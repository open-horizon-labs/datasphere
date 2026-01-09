//! Persistent job queue for daemon processing
//!
//! AIDEV-NOTE: Append-only JSONL file at ~/.datasphere/queue.jsonl
//! Latest entry per source_id wins (allows status updates without rewriting).
//! Supports pending → processing → done/failed transitions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

/// Job status in the queue
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Pending,
    Processing,
    Done,
    Failed,
}

/// A job in the queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique identifier (session UUID or file path)
    pub source_id: String,
    /// Type of source
    pub source_type: String,
    /// Project ID (directory name in ~/.claude/projects/)
    pub project_id: String,
    /// Path to transcript file
    pub transcript_path: String,
    /// When the job was queued
    pub queued_at: DateTime<Utc>,
    /// Current status
    pub status: JobStatus,
    /// Error message if failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Persistent job queue backed by JSONL file
pub struct Queue {
    path: PathBuf,
}

impl Queue {
    /// Open or create queue at the default location (~/.datasphere/queue.jsonl)
    pub fn open_default() -> Result<Self, String> {
        let ds_dir = dirs::home_dir()
            .ok_or("Could not determine home directory")?
            .join(".datasphere");

        std::fs::create_dir_all(&ds_dir)
            .map_err(|e| format!("Failed to create datasphere directory: {}", e))?;

        let path = ds_dir.join("queue.jsonl");
        Ok(Self { path })
    }

    /// Add a job to the queue (deduplicates by source_id)
    /// Returns Ok(true) if job was added, Ok(false) if skipped (already pending/processing)
    pub fn add(&self, job: Job) -> Result<bool, String> {
        // Check if already pending/processing
        let jobs = self.load_all()?;
        if let Some(existing) = jobs.get(&job.source_id) {
            if existing.status == JobStatus::Pending || existing.status == JobStatus::Processing {
                // Already queued, skip
                return Ok(false);
            }
        }

        self.append(&job)?;
        Ok(true)
    }

    /// Get oldest pending job and mark it as processing
    pub fn pop_pending(&self) -> Result<Option<Job>, String> {
        let jobs = self.load_all()?;

        // Find oldest pending job
        let mut pending: Vec<_> = jobs
            .into_values()
            .filter(|j| j.status == JobStatus::Pending)
            .collect();

        pending.sort_by(|a, b| a.queued_at.cmp(&b.queued_at));

        if let Some(mut job) = pending.into_iter().next() {
            job.status = JobStatus::Processing;
            self.append(&job)?;
            Ok(Some(job))
        } else {
            Ok(None)
        }
    }

    /// Mark a job as done
    pub fn mark_done(&self, source_id: &str) -> Result<(), String> {
        let jobs = self.load_all()?;
        if let Some(mut job) = jobs.get(source_id).cloned() {
            job.status = JobStatus::Done;
            job.error = None;
            self.append(&job)?;
        }
        Ok(())
    }

    /// Mark a job as failed with error message
    pub fn mark_failed(&self, source_id: &str, error: &str) -> Result<(), String> {
        let jobs = self.load_all()?;
        if let Some(mut job) = jobs.get(source_id).cloned() {
            job.status = JobStatus::Failed;
            job.error = Some(error.to_string());
            self.append(&job)?;
        }
        Ok(())
    }

    /// Mark a job as pending (for returning processing jobs to the queue)
    pub fn mark_pending(&self, source_id: &str) -> Result<(), String> {
        let jobs = self.load_all()?;
        if let Some(mut job) = jobs.get(source_id).cloned() {
            job.status = JobStatus::Pending;
            job.error = None;
            self.append(&job)?;
        }
        Ok(())
    }

    /// List all pending jobs
    pub fn list_pending(&self) -> Result<Vec<Job>, String> {
        let jobs = self.load_all()?;
        let mut pending: Vec<_> = jobs
            .into_values()
            .filter(|j| j.status == JobStatus::Pending)
            .collect();
        pending.sort_by(|a, b| a.queued_at.cmp(&b.queued_at));
        Ok(pending)
    }

    /// List all processing jobs
    pub fn list_processing(&self) -> Result<Vec<Job>, String> {
        let jobs = self.load_all()?;
        Ok(jobs
            .into_values()
            .filter(|j| j.status == JobStatus::Processing)
            .collect())
    }

    /// Count jobs by status
    pub fn counts(&self) -> Result<(usize, usize, usize, usize), String> {
        let jobs = self.load_all()?;
        let pending = jobs.values().filter(|j| j.status == JobStatus::Pending).count();
        let processing = jobs.values().filter(|j| j.status == JobStatus::Processing).count();
        let done = jobs.values().filter(|j| j.status == JobStatus::Done).count();
        let failed = jobs.values().filter(|j| j.status == JobStatus::Failed).count();
        Ok((pending, processing, done, failed))
    }

    /// Clear completed (done) jobs by compacting the file
    pub fn clear_done(&self) -> Result<usize, String> {
        let jobs = self.load_all()?;
        let done_count = jobs.values().filter(|j| j.status == JobStatus::Done).count();

        // Rewrite file without done jobs
        let remaining: Vec<_> = jobs
            .into_values()
            .filter(|j| j.status != JobStatus::Done)
            .collect();

        self.rewrite(&remaining)?;
        Ok(done_count)
    }

    /// Delete entire queue (all jobs, all statuses)
    pub fn nuke(&self) -> Result<usize, String> {
        let jobs = self.load_all()?;
        let total = jobs.len();

        // Truncate file to empty
        File::create(&self.path)
            .map_err(|e| format!("Failed to nuke queue: {}", e))?;

        Ok(total)
    }

    /// Retry all failed jobs (moves them from Failed → Pending)
    /// Returns number of jobs requeued
    pub fn retry_all(&self) -> Result<usize, String> {
        let jobs = self.load_all()?;
        let mut count = 0;

        for (_, mut job) in jobs.into_iter() {
            if job.status == JobStatus::Failed {
                job.status = JobStatus::Pending;
                job.error = None;
                job.queued_at = Utc::now();
                self.append(&job)?;
                count += 1;
            }
        }

        Ok(count)
    }

    /// Retry a specific failed job by source_id
    /// Returns Ok(true) if job was requeued, Ok(false) if not found or not failed
    pub fn retry_one(&self, source_id: &str) -> Result<bool, String> {
        let jobs = self.load_all()?;

        if let Some(mut job) = jobs.get(source_id).cloned() {
            if job.status == JobStatus::Failed {
                job.status = JobStatus::Pending;
                job.error = None;
                job.queued_at = Utc::now();
                self.append(&job)?;
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// List all failed jobs
    pub fn list_failed(&self) -> Result<Vec<Job>, String> {
        let jobs = self.load_all()?;
        let mut failed: Vec<_> = jobs
            .into_values()
            .filter(|j| j.status == JobStatus::Failed)
            .collect();
        failed.sort_by(|a, b| a.queued_at.cmp(&b.queued_at));
        Ok(failed)
    }

    /// Load all jobs, deduplicating by source_id (latest wins)
    fn load_all(&self) -> Result<HashMap<String, Job>, String> {
        let mut jobs = HashMap::new();

        if !self.path.exists() {
            return Ok(jobs);
        }

        let file = File::open(&self.path)
            .map_err(|e| format!("Failed to open queue file: {}", e))?;

        for line in BufReader::new(file).lines() {
            let line = line.map_err(|e| format!("Failed to read line: {}", e))?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<Job>(&line) {
                Ok(job) => {
                    jobs.insert(job.source_id.clone(), job);
                }
                Err(e) => {
                    eprintln!("Warning: skipping malformed queue entry: {}", e);
                }
            }
        }

        Ok(jobs)
    }

    /// Append a job entry to the file
    fn append(&self, job: &Job) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("Failed to open queue file for append: {}", e))?;

        let line = serde_json::to_string(job)
            .map_err(|e| format!("Failed to serialize job: {}", e))?;

        writeln!(file, "{}", line)
            .map_err(|e| format!("Failed to write to queue file: {}", e))?;

        Ok(())
    }

    /// Rewrite the entire file with the given jobs
    fn rewrite(&self, jobs: &[Job]) -> Result<(), String> {
        let mut file = File::create(&self.path)
            .map_err(|e| format!("Failed to create queue file: {}", e))?;

        for job in jobs {
            let line = serde_json::to_string(job)
                .map_err(|e| format!("Failed to serialize job: {}", e))?;
            writeln!(file, "{}", line)
                .map_err(|e| format!("Failed to write to queue file: {}", e))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::{tempdir, TempDir};

    /// Returns (queue, _tempdir) - must keep _tempdir alive for test duration
    fn test_queue() -> (Queue, TempDir) {
        let dir = tempdir().unwrap();
        let queue = Queue {
            path: dir.path().join("queue.jsonl"),
        };
        (queue, dir)
    }

    #[test]
    fn test_add_and_pop() {
        let (queue, _dir) = test_queue();

        let job = Job {
            source_id: "test-123".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };

        queue.add(job.clone()).unwrap();

        let popped = queue.pop_pending().unwrap();
        assert!(popped.is_some());
        let popped = popped.unwrap();
        assert_eq!(popped.source_id, "test-123");
        assert_eq!(popped.status, JobStatus::Processing);

        // Should be no more pending
        let next = queue.pop_pending().unwrap();
        assert!(next.is_none());
    }

    #[test]
    fn test_deduplication() {
        let (queue, _dir) = test_queue();

        let job = Job {
            source_id: "test-123".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };

        let added = queue.add(job.clone()).unwrap();
        assert!(added); // First add should succeed

        let added = queue.add(job.clone()).unwrap();
        assert!(!added); // Duplicate should be skipped

        let pending = queue.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
    }

    #[test]
    fn test_mark_pending() {
        let (queue, _dir) = test_queue();

        let job = Job {
            source_id: "test-123".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };

        queue.add(job.clone()).unwrap();

        // Pop to mark as processing
        let popped = queue.pop_pending().unwrap().unwrap();
        assert_eq!(popped.status, JobStatus::Processing);

        // Verify it's processing
        let processing = queue.list_processing().unwrap();
        assert_eq!(processing.len(), 1);

        // Mark back to pending
        queue.mark_pending("test-123").unwrap();

        // Verify it's pending again
        let pending = queue.list_pending().unwrap();
        assert_eq!(pending.len(), 1);
        let processing = queue.list_processing().unwrap();
        assert_eq!(processing.len(), 0);
    }

    #[test]
    fn test_retry_all() {
        let (queue, _dir) = test_queue();

        // Add two jobs and mark them as failed
        let job1 = Job {
            source_id: "test-1".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test1.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };
        let job2 = Job {
            source_id: "test-2".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test2.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };

        queue.add(job1).unwrap();
        queue.add(job2).unwrap();

        // Mark both as failed
        queue.mark_failed("test-1", "error 1").unwrap();
        queue.mark_failed("test-2", "error 2").unwrap();

        // Verify they're failed
        let failed = queue.list_failed().unwrap();
        assert_eq!(failed.len(), 2);
        assert_eq!(queue.list_pending().unwrap().len(), 0);

        // Retry all
        let retried = queue.retry_all().unwrap();
        assert_eq!(retried, 2);

        // Verify they're now pending
        let pending = queue.list_pending().unwrap();
        assert_eq!(pending.len(), 2);
        assert_eq!(queue.list_failed().unwrap().len(), 0);

        // Error should be cleared
        for job in &pending {
            assert!(job.error.is_none());
        }
    }

    #[test]
    fn test_retry_one() {
        let (queue, _dir) = test_queue();

        let job = Job {
            source_id: "test-123".to_string(),
            source_type: "session".to_string(),
            project_id: "-test-project".to_string(),
            transcript_path: "/path/to/test.jsonl".to_string(),
            queued_at: Utc::now(),
            status: JobStatus::Pending,
            error: None,
        };

        queue.add(job).unwrap();
        queue.mark_failed("test-123", "some error").unwrap();

        // Verify it's failed
        assert_eq!(queue.list_failed().unwrap().len(), 1);
        assert_eq!(queue.list_pending().unwrap().len(), 0);

        // Retry non-existent job should return false
        assert!(!queue.retry_one("nonexistent").unwrap());

        // Retry the failed job
        assert!(queue.retry_one("test-123").unwrap());

        // Verify it's now pending
        assert_eq!(queue.list_pending().unwrap().len(), 1);
        assert_eq!(queue.list_failed().unwrap().len(), 0);

        // Retrying again should return false (not failed anymore)
        assert!(!queue.retry_one("test-123").unwrap());
    }
}
