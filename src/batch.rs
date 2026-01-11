//! Batch processing for Anthropic Message Batches API
//!
//! AIDEV-NOTE: Batch API provides 50% cost savings for non-time-sensitive workloads.
//! Large sessions (>10K tokens) are queued for batch processing while small sessions
//! are processed in real-time.
//!
//! Batch lifecycle:
//! 1. Queue requests until threshold (count or time)
//! 2. Submit batch to Anthropic API
//! 3. Poll for completion (up to 24 hours)
//! 4. Download and process results

use chrono::{DateTime, Utc};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

const ANTHROPIC_BATCH_URL: &str = "https://api.anthropic.com/v1/messages/batches";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// Minimum requests before submitting a batch
const BATCH_MIN_COUNT: usize = 10;

/// Maximum requests per batch (Anthropic limit)
const BATCH_MAX_COUNT: usize = 10_000;

/// Maximum age of oldest request before forcing batch submission (1 hour)
const BATCH_MAX_AGE_SECS: i64 = 3600;

/// Error types for batch operations
#[derive(Debug)]
pub enum BatchError {
    /// API key not configured
    MissingApiKey,
    /// HTTP request failed
    RequestFailed(String),
    /// API returned error
    ApiError(String),
    /// Failed to parse response
    ParseError(String),
    /// Batch not found
    NotFound(String),
    /// IO error (state persistence)
    IoError(String),
}

impl std::fmt::Display for BatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchError::MissingApiKey => write!(f, "ANTHROPIC_API_KEY not set"),
            BatchError::RequestFailed(e) => write!(f, "Request failed: {}", e),
            BatchError::ApiError(e) => write!(f, "API error: {}", e),
            BatchError::ParseError(e) => write!(f, "Parse error: {}", e),
            BatchError::NotFound(id) => write!(f, "Batch not found: {}", id),
            BatchError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for BatchError {}

/// A single request to be included in a batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Unique identifier for matching results (session_id:chunk:timestamp)
    pub custom_id: String,
    /// Session ID this request belongs to
    pub session_id: String,
    /// System prompt for the LLM
    pub system_prompt: String,
    /// User message (transcript content)
    pub message: String,
    /// When this request was queued
    pub queued_at: DateTime<Utc>,
    /// SimHash of the transcript content (for deduplication on completion)
    pub simhash: i64,
    /// Chunk index (0-based) if this is part of a chunked session
    #[serde(default)]
    pub chunk_index: Option<usize>,
    /// Total number of chunks for this session
    #[serde(default)]
    pub total_chunks: Option<usize>,
}

/// Metadata for a batch request (preserved through submit → poll cycle)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequestMeta {
    /// Custom ID for matching results
    pub custom_id: String,
    /// Session ID this request belongs to
    pub session_id: String,
    /// SimHash of the transcript content
    pub simhash: i64,
    /// Chunk index (0-based) if this is part of a chunked session
    #[serde(default)]
    pub chunk_index: Option<usize>,
    /// Total number of chunks for this session
    #[serde(default)]
    pub total_chunks: Option<usize>,
}

/// A batch that has been submitted and is pending completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingBatch {
    /// Anthropic batch ID
    pub batch_id: String,
    /// When the batch was submitted
    pub submitted_at: DateTime<Utc>,
    /// Custom IDs of requests in this batch
    pub request_ids: Vec<String>,
    /// Model used for this batch
    pub model: String,
    /// Request metadata (session_id, simhash) preserved for result processing
    #[serde(default)]
    pub request_meta: Vec<BatchRequestMeta>,
}

/// Status of a batch
#[derive(Debug, Clone, PartialEq)]
pub enum BatchStatus {
    /// Still processing
    InProgress,
    /// All requests completed
    Ended,
    /// Cancellation in progress
    Canceling,
}

/// Result of a single request within a batch
#[derive(Debug)]
pub struct BatchResultItem {
    /// Custom ID matching the original request
    pub custom_id: String,
    /// Result type: "succeeded", "errored", "expired", "canceled"
    pub result_type: String,
    /// Generated text (if succeeded)
    pub text: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Token usage
    pub usage: Option<BatchUsage>,
}

#[derive(Debug)]
pub struct BatchUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Queue for accumulating batch requests
pub struct BatchQueue {
    /// Pending requests not yet submitted
    requests: Vec<BatchRequest>,
    /// Batches that have been submitted
    pending_batches: Vec<PendingBatch>,
    /// Model to use for batches
    model: String,
    /// Path to state file
    state_path: PathBuf,
}

// API request/response types

#[derive(Debug, Serialize)]
struct ApiBatchRequest<'a> {
    requests: Vec<ApiBatchRequestItem<'a>>,
}

#[derive(Debug, Serialize)]
struct ApiBatchRequestItem<'a> {
    custom_id: &'a str,
    params: ApiBatchParams<'a>,
}

#[derive(Debug, Serialize)]
struct ApiBatchParams<'a> {
    model: &'a str,
    max_tokens: u32,
    system: Vec<ApiSystemBlock<'a>>,
    messages: Vec<ApiMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct ApiSystemBlock<'a> {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: &'a str,
    cache_control: ApiCacheControl,
}

#[derive(Debug, Serialize)]
struct ApiCacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

#[derive(Debug, Serialize)]
struct ApiMessage<'a> {
    role: &'static str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ApiBatchResponse {
    id: String,
    processing_status: String,
    results_url: Option<String>,
    request_counts: ApiRequestCounts,
}

#[derive(Debug, Deserialize)]
struct ApiRequestCounts {
    processing: u32,
    succeeded: u32,
    errored: u32,
    canceled: u32,
    expired: u32,
}

#[derive(Debug, Deserialize)]
struct ApiResultLine {
    custom_id: String,
    result: ApiResultData,
}

#[derive(Debug, Deserialize)]
struct ApiResultData {
    #[serde(rename = "type")]
    result_type: String,
    message: Option<ApiResultMessage>,
    error: Option<ApiResultError>,
}

#[derive(Debug, Deserialize)]
struct ApiResultMessage {
    content: Vec<ApiContentBlock>,
    usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
struct ApiContentBlock {
    text: String,
}

#[derive(Debug, Deserialize)]
struct ApiUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ApiResultError {
    message: String,
}

impl BatchQueue {
    /// Create a new batch queue
    pub fn new(model: String, state_path: PathBuf) -> Self {
        Self {
            requests: Vec::new(),
            pending_batches: Vec::new(),
            model,
            state_path,
        }
    }

    /// Load state from disk
    pub fn load_state(&mut self) -> Result<(), BatchError> {
        if !self.state_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(&self.state_path)
            .map_err(|e| BatchError::IoError(e.to_string()))?;

        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }

            // Try to parse as PendingBatch first
            if let Ok(batch) = serde_json::from_str::<PendingBatch>(line) {
                self.pending_batches.push(batch);
            } else if let Ok(req) = serde_json::from_str::<BatchRequest>(line) {
                // Fall back to BatchRequest
                self.requests.push(req);
            }
        }

        Ok(())
    }

    /// Save state to disk
    pub fn save_state(&self) -> Result<(), BatchError> {
        let mut lines = Vec::new();

        // Save pending batches
        for batch in &self.pending_batches {
            let line = serde_json::to_string(batch)
                .map_err(|e| BatchError::ParseError(e.to_string()))?;
            lines.push(line);
        }

        // Save queued requests
        for req in &self.requests {
            let line = serde_json::to_string(req)
                .map_err(|e| BatchError::ParseError(e.to_string()))?;
            lines.push(line);
        }

        std::fs::write(&self.state_path, lines.join("\n"))
            .map_err(|e| BatchError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Add a request to the queue
    /// For unchunked sessions, chunk_index and total_chunks should be None
    pub fn add(&mut self, session_id: String, system_prompt: String, message: String, simhash: i64) {
        self.add_chunk(session_id, system_prompt, message, simhash, None, None);
    }

    /// Add a chunked request to the queue
    /// AIDEV-NOTE: Each chunk gets its own batch request, results aggregated on completion
    pub fn add_chunk(
        &mut self,
        session_id: String,
        system_prompt: String,
        message: String,
        simhash: i64,
        chunk_index: Option<usize>,
        total_chunks: Option<usize>,
    ) {
        // Use nanoseconds to prevent collisions when multiple requests are queued in same second
        let timestamp_nanos = Utc::now().timestamp_nanos_opt().unwrap_or_else(|| Utc::now().timestamp() * 1_000_000_000);
        let custom_id = match chunk_index {
            Some(idx) => format!("{}:c{}:{}", session_id, idx, timestamp_nanos),
            None => format!("{}:{}", session_id, timestamp_nanos),
        };
        self.requests.push(BatchRequest {
            custom_id,
            session_id,
            system_prompt,
            message,
            queued_at: Utc::now(),
            simhash,
            chunk_index,
            total_chunks,
        });
    }

    /// Check if batch should be submitted
    pub fn should_submit(&self) -> bool {
        if self.requests.is_empty() {
            return false;
        }

        // Submit if we have enough requests
        if self.requests.len() >= BATCH_MIN_COUNT {
            return true;
        }

        // Submit if oldest request is too old
        // Note: Use min_by_key instead of first() because re-queued requests go to end
        if let Some(oldest) = self.requests.iter().min_by_key(|r| r.queued_at) {
            let age = Utc::now().signed_duration_since(oldest.queued_at);
            if age.num_seconds() >= BATCH_MAX_AGE_SECS {
                return true;
            }
        }

        false
    }

    /// Force submit regardless of thresholds (e.g., on shutdown)
    pub fn has_pending_requests(&self) -> bool {
        !self.requests.is_empty()
    }

    /// Get count of pending batches
    pub fn pending_batch_count(&self) -> usize {
        self.pending_batches.len()
    }

    /// Get count of queued requests
    pub fn queued_request_count(&self) -> usize {
        self.requests.len()
    }

    /// Submit queued requests as a batch
    pub async fn submit(&mut self) -> Result<String, BatchError> {
        if self.requests.is_empty() {
            return Err(BatchError::ApiError("No requests to submit".to_string()));
        }

        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| BatchError::MissingApiKey)?;

        // Clone requests to submit (don't drain yet - wait for successful response)
        let submit_count = self.requests.len().min(BATCH_MAX_COUNT);
        let to_submit: Vec<_> = self.requests[..submit_count].to_vec();
        let request_ids: Vec<_> = to_submit.iter().map(|r| r.custom_id.clone()).collect();

        // Build API request
        let api_requests: Vec<_> = to_submit
            .iter()
            .map(|req| ApiBatchRequestItem {
                custom_id: &req.custom_id,
                params: ApiBatchParams {
                    model: &self.model,
                    max_tokens: 16000,
                    system: vec![ApiSystemBlock {
                        block_type: "text",
                        text: &req.system_prompt,
                        cache_control: ApiCacheControl {
                            cache_type: "ephemeral",
                        },
                    }],
                    messages: vec![ApiMessage {
                        role: "user",
                        content: &req.message,
                    }],
                },
            })
            .collect();

        let body = ApiBatchRequest {
            requests: api_requests,
        };

        let client = Client::new();
        let response = client
            .post(ANTHROPIC_BATCH_URL)
            .header("x-api-key", &api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .header("anthropic-beta", "message-batches-2024-09-24")
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| BatchError::RequestFailed(e.to_string()))?;

        let status = response.status();
        let body_text = response
            .text()
            .await
            .map_err(|e| BatchError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(BatchError::ApiError(format!("{}: {}", status, body_text)));
        }

        // Success - now drain the submitted requests from queue
        self.requests.drain(..submit_count);

        let parsed: ApiBatchResponse = serde_json::from_str(&body_text)
            .map_err(|e| BatchError::ParseError(format!("{}: {}", e, body_text)))?;

        let batch_id = parsed.id.clone();

        // Build metadata from submitted requests (preserves simhash + chunk info through batch lifecycle)
        let request_meta: Vec<BatchRequestMeta> = to_submit
            .iter()
            .map(|r| BatchRequestMeta {
                custom_id: r.custom_id.clone(),
                session_id: r.session_id.clone(),
                simhash: r.simhash,
                chunk_index: r.chunk_index,
                total_chunks: r.total_chunks,
            })
            .collect();

        // Track pending batch
        self.pending_batches.push(PendingBatch {
            batch_id: batch_id.clone(),
            submitted_at: Utc::now(),
            request_ids,
            model: self.model.clone(),
            request_meta,
        });

        // Save state
        self.save_state()?;

        Ok(batch_id)
    }

    /// Poll all pending batches for completion
    /// Returns completed batches with their results
    ///
    /// On transient errors, batches remain in pending state for retry.
    /// Only successfully fetched results cause batch removal.
    pub async fn poll_pending(&mut self) -> Result<Vec<(PendingBatch, Vec<BatchResultItem>)>, BatchError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| BatchError::MissingApiKey)?;

        let client = Client::new();
        let mut completed = Vec::new();
        let mut indices_to_remove = Vec::new();

        // Process each batch, tracking which ones complete successfully
        for (idx, batch) in self.pending_batches.iter().enumerate() {
            let url = format!("{}/{}", ANTHROPIC_BATCH_URL, batch.batch_id);

            let response = match client
                .get(&url)
                .header("x-api-key", &api_key)
                .header("anthropic-version", ANTHROPIC_API_VERSION)
                .header("anthropic-beta", "message-batches-2024-09-24")
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("Network error polling batch {}: {}", batch.batch_id, e);
                    continue;
                }
            };

            if response.status() == StatusCode::NOT_FOUND {
                eprintln!("Batch {} not found (expired or invalid), removing", batch.batch_id);
                indices_to_remove.push(idx);
                continue;
            }

            let body = match response.text().await {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Read error polling batch {}: {}", batch.batch_id, e);
                    continue;
                }
            };

            let parsed: ApiBatchResponse = match serde_json::from_str(&body) {
                Ok(p) => p,
                Err(e) => {
                    let preview: String = body.chars().take(200).collect();
                    eprintln!("Parse error polling batch {}: {} (body: {})", batch.batch_id, e, preview);
                    continue;
                }
            };

            match parsed.processing_status.as_str() {
                "ended" => {
                    if let Some(results_url) = parsed.results_url {
                        // Try to fetch results; only remove batch on success
                        match Self::fetch_results_static(&client, &api_key, &results_url).await {
                            Ok(results) => {
                                completed.push((batch.clone(), results));
                                indices_to_remove.push(idx);
                            }
                            Err(e) => {
                                eprintln!("Fetch error for batch {}: {}", batch.batch_id, e);
                                continue;
                            }
                        }
                    } else {
                        // Ended but no results URL - remove as there's nothing to fetch
                        indices_to_remove.push(idx);
                    }
                }
                _ => {
                    // Still processing or unknown status, keep polling
                }
            }
        }

        // Remove completed/invalid batches in reverse order to preserve indices
        for idx in indices_to_remove.into_iter().rev() {
            self.pending_batches.remove(idx);
        }

        // Save state
        self.save_state()?;

        Ok(completed)
    }

    /// Fetch results from a completed batch
    async fn fetch_results_static(
        client: &Client,
        api_key: &str,
        results_url: &str,
    ) -> Result<Vec<BatchResultItem>, BatchError> {
        let response = client
            .get(results_url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_API_VERSION)
            .send()
            .await
            .map_err(|e| BatchError::RequestFailed(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| BatchError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(BatchError::ApiError(format!("{}: {}", status, body)));
        }

        let mut results = Vec::new();

        for line in body.lines() {
            if line.trim().is_empty() {
                continue;
            }

            let parsed: ApiResultLine = serde_json::from_str(line)
                .map_err(|e| BatchError::ParseError(format!("{}: {}", e, line)))?;

            let item = BatchResultItem {
                custom_id: parsed.custom_id,
                result_type: parsed.result.result_type.clone(),
                text: parsed
                    .result
                    .message
                    .as_ref()
                    .and_then(|m| m.content.first())
                    .map(|c| c.text.clone()),
                error: parsed.result.error.map(|e| e.message),
                usage: parsed.result.message.map(|m| BatchUsage {
                    input_tokens: m.usage.input_tokens,
                    output_tokens: m.usage.output_tokens,
                }),
            };

            results.push(item);
        }

        Ok(results)
    }

    /// Get mapping of custom_id to session_id for a batch
    pub fn get_session_mapping(&self, batch: &PendingBatch) -> HashMap<String, String> {
        // First try to use request_meta (preserves exact session_id)
        if !batch.request_meta.is_empty() {
            return batch
                .request_meta
                .iter()
                .map(|m| (m.custom_id.clone(), m.session_id.clone()))
                .collect();
        }

        // Fallback: parse custom_id format "session_id:timestamp"
        batch
            .request_ids
            .iter()
            .filter_map(|cid| {
                cid.split(':').next().map(|sid| (cid.clone(), sid.to_string()))
            })
            .collect()
    }

    /// Get full metadata for a batch (for result processing)
    /// Returns mapping of custom_id to BatchRequestMeta
    /// AIDEV-NOTE: Critical for deduplication and chunk aggregation
    pub fn get_request_meta(&self, batch: &PendingBatch) -> HashMap<String, BatchRequestMeta> {
        batch
            .request_meta
            .iter()
            .map(|m| (m.custom_id.clone(), m.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_submit_empty() {
        let queue = BatchQueue::new("test".to_string(), PathBuf::from("/tmp/test"));
        assert!(!queue.should_submit());
    }

    #[test]
    fn test_should_submit_count_threshold() {
        let mut queue = BatchQueue::new("test".to_string(), PathBuf::from("/tmp/test"));
        for i in 0..BATCH_MIN_COUNT {
            queue.add(format!("sess_{}", i), "system".to_string(), "message".to_string(), i as i64);
        }
        assert!(queue.should_submit());
    }
}
