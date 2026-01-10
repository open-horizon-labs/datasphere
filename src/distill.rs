//! Distillation - extract knowledge nodes from transcripts
//!
//! AIDEV-NOTE: Extracts knowledge from sessions per engram-spec-v1:
//! - Concepts: ideas, patterns, principles
//! - Decisions: choices with rationale
//! - Intents: goals, desires, plans
//! - Entities: people, projects, tools
//! - Questions: open/unresolved items
//!
//! One transcript → one distillation → one node.
//! Coarser granularity is intentional - enables serendipitous connections.
//! LLM produces a narrative summary - no structured parsing needed.
//!
//! For large transcripts (>12K tokens), we chunk semantically, distill each
//! chunk in parallel (up to 4 concurrent), then synthesize into one summary.

use crate::core::{Node, SourceType};
use crate::llm;
use futures::{stream, StreamExt};
use semchunk_rs::Chunker;
use std::sync::OnceLock;
use std::time::Instant;
use tiktoken_rs::{cl100k_base, CoreBPE};

/// An extracted piece of knowledge ready to become a Node
#[derive(Debug, Clone)]
pub struct ExtractedInsight {
    /// The knowledge content (full LLM distillation narrative)
    pub content: String,

    /// Confidence score (0.0-1.0), currently always 1.0
    pub confidence: f32,
}

impl ExtractedInsight {
    /// Convert this insight into a Node for storage
    ///
    /// # Arguments
    /// * `source` - Source identifier (e.g., session ID, file path)
    /// * `source_type` - Type of source (Session, File, etc.)
    /// * `embedding` - Pre-computed embedding vector (1536 dims for text-embedding-3-small)
    pub fn into_node(self, source: String, source_type: SourceType, embedding: Vec<f32>) -> Node {
        Node::new(self.content, source, source_type, embedding, self.confidence)
    }
}

/// Minimum length for a distillation to be considered substantive
const MIN_CONTENT_LEN: usize = 50;

/// Threshold for chunking (in tokens). Above this, we chunk and synthesize.
/// AIDEV-NOTE: ~12K tokens leaves room for system prompt + response in Claude's context.
const CHUNK_THRESHOLD_TOKENS: usize = 12000;

/// Max tokens per chunk when splitting large transcripts
const MAX_CHUNK_TOKENS: usize = 10000;

/// Max parallel LLM calls for chunk distillation
const PARALLEL_DISTILLATIONS: usize = 4;

/// Global BPE tokenizer instance (cached for performance)
static BPE: OnceLock<CoreBPE> = OnceLock::new();

fn get_bpe() -> &'static CoreBPE {
    BPE.get_or_init(|| cl100k_base().expect("Failed to load cl100k_base tokenizer"))
}

/// Count tokens in text
fn count_tokens(text: &str) -> usize {
    get_bpe().encode_with_special_tokens(text).len()
}

/// Split transcript into semantic chunks
fn chunk_transcript(transcript: &str) -> Vec<String> {
    let chunker = Chunker::new(MAX_CHUNK_TOKENS, Box::new(count_tokens));
    chunker.chunk(transcript)
}

/// Result of extraction with metadata about chunking
pub struct ExtractionResult {
    /// All extracted insights (one per chunk, or single for small transcripts)
    pub insights: Vec<ExtractedInsight>,
    pub chunks_used: usize,
    /// Total cost in USD (if available from API)
    pub total_cost: Option<f64>,
}

/// Extract knowledge from a formatted transcript
///
/// Returns ExtractedInsights - one per chunk for large transcripts,
/// or a single insight for small ones.
///
/// For large transcripts, chunks semantically and distills each chunk
/// in parallel (up to 4 concurrent). Each chunk becomes a separate node.
///
/// # Arguments
/// * `transcript` - Formatted transcript text (from format_context)
///
/// # Returns
/// ExtractionResult with insights (may be empty) and chunk count
pub async fn extract_knowledge(transcript: &str) -> Result<ExtractionResult, llm::LlmError> {
    if transcript.trim().is_empty() {
        return Ok(ExtractionResult {
            insights: Vec::new(),
            chunks_used: 0,
            total_cost: None,
        });
    }

    let token_count = count_tokens(transcript);

    // Small transcript: single LLM call
    if token_count <= CHUNK_THRESHOLD_TOKENS {
        let result = call_extraction_llm(transcript).await?;
        let cost = result.cost.map(|(_, _, usd)| usd);
        let content = result.text.trim().to_string();

        if content.len() < MIN_CONTENT_LEN {
            eprintln!("    Distillation too short ({} chars < {}), skipping", content.len(), MIN_CONTENT_LEN);
            return Ok(ExtractionResult {
                insights: Vec::new(),
                chunks_used: 1,
                total_cost: cost,
            });
        }

        return Ok(ExtractionResult {
            insights: vec![ExtractedInsight {
                content,
                confidence: 1.0,
            }],
            chunks_used: 1,
            total_cost: cost,
        });
    }

    // Large transcript: chunk → distill in parallel (no synthesis)
    let chunks = chunk_transcript(transcript);
    let chunk_count = chunks.len();

    eprintln!(
        "    Distilling {} chunks ({} parallel)...",
        chunk_count, PARALLEL_DISTILLATIONS
    );

    // Distill chunks in parallel with bounded concurrency
    let chunk_start = Instant::now();
    let chunk_results: Vec<(usize, Result<llm::LlmResult, llm::LlmError>)> = stream::iter(chunks.into_iter().enumerate())
        .map(|(i, chunk)| async move {
            let result = call_extraction_llm(&chunk).await;
            (i, result)
        })
        .buffer_unordered(PARALLEL_DISTILLATIONS)
        .collect()
        .await;
    eprintln!("    Chunk distillation completed in {:.1}s", chunk_start.elapsed().as_secs_f32());

    // Collect successful distillations (preserving order for consistency)
    // If any chunk hit a rate limit, propagate that error immediately
    let mut indexed_distillations: Vec<(usize, String)> = Vec::new();
    let mut dropped_count = 0;
    let mut total_cost: f64 = 0.0;

    for (i, result) in chunk_results {
        match result {
            Ok(llm_result) => {
                if let Some((_, _, cost)) = llm_result.cost {
                    total_cost += cost;
                }
                let trimmed = llm_result.text.trim();
                if !trimmed.is_empty() && trimmed.len() >= MIN_CONTENT_LEN {
                    indexed_distillations.push((i, llm_result.text));
                } else {
                    dropped_count += 1;
                    eprintln!("    Chunk {} distillation too short ({} chars), skipping", i + 1, trimmed.len());
                }
            }
            Err(llm::LlmError::RateLimit(msg)) => {
                // Rate limit on any chunk should propagate immediately
                return Err(llm::LlmError::RateLimit(msg));
            }
            Err(e) => {
                dropped_count += 1;
                eprintln!("    Warning: chunk {} failed: {}", i + 1, e);
            }
        }
    }

    if dropped_count > 0 {
        eprintln!("    Dropped {} of {} chunks", dropped_count, chunk_count);
    }

    // Sort by original chunk order
    indexed_distillations.sort_by_key(|(i, _)| *i);

    // Convert to insights (no synthesis - each chunk becomes its own node)
    let insights: Vec<ExtractedInsight> = indexed_distillations
        .into_iter()
        .map(|(_, content)| ExtractedInsight {
            content,
            confidence: 1.0,
        })
        .collect();

    eprintln!("    Extracted {} insights from {} chunks", insights.len(), chunk_count);

    Ok(ExtractionResult {
        insights,
        chunks_used: chunk_count,
        total_cost: if total_cost > 0.0 { Some(total_cost) } else { None },
    })
}

/// Call LLM to extract knowledge from transcript
async fn call_extraction_llm(transcript: &str) -> Result<llm::LlmResult, llm::LlmError> {
    // AIDEV-NOTE: Prompt based on engram-spec-v1.md extraction categories.
    // No structured format enforced - LLM narrates naturally.
    // The whole response becomes the node content.
    let system_prompt = r#"You are extracting knowledge from an AI <> Human session transcript for a knowledge graph.

Your output will be stored and retrieved via semantic search in future sessions.

Extract these types of knowledge:

**CONCEPTS** - Ideas, patterns, and principles discussed
- Architectural patterns applied or considered
- Design principles referenced
- Technical concepts explained

**DECISIONS** - Choices made with their rationale
- Why one approach was chosen over another
- Trade-offs considered
- Constraints that influenced choices

**INTENTS** - Goals, desires, and plans expressed
- What the user is trying to achieve
- Future work mentioned
- Success criteria discussed

**ENTITIES** - Significant named things
- Projects, tools, libraries mentioned
- People or teams referenced
- Services or systems involved

**QUESTIONS** - Open/unresolved items
- Things left for later
- Uncertainties acknowledged
- Areas needing investigation

Write a concise summary capturing the key knowledge. Use whatever format (prose, bullets, headers) best fits the content. Each insight should be self-contained and useful without the original transcript.

Skip routine actions (ran tests, fixed typo). Focus on knowledge valuable for future sessions.

If the session has no substantive knowledge, just say so briefly.

IMPORTANT: Keep your response under 2000 words. Be concise and focus."#;

    let message = format!("TRANSCRIPT:\n{}", transcript);
    llm::call_claude(system_prompt, &message).await
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EMBEDDING_DIM;

    #[test]
    fn test_insight_into_node() {
        let insight = ExtractedInsight {
            content: "Test insight content".to_string(),
            confidence: 0.9,
        };
        let embedding = vec![0.0f32; EMBEDDING_DIM];

        let node = insight.into_node("session-123".to_string(), SourceType::Session, embedding);

        assert_eq!(node.content, "Test insight content");
        assert_eq!(node.confidence, 0.9);
        assert_eq!(node.source, "session-123");
        assert!(matches!(node.source_type, SourceType::Session));
        assert_eq!(node.embedding.len(), EMBEDDING_DIM);
    }

    #[tokio::test]
    async fn test_extract_knowledge_empty_transcript() {
        let result = extract_knowledge("").await.unwrap();
        assert!(result.insights.is_empty());
        assert_eq!(result.chunks_used, 0);

        let result = extract_knowledge("   ").await.unwrap();
        assert!(result.insights.is_empty());
        assert_eq!(result.chunks_used, 0);
    }
}
