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
//! For large transcripts (>50K chars), we chunk semantically, distill each
//! chunk separately, then synthesize into a single coherent summary.

use crate::core::{Node, SourceType};
use crate::llm;
use semchunk_rs::Chunker;
use std::sync::OnceLock;
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
    pub insight: Option<ExtractedInsight>,
    pub chunks_used: usize,
}

/// Extract knowledge from a formatted transcript
///
/// Returns a single ExtractedInsight containing the full LLM distillation,
/// or None if no knowledge worth capturing was found.
///
/// For large transcripts, chunks semantically, distills each chunk,
/// then synthesizes into one coherent summary.
///
/// # Arguments
/// * `transcript` - Formatted transcript text (from format_context)
///
/// # Returns
/// ExtractionResult with insight (if found) and chunk count
pub fn extract_knowledge(transcript: &str) -> Result<ExtractionResult, String> {
    if transcript.trim().is_empty() {
        return Ok(ExtractionResult {
            insight: None,
            chunks_used: 0,
        });
    }

    let token_count = count_tokens(transcript);

    // Small transcript: single LLM call
    if token_count <= CHUNK_THRESHOLD_TOKENS {
        let content = call_extraction_llm(transcript)?;
        let content = content.trim().to_string();

        if content.len() < MIN_CONTENT_LEN {
            return Ok(ExtractionResult {
                insight: None,
                chunks_used: 1,
            });
        }

        return Ok(ExtractionResult {
            insight: Some(ExtractedInsight {
                content,
                confidence: 1.0,
            }),
            chunks_used: 1,
        });
    }

    // Large transcript: chunk → distill each → synthesize
    let chunks = chunk_transcript(transcript);
    let chunk_count = chunks.len();

    // Distill each chunk
    let mut chunk_distillations = Vec::with_capacity(chunk_count);
    for (i, chunk) in chunks.iter().enumerate() {
        eprintln!("    Distilling chunk {}/{}...", i + 1, chunk_count);
        match call_extraction_llm(chunk) {
            Ok(distillation) => {
                let trimmed = distillation.trim();
                if !trimmed.is_empty() && trimmed.len() >= MIN_CONTENT_LEN {
                    chunk_distillations.push(distillation);
                }
            }
            Err(e) => {
                eprintln!("    Warning: chunk {} failed: {}", i + 1, e);
            }
        }
    }

    if chunk_distillations.is_empty() {
        return Ok(ExtractionResult {
            insight: None,
            chunks_used: chunk_count,
        });
    }

    // Synthesize chunk distillations into final summary
    eprintln!("    Synthesizing {} chunk distillations...", chunk_distillations.len());
    let synthesized = call_synthesis_llm(&chunk_distillations)?;
    let synthesized = synthesized.trim().to_string();

    if synthesized.len() < MIN_CONTENT_LEN {
        return Ok(ExtractionResult {
            insight: None,
            chunks_used: chunk_count,
        });
    }

    Ok(ExtractionResult {
        insight: Some(ExtractedInsight {
            content: synthesized,
            confidence: 1.0,
        }),
        chunks_used: chunk_count,
    })
}

/// Call LLM to extract knowledge from transcript
fn call_extraction_llm(transcript: &str) -> Result<String, String> {
    // AIDEV-NOTE: Prompt based on engram-spec-v1.md extraction categories.
    // No structured format enforced - LLM narrates naturally.
    // The whole response becomes the node content.
    let system_prompt = r#"You are extracting knowledge from an AI coding session transcript for a knowledge graph.

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

If the session has no substantive knowledge, just say so briefly."#;

    let message = format!("TRANSCRIPT:\n{}", transcript);
    llm::call_claude(system_prompt, &message)
}

/// Call LLM to synthesize multiple chunk distillations into one summary
fn call_synthesis_llm(distillations: &[String]) -> Result<String, String> {
    let system_prompt = r#"You are synthesizing knowledge summaries from multiple chunks of an AI coding session.

Each chunk summary below was extracted from a different part of the same session transcript.

Your task:
1. Combine the insights into a single coherent summary
2. Remove redundant or duplicate information
3. Preserve all unique knowledge, decisions, concepts, and questions
4. Organize by theme rather than by chunk order
5. Keep the same format (prose, bullets, headers) as appropriate

The result should read as if it was extracted from the whole session at once, not as a collection of separate summaries."#;

    let mut message = String::from("CHUNK SUMMARIES:\n\n");
    for (i, distillation) in distillations.iter().enumerate() {
        message.push_str(&format!("--- Chunk {} ---\n{}\n\n", i + 1, distillation));
    }

    llm::call_claude(system_prompt, &message)
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

    #[test]
    fn test_extract_knowledge_empty_transcript() {
        let result = extract_knowledge("").unwrap();
        assert!(result.insight.is_none());
        assert_eq!(result.chunks_used, 0);

        let result = extract_knowledge("   ").unwrap();
        assert!(result.insight.is_none());
        assert_eq!(result.chunks_used, 0);
    }
}
