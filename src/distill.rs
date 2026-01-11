//! Distillation - extract knowledge nodes from transcripts
//!
//! AIDEV-NOTE: Extracts knowledge from sessions per engram-spec-v1:
//! - Concepts: ideas, patterns, principles
//! - Decisions: choices with rationale
//! - Intents: goals, desires, plans
//! - Entities: people, projects, tools
//! - Questions: open/unresolved items
//!
//! All distillation goes through the Batch API (50% cost savings).
//! This module provides chunking and the extraction prompt.
//! Actual LLM calls are handled by the batch subsystem.

use semchunk_rs::Chunker;
use std::sync::OnceLock;
use tiktoken_rs::{cl100k_base, CoreBPE};

/// Minimum length for a distillation to be considered substantive
pub const MIN_CONTENT_LEN: usize = 50;

/// Max tokens per chunk when splitting large transcripts
/// AIDEV-NOTE: 40K tokens per chunk balances context richness with API limits
const MAX_CHUNK_TOKENS: usize = 40000;

/// System prompt for knowledge extraction
/// AIDEV-NOTE: Prompt based on engram-spec-v1.md extraction categories.
/// Used by batch processing.
pub const EXTRACTION_SYSTEM_PROMPT: &str = r#"You are extracting knowledge from an AI <> Human session transcript for a knowledge graph.

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

/// Global BPE tokenizer instance (cached for performance)
static BPE: OnceLock<CoreBPE> = OnceLock::new();

fn get_bpe() -> &'static CoreBPE {
    BPE.get_or_init(|| cl100k_base().expect("Failed to load cl100k_base tokenizer"))
}

/// Count tokens in text
pub fn count_tokens(text: &str) -> usize {
    get_bpe().encode_with_special_tokens(text).len()
}

/// Split transcript into semantic chunks
/// AIDEV-NOTE: Used by process_session to chunk before queueing to batch
pub fn chunk_transcript(transcript: &str) -> Vec<String> {
    let chunker = Chunker::new(MAX_CHUNK_TOKENS, Box::new(count_tokens));
    chunker.chunk(transcript)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_transcript_small() {
        let small = "Hello world";
        let chunks = chunk_transcript(small);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], small);
    }

    #[test]
    fn test_count_tokens() {
        // Simple sanity check
        let tokens = count_tokens("Hello world");
        assert!(tokens > 0);
        assert!(tokens < 10);
    }
}
