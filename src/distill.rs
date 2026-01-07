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

use crate::core::{Node, SourceType};
use crate::llm;

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

/// Extract knowledge from a formatted transcript
///
/// Returns a single ExtractedInsight containing the full LLM distillation,
/// or None if no knowledge worth capturing was found.
///
/// # Arguments
/// * `transcript` - Formatted transcript text (from format_context)
///
/// # Returns
/// Some(ExtractedInsight) if knowledge found, None otherwise
pub fn extract_knowledge(transcript: &str) -> Result<Option<ExtractedInsight>, String> {
    if transcript.trim().is_empty() {
        return Ok(None);
    }

    let content = call_extraction_llm(transcript)?;
    let content = content.trim().to_string();

    // If the LLM produced very little, treat as no knowledge
    if content.len() < MIN_CONTENT_LEN {
        return Ok(None);
    }

    Ok(Some(ExtractedInsight {
        content,
        confidence: 1.0,
    }))
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
        assert!(result.is_none());

        let result = extract_knowledge("   ").unwrap();
        assert!(result.is_none());
    }
}
