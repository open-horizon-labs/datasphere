use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// AIDEV-NOTE: Using text-embedding-3-small (1536 dims, 8191 max tokens, $0.02/1M tokens)
// If switching to text-embedding-3-large, change to 3072 dims
pub const EMBEDDING_DIM: usize = 1536;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Session,
    File,
    // Future: Git
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: Uuid,
    pub content: String,
    pub source: String,
    pub source_type: SourceType,
    pub timestamp: DateTime<Utc>,
    pub embedding: Vec<f32>,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    #[serde(default = "default_namespace")]
    pub namespace: String,
}

fn default_namespace() -> String {
    "personal".to_string()
}

impl Node {
    pub fn new(
        content: String,
        source: String,
        source_type: SourceType,
        embedding: Vec<f32>,
        confidence: f32,
    ) -> Self {
        Self::with_namespace(content, source, source_type, embedding, confidence, "personal".to_string())
    }

    pub fn with_namespace(
        content: String,
        source: String,
        source_type: SourceType,
        embedding: Vec<f32>,
        confidence: f32,
        namespace: String,
    ) -> Self {
        debug_assert!(
            embedding.len() == EMBEDDING_DIM,
            "expected {} dims, got {}",
            EMBEDDING_DIM,
            embedding.len()
        );
        Self {
            id: Uuid::new_v4(),
            content,
            source,
            source_type,
            timestamp: Utc::now(),
            embedding,
            confidence,
            metadata: None,
            namespace,
        }
    }
}
