use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const EMBEDDING_DIM: usize = 3072;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    Session,
    // Future: File, Git
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
}

impl Node {
    pub fn new(
        content: String,
        source: String,
        source_type: SourceType,
        embedding: Vec<f32>,
        confidence: f32,
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
        }
    }
}
