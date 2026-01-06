use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: Uuid,
    pub source_node: Uuid,
    pub target_node: Uuid,
    pub weight: f32,
    pub created_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

impl Edge {
    pub fn new(source_node: Uuid, target_node: Uuid, weight: f32) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_node,
            target_node,
            created_at: Utc::now(),
            weight,
            metadata: None,
        }
    }
}
