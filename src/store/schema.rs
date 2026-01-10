use arrow_schema::{DataType, Field, Schema};
use std::sync::Arc;

use crate::core::EMBEDDING_DIM;

pub fn nodes_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("source_type", DataType::Utf8, false),
        Field::new("timestamp", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                EMBEDDING_DIM as i32,
            ),
            false,
        ),
        Field::new("confidence", DataType::Float32, false),
        Field::new("metadata", DataType::Utf8, true),
        Field::new("namespace", DataType::Utf8, false), // e.g., "personal", "team:xyz"
    ]))
}

/// Schema for tracking processed sources (sessions, files, etc.)
/// AIDEV-NOTE: source_id is the key (session UUID or canonical file path).
/// source_type distinguishes sessions from files. node_ids is JSON array.
pub fn processed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("source_id", DataType::Utf8, false),    // primary key (session UUID or file path)
        Field::new("source_type", DataType::Utf8, false),  // "session" or "file"
        Field::new("simhash", DataType::Int64, false),     // SimHash of content
        Field::new("processed_at", DataType::Utf8, false), // ISO timestamp
        Field::new("node_count", DataType::Int32, false),  // nodes created
        Field::new("node_ids", DataType::Utf8, true),      // JSON array of node UUIDs
    ]))
}
