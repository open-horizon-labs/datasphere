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
    ]))
}

pub fn edges_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("source_node", DataType::Utf8, false),
        Field::new("target_node", DataType::Utf8, false),
        Field::new("weight", DataType::Float32, false),
        Field::new("created_at", DataType::Utf8, false),
        Field::new("metadata", DataType::Utf8, true),
    ]))
}

/// Schema for tracking processed transcripts
pub fn processed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("hash", DataType::Utf8, false),        // file content hash (primary key)
        Field::new("session_id", DataType::Utf8, false),  // session UUID
        Field::new("processed_at", DataType::Utf8, false), // ISO timestamp
        Field::new("node_count", DataType::Int32, false), // nodes created
        Field::new("file_size", DataType::Int64, false),  // original file size in bytes
    ]))
}
