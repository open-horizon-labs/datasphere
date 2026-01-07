use arrow_array::{
    Array, Float32Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    types::Float32Type, FixedSizeListArray,
};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::{connect, Connection, Table, query::{QueryBase, ExecutableQuery}, DistanceType};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::{Node, SourceType, EMBEDDING_DIM};
use super::schema::{nodes_schema, processed_schema};

/// Record of a processed source (session or file)
/// AIDEV-NOTE: source_id is the key (session UUID or canonical file path).
/// source_type distinguishes sessions from files. node_ids tracks all created nodes.
#[derive(Debug, Clone)]
pub struct Processed {
    pub source_id: String,       // session UUID or canonical file path
    pub source_type: String,     // "session" or "file"
    pub simhash: i64,
    pub processed_at: DateTime<Utc>,
    pub node_count: i32,
    pub node_ids: Vec<String>,   // UUIDs of created nodes (for updates/deletion)
}

pub struct Store {
    #[allow(dead_code)]
    db: Connection,
    nodes: Table,
    processed: Table,
}

impl Store {
    pub async fn open(path: &str) -> Result<Self, lancedb::Error> {
        let db = connect(path).execute().await?;

        // Open or create nodes table
        let nodes = match db.open_table("nodes").execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table("nodes", nodes_schema())
                    .execute()
                    .await?
            }
        };

        // Open or create processed table
        let processed = match db.open_table("processed").execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table("processed", processed_schema())
                    .execute()
                    .await?
            }
        };

        Ok(Self { db, nodes, processed })
    }

    pub async fn insert_node(&self, node: &Node) -> Result<(), lancedb::Error> {
        let batch = node_to_batch(node)?;
        let schema = nodes_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.nodes.add(Box::new(batches)).execute().await?;
        Ok(())
    }

    pub async fn search_similar(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Node>, lancedb::Error> {
        let mut results = self
            .nodes
            .vector_search(embedding)?
            .limit(limit)
            .execute()
            .await?;

        let mut nodes = Vec::new();
        while let Some(batch) = results.try_next().await? {
            nodes.extend(batch_to_nodes(&batch)?);
        }
        Ok(nodes)
    }

    /// Search for similar nodes and return with similarity scores
    /// AIDEV-NOTE: Uses cosine distance. Similarity = 1.0 - distance.
    /// OpenAI embeddings are normalized, so cosine distance is appropriate.
    pub async fn search_similar_with_scores(
        &self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<(Node, f32)>, lancedb::Error> {
        let mut results = self
            .nodes
            .vector_search(embedding)?
            .distance_type(DistanceType::Cosine)
            .limit(limit)
            .execute()
            .await?;

        let mut nodes_with_scores = Vec::new();
        while let Some(batch) = results.try_next().await? {
            let nodes = batch_to_nodes(&batch)?;
            let distances = extract_distances(&batch)?;

            for (node, distance) in nodes.into_iter().zip(distances.into_iter()) {
                // Cosine distance to similarity: similarity = 1.0 - distance
                let similarity = 1.0 - distance;
                nodes_with_scores.push((node, similarity));
            }
        }
        Ok(nodes_with_scores)
    }

    /// Get processed record by source_id
    pub async fn get_processed(&self, source_id: &str) -> Result<Option<Processed>, lancedb::Error> {
        let filter = format!("source_id = '{}'", source_id);

        let mut results = self
            .processed
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = results.try_next().await? {
            let records = batch_to_processed(&batch)?;
            Ok(records.into_iter().next())
        } else {
            Ok(None)
        }
    }

    /// Record that a transcript has been processed
    pub async fn insert_processed(&self, record: &Processed) -> Result<(), lancedb::Error> {
        let batch = processed_to_batch(record)?;
        let schema = processed_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.processed.add(Box::new(batches)).execute().await?;
        Ok(())
    }

    /// Delete processed record by source_id (for re-processing)
    pub async fn delete_processed(&self, source_id: &str) -> Result<(), lancedb::Error> {
        let filter = format!("source_id = '{}'", source_id);
        self.processed.delete(&filter).await?;
        Ok(())
    }

    /// Delete multiple nodes by ID (for re-processing files with multiple chunks)
    pub async fn delete_nodes(&self, ids: &[Uuid]) -> Result<(), lancedb::Error> {
        for id in ids {
            let filter = format!("id = '{}'", id);
            self.nodes.delete(&filter).await?;
        }
        Ok(())
    }

    /// Get a node by ID
    pub async fn get_node(&self, id: Uuid) -> Result<Option<Node>, lancedb::Error> {
        let filter = format!("id = '{}'", id);

        let mut results = self
            .nodes
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = results.try_next().await? {
            let nodes = batch_to_nodes(&batch)?;
            Ok(nodes.into_iter().next())
        } else {
            Ok(None)
        }
    }

    /// Count total nodes in the store
    pub async fn count_nodes(&self) -> Result<usize, lancedb::Error> {
        self.nodes.count_rows(None).await
    }

    /// Count total processed transcripts
    pub async fn count_processed(&self) -> Result<usize, lancedb::Error> {
        self.processed.count_rows(None).await
    }

    /// List recent nodes (by timestamp, newest first)
    pub async fn list_nodes(&self, limit: usize) -> Result<Vec<Node>, lancedb::Error> {
        let mut results = self
            .nodes
            .query()
            .limit(limit)
            .execute()
            .await?;

        let mut nodes = Vec::new();
        while let Some(batch) = results.try_next().await? {
            nodes.extend(batch_to_nodes(&batch)?);
        }
        Ok(nodes)
    }

    /// Delete a node by ID (for re-processing)
    pub async fn delete_node(&self, id: Uuid) -> Result<(), lancedb::Error> {
        let filter = format!("id = '{}'", id);
        self.nodes.delete(&filter).await?;
        Ok(())
    }
}

fn node_to_batch(node: &Node) -> Result<RecordBatch, lancedb::Error> {
    let ids = StringArray::from(vec![node.id.to_string()]);
    let contents = StringArray::from(vec![node.content.as_str()]);
    let sources = StringArray::from(vec![node.source.as_str()]);
    let source_types = StringArray::from(vec![match node.source_type {
        SourceType::Session => "session",
        SourceType::File => "file",
    }]);
    let timestamps = StringArray::from(vec![node.timestamp.to_rfc3339()]);

    let embeddings = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![Some(node.embedding.iter().map(|&v| Some(v)).collect::<Vec<_>>())],
        EMBEDDING_DIM as i32,
    );

    let confidences = Float32Array::from(vec![node.confidence]);
    let metadata = StringArray::from(vec![node
        .metadata
        .as_ref()
        .map(|m| m.to_string())]);

    let batch = RecordBatch::try_new(
        nodes_schema(),
        vec![
            Arc::new(ids),
            Arc::new(contents),
            Arc::new(sources),
            Arc::new(source_types),
            Arc::new(timestamps),
            Arc::new(embeddings),
            Arc::new(confidences),
            Arc::new(metadata),
        ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    Ok(batch)
}

fn batch_to_nodes(batch: &RecordBatch) -> Result<Vec<Node>, lancedb::Error> {
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "id column not found".to_string(),
        })?;

    let contents = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "content column not found".to_string(),
        })?;

    let sources = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "source column not found".to_string(),
        })?;

    let source_types = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "source_type column not found".to_string(),
        })?;

    let timestamps = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "timestamp column not found".to_string(),
        })?;

    let embeddings = batch
        .column(5)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "embedding column not found".to_string(),
        })?;

    let confidences = batch
        .column(6)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "confidence column not found".to_string(),
        })?;

    let metadata_col = batch
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "metadata column not found".to_string(),
        })?;

    let mut nodes = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let embedding_array = embeddings.value(i);
        let embedding_f32 = embedding_array
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| lancedb::Error::InvalidInput {
                message: "embedding values not f32".to_string(),
            })?;
        let embedding: Vec<f32> = embedding_f32.values().to_vec();

        let metadata = if metadata_col.is_null(i) {
            None
        } else {
            metadata_col
                .value(i)
                .parse::<serde_json::Value>()
                .ok()
        };

        let node = Node {
            id: ids
                .value(i)
                .parse()
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid uuid".to_string(),
                })?,
            content: contents.value(i).to_string(),
            source: sources.value(i).to_string(),
            source_type: match source_types.value(i) {
                "session" => SourceType::Session,
                "file" => SourceType::File,
                _ => SourceType::Session,
            },
            timestamp: chrono::DateTime::parse_from_rfc3339(timestamps.value(i))
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid timestamp".to_string(),
                })?
                .with_timezone(&chrono::Utc),
            embedding,
            confidence: confidences.value(i),
            metadata,
        };
        nodes.push(node);
    }

    Ok(nodes)
}

/// Extract _distance column from vector search results
fn extract_distances(batch: &RecordBatch) -> Result<Vec<f32>, lancedb::Error> {
    // LanceDB adds _distance column to vector search results
    let distance_col = batch
        .column_by_name("_distance")
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "_distance column not found in vector search results".to_string(),
        })?;

    let distances = distance_col
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "_distance column is not Float32".to_string(),
        })?;

    Ok(distances.values().to_vec())
}

fn processed_to_batch(record: &Processed) -> Result<RecordBatch, lancedb::Error> {
    let source_ids = StringArray::from(vec![record.source_id.as_str()]);
    let source_types = StringArray::from(vec![record.source_type.as_str()]);
    let simhashes = Int64Array::from(vec![record.simhash]);
    let processed_ats = StringArray::from(vec![record.processed_at.to_rfc3339()]);
    let node_counts = Int32Array::from(vec![record.node_count]);
    // Serialize node_ids as JSON array
    let node_ids_json = serde_json::to_string(&record.node_ids).unwrap_or_else(|_| "[]".to_string());
    let node_ids = StringArray::from(vec![Some(node_ids_json.as_str())]);

    let batch = RecordBatch::try_new(
        processed_schema(),
        vec![
            Arc::new(source_ids),
            Arc::new(source_types),
            Arc::new(simhashes),
            Arc::new(processed_ats),
            Arc::new(node_counts),
            Arc::new(node_ids),
        ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    Ok(batch)
}

fn batch_to_processed(batch: &RecordBatch) -> Result<Vec<Processed>, lancedb::Error> {
    let source_ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "source_id column not found".to_string(),
        })?;

    let source_types = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "source_type column not found".to_string(),
        })?;

    let simhashes = batch
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "simhash column not found".to_string(),
        })?;

    let processed_ats = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "processed_at column not found".to_string(),
        })?;

    let node_counts = batch
        .column(4)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "node_count column not found".to_string(),
        })?;

    let node_ids_col = batch
        .column(5)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "node_ids column not found".to_string(),
        })?;

    let mut records = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        // Parse node_ids from JSON array
        let node_ids: Vec<String> = if node_ids_col.is_null(i) {
            Vec::new()
        } else {
            serde_json::from_str(node_ids_col.value(i)).unwrap_or_default()
        };

        let record = Processed {
            source_id: source_ids.value(i).to_string(),
            source_type: source_types.value(i).to_string(),
            simhash: simhashes.value(i),
            processed_at: chrono::DateTime::parse_from_rfc3339(processed_ats.value(i))
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid processed_at timestamp".to_string(),
                })?
                .with_timezone(&chrono::Utc),
            node_count: node_counts.value(i),
            node_ids,
        };
        records.push(record);
    }

    Ok(records)
}
