use arrow_array::{
    Array, Float32Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator, StringArray,
    types::Float32Type, FixedSizeListArray,
};
use chrono::{DateTime, Utc};
use futures::TryStreamExt;
use lancedb::{connect, Connection, Table, query::{QueryBase, ExecutableQuery}};
use std::sync::Arc;
use uuid::Uuid;

use crate::core::{Edge, Node, SourceType, EMBEDDING_DIM};
use super::schema::{edges_schema, nodes_schema, processed_schema};

/// Record of a processed transcript
#[derive(Debug, Clone)]
pub struct Processed {
    pub hash: String,
    pub session_id: String,
    pub processed_at: DateTime<Utc>,
    pub node_count: i32,
    pub file_size: i64,
}

pub struct Store {
    #[allow(dead_code)]
    db: Connection,
    nodes: Table,
    edges: Table,
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

        // Open or create edges table
        let edges = match db.open_table("edges").execute().await {
            Ok(t) => t,
            Err(_) => {
                db.create_empty_table("edges", edges_schema())
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

        Ok(Self { db, nodes, edges, processed })
    }

    pub async fn insert_node(&self, node: &Node) -> Result<(), lancedb::Error> {
        let batch = node_to_batch(node)?;
        let schema = nodes_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.nodes.add(Box::new(batches)).execute().await?;
        Ok(())
    }

    pub async fn insert_edge(&self, edge: &Edge) -> Result<(), lancedb::Error> {
        let batch = edge_to_batch(edge)?;
        let schema = edges_schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        self.edges.add(Box::new(batches)).execute().await?;
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

    pub async fn get_edges_for(&self, node_id: Uuid) -> Result<Vec<Edge>, lancedb::Error> {
        let id_str = node_id.to_string();
        let filter = format!(
            "source_node = '{}' OR target_node = '{}'",
            id_str, id_str
        );

        let mut results = self
            .edges
            .query()
            .only_if(filter)
            .execute()
            .await?;

        let mut edges = Vec::new();
        while let Some(batch) = results.try_next().await? {
            edges.extend(batch_to_edges(&batch)?);
        }
        Ok(edges)
    }

    /// Check if a transcript hash has been processed
    pub async fn is_processed(&self, hash: &str) -> Result<bool, lancedb::Error> {
        let filter = format!("hash = '{}'", hash);

        let mut results = self
            .processed
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?;

        Ok(results.try_next().await?.is_some())
    }

    /// Get processed record by hash
    pub async fn get_processed(&self, hash: &str) -> Result<Option<Processed>, lancedb::Error> {
        let filter = format!("hash = '{}'", hash);

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

    /// Get an edge by ID
    pub async fn get_edge(&self, id: Uuid) -> Result<Option<Edge>, lancedb::Error> {
        let filter = format!("id = '{}'", id);

        let mut results = self
            .edges
            .query()
            .only_if(filter)
            .limit(1)
            .execute()
            .await?;

        if let Some(batch) = results.try_next().await? {
            let edges = batch_to_edges(&batch)?;
            Ok(edges.into_iter().next())
        } else {
            Ok(None)
        }
    }

    /// Count total edges in the store
    pub async fn count_edges(&self) -> Result<usize, lancedb::Error> {
        self.edges.count_rows(None).await
    }

    /// Count total processed transcripts
    pub async fn count_processed(&self) -> Result<usize, lancedb::Error> {
        self.processed.count_rows(None).await
    }
}

fn node_to_batch(node: &Node) -> Result<RecordBatch, lancedb::Error> {
    let ids = StringArray::from(vec![node.id.to_string()]);
    let contents = StringArray::from(vec![node.content.as_str()]);
    let sources = StringArray::from(vec![node.source.as_str()]);
    let source_types = StringArray::from(vec![match node.source_type {
        SourceType::Session => "session",
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

fn edge_to_batch(edge: &Edge) -> Result<RecordBatch, lancedb::Error> {
    let ids = StringArray::from(vec![edge.id.to_string()]);
    let source_nodes = StringArray::from(vec![edge.source_node.to_string()]);
    let target_nodes = StringArray::from(vec![edge.target_node.to_string()]);
    let weights = Float32Array::from(vec![edge.weight]);
    let created_ats = StringArray::from(vec![edge.created_at.to_rfc3339()]);
    let metadata = StringArray::from(vec![edge
        .metadata
        .as_ref()
        .map(|m| m.to_string())]);

    let batch = RecordBatch::try_new(
        edges_schema(),
        vec![
            Arc::new(ids),
            Arc::new(source_nodes),
            Arc::new(target_nodes),
            Arc::new(weights),
            Arc::new(created_ats),
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

fn batch_to_edges(batch: &RecordBatch) -> Result<Vec<Edge>, lancedb::Error> {
    let ids = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "id column not found".to_string(),
        })?;

    let source_nodes = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "source_node column not found".to_string(),
        })?;

    let target_nodes = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "target_node column not found".to_string(),
        })?;

    let weights = batch
        .column(3)
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "weight column not found".to_string(),
        })?;

    let created_ats = batch
        .column(4)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "created_at column not found".to_string(),
        })?;

    let metadata_col = batch
        .column(5)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "metadata column not found".to_string(),
        })?;

    let mut edges = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let metadata = if metadata_col.is_null(i) {
            None
        } else {
            metadata_col
                .value(i)
                .parse::<serde_json::Value>()
                .ok()
        };

        let edge = Edge {
            id: ids
                .value(i)
                .parse()
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid uuid".to_string(),
                })?,
            source_node: source_nodes
                .value(i)
                .parse()
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid source_node uuid".to_string(),
                })?,
            target_node: target_nodes
                .value(i)
                .parse()
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid target_node uuid".to_string(),
                })?,
            weight: weights.value(i),
            created_at: chrono::DateTime::parse_from_rfc3339(created_ats.value(i))
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid created_at timestamp".to_string(),
                })?
                .with_timezone(&chrono::Utc),
            metadata,
        };
        edges.push(edge);
    }

    Ok(edges)
}

fn processed_to_batch(record: &Processed) -> Result<RecordBatch, lancedb::Error> {
    let hashes = StringArray::from(vec![record.hash.as_str()]);
    let session_ids = StringArray::from(vec![record.session_id.as_str()]);
    let processed_ats = StringArray::from(vec![record.processed_at.to_rfc3339()]);
    let node_counts = Int32Array::from(vec![record.node_count]);
    let file_sizes = Int64Array::from(vec![record.file_size]);

    let batch = RecordBatch::try_new(
        processed_schema(),
        vec![
            Arc::new(hashes),
            Arc::new(session_ids),
            Arc::new(processed_ats),
            Arc::new(node_counts),
            Arc::new(file_sizes),
        ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    Ok(batch)
}

fn batch_to_processed(batch: &RecordBatch) -> Result<Vec<Processed>, lancedb::Error> {
    let hashes = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "hash column not found".to_string(),
        })?;

    let session_ids = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "session_id column not found".to_string(),
        })?;

    let processed_ats = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "processed_at column not found".to_string(),
        })?;

    let node_counts = batch
        .column(3)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "node_count column not found".to_string(),
        })?;

    let file_sizes = batch
        .column(4)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| lancedb::Error::InvalidInput {
            message: "file_size column not found".to_string(),
        })?;

    let mut records = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let record = Processed {
            hash: hashes.value(i).to_string(),
            session_id: session_ids.value(i).to_string(),
            processed_at: chrono::DateTime::parse_from_rfc3339(processed_ats.value(i))
                .map_err(|_| lancedb::Error::InvalidInput {
                    message: "invalid processed_at timestamp".to_string(),
                })?
                .with_timezone(&chrono::Utc),
            node_count: node_counts.value(i),
            file_size: file_sizes.value(i),
        };
        records.push(record);
    }

    Ok(records)
}
