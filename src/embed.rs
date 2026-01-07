//! Embedding generation via OpenAI API
//!
//! AIDEV-NOTE: Uses text-embedding-3-small (1536 dims, 8191 max tokens)
//! - API endpoint: POST https://api.openai.com/v1/embeddings
//! - Requires OPENAI_API_KEY environment variable
//! - Uses tiktoken (cl100k_base) for accurate token counting
//! - Uses semchunk-rs for semantic chunking of long inputs

use crate::core::EMBEDDING_DIM;
use reqwest::Client;
use semchunk_rs::Chunker;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;
use tiktoken_rs::{cl100k_base, CoreBPE};

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";
const MODEL: &str = "text-embedding-3-small";

// AIDEV-NOTE: text-embedding-3-small max is 8191 tokens
// Using 8000 to leave a small safety margin
const MAX_TOKENS: usize = 8000;

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    model: &'static str,
    input: &'a str,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

/// Generate embedding for text using OpenAI API
///
/// For inputs longer than the token limit, chunks the text semantically
/// and averages the embeddings.
///
/// # Errors
/// Returns error if OPENAI_API_KEY not set or API call fails
pub async fn embed(text: &str) -> Result<Vec<f32>, EmbedError> {
    let api_key =
        std::env::var("OPENAI_API_KEY").map_err(|_| EmbedError::MissingApiKey)?;

    let text = text.trim();
    if text.is_empty() {
        return Err(EmbedError::EmptyInput);
    }

    let client = Client::new();
    let chunks = chunk_text(text);

    // Single chunk - simple case
    if chunks.len() == 1 {
        return call_openai(&client, &api_key, &chunks[0]).await;
    }

    // Multiple chunks - embed each and average
    let mut embeddings = Vec::with_capacity(chunks.len());
    for chunk in &chunks {
        let emb = call_openai(&client, &api_key, chunk).await?;
        embeddings.push(emb);
    }

    Ok(average_embeddings(&embeddings))
}

/// Call OpenAI embeddings API for a single text chunk
async fn call_openai(
    client: &Client,
    api_key: &str,
    text: &str,
) -> Result<Vec<f32>, EmbedError> {
    let request = EmbeddingRequest {
        model: MODEL,
        input: text,
    };

    let response = client
        .post(OPENAI_EMBEDDINGS_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request)
        .send()
        .await
        .map_err(|e| EmbedError::ApiError(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(EmbedError::ApiError(format!("{}: {}", status, body)));
    }

    let result: EmbeddingResponse = response
        .json()
        .await
        .map_err(|e| EmbedError::ApiError(e.to_string()))?;

    result
        .data
        .into_iter()
        .next()
        .map(|d| d.embedding)
        .ok_or_else(|| EmbedError::ApiError("No embedding in response".to_string()))
}

/// Global BPE tokenizer instance (cached for performance)
/// AIDEV-NOTE: semchunk calls count_tokens many times, so caching is critical
static BPE: OnceLock<CoreBPE> = OnceLock::new();

fn get_bpe() -> &'static CoreBPE {
    BPE.get_or_init(|| cl100k_base().expect("Failed to load cl100k_base tokenizer"))
}

/// Count tokens in text using cl100k_base encoding (used by text-embedding-3-*)
fn count_tokens(text: &str) -> usize {
    get_bpe().encode_with_special_tokens(text).len()
}

/// Split text into semantically meaningful chunks that fit within token limit
///
/// Uses semchunk-rs for semantic chunking with tiktoken for accurate token counting.
/// AIDEV-NOTE: Chunker can't be cached (not Sync), but BPE is cached so this is fast enough.
pub fn chunk_text(text: &str) -> Vec<String> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    // Check if we even need to chunk
    if count_tokens(text) <= MAX_TOKENS {
        return vec![text.to_string()];
    }

    // Create chunker per-call (cheap, the BPE tokenizer is cached)
    let chunker = Chunker::new(MAX_TOKENS, Box::new(count_tokens));
    chunker.chunk(text)
}

/// Average multiple embedding vectors
fn average_embeddings(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![0.0; EMBEDDING_DIM];
    }

    let n = embeddings.len() as f32;
    let mut result = vec![0.0; EMBEDDING_DIM];

    for emb in embeddings {
        for (i, val) in emb.iter().enumerate() {
            if i < EMBEDDING_DIM {
                result[i] += val / n;
            }
        }
    }

    result
}

#[derive(Debug)]
pub enum EmbedError {
    MissingApiKey,
    EmptyInput,
    ApiError(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::MissingApiKey => write!(f, "OPENAI_API_KEY not set"),
            EmbedError::EmptyInput => write!(f, "Empty input text"),
            EmbedError::ApiError(e) => write!(f, "OpenAI API error: {}", e),
        }
    }
}

impl std::error::Error for EmbedError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens() {
        // Simple test - "hello world" should be 2 tokens
        let count = count_tokens("hello world");
        assert!(count > 0);
        assert!(count < 10);
    }

    #[test]
    fn test_chunk_text_short() {
        let text = "Hello world";
        let chunks = chunk_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_chunk_text_empty() {
        assert!(chunk_text("").is_empty());
        assert!(chunk_text("   ").is_empty());
    }

    #[test]
    fn test_chunk_text_long() {
        // Create text with semantic structure (paragraphs) that exceeds MAX_TOKENS
        // Each paragraph is ~50 tokens, need >8000 tokens, so ~200 paragraphs
        let paragraph = "This is a sample paragraph with enough words to be meaningful. \
            It contains multiple sentences that discuss various topics. \
            The semantic chunker needs real structure to work efficiently.\n\n";
        let text = paragraph.repeat(400);

        let total_tokens = count_tokens(&text);
        assert!(
            total_tokens > MAX_TOKENS,
            "Test setup: expected > {} tokens, got {}",
            MAX_TOKENS,
            total_tokens
        );

        let chunks = chunk_text(&text);
        assert!(
            chunks.len() > 1,
            "Long text ({} tokens) should split into multiple chunks, got {}",
            total_tokens,
            chunks.len()
        );

        // Verify each chunk is within token limit
        for chunk in &chunks {
            let tokens = count_tokens(chunk);
            assert!(
                tokens <= MAX_TOKENS,
                "Chunk has {} tokens, exceeds max {}",
                tokens,
                MAX_TOKENS
            );
        }
    }

    #[test]
    fn test_average_embeddings() {
        let emb1 = vec![1.0, 2.0, 3.0];
        let emb2 = vec![3.0, 4.0, 5.0];
        let avg = average_embeddings(&[emb1, emb2]);
        // Note: average_embeddings pads to EMBEDDING_DIM
        assert_eq!(avg[0], 2.0);
        assert_eq!(avg[1], 3.0);
        assert_eq!(avg[2], 4.0);
        assert_eq!(avg.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_average_embeddings_empty() {
        let avg = average_embeddings(&[]);
        assert_eq!(avg.len(), EMBEDDING_DIM);
        assert!(avg.iter().all(|&v| v == 0.0));
    }
}
