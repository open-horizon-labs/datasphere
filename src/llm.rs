//! LLM utilities for calling Claude CLI, Anthropic API, or OpenAI API
//!
//! AIDEV-NOTE: Adapted from wm/src/llm.rs. Supports multiple providers:
//! - Anthropic API (preferred): direct HTTP calls when ANTHROPIC_API_KEY is set
//! - Claude CLI (fallback): uses `claude` binary with system prompt
//! - OpenAI API: direct HTTP calls to chat completions endpoint
//!
//! Set DATASPHERE_MODEL to control model:
//! - "haiku", "sonnet", "opus" → Anthropic API (if key set) or Claude CLI
//! - "gpt-4o", "gpt-4.5-preview", "o1", etc. → OpenAI API

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;
use tokio::process::Command;

const OPENAI_CHAT_URL: &str = "https://api.openai.com/v1/chat/completions";
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";

/// Error types for LLM calls
#[derive(Debug)]
pub enum LlmError {
    /// Rate limit hit - caller should back off
    RateLimit(String),
    /// Other errors
    Other(String),
}

impl std::fmt::Display for LlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmError::RateLimit(msg) => write!(f, "Rate limit: {}", msg),
            LlmError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for LlmError {}

/// Check if an error message indicates a rate limit
/// NOTE: Patterns may need updates as Claude CLI error messages evolve
fn is_rate_limit_error(error_text: &str) -> bool {
    let lower = error_text.to_lowercase();
    lower.contains("hit your limit")
        || lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("429")
}

/// Cached path to claude CLI binary (resolved once at first use)
static CLAUDE_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Get the path to the claude CLI, resolving it once via PATH lookup
fn get_claude_path() -> &'static PathBuf {
    CLAUDE_PATH.get_or_init(|| {
        which::which("claude").unwrap_or_else(|_| PathBuf::from("claude"))
    })
}

/// Result of calling the LLM with a marker-based response format
#[derive(Debug)]
pub struct MarkerResponse {
    /// Whether the marker indicated yes/true
    pub is_positive: bool,

    /// Content after the marker line (if positive)
    pub content: String,
}

// OpenAI API types
#[derive(Debug, Serialize)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: Vec<OpenAIMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
}

/// Check if model uses new-style max_completion_tokens param
fn uses_completion_tokens(model: &str) -> bool {
    model.starts_with("gpt-5")
        || model.starts_with("o1")
        || model.starts_with("o3")
}

#[derive(Debug, Serialize)]
struct OpenAIMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponseMessage {
    content: String,
}

/// Check if model name indicates OpenAI
fn is_openai_model(model: &str) -> bool {
    model.starts_with("gpt-")
        || model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("chatgpt")
}

// Anthropic API types
#[derive(Debug, Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<AnthropicMessage<'a>>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContent {
    text: String,
}

/// Map short model names to full Anthropic model IDs
fn resolve_anthropic_model(model: &str) -> &str {
    match model {
        "haiku" => "claude-haiku-4-5",
        "sonnet" => "claude-sonnet-4-5",
        "opus" => "claude-opus-4-5",
        _ => model, // Pass through full model IDs
    }
}

/// Call Anthropic Messages API directly
async fn call_anthropic(
    system_prompt: &str,
    message: &str,
    model: &str,
) -> Result<String, LlmError> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .map_err(|_| LlmError::Other("ANTHROPIC_API_KEY not set".to_string()))?;

    let resolved_model = resolve_anthropic_model(model);
    let client = Client::new();
    let request = AnthropicRequest {
        model: resolved_model,
        max_tokens: 16000,
        system: system_prompt,
        messages: vec![AnthropicMessage {
            role: "user",
            content: message,
        }],
    };

    let response = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", &api_key)
        .header("anthropic-version", ANTHROPIC_API_VERSION)
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| LlmError::Other(format!("Anthropic request failed: {}", e)))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| LlmError::Other(format!("Failed to read response: {}", e)))?;

    if status == 429 {
        return Err(LlmError::RateLimit(body));
    }

    if !status.is_success() {
        return Err(LlmError::Other(format!(
            "Anthropic API error ({}): {}",
            status, body
        )));
    }

    let parsed: AnthropicResponse = serde_json::from_str(&body)
        .map_err(|e| LlmError::Other(format!("Failed to parse response: {} - {}", e, body)))?;

    parsed
        .content
        .first()
        .map(|c| c.text.clone())
        .ok_or_else(|| LlmError::Other("No content in response".to_string()))
}

/// Call OpenAI chat completions API
async fn call_openai(
    system_prompt: &str,
    message: &str,
    model: &str,
) -> Result<String, LlmError> {
    let api_key = std::env::var("OPENAI_API_KEY")
        .map_err(|_| LlmError::Other("OPENAI_API_KEY not set".to_string()))?;

    let client = Client::new();
    let use_completion_tokens = uses_completion_tokens(model);
    let request = OpenAIChatRequest {
        model,
        messages: vec![
            OpenAIMessage {
                role: "system",
                content: system_prompt,
            },
            OpenAIMessage {
                role: "user",
                content: message,
            },
        ],
        max_tokens: if use_completion_tokens { None } else { Some(16000) },
        max_completion_tokens: if use_completion_tokens { Some(16000) } else { None },
    };

    let response = client
        .post(OPENAI_CHAT_URL)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await
        .map_err(|e| LlmError::Other(format!("OpenAI request failed: {}", e)))?;

    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| LlmError::Other(format!("Failed to read response: {}", e)))?;

    if status == 429 {
        return Err(LlmError::RateLimit(body));
    }

    if !status.is_success() {
        return Err(LlmError::Other(format!(
            "OpenAI API error ({}): {}",
            status, body
        )));
    }

    let parsed: OpenAIChatResponse = serde_json::from_str(&body)
        .map_err(|e| LlmError::Other(format!("Failed to parse response: {} - {}", e, body)))?;

    parsed
        .choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| LlmError::Other("No choices in response".to_string()))
}

/// Call LLM with a system prompt and message (async)
///
/// Provider selection priority:
/// 1. OpenAI models (gpt-*, o1, o3) → OpenAI API
/// 2. ANTHROPIC_API_KEY set → Anthropic API (preferred, avoids CLI session limits)
/// 3. Fallback → Claude CLI
///
/// # Errors
/// - `LlmError::RateLimit` if the API rate limit was hit
/// - `LlmError::Other` for all other errors
pub async fn call_claude(system_prompt: &str, message: &str) -> Result<String, LlmError> {
    let model = std::env::var("DATASPHERE_MODEL").unwrap_or_else(|_| "sonnet".to_string());

    // 1. OpenAI models → OpenAI API
    if is_openai_model(&model) {
        return call_openai(system_prompt, message, &model).await;
    }

    // 2. ANTHROPIC_API_KEY set → Anthropic API (preferred)
    if std::env::var("ANTHROPIC_API_KEY").is_ok() {
        return call_anthropic(system_prompt, message, &model).await;
    }

    // 3. Fallback: Claude CLI
    let mut cmd = Command::new(get_claude_path());
    cmd.arg("-p")
        .arg("--output-format")
        .arg("json")
        .arg("--no-session-persistence")
        .arg("--system-prompt")
        .arg(system_prompt)
        .arg(message)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .stdin(Stdio::null())
        .env("WM_DISABLED", "1")
        .env("SUPEREGO_DISABLED", "1")
        .env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "16000")
        .arg("--model")
        .arg(&model);

    let output = cmd
        .output()
        .await
        .map_err(|e| LlmError::Other(format!("Failed to run claude CLI: {}", e)))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Try to parse JSON even on failure - Claude CLI may return is_error:true with valid JSON
    if let Ok(cli_response) = serde_json::from_str::<serde_json::Value>(&stdout) {
        // Check for is_error flag in response
        if cli_response.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false) {
            let error_msg = cli_response
                .get("result")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");

            if is_rate_limit_error(error_msg) {
                return Err(LlmError::RateLimit(error_msg.to_string()));
            }
            return Err(LlmError::Other(error_msg.to_string()));
        }

        // Success case - extract result
        if let Some(result) = cli_response.get("result").and_then(|v| v.as_str()) {
            return Ok(result.to_string());
        }

        return Err(LlmError::Other("Claude CLI response missing 'result' field".to_string()));
    }

    // Couldn't parse JSON - check raw output for rate limit indicators
    let combined = format!("{} {}", stdout, stderr);
    if is_rate_limit_error(&combined) {
        return Err(LlmError::RateLimit(combined));
    }

    if !output.status.success() {
        return Err(LlmError::Other(format!(
            "Claude CLI failed (exit {:?}):\nstderr: {}\nstdout: {}",
            output.status.code(),
            stderr,
            stdout
        )));
    }

    Err(LlmError::Other("Failed to parse Claude CLI response".to_string()))
}

/// Parse a marker-based response (e.g., "HAS_KNOWLEDGE: YES\n<content>")
///
/// The marker format is: `MARKER_NAME: YES|NO|TRUE|FALSE`
/// If positive, content is everything after the marker line.
pub fn parse_marker_response(text: &str, marker_name: &str) -> MarkerResponse {
    let lines: Vec<&str> = text.lines().collect();
    let marker_prefix = format!("{}:", marker_name);

    for (i, line) in lines.iter().enumerate() {
        let stripped = strip_markdown_prefix(line);

        if let Some(value) = stripped.strip_prefix(&marker_prefix) {
            let value = value.trim().to_uppercase();
            if value == "YES" || value == "TRUE" {
                let content = lines[i + 1..].join("\n").trim().to_string();
                return MarkerResponse {
                    is_positive: true,
                    content,
                };
            }
            return MarkerResponse {
                is_positive: false,
                content: String::new(),
            };
        }
    }

    // No marker found
    MarkerResponse {
        is_positive: false,
        content: String::new(),
    }
}

/// Strip markdown prefixes from a line for lenient marker matching
fn strip_markdown_prefix(line: &str) -> &str {
    line.trim().trim_start_matches(['#', '>', '*']).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_marker_yes() {
        let text = "HAS_KNOWLEDGE: YES\n- First insight\n- Second insight";
        let result = parse_marker_response(text, "HAS_KNOWLEDGE");
        assert!(result.is_positive);
        assert_eq!(result.content, "- First insight\n- Second insight");
    }

    #[test]
    fn test_parse_marker_no() {
        let text = "HAS_KNOWLEDGE: NO";
        let result = parse_marker_response(text, "HAS_KNOWLEDGE");
        assert!(!result.is_positive);
        assert!(result.content.is_empty());
    }

    #[test]
    fn test_parse_marker_with_markdown() {
        let text = "## HAS_RELEVANT: TRUE\nSome content here";
        let result = parse_marker_response(text, "HAS_RELEVANT");
        assert!(result.is_positive);
        assert_eq!(result.content, "Some content here");
    }

    #[test]
    fn test_parse_marker_not_found() {
        let text = "No markers here";
        let result = parse_marker_response(text, "HAS_KNOWLEDGE");
        assert!(!result.is_positive);
        assert!(result.content.is_empty());
    }

    #[test]
    fn test_is_rate_limit_error() {
        // Common rate limit messages
        assert!(is_rate_limit_error("You've hit your limit for today"));
        assert!(is_rate_limit_error("Rate limit exceeded"));
        assert!(is_rate_limit_error("Too many requests, please try again later"));
        assert!(is_rate_limit_error("Error 429: Too many requests"));

        // Case insensitive
        assert!(is_rate_limit_error("HIT YOUR LIMIT"));
        assert!(is_rate_limit_error("RATE LIMIT"));

        // Not rate limit errors
        assert!(!is_rate_limit_error("Connection failed"));
        assert!(!is_rate_limit_error("Invalid API key"));
        assert!(!is_rate_limit_error("Server error 500"));
    }

    #[test]
    fn test_llm_error_display() {
        let rate_limit = LlmError::RateLimit("hit your limit".to_string());
        assert_eq!(format!("{}", rate_limit), "Rate limit: hit your limit");

        let other = LlmError::Other("connection failed".to_string());
        assert_eq!(format!("{}", other), "connection failed");
    }
}
