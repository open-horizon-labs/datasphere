//! LLM utilities for calling Claude CLI
//!
//! AIDEV-NOTE: Adapted from wm/src/llm.rs. Engram uses same pattern:
//! call Claude CLI with a system prompt, parse response using text markers.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;
use tokio::process::Command;

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

/// Call Claude CLI with a system prompt and message (async)
///
/// Returns the raw result string from the Claude CLI JSON response.
/// Sets WM_DISABLED and SUPEREGO_DISABLED to prevent recursion.
///
/// # Errors
/// - `LlmError::RateLimit` if the API rate limit was hit
/// - `LlmError::Other` for all other errors
pub async fn call_claude(system_prompt: &str, message: &str) -> Result<String, LlmError> {
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
        .env("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "16000");

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
