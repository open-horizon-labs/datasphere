//! LLM utilities for calling Claude CLI
//!
//! AIDEV-NOTE: Adapted from wm/src/llm.rs. Engram uses same pattern:
//! call Claude CLI with a system prompt, parse response using text markers.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::OnceLock;
use tokio::process::Command;

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
pub async fn call_claude(system_prompt: &str, message: &str) -> Result<String, String> {
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
        .map_err(|e| format!("Failed to run claude CLI: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(format!(
            "Claude CLI failed (exit {:?}):\nstderr: {}\nstdout: {}",
            output.status.code(),
            stderr,
            stdout
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse Claude CLI JSON wrapper to extract result field
    let cli_response: serde_json::Value = serde_json::from_str(&stdout)
        .map_err(|e| format!("Failed to parse Claude CLI response: {}", e))?;

    cli_response
        .get("result")
        .and_then(|v| v.as_str())
        .map(String::from)
        .ok_or_else(|| "Claude CLI response missing 'result' field".to_string())
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
}
