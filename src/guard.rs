//! Deterministic destructive command detection and file path sandboxing for
//! Claude Code PreToolUse hooks.
//!
//! Reads hook JSON from stdin, evaluates the tool input, and returns structured
//! JSON to block or allow execution. No LLM evaluation — pure pattern matching
//! in Rust.
//!
//! Two modes:
//! - **Bash guard**: Evaluates Bash commands for destructive patterns (always active).
//! - **File path sandbox**: When `AGENT_ALLOWED_PATHS` is set, restricts Edit/Write/Read/
//!   NotebookEdit tools to allowed directory prefixes. Inactive in interactive mode.
//!
//! Configuration is loaded from `.claude/agent-guard.toml` (project-level) or
//! `~/.claude/agent-guard.toml` (user-level), with embedded defaults as fallback.

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::Path;
use std::sync::OnceLock;

// ── Configuration ───────────────────────────────────────

/// Default agent guard configuration embedded in the binary.
const DEFAULT_CONFIG: &str = include_str!("../.claude/agent-guard.toml");

/// Cached compiled patterns loaded once per process.
/// This avoids repeated file I/O, TOML parsing, and regex compilation.
static CACHED_PATTERNS: OnceLock<Vec<CompiledPattern>> = OnceLock::new();

/// Agent guard configuration structure (versioned schema).
/// Fields `schema_version` and `metadata` are deserialized for forward compat
/// and future schema migration; only `patterns` is actively used today.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct GuardConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub metadata: Option<ConfigMetadata>,
    #[serde(default)]
    pub patterns: Vec<PatternDefinition>,
}

/// Metadata about the configuration file (reserved for future tooling).
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ConfigMetadata {
    pub source: String,
    pub version: String,
    pub description: Option<String>,
}

/// Pattern definition from configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct PatternDefinition {
    pub id: String,
    #[serde(default = "default_priority")]
    pub priority: u32,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub matcher: MatcherConfig,
    #[serde(default)]
    pub validator: Option<ValidatorConfig>,
    pub message: String,
}

/// Matcher configuration (currently only regex, extensible for future types).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum MatcherConfig {
    #[serde(rename = "regex")]
    Regex { pattern: String },
}

/// Validator configuration for additional pattern checks.
/// Validators are composable and can express complex logic without hardcoding.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ValidatorConfig {
    /// Reject if segment contains a specific substring
    #[serde(rename = "not_contains")]
    NotContains { value: String },

    /// Ensure all specified flags are present after a command
    #[serde(rename = "flags_present")]
    FlagsPresent { command: String, flags: Vec<String> },

    /// Check if any command arguments match values in a list
    #[serde(rename = "args_match_any")]
    ArgsMatchAny {
        command: String,
        values: Vec<String>,
    },

    /// All sub-validators must pass
    #[serde(rename = "all_of")]
    AllOf { validators: Vec<ValidatorConfig> },

    /// At least one sub-validator must pass
    #[serde(rename = "any_of")]
    AnyOf { validators: Vec<ValidatorConfig> },

    /// Negate the result of a sub-validator
    #[serde(rename = "not")]
    Not { validator: Box<ValidatorConfig> },
}

/// Compiled pattern ready for evaluation.
/// Regex is compiled once during initialization and cached for the process lifetime.
struct CompiledPattern {
    id: String,
    priority: u32,
    regex: Regex,
    message: String,
    validator: Option<ValidatorConfig>,
}

fn default_schema_version() -> String {
    "1.0".to_string()
}

fn default_priority() -> u32 {
    100
}

fn default_enabled() -> bool {
    true
}

// ── Validator Implementation ────────────────────────────

/// Execute a validator configuration against a command segment.
fn execute_validator(segment: &str, validator: &ValidatorConfig) -> bool {
    match validator {
        ValidatorConfig::NotContains { value } => !segment.contains(value.as_str()),

        ValidatorConfig::FlagsPresent { command, flags } => {
            validate_flags_present(segment, command, flags)
        }

        ValidatorConfig::ArgsMatchAny { command, values } => {
            validate_args_match_any(segment, command, values)
        }

        ValidatorConfig::AllOf { validators } => {
            validators.iter().all(|v| execute_validator(segment, v))
        }

        ValidatorConfig::AnyOf { validators } => {
            validators.iter().any(|v| execute_validator(segment, v))
        }

        ValidatorConfig::Not { validator } => !execute_validator(segment, validator),
    }
}

/// Check if all specified flags are present after a command.
fn validate_flags_present(segment: &str, command: &str, required_flags: &[String]) -> bool {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let cmd_pos = match words.iter().position(|w| *w == command) {
        Some(pos) => pos,
        None => return false,
    };

    // Collect all flag characters after the command
    let mut flag_chars = String::new();
    for word in &words[cmd_pos + 1..] {
        if word.starts_with('-') && !word.starts_with("--") {
            flag_chars.push_str(&word[1..]); // Strip leading '-'
        }
    }

    // Check that all required flags are present
    required_flags
        .iter()
        .all(|flag| flag.chars().all(|c| flag_chars.contains(c)))
}

/// Check if any arguments after a command match values in a list.
fn validate_args_match_any(segment: &str, command: &str, values: &[String]) -> bool {
    let words: Vec<&str> = segment.split_whitespace().collect();
    let cmd_pos = match words.iter().position(|w| *w == command) {
        Some(pos) => pos,
        None => return false,
    };

    // Check arguments after the command (skip flags)
    for word in &words[cmd_pos + 1..] {
        if word.starts_with('-') {
            continue; // Skip flags
        }

        // Normalize path for comparison
        let normalized = word.trim_end_matches('/');
        let normalized = if normalized.is_empty() {
            word // Keep original if it becomes empty (like "/")
        } else {
            normalized
        };

        if values.iter().any(|v| v == normalized || v == word) {
            return true;
        }
    }

    false
}

impl GuardConfig {
    /// Load configuration from the hierarchy: project → user → embedded defaults.
    pub fn load() -> Self {
        // Try project-level config first
        if let Some(config) = Self::load_from_project() {
            return config;
        }

        // Try user-level config
        if let Some(config) = Self::load_from_user() {
            return config;
        }

        // Fall back to embedded defaults
        Self::load_from_embedded()
    }

    /// Load config from project-level `.claude/agent-guard.toml`.
    fn load_from_project() -> Option<Self> {
        let path = Path::new(".claude/agent-guard.toml");
        Self::load_from_file(path)
    }

    /// Load config from user-level `~/.claude/agent-guard.toml`.
    fn load_from_user() -> Option<Self> {
        let home = dirs::home_dir()?;
        let path = home.join(".claude/agent-guard.toml");
        Self::load_from_file(&path)
    }

    /// Load config from embedded default string.
    fn load_from_embedded() -> Self {
        toml::from_str(DEFAULT_CONFIG).expect("BUG: embedded default config is invalid TOML")
    }

    /// Load config from a specific file path.
    fn load_from_file(path: &Path) -> Option<Self> {
        let contents = std::fs::read_to_string(path).ok()?;
        toml::from_str(&contents).ok()
    }

    /// Compile patterns from configuration into regex matchers.
    /// Returns compiled patterns sorted by priority (highest first).
    fn compile_patterns(self) -> Vec<CompiledPattern> {
        let mut compiled = Vec::new();

        for pattern_def in self.patterns {
            if !pattern_def.enabled {
                continue; // Skip disabled patterns
            }

            let MatcherConfig::Regex { pattern: regex_str } = &pattern_def.matcher;

            let regex = match Regex::new(regex_str) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!(
                        "[agent-guard] WARNING: Failed to compile regex for pattern '{}': {}",
                        pattern_def.id, e
                    );
                    continue;
                }
            };

            compiled.push(CompiledPattern {
                id: pattern_def.id,
                priority: pattern_def.priority,
                regex,
                message: pattern_def.message,
                validator: pattern_def.validator,
            });
        }

        // Sort by priority (highest first)
        compiled.sort_by(|a, b| b.priority.cmp(&a.priority));

        compiled
    }
}

// ── Public API ──────────────────────────────────────────

/// Entry point for `meta agent guard`.
///
/// Reads PreToolUse hook JSON from stdin, evaluates the tool input,
/// prints denial JSON to stdout if blocked, exits silently if safe.
///
/// For Bash tools: checks command against destructive patterns.
/// For Edit/Write/Read/NotebookEdit: validates file_path against `AGENT_ALLOWED_PATHS`.
pub fn handle_guard() -> Result<()> {
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;

    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(());
    }

    let hook_input: HookInput = match serde_json::from_str(trimmed) {
        Ok(hi) => hi,
        Err(_) => return Ok(()), // Malformed input — allow
    };

    let tool_name = hook_input.tool_name.as_deref().unwrap_or("");

    // File-path tools: validate ALL path fields against allowed directories.
    // Both file_path and notebook_path are checked when present, preventing
    // bypass via a payload that smuggles a second path field.
    if matches!(tool_name, "Edit" | "Write" | "Read" | "NotebookEdit") {
        let paths: Vec<&String> = hook_input
            .tool_input
            .as_ref()
            .map(|ti| {
                let mut v = Vec::new();
                if let Some(fp) = ti.file_path.as_ref() {
                    v.push(fp);
                }
                if let Some(np) = ti.notebook_path.as_ref() {
                    v.push(np);
                }
                v
            })
            .unwrap_or_default();

        if paths.is_empty() {
            // When sandboxing is active, deny file-path tools with no path
            // to prevent bypass via malformed payloads
            if std::env::var_os("AGENT_ALLOWED_PATHS").is_some_and(|v| !v.is_empty()) {
                emit_denial(format!(
                    "{} blocked: no file path provided for sandboxed tool.",
                    tool_name
                ))?;
            }
        } else {
            for fp in &paths {
                if let Some(denial) = evaluate_file_path(tool_name, fp) {
                    emit_denial(denial.reason)?;
                    break;
                }
            }
        }
        return Ok(());
    }

    // Only evaluate Bash tool for destructive command patterns
    if !matches!(tool_name, "" | "Bash") {
        return Ok(());
    }

    let command = match parse_command(&input) {
        Some(cmd) => cmd,
        None => return Ok(()), // No command to evaluate — allow
    };

    if let Some(denial) = evaluate_command(&command) {
        emit_denial(denial.reason)?;
    }

    Ok(())
}

// ── Types ───────────────────────────────────────────────

#[derive(Deserialize)]
struct HookInput {
    tool_name: Option<String>,
    tool_input: Option<ToolInput>,
}

#[derive(Deserialize)]
struct ToolInput {
    command: Option<String>,
    file_path: Option<String>,
    notebook_path: Option<String>,
}

#[derive(Serialize)]
struct HookOutput {
    #[serde(rename = "hookSpecificOutput")]
    hook_specific_output: HookSpecificOutput,
}

#[derive(Serialize)]
struct HookSpecificOutput {
    #[serde(rename = "hookEventName")]
    hook_event_name: String,
    #[serde(rename = "permissionDecision")]
    permission_decision: String,
    #[serde(rename = "permissionDecisionReason")]
    permission_decision_reason: String,
}

/// A denial reason returned when a destructive pattern is detected.
#[derive(Debug, Clone, PartialEq)]
pub struct DenyReason {
    pub reason: String,
}

/// Emit a denial JSON response to stdout.
fn emit_denial(reason: String) -> Result<()> {
    let output = HookOutput {
        hook_specific_output: HookSpecificOutput {
            hook_event_name: "PreToolUse".to_string(),
            permission_decision: "deny".to_string(),
            permission_decision_reason: reason,
        },
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

// ── File Path Sandboxing ────────────────────────────────

/// Validate a file path against `AGENT_ALLOWED_PATHS` (OS-native path list: `:` on Unix, `;` on Windows).
///
/// When the env var is unset or empty, all paths are allowed (interactive mode).
/// When set, the resolved path must start with at least one allowed prefix.
/// Resolves symlinks and `..` components to prevent path traversal escapes.
pub fn evaluate_file_path(tool_name: &str, file_path: &str) -> Option<DenyReason> {
    let allowed = match std::env::var_os("AGENT_ALLOWED_PATHS") {
        Some(v) if !v.is_empty() => v.to_string_lossy().to_string(),
        _ => return None, // No restriction in interactive mode
    };

    evaluate_file_path_with_allowed(tool_name, file_path, &allowed)
}

/// Inner implementation that accepts allowed paths explicitly (testable without env vars).
fn evaluate_file_path_with_allowed(
    tool_name: &str,
    file_path: &str,
    allowed: &str,
) -> Option<DenyReason> {
    // Reject empty and relative paths early — Claude Code should always send absolute paths,
    // so anything else is suspicious. This prevents CWD-dependent canonicalization surprises.
    let trimmed_path = file_path.trim();
    if trimmed_path.is_empty() || !Path::new(trimmed_path).is_absolute() {
        return Some(DenyReason {
            reason: format!(
                "{} blocked: path must be absolute, got '{}'.",
                tool_name, file_path
            ),
        });
    }

    // Resolve the path to catch traversal (../../..) and symlink escapes.
    // For files that don't exist yet (Write creating new file), resolve the parent.
    let resolved = resolve_path(trimmed_path);

    // Debug logging
    if std::env::var("META_DEBUG_GUARD").is_ok() {
        eprintln!(
            "[agent-guard] File path check: tool={}, path={}, resolved={}",
            tool_name, file_path, resolved
        );
    }

    // Use split_paths for OS-native parsing (`:` on Unix, `;` on Windows)
    for prefix in std::env::split_paths(std::ffi::OsStr::new(allowed)) {
        if prefix.as_os_str().is_empty() {
            continue;
        }
        // Resolve the prefix too, so symlinks match (e.g., /tmp -> /private/tmp on macOS)
        let resolved_prefix = resolve_path(&prefix.to_string_lossy());
        // Use Path::starts_with for component-aware comparison, preventing
        // prefix bypass (e.g., /tmp/safe matching /tmp/safevil)
        if Path::new(&resolved).starts_with(Path::new(&resolved_prefix)) {
            return None; // Path is within an allowed prefix
        }
    }

    Some(DenyReason {
        reason: format!(
            "{} blocked: '{}' is outside the allowed workspace. Stay within your worktree.",
            tool_name, file_path
        ),
    })
}

/// Resolve a file path to an absolute, canonical form.
/// Handles symlinks and `..` components. Falls back to the raw path if resolution fails.
///
/// Walks up the path tree to find the deepest existing ancestor, canonicalizes it,
/// then appends the remaining non-existent components. This ensures consistent
/// resolution even when only part of the path exists (e.g., `/tmp` is a symlink
/// to `/private/tmp` on macOS, but `/tmp/worktrees/myworktree` doesn't exist yet).
fn resolve_path(path: &str) -> String {
    let p = Path::new(path);

    // Try full canonicalize first (entire path exists)
    if let Ok(canonical) = p.canonicalize() {
        return canonical.to_string_lossy().to_string();
    }

    // Walk up the path tree to find the deepest existing ancestor,
    // then rebuild with remaining components.
    let mut to_append = Vec::new();
    let mut current = p.to_path_buf();

    while let Some(name) = current.file_name() {
        to_append.push(name.to_os_string());

        let Some(parent) = current.parent() else {
            break;
        };

        if let Ok(canonical) = parent.canonicalize() {
            let mut result = canonical;
            for component in to_append.iter().rev() {
                result = result.join(component);
            }
            // Normalize away any remaining `..` components to prevent traversal
            return normalize_path(&result).to_string_lossy().to_string();
        }
        current = parent.to_path_buf();
    }

    // Last resort: normalize and return (already absolute from Claude Code)
    normalize_path(Path::new(path))
        .to_string_lossy()
        .to_string()
}

/// Normalize a path by collapsing `.` and `..` components logically.
/// Unlike `canonicalize`, this does not require the path to exist.
fn normalize_path(path: &Path) -> std::path::PathBuf {
    use std::path::Component;
    let mut normalized = std::path::PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                normalized.pop();
            }
            Component::CurDir => {}
            _ => {
                normalized.push(component);
            }
        }
    }
    normalized
}

// ── Input Parsing ───────────────────────────────────────

/// Extract the command string from hook JSON input.
/// Returns None if input is empty, malformed, or missing the command field.
fn parse_command(input: &str) -> Option<String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    let hook_input: HookInput = serde_json::from_str(trimmed).ok()?;
    let command = hook_input.tool_input?.command?;
    if command.trim().is_empty() {
        return None;
    }
    Some(command)
}

// ── Command Evaluation ──────────────────────────────────

/// Evaluate a command string for destructive patterns.
/// Returns a DenyReason if the command should be blocked, None if safe.
///
/// Patterns are loaded and compiled once, then cached for the lifetime of the process.
pub fn evaluate_command(command: &str) -> Option<DenyReason> {
    let patterns = CACHED_PATTERNS.get_or_init(|| {
        let config = GuardConfig::load();
        config.compile_patterns()
    });

    for segment in split_compound_command(command) {
        let trimmed = segment.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(denial) = evaluate_segment(trimmed, patterns) {
            return Some(denial);
        }
    }
    None
}

/// Evaluate a single command segment using compiled regex patterns.
fn evaluate_segment(segment: &str, patterns: &[CompiledPattern]) -> Option<DenyReason> {
    for pattern in patterns {
        if pattern.regex.is_match(segment) {
            // Additional validation if required
            if let Some(ref validator) = pattern.validator {
                if !execute_validator(segment, validator) {
                    continue; // Regex matched but validator rejected
                }
            }

            // Debug logging when META_DEBUG_GUARD is set
            if std::env::var("META_DEBUG_GUARD").is_ok() {
                eprintln!(
                    "[agent-guard] Pattern '{}' triggered for: {}",
                    pattern.id, segment
                );
            }

            return Some(DenyReason {
                reason: pattern.message.clone(),
            });
        }
    }
    None
}

/// Split a compound command on `&&`, `||`, `;`, and `|` delimiters.
/// Simple split — does not handle quoting. Sufficient for Claude-generated commands.
/// Returns trimmed segments.
fn split_compound_command(command: &str) -> Vec<&str> {
    let mut segments = Vec::new();
    let mut rest = command;

    loop {
        // Find the earliest delimiter.
        // Order matters: check `||` before `|`, and multi-char before single-char.
        let delimiters: &[&str] = &["||", "&&", ";"];
        let earliest = delimiters
            .iter()
            .filter_map(|d| rest.find(d).map(|pos| (pos, d.len())))
            .min_by_key(|(pos, _)| *pos);

        // Also check for standalone pipe `|` (not part of ||)
        let pipe_pos = find_standalone_pipe(rest);

        // Take whichever delimiter comes first
        let next_delimiter = match (earliest, pipe_pos) {
            (Some((pos1, len1)), Some(pos2)) => {
                if pos2 < pos1 {
                    Some((pos2, 1)) // pipe comes first
                } else {
                    Some((pos1, len1)) // other delimiter comes first
                }
            }
            (Some(delim), None) => Some(delim),
            (None, Some(pos)) => Some((pos, 1)),
            (None, None) => None,
        };

        match next_delimiter {
            Some((pos, len)) => {
                segments.push(rest[..pos].trim());
                rest = &rest[pos + len..];
            }
            None => {
                segments.push(rest.trim());
                break;
            }
        }
    }

    segments
}

/// Find a standalone pipe `|` that is NOT part of `||`.
/// Returns the position of the first such pipe, or None if not found.
fn find_standalone_pipe(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'|' {
            // Check if it's part of ||
            let prev_is_pipe = i > 0 && bytes[i - 1] == b'|';
            let next_is_pipe = i + 1 < bytes.len() && bytes[i + 1] == b'|';
            if !prev_is_pipe && !next_is_pipe {
                return Some(i);
            }
        }
    }
    None
}

// ── Tests ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_command ──────────────────────────────────

    #[test]
    fn parse_command_extracts_command() {
        let input = r#"{"tool_input": {"command": "git status"}}"#;
        assert_eq!(parse_command(input), Some("git status".to_string()));
    }

    #[test]
    fn parse_command_returns_none_for_empty_input() {
        assert_eq!(parse_command(""), None);
        assert_eq!(parse_command("  "), None);
    }

    #[test]
    fn parse_command_returns_none_for_malformed_json() {
        assert_eq!(parse_command("not json"), None);
        assert_eq!(parse_command("{"), None);
    }

    #[test]
    fn parse_command_returns_none_for_missing_fields() {
        assert_eq!(parse_command(r#"{}"#), None);
        assert_eq!(parse_command(r#"{"tool_input": {}}"#), None);
        assert_eq!(parse_command(r#"{"tool_input": {"command": ""}}"#), None);
    }

    // ── split_compound_command ─────────────────────────

    #[test]
    fn split_simple_command() {
        assert_eq!(split_compound_command("git status"), vec!["git status"]);
    }

    #[test]
    fn split_and_chain() {
        assert_eq!(
            split_compound_command("git add . && git commit -m msg"),
            vec!["git add .", "git commit -m msg"]
        );
    }

    #[test]
    fn split_or_chain() {
        assert_eq!(split_compound_command("cmd1 || cmd2"), vec!["cmd1", "cmd2"]);
    }

    #[test]
    fn split_semicolon() {
        assert_eq!(split_compound_command("cmd1; cmd2"), vec!["cmd1", "cmd2"]);
    }

    #[test]
    fn split_mixed_delimiters() {
        assert_eq!(
            split_compound_command("cmd1 && cmd2; cmd3 || cmd4"),
            vec!["cmd1", "cmd2", "cmd3", "cmd4"]
        );
    }

    // ── git push --force ──────────────────────────────

    #[test]
    fn denies_git_push_force() {
        assert!(evaluate_command("git push --force origin main").is_some());
    }

    #[test]
    fn denies_git_push_f() {
        assert!(evaluate_command("git push -f origin main").is_some());
    }

    #[test]
    fn allows_git_push_force_with_lease() {
        assert!(evaluate_command("git push --force-with-lease origin main").is_none());
    }

    #[test]
    fn allows_git_push_force_with_lease_equals() {
        assert!(evaluate_command("git push --force-with-lease=main origin main").is_none());
    }

    #[test]
    fn allows_normal_git_push() {
        assert!(evaluate_command("git push origin main").is_none());
    }

    #[test]
    fn allows_git_push_no_force() {
        assert!(evaluate_command("git push").is_none());
    }

    // ── git reset --hard ──────────────────────────────

    #[test]
    fn denies_git_reset_hard() {
        assert!(evaluate_command("git reset --hard").is_some());
    }

    #[test]
    fn denies_git_reset_hard_with_ref() {
        assert!(evaluate_command("git reset --hard HEAD~3").is_some());
    }

    #[test]
    fn allows_git_reset_soft() {
        assert!(evaluate_command("git reset --soft HEAD~1").is_none());
    }

    #[test]
    fn allows_git_reset_no_flag() {
        assert!(evaluate_command("git reset HEAD file.txt").is_none());
    }

    // ── git clean ─────────────────────────────────────

    #[test]
    fn denies_git_clean_fd() {
        assert!(evaluate_command("git clean -fd").is_some());
    }

    #[test]
    fn denies_git_clean_fdx() {
        assert!(evaluate_command("git clean -fdx").is_some());
    }

    #[test]
    fn denies_git_clean_fxd() {
        assert!(evaluate_command("git clean -fxd").is_some());
    }

    #[test]
    fn denies_git_clean_df() {
        assert!(evaluate_command("git clean -df").is_some());
    }

    #[test]
    fn allows_git_clean_dry_run() {
        assert!(evaluate_command("git clean -nd").is_none());
    }

    #[test]
    fn allows_git_clean_no_force() {
        assert!(evaluate_command("git clean -n").is_none());
    }

    // ── git checkout . ────────────────────────────────

    #[test]
    fn denies_git_checkout_dot() {
        assert!(evaluate_command("git checkout .").is_some());
    }

    #[test]
    fn denies_git_checkout_dashdash_dot() {
        assert!(evaluate_command("git checkout -- .").is_some());
    }

    #[test]
    fn allows_git_checkout_branch() {
        assert!(evaluate_command("git checkout main").is_none());
    }

    #[test]
    fn allows_git_checkout_specific_file() {
        assert!(evaluate_command("git checkout -- src/main.rs").is_none());
    }

    #[test]
    fn allows_git_checkout_b() {
        assert!(evaluate_command("git checkout -b feature/new").is_none());
    }

    // ── rm -rf ────────────────────────────────────────

    #[test]
    fn denies_rm_rf_dot() {
        assert!(evaluate_command("rm -rf .").is_some());
    }

    #[test]
    fn denies_rm_rf_parent() {
        assert!(evaluate_command("rm -rf ..").is_some());
    }

    #[test]
    fn denies_rm_rf_slash() {
        assert!(evaluate_command("rm -rf /").is_some());
    }

    #[test]
    fn denies_rm_rf_meta() {
        assert!(evaluate_command("rm -rf .meta").is_some());
    }

    #[test]
    fn denies_rm_rf_star() {
        assert!(evaluate_command("rm -rf *").is_some());
    }

    #[test]
    fn denies_rm_fr_dot() {
        assert!(evaluate_command("rm -fr .").is_some());
    }

    #[test]
    fn allows_rm_rf_specific_dir() {
        assert!(evaluate_command("rm -rf node_modules").is_none());
    }

    #[test]
    fn allows_rm_rf_specific_path() {
        assert!(evaluate_command("rm -rf target/debug").is_none());
    }

    #[test]
    fn allows_rm_without_rf() {
        assert!(evaluate_command("rm file.txt").is_none());
    }

    // ── Compound commands ─────────────────────────────

    #[test]
    fn denies_destructive_in_compound() {
        assert!(evaluate_command("git add . && git push --force").is_some());
    }

    #[test]
    fn allows_safe_compound() {
        assert!(evaluate_command("git add . && git commit -m msg && git push").is_none());
    }

    #[test]
    fn denies_second_segment_in_semicolon() {
        assert!(evaluate_command("echo hi; git reset --hard").is_some());
    }

    // ── Safe commands ─────────────────────────────────

    #[test]
    fn allows_git_status() {
        assert!(evaluate_command("git status").is_none());
    }

    #[test]
    fn allows_cargo_build() {
        assert!(evaluate_command("cargo build").is_none());
    }

    #[test]
    fn allows_ls() {
        assert!(evaluate_command("ls -la").is_none());
    }

    #[test]
    fn allows_meta_commands() {
        assert!(evaluate_command("meta git status").is_none());
        assert!(evaluate_command("meta exec -- cargo test").is_none());
    }

    // ── Denial reason content ─────────────────────────

    #[test]
    fn force_push_reason_suggests_lease() {
        let denial = evaluate_command("git push --force").unwrap();
        assert!(denial.reason.contains("--force-with-lease"));
    }

    #[test]
    fn reset_hard_reason_suggests_snapshot() {
        let denial = evaluate_command("git reset --hard").unwrap();
        assert!(denial.reason.contains("snapshot"));
    }

    #[test]
    fn clean_reason_suggests_dry_run() {
        let denial = evaluate_command("git clean -fd").unwrap();
        assert!(denial.reason.contains("-nd"));
    }

    // ── JSON output structure ─────────────────────────

    #[test]
    fn hook_output_serializes_correctly() {
        let output = HookOutput {
            hook_specific_output: HookSpecificOutput {
                hook_event_name: "PreToolUse".to_string(),
                permission_decision: "deny".to_string(),
                permission_decision_reason: "test reason".to_string(),
            },
        };
        let json = serde_json::to_string(&output).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "PreToolUse");
        assert_eq!(v["hookSpecificOutput"]["permissionDecision"], "deny");
        assert_eq!(
            v["hookSpecificOutput"]["permissionDecisionReason"],
            "test reason"
        );
    }

    // ── Pipe delimiter ───────────────────────────────

    #[test]
    fn split_pipe_delimiter() {
        assert_eq!(
            split_compound_command("git push --force | tee log.txt"),
            vec!["git push --force", "tee log.txt"]
        );
    }

    #[test]
    fn denies_force_push_piped() {
        assert!(evaluate_command("git push --force origin main | tee output.log").is_some());
    }

    #[test]
    fn denies_reset_hard_piped() {
        assert!(evaluate_command("git reset --hard | cat").is_some());
    }

    #[test]
    fn split_pipe_does_not_confuse_or() {
        // " || " should be matched as OR, not as two pipes
        assert_eq!(split_compound_command("cmd1 || cmd2"), vec!["cmd1", "cmd2"]);
    }

    // ── git clean separate flags ─────────────────────

    #[test]
    fn denies_git_clean_f_d_separate() {
        assert!(evaluate_command("git clean -f -d").is_some());
    }

    #[test]
    fn denies_git_clean_d_f_separate() {
        assert!(evaluate_command("git clean -d -f").is_some());
    }

    #[test]
    fn denies_git_clean_f_d_x_separate() {
        assert!(evaluate_command("git clean -f -d -x").is_some());
    }

    #[test]
    fn allows_git_clean_f_only() {
        // -f alone without -d should be allowed (only removes files, not dirs)
        assert!(evaluate_command("git clean -f").is_none());
    }

    // ── rm -rf edge cases ────────────────────────────

    #[test]
    fn denies_rm_rf_meta_yaml() {
        assert!(evaluate_command("rm -rf .meta.yaml").is_some());
    }

    #[test]
    fn denies_rm_rf_meta_yml() {
        assert!(evaluate_command("rm -rf .meta.yml").is_some());
    }

    #[test]
    fn denies_rm_rf_home_tilde() {
        assert!(evaluate_command("rm -rf ~").is_some());
    }

    #[test]
    fn denies_rm_rf_home_var() {
        assert!(evaluate_command("rm -rf $HOME").is_some());
    }

    #[test]
    fn denies_rm_rf_dot_star() {
        assert!(evaluate_command("rm -rf ./*").is_some());
    }

    #[test]
    fn denies_rm_rf_parent_star() {
        assert!(evaluate_command("rm -rf ../*").is_some());
    }

    #[test]
    fn denies_rm_rf_trailing_slash() {
        assert!(evaluate_command("rm -rf ./").is_some());
    }

    #[test]
    fn denies_rm_rf_multiple_targets_with_dangerous() {
        // Should catch .meta even among safe targets
        assert!(evaluate_command("rm -rf node_modules .meta target").is_some());
    }

    // ── parse_command edge cases ─────────────────────

    #[test]
    fn parse_command_handles_null_tool_input() {
        assert_eq!(parse_command(r#"{"tool_input": null}"#), None);
    }

    #[test]
    fn parse_command_handles_null_command() {
        assert_eq!(parse_command(r#"{"tool_input": {"command": null}}"#), None);
    }

    #[test]
    fn parse_command_ignores_extra_fields() {
        let input = r#"{"hook_event_name":"PreToolUse","tool_name":"Bash","tool_input":{"command":"git status","description":"check status"},"session_id":"abc"}"#;
        assert_eq!(parse_command(input), Some("git status".to_string()));
    }

    // ── git branch -D ────────────────────────────────────

    #[test]
    fn denies_git_branch_force_delete() {
        assert!(evaluate_command("git branch -D feature-branch").is_some());
    }

    #[test]
    fn denies_git_branch_force_delete_multiple() {
        assert!(evaluate_command("git branch -D feat1 feat2").is_some());
    }

    #[test]
    fn allows_git_branch_safe_delete() {
        assert!(evaluate_command("git branch -d feature-branch").is_none());
    }

    #[test]
    fn allows_git_branch_list() {
        assert!(evaluate_command("git branch").is_none());
        assert!(evaluate_command("git branch -v").is_none());
        assert!(evaluate_command("git branch -a").is_none());
    }

    #[test]
    fn allows_git_branch_create() {
        assert!(evaluate_command("git branch new-feature").is_none());
    }

    #[test]
    fn branch_delete_reason_suggests_safe_alternative() {
        let denial = evaluate_command("git branch -D old-branch").unwrap();
        assert!(denial.reason.contains("git branch -d"));
        assert!(denial.reason.contains("safe delete"));
    }

    // ── git stash drop/clear ──────────────────────────

    #[test]
    fn denies_git_stash_drop() {
        assert!(evaluate_command("git stash drop").is_some());
    }

    #[test]
    fn denies_git_stash_drop_with_ref() {
        assert!(evaluate_command("git stash drop stash@{0}").is_some());
    }

    #[test]
    fn denies_git_stash_clear() {
        assert!(evaluate_command("git stash clear").is_some());
    }

    #[test]
    fn allows_git_stash() {
        assert!(evaluate_command("git stash").is_none());
    }

    #[test]
    fn allows_git_stash_push() {
        assert!(evaluate_command("git stash push -m 'WIP'").is_none());
    }

    #[test]
    fn allows_git_stash_list() {
        assert!(evaluate_command("git stash list").is_none());
    }

    #[test]
    fn allows_git_stash_show() {
        assert!(evaluate_command("git stash show").is_none());
        assert!(evaluate_command("git stash show stash@{0}").is_none());
    }

    #[test]
    fn allows_git_stash_apply() {
        assert!(evaluate_command("git stash apply").is_none());
        assert!(evaluate_command("git stash apply stash@{1}").is_none());
    }

    #[test]
    fn allows_git_stash_pop() {
        assert!(evaluate_command("git stash pop").is_none());
    }

    #[test]
    fn stash_drop_reason_suggests_alternatives() {
        let denial = evaluate_command("git stash drop").unwrap();
        assert!(denial.reason.contains("git stash list"));
        assert!(denial.reason.contains("git stash apply"));
    }

    #[test]
    fn stash_clear_reason_suggests_alternatives() {
        let denial = evaluate_command("git stash clear").unwrap();
        assert!(denial.reason.contains("ALL stash entries"));
        assert!(denial.reason.contains("git stash drop"));
    }

    // ── Pipe handling without spaces ──────────────────

    #[test]
    fn split_pipe_no_spaces() {
        assert_eq!(
            split_compound_command("git status|tee log.txt"),
            vec!["git status", "tee log.txt"]
        );
    }

    #[test]
    fn denies_force_push_piped_no_spaces() {
        assert!(evaluate_command("git push --force|tee output.log").is_some());
    }

    #[test]
    fn denies_reset_hard_piped_no_spaces() {
        assert!(evaluate_command("git reset --hard|cat").is_some());
    }

    #[test]
    fn split_pipe_mixed_spacing() {
        assert_eq!(
            split_compound_command("cmd1|cmd2 | cmd3"),
            vec!["cmd1", "cmd2", "cmd3"]
        );
    }

    #[test]
    fn split_does_not_break_or_without_spaces() {
        // "cmd1||cmd2" should still be treated as OR (not two pipes)
        assert_eq!(split_compound_command("cmd1||cmd2"), vec!["cmd1", "cmd2"]);
    }

    #[test]
    fn pipe_in_compound_with_destructive() {
        assert!(evaluate_command("git add .|git commit -m msg && git push --force").is_some());
    }

    // ── Edge cases for new patterns ───────────────────

    #[test]
    fn compound_with_branch_delete() {
        assert!(evaluate_command("git checkout main && git branch -D old-feature").is_some());
    }

    #[test]
    fn compound_with_stash_clear() {
        assert!(evaluate_command("git stash && git stash clear").is_some());
    }

    #[test]
    fn all_new_patterns_in_one_chain() {
        assert!(
            evaluate_command("git branch -D feat1 && git stash drop && git reset --hard").is_some()
        );
    }

    // ── Configuration loading tests ────────────────────

    #[test]
    fn config_loads_embedded_defaults() {
        let config = GuardConfig::load_from_embedded();
        assert_eq!(config.schema_version, "1.0");
        assert!(!config.patterns.is_empty());

        // Verify all expected patterns are present
        let pattern_ids: Vec<&str> = config.patterns.iter().map(|p| p.id.as_str()).collect();
        assert!(pattern_ids.contains(&"meta.git.force_push"));
        assert!(pattern_ids.contains(&"meta.git.reset_hard"));
        assert!(pattern_ids.contains(&"meta.git.branch_force_delete"));
        assert!(pattern_ids.contains(&"meta.git.stash_drop"));
        assert!(pattern_ids.contains(&"meta.git.stash_clear"));
    }

    #[test]
    fn config_default_patterns_are_enabled() {
        let config = GuardConfig::load();

        // All default patterns should be enabled
        for pattern in &config.patterns {
            assert!(
                pattern.enabled,
                "Pattern {} should be enabled by default",
                pattern.id
            );
        }

        // Verify we have the expected number of patterns
        assert!(
            config.patterns.len() >= 8,
            "Should have at least 8 default patterns"
        );
    }

    #[test]
    fn config_can_parse_custom_toml() {
        let toml = r#"
schema_version = "1.0"

[[patterns]]
id = "meta.git.force_push"
enabled = false
matcher = { type = "regex", pattern = 'git\s+push.*--force' }
message = "Force push disabled"

[[patterns]]
id = "meta.git.reset_hard"
enabled = true
matcher = { type = "regex", pattern = 'git\s+reset.*--hard' }
message = "Custom reset message"
"#;
        let config: GuardConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.schema_version, "1.0");
        assert_eq!(config.patterns.len(), 2);

        assert!(!config.patterns[0].enabled);
        assert_eq!(config.patterns[0].id, "meta.git.force_push");

        assert!(config.patterns[1].enabled);
        assert_eq!(config.patterns[1].message, "Custom reset message");
    }

    #[test]
    fn disabled_pattern_is_not_checked() {
        // Create a custom config with git_force_push disabled
        let toml = r#"
schema_version = "1.0"

[[patterns]]
id = "meta.git.force_push"
enabled = false
matcher = { type = "regex", pattern = 'git\s+push.*\s+(--force|-f)\b(?!-with-lease)' }
message = "test"
"#;
        let config: GuardConfig = toml::from_str(toml).unwrap();
        let patterns = config.compile_patterns();

        // This command should normally be denied, but with the pattern disabled it should pass
        let result = evaluate_segment("git push --force origin main", &patterns);
        assert!(result.is_none());
    }

    #[test]
    fn custom_message_overrides_default() {
        let toml = r#"
schema_version = "1.0"

[[patterns]]
id = "meta.git.force_push"
enabled = true
matcher = { type = "regex", pattern = 'git\s+push.*(--force|-f)\b' }
message = "TEAM POLICY: No force push ever!"
"#;
        let config: GuardConfig = toml::from_str(toml).unwrap();
        let patterns = config.compile_patterns();
        let result = evaluate_segment("git push --force", &patterns).unwrap();
        assert_eq!(result.reason, "TEAM POLICY: No force push ever!");
    }

    #[test]
    fn pattern_registry_covers_all_patterns() {
        // Ensure all expected patterns are in the default config
        let config = GuardConfig::load_from_embedded();
        let pattern_ids: Vec<&str> = config.patterns.iter().map(|p| p.id.as_str()).collect();

        assert!(pattern_ids.contains(&"meta.git.force_push"));
        assert!(pattern_ids.contains(&"meta.git.reset_hard"));
        assert!(pattern_ids.contains(&"meta.git.clean_force"));
        assert!(pattern_ids.contains(&"meta.git.checkout_dot"));
        assert!(pattern_ids.contains(&"meta.git.branch_force_delete"));
        assert!(pattern_ids.contains(&"meta.git.stash_drop"));
        assert!(pattern_ids.contains(&"meta.git.stash_clear"));
        assert!(pattern_ids.contains(&"meta.rm.dangerous_paths"));
    }

    #[test]
    fn patterns_are_cached_across_evaluations() {
        // First evaluation loads and compiles patterns
        let result1 = evaluate_command("git status");
        assert!(result1.is_none());

        // Second evaluation should use cached patterns (no additional file I/O or compilation)
        let result2 = evaluate_command("git push --force");
        assert!(result2.is_some());

        // Verify both evaluations worked correctly
        let result3 = evaluate_command("git branch -D test");
        assert!(result3.is_some());
    }

    #[test]
    fn debug_logging_available() {
        // Test that debug env var is checked (doesn't crash)
        std::env::set_var("META_DEBUG_GUARD", "1");
        let result = evaluate_command("git push --force");
        assert!(result.is_some());
        std::env::remove_var("META_DEBUG_GUARD");
    }

    #[test]
    fn patterns_sorted_by_priority() {
        let toml = r#"
schema_version = "1.0"

[[patterns]]
id = "low"
priority = 50
enabled = true
matcher = { type = "regex", pattern = 'test' }
message = "low priority"

[[patterns]]
id = "high"
priority = 200
enabled = true
matcher = { type = "regex", pattern = 'test' }
message = "high priority"

[[patterns]]
id = "medium"
priority = 100
enabled = true
matcher = { type = "regex", pattern = 'test' }
message = "medium priority"
"#;
        let config: GuardConfig = toml::from_str(toml).unwrap();
        let patterns = config.compile_patterns();

        // Should be sorted by priority (highest first)
        assert_eq!(patterns[0].priority, 200);
        assert_eq!(patterns[1].priority, 100);
        assert_eq!(patterns[2].priority, 50);
    }

    // ── File path sandboxing ────────────────────────────
    //
    // Tests use evaluate_file_path_with_allowed() to avoid env var races
    // when tests run in parallel. Tests use Unix-style paths and are
    // skipped on Windows where `/tmp/...` is not considered absolute.

    #[cfg(not(windows))]
    #[test]
    fn file_path_allows_within_prefix() {
        let allowed = std::env::join_paths(["/tmp/worktrees/test", "/tmp"])
            .unwrap()
            .to_string_lossy()
            .to_string();
        let allowed = allowed.as_str();
        assert!(evaluate_file_path_with_allowed(
            "Edit",
            "/tmp/worktrees/test/src/main.rs",
            allowed
        )
        .is_none());
        assert!(evaluate_file_path_with_allowed("Write", "/tmp/somefile.txt", allowed).is_none());
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_outside_prefix() {
        let allowed = "/tmp/worktrees/test";
        let result =
            evaluate_file_path_with_allowed("Edit", "/Users/matt/real-repo/src/main.rs", allowed);
        assert!(result.is_some());
        assert!(result
            .unwrap()
            .reason
            .contains("outside the allowed workspace"));
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_read_outside() {
        let allowed = "/tmp/worktrees/test";
        let result =
            evaluate_file_path_with_allowed("Read", "/Users/matt/real-repo/secrets.env", allowed);
        assert!(result.is_some());
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_multiple_prefixes() {
        let allowed = std::env::join_paths(["/tmp/worktrees/test", "/home/user/.kb", "/tmp"])
            .unwrap()
            .to_string_lossy()
            .to_string();
        let allowed = allowed.as_str();
        assert!(evaluate_file_path_with_allowed(
            "Write",
            "/home/user/.kb/workspace/task.md",
            allowed,
        )
        .is_none());
        assert!(
            evaluate_file_path_with_allowed("Edit", "/tmp/worktrees/test/code.rs", allowed)
                .is_none()
        );
        let result =
            evaluate_file_path_with_allowed("Edit", "/home/user/real-code/main.rs", allowed);
        assert!(result.is_some());
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denial_includes_tool_name() {
        let allowed = "/tmp/allowed";
        let result =
            evaluate_file_path_with_allowed("NotebookEdit", "/forbidden/notebook.ipynb", allowed)
                .unwrap();
        assert!(result.reason.contains("NotebookEdit"));
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_traversal_via_dotdot() {
        let allowed = "/tmp/worktrees/test";
        // Attempt to escape via .. in a non-existent path
        let result = evaluate_file_path_with_allowed(
            "Write",
            "/tmp/worktrees/test/nonexistent/../../etc/passwd",
            allowed,
        );
        assert!(result.is_some(), "path traversal via .. should be denied");
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_prefix_partial_match() {
        let allowed = "/tmp/safe";
        // /tmp/safevil should NOT match /tmp/safe
        let result = evaluate_file_path_with_allowed("Edit", "/tmp/safevil/malicious.sh", allowed);
        assert!(
            result.is_some(),
            "partial directory name match should be denied"
        );
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_empty_path() {
        let allowed = "/tmp/worktrees/test";
        let result = evaluate_file_path_with_allowed("Edit", "", allowed);
        assert!(result.is_some(), "empty path should be denied");
        assert!(result.unwrap().reason.contains("must be absolute"));
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_whitespace_only_path() {
        let allowed = "/tmp/worktrees/test";
        let result = evaluate_file_path_with_allowed("Write", "   ", allowed);
        assert!(result.is_some(), "whitespace-only path should be denied");
    }

    #[cfg(not(windows))]
    #[test]
    fn file_path_denies_relative_path() {
        let allowed = "/tmp/worktrees/test";
        let result = evaluate_file_path_with_allowed("Write", "../../etc/passwd", allowed);
        assert!(result.is_some(), "relative path should be denied");
        assert!(result.unwrap().reason.contains("must be absolute"));
    }

    // ── handle_guard with tool_name ─────────────────────

    #[test]
    fn parse_hook_input_with_tool_name() {
        let input = r#"{"tool_name":"Edit","tool_input":{"file_path":"/tmp/test.rs"}}"#;
        let hi: HookInput = serde_json::from_str(input).unwrap();
        assert_eq!(hi.tool_name.as_deref(), Some("Edit"));
        assert_eq!(
            hi.tool_input.as_ref().unwrap().file_path.as_deref(),
            Some("/tmp/test.rs")
        );
    }

    #[test]
    fn parse_hook_input_with_notebook_path() {
        let input =
            r#"{"tool_name":"NotebookEdit","tool_input":{"notebook_path":"/tmp/nb.ipynb"}}"#;
        let hi: HookInput = serde_json::from_str(input).unwrap();
        assert_eq!(hi.tool_name.as_deref(), Some("NotebookEdit"));
        assert_eq!(
            hi.tool_input.as_ref().unwrap().notebook_path.as_deref(),
            Some("/tmp/nb.ipynb")
        );
    }

    #[test]
    fn parse_hook_input_bash_still_works() {
        let input = r#"{"tool_name":"Bash","tool_input":{"command":"git status"}}"#;
        let hi: HookInput = serde_json::from_str(input).unwrap();
        assert_eq!(hi.tool_name.as_deref(), Some("Bash"));
        assert_eq!(
            hi.tool_input.as_ref().unwrap().command.as_deref(),
            Some("git status")
        );
        assert!(hi.tool_input.as_ref().unwrap().file_path.is_none());
    }

    #[cfg(not(windows))]
    #[test]
    fn resolve_path_handles_existing_paths() {
        let resolved = resolve_path("/tmp");
        assert!(resolved.starts_with('/'));
        // On macOS, /tmp -> /private/tmp
        assert!(resolved == "/tmp" || resolved == "/private/tmp");
    }

    #[cfg(not(windows))]
    #[test]
    fn resolve_path_handles_nonexistent_files() {
        let resolved = resolve_path("/tmp/nonexistent_guard_test_file.rs");
        // Should resolve parent (/tmp or /private/tmp) + filename
        assert!(resolved.ends_with("nonexistent_guard_test_file.rs"));
    }
}
