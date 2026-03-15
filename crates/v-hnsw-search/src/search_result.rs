//! Search result building and query-language utilities.

use v_hnsw_core::{PayloadStore, PayloadValue};

/// Search result common to both find and serve.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResultItem {
    #[serde(default, skip_serializing_if = "is_id_zero")]
    pub id: u64,
    #[serde(default, skip_serializing_if = "is_score_zero")]
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
}

fn is_id_zero(v: &u64) -> bool {
    *v == 0
}

fn is_score_zero(v: &f32) -> bool {
    *v == 0.0
}

/// Build search result items from raw (id, score) pairs.
pub fn build_results(
    results: &[(u64, f32)],
    payload_store: &dyn PayloadStore,
) -> Vec<SearchResultItem> {
    let max_score = results.first().map(|(_, s)| *s).unwrap_or(1.0);

    results
        .iter()
        .map(|&(id, score)| {
            let text = payload_store.get_text(id).ok().flatten();
            let payload = payload_store.get_payload(id).ok().flatten();
            let source = payload
                .as_ref()
                .map(|p| p.source.clone())
                .filter(|s: &String| !s.is_empty());
            let title = payload
                .as_ref()
                .and_then(|p| p.custom.get("title"))
                .and_then(|v| match v {
                    PayloadValue::String(s) => Some(s.clone()),
                    _ => None,
                });
            let url = payload
                .as_ref()
                .and_then(|p| p.custom.get("url"))
                .and_then(|v| match v {
                    PayloadValue::String(s) => Some(s.clone()),
                    _ => None,
                });

            SearchResultItem {
                id,
                score: if max_score > 0.0 {
                    score / max_score
                } else {
                    0.0
                },
                text,
                source,
                title,
                url,
            }
        })
        .collect()
}

// ── Find output ──────────────────────────────────────────────────────

/// Search output for JSON formatting, shared by v-hnsw find and v-code find.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct FindOutput {
    pub results: Vec<SearchResultItem>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub query: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub model: String,
    #[serde(default, skip_serializing_if = "is_doc_count_zero")]
    pub total_docs: usize,
    #[serde(default, skip_serializing_if = "is_elapsed_zero")]
    pub elapsed_ms: f64,
}

fn is_doc_count_zero(v: &usize) -> bool {
    *v == 0
}

fn is_elapsed_zero(v: &f64) -> bool {
    *v == 0.0
}

/// Truncate text to `max_len` chars, appending "..." if truncated.
pub fn truncate_text(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        return text.to_string();
    }
    let mut end = max_len;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}...", &text[..end])
}

/// Compact the output: truncate text, strip home prefix, zero out metadata.
pub fn compact_output(mut output: FindOutput) -> FindOutput {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default()
        .replace('\\', "/");

    for item in &mut output.results {
        item.id = 0;
        item.score = 0.0;
        item.url = None;
        if let Some(ref text) = item.text {
            let cleaned = text.replace('\n', " ");
            item.text = Some(truncate_text(&cleaned, 150));
        }
        if let Some(ref source) = item.source {
            let short = source
                .strip_prefix(&home)
                .unwrap_or(source)
                .trim_start_matches('/');
            item.source = Some(short.to_string());
        }
    }
    output.model = String::new();
    output.total_docs = 0;
    output.elapsed_ms = 0.0;
    output
}

/// Print `FindOutput` as JSON with optional compaction and score filtering.
pub fn print_find_output(output: FindOutput, full: bool, min_score: f32) -> anyhow::Result<()> {
    let mut output = output;
    if min_score > 0.0 {
        output.results.retain(|item| item.score >= min_score);
    }
    if full {
        let json = serde_json::to_string_pretty(&output)?;
        println!("{json}");
    } else {
        print_compact_grouped(&output);
    }
    Ok(())
}

/// Print results in compact text format, grouped by source file.
fn print_compact_grouped(output: &FindOutput) {
    use std::collections::HashMap;

    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_default()
        .replace('\\', "/");
    let cwd = std::env::current_dir()
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_default();

    // Group results by source file, preserving insertion order
    let mut keys: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<&SearchResultItem>> = HashMap::new();
    for item in &output.results {
        let source = item.source.as_deref().unwrap_or("(unknown)");
        let normalized = source.replace('\\', "/");
        // Try stripping cwd first (more specific), then home
        let short = normalized
            .strip_prefix(&cwd)
            .or_else(|| normalized.strip_prefix(&home))
            .unwrap_or(&normalized)
            .trim_start_matches('/')
            .to_string();
        if !groups.contains_key(&short) {
            keys.push(short.clone());
        }
        groups.entry(short).or_default().push(item);
    }

    for file in &keys {
        let items = &groups[file];
        println!("{file}");
        for item in items {
            let title = item.title.as_deref().unwrap_or("?");
            let desc = item
                .text
                .as_deref()
                .and_then(|t| extract_description(t))
                .unwrap_or_default();
            let lines = item
                .text
                .as_deref()
                .and_then(|t| extract_lines(t))
                .unwrap_or_default();

            if lines.is_empty() {
                print!("  {title}");
            } else {
                print!("  {title} :{lines}");
            }
            if !desc.is_empty() {
                print!(" — {desc}");
            }
            println!();
        }
        println!();
    }
}

/// Extract the description line from chunk text (after File: line).
pub(crate) fn extract_description(text: &str) -> Option<&str> {
    // Text format: "[kind] name\nFile: path:lines\nDescription...\nTypes: ..."
    let lines: Vec<&str> = text.lines().collect();
    // Find first line that's not [kind], File:, Signature:, Types:, Calls:, Called by:
    for line in &lines[1..] {
        let trimmed = line.trim();
        if trimmed.is_empty()
            || trimmed.starts_with("File:")
            || trimmed.starts_with("Signature:")
            || trimmed.starts_with("Types:")
            || trimmed.starts_with("Calls:")
            || trimmed.starts_with("Called by:")
        {
            continue;
        }
        let desc = truncate_text(trimmed, 80);
        return Some(if desc.len() == trimmed.len() {
            trimmed
        } else {
            // Return the original trimmed for lifetime; caller will truncate
            trimmed
        });
    }
    None
}

/// Extract line range from chunk text "File: path:START-END".
pub(crate) fn extract_lines(text: &str) -> Option<&str> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("File:") {
            // Format: "./path:10-20" — extract the line range after last ':'
            let rest = rest.trim();
            if let Some(colon_pos) = rest.rfind(':') {
                let range = &rest[colon_pos + 1..];
                if range.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                    return Some(range);
                }
            }
        }
    }
    None
}

// ── Query language utilities ────────────────────────────────────────

/// Check if query contains Korean (Hangul) characters.
///
/// Returns `true` if any character is in the Hangul Syllables or Jamo range.
pub fn has_korean(text: &str) -> bool {
    text.chars().any(|c| {
        ('\u{AC00}'..='\u{D7AF}').contains(&c)  // Hangul Syllables
            || ('\u{1100}'..='\u{11FF}').contains(&c) // Hangul Jamo
            || ('\u{3130}'..='\u{318F}').contains(&c) // Hangul Compatibility Jamo
    })
}

/// Fusion alpha for convex combination, adjusted by query language.
///
/// Returns alpha in [0, 1]: higher = more weight on dense (vector) search.
pub fn fusion_alpha(query: &str) -> f32 {
    if has_korean(query) {
        0.4 // Korean: BM25 형태소 매칭이 더 중요
    } else {
        0.7 // English: 벡터 유사도 우선
    }
}
