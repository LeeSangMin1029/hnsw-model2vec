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
    let output = if full { output } else { compact_output(output) };
    let json = serde_json::to_string_pretty(&output)?;
    println!("{json}");
    Ok(())
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
