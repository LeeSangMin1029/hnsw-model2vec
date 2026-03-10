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
