use std::ffi::OsStr;

use crate::commands::common::*;

// ── has_korean ───────────────────────────────────────────────────────

#[test]
fn has_korean_hangul_syllable() {
    assert!(has_korean("안녕하세요"));
}

#[test]
fn has_korean_mixed() {
    assert!(has_korean("hello 세계"));
}

#[test]
fn has_korean_english_only() {
    assert!(!has_korean("hello world"));
}

#[test]
fn has_korean_empty() {
    assert!(!has_korean(""));
}

#[test]
fn has_korean_jamo() {
    // Hangul Jamo range
    assert!(has_korean("\u{1100}\u{1161}"));
}

#[test]
fn has_korean_compatibility_jamo() {
    // Hangul Compatibility Jamo
    assert!(has_korean("\u{3131}"));
}

#[test]
fn has_korean_japanese_not_korean() {
    // Katakana is not Korean
    assert!(!has_korean("カタカナ"));
}

// ── fusion_alpha ─────────────────────────────────────────────────────

#[test]
fn fusion_alpha_korean() {
    let alpha = fusion_alpha("한국어 질문입니다");
    assert!((alpha - 0.4).abs() < 1e-6, "Korean query should get alpha=0.4");
}

#[test]
fn fusion_alpha_english() {
    let alpha = fusion_alpha("english query");
    assert!((alpha - 0.7).abs() < 1e-6, "English query should get alpha=0.7");
}

#[test]
fn fusion_alpha_mixed_has_korean() {
    // Mixed text with Korean characters should be treated as Korean
    let alpha = fusion_alpha("search 검색");
    assert!((alpha - 0.4).abs() < 1e-6);
}

// ── truncate_for_embed ───────────────────────────────────────────────

#[test]
fn truncate_for_embed_short_text() {
    let text = "short text";
    let result = truncate_for_embed(text);
    assert_eq!(result, text);
}

#[test]
fn truncate_for_embed_exact_limit() {
    let text = "a".repeat(8000);
    let result = truncate_for_embed(&text);
    assert_eq!(result.len(), 8000);
}

#[test]
fn truncate_for_embed_over_limit() {
    let text = "a".repeat(9000);
    let result = truncate_for_embed(&text);
    assert_eq!(result.len(), 8000);
}

#[test]
fn truncate_for_embed_multibyte() {
    // Each Korean char is 3 bytes; create a string near the boundary
    let mut text = "a".repeat(7999);
    text.push('\u{AC00}'); // 3 bytes, total = 8002
    let result = truncate_for_embed(&text);
    assert!(result.len() <= 8000);
    // Verify it's a valid string (no panic on boundary)
    assert!(result.is_char_boundary(result.len()));
}

// ── generate_id ──────────────────────────────────────────────────────

#[test]
fn generate_id_deterministic() {
    let id1 = generate_id("test/file.rs", 0);
    let id2 = generate_id("test/file.rs", 0);
    assert_eq!(id1, id2, "same source+index should produce same ID");
}

#[test]
fn generate_id_different_source() {
    let id1 = generate_id("file_a.rs", 0);
    let id2 = generate_id("file_b.rs", 0);
    assert_ne!(id1, id2);
}

#[test]
fn generate_id_different_index() {
    let id1 = generate_id("test.rs", 0);
    let id2 = generate_id("test.rs", 1);
    assert_ne!(id1, id2);
}

// ── content_hash_bytes ───────────────────────────────────────────────

#[test]
fn content_hash_bytes_deterministic() {
    let h1 = content_hash_bytes(b"hello world");
    let h2 = content_hash_bytes(b"hello world");
    assert_eq!(h1, h2);
}

#[test]
fn content_hash_bytes_different_input() {
    let h1 = content_hash_bytes(b"hello");
    let h2 = content_hash_bytes(b"world");
    assert_ne!(h1, h2);
}

#[test]
fn content_hash_bytes_empty() {
    // Should not panic on empty input
    let _h = content_hash_bytes(b"");
}

// ── should_skip_dir ──────────────────────────────────────────────────

#[test]
fn skip_target_dir() {
    assert!(should_skip_dir(OsStr::new("target"), &[]));
}

#[test]
fn skip_node_modules() {
    assert!(should_skip_dir(OsStr::new("node_modules"), &[]));
}

#[test]
fn skip_git_dir() {
    assert!(should_skip_dir(OsStr::new(".git"), &[]));
}

#[test]
fn skip_vhnsw_dirs() {
    assert!(should_skip_dir(OsStr::new(".v-hnsw-code.db"), &[]));
    assert!(should_skip_dir(OsStr::new(".v-hnsw-sessions.db"), &[]));
}

#[test]
fn skip_custom_exclude() {
    assert!(should_skip_dir(OsStr::new("my_custom_dir"), &["my_custom_dir".to_string()]));
}

#[test]
fn dont_skip_src() {
    assert!(!should_skip_dir(OsStr::new("src"), &[]));
}

#[test]
fn dont_skip_regular_dir() {
    assert!(!should_skip_dir(OsStr::new("my_module"), &[]));
}

#[test]
fn skip_all_builtin_dirs() {
    let builtins = [
        "target", "node_modules", ".git", ".swarm", "__pycache__",
        ".venv", "dist", "vendor", ".tox", ".mypy_cache",
        ".pytest_cache", ".claude", "build", "mutants.out",
    ];
    for dir in builtins {
        assert!(should_skip_dir(OsStr::new(dir), &[]), "{dir} should be skipped");
    }
}

// ── make_payload ─────────────────────────────────────────────────────

#[test]
fn make_payload_basic() {
    let payload = make_payload("src/main.rs", Some("main"), &["lang:rust".to_string()], 0, 5, 1000, &Default::default());
    assert_eq!(payload.source, "src/main.rs");
    assert_eq!(payload.tags, vec!["lang:rust"]);
    assert_eq!(payload.chunk_index, 0);
    assert_eq!(payload.chunk_total, 5);
    assert_eq!(payload.source_modified_at, 1000);
    // title should be in custom map
    assert!(payload.custom.contains_key("title"));
}

#[test]
fn make_payload_no_title() {
    let payload = make_payload("test.md", None, &[], 0, 1, 0, &Default::default());
    assert!(!payload.custom.contains_key("title"));
}

#[test]
fn make_payload_multiple_tags() {
    let tags = vec!["kind:function".to_string(), "lang:rust".to_string(), "vis:pub".to_string()];
    let payload = make_payload("lib.rs", None, &tags, 2, 10, 500, &Default::default());
    assert_eq!(payload.tags.len(), 3);
    assert_eq!(payload.chunk_index, 2);
    assert_eq!(payload.chunk_total, 10);
}

// ── SearchResultItem serialization ───────────────────────────────────

#[test]
fn search_result_item_skips_none_fields() {
    let item = SearchResultItem {
        id: 42,
        score: 0.95,
        text: None,
        source: None,
        title: None,
        url: None,
    };
    let json = serde_json::to_value(&item).unwrap();
    assert!(json.get("text").is_none());
    assert!(json.get("source").is_none());
    assert!(json.get("title").is_none());
    assert!(json.get("url").is_none());
    assert_eq!(json["id"], 42);
}

#[test]
fn search_result_item_includes_present_fields() {
    let item = SearchResultItem {
        id: 1,
        score: 0.8,
        text: Some("hello".to_string()),
        source: Some("file.md".to_string()),
        title: Some("Title".to_string()),
        url: Some("https://example.com".to_string()),
    };
    let json = serde_json::to_value(&item).unwrap();
    assert_eq!(json["text"], "hello");
    assert_eq!(json["source"], "file.md");
    assert_eq!(json["title"], "Title");
    assert_eq!(json["url"], "https://example.com");
}
