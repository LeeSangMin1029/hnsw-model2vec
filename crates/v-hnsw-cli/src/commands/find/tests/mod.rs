use super::*;

// ── truncate_text ────────────────────────────────────────────────────

#[test]
fn truncate_text_short_string() {
    assert_eq!(truncate_text("hello", 10), "hello");
}

#[test]
fn truncate_text_exact_length() {
    assert_eq!(truncate_text("hello", 5), "hello");
}

#[test]
fn truncate_text_long_string() {
    let result = truncate_text("hello world", 5);
    assert_eq!(result, "hello...");
}

#[test]
fn truncate_text_empty() {
    assert_eq!(truncate_text("", 10), "");
}

#[test]
fn truncate_text_multibyte_boundary() {
    // Korean chars are 3 bytes each in UTF-8
    let korean = "안녕하세요";
    let result = truncate_text(korean, 7);
    // Should truncate at a valid char boundary (6 bytes = 2 chars)
    assert!(result.ends_with("..."));
    assert!(result.starts_with("안녕"));
}

#[test]
fn truncate_text_single_char_limit() {
    let result = truncate_text("abcdef", 1);
    assert_eq!(result, "a...");
}

// ── is_zero ──────────────────────────────────────────────────────────

#[test]
fn is_zero_true() {
    assert!(is_zero(&0));
}

#[test]
fn is_zero_false() {
    assert!(!is_zero(&1));
    assert!(!is_zero(&42));
}

// ── parse_vector ─────────────────────────────────────────────────────

#[test]
fn parse_vector_basic() {
    let result = parse_vector("0.1,0.2,0.3").unwrap();
    assert_eq!(result.len(), 3);
    assert!((result[0] - 0.1).abs() < 1e-6);
    assert!((result[1] - 0.2).abs() < 1e-6);
    assert!((result[2] - 0.3).abs() < 1e-6);
}

#[test]
fn parse_vector_with_spaces() {
    let result = parse_vector(" 1.0 , 2.0 , 3.0 ").unwrap();
    assert_eq!(result.len(), 3);
    assert!((result[0] - 1.0).abs() < 1e-6);
}

#[test]
fn parse_vector_single_value() {
    let result = parse_vector("0.5").unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn parse_vector_negative_values() {
    let result = parse_vector("-1.0,0.0,1.0").unwrap();
    assert!((result[0] - (-1.0)).abs() < 1e-6);
    assert!((result[1]).abs() < 1e-6);
    assert!((result[2] - 1.0).abs() < 1e-6);
}

#[test]
fn parse_vector_invalid_number() {
    let result = parse_vector("0.1,abc,0.3");
    assert!(result.is_err());
}

#[test]
fn parse_vector_empty_segment() {
    let result = parse_vector("0.1,,0.3");
    assert!(result.is_err());
}

// ── looks_like_code_symbol ───────────────────────────────────────────

#[test]
fn code_symbol_rust_path() {
    assert!(looks_like_code_symbol("std::collections::HashMap"));
}

#[test]
fn code_symbol_snake_case() {
    assert!(looks_like_code_symbol("my_function"));
}

#[test]
fn code_symbol_camel_case() {
    assert!(looks_like_code_symbol("MyStruct"));
}

#[test]
fn code_symbol_plain_lowercase() {
    // All alphanumeric + underscore => treated as symbol
    assert!(looks_like_code_symbol("main"));
}

#[test]
fn code_symbol_natural_language() {
    assert!(!looks_like_code_symbol("how to use HashMap"));
}

#[test]
fn code_symbol_empty() {
    // Empty string has no spaces, all chars are alphanumeric/underscore (vacuously true)
    // so it returns true - that's fine, edge case
    assert!(looks_like_code_symbol(""));
}

#[test]
fn code_symbol_with_spaces() {
    assert!(!looks_like_code_symbol("hello world"));
}

// ── compact_output ───────────────────────────────────────────────────

#[test]
fn compact_output_truncates_text() {
    let output = FindOutput {
        results: vec![SearchResultItem {
            id: 1,
            score: 0.95,
            text: Some("a".repeat(200)),
            source: None,
            title: None,
            url: None,
        }],
        query: "test".to_string(),
        model: String::new(),
        total_docs: 100,
        elapsed_ms: 1.5,
    };

    let compacted = compact_output(output);
    let text = compacted.results[0].text.as_ref().unwrap();
    // 150 chars + "..." = 153
    assert!(text.len() <= 153, "text should be truncated to ~150 chars, got {}", text.len());
    assert!(text.ends_with("..."));
}

#[test]
fn compact_output_replaces_newlines() {
    let output = FindOutput {
        results: vec![SearchResultItem {
            id: 1,
            score: 0.5,
            text: Some("line1\nline2\nline3".to_string()),
            source: None,
            title: None,
            url: None,
        }],
        query: String::new(),
        model: String::new(),
        total_docs: 0,
        elapsed_ms: 0.0,
    };

    let compacted = compact_output(output);
    let text = compacted.results[0].text.as_ref().unwrap();
    assert!(!text.contains('\n'));
    assert!(text.contains("line1 line2 line3"));
}

#[test]
fn compact_output_short_text_unchanged() {
    let output = FindOutput {
        results: vec![SearchResultItem {
            id: 1,
            score: 0.5,
            text: Some("short text".to_string()),
            source: None,
            title: None,
            url: None,
        }],
        query: String::new(),
        model: String::new(),
        total_docs: 0,
        elapsed_ms: 0.0,
    };

    let compacted = compact_output(output);
    assert_eq!(compacted.results[0].text.as_deref(), Some("short text"));
}

// ── FindOutput serialization ─────────────────────────────────────────

#[test]
fn find_output_serialization_skips_empty_fields() {
    let output = FindOutput {
        results: vec![],
        query: String::new(),
        model: String::new(),
        total_docs: 0,
        elapsed_ms: 1.0,
    };

    let json = serde_json::to_value(&output).unwrap();
    // query, model should be skipped when empty
    assert!(json.get("query").is_none());
    assert!(json.get("model").is_none());
    // total_docs should be skipped when zero
    assert!(json.get("total_docs").is_none());
    // elapsed_ms should always be present
    assert!(json.get("elapsed_ms").is_some());
}

#[test]
fn find_output_serialization_includes_nonempty() {
    let output = FindOutput {
        results: vec![],
        query: "test query".to_string(),
        model: "my-model".to_string(),
        total_docs: 42,
        elapsed_ms: 2.5,
    };

    let json = serde_json::to_value(&output).unwrap();
    assert_eq!(json["query"], "test query");
    assert_eq!(json["model"], "my-model");
    assert_eq!(json["total_docs"], 42);
}
