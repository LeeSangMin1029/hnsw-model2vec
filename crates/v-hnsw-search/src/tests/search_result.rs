use crate::search_result::{truncate_text, has_korean, fusion_alpha, extract_description, extract_lines};
use crate::split_identifier;

// ============================================================================
// truncate_text tests
// ============================================================================

#[test]
fn truncate_text_short_string_unchanged() {
    let result = truncate_text("hello", 10);
    assert_eq!(result, "hello");
}

#[test]
fn truncate_text_exact_length_unchanged() {
    let result = truncate_text("hello", 5);
    assert_eq!(result, "hello");
}

#[test]
fn truncate_text_long_string_truncated() {
    let result = truncate_text("hello world", 5);
    assert_eq!(result, "hello...");
}

#[test]
fn truncate_text_empty_string() {
    let result = truncate_text("", 10);
    assert_eq!(result, "");
}

#[test]
fn truncate_text_multibyte_boundary() {
    // "café" — 'é' is 2 bytes in UTF-8; truncating at byte 4 would split it
    let result = truncate_text("café latte", 4);
    // Should back up to a valid char boundary (before 'é' or after it)
    assert!(result.ends_with("..."));
    // The truncated part must be valid UTF-8
    assert!(result.is_char_boundary(result.len()));
}

#[test]
fn truncate_text_cjk_characters() {
    // Each CJK char is 3 bytes in UTF-8
    let text = "你好世界测试";
    let result = truncate_text(text, 6);
    // 6 bytes = 2 CJK chars
    assert!(result.ends_with("..."));
    assert_eq!(&result[..result.len() - 3], "你好");
}

#[test]
fn truncate_text_max_len_one() {
    let result = truncate_text("ab", 1);
    assert_eq!(result, "a...");
}

// ============================================================================
// has_korean tests
// ============================================================================

#[test]
fn has_korean_with_hangul_syllables() {
    assert!(has_korean("안녕하세요"));
}

#[test]
fn has_korean_mixed_with_english() {
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
fn has_korean_japanese_not_korean() {
    // Katakana and Hiragana are not Korean
    assert!(!has_korean("こんにちは"));
}

#[test]
fn has_korean_jamo() {
    // Hangul Jamo range (U+1100-U+11FF)
    assert!(has_korean("\u{1100}"));
}

#[test]
fn has_korean_compatibility_jamo() {
    // Hangul Compatibility Jamo (U+3130-U+318F)
    assert!(has_korean("\u{3131}")); // ㄱ
}

// ============================================================================
// fusion_alpha tests
// ============================================================================

#[test]
fn fusion_alpha_english_query() {
    let alpha = fusion_alpha("hello world");
    assert!((alpha - 0.7).abs() < f32::EPSILON);
}

#[test]
fn fusion_alpha_korean_query() {
    let alpha = fusion_alpha("안녕하세요");
    assert!((alpha - 0.4).abs() < f32::EPSILON);
}

#[test]
fn fusion_alpha_mixed_favors_korean() {
    // If any Korean character is present, use Korean alpha
    let alpha = fusion_alpha("search 검색어");
    assert!((alpha - 0.4).abs() < f32::EPSILON);
}

#[test]
fn fusion_alpha_empty_query() {
    let alpha = fusion_alpha("");
    assert!((alpha - 0.7).abs() < f32::EPSILON);
}

// ============================================================================
// extract_description tests
// ============================================================================

#[test]
fn extract_description_standard_format() {
    let text = "[function] my_func\nFile: src/lib.rs:10-20\nThis function does something useful.";
    let desc = extract_description(text);
    assert_eq!(desc, Some("This function does something useful."));
}

#[test]
fn extract_description_skips_metadata_lines() {
    let text = "[function] my_func\nFile: src/lib.rs:10-20\nSignature: fn my_func()\nTypes: u32\nActual description here";
    let desc = extract_description(text);
    assert_eq!(desc, Some("Actual description here"));
}

#[test]
fn extract_description_no_description() {
    let text = "[function] my_func\nFile: src/lib.rs:10-20\nSignature: fn my_func()";
    let desc = extract_description(text);
    assert_eq!(desc, None);
}

#[test]
fn extract_description_skips_calls_and_called_by() {
    let text = "[function] my_func\nFile: src/lib.rs:10-20\nCalls: foo, bar\nCalled by: baz\nThe real description";
    let desc = extract_description(text);
    assert_eq!(desc, Some("The real description"));
}

// ============================================================================
// extract_lines tests
// ============================================================================

#[test]
fn extract_lines_standard_format() {
    let text = "[function] my_func\nFile: ./src/lib.rs:10-20\nSome description";
    let lines = extract_lines(text);
    assert_eq!(lines, Some("10-20"));
}

#[test]
fn extract_lines_single_line() {
    let text = "[function] my_func\nFile: ./src/lib.rs:42\nSome description";
    let lines = extract_lines(text);
    assert_eq!(lines, Some("42"));
}

#[test]
fn extract_lines_no_file_line() {
    let text = "[function] my_func\nSome description";
    let lines = extract_lines(text);
    assert_eq!(lines, None);
}

#[test]
fn extract_lines_file_no_line_range() {
    let text = "[function] my_func\nFile: ./src/lib.rs\nSome description";
    let lines = extract_lines(text);
    // No colon after path with digits → should return None
    // Actually the path has colons from drive letters on Windows, but the format
    // uses rfind(':') so it finds the last colon. The range after it is "lib.rs"
    // which doesn't start with a digit, so None.
    assert_eq!(lines, None);
}

// ============================================================================
// split_identifier tests
// ============================================================================

#[test]
fn split_identifier_camel_case() {
    let parts = split_identifier("camelCase");
    assert_eq!(parts, vec!["camel", "Case"]);
}

#[test]
fn split_identifier_snake_case() {
    let parts = split_identifier("snake_case_name");
    assert_eq!(parts, vec!["snake", "case", "name"]);
}

#[test]
fn split_identifier_screaming_snake() {
    let parts = split_identifier("MAX_BUFFER_SIZE");
    assert_eq!(parts, vec!["MAX", "BUFFER", "SIZE"]);
}

#[test]
fn split_identifier_acronym_then_word() {
    // HTMLParser → HTML + Parser
    let parts = split_identifier("HTMLParser");
    assert_eq!(parts, vec!["HTML", "Parser"]);
}

#[test]
fn split_identifier_digit_boundaries() {
    // vec3d → vec + 3 + d
    let parts = split_identifier("vec3d");
    assert_eq!(parts, vec!["vec", "3", "d"]);
}

#[test]
fn split_identifier_single_word() {
    let parts = split_identifier("hello");
    assert_eq!(parts, vec!["hello"]);
}

#[test]
fn split_identifier_empty_string() {
    let parts = split_identifier("");
    assert!(parts.is_empty());
}

#[test]
fn split_identifier_leading_underscores() {
    let parts = split_identifier("__private");
    assert_eq!(parts, vec!["private"]);
}

#[test]
fn split_identifier_mixed_camel_and_snake() {
    let parts = split_identifier("myFunc_name");
    assert_eq!(parts, vec!["my", "Func", "name"]);
}

#[test]
fn split_identifier_all_uppercase() {
    let parts = split_identifier("ALLCAPS");
    assert_eq!(parts, vec!["ALLCAPS"]);
}

#[test]
fn split_identifier_number_prefix() {
    let parts = split_identifier("3dVector");
    assert_eq!(parts, vec!["3", "d", "Vector"]);
}
