use crate::tokenizer::jamo::*;

#[test]
fn test_is_hangul_syllable() {
    assert!(is_hangul_syllable('가'));
    assert!(is_hangul_syllable('힣'));
    assert!(is_hangul_syllable('한'));
    assert!(!is_hangul_syllable('ㄱ'));
    assert!(!is_hangul_syllable('a'));
    assert!(!is_hangul_syllable('1'));
}

#[test]
fn test_is_hangul_jamo() {
    assert!(is_hangul_jamo('ㄱ'));
    assert!(is_hangul_jamo('ㅎ'));
    assert!(is_hangul_jamo('ㅏ'));
    assert!(is_hangul_jamo('ㅣ'));
    assert!(!is_hangul_jamo('가'));
    assert!(!is_hangul_jamo('a'));
}

#[test]
fn test_decompose_syllable() {
    // 한 = ㅎ + ㅏ + ㄴ
    let (cho, jung, jong) = decompose_syllable('한').expect("valid syllable");
    assert_eq!(cho, 'ㅎ');
    assert_eq!(jung, 'ㅏ');
    assert_eq!(jong, Some('ㄴ'));

    // 가 = ㄱ + ㅏ + (none)
    let (cho, jung, jong) = decompose_syllable('가').expect("valid syllable");
    assert_eq!(cho, 'ㄱ');
    assert_eq!(jung, 'ㅏ');
    assert_eq!(jong, None);

    // 글 = ㄱ + ㅡ + ㄹ
    let (cho, jung, jong) = decompose_syllable('글').expect("valid syllable");
    assert_eq!(cho, 'ㄱ');
    assert_eq!(jung, 'ㅡ');
    assert_eq!(jong, Some('ㄹ'));

    // Non-Hangul
    assert!(decompose_syllable('a').is_none());
    assert!(decompose_syllable('ㄱ').is_none());
}

#[test]
fn test_decompose_hangul() {
    assert_eq!(decompose_hangul("한글"), "ㅎㅏㄴㄱㅡㄹ");
    assert_eq!(decompose_hangul("가나다"), "ㄱㅏㄴㅏㄷㅏ");
    assert_eq!(decompose_hangul(""), "");
    assert_eq!(decompose_hangul("abc"), "abc");
    assert_eq!(decompose_hangul("한a글"), "ㅎㅏㄴaㄱㅡㄹ");
}

#[test]
fn test_extract_choseong() {
    assert_eq!(extract_choseong("한글"), "ㅎㄱ");
    assert_eq!(extract_choseong("대한민국"), "ㄷㅎㅁㄱ");
    assert_eq!(extract_choseong(""), "");
    assert_eq!(extract_choseong("Hello"), "Hello");
    assert_eq!(extract_choseong("ㅎㄱ"), "ㅎㄱ");
    assert_eq!(extract_choseong("한글 세상"), "ㅎㄱ ㅅㅅ");
}

#[test]
fn test_matches_choseong() {
    assert!(matches_choseong("한글", "ㅎㄱ"));
    assert!(matches_choseong("한글 프로그래밍", "ㅎㄱ"));
    assert!(matches_choseong("대한민국", "ㄷㅎㅁㄱ"));
    assert!(!matches_choseong("한글", "ㄱㅎ"));
    assert!(!matches_choseong("영어", "ㅎㄱ"));
}

// ============================================================================
// Additional jamo tests: edge cases, boundary characters
// ============================================================================

#[test]
fn test_is_hangul_syllable_boundaries() {
    // First syllable: 가 (U+AC00)
    assert!(is_hangul_syllable('\u{AC00}'));
    // Last syllable: 힣 (U+D7A3)
    assert!(is_hangul_syllable('\u{D7A3}'));
    // Just before range
    assert!(!is_hangul_syllable('\u{ABFF}'));
    // Just after range
    assert!(!is_hangul_syllable('\u{D7A4}'));
}

#[test]
fn test_is_hangul_jamo_boundaries() {
    // Start: U+3130
    assert!(!is_hangul_jamo('\u{312F}')); // just before
    assert!(is_hangul_jamo('\u{3131}')); // ㄱ
    // End: U+318F
    assert!(is_hangul_jamo('\u{318F}'));
    assert!(!is_hangul_jamo('\u{3190}')); // just after
}

#[test]
fn test_decompose_syllable_first_and_last() {
    // 가 = ㄱ + ㅏ + (none) — first syllable
    let (cho, jung, jong) = decompose_syllable('가').unwrap();
    assert_eq!(cho, 'ㄱ');
    assert_eq!(jung, 'ㅏ');
    assert_eq!(jong, None);

    // 힣 = ㅎ + ㅣ + ㅎ — last syllable
    let (cho, jung, jong) = decompose_syllable('힣').unwrap();
    assert_eq!(cho, 'ㅎ');
    assert_eq!(jung, 'ㅣ');
    assert_eq!(jong, Some('ㅎ'));
}

#[test]
fn test_decompose_syllable_no_jongseong() {
    // 하 = ㅎ + ㅏ + (none)
    let (cho, jung, jong) = decompose_syllable('하').unwrap();
    assert_eq!(cho, 'ㅎ');
    assert_eq!(jung, 'ㅏ');
    assert_eq!(jong, None);
}

#[test]
fn test_decompose_syllable_all_double_consonants() {
    // 까 = ㄲ + ㅏ
    let (cho, _, _) = decompose_syllable('까').unwrap();
    assert_eq!(cho, 'ㄲ');

    // 빠 = ㅃ + ㅏ
    let (cho, _, _) = decompose_syllable('빠').unwrap();
    assert_eq!(cho, 'ㅃ');

    // 싸 = ㅆ + ㅏ
    let (cho, _, _) = decompose_syllable('싸').unwrap();
    assert_eq!(cho, 'ㅆ');
}

#[test]
fn test_decompose_syllable_non_hangul_returns_none() {
    assert!(decompose_syllable('A').is_none());
    assert!(decompose_syllable('1').is_none());
    assert!(decompose_syllable(' ').is_none());
    assert!(decompose_syllable('ㄱ').is_none()); // jamo, not syllable
    assert!(decompose_syllable('ㅏ').is_none()); // vowel jamo
    assert!(decompose_syllable('你').is_none()); // CJK
    assert!(decompose_syllable('😀').is_none()); // emoji
}

#[test]
fn test_decompose_hangul_empty() {
    assert_eq!(decompose_hangul(""), "");
}

#[test]
fn test_decompose_hangul_pure_ascii() {
    assert_eq!(decompose_hangul("Hello World 123!"), "Hello World 123!");
}

#[test]
fn test_decompose_hangul_mixed_scripts() {
    assert_eq!(decompose_hangul("Hello세계World"), "HelloㅅㅔㄱㅖWorld");
}

#[test]
fn test_decompose_hangul_spaces_preserved() {
    assert_eq!(decompose_hangul("한 글"), "ㅎㅏㄴ ㄱㅡㄹ");
}

#[test]
fn test_decompose_hangul_numbers_and_punctuation() {
    assert_eq!(decompose_hangul("가1나2다3"), "ㄱㅏ1ㄴㅏ2ㄷㅏ3");
}

#[test]
fn test_extract_choseong_empty() {
    assert_eq!(extract_choseong(""), "");
}

#[test]
fn test_extract_choseong_spaces_preserved() {
    assert_eq!(extract_choseong("한글 테스트"), "ㅎㄱ ㅌㅅㅌ");
}

#[test]
fn test_extract_choseong_mixed_jamo_syllable() {
    // ㅎ is already a jamo, 한 decomposes to ㅎ
    assert_eq!(extract_choseong("ㅎ한"), "ㅎㅎ");
}

#[test]
fn test_extract_choseong_emoji_passthrough() {
    assert_eq!(extract_choseong("😀"), "😀");
    assert_eq!(extract_choseong("한😀글"), "ㅎ😀ㄱ");
}

#[test]
fn test_matches_choseong_empty_pattern() {
    // Empty pattern should match (contains "" is always true)
    assert!(matches_choseong("한글", ""));
}

#[test]
fn test_matches_choseong_empty_text() {
    assert!(!matches_choseong("", "ㅎ"));
    assert!(matches_choseong("", "")); // "" contains ""
}

#[test]
fn test_matches_choseong_ascii_in_pattern() {
    // Pattern with ASCII should match if text contains that ASCII
    assert!(matches_choseong("Hello한글", "Hello"));
    assert!(matches_choseong("한글Hello", "ㅎㄱHello"));
}

#[test]
fn test_matches_choseong_partial() {
    // Match a substring of the choseong
    assert!(matches_choseong("대한민국만세", "ㅁㄱㅁㅅ"));
    assert!(matches_choseong("대한민국만세", "ㄷㅎ"));
}
