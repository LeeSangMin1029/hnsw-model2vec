//! Hangul Jamo decomposition utilities.
//!
//! Provides functions to decompose Korean syllables into their constituent
//! Jamo (consonants and vowels) for advanced search features like
//! consonant-only search (초성 검색).

/// Initial consonants (초성) in Unicode Jamo order.
const CHOSEONG: [char; 19] = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ',
    'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
];

/// Medial vowels (중성) in Unicode Jamo order.
const JUNGSEONG: [char; 21] = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ',
    'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
];

/// Final consonants (종성) in Unicode Jamo order.
/// Index 0 is empty (no final consonant).
const JONGSEONG: [Option<char>; 28] = [
    None,
    Some('ㄱ'),
    Some('ㄲ'),
    Some('ㄳ'),
    Some('ㄴ'),
    Some('ㄵ'),
    Some('ㄶ'),
    Some('ㄷ'),
    Some('ㄹ'),
    Some('ㄺ'),
    Some('ㄻ'),
    Some('ㄼ'),
    Some('ㄽ'),
    Some('ㄾ'),
    Some('ㄿ'),
    Some('ㅀ'),
    Some('ㅁ'),
    Some('ㅂ'),
    Some('ㅄ'),
    Some('ㅅ'),
    Some('ㅆ'),
    Some('ㅇ'),
    Some('ㅈ'),
    Some('ㅊ'),
    Some('ㅋ'),
    Some('ㅌ'),
    Some('ㅍ'),
    Some('ㅎ'),
];

/// Hangul syllable block start (가).
const HANGUL_SYLLABLE_START: u32 = 0xAC00;
/// Hangul syllable block end (힣).
const HANGUL_SYLLABLE_END: u32 = 0xD7A3;

/// Number of medial vowels.
const JUNGSEONG_COUNT: u32 = 21;
/// Number of final consonants (including none).
const JONGSEONG_COUNT: u32 = 28;

/// Check if a character is a Hangul syllable block (가-힣).
#[inline]
pub fn is_hangul_syllable(c: char) -> bool {
    let code = c as u32;
    (HANGUL_SYLLABLE_START..=HANGUL_SYLLABLE_END).contains(&code)
}

/// Check if a character is a Hangul Jamo (ㄱ-ㅎ, ㅏ-ㅣ).
#[inline]
pub fn is_hangul_jamo(c: char) -> bool {
    let code = c as u32;
    // Hangul Compatibility Jamo: U+3130 - U+318F
    (0x3130..=0x318F).contains(&code)
}

/// Decompose a single Hangul syllable into (초성, 중성, Option<종성>).
///
/// Returns `None` if the character is not a Hangul syllable.
///
/// # Example
/// ```
/// use v_hnsw_search::tokenizer::jamo::decompose_syllable;
///
/// let (cho, jung, jong) = decompose_syllable('한').unwrap();
/// assert_eq!(cho, 'ㅎ');
/// assert_eq!(jung, 'ㅏ');
/// assert_eq!(jong, Some('ㄴ'));
/// ```
pub fn decompose_syllable(c: char) -> Option<(char, char, Option<char>)> {
    if !is_hangul_syllable(c) {
        return None;
    }

    let code = c as u32 - HANGUL_SYLLABLE_START;
    let cho_idx = (code / (JUNGSEONG_COUNT * JONGSEONG_COUNT)) as usize;
    let jung_idx = ((code % (JUNGSEONG_COUNT * JONGSEONG_COUNT)) / JONGSEONG_COUNT) as usize;
    let jong_idx = (code % JONGSEONG_COUNT) as usize;

    Some((CHOSEONG[cho_idx], JUNGSEONG[jung_idx], JONGSEONG[jong_idx]))
}

/// Fully decompose a string into Hangul Jamo.
///
/// Each Hangul syllable is decomposed into its constituent Jamo.
/// Non-Hangul characters are preserved as-is.
///
/// # Example
/// ```
/// use v_hnsw_search::tokenizer::jamo::decompose_hangul;
///
/// assert_eq!(decompose_hangul("한글"), "ㅎㅏㄴㄱㅡㄹ");
/// assert_eq!(decompose_hangul("Hello 세계"), "Hello ㅅㅔㄱㅖ");
/// ```
pub fn decompose_hangul(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 3);

    for c in s.chars() {
        if let Some((cho, jung, jong)) = decompose_syllable(c) {
            result.push(cho);
            result.push(jung);
            if let Some(j) = jong {
                result.push(j);
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Extract only the initial consonants (초성) from a string.
///
/// Useful for consonant search (초성 검색) where users type only
/// the first consonant of each syllable.
///
/// # Example
/// ```
/// use v_hnsw_search::tokenizer::jamo::extract_choseong;
///
/// assert_eq!(extract_choseong("한글"), "ㅎㄱ");
/// assert_eq!(extract_choseong("대한민국"), "ㄷㅎㅁㄱ");
/// assert_eq!(extract_choseong("Hello"), "Hello");
/// ```
pub fn extract_choseong(s: &str) -> String {
    let mut result = String::with_capacity(s.len());

    for c in s.chars() {
        if let Some((cho, _, _)) = decompose_syllable(c) {
            result.push(cho);
        } else if is_hangul_jamo(c) {
            // Already a jamo, keep it
            result.push(c);
        } else {
            result.push(c);
        }
    }

    result
}

/// Check if a query pattern matches text using choseong matching.
///
/// This enables users to search by typing only initial consonants.
///
/// # Example
/// ```
/// use v_hnsw_search::tokenizer::jamo::matches_choseong;
///
/// assert!(matches_choseong("한글 프로그래밍", "ㅎㄱ"));
/// assert!(matches_choseong("대한민국", "ㄷㅎㅁㄱ"));
/// assert!(!matches_choseong("한글", "ㄱㅎ"));
/// ```
pub fn matches_choseong(text: &str, pattern: &str) -> bool {
    let text_choseong = extract_choseong(text);
    text_choseong.contains(pattern)
}