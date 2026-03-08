use std::path::PathBuf;

use crate::korean_tokenizer::*;
use crate::Tokenizer;

fn test_dict_path() -> PathBuf {
    if let Ok(path) = std::env::var("LINDERA_KO_DIC_PATH") {
        return PathBuf::from(path);
    }
    v_hnsw_core::ko_dic_dir()
}

fn ensure_init() {
    let _ = init_korean_tokenizer(&test_dict_path());
}

#[test]
fn test_korean_tokenizer_basic() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();
    let tokens = tokenizer.tokenize("안녕하세요 세계");
    assert!(!tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_morphological() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();
    let tokens = tokenizer.tokenize("한국은 아름다운 나라입니다");
    assert!(!tokens.contains(&"은".to_string()));
}

#[test]
fn test_korean_tokenizer_english() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();
    let tokens = tokenizer.tokenize("hello world test");
    assert!(!tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_mixed() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();
    let tokens = tokenizer.tokenize("벡터 데이터베이스 vector database");
    assert!(tokens.len() >= 2);
}

#[test]
fn test_korean_tokenizer_empty() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();
    let tokens = tokenizer.tokenize("");
    assert!(tokens.is_empty());
}

#[test]
fn test_korean_tokenizer_clone() {
    ensure_init();
    let t1 = KoreanBm25Tokenizer::new();
    let t2 = t1.clone();
    let tokens1 = t1.tokenize("테스트");
    let tokens2 = t2.tokenize("테스트");
    assert_eq!(tokens1, tokens2);
}

#[test]
fn test_korean_tokenizer_serialization() {
    ensure_init();
    let tokenizer = KoreanBm25Tokenizer::new();

    // serde roundtrip
    let json = serde_json::to_string(&tokenizer).expect("serialize");
    let _: KoreanBm25Tokenizer = serde_json::from_str(&json).expect("deserialize");

    // bincode roundtrip
    let bytes =
        bincode::encode_to_vec(&tokenizer, bincode::config::standard()).expect("encode");
    let (decoded, _): (KoreanBm25Tokenizer, usize) =
        bincode::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");

    // Verify it still works after deserialization
    let tokens = decoded.tokenize("테스트 문장");
    assert!(!tokens.is_empty());
}
