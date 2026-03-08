use crate::types::{Payload, PayloadValue};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// PayloadValue variants
// ---------------------------------------------------------------------------

#[test]
fn test_payload_value_string() {
    let val = PayloadValue::String("hello".to_string());
    let debug = format!("{val:?}");
    assert!(debug.contains("hello"));
}

#[test]
fn test_payload_value_integer() {
    let val = PayloadValue::Integer(-42);
    let debug = format!("{val:?}");
    assert!(debug.contains("-42"));
}

#[test]
fn test_payload_value_float() {
    let val = PayloadValue::Float(3.14);
    let debug = format!("{val:?}");
    assert!(debug.contains("3.14"));
}

#[test]
fn test_payload_value_bool() {
    let val = PayloadValue::Bool(true);
    let debug = format!("{val:?}");
    assert!(debug.contains("true"));
}

#[test]
fn test_payload_value_string_list() {
    let val = PayloadValue::StringList(vec!["a".into(), "b".into()]);
    let debug = format!("{val:?}");
    assert!(debug.contains("a"));
    assert!(debug.contains("b"));
}

#[test]
fn test_payload_value_clone_debug() {
    let val = PayloadValue::String("hello".to_string());
    let cloned = val.clone();
    let debug = format!("{cloned:?}");
    assert!(debug.contains("hello"));
}

// ---------------------------------------------------------------------------
// PayloadValue bincode round-trip per variant
// ---------------------------------------------------------------------------

#[test]
fn test_payload_value_bincode_integer() {
    let config = bincode::config::standard();
    let val = PayloadValue::Integer(i64::MAX);
    let encoded = bincode::encode_to_vec(&val, config).unwrap();
    let (decoded, _): (PayloadValue, _) = bincode::decode_from_slice(&encoded, config).unwrap();
    match decoded {
        PayloadValue::Integer(n) => assert_eq!(n, i64::MAX),
        other => panic!("expected Integer, got: {other:?}"),
    }
}

#[test]
fn test_payload_value_bincode_float() {
    let config = bincode::config::standard();
    let val = PayloadValue::Float(2.718);
    let encoded = bincode::encode_to_vec(&val, config).unwrap();
    let (decoded, _): (PayloadValue, _) = bincode::decode_from_slice(&encoded, config).unwrap();
    match decoded {
        PayloadValue::Float(f) => assert!((f - 2.718).abs() < 1e-10),
        other => panic!("expected Float, got: {other:?}"),
    }
}

#[test]
fn test_payload_value_bincode_bool() {
    let config = bincode::config::standard();
    let val = PayloadValue::Bool(false);
    let encoded = bincode::encode_to_vec(&val, config).unwrap();
    let (decoded, _): (PayloadValue, _) = bincode::decode_from_slice(&encoded, config).unwrap();
    match decoded {
        PayloadValue::Bool(b) => assert!(!b),
        other => panic!("expected Bool, got: {other:?}"),
    }
}

#[test]
fn test_payload_value_bincode_string_list() {
    let config = bincode::config::standard();
    let val = PayloadValue::StringList(vec!["x".into(), "y".into(), "z".into()]);
    let encoded = bincode::encode_to_vec(&val, config).unwrap();
    let (decoded, _): (PayloadValue, _) = bincode::decode_from_slice(&encoded, config).unwrap();
    match decoded {
        PayloadValue::StringList(v) => assert_eq!(v, vec!["x", "y", "z"]),
        other => panic!("expected StringList, got: {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// Payload bincode round-trip
// ---------------------------------------------------------------------------

#[test]
fn test_payload_bincode_roundtrip() {
    let payload = Payload {
        source: "doc.txt".to_string(),
        tags: vec![],
        created_at: 100,
        source_modified_at: 200,
        chunk_index: 1,
        chunk_total: 2,
        custom: HashMap::new(),
    };

    let config = bincode::config::standard();
    let encoded = bincode::encode_to_vec(&payload, config).unwrap();
    let (decoded, _): (Payload, _) = bincode::decode_from_slice(&encoded, config).unwrap();

    assert_eq!(decoded.source, "doc.txt");
    assert_eq!(decoded.chunk_index, 1);
    assert_eq!(decoded.chunk_total, 2);
}

#[test]
fn test_payload_default_fields() {
    let payload = Payload {
        source: String::new(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 0,
        custom: HashMap::new(),
    };

    assert!(payload.source.is_empty());
    assert!(payload.tags.is_empty());
    assert_eq!(payload.created_at, 0);
    assert_eq!(payload.source_modified_at, 0);
    assert_eq!(payload.chunk_index, 0);
    assert_eq!(payload.chunk_total, 0);
    assert!(payload.custom.is_empty());
}

#[test]
fn test_payload_with_tags_and_custom() {
    let mut custom = HashMap::new();
    custom.insert("lang".into(), PayloadValue::String("ko".into()));
    custom.insert("priority".into(), PayloadValue::Integer(1));

    let payload = Payload {
        source: "readme.md".to_string(),
        tags: vec!["doc".into(), "important".into()],
        created_at: 1_700_000_000,
        source_modified_at: 1_700_000_100,
        chunk_index: 0,
        chunk_total: 1,
        custom,
    };

    let config = bincode::config::standard();
    let encoded = bincode::encode_to_vec(&payload, config).unwrap();
    let (decoded, _): (Payload, _) = bincode::decode_from_slice(&encoded, config).unwrap();

    assert_eq!(decoded.tags, vec!["doc", "important"]);
    assert_eq!(decoded.custom.len(), 2);
    assert!(decoded.custom.contains_key("lang"));
    assert!(decoded.custom.contains_key("priority"));
}

#[test]
fn test_payload_empty_custom_map_roundtrip() {
    let payload = Payload {
        source: "x".into(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 0,
        custom: HashMap::new(),
    };

    let config = bincode::config::standard();
    let encoded = bincode::encode_to_vec(&payload, config).unwrap();
    let (decoded, _): (Payload, _) = bincode::decode_from_slice(&encoded, config).unwrap();
    assert!(decoded.custom.is_empty());
}
