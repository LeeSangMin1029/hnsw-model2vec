//! Tests for export.rs — ExportRecord serialization.

use serde_json::Value;

/// Mirror of the private ExportRecord for testing serialization shape.
#[derive(Debug, serde::Serialize)]
struct ExportRecord {
    id: u64,
    vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

#[test]
fn export_record_serialize_with_text() {
    let record = ExportRecord {
        id: 1,
        vector: vec![0.1, 0.2, 0.3],
        text: Some("hello".into()),
    };
    let json = serde_json::to_string(&record).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["id"], 1);
    assert_eq!(v["vector"], serde_json::json!([0.1, 0.2, 0.3]));
    assert_eq!(v["text"], "hello");
}

#[test]
fn export_record_serialize_without_text() {
    let record = ExportRecord {
        id: 42,
        vector: vec![1.0, 2.0],
        text: None,
    };
    let json = serde_json::to_string(&record).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["id"], 42);
    assert!(v.get("text").is_none(), "text should be skipped when None");
}

#[test]
fn export_record_serialize_empty_vector() {
    let record = ExportRecord {
        id: 0,
        vector: vec![],
        text: None,
    };
    let json = serde_json::to_string(&record).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["vector"], serde_json::json!([]));
}

#[test]
fn export_record_roundtrip() {
    let record = ExportRecord {
        id: 99,
        vector: vec![0.5; 128],
        text: Some("long text".into()),
    };
    let json = serde_json::to_string(&record).unwrap();

    // Deserialize as generic Value and verify fields
    let v: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["id"], 99);
    assert_eq!(v["vector"].as_array().unwrap().len(), 128);
    assert_eq!(v["text"], "long text");
}
