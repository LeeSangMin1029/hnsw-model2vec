//! Tests for get.rs — PointOutput serialization.

use serde_json::Value;

use crate::commands::get::PointOutput;

#[test]
fn point_output_serialize_with_text() {
    let output = PointOutput {
        id: 10,
        text: Some("sample text".into()),
        vector_preview: vec![0.1, 0.2, 0.3, 0.4, 0.5],
        vector_dim: 128,
    };
    let json = serde_json::to_string(&output).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["id"], 10);
    assert_eq!(v["text"], "sample text");
    assert_eq!(v["vector_preview"].as_array().unwrap().len(), 5);
    assert_eq!(v["vector_dim"], 128);
}

#[test]
fn point_output_serialize_without_text() {
    let output = PointOutput {
        id: 20,
        text: None,
        vector_preview: vec![],
        vector_dim: 0,
    };
    let json = serde_json::to_string(&output).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();

    assert_eq!(v["id"], 20);
    assert!(v["text"].is_null());
    assert!(v["vector_preview"].as_array().unwrap().is_empty());
    assert_eq!(v["vector_dim"], 0);
}

#[test]
fn point_output_serialize_pretty() {
    let outputs = vec![
        PointOutput {
            id: 1,
            text: Some("a".into()),
            vector_preview: vec![0.1],
            vector_dim: 64,
        },
        PointOutput {
            id: 2,
            text: None,
            vector_preview: vec![0.2, 0.3],
            vector_dim: 64,
        },
    ];
    let json = serde_json::to_string_pretty(&outputs).unwrap();
    let v: Vec<Value> = serde_json::from_str(&json).unwrap();

    assert_eq!(v.len(), 2);
    assert_eq!(v[0]["id"], 1);
    assert_eq!(v[1]["id"], 2);
}

#[test]
fn point_output_field_names() {
    let output = PointOutput {
        id: 0,
        text: None,
        vector_preview: vec![],
        vector_dim: 0,
    };
    let json = serde_json::to_string(&output).unwrap();

    // Verify exact field names in JSON
    assert!(json.contains("\"id\""));
    assert!(json.contains("\"text\""));
    assert!(json.contains("\"vector_preview\""));
    assert!(json.contains("\"vector_dim\""));
}
