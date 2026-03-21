use crate::lsp::jsonrpc::{Message, RequestId};
use serde_json::{from_value, json, to_value};

fn roundtrip(input: serde_json::Value) {
    let deserialized = from_value::<Message>(input.clone()).expect("deserialize");
    let serialized = to_value(deserialized).expect("serialize");
    assert_eq!(input, serialized);
}

#[test]
fn request_positional_params() {
    roundtrip(json!({
        "jsonrpc": "2.0",
        "method": "subtract",
        "params": [42, 23],
        "id": 1,
    }));
}

#[test]
fn request_named_params() {
    roundtrip(json!({
        "jsonrpc": "2.0",
        "method": "subtract",
        "params": { "minuend": 42, "subtrahend": 23 },
        "id": 3,
    }));
}

#[test]
fn response_success() {
    roundtrip(json!({
        "jsonrpc": "2.0",
        "result": 19,
        "id": 1,
    }));
}

#[test]
fn notification() {
    roundtrip(json!({
        "jsonrpc": "2.0",
        "method": "update",
        "params": [1, 2, 3, 4, 5],
    }));
}

#[test]
fn error_response() {
    roundtrip(json!({
        "jsonrpc": "2.0",
        "error": {
            "code": -32601,
            "message": "Method not found",
        },
        "id": "1",
    }));
}

#[test]
fn request_id_display() {
    assert_eq!(RequestId::Number(42).to_string(), "42");
    assert_eq!(RequestId::String("abc".into()).to_string(), "abc");
}
