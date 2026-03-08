//! Tests for the readers module dispatch (open_reader).

use std::io::Write;

use crate::commands::readers::{open_reader, ReaderConfig};

#[test]
fn open_reader_jsonl_extension() {
    let mut tmp = tempfile::Builder::new()
        .suffix(".jsonl")
        .tempfile()
        .unwrap();
    writeln!(tmp, r#"{{"id": 1, "vector": [1.0]}}"#).unwrap();

    let cfg = ReaderConfig::with_vector("vector");
    let mut reader = open_reader(tmp.path(), &cfg).unwrap();
    assert_eq!(reader.count().unwrap(), 1);
}

#[test]
fn open_reader_ndjson_extension() {
    let mut tmp = tempfile::Builder::new()
        .suffix(".ndjson")
        .tempfile()
        .unwrap();
    writeln!(tmp, r#"{{"id": 1, "vector": [1.0]}}"#).unwrap();

    let cfg = ReaderConfig::with_vector("vector");
    let mut reader = open_reader(tmp.path(), &cfg).unwrap();
    assert_eq!(reader.count().unwrap(), 1);
}

#[test]
fn open_reader_fvecs_extension() {
    let mut data = Vec::new();
    // dim=2, two f32 values
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let cfg = ReaderConfig::with_vector("vector");
    let mut reader = open_reader(tmp.path(), &cfg).unwrap();
    assert_eq!(reader.count().unwrap(), 1);
}

#[test]
fn open_reader_fvecs_rejected_in_embed_mode() {
    let mut data = Vec::new();
    data.extend_from_slice(&2u32.to_le_bytes());
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let cfg = ReaderConfig { vector_column: None }; // embed mode
    let result = open_reader(tmp.path(), &cfg);
    assert!(result.is_err());
    let msg = format!("{}", result.err().unwrap());
    assert!(msg.contains("embed"), "error should mention embed mode: {msg}");
}

#[test]
fn open_reader_unsupported_extension() {
    let tmp = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .unwrap();
    let cfg = ReaderConfig::with_vector("vector");
    let result = open_reader(tmp.path(), &cfg);
    assert!(result.is_err());
    let msg = format!("{}", result.err().unwrap());
    assert!(msg.contains("csv"), "error should mention csv: {msg}");
}
