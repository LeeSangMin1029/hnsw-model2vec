//! Tests for the JSONL reader.

use std::io::Write;

use crate::commands::readers::jsonl::JsonlReader;
use crate::commands::readers::VectorReader;

fn write_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
    let mut tmp = tempfile::Builder::new()
        .suffix(".jsonl")
        .tempfile()
        .unwrap();
    for line in lines {
        writeln!(tmp, "{line}").unwrap();
    }
    tmp
}

// ---------------------------------------------------------------------------
// JsonlReader::open + count
// ---------------------------------------------------------------------------

#[test]
fn open_counts_lines() {
    let tmp = write_jsonl(&[
        r#"{"id": 1, "vector": [1.0, 2.0]}"#,
        r#"{"id": 2, "vector": [3.0, 4.0]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    assert_eq!(reader.count().unwrap(), 2);
}

#[test]
fn open_counts_lines_including_blank() {
    let tmp = write_jsonl(&[
        r#"{"id": 1, "vector": [1.0]}"#,
        "",
        r#"{"id": 2, "vector": [2.0]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    // count() counts raw lines (including blank)
    assert_eq!(reader.count().unwrap(), 3);
}

// ---------------------------------------------------------------------------
// JsonlReader::records
// ---------------------------------------------------------------------------

#[test]
fn records_parses_full_fields() {
    let tmp = write_jsonl(&[
        r#"{"id": 42, "vector": [1.0, 2.0, 3.0], "text": "hello", "source": "doc.md", "tags": ["a", "b"]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    let recs: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, 42);
    assert_eq!(recs[0].vector, vec![1.0, 2.0, 3.0]);
    assert_eq!(recs[0].text.as_deref(), Some("hello"));
    assert_eq!(recs[0].source.as_deref(), Some("doc.md"));
    assert_eq!(recs[0].tags.as_deref(), Some(&["a".to_string(), "b".to_string()][..]));
}

#[test]
fn records_optional_fields_default() {
    let tmp = write_jsonl(&[
        r#"{"id": 1}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    let recs: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(recs.len(), 1);
    assert_eq!(recs[0].id, 1);
    assert!(recs[0].vector.is_empty(), "vector defaults to empty");
    assert!(recs[0].text.is_none());
    assert!(recs[0].source.is_none());
    assert!(recs[0].tags.is_none());
}

#[test]
fn records_skips_blank_lines() {
    let tmp = write_jsonl(&[
        r#"{"id": 1, "vector": [1.0]}"#,
        "",
        "   ",
        r#"{"id": 2, "vector": [2.0]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    let recs: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].id, 1);
    assert_eq!(recs[1].id, 2);
}

#[test]
fn records_returns_error_for_malformed_json() {
    let tmp = write_jsonl(&[
        r#"{"id": 1, "vector": [1.0]}"#,
        r#"not valid json"#,
        r#"{"id": 3, "vector": [3.0]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    let results: Vec<_> = reader.records().collect();

    assert_eq!(results.len(), 3);
    assert!(results[0].is_ok());
    assert!(results[1].is_err(), "malformed JSON should produce error");
    let err_msg = format!("{}", results[1].as_ref().err().unwrap());
    assert!(err_msg.contains("parse error"), "error message: {err_msg}");
    assert!(results[2].is_ok());
}

#[test]
fn records_multiple_records() {
    let tmp = write_jsonl(&[
        r#"{"id": 10, "vector": [1.0, 2.0]}"#,
        r#"{"id": 20, "vector": [3.0, 4.0]}"#,
        r#"{"id": 30, "vector": [5.0, 6.0]}"#,
    ]);
    let mut reader = JsonlReader::open(tmp.path()).unwrap();
    let recs: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(recs.len(), 3);
    assert_eq!(recs[0].id, 10);
    assert_eq!(recs[1].id, 20);
    assert_eq!(recs[2].id, 30);
}
