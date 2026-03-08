//! Tests for insert module pure functions.
//!
//! Modules `standard.rs` and `embed_mode.rs` are integration-only
//! (require StorageEngine / Model); skipped.

use crate::commands::insert::make_payload;

#[test]
fn make_payload_with_all_fields() {
    let payload = make_payload(
        Some("source.md".to_string()),
        Some(vec!["tag1".to_string(), "tag2".to_string()]),
    );
    assert_eq!(payload.source, "source.md");
    assert_eq!(payload.tags, vec!["tag1", "tag2"]);
    assert_eq!(payload.chunk_index, 0);
    assert_eq!(payload.chunk_total, 1);
    assert!(payload.created_at > 0, "created_at should be a valid timestamp");
    assert_eq!(payload.source_modified_at, payload.created_at);
}

#[test]
fn make_payload_with_none_fields() {
    let payload = make_payload(None, None);
    assert!(payload.source.is_empty(), "None source should become empty string");
    assert!(payload.tags.is_empty(), "None tags should become empty vec");
}

#[test]
fn make_payload_source_preserved() {
    let payload = make_payload(Some("path/to/file.rs".to_string()), None);
    assert_eq!(payload.source, "path/to/file.rs");
}
