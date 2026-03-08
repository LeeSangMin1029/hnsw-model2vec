use crate::VhnswError;

// ---------------------------------------------------------------------------
// VhnswError Display
// ---------------------------------------------------------------------------

#[test]
fn test_error_display_dimension_mismatch() {
    let err = VhnswError::DimensionMismatch {
        expected: 256,
        got: 128,
    };
    assert_eq!(err.to_string(), "dimension mismatch: expected 256, got 128");
}

#[test]
fn test_error_display_point_not_found() {
    let err = VhnswError::PointNotFound(42);
    assert_eq!(err.to_string(), "point not found: 42");
}

#[test]
fn test_error_display_index_full() {
    let err = VhnswError::IndexFull { capacity: 1000 };
    assert_eq!(err.to_string(), "index full: capacity 1000");
}

#[test]
fn test_error_display_tokenizer() {
    let err = VhnswError::Tokenizer("bad token".to_string());
    assert_eq!(err.to_string(), "tokenizer error: bad token");
}

#[test]
fn test_error_display_payload() {
    let err = VhnswError::Payload("decode failed".to_string());
    assert_eq!(err.to_string(), "payload error: decode failed");
}

#[test]
fn test_error_display_wal() {
    let err = VhnswError::Wal("corrupted".to_string());
    assert_eq!(err.to_string(), "wal error: corrupted");
}

#[test]
fn test_error_display_storage() {
    let io_err = std::io::Error::other("disk full");
    let err = VhnswError::Storage(io_err);
    assert_eq!(err.to_string(), "storage error: disk full");
}

// ---------------------------------------------------------------------------
// From<io::Error> conversion
// ---------------------------------------------------------------------------

#[test]
fn test_from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err: VhnswError = io_err.into();
    match &err {
        VhnswError::Storage(e) => {
            assert_eq!(e.kind(), std::io::ErrorKind::NotFound);
            assert_eq!(e.to_string(), "file missing");
        }
        other => panic!("expected Storage variant, got: {other}"),
    }
}

#[test]
fn test_from_io_error_permission_denied() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err: VhnswError = io_err.into();
    match &err {
        VhnswError::Storage(e) => assert_eq!(e.kind(), std::io::ErrorKind::PermissionDenied),
        other => panic!("expected Storage variant, got: {other}"),
    }
}

// ---------------------------------------------------------------------------
// Error source chain (std::error::Error)
// ---------------------------------------------------------------------------

#[test]
fn test_error_source_storage() {
    use std::error::Error;
    let io_err = std::io::Error::other("underlying cause");
    let err = VhnswError::Storage(io_err);
    // Storage variant should have an error source (the io::Error)
    assert!(err.source().is_some());
}

#[test]
fn test_error_source_non_storage() {
    use std::error::Error;
    // Non-Storage variants should have no source
    let err = VhnswError::Tokenizer("oops".into());
    assert!(err.source().is_none());

    let err = VhnswError::Payload("oops".into());
    assert!(err.source().is_none());

    let err = VhnswError::Wal("oops".into());
    assert!(err.source().is_none());

    let err = VhnswError::PointNotFound(1);
    assert!(err.source().is_none());

    let err = VhnswError::IndexFull { capacity: 10 };
    assert!(err.source().is_none());

    let err = VhnswError::DimensionMismatch {
        expected: 1,
        got: 2,
    };
    assert!(err.source().is_none());
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

#[test]
fn test_error_debug() {
    let err = VhnswError::PointNotFound(99);
    let debug = format!("{err:?}");
    assert!(debug.contains("PointNotFound"));
    assert!(debug.contains("99"));
}
