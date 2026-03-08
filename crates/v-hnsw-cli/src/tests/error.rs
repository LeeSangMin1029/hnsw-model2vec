//! Tests for error.rs — CliError display and From conversions.

use crate::error::CliError;
use v_hnsw_core::VhnswError;
use v_hnsw_embed::EmbedError;

// ── Display output per variant ──

#[test]
fn display_database_variant() {
    let err = CliError::Database(VhnswError::PointNotFound(42));
    let msg = err.to_string();
    assert!(msg.contains("database:"), "got: {msg}");
    assert!(msg.contains("42"), "got: {msg}");
}

#[test]
fn display_embed_variant() {
    let err = CliError::Embed(EmbedError::ModelInit("oom".into()));
    let msg = err.to_string();
    assert!(msg.contains("embedding:"), "got: {msg}");
    assert!(msg.contains("oom"), "got: {msg}");
}

#[test]
fn display_daemon_variant() {
    let err = CliError::Daemon("connection refused".into());
    assert_eq!(err.to_string(), "daemon: connection refused");
}

#[test]
fn display_input_variant() {
    let err = CliError::Input("bad arg".into());
    assert_eq!(err.to_string(), "input: bad arg");
}

#[test]
fn display_io_variant() {
    let inner = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
    let err = CliError::Io(inner);
    let msg = err.to_string();
    assert!(msg.contains("io:"), "got: {msg}");
    assert!(msg.contains("file missing"), "got: {msg}");
}

#[test]
fn display_interrupted_variant() {
    let err = CliError::Interrupted;
    assert_eq!(err.to_string(), "interrupted");
}

// ── From<VhnswError> ──

#[test]
fn from_vhnsw_dimension_mismatch() {
    let vhnsw_err = VhnswError::DimensionMismatch { expected: 128, got: 64 };
    let cli_err: CliError = vhnsw_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
    assert!(cli_err.to_string().contains("dimension mismatch"));
}

#[test]
fn from_vhnsw_point_not_found() {
    let vhnsw_err = VhnswError::PointNotFound(99);
    let cli_err: CliError = vhnsw_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
    assert!(cli_err.to_string().contains("99"));
}

#[test]
fn from_vhnsw_index_full() {
    let vhnsw_err = VhnswError::IndexFull { capacity: 1000 };
    let cli_err: CliError = vhnsw_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
    assert!(cli_err.to_string().contains("1000"));
}

// ── From<EmbedError> ──

#[test]
fn from_embed_model_init() {
    let embed_err = EmbedError::ModelInit("bad model".into());
    let cli_err: CliError = embed_err.into();
    assert!(matches!(cli_err, CliError::Embed(_)));
    assert!(cli_err.to_string().contains("bad model"));
}

#[test]
fn from_embed_invalid_input() {
    let embed_err = EmbedError::InvalidInput("empty".into());
    let cli_err: CliError = embed_err.into();
    assert!(matches!(cli_err, CliError::Embed(_)));
}

// ── From<std::io::Error> ──

#[test]
fn from_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "no access");
    let cli_err: CliError = io_err.into();
    assert!(matches!(cli_err, CliError::Io(_)));
    assert!(cli_err.to_string().contains("no access"));
}

// ── From<anyhow::Error> ──

#[test]
fn from_anyhow_generic_becomes_input() {
    let anyhow_err = anyhow::anyhow!("something went wrong");
    let cli_err: CliError = anyhow_err.into();
    assert!(matches!(cli_err, CliError::Input(_)));
    assert!(cli_err.to_string().contains("something went wrong"));
}

#[test]
fn from_anyhow_wrapping_io_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing file");
    let anyhow_err: anyhow::Error = io_err.into();
    let cli_err: CliError = anyhow_err.into();
    assert!(matches!(cli_err, CliError::Io(_)));
    assert!(cli_err.to_string().contains("missing file"));
}

#[test]
fn from_anyhow_wrapping_vhnsw_point_not_found() {
    let vhnsw_err = VhnswError::PointNotFound(7);
    let anyhow_err: anyhow::Error = vhnsw_err.into();
    let cli_err: CliError = anyhow_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
}

#[test]
fn from_anyhow_wrapping_vhnsw_dimension_mismatch() {
    let vhnsw_err = VhnswError::DimensionMismatch { expected: 64, got: 32 };
    let anyhow_err: anyhow::Error = vhnsw_err.into();
    let cli_err: CliError = anyhow_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
    assert!(cli_err.to_string().contains("dimension mismatch"));
}

#[test]
fn from_anyhow_wrapping_vhnsw_index_full() {
    let vhnsw_err = VhnswError::IndexFull { capacity: 500 };
    let anyhow_err: anyhow::Error = vhnsw_err.into();
    let cli_err: CliError = anyhow_err.into();
    assert!(matches!(cli_err, CliError::Database(_)));
}
