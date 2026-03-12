//! Central error type for the v-hnsw CLI.

use v_hnsw_core::VhnswError;
#[cfg(feature = "doc")]
use v_hnsw_embed::EmbedError;

/// CLI error with categorized variants for structured error handling.
#[derive(thiserror::Error, Debug)]
pub enum CliError {
    /// Database/storage layer error.
    #[error("database: {0}")]
    Database(#[from] VhnswError),

    /// Embedding model error.
    #[cfg(feature = "doc")]
    #[error("embedding: {0}")]
    Embed(#[from] EmbedError),

    /// Daemon communication error.
    #[error("daemon: {0}")]
    Daemon(String),

    /// Invalid user input or arguments.
    #[error("input: {0}")]
    Input(String),

    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Operation was interrupted by Ctrl+C.
    #[error("interrupted")]
    Interrupted,
}

impl From<anyhow::Error> for CliError {
    fn from(err: anyhow::Error) -> Self {
        // Try to downcast to known error types
        if let Some(e) = err.downcast_ref::<VhnswError>() {
            return CliError::Database(match e {
                VhnswError::DimensionMismatch { expected, got } => {
                    VhnswError::DimensionMismatch {
                        expected: *expected,
                        got: *got,
                    }
                }
                VhnswError::PointNotFound(id) => VhnswError::PointNotFound(*id),
                VhnswError::IndexFull { capacity } => VhnswError::IndexFull {
                    capacity: *capacity,
                },
                _ => return CliError::Input(err.to_string()),
            });
        }
        if let Some(e) = err.downcast_ref::<std::io::Error>() {
            return CliError::Io(std::io::Error::new(e.kind(), e.to_string()));
        }
        CliError::Input(err.to_string())
    }
}
