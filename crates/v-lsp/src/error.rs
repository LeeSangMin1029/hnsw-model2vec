//! Error types for v-lsp.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum LspError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("protocol error: {0}")]
    Protocol(String),

    #[error("RA process exited unexpectedly")]
    ProcessExited,

    #[error("shared memory error: {0}")]
    Shm(String),

    #[error("timeout waiting for response")]
    Timeout,

    #[error("instance not ready")]
    NotReady,
}

pub type Result<T> = std::result::Result<T, LspError>;
