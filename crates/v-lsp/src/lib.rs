//! v-lsp — Shared rust-analyzer instance with mmap IPC.
//!
//! Architecture:
//! - `lsp`: JSON-RPC 2.0 types + sync transport (ported from lspmux, no tokio)
//! - `instance`: RA process lifecycle + stdin/stdout message routing
//! - `shm`: mmap-based ring buffer for zero-copy IPC between daemon and clients
//! - `client`: Client-side API for sending LSP requests via shared memory

pub mod lsp;
pub mod instance;
pub mod shm;
pub mod chunker;
pub mod client;
pub mod error;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;
