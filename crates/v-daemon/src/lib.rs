//! v-daemon — background daemon for model/index caching + code intelligence.
//!
//! Client utilities (port discovery, RPC, auto-start) in `client` module.
//! Server implementation in `server`, `state`, `handler` modules.

pub mod client;
pub mod code;
pub mod doc;
pub mod handler;
pub mod interrupt;
pub mod server;
pub mod state;
pub mod watcher;

// Re-export client functions for backward compatibility.
pub use client::*;
