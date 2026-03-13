//! Code intelligence library — structural queries on code-chunked databases.
//!
//! Provides reusable types and algorithms for code navigation:
//! parsing, call graph construction, BFS traversal, and statistics.
//!
//! CLI command handlers live in `v-hnsw-cli`; this crate contains only
//! the pure analysis logic.

pub mod bfs;
pub mod blast;
pub mod clones;
pub mod context;
pub mod context_cmd;
pub mod deps;
pub mod gather;
pub mod graph;
pub mod helpers;
pub mod impact;
pub mod jump;
pub mod loader;
pub mod lsp;
pub mod parse;
pub mod reason;
pub mod stats;
pub mod trace;
