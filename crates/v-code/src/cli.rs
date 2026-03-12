//! v-code CLI definition — code intelligence commands only.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::commands::intel::OutputFormat;

/// v-code: Code intelligence CLI.
#[derive(Parser)]
#[command(name = "v-code")]
#[command(author, version, about = "Code intelligence: structural analysis, clone detection, and reasoning")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
#[expect(clippy::large_enum_variant, reason = "clap derive enums are parsed once, size is irrelevant")]
pub enum Commands {
    /// List symbols in the database (functions, structs, enums, impls).
    Symbols {
        /// Path to the database directory.
        db: PathBuf,
        /// Filter by symbol name (substring match).
        #[arg(short, long)]
        name: Option<String>,
        /// Filter by kind (function, struct, enum, impl, trait, etc.).
        #[arg(short, long)]
        kind: Option<String>,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Find symbol definition location.
    #[command(visible_alias = "d")]
    Def {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to find.
        name: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Find all references to a symbol.
    #[command(visible_alias = "r")]
    Refs {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to find references to.
        name: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Show file-level dependency graph from code chunks.
    Deps {
        /// Path to the database directory.
        db: PathBuf,
        /// Show dependencies for a specific file only (suffix match).
        file: Option<String>,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
        /// Transitive dependency depth (default: 1 = direct only).
        #[arg(long, default_value = "1")]
        depth: usize,
    },
    /// Show impact of changing a symbol (reverse BFS: callers).
    #[command(visible_alias = "imp")]
    Impact {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to analyse.
        symbol: String,
        /// Max BFS depth (default: 2).
        #[arg(long, default_value = "2")]
        depth: u32,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
        /// Include test symbols in results (hidden by default).
        #[arg(long)]
        include_tests: bool,
        /// Show reasoning details for each symbol in results.
        #[arg(long)]
        detail: bool,
    },
    /// Find shortest call path between two symbols.
    #[command(visible_alias = "tr")]
    Trace {
        /// Path to the database directory.
        db: PathBuf,
        /// Source symbol name.
        from: String,
        /// Target symbol name.
        to: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Gather context + impact for full symbol understanding.
    #[command(visible_alias = "g")]
    Gather {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to gather around.
        symbol: String,
        /// Max BFS depth (default: 2).
        #[arg(long, default_value = "2")]
        depth: u32,
        /// Max results to show (default: 15).
        #[arg(short, long, default_value = "15")]
        k: usize,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
        /// Include test symbols in results (hidden by default).
        #[arg(long)]
        include_tests: bool,
        /// Show reasoning details for each symbol in results.
        #[arg(long)]
        detail: bool,
    },
    /// View or manage reasoning (design decisions, history) for a symbol.
    #[command(visible_alias = "dt")]
    Detail {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to manage reasoning for.
        symbol: String,
        /// Add a general note.
        #[arg(long)]
        add: Option<String>,
        /// Set or update the design decision.
        #[arg(long)]
        decision: Option<String>,
        /// Set or update the rationale (why).
        #[arg(long)]
        why: Option<String>,
        /// Add a constraint.
        #[arg(long)]
        constraint: Option<String>,
        /// Add a rejected alternative.
        #[arg(long)]
        rejected: Option<String>,
        /// Record a failure.
        #[arg(long)]
        failure: Option<String>,
        /// Record a fix (usually paired with --failure).
        #[arg(long)]
        fix: Option<String>,
        /// Root cause symbol for a failure (used with --failure).
        #[arg(long)]
        root_cause: Option<String>,
        /// Reason for rejecting an alternative (used with --rejected).
        #[arg(long)]
        reject_reason: Option<String>,
        /// Condition under which the rejection applies (used with --rejected).
        #[arg(long)]
        reject_condition: Option<String>,
        /// Mark the last failure as resolved.
        #[arg(long)]
        resolve: bool,
        /// Invalidate the last failure with a reason.
        #[arg(long)]
        invalidate: Option<String>,
        /// Show all history including resolved failures (default: hide resolved).
        #[arg(long)]
        all: bool,
        /// Delete the reasoning entry.
        #[arg(long)]
        delete: bool,
        /// Source file path for location tracking (rename resilience).
        #[arg(long)]
        file_path: Option<String>,
        /// Line range "start:end" for location tracking.
        #[arg(long)]
        line_range: Option<String>,
        /// Related symbol for cross-referencing.
        #[arg(long)]
        relate: Option<String>,
    },
    /// Find duplicate code (token Jaccard default, --ast structural).
    #[command(visible_alias = "dup")]
    Dupes {
        /// Path to the database directory.
        db: PathBuf,
        /// Similarity threshold (Jaccard, 0.0-1.0).
        #[arg(long, default_value = "0.5")]
        threshold: f32,
        /// Exclude test functions from comparison.
        #[arg(long)]
        exclude_tests: bool,
        /// Max number of results to show.
        #[arg(short, long, default_value = "50")]
        k: usize,
        /// Output as JSON.
        #[arg(long)]
        json: bool,
        /// Use AST structural hash (Type-1/2, ignores identifier names).
        #[arg(long)]
        ast: bool,
        /// Unified pipeline: Filter (AST+MinHash) → Verify (all signals).
        #[arg(long)]
        all: bool,
        /// Skip functions shorter than N lines.
        #[arg(long, default_value = "5")]
        min_lines: usize,
    },
    /// Show per-crate code statistics (functions, structs, enums).
    Stats {
        /// Path to the database directory.
        db: PathBuf,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Start persistent daemon server for fast search.
    Serve {
        /// Path to database directory to preload (optional).
        db: Option<PathBuf>,
        /// TCP port to listen on (default: 19530).
        #[arg(long, default_value = "19530")]
        port: u16,
        /// Idle timeout in seconds (0 = never, default: 0).
        #[arg(long, default_value = "0")]
        timeout: u64,
        /// Run in background (daemonize).
        #[arg(long)]
        background: bool,
    },
    /// Add code files to the database (auto-embed with jina-code).
    Add {
        /// Path to the database directory.
        db: PathBuf,
        /// Path to code folder or single file.
        input: PathBuf,
        /// Glob patterns to exclude from scanning.
        #[arg(short, long)]
        exclude: Vec<String>,
    },
    /// Incrementally update code database from source.
    Update {
        /// Path to the database directory.
        db: PathBuf,
        /// Path to source folder.
        input: PathBuf,
        /// Glob patterns to exclude from scanning.
        #[arg(short, long)]
        exclude: Vec<String>,
    },
    /// Search code with hybrid BM25+HNSW and cross-encoder reranking.
    #[command(visible_alias = "f")]
    Find {
        /// Path to the database directory.
        db: PathBuf,
        /// Search query.
        query: String,
        /// Number of results.
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Show full output (scores, model, timing).
        #[arg(long)]
        full: bool,
        /// Minimum score threshold.
        #[arg(long, default_value = "0.0")]
        min_score: f32,
        /// Skip cross-encoder reranking.
        #[arg(long)]
        no_rerank: bool,
    },
}
