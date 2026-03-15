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
        /// Include test symbols in results (excluded by default).
        #[arg(long)]
        include_tests: bool,
        /// Max number of symbols to show.
        #[arg(short, long)]
        limit: Option<usize>,
        /// Compact output: name and location only (no signatures).
        #[arg(long)]
        compact: bool,
    },
    /// Unified context: definition + callers + callees + types + tests.
    #[command(visible_alias = "ctx")]
    Context {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to look up.
        symbol: String,
        /// Max BFS depth (default: 1).
        #[arg(long, default_value = "1")]
        depth: u32,
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
    /// Show blast radius of changing a symbol (transitive callers + summary).
    #[command(visible_alias = "bl")]
    Blast {
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
    /// Show execution flow tree (DFS callee traversal).
    #[command(visible_alias = "j")]
    Jump {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to trace execution flow for.
        symbol: String,
        /// Max DFS depth (default: 2).
        #[arg(long, default_value = "2")]
        depth: u32,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
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
        /// Minimum sub-block size (lines) for intra-function clone detection.
        #[arg(long, default_value = "5")]
        min_sub_lines: usize,
        /// Analyze duplicate pairs: callee/caller match, blast radius, merge safety.
        #[arg(long)]
        analyze: bool,
    },
    /// Show per-crate code statistics (functions, structs, enums).
    Stats {
        /// Path to the database directory.
        db: PathBuf,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Search string literal arguments across all chunks.
    #[command(visible_alias = "str")]
    Strings {
        /// Path to the database directory.
        db: PathBuf,
        /// String value to search (substring match, case-insensitive).
        query: String,
        /// Filter by callee function name (substring match).
        #[arg(long)]
        callee: Option<String>,
    },
    /// Trace interprocedural string flow through wrapper functions.
    #[command(visible_alias = "fl")]
    Flow {
        /// Path to the database directory.
        db: PathBuf,
        /// String value to trace (substring match, case-insensitive).
        query: String,
        /// Max depth to follow parameter flows (default: 3).
        #[arg(long, default_value = "3")]
        depth: u32,
    },
    /// Add/update code files in the database (auto-incremental).
    Add {
        /// Path to the database directory.
        db: PathBuf,
        /// Path to code folder or single file.
        input: PathBuf,
        /// Glob patterns to exclude from scanning.
        #[arg(short, long)]
        exclude: Vec<String>,
    },
}
