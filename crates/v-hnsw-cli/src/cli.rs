//! CLI argument definitions using clap derive.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

/// v-hnsw: Local vector database CLI.
#[derive(Parser)]
#[command(name = "v-hnsw")]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Create a new database.
    Create {
        /// Path to the database directory.
        path: PathBuf,
        /// Vector dimension (required).
        #[arg(short = 'd', long)]
        dim: usize,
        /// Distance metric.
        #[arg(long, default_value = "cosine")]
        metric: MetricType,
        /// Max neighbors per layer (HNSW M parameter).
        #[arg(short = 'm', long = "neighbors", default_value = "16")]
        m: usize,
        /// Construction beam width (ef_construction).
        #[arg(short = 'e', long, default_value = "200")]
        ef: usize,
        /// Enable Korean tokenizer for BM25.
        #[arg(long)]
        korean: bool,
    },
    /// Show database information.
    Info {
        /// Path to the database directory.
        path: PathBuf,
    },
    /// Insert vectors from a file (JSONL, fvecs, bvecs).
    Insert {
        /// Path to the database directory.
        path: PathBuf,
        /// Input file (.jsonl, .ndjson, .fvecs, .bvecs).
        #[arg(short, long)]
        input: PathBuf,
        /// Vector column name for input files.
        #[arg(long, default_value = "vector")]
        vector_column: String,
        /// Auto-embed text using model2vec (skips vector column requirement).
        #[arg(long)]
        embed: bool,
        /// Text column name to embed (used with --embed).
        #[arg(long, default_value = "text")]
        text_column: String,
        /// Model2Vec model ID from HuggingFace (used with --embed).
        #[arg(long, default_value = "minishlab/potion-multilingual-128M")]
        model: String,
        /// Batch size for embedding (used with --embed).
        #[arg(long, default_value = "1024")]
        batch_size: usize,
    },
    /// Delete a point by ID.
    Delete {
        /// Path to the database directory.
        path: PathBuf,
        /// Point ID to delete.
        #[arg(long)]
        id: u64,
    },
    /// Run a benchmark.
    Bench {
        /// Path to the database directory.
        path: PathBuf,
        /// Number of random queries.
        #[arg(short, long, default_value = "100")]
        queries: usize,
        /// Number of results per query.
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Export database to JSONL file.
    Export {
        /// Path to the database directory.
        path: PathBuf,
        /// Output JSONL file.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Manage collections.
    Collection {
        /// Path to the database root directory.
        path: PathBuf,
        #[command(subcommand)]
        action: crate::commands::collection::CollectionAction,
    },
    /// Rebuild HNSW and BM25 indexes from existing data.
    BuildIndex {
        /// Path to the database directory.
        path: PathBuf,
    },
    /// Get point details by ID.
    Get {
        /// Path to the database directory.
        path: PathBuf,
        /// Point IDs to retrieve.
        #[arg(required = true)]
        ids: Vec<u64>,
    },
    /// Add data to database (auto-detect input type, auto-embed, auto-create DB).
    Add {
        /// Path to the database directory.
        db: PathBuf,
        /// Input file or folder (folder with .md files, .jsonl, or .parquet).
        input: PathBuf,
        /// Directories to exclude from scanning (can be specified multiple times).
        #[arg(long)]
        exclude: Vec<String>,
    },
    /// Incrementally update database with changed files.
    Update {
        /// Path to the database directory.
        db: PathBuf,
        /// Input folder to scan for changes.
        input: PathBuf,
        /// Directories to exclude from scanning (can be specified multiple times).
        #[arg(long)]
        exclude: Vec<String>,
    },
    /// Search database (hybrid HNSW+BM25 by default, supports raw vector mode).
    Find {
        /// Path to the database directory.
        db: PathBuf,
        /// Search query text (required unless --vector is provided).
        query: Option<String>,
        /// Number of results to return.
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Filter by tags (can be specified multiple times, AND logic).
        #[arg(long)]
        tag: Vec<String>,
        /// Show full text (default: truncated to 150 chars).
        #[arg(long)]
        full: bool,
        /// Raw query vector as comma-separated floats (bypasses auto-embedding).
        #[arg(long)]
        vector: Option<String>,
        /// Search beam width for HNSW (ef_search).
        #[arg(long, default_value = "200")]
        ef: usize,
        /// Minimum normalized score threshold (0.0-1.0). Results below this are dropped.
        #[arg(long, default_value = "0.25")]
        min_score: f32,
    },
    /// List symbols matching filters.
    ///
    /// JSON schema: {_s, "file":[{l=lines,k=kind,n=name}]}
    #[command(visible_alias = "sym")]
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
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Find symbol definition location.
    ///
    /// JSON schema: {_s, "file":[{l=lines,k=kind,n=name}]}
    #[command(visible_alias = "d")]
    Def {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to find.
        name: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Find all callers of a function.
    ///
    /// JSON schema: {_s, "file":[{l=lines,k=kind,n=name}]}
    #[command(visible_alias = "c")]
    Callers {
        /// Path to the database directory.
        db: PathBuf,
        /// Function name to find callers of.
        function: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Find all references to a symbol.
    ///
    /// JSON schema: {_s, "file":[{l=lines,k=kind,n=name,v=via}]}
    #[command(visible_alias = "r")]
    Refs {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to find references to.
        name: String,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Show file-level dependency graph from code chunks.
    ///
    /// Analyses `calls` and `types` fields to map file → file dependencies.
    /// JSON schema: {_s, "file":[{n=name,v=via}]}
    Deps {
        /// Path to the database directory.
        db: PathBuf,
        /// Show dependencies for a specific file only (suffix match).
        file: Option<String>,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
        /// Transitive dependency depth (default: 1 = direct only).
        #[arg(long, default_value = "1")]
        depth: usize,
    },
    /// Show call-graph context of a symbol (forward BFS: callees).
    ///
    /// BFS from target symbol through callees, depth-limited.
    /// Score: `1/(depth+1)`, test code weighted at 0.1.
    /// Test symbols are hidden by default; use `--include-tests` to show them.
    #[command(visible_alias = "ctx")]
    Context {
        /// Path to the database directory.
        db: PathBuf,
        /// Symbol name to explore.
        symbol: String,
        /// Max BFS depth (default: 2).
        #[arg(long, default_value = "2")]
        depth: u32,
        /// Max results to show (default: 20).
        #[arg(short, long, default_value = "20")]
        k: usize,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
        /// Include test symbols in results (hidden by default).
        #[arg(long)]
        include_tests: bool,
        /// Show reasoning details for each symbol in results.
        #[arg(long)]
        detail: bool,
    },
    /// Show impact of changing a symbol (reverse BFS: callers).
    ///
    /// "If I change this, what breaks?" — traverses callers direction.
    /// Test symbols are hidden by default; use `--include-tests` to show them.
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
        format: crate::commands::code_intel::OutputFormat,
        /// Include test symbols in results (hidden by default).
        #[arg(long)]
        include_tests: bool,
        /// Show reasoning details for each symbol in results.
        #[arg(long)]
        detail: bool,
    },
    /// Find shortest call path between two symbols.
    ///
    /// BFS on call graph (callees direction) from symbol A to symbol B.
    /// Shows the shortest chain of function calls connecting them.
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
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Gather context + impact for full symbol understanding.
    ///
    /// Merges forward (callees) and reverse (callers) BFS results into
    /// a single "read this code to understand the symbol" view.
    /// Test symbols are hidden by default; use `--include-tests` to show them.
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
        format: crate::commands::code_intel::OutputFormat,
        /// Include test symbols in results (hidden by default).
        #[arg(long)]
        include_tests: bool,
        /// Show reasoning details for each symbol in results.
        #[arg(long)]
        detail: bool,
    },
    /// View or manage reasoning (design decisions, history) for a symbol.
    ///
    /// Without mutation flags, shows the current reasoning entry.
    /// With flags, creates or updates the entry.
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
        /// Delete the reasoning entry.
        #[arg(long)]
        delete: bool,
    },
    /// Show per-crate code statistics (functions, structs, enums).
    ///
    /// JSON schema: {_s, "crate":{p=prod_fn,t=test_fn,s=struct,e=enum}}
    Stats {
        /// Path to the database directory.
        db: PathBuf,
        /// Output format (text or json).
        #[arg(long, default_value = "text")]
        format: crate::commands::code_intel::OutputFormat,
    },
    /// Start daemon server for fast embedding search.
    Serve {
        /// Path to database directory to preload (optional).
        db: Option<PathBuf>,
        /// TCP port to listen on (default: 19530).
        #[arg(long, default_value = "19530")]
        port: u16,
        /// Idle timeout in seconds (default: 300 = 5 minutes).
        #[arg(long, default_value = "300")]
        timeout: u64,
        /// Run in background (daemonize).
        #[arg(long)]
        background: bool,
    },
}

/// Distance metric type.
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum MetricType {
    /// Cosine similarity (1 - cosine).
    Cosine,
    /// L2 (Euclidean) distance.
    L2,
    /// Dot product distance (1 - dot).
    Dot,
}
