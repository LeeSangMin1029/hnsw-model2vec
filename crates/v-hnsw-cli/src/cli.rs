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
    /// Insert vectors from a file (JSONL, Parquet, fvecs, bvecs).
    Insert {
        /// Path to the database directory.
        path: PathBuf,
        /// Input file (.jsonl, .ndjson, .parquet, .fvecs, .bvecs).
        #[arg(short, long)]
        input: PathBuf,
        /// Vector column name for Parquet files.
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
    /// Import data from JSONL file.
    Import {
        /// Path to the database directory.
        path: PathBuf,
        /// Input JSONL file.
        #[arg(short, long)]
        input: PathBuf,
    },
    /// Compare v-hnsw performance on JSONL data (e.g., Claude sessions).
    Compare {
        /// Path to directory containing JSONL files.
        #[arg(long)]
        jsonl_dir: PathBuf,
        /// Number of queries to run.
        #[arg(long, default_value = "100")]
        queries: usize,
        /// Top-k results per query.
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Chunk size in characters.
        #[arg(long, default_value = "500")]
        chunk_size: usize,
        /// Maximum number of chunks to process (optional).
        #[arg(long)]
        max_chunks: Option<usize>,
    },
    /// Index markdown files (compatible with qmd collections).
    IndexMd {
        /// Input directory containing markdown files.
        #[arg(short, long)]
        input: PathBuf,
        /// Output directory for the index.
        #[arg(short, long)]
        output: PathBuf,
        /// Chunk size in characters.
        #[arg(long, default_value = "500")]
        chunk_size: usize,
        /// Chunk overlap in characters.
        #[arg(long, default_value = "100")]
        chunk_overlap: usize,
        /// Glob pattern for markdown files.
        #[arg(long)]
        pattern: Option<String>,
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
    },
    /// Incrementally update database with changed files.
    Update {
        /// Path to the database directory.
        db: PathBuf,
        /// Input folder to scan for changes.
        input: PathBuf,
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
