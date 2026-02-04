//! v-hnsw CLI - Command-line interface for v-hnsw vector database.

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Global flag for Ctrl+C handling.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Check if Ctrl+C was pressed.
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

/// v-hnsw: Local vector database CLI.
#[derive(Parser)]
#[command(name = "v-hnsw")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
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
    },
    /// Search for nearest neighbors.
    Search {
        /// Path to the database directory.
        path: PathBuf,
        /// Query vector as comma-separated floats.
        #[arg(short, long)]
        vector: Option<String>,
        /// Query text for BM25 search.
        #[arg(short, long)]
        text: Option<String>,
        /// Number of results to return.
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Search beam width (ef_search).
        #[arg(long, default_value = "200")]
        ef: usize,
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

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::WARN.into()),
        )
        .init();

    // Setup Ctrl+C handler
    let interrupted = Arc::new(AtomicBool::new(false));
    let i = interrupted.clone();
    if let Err(e) = ctrlc::set_handler(move || {
        i.store(true, Ordering::SeqCst);
        INTERRUPTED.store(true, Ordering::SeqCst);
        eprintln!("\nInterrupted. Cleaning up...");
    }) {
        eprintln!("Warning: Failed to set Ctrl+C handler: {e}");
    }

    let cli = Cli::parse();

    match cli.command {
        Commands::Create {
            path,
            dim,
            metric,
            m,
            ef,
            korean,
        } => commands::create::run(path, dim, metric, m, ef, korean),
        Commands::Info { path } => commands::info::run(path),
        Commands::Insert {
            path,
            input,
            vector_column,
        } => commands::insert::run(path, input, &vector_column),
        Commands::Search {
            path,
            vector,
            text,
            k,
            ef,
        } => commands::search::run(path, vector, text, k, ef),
        Commands::Delete { path, id } => commands::delete::run(path, id),
        Commands::Bench { path, queries, k } => commands::bench::run(path, queries, k),
        Commands::Export { path, output } => commands::export::run(path, output),
        Commands::Import { path, input } => commands::import::run(path, input),
        Commands::Compare {
            jsonl_dir,
            queries,
            k,
            chunk_size,
            max_chunks,
        } => commands::compare::run(jsonl_dir, queries, k, chunk_size, max_chunks),
        Commands::IndexMd {
            input,
            output,
            chunk_size,
            chunk_overlap,
            pattern,
        } => commands::index_md::run(input, output, chunk_size, chunk_overlap, pattern),
    }
}
