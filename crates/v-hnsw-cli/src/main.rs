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
        /// Auto-embed text using fastembed model (skips vector column requirement).
        #[arg(long)]
        embed: bool,
        /// Text column name to embed (used with --embed).
        #[arg(long, default_value = "text")]
        text_column: String,
        /// Embedding model to use (used with --embed).
        #[arg(long, default_value = "all-mini-lm-l6-v2")]
        model: String,
        /// Batch size for embedding (used with --embed).
        #[arg(long, default_value = "128")]
        batch_size: usize,
        /// Device for embedding inference (cpu, cuda, cuda:N, directml, directml:N).
        #[arg(long, default_value = "cpu")]
        device: String,
        /// Use FP16 model via direct ort Session (faster, less VRAM).
        #[arg(long)]
        fp16: bool,
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
        /// Collection to search in.
        #[arg(long, default_value = "default")]
        collection: String,
        /// Enable cross-encoder reranking.
        #[arg(long)]
        rerank: bool,
        /// Cross-encoder model (minilm or bge).
        #[arg(long, default_value = "minilm")]
        rerank_model: String,
        /// Number of candidates to rerank (default: 3x k).
        #[arg(long)]
        rerank_top: Option<usize>,
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
        action: commands::collection::CollectionAction,
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
    /// Semantic vector search with auto-embedding.
    Vsearch {
        /// Path to the database directory.
        path: PathBuf,
        /// Query text to search for.
        query: String,
        /// Number of results to return.
        #[arg(short, long, default_value = "10")]
        k: usize,
        /// Search beam width (ef_search).
        #[arg(long, default_value = "200")]
        ef: usize,
        /// Embedding model (auto-detected from DB if not specified).
        #[arg(long)]
        model: Option<String>,
        /// Show document text in results.
        #[arg(long)]
        show_text: bool,
    },
    /// Add data to database (auto-detect input type, auto-embed, auto-create DB).
    Add {
        /// Path to the database directory.
        db: PathBuf,
        /// Input file or folder (folder with .md files, .jsonl, or .parquet).
        input: PathBuf,
    },
    /// Search database with hybrid HNSW + BM25 (auto-load model from config).
    Find {
        /// Path to the database directory.
        db: PathBuf,
        /// Search query text.
        query: String,
        /// Number of results to return.
        #[arg(short, long, default_value = "10")]
        k: usize,
    },
    /// Start daemon server for fast embedding search.
    Serve {
        /// Path to the database directory.
        db: PathBuf,
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
            embed,
            text_column,
            model,
            batch_size,
            device,
            fp16,
        } => commands::insert::run(path, input, &vector_column, embed, &text_column, &model, batch_size, &device, fp16),
        Commands::Search {
            path,
            vector,
            text,
            k,
            ef,
            collection,
            rerank,
            rerank_model,
            rerank_top,
        } => commands::search::run(commands::search::SearchParams {
            path,
            vector,
            text,
            k,
            ef,
            collection,
            rerank,
            rerank_model,
            rerank_top,
        }),
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
        Commands::Collection { path, action } => commands::collection::run(path, action),
        Commands::BuildIndex { path } => commands::buildindex::run(path),
        Commands::Get { path, ids } => commands::get::run(path, ids),
        Commands::Vsearch {
            path,
            query,
            k,
            ef,
            model,
            show_text,
        } => commands::vsearch::run(commands::vsearch::VSearchParams {
            path,
            query,
            k,
            ef,
            model,
            show_text,
        }),
        Commands::Add { db, input } => commands::add::run(db, input),
        Commands::Find { db, query, k } => commands::find::run(db, query, k),
        Commands::Serve { db, port, timeout, background } => commands::serve::run(db, port, timeout, background),
    }
}
