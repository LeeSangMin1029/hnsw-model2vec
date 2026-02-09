//! v-hnswd - Daemon server and heavy CLI for v-hnsw vector database.
//!
//! Contains all heavy operations: embedding, search, indexing.
//! Can run as both a daemon server and a direct CLI.

pub mod chunk;
mod cli;
mod commands;
pub mod error;

use anyhow::Result;
use clap::Parser;
use std::sync::atomic::{AtomicBool, Ordering};

use cli::{Cli, Commands};

/// Global flag for Ctrl+C handling.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Check if Ctrl+C was pressed.
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

fn main() -> Result<()> {
    // Initialize tracing (default: v_hnsw=info, override via RUST_LOG)
    #[allow(clippy::expect_used)]
    let directive = "v_hnsw=info".parse().expect("static directive");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(directive),
        )
        .init();

    // Setup Ctrl+C handler
    if let Err(e) = ctrlc::set_handler(move || {
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
        } => commands::insert::run(
            path,
            input,
            &vector_column,
            embed,
            &text_column,
            &model,
            batch_size,
        ),
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
        Commands::Add { db, input } => commands::add::run(db, input),
        Commands::Update { db, input } => commands::update::run(db, input),
        Commands::Find { db, query, k, tag, full, fast, vector, ef } => {
            commands::find::run(commands::find::FindParams {
                db, query, k, tags: tag, full, fast, vector, ef,
            })
        }
        Commands::Serve {
            db,
            port,
            timeout,
            background,
        } => commands::serve::run(db, port, timeout, background),
    }
}
