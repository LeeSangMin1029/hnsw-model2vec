//! v-hnswd - Daemon server and heavy CLI for v-hnsw vector database.
//!
//! Contains all heavy operations: embedding, search, indexing.
//! Can run as both a daemon server and a direct CLI.

pub mod chunk;
mod cli;
mod commands;
pub mod error;

use clap::Parser;
use std::sync::atomic::{AtomicBool, Ordering};

use cli::{Cli, Commands};
use error::CliError;

/// Global flag for Ctrl+C handling.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

/// Check if Ctrl+C was pressed.
pub fn is_interrupted() -> bool {
    INTERRUPTED.load(Ordering::Relaxed)
}

fn main() {
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

    if let Err(err) = run() {
        let cli_err = CliError::from(err);
        tracing::error!("{cli_err:#}");
        log_to_file(&cli_err);
        eprintln!("Error: {cli_err}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
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
        Commands::Find { db, query, k, tag, full, vector, ef } => {
            commands::find::run(commands::find::FindParams {
                db, query, k, tags: tag, full, vector, ef,
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

/// Best-effort append error to `~/.v-hnsw/logs/v-hnsw.log`.
fn log_to_file(err: &CliError) {
    use std::io::Write;

    let log_dir = v_hnsw_core::data_dir().join("logs");
    let _ = std::fs::create_dir_all(&log_dir);

    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("v-hnsw.log"))
    else {
        return;
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let _ = writeln!(f, "[{ts}] {err:?}");
}
