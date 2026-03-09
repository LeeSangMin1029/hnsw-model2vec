//! v-hnswd - Daemon server and heavy CLI for v-hnsw vector database.
//!
//! Contains all heavy operations: embedding, search, indexing.
//! Can run as both a daemon server and a direct CLI.

pub mod chunk;
pub use v_hnsw_code as chunk_code;
mod cli;
mod commands;
pub mod error;

#[cfg(test)]
mod tests;

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
        Commands::Collection { path, action } => commands::collection::run(path, action),
        Commands::BuildIndex { path } => commands::buildindex::run(path),
        Commands::Get { path, ids } => commands::get::run(path, ids),
        Commands::Add { db, input, exclude } => commands::add::run(db, input, &exclude),
        Commands::Update { db, input, exclude } => commands::update::run(db, input, &exclude),
        Commands::Find { db, query, k, tag, full, vector, ef, min_score } => {
            commands::find::run(commands::find::FindParams {
                db, query, k, tags: tag, full, vector, ef, min_score,
            })
        }
        Commands::Symbols { db, name, kind, format } => {
            commands::code_intel::run_symbols(db, name, kind, format)
        }
        Commands::Def { db, name, format } => commands::code_intel::run_def(db, name, format),
        Commands::Refs { db, name, format } => {
            commands::code_intel::run_refs(db, name, format)
        }
        Commands::Deps { db, file, format, depth } => {
            commands::code_intel::deps::run_deps(db, file, format, depth)
        }
        Commands::Impact { db, symbol, depth, format, include_tests, detail } => {
            commands::code_intel::impact::run_impact(db, symbol, depth, format, include_tests, detail)
        }
        Commands::Trace { db, from, to, format } => {
            commands::code_intel::trace::run_trace(db, from, to, format)
        }
        Commands::Gather { db, symbol, depth, k, format, include_tests, detail } => {
            commands::code_intel::gather::run_gather(db, symbol, depth, k, format, include_tests, detail)
        }
        Commands::Detail { db, symbol, add, decision, why, constraint, rejected, failure, fix, delete } => {
            commands::code_intel::detail::run_detail(commands::code_intel::detail::DetailParams {
                db, symbol, add, decision, why, constraint, rejected, failure, fix, delete,
            })
        }
        Commands::Dupes { db, threshold, exclude_tests, k, json, embed, ast, min_lines } => {
            commands::dupes::run(db, threshold, exclude_tests, k, json, embed, ast, min_lines)
        }
        Commands::Stats { db, format } => commands::code_intel::run_stats(db, format),
        Commands::Serve {
            db,
            port,
            timeout,
            background,
        } => commands::serve::run(db, port, timeout, background),
    }
}

/// Best-effort append error to `~/.v-hnsw/logs/v-hnsw.log`.
///
/// Truncates and restarts the log file when it exceeds 1 MB.
fn log_to_file(err: &CliError) {
    use std::io::Write;

    let log_dir = v_hnsw_core::data_dir().join("logs");
    let _ = std::fs::create_dir_all(&log_dir);

    let log_path = log_dir.join("v-hnsw.log");

    // Truncate if file exceeds 1 MB
    if let Ok(meta) = std::fs::metadata(&log_path)
        && meta.len() > 1_000_000 {
            let _ = std::fs::remove_file(&log_path);
        }

    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    else {
        return;
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let _ = writeln!(f, "[{ts}] {err:?}");
}
