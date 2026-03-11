//! v-code — Code intelligence CLI.
//!
//! Standalone binary for code-only commands: symbols, def, refs, deps,
//! impact, trace, gather, detail, dupes, stats, add, update.

pub mod chunk;
pub use v_hnsw_code as chunk_code;
mod cli;
mod commands;
pub mod error;
pub mod interrupt;
mod v_code_cli;

#[cfg(test)]
mod tests;

use clap::Parser;

use error::CliError;
use v_code_cli::{Cli, Commands};

pub use interrupt::is_interrupted;

fn main() {
    #[allow(clippy::expect_used)]
    let directive = "v_hnsw=info".parse().expect("static directive");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(directive),
        )
        .init();

    if let Err(err) = run() {
        let cli_err = CliError::from(err);
        eprintln!("Error: {cli_err}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Symbols { db, name, kind, format } => {
            commands::code_intel::run_symbols(db, name, kind, format)
        }
        Commands::Def { db, name, format } => commands::code_intel::run_def(db, name, format),
        Commands::Refs { db, name, format } => commands::code_intel::run_refs(db, name, format),
        Commands::Deps { db, file, format, depth } => {
            commands::code_intel::deps::run_deps(db, file, format, depth)
        }
        Commands::Impact { db, symbol, depth, format, include_tests, detail } => {
            commands::code_intel::run_impact(db, symbol, depth, format, include_tests, detail)
        }
        Commands::Trace { db, from, to, format } => {
            commands::code_intel::run_trace(db, from, to, format)
        }
        Commands::Gather { db, symbol, depth, k, format, include_tests, detail } => {
            commands::code_intel::run_gather(db, symbol, depth, k, format, include_tests, detail)
        }
        Commands::Detail {
            db, symbol, add, decision, why, constraint, rejected, failure, fix,
            root_cause, reject_reason, reject_condition, resolve, invalidate, all, delete,
            file_path, line_range, relate,
        } => {
            commands::code_intel::detail::run_detail(commands::code_intel::detail::DetailParams {
                db, symbol, add, decision, why, constraint, rejected, failure, fix,
                root_cause, reject_reason, reject_condition, resolve, invalidate, show_all: all, delete,
                file_path, line_range, relate,
            })
        }
        Commands::Dupes { db, threshold, exclude_tests, k, json, ast, all, min_lines } => {
            commands::dupes::run(commands::dupes::DupesConfig {
                db, threshold, exclude_tests, k, json,
                ast_mode: ast, all_mode: all, min_lines,
            })
        }
        Commands::Stats { db, format } => commands::code_intel::run_stats(db, format),
        Commands::Add { db, input, exclude } => commands::add::run(db, input, &exclude),
        Commands::Update { db, input, exclude } => commands::update::run(db, Some(input), &exclude),
    }
}
