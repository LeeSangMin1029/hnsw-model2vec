//! v-code — Code intelligence CLI.

mod cli;
mod commands;

use clap::Parser;
use v_hnsw_cli::error::CliError;

fn main() {
    #[allow(clippy::expect_used)]
    let directive = "v_hnsw=info".parse().expect("static directive");
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env().add_directive(directive),
        )
        .init();

    v_hnsw_cli::interrupt::install_handler();

    if let Err(err) = run() {
        let cli_err = CliError::from(err);
        eprintln!("Error: {cli_err}");
        std::process::exit(1);
    }
}

fn run() -> anyhow::Result<()> {
    use cli::{Cli, Commands};
    use commands::intel;

    let cli = Cli::parse();

    match cli.command {
        Commands::Symbols { db, name, kind, format } => {
            intel::run_symbols(db, name, kind, format)
        }
        Commands::Def { db, name, format } => intel::run_def(db, name, format),
        Commands::Refs { db, name, format } => intel::run_refs(db, name, format),
        Commands::Deps { db, file, format, depth } => {
            intel::deps::run_deps(db, file, format, depth)
        }
        Commands::Impact { db, symbol, depth, format, include_tests, detail } => {
            intel::run_impact(db, symbol, depth, format, include_tests, detail)
        }
        Commands::Trace { db, from, to, format } => {
            intel::run_trace(db, from, to, format)
        }
        Commands::Gather { db, symbol, depth, k, format, include_tests, detail } => {
            intel::run_gather(db, symbol, depth, k, format, include_tests, detail)
        }
        Commands::Detail {
            db, symbol, add, decision, why, constraint, rejected,
            failure, fix, root_cause, reject_reason, reject_condition,
            resolve, invalidate, all, delete, file_path, line_range, relate,
        } => intel::detail::run_detail(intel::detail::DetailParams {
            db, symbol, add, decision, why, constraint, rejected,
            failure, fix, root_cause, reject_reason, reject_condition,
            resolve, invalidate, show_all: all, delete, file_path, line_range, relate,
        }),
        Commands::Dupes { db, threshold, exclude_tests, k, json, ast, all, min_lines } => {
            commands::dupes::run(commands::dupes::DupesConfig {
                db, threshold, exclude_tests, k, json, ast_mode: ast, all_mode: all, min_lines,
            })
        }
        Commands::Stats { db, format } => intel::run_stats(db, format),
        Commands::Add { db, input, exclude } => commands::add::run(db, input, &exclude),
        Commands::Update { db, input, exclude } => {
            commands::add::run_update(db, input, &exclude)
        }
        Commands::Find { db, query, k, full, min_score, no_rerank } => {
            commands::find::run(commands::find::FindParams {
                db, query, k, full, min_score, no_rerank,
            })
        }
    }
}
