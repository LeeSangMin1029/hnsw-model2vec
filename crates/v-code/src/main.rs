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

/// V_CODE_COMPACT=1 makes --compact the default for all commands.
fn env_compact() -> bool {
    std::env::var("V_CODE_COMPACT").is_ok_and(|v| v == "1" || v == "true")
}

fn run() -> anyhow::Result<()> {
    use cli::{Cli, Commands};
    use commands::intel;

    let cli = Cli::parse();

    match cli.command {
        Commands::Symbols { db, name, kind, format, include_tests, limit, compact } => {
            intel::run_symbols(db, name, kind, format, include_tests, limit, compact || env_compact())
        }
        Commands::Context { db, symbol, depth, format } => {
            intel::run_context(db, symbol, depth, format)
        }
        Commands::Deps { db, file, format, depth } => {
            intel::deps::run_deps(db, file, format, depth)
        }
        Commands::Blast { db, symbol, depth, format, include_tests } => {
            intel::run_blast(db, symbol, depth, format, include_tests)
        }
        Commands::Jump { db, symbol, depth, format } => {
            intel::run_jump(db, symbol, depth, format)
        }
        Commands::Trace { db, from, to, format } => {
            intel::run_trace(db, from, to, format)
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
        Commands::Dupes { db, threshold, exclude_tests, k, json, ast, all, min_lines, min_sub_lines, analyze } => {
            commands::dupes::run(commands::dupes::DupesConfig {
                db, threshold, exclude_tests, k, json, ast_mode: ast, all_mode: all, min_lines, min_sub_lines, analyze,
            })
        }
        Commands::Stats { db, format } => intel::run_stats(db, format),
        Commands::Add { db, input, exclude } => commands::add::run(db, input, &exclude),
        Commands::Serve { db, port, timeout, background } => {
            v_hnsw_cli::commands::serve::run(db, port, timeout, background, None)
        }
    }
}
