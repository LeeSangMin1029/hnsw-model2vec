//! v-code — Code intelligence CLI.

mod cli;
use v_code::commands;

use anyhow::Context as _;
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

    // Run on a thread with 32 MB stack — graph build + SCIP parsing need deep stacks.
    let result = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(run)
        .expect("failed to spawn main thread")
        .join()
        .expect("main thread panicked");

    if let Err(err) = result {
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
        Commands::Context { db, symbol, depth, format, source, include_tests } => {
            intel::run_context(db, symbol, depth, format, source, include_tests)
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
        Commands::Untested { db, depth, format, file } => intel::run_untested(db, depth, format, file),
        Commands::Strings { db, query, callee } => intel::run_strings(db, query, callee),
        Commands::Flow { db, query, depth } => intel::run_flow(db, query, depth),
        Commands::Add { db, input, exclude } => commands::add::run(db, input, &exclude),
        Commands::Embed { db } => commands::embed::run(db),
        Commands::Replace { db, symbol, file, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::replace(db, symbol, file, body)
        }
        Commands::InsertAfter { db, symbol, file, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::insert_after(db, symbol, file, body)
        }
        Commands::InsertBefore { db, symbol, file, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::insert_before(db, symbol, file, body)
        }
        Commands::DeleteSymbol { db, symbol, file } => {
            commands::edit::delete_symbol(db, symbol, file)
        }
        Commands::InsertAt { db, file, line, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::insert_at(db, file, line, body)
        }
        Commands::DeleteLines { db, file, start, end } => {
            commands::edit::delete_lines(db, file, start, end)
        }
        Commands::ReplaceLines { db, file, start, end, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::replace_lines(db, file, start, end, body)
        }
        Commands::CreateFile { db, file, body, body_file } => {
            let body = read_body(body, body_file)?;
            commands::edit::create_file(db, file, body)
        }
    }
}

/// Read body from `--body`, `--body-file`, or stdin (in that priority).
fn read_body(body: Option<String>, body_file: Option<std::path::PathBuf>) -> anyhow::Result<String> {
    if let Some(b) = body {
        return Ok(b);
    }
    if let Some(path) = body_file {
        return std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read body file: {}", path.display()));
    }
    use std::io::Read;
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    Ok(buf)
}
