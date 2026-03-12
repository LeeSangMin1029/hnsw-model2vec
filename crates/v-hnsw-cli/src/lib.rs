//! v-hnsw CLI library — shared code for v-hnsw and v-code binaries.

#[cfg(feature = "doc")]
pub mod chunk;
#[cfg(feature = "doc")]
pub mod cli;
pub use v_code_chunk as chunk_code;
pub mod commands;
pub mod error;
pub mod interrupt;

#[cfg(test)]
mod tests;

pub use interrupt::is_interrupted;

// ── Entry points called by thin binary crates ─────────────────────────

/// Run the v-hnsw (document) CLI.
#[cfg(feature = "doc")]
pub fn run_doc() -> anyhow::Result<()> {
    use clap::Parser;
    use cli::{Cli, Commands};

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
        Commands::Find {
            db,
            query,
            k,
            tag,
            exclude_tests,
            full,
            vector,
            ef,
            min_score,
        } => {
            let mut tags = tag;
            if exclude_tests {
                tags.push("role:prod".to_string());
            }
            commands::find::run(commands::find::FindParams {
                db,
                query,
                k,
                tags,
                full,
                vector,
                ef,
                min_score,
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

