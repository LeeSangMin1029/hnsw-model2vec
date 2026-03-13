//! v-hnsw CLI library — shared code for v-hnsw and v-code binaries.

pub mod chunk;
pub mod cli;
pub mod commands;
pub mod error;
pub mod interrupt;

#[cfg(test)]
mod tests;

pub use interrupt::is_interrupted;

// ── Entry points called by thin binary crates ─────────────────────────

/// Run the v-hnsw (document) CLI.
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
        } => commands::serve::run(db, port, timeout, background, Some(doc_method_handler)),
    }
}

/// Extension handler for v-hnsw daemon: handles "update" RPC method.
fn doc_method_handler(
    method: &str,
    params: serde_json::Value,
    state: &mut commands::serve::daemon::DaemonState,
) -> Option<anyhow::Result<serde_json::Value>> {
    match method {
        "update" => Some(handle_daemon_update(params, state)),
        _ => None,
    }
}

fn handle_daemon_update(
    params: serde_json::Value,
    state: &mut commands::serve::daemon::DaemonState,
) -> anyhow::Result<serde_json::Value> {
    use std::path::PathBuf;

    #[derive(serde::Deserialize)]
    struct UpdateParams { db: String, input: String, #[serde(default)] exclude: Vec<String> }

    let p: UpdateParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid update params: {e}"))?;
    let db_path = PathBuf::from(&p.db);
    let input_path = PathBuf::from(&p.input);
    let t0 = std::time::Instant::now();

    let key = state.evict_db(&db_path)?;
    let model = state.model()?;

    let stats = commands::update::run_core(&key, &input_path, Some(model), &p.exclude)?;

    state.reload(&key)?;

    eprintln!(
        "[daemon] Update complete: new={} mod={} del={} unchanged={} ({:.0}ms)",
        stats.new, stats.modified, stats.deleted, stats.unchanged,
        t0.elapsed().as_millis()
    );

    Ok(serde_json::to_value(stats)?)
}

