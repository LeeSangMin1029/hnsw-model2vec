//! Add command - Unified data ingestion with automatic embedding.
//!
//! Detects input type (folder with .md files, .jsonl, .parquet) and processes accordingly.
//! Auto-creates database, embeds text, builds indexes.

mod ingest;
mod pipeline;

use std::path::{Path, PathBuf};

use anyhow::Result;
use v_hnsw_embed::EmbeddingModel;

use super::common;
use super::create::DbConfig;
use crate::is_interrupted;

/// Input type detected from the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputType {
    /// Folder containing markdown files.
    MarkdownFolder,
    /// JSONL file.
    Jsonl,
    /// Parquet file.
    Parquet,
}

/// Detect input type from path (recursive for directories).
fn detect_input_type(path: &Path) -> Result<InputType> {
    if path.is_dir() {
        // Recursively check if folder contains markdown files
        let has_md = walkdir::WalkDir::new(path)
            .max_depth(3)
            .into_iter()
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "md" || ext == "markdown")
                    .unwrap_or(false)
            });
        if has_md {
            Ok(InputType::MarkdownFolder)
        } else {
            anyhow::bail!(
                "Directory contains no markdown files: {}",
                path.display()
            );
        }
    } else if path.is_file() {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        match ext.as_deref() {
            Some("jsonl") | Some("ndjson") => Ok(InputType::Jsonl),
            Some("parquet") => Ok(InputType::Parquet),
            Some("md") | Some("markdown") => {
                anyhow::bail!(
                    "Single markdown file not supported. Use a folder containing .md files."
                )
            }
            _ => anyhow::bail!(
                "Unsupported file type: {}. Supported: folder with .md files, .jsonl, .parquet",
                path.display()
            ),
        }
    } else {
        anyhow::bail!("Path not found: {}", path.display());
    }
}

/// Run the add command.
pub fn run(db_path: PathBuf, input_path: PathBuf) -> Result<()> {
    // Detect input type
    let input_type = detect_input_type(&input_path)?;

    tracing::info!(
        input_type = ?input_type,
        input = %input_path.display(),
        db = %db_path.display(),
        "Starting add command"
    );
    println!("Input type: {:?}", input_type);
    println!("Input path: {}", input_path.display());
    println!("Database:   {}", db_path.display());
    println!();

    // Create model
    let model = common::create_model()?;
    let model_name = common::DEFAULT_MODEL;

    // Ensure database exists
    let mut engine = common::ensure_database(&db_path, model.dim(), model_name, true)?;

    // Process based on input type
    let (inserted, errors) = match input_type {
        InputType::MarkdownFolder => {
            ingest::process_markdown_folder(&db_path, &input_path, &model, &mut engine)?
        }
        InputType::Jsonl => {
            ingest::process_jsonl(&db_path, &input_path, &model, &mut engine)?
        }
        InputType::Parquet => {
            ingest::process_parquet(&db_path, &input_path, &model, &mut engine)?
        }
    };

    if is_interrupted() {
        println!();
        println!("Operation interrupted. Partial data may have been inserted.");
        return Ok(());
    }

    if inserted == 0 && errors == 0 {
        println!("No data to process.");
        return Ok(());
    }

    // Build indexes
    let config = DbConfig::load(&db_path)?;
    common::build_indexes(&db_path, &engine, &config)?;

    // Notify daemon to reload if running
    if let Ok(()) = super::serve::notify_daemon_reload(&db_path) {
        println!("Daemon notified to reload indexes.");
    }

    tracing::info!(inserted, errors, "Add command completed");
    println!();
    println!("Done! Database ready at: {}", db_path.display());

    Ok(())
}
