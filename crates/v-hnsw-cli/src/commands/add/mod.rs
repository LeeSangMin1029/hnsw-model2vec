//! Add command - Unified data ingestion with automatic embedding.
//!
//! Detects input type (folder with .md files, .jsonl) and processes accordingly.
//! Auto-creates database, embeds text, builds indexes.

pub(crate) mod ingest;

use std::path::{Path, PathBuf};

use anyhow::Result;
use v_hnsw_embed::EmbeddingModel;

use super::common;
use super::db_config::DbConfig;
use crate::is_interrupted;

/// Input type detected from the path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputType {
    /// Folder containing markdown files.
    MarkdownFolder,
    /// Single markdown file.
    SingleMarkdown,
    /// JSONL file.
    Jsonl,
}

/// Detect input type from path (recursive for directories).
fn detect_input_type(path: &Path, exclude: &[String]) -> Result<InputType> {
    if path.is_dir() {
        // Recursively check for markdown files
        let mut has_md = false;
        for entry in walkdir::WalkDir::new(path)
            .max_depth(5)
            .into_iter()
            .filter_entry(|e| {
                if e.file_type().is_dir() {
                    !common::should_skip_dir(e.file_name(), exclude)
                } else {
                    true
                }
            })
            .filter_map(|e| e.ok())
        {
            if let Some(ext) = entry.path().extension().and_then(|e| e.to_str()) {
                if ext == "md" || ext == "markdown" {
                    has_md = true;
                    break;
                }
            }
        }
        if has_md {
            Ok(InputType::MarkdownFolder)
        } else {
            anyhow::bail!(
                "Directory contains no supported files: {}",
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
            Some("md") | Some("markdown") => Ok(InputType::SingleMarkdown),
            _ => anyhow::bail!(
                "Unsupported file type: {}. Supported: .md, .markdown, .jsonl, or folder",
                path.display()
            ),
        }
    } else {
        anyhow::bail!("Path not found: {}", path.display());
    }
}

/// Run the add command.
pub fn run(db_path: PathBuf, input_path: PathBuf, exclude: &[String]) -> Result<()> {
    // Detect input type
    let input_type = detect_input_type(&input_path, exclude)?;

    tracing::info!(
        input_type = ?input_type,
        input = %input_path.display(),
        db = %db_path.display(),
        "Starting add command"
    );
    println!("{:?}: {}", input_type, input_path.display());
    println!("Database: {}", db_path.display());

    // Create model
    let model = common::create_model()?;
    let model_name = common::DEFAULT_MODEL;

    // Ensure database exists
    let mut engine = common::ensure_database(&db_path, model.dim(), model_name, true, false)?;

    // Update config: store input_path
    if let Ok(mut config) = DbConfig::load(&db_path) {
        if let Ok(canonical) = input_path.canonicalize() {
            config.input_path = Some(canonical.to_string_lossy().into_owned());
        }
        let _ = config.save(&db_path);
    }

    // Process based on input type
    let result = match input_type {
        InputType::MarkdownFolder => {
            ingest::process_markdown_folder(&db_path, &input_path, &model, &mut engine, exclude)?
        }
        InputType::SingleMarkdown => {
            ingest::process_markdown_files(&db_path, std::slice::from_ref(&input_path), &model, &mut engine)?
        }
        InputType::Jsonl => {
            ingest::process_jsonl(&db_path, &input_path, &model, &mut engine)?
        }
    };

    let inserted = result.inserted;
    let errors = result.errors;

    if is_interrupted() {
        println!();
        println!("Operation interrupted. Partial data may have been inserted.");
        return Ok(());
    }

    if inserted == 0 && errors == 0 {
        println!("No data to process.");
        return Ok(());
    }

    // Update indexes incrementally (falls back to full rebuild if no existing index)
    let config = DbConfig::load(&db_path)?;
    common::update_indexes_incremental(&db_path, &engine, &config, &result.added_ids, &result.removed_ids)?;

    // Notify daemon to reload if running
    if let Ok(()) = super::serve::notify_daemon_reload(&db_path) {
        println!("Daemon notified to reload indexes.");
    }

    tracing::info!(inserted, errors, "Add command completed");
    println!("Done: {}", db_path.display());

    Ok(())
}
