//! Add command - Unified data ingestion with automatic embedding.
//!
//! Detects input type (folder with .md files, .jsonl) and processes accordingly.
//! Auto-creates database, embeds text, builds indexes.

mod ingest;
mod pipeline;

#[cfg(test)]
mod tests;

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
    /// Folder containing code files (.rs, etc.).
    CodeFolder,
    /// JSONL file.
    Jsonl,
}

/// Detect input type from path (recursive for directories).
fn detect_input_type(path: &Path, exclude: &[String]) -> Result<InputType> {
    if path.is_dir() {
        // Recursively check what kind of files the folder contains
        let mut has_md = false;
        let mut has_code = false;
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
                match ext {
                    "md" | "markdown" => has_md = true,
                    e if crate::chunk_code::is_supported_code_file(e) => has_code = true,
                    _ => {}
                }
            }
            if has_md && has_code {
                break;
            }
        }
        // Code takes priority when both exist (code folder may contain README.md)
        if has_code {
            Ok(InputType::CodeFolder)
        } else if has_md {
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
            Some("md") | Some("markdown") => {
                anyhow::bail!(
                    "Single markdown file not supported. Use a folder containing .md files."
                )
            }
            _ => anyhow::bail!(
                "Unsupported file type: {}. Supported: folder with .md files, .jsonl",
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
    println!("Input type: {:?}", input_type);
    println!("Input path: {}", input_path.display());
    println!("Database:   {}", db_path.display());
    println!();

    // Create model
    let model = common::create_model()?;
    let model_name = common::DEFAULT_MODEL;

    // Ensure database exists
    let mut engine = common::ensure_database(&db_path, model.dim(), model_name, true)?;

    // Record content type in config
    let content_type = match input_type {
        InputType::CodeFolder => "code",
        InputType::MarkdownFolder => "markdown",
        InputType::Jsonl => "mixed",
    };
    if let Ok(mut config) = DbConfig::load(&db_path)
        && (config.content_type.is_empty() || config.content_type == "mixed") {
            config.content_type = content_type.to_owned();
            let _ = config.save(&db_path);
        }

    // Process based on input type
    let (inserted, errors, added_ids) = match input_type {
        InputType::MarkdownFolder => {
            ingest::process_markdown_folder(&db_path, &input_path, &model, &mut engine, exclude)?
        }
        InputType::CodeFolder => {
            ingest::process_code_folder(&db_path, &input_path, &model, &mut engine, exclude)?
        }
        InputType::Jsonl => {
            ingest::process_jsonl(&db_path, &input_path, &model, &mut engine)?
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

    // Update indexes incrementally (falls back to full rebuild if no existing index)
    let config = DbConfig::load(&db_path)?;
    common::update_indexes_incremental(&db_path, &engine, &config, &added_ids, &[])?;

    // Notify daemon to reload if running
    if let Ok(()) = super::serve::notify_daemon_reload(&db_path) {
        println!("Daemon notified to reload indexes.");
    }

    tracing::info!(inserted, errors, "Add command completed");
    println!();
    println!("Done! Database ready at: {}", db_path.display());

    Ok(())
}
