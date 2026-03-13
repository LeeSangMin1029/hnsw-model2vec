//! Vector file readers — auto-detect format by extension.

pub mod fvecs;
pub mod jsonl;

use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};

/// Open a file or return a single-element error iterator.
///
/// Intended for [`VectorReader::records`] implementations that need to re-open
/// the source file from the beginning.
pub(crate) fn open_or_error_iter(
    path: &Path,
) -> std::result::Result<File, Box<dyn Iterator<Item = Result<InputRecord>>>> {
    File::open(path).map_err(|e| {
        Box::new(std::iter::once(Err(anyhow::anyhow!(
            "cannot re-open {}: {e}",
            path.display()
        )))) as Box<dyn Iterator<Item = Result<InputRecord>>>
    })
}

/// A single record produced by any reader.
pub struct InputRecord {
    /// Point ID.
    pub id: u64,
    /// Dense vector.
    pub vector: Vec<f32>,
    /// Optional text for BM25.
    pub text: Option<String>,
    /// Optional source document path.
    pub source: Option<String>,
    /// Optional tags.
    pub tags: Option<Vec<String>>,
}

/// Common interface for all vector file readers.
pub trait VectorReader: Send {
    /// Total number of records (or best estimate).
    fn count(&mut self) -> Result<usize>;
    /// Lazily iterate over records.
    fn records(&mut self) -> Box<dyn Iterator<Item = Result<InputRecord>> + '_>;
}

/// Reader configuration.
pub struct ReaderConfig<'a> {
    /// Vector column name. `None` in embed mode (vector not required).
    pub vector_column: Option<&'a str>,
}

impl<'a> ReaderConfig<'a> {
    /// Shorthand for the common non-embed case.
    pub fn with_vector(vector_column: &'a str) -> Self {
        Self {
            vector_column: Some(vector_column),
        }
    }
}

/// Open a reader appropriate for `path`, auto-detected by extension.
///
/// * `.jsonl` / `.ndjson` -> [`jsonl::JsonlReader`]
/// * `.fvecs` / `.bvecs`   -> [`fvecs::FvecsReader`]
pub fn open_reader(path: &Path, cfg: &ReaderConfig<'_>) -> Result<Box<dyn VectorReader>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    match ext.as_str() {
        "jsonl" | "ndjson" => {
            let r = jsonl::JsonlReader::open(path)
                .with_context(|| format!("opening JSONL reader for {}", path.display()))?;
            Ok(Box::new(r))
        }
        "fvecs" | "bvecs" => {
            if cfg.vector_column.is_none() {
                anyhow::bail!("--embed mode is not supported with fvecs/bvecs files (no text column)");
            }
            let r = fvecs::FvecsReader::open(path)
                .with_context(|| format!("opening fvecs/bvecs reader for {}", path.display()))?;
            Ok(Box::new(r))
        }
        other => anyhow::bail!(
            "unsupported file extension '.{other}' — expected .jsonl, .ndjson, .fvecs, or .bvecs"
        ),
    }
}
