//! Vector file readers — auto-detect format by extension.

pub mod fvecs;
pub mod jsonl;
pub mod parquet;

use std::path::Path;

use anyhow::{Context, Result};

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
    /// Text column name override (default: `"text"`).
    pub text_column: &'a str,
}

impl<'a> ReaderConfig<'a> {
    /// Shorthand for the common non-embed case.
    pub fn with_vector(vector_column: &'a str) -> Self {
        Self {
            vector_column: Some(vector_column),
            text_column: "text",
        }
    }
}

/// Open a reader appropriate for `path`, auto-detected by extension.
///
/// * `.jsonl` / `.ndjson` -> [`jsonl::JsonlReader`]
/// * `.parquet`            -> [`parquet::ParquetReader`]
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
        "parquet" => {
            let r = parquet::ParquetReader::open(path, cfg.vector_column, cfg.text_column)
                .with_context(|| format!("opening Parquet reader for {}", path.display()))?;
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
            "unsupported file extension '.{other}' — expected .jsonl, .ndjson, .parquet, .fvecs, or .bvecs"
        ),
    }
}
