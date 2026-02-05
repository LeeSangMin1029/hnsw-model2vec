//! JSONL / NDJSON reader.

use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::path::Path;

use anyhow::{Context, Result};
use serde::Deserialize;

use super::{InputRecord, VectorReader};

/// Deserializable record matching the JSONL schema.
#[derive(Debug, Deserialize)]
struct JsonlRecord {
    id: u64,
    /// Dense vector. Optional in embed mode (generated from text).
    #[serde(default)]
    vector: Option<Vec<f32>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    tags: Option<Vec<String>>,
}

/// Reads records from a JSONL (newline-delimited JSON) file.
pub struct JsonlReader {
    path: std::path::PathBuf,
    line_count: usize,
}

impl JsonlReader {
    /// Open and pre-count lines.
    pub fn open(path: &Path) -> Result<Self> {
        let f = File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let line_count = BufReader::new(f).lines().count();
        Ok(Self {
            path: path.to_path_buf(),
            line_count,
        })
    }
}

impl VectorReader for JsonlReader {
    fn count(&mut self) -> Result<usize> {
        Ok(self.line_count)
    }

    fn records(&mut self) -> Box<dyn Iterator<Item = Result<InputRecord>> + '_> {
        // Re-open to iterate from the beginning.
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) => {
                return Box::new(std::iter::once(Err(anyhow::anyhow!(
                    "cannot re-open {}: {e}",
                    self.path.display()
                ))));
            }
        };

        let lines: Lines<BufReader<File>> = BufReader::new(file).lines();

        Box::new(lines.enumerate().filter_map(|(idx, line_res)| {
            let line = match line_res {
                Ok(l) => l,
                Err(e) => return Some(Err(anyhow::anyhow!("line {}: read error: {e}", idx + 1))),
            };
            if line.trim().is_empty() {
                return None; // skip blanks
            }
            let rec: JsonlRecord = match serde_json::from_str(&line) {
                Ok(r) => r,
                Err(e) => {
                    return Some(Err(anyhow::anyhow!("line {}: parse error: {e}", idx + 1)));
                }
            };
            Some(Ok(InputRecord {
                id: rec.id,
                vector: rec.vector.unwrap_or_default(),
                text: rec.text,
                source: rec.source,
                tags: rec.tags,
            }))
        }))
    }
}
