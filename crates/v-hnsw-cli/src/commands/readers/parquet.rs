//! Apache Parquet reader.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{Array, AsArray, ListArray};
use arrow::datatypes::{Float32Type, Int64Type, UInt64Type};
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::file::reader::{FileReader, SerializedFileReader};

use super::{InputRecord, VectorReader};

/// Column indices resolved once on open.
struct ColumnMap {
    id: Option<usize>,
    vector: usize,
    text: Option<usize>,
    source: Option<usize>,
}

/// Reads vectors from an Apache Parquet file.
pub struct ParquetReader {
    row_count: usize,
    file_path: std::path::PathBuf,
    col_map: ColumnMap,
    vector_column: String,
}

/// Case-insensitive column lookup.
fn find_col(schema: &arrow::datatypes::Schema, name: &str) -> Option<usize> {
    let lower = name.to_ascii_lowercase();
    schema
        .fields()
        .iter()
        .position(|f| f.name().to_ascii_lowercase() == lower)
}

impl ParquetReader {
    pub fn open(path: &Path, vector_column: &str) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let file_reader = SerializedFileReader::new(file)
            .with_context(|| "invalid parquet file")?;
        let row_count: usize = file_reader
            .metadata()
            .file_metadata()
            .num_rows()
            .try_into()
            .unwrap_or(0usize);

        // Build a temporary batch reader just to inspect the Arrow schema.
        let file2 = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file2)
            .with_context(|| "cannot build parquet reader")?;
        let schema = builder.schema().clone();

        let vector_idx = find_col(&schema, vector_column).ok_or_else(|| {
            anyhow::anyhow!(
                "parquet file has no '{vector_column}' column — available: {}",
                schema
                    .fields()
                    .iter()
                    .map(|f: &arrow::datatypes::FieldRef| f.name().as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;

        let col_map = ColumnMap {
            id: find_col(&schema, "id"),
            vector: vector_idx,
            text: find_col(&schema, "text"),
            source: find_col(&schema, "source"),
        };

        Ok(Self {
            row_count,
            file_path: path.to_path_buf(),
            col_map,
            vector_column: vector_column.to_owned(),
        })
    }

    /// Open a fresh batch reader.
    fn batch_reader(&self) -> Result<ParquetRecordBatchReader> {
        let file = File::open(&self.file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;
        Ok(reader)
    }
}

impl VectorReader for ParquetReader {
    fn count(&mut self) -> Result<usize> {
        Ok(self.row_count)
    }

    fn records(&mut self) -> Box<dyn Iterator<Item = Result<InputRecord>> + '_> {
        let batch_reader = match self.batch_reader() {
            Ok(r) => r,
            Err(e) => return Box::new(std::iter::once(Err(e))),
        };

        let col_map_id = self.col_map.id;
        let col_map_vector = self.col_map.vector;
        let col_map_text = self.col_map.text;
        let col_map_source = self.col_map.source;
        let vec_col_name = self.vector_column.clone();

        // Global auto-increment for rows missing an id column.
        let mut global_row: u64 = 0;

        Box::new(
            batch_reader
                .flat_map(move |batch_res| {
                    let batch = match batch_res {
                        Ok(b) => b,
                        Err(e) => {
                            return vec![Err(anyhow::anyhow!("parquet batch read error: {e}"))];
                        }
                    };

                    let num_rows = batch.num_rows();
                    let mut out = Vec::with_capacity(num_rows);

                    // --- id column ---
                    let id_col: Option<Arc<dyn Array>> =
                        col_map_id.map(|i| Arc::clone(batch.column(i)));

                    // --- vector column ---
                    let vec_col = Arc::clone(batch.column(col_map_vector));

                    // --- text column ---
                    let text_col: Option<Arc<dyn Array>> =
                        col_map_text.map(|i| Arc::clone(batch.column(i)));

                    // --- source column ---
                    let source_col: Option<Arc<dyn Array>> =
                        col_map_source.map(|i| Arc::clone(batch.column(i)));

                    for row in 0..num_rows {
                        // id
                        let id = match &id_col {
                            Some(col) => extract_id(col.as_ref(), row, global_row + row as u64),
                            None => global_row + row as u64,
                        };

                        // vector
                        let vector = match extract_vector(vec_col.as_ref(), row) {
                            Ok(v) => v,
                            Err(e) => {
                                out.push(Err(anyhow::anyhow!(
                                    "row {}: failed to read '{vec_col_name}': {e}",
                                    global_row + row as u64
                                )));
                                continue;
                            }
                        };

                        // text
                        let text = text_col
                            .as_ref()
                            .and_then(|c| c.as_string_opt::<i32>())
                            .and_then(|a| {
                                if a.is_null(row) {
                                    None
                                } else {
                                    Some(a.value(row).to_owned())
                                }
                            });

                        // source
                        let source = source_col
                            .as_ref()
                            .and_then(|c| c.as_string_opt::<i32>())
                            .and_then(|a| {
                                if a.is_null(row) {
                                    None
                                } else {
                                    Some(a.value(row).to_owned())
                                }
                            });

                        out.push(Ok(InputRecord {
                            id,
                            vector,
                            text,
                            source,
                            tags: None,
                        }));
                    }

                    global_row += num_rows as u64;
                    out
                }),
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a u64 id from either UInt64 or Int64 columns.
fn extract_id(col: &dyn Array, row: usize, fallback: u64) -> u64 {
    if col.is_null(row) {
        return fallback;
    }
    if let Some(a) = col.as_primitive_opt::<UInt64Type>() {
        return a.value(row);
    }
    if let Some(a) = col.as_primitive_opt::<Int64Type>() {
        let v = a.value(row);
        if v >= 0 {
            return v as u64;
        }
    }
    fallback
}

/// Extract a `Vec<f32>` from a `List<Float32>` or `FixedSizeList<Float32>` cell.
fn extract_vector(col: &dyn Array, row: usize) -> Result<Vec<f32>> {
    // Try FixedSizeList<Float32>
    if let Some(fsl) = col.as_fixed_size_list_opt() {
        let values = fsl.value(row);
        let f32arr = values
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| anyhow::anyhow!("FixedSizeList child is not Float32"))?;
        return Ok(f32arr.values().to_vec());
    }

    // Try List<Float32>
    if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
        let values = list.value(row);
        let f32arr = values
            .as_primitive_opt::<Float32Type>()
            .ok_or_else(|| anyhow::anyhow!("List child is not Float32"))?;
        return Ok(f32arr.values().to_vec());
    }

    anyhow::bail!("vector column is neither FixedSizeList<Float32> nor List<Float32>");
}
