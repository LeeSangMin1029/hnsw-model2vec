//! fvecs / bvecs binary format reader (ann-benchmarks standard).
//!
//! **fvecs**: repeated `[dim: u32_le][dim x f32_le]`
//! **bvecs**: repeated `[dim: u32_le][dim x u8]`  (converted to f32)
//!
//! No unsafe code: byte conversion via `from_le_bytes`.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use anyhow::{Context, Result};

use super::{InputRecord, VectorReader};

/// Binary format variant.
#[derive(Clone, Copy)]
enum Variant {
    /// Each element is `f32` (4 bytes).
    Fvecs,
    /// Each element is `u8` (1 byte), promoted to `f32`.
    Bvecs,
}

/// Reads ann-benchmarks fvecs / bvecs files.
pub struct FvecsReader {
    path: std::path::PathBuf,
    variant: Variant,
    /// Pre-computed record count.
    record_count: usize,
}

impl FvecsReader {
    pub fn open(path: &Path) -> Result<Self> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .unwrap_or_default();

        let variant = match ext.as_str() {
            "fvecs" => Variant::Fvecs,
            "bvecs" => Variant::Bvecs,
            other => anyhow::bail!("expected .fvecs or .bvecs, got '.{other}'"),
        };

        // Peek the first 4 bytes to learn the dimension, then compute count.
        let file_len = std::fs::metadata(path)
            .with_context(|| format!("cannot stat {}", path.display()))?
            .len();

        if file_len < 4 {
            anyhow::bail!("file too small to contain a dimension header");
        }

        let mut f = File::open(path)
            .with_context(|| format!("cannot open {}", path.display()))?;
        let dim = read_le::<u32>(&mut f)? as usize;
        if dim == 0 {
            anyhow::bail!("dimension is 0");
        }

        let elem_bytes: usize = match variant {
            Variant::Fvecs => 4,
            Variant::Bvecs => 1,
        };
        let record_bytes = 4 + dim * elem_bytes; // 4-byte header + payload
        if !(file_len as usize).is_multiple_of(record_bytes) {
            anyhow::bail!(
                "file size ({file_len}) is not a multiple of record size ({record_bytes})"
            );
        }
        let record_count = file_len as usize / record_bytes;

        Ok(Self {
            path: path.to_path_buf(),
            variant,
            record_count,
        })
    }
}

impl VectorReader for FvecsReader {
    fn count(&mut self) -> Result<usize> {
        Ok(self.record_count)
    }

    fn records(&mut self) -> Box<dyn Iterator<Item = Result<InputRecord>> + '_> {
        let file = match super::open_or_error_iter(&self.path) {
            Ok(f) => f,
            Err(it) => return it,
        };

        let variant = self.variant;
        let mut reader = BufReader::new(file);
        let mut id: u64 = 0;

        Box::new(std::iter::from_fn(move || {
            match read_one_record(&mut reader, variant) {
                Ok(Some(vector)) => {
                    let rec = InputRecord {
                        id,
                        vector,
                        text: None,
                        source: None,
                        tags: None,
                    };
                    id += 1;
                    Some(Ok(rec))
                }
                Ok(None) => None, // EOF
                Err(e) => {
                    let err_id = id;
                    id += 1;
                    Some(Err(anyhow::anyhow!("record {err_id}: {e}")))
                }
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Low-level helpers (no unsafe)
// ---------------------------------------------------------------------------

/// Read a little-endian primitive (`u32` or `f32`) from a byte stream.
fn read_le<T: LeBytesReadable>(r: &mut impl Read) -> Result<T> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)
        .with_context(|| format!("failed to read {}", std::any::type_name::<T>()))?;
    Ok(T::from_le_bytes(buf))
}

/// Trait for 4-byte little-endian primitives.
trait LeBytesReadable {
    fn from_le_bytes(bytes: [u8; 4]) -> Self;
}

impl LeBytesReadable for u32 {
    fn from_le_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}

impl LeBytesReadable for f32 {
    fn from_le_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

/// Read one complete record (header + vector). Returns `None` on clean EOF.
fn read_one_record(r: &mut impl Read, variant: Variant) -> Result<Option<Vec<f32>>> {
    // Try reading the 4-byte dimension header.
    let mut dim_buf = [0u8; 4];
    match r.read_exact(&mut dim_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let dim = u32::from_le_bytes(dim_buf) as usize;

    let vector = match variant {
        Variant::Fvecs => {
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(read_le::<f32>(r)?);
            }
            vec
        }
        Variant::Bvecs => {
            let mut bytes = vec![0u8; dim];
            r.read_exact(&mut bytes)
                .context("failed to read bvecs payload")?;
            bytes.into_iter().map(f32::from).collect()
        }
    };

    Ok(Some(vector))
}
