//! Add command - Unified data ingestion with automatic embedding.
//!
//! Detects input type (folder with .md files, .jsonl, .parquet) and processes accordingly.
//! Auto-creates database, embeds text, builds indexes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_chunk::{ChunkConfig, MarkdownChunker};
use v_hnsw_core::{Payload, PayloadStore, PayloadValue, VectorIndex, VectorStore};
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::{EmbeddingModel, Model2VecModel};
use v_hnsw_graph::{HnswConfig, HnswGraph};
use v_hnsw_search::{Bm25Index, KoreanBm25Tokenizer};
use v_hnsw_storage::{StorageConfig, StorageEngine};

use super::create::DbConfig;
use super::file_index;
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

/// Detect input type from path.
fn detect_input_type(path: &Path) -> Result<InputType> {
    if path.is_dir() {
        // Check if folder contains markdown files
        let has_md = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .any(|e| {
                e.path().extension()
                    .map(|ext| ext == "md" || ext == "markdown")
                    .unwrap_or(false)
            });
        if has_md {
            Ok(InputType::MarkdownFolder)
        } else {
            anyhow::bail!("Directory contains no markdown files: {}", path.display());
        }
    } else if path.is_file() {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase());
        match ext.as_deref() {
            Some("jsonl") | Some("ndjson") => Ok(InputType::Jsonl),
            Some("parquet") => Ok(InputType::Parquet),
            Some("md") | Some("markdown") => {
                // Single markdown file - treat as folder with one file
                anyhow::bail!("Single markdown file not supported. Use a folder containing .md files.")
            }
            _ => anyhow::bail!("Unsupported file type: {}. Supported: folder with .md files, .jsonl, .parquet", path.display()),
        }
    } else {
        anyhow::bail!("Path not found: {}", path.display());
    }
}

/// Default model2vec model ID.
pub const DEFAULT_MODEL: &str = "minishlab/potion-multilingual-128M";

/// Create embedding model (model2vec).
pub fn create_model() -> Result<Model2VecModel> {
    println!("Loading model2vec model: {}", DEFAULT_MODEL);

    let model = Model2VecModel::from_pretrained(DEFAULT_MODEL)
        .context("Failed to load model2vec model")?;

    println!("Model loaded (dim={}).", model.dim());
    Ok(model)
}

/// Auto-create database if it doesn't exist.
fn ensure_database(path: &Path, dim: usize, model_name: &str) -> Result<StorageEngine> {
    if path.exists() {
        let config = DbConfig::load(path)?;
        if config.dim != dim {
            anyhow::bail!(
                "Dimension mismatch: database has dim={}, but model produces dim={}",
                config.dim, dim
            );
        }
        // Update embed_model if not set
        if config.embed_model.is_none() {
            let mut config = config;
            config.embed_model = Some(model_name.to_string());
            config.save(path)?;
        }
        StorageEngine::open(path)
            .with_context(|| format!("Failed to open database at {}", path.display()))
    } else {
        println!("Creating new database at {}", path.display());

        let storage_config = StorageConfig {
            dim,
            initial_capacity: 10_000,
            checkpoint_threshold: 50_000,
        };

        let engine = StorageEngine::create(path, storage_config)
            .with_context(|| format!("Failed to create storage at {}", path.display()))?;

        let db_config = DbConfig {
            version: DbConfig::CURRENT_VERSION,
            dim,
            metric: "cosine".to_string(),
            m: 16,
            ef_construction: 200,
            korean: true,
            embed_model: Some(model_name.to_string()),
        };
        db_config.save(path)?;

        println!("  Dimension:  {dim}");
        println!("  Metric:     cosine");
        println!("  M:          16");
        println!("  ef:         200");
        println!("  Model:      {model_name}");
        println!();

        Ok(engine)
    }
}

/// Build progress bar with standard template.
pub fn make_progress_bar(total: u64) -> Result<ProgressBar> {
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec})")
            .map_err(|e| anyhow::anyhow!("Invalid progress template: {e}"))?
            .progress_chars("#>-"),
    );
    Ok(pb)
}

/// Build payload from source info.
pub fn make_payload(source: &str, title: Option<&str>, tags: &[String], chunk_index: usize, chunk_total: usize) -> Payload {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let mut custom = HashMap::new();
    if let Some(t) = title {
        custom.insert("title".to_string(), PayloadValue::String(t.to_string()));
    }

    Payload {
        source: source.to_string(),
        tags: tags.to_vec(),
        created_at: now,
        source_modified_at: now,
        chunk_index: chunk_index as u32,
        chunk_total: chunk_total as u32,
        custom,
    }
}

/// Generate a stable ID from source path and chunk index.
pub fn generate_id(source: &str, chunk_index: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    chunk_index.hash(&mut hasher);
    hasher.finish()
}

/// Embed texts with length-sorted batching to minimize padding waste.
pub fn embed_sorted(model: &dyn EmbeddingModel, texts: &[String]) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }

    // Sort indices by text length
    let mut indices: Vec<usize> = (0..texts.len()).collect();
    indices.sort_by_key(|&i| texts[i].len());

    let sorted: Vec<&str> = indices.iter().map(|&i| texts[i].as_str()).collect();
    let sorted_embs = model
        .embed(&sorted)
        .map_err(|e| anyhow::anyhow!("Embedding failed: {e}"))?;

    // Restore original order
    let mut embeddings = vec![Vec::new(); texts.len()];
    for (sorted_idx, &orig_idx) in indices.iter().enumerate() {
        embeddings[orig_idx] = sorted_embs[sorted_idx].clone();
    }
    Ok(embeddings)
}

/// A record for batch processing.
#[derive(Clone)]
struct AddRecord {
    id: u64,
    text: String,
    source: String,
    title: Option<String>,
    tags: Vec<String>,
    chunk_index: usize,
    chunk_total: usize,
}

/// Process markdown folder.
fn process_markdown_folder(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, u64)> {
    let chunker = MarkdownChunker::new(ChunkConfig {
        target_size: 1000,
        overlap: 200,
        min_size: 100,
        include_heading_context: true,
    });

    // Collect all markdown files
    let md_files: Vec<PathBuf> = walkdir::WalkDir::new(input_path)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension()
                .map(|ext| ext == "md" || ext == "markdown")
                .unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    if md_files.is_empty() {
        anyhow::bail!("No markdown files found in {}", input_path.display());
    }

    println!("Found {} markdown files", md_files.len());

    // First pass: collect all chunks and track file metadata
    let mut records: Vec<AddRecord> = Vec::new();
    let mut file_metadata_map: HashMap<String, (u64, u64, Vec<u64>)> = HashMap::new(); // path -> (mtime, size, chunk_ids)

    for md_path in &md_files {
        if is_interrupted() {
            break;
        }

        let (frontmatter, chunks) = match chunker.chunk_file(md_path) {
            Ok(result) => result,
            Err(e) => {
                eprintln!("Error reading {}: {e}", md_path.display());
                continue;
            }
        };

        let source = md_path.to_string_lossy().to_string();
        let title = frontmatter.as_ref().and_then(|f| f.title.clone());
        let tags = frontmatter.as_ref().map(|f| f.tags.clone()).unwrap_or_default();
        let chunk_total = chunks.len();

        // Get file metadata
        let mtime = file_index::get_file_mtime(md_path).unwrap_or(0);
        let size = file_index::get_file_size(md_path).unwrap_or(0);
        let mut chunk_ids = Vec::new();

        for chunk in chunks {
            let id = generate_id(&source, chunk.chunk_index);
            chunk_ids.push(id);
            records.push(AddRecord {
                id,
                text: chunk.text,
                source: source.clone(),
                title: title.clone(),
                tags: tags.clone(),
                chunk_index: chunk.chunk_index,
                chunk_total,
            });
        }

        // Store metadata for this file
        file_metadata_map.insert(source, (mtime, size, chunk_ids));
    }

    println!("Total chunks to process: {}", records.len());

    // Process in batches
    let result = process_records(db_path, records, model, engine)?;

    // Save file metadata index
    let mut file_index = file_index::load_file_index(db_path)?;
    for (path, (mtime, size, chunk_ids)) in file_metadata_map {
        file_index.update_file(path, mtime, size, chunk_ids);
    }
    file_index::save_file_index(db_path, &file_index)?;

    Ok(result)
}

/// Process JSONL file.
fn process_jsonl(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, u64)> {
    use std::io::{BufRead, BufReader};

    let file = std::fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;
    let reader = BufReader::new(file);

    let mut records: Vec<AddRecord> = Vec::new();
    let source = input_path.to_string_lossy().to_string();

    for (line_num, line_result) in reader.lines().enumerate() {
        if is_interrupted() {
            break;
        }

        let line = match line_result {
            Ok(l) => l,
            Err(e) => {
                eprintln!("Line {}: read error: {e}", line_num + 1);
                continue;
            }
        };

        if line.trim().is_empty() {
            continue;
        }

        let json: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Line {}: parse error: {e}", line_num + 1);
                continue;
            }
        };

        // Extract text field
        let text = json.get("text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        if text.is_none() || text.as_ref().map(|t| t.is_empty()).unwrap_or(true) {
            continue;
        }

        let text = text.unwrap();

        // Extract optional fields
        let id = json.get("id")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| generate_id(&source, line_num));

        let title = json.get("title")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let tags = json.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();

        let item_source = json.get("source")
            .and_then(|v| v.as_str())
            .unwrap_or(&source)
            .to_string();

        records.push(AddRecord {
            id,
            text,
            source: item_source,
            title,
            tags,
            chunk_index: 0,
            chunk_total: 1,
        });
    }

    println!("Total records to process: {}", records.len());

    process_records(db_path, records, model, engine)
}

/// Process Parquet file.
fn process_parquet(
    db_path: &Path,
    input_path: &Path,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, u64)> {
    use arrow::array::{Array, StringArray, UInt64Array};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(input_path)
        .with_context(|| format!("Failed to open {}", input_path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| "Failed to create Parquet reader")?;

    let reader = builder.build()
        .with_context(|| "Failed to build Parquet reader")?;

    let mut records: Vec<AddRecord> = Vec::new();
    let source = input_path.to_string_lossy().to_string();
    let mut row_idx = 0u64;

    for batch_result in reader {
        if is_interrupted() {
            break;
        }

        let batch = batch_result.with_context(|| "Failed to read Parquet batch")?;
        let schema = batch.schema();

        // Find text column
        let text_col_idx = schema.fields()
            .iter()
            .position(|f| f.name() == "text" || f.name() == "content")
            .ok_or_else(|| anyhow::anyhow!("No 'text' or 'content' column found in Parquet file"))?;

        let text_array = batch.column(text_col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow::anyhow!("Text column is not a string array"))?;

        // Optional id column
        let id_col_idx = schema.fields().iter().position(|f| f.name() == "id");
        let id_array = id_col_idx.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<UInt64Array>()
        }).flatten();

        // Optional title column
        let title_col_idx = schema.fields().iter().position(|f| f.name() == "title");
        let title_array = title_col_idx.map(|idx| {
            batch.column(idx).as_any().downcast_ref::<StringArray>()
        }).flatten();

        for i in 0..batch.num_rows() {
            if text_array.is_null(i) {
                row_idx += 1;
                continue;
            }

            let text = text_array.value(i).to_string();
            if text.is_empty() {
                row_idx += 1;
                continue;
            }

            let id = id_array
                .and_then(|arr| if arr.is_null(i) { None } else { Some(arr.value(i)) })
                .unwrap_or_else(|| generate_id(&source, row_idx as usize));

            let title = title_array
                .and_then(|arr| if arr.is_null(i) { None } else { Some(arr.value(i).to_string()) });

            records.push(AddRecord {
                id,
                text,
                source: source.clone(),
                title,
                tags: Vec::new(),
                chunk_index: 0,
                chunk_total: 1,
            });

            row_idx += 1;
        }
    }

    println!("Total records to process: {}", records.len());

    process_records(db_path, records, model, engine)
}

/// Embedded batch ready for storage insertion.
struct EmbeddedBatch {
    records: Vec<AddRecord>,
    embeddings: Vec<Vec<f32>>,
}

/// Process records in batches with embedding using pipeline parallelism.
/// Producer thread: prepare texts + embedding
/// Consumer thread: storage insertion
fn process_records(
    _db_path: &Path,
    records: Vec<AddRecord>,
    model: &Model2VecModel,
    engine: &mut StorageEngine,
) -> Result<(u64, u64, u64)> {
    if records.is_empty() {
        return Ok((0, 0, 0));
    }

    let batch_size = 256; // model2vec is CPU-based, can handle larger batches
    let max_chars = 8000; // model2vec uses simple tokenization

    let pb = make_progress_bar(records.len() as u64)?;
    let start = Instant::now();

    // Bounded channel: buffer batches so producer can stay ahead during storage I/O
    let (sender, receiver) = crossbeam::channel::bounded::<EmbeddedBatch>(4);

    // Pipeline: Producer embeds, Consumer inserts
    let result = std::thread::scope(|scope| {
        let pb_ref = &pb;

        // Producer thread: prepare texts + embed on GPU
        let producer = scope.spawn(move || -> anyhow::Result<u64> {
            let mut producer_errors = 0u64;

            for chunk in records.chunks(batch_size) {
                if is_interrupted() {
                    break;
                }

                // Prepare texts for embedding (truncate if needed)
                let texts: Vec<String> = chunk
                    .iter()
                    .map(|r| {
                        if r.text.len() > max_chars {
                            let mut end = max_chars;
                            while !r.text.is_char_boundary(end) {
                                end -= 1;
                            }
                            r.text[..end].to_string()
                        } else {
                            r.text.clone()
                        }
                    })
                    .collect();

                // Embed batch on GPU
                let embeddings = match embed_sorted(model, &texts) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("Embedding error: {e}");
                        producer_errors += chunk.len() as u64;
                        pb_ref.inc(chunk.len() as u64);
                        continue;
                    }
                };

                // Send to consumer
                let batch = EmbeddedBatch {
                    records: chunk.to_vec(),
                    embeddings,
                };

                if sender.send(batch).is_err() {
                    break; // Consumer dropped
                }
            }

            drop(sender); // Signal completion
            Ok(producer_errors)
        });

        // Consumer (this thread): receive batches + insert into storage
        let mut inserted = 0u64;
        let mut consumer_errors = 0u64;

        for batch in receiver {
            if is_interrupted() {
                break;
            }

            // Insert into storage
            let items: Vec<(u64, &[f32], Payload, &str)> = batch
                .records
                .iter()
                .zip(batch.embeddings.iter())
                .map(|(rec, emb)| {
                    let payload = make_payload(
                        &rec.source,
                        rec.title.as_deref(),
                        &rec.tags,
                        rec.chunk_index,
                        rec.chunk_total,
                    );
                    (rec.id, emb.as_slice(), payload, rec.text.as_str())
                })
                .collect();

            if let Err(e) = engine.insert_batch(&items) {
                eprintln!("Insert error: {e}");
                consumer_errors += batch.records.len() as u64;
            } else {
                inserted += batch.records.len() as u64;
            }

            pb_ref.inc(batch.records.len() as u64);
        }

        // Wait for producer and get its error count
        let producer_errors = producer.join().unwrap_or_else(|_| Ok(0)).unwrap_or(0);

        (inserted, producer_errors + consumer_errors)
    });

    let (inserted, errors) = result;

    if !is_interrupted() {
        pb.finish_with_message("Done");
    }

    engine.checkpoint()
        .with_context(|| "Failed to checkpoint database")?;

    let elapsed = start.elapsed();

    println!();
    println!("Add completed:");
    println!("  Inserted: {inserted}");
    println!("  Errors:   {errors}");
    println!("  Elapsed:  {:.2}s", elapsed.as_secs_f64());
    if inserted > 0 {
        println!("  Rate:     {:.0} items/s", inserted as f64 / elapsed.as_secs_f64());
    }

    Ok((inserted, 0, errors))
}

/// Build and save HNSW and BM25 indexes.
fn build_indexes(path: &Path, engine: &StorageEngine, config: &DbConfig) -> Result<()> {
    if engine.is_empty() {
        println!("No vectors to index, skipping index building.");
        return Ok(());
    }

    println!();
    println!("Building indexes...");

    // Build HNSW graph
    let hnsw_config = HnswConfig::builder()
        .dim(config.dim)
        .m(config.m)
        .ef_construction(config.ef_construction)
        .build()
        .with_context(|| "Failed to create HNSW config")?;

    let hnsw_path = path.join("hnsw.bin");
    let vector_store = engine.vector_store();

    println!("  Building HNSW graph (M={}, ef_construction={})...", config.m, config.ef_construction);

    let mut hnsw = HnswGraph::new(hnsw_config, CosineDistance);
    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            println!("  Interrupted during HNSW build");
            return Ok(());
        }
        if let Ok(vec) = vector_store.get(*id) {
            let _ = hnsw.insert(*id, vec);
        }
    }

    hnsw.save(&hnsw_path)
        .with_context(|| format!("Failed to save HNSW graph to {}", hnsw_path.display()))?;
    println!("  HNSW graph saved: {}", hnsw_path.display());

    // Build BM25 index
    println!("  Building BM25 index...");
    let bm25_path = path.join("bm25.bin");
    let mut bm25: Bm25Index<KoreanBm25Tokenizer> = Bm25Index::new(KoreanBm25Tokenizer::new());
    let payload_store = engine.payload_store();

    for id in vector_store.id_map().keys() {
        if is_interrupted() {
            println!("  Interrupted during BM25 build");
            return Ok(());
        }
        if let Ok(Some(text)) = payload_store.get_text(*id) {
            bm25.add_document(*id, &text);
        }
    }

    bm25.save(&bm25_path)
        .with_context(|| format!("Failed to save BM25 index to {}", bm25_path.display()))?;
    println!("  BM25 index saved: {}", bm25_path.display());

    println!("Index building completed.");
    Ok(())
}

/// Run the add command.
pub fn run(db_path: PathBuf, input_path: PathBuf) -> Result<()> {
    // Detect input type
    let input_type = detect_input_type(&input_path)?;

    println!("Input type: {:?}", input_type);
    println!("Input path: {}", input_path.display());
    println!("Database:   {}", db_path.display());
    println!();

    // Create model
    let model = create_model()?;
    let model_name = DEFAULT_MODEL;

    // Ensure database exists
    let mut engine = ensure_database(&db_path, model.dim(), model_name)?;

    // Process based on input type
    let (inserted, _skipped, errors) = match input_type {
        InputType::MarkdownFolder => process_markdown_folder(&db_path, &input_path, &model, &mut engine)?,
        InputType::Jsonl => process_jsonl(&db_path, &input_path, &model, &mut engine)?,
        InputType::Parquet => process_parquet(&db_path, &input_path, &model, &mut engine)?,
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
    build_indexes(&db_path, &engine, &config)?;

    println!();
    println!("Done! Database ready at: {}", db_path.display());

    Ok(())
}
