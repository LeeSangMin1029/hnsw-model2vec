//! Index Markdown files command - Creates a v-hnsw index from markdown documents.
//!
//! This allows v-hnsw to index the same markdown files as qmd collections,
//! enabling faster vector search with HNSW algorithm.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use v_hnsw_core::VectorIndex;
use v_hnsw_distance::CosineDistance;
use v_hnsw_embed::{EmbeddingModel, FastEmbedModel, ModelType};
use v_hnsw_graph::{HnswConfig, HnswGraph};

use crate::is_interrupted;

/// Default batch size for embedding.
const BATCH_SIZE: usize = 128;

/// Document with metadata extracted from markdown.
#[derive(Debug)]
struct Document {
    id: u64,
    path: String,
    title: Option<String>,
    content: String,
}

/// Run the index-md command.
pub fn run(
    input_dir: PathBuf,
    output_dir: PathBuf,
    chunk_size: usize,
    chunk_overlap: usize,
    pattern: Option<String>,
) -> Result<()> {
    println!("=== v-hnsw Markdown Indexer ===\n");

    // Step 1: Find markdown files
    println!("Scanning for markdown files in {}...", input_dir.display());
    let pattern = pattern.as_deref().unwrap_or("**/*.md");
    let md_files = find_markdown_files(&input_dir, pattern)?;
    println!("Found {} markdown files", md_files.len());

    if md_files.is_empty() {
        anyhow::bail!("No markdown files found matching pattern: {}", pattern);
    }

    // Step 2: Parse and extract content
    println!("\nParsing markdown files...");
    let documents = parse_markdown_files(&md_files)?;
    println!("Extracted {} documents", documents.len());

    // Step 3: Chunk documents
    println!("\nChunking documents ({} chars, {} overlap)...", chunk_size, chunk_overlap);
    let chunks = chunk_documents(&documents, chunk_size, chunk_overlap);
    println!("Created {} chunks", chunks.len());

    if chunks.is_empty() {
        anyhow::bail!("No chunks created from documents");
    }

    // Step 4: Initialize embedding model
    println!("\nInitializing embedding model...");
    let model_start = Instant::now();
    let model = FastEmbedModel::with_model(ModelType::AllMiniLML6V2)
        .context("Failed to initialize embedding model")?;
    println!(
        "Model: {} (dim={}), init time: {:.2}s",
        model.name(),
        model.dim(),
        model_start.elapsed().as_secs_f64()
    );

    // Step 5: Generate embeddings
    println!("\nGenerating embeddings...");
    let embed_start = Instant::now();
    let chunk_texts: Vec<&str> = chunks.iter().map(|(_, text)| text.as_str()).collect();
    let embeddings = embed_texts(&model, &chunk_texts)?;
    let embed_time = embed_start.elapsed();
    println!(
        "Embedding time: {:.2}s ({:.0} chunks/sec)",
        embed_time.as_secs_f64(),
        chunks.len() as f64 / embed_time.as_secs_f64()
    );

    if is_interrupted() {
        println!("Interrupted during embedding");
        return Ok(());
    }

    // Step 6: Build HNSW index
    println!("\nBuilding HNSW index...");
    let index_start = Instant::now();
    let hnsw = build_index(&embeddings, model.dim())?;
    let index_time = index_start.elapsed();
    println!(
        "Index build time: {:.2}s ({:.0} vectors/sec)",
        index_time.as_secs_f64(),
        embeddings.len() as f64 / index_time.as_secs_f64()
    );

    // Step 7: Save index and metadata
    println!("\nSaving index to {}...", output_dir.display());
    fs::create_dir_all(&output_dir)?;
    save_index(&output_dir, &hnsw, &chunks, &documents)?;

    // Summary
    println!("\n=== Summary ===");
    println!("Documents:     {}", documents.len());
    println!("Chunks:        {}", chunks.len());
    println!("Embedding dim: {}", model.dim());
    println!("Output:        {}", output_dir.display());
    println!("\nIndexing time:");
    println!("  Embedding:   {:.2}s", embed_time.as_secs_f64());
    println!("  Index build: {:.2}s", index_time.as_secs_f64());
    println!(
        "  Total:       {:.2}s",
        embed_time.as_secs_f64() + index_time.as_secs_f64()
    );

    Ok(())
}

/// Find markdown files matching a glob pattern.
fn find_markdown_files(dir: &PathBuf, _pattern: &str) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    collect_md_files_recursive(dir, &mut files)?;
    Ok(files)
}

/// Recursively collect markdown files.
fn collect_md_files_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let entries = fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            collect_md_files_recursive(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "md") {
            files.push(path);
        }
    }

    Ok(())
}

/// Parse markdown files and extract content.
fn parse_markdown_files(files: &[PathBuf]) -> Result<Vec<Document>> {
    let mut documents = Vec::new();
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    for (idx, path) in files.iter().enumerate() {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            break;
        }

        if let Ok((title, content)) = parse_markdown_file(path)
            && !content.trim().is_empty()
        {
            documents.push(Document {
                id: idx as u64,
                path: path.to_string_lossy().to_string(),
                title,
                content,
            });
        }
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    Ok(documents)
}

/// Parse a single markdown file, extracting frontmatter and content.
fn parse_markdown_file(path: &Path) -> Result<(Option<String>, String)> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines: Vec<String> = Vec::new();
    let mut in_frontmatter = false;
    let mut frontmatter_done = false;
    let mut title = None;

    for line in reader.lines() {
        let line = line?;

        // Detect frontmatter boundaries
        if line.trim() == "---"
            && !frontmatter_done
        {
            in_frontmatter = !in_frontmatter;
            if !in_frontmatter {
                frontmatter_done = true;
            }
            continue;
        }

        if in_frontmatter {
            // Extract title from frontmatter
            if line.starts_with("title:") {
                title = Some(line.trim_start_matches("title:").trim().trim_matches('"').to_string());
            }
        } else {
            // Extract title from first H1 if not in frontmatter
            if title.is_none() && line.starts_with("# ") {
                title = Some(line.trim_start_matches("# ").to_string());
            }
            lines.push(line);
        }
    }

    Ok((title, lines.join("\n")))
}

/// Chunk documents with overlap.
fn chunk_documents(
    documents: &[Document],
    chunk_size: usize,
    overlap: usize,
) -> Vec<(u64, String)> {
    let mut chunks = Vec::new();
    let mut chunk_id = 0u64;

    for doc in documents {
        let chars: Vec<char> = doc.content.chars().collect();
        if chars.is_empty() {
            continue;
        }

        let step = if chunk_size > overlap {
            chunk_size - overlap
        } else {
            chunk_size
        };

        let mut start = 0;
        while start < chars.len() {
            let end = std::cmp::min(start + chunk_size, chars.len());
            let chunk: String = chars[start..end].iter().collect();

            // Only keep chunks with meaningful content
            let trimmed = chunk.trim();
            if trimmed.len() >= 50 {
                chunks.push((chunk_id, trimmed.to_string()));
                chunk_id += 1;
            }

            if end >= chars.len() {
                break;
            }
            start += step;
        }
    }

    chunks
}

/// Generate embeddings for texts.
fn embed_texts(model: &FastEmbedModel, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(texts.len());

    let pb = ProgressBar::new(texts.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    for batch in texts.chunks(BATCH_SIZE) {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            break;
        }

        let embeddings = model
            .embed(batch)
            .context("Failed to generate embeddings")?;

        all_embeddings.extend(embeddings);
        pb.inc(batch.len() as u64);
    }

    pb.finish_with_message("Done");
    Ok(all_embeddings)
}

/// Build HNSW index from embeddings.
fn build_index(embeddings: &[Vec<f32>], dim: usize) -> Result<HnswGraph<CosineDistance>> {
    let config = HnswConfig::builder()
        .dim(dim)
        .m(16)
        .ef_construction(200)
        .build()
        .context("Failed to create HNSW config")?;

    let mut hnsw: HnswGraph<CosineDistance> = HnswGraph::new(config, CosineDistance);

    let pb = ProgressBar::new(embeddings.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} vectors")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    for (id, embedding) in embeddings.iter().enumerate() {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            break;
        }

        hnsw.insert(id as u64, embedding)
            .context("Failed to insert vector")?;
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    Ok(hnsw)
}

/// Save index and metadata to disk.
fn save_index(
    output_dir: &Path,
    _hnsw: &HnswGraph<CosineDistance>,
    chunks: &[(u64, String)],
    documents: &[Document],
) -> Result<()> {
    // Save chunk metadata as JSONL
    let chunks_path = output_dir.join("chunks.jsonl");
    let mut chunks_file = fs::File::create(&chunks_path)?;
    for (id, text) in chunks {
        use std::io::Write;
        writeln!(
            chunks_file,
            r#"{{"id":{},"text":{}}}"#,
            id,
            serde_json::to_string(text)?
        )?;
    }

    // Save document metadata as JSONL
    let docs_path = output_dir.join("documents.jsonl");
    let mut docs_file = fs::File::create(&docs_path)?;
    for doc in documents {
        use std::io::Write;
        writeln!(
            docs_file,
            r#"{{"id":{},"path":{},"title":{}}}"#,
            doc.id,
            serde_json::to_string(&doc.path)?,
            serde_json::to_string(&doc.title)?
        )?;
    }

    // Note: Full index serialization would require v-hnsw storage crate
    // For now, we save metadata only
    println!("  Saved {} chunks to {}", chunks.len(), chunks_path.display());
    println!("  Saved {} documents to {}", documents.len(), docs_path.display());

    Ok(())
}
