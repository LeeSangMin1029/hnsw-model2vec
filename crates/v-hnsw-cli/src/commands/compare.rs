//! Compare command - Benchmark v-hnsw vs qmd performance on real data.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;
use v_hnsw_core::VectorIndex;
use v_hnsw_embed::{Model2VecModel, EmbeddingModel};
use v_hnsw_graph::{CosineDistance, HnswConfig, HnswGraph};

use crate::is_interrupted;

/// Default batch size for embedding.
const BATCH_SIZE: usize = 128;

/// Minimal JSONL message structure from Claude sessions.
#[derive(Debug, Deserialize)]
struct SessionMessage {
    #[serde(rename = "type")]
    msg_type: Option<String>,
    role: Option<String>,
    message: Option<MessageContent>,
}

#[derive(Debug, Deserialize)]
struct MessageContent {
    content: Option<Vec<ContentItem>>,
}

#[derive(Debug, Deserialize)]
struct ContentItem {
    #[serde(rename = "type")]
    item_type: Option<String>,
    text: Option<String>,
}

/// Run the compare benchmark command.
pub fn run(
    jsonl_dir: PathBuf,
    queries: usize,
    k: usize,
    chunk_size: usize,
    max_chunks: Option<usize>,
) -> Result<()> {
    println!("=== v-hnsw Benchmark ===\n");

    // Step 1: Load and extract texts from JSONL files
    println!("Loading JSONL files from {}...", jsonl_dir.display());
    let texts = load_jsonl_texts(&jsonl_dir)?;
    println!("Extracted {} assistant messages", texts.len());

    if texts.is_empty() {
        anyhow::bail!("No text extracted from JSONL files");
    }

    // Step 2: Chunk texts
    println!("\nChunking texts ({} chars per chunk)...", chunk_size);
    let chunks: Vec<String> = texts
        .iter()
        .flat_map(|text| chunk_text(text, chunk_size))
        .collect();

    let chunks = if let Some(max) = max_chunks {
        if chunks.len() > max {
            println!("Limiting to {} chunks (from {})", max, chunks.len());
            chunks.into_iter().take(max).collect()
        } else {
            chunks
        }
    } else {
        chunks
    };

    println!("Created {} chunks", chunks.len());

    if chunks.is_empty() {
        anyhow::bail!("No chunks created from texts");
    }

    // Step 3: Initialize embedding model
    println!("\nInitializing embedding model...");
    let model_start = Instant::now();
    let model = Model2VecModel::new()
        .context("Failed to initialize embedding model")?;
    let model_init_time = model_start.elapsed();
    println!(
        "Model: {} (dim={})",
        model.name(),
        model.dim()
    );
    println!("Model init time: {:.2}s", model_init_time.as_secs_f64());

    // Step 4: Generate embeddings
    println!("\nGenerating embeddings...");
    let embed_start = Instant::now();
    let embeddings = embed_chunks(&model, &chunks)?;
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

    // Step 5: Build HNSW index
    println!("\nBuilding HNSW index...");
    let index_start = Instant::now();
    let hnsw = build_hnsw_index(&embeddings, model.dim())?;
    let index_time = index_start.elapsed();
    println!(
        "Index build time: {:.2}s ({:.0} vectors/sec)",
        index_time.as_secs_f64(),
        embeddings.len() as f64 / index_time.as_secs_f64()
    );

    if is_interrupted() {
        println!("Interrupted during index build");
        return Ok(());
    }

    // Step 6: Run search benchmark
    println!("\nRunning search benchmark ({} queries, k={})...", queries, k);
    let query_embeddings = select_query_embeddings(&embeddings, queries);

    run_search_benchmark(&hnsw, &query_embeddings, k)?;

    // Summary
    println!("\n=== Summary ===");
    println!("Documents:     {}", texts.len());
    println!("Chunks:        {}", chunks.len());
    println!("Embedding dim: {}", model.dim());
    println!("\nIndexing:");
    println!("  Embedding time:   {:.2}s", embed_time.as_secs_f64());
    println!("  Index build time: {:.2}s", index_time.as_secs_f64());
    println!(
        "  Total:            {:.2}s",
        embed_time.as_secs_f64() + index_time.as_secs_f64()
    );

    Ok(())
}

/// Load text content from JSONL files in a directory (recursively).
fn load_jsonl_texts(dir: &PathBuf) -> Result<Vec<String>> {
    let mut texts = Vec::new();

    // Find all .jsonl files recursively
    let mut jsonl_files = Vec::new();
    collect_jsonl_files_recursive(dir, &mut jsonl_files)?;

    if jsonl_files.is_empty() {
        anyhow::bail!("No .jsonl files found in {}", dir.display());
    }

    println!("Found {} JSONL files (recursive)", jsonl_files.len());

    let pb = ProgressBar::new(jsonl_files.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    for path in jsonl_files {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            return Ok(texts);
        }

        if let Ok(file_texts) = extract_texts_from_jsonl(&path) {
            texts.extend(file_texts);
        }
        pb.inc(1);
    }

    pb.finish_with_message("Done");
    Ok(texts)
}

/// Recursively collect all .jsonl files from a directory.
fn collect_jsonl_files_recursive(dir: &PathBuf, files: &mut Vec<PathBuf>) -> Result<()> {
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory: {}", dir.display()))?;

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            // Recurse into subdirectories
            collect_jsonl_files_recursive(&path, files)?;
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            files.push(path);
        }
    }

    Ok(())
}

/// Extract assistant message texts from a single JSONL file.
fn extract_texts_from_jsonl(path: &PathBuf) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut texts = Vec::new();

    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        if line.trim().is_empty() {
            continue;
        }

        // Try to parse as SessionMessage
        if let Ok(msg) = serde_json::from_str::<SessionMessage>(&line) {
            // Check if this is an assistant message
            let is_assistant = msg.role.as_deref() == Some("assistant")
                || msg.msg_type.as_deref() == Some("assistant");

            if is_assistant
                && let Some(message) = msg.message
                && let Some(content) = message.content
            {
                for item in content {
                    if item.item_type.as_deref() == Some("text")
                        && let Some(text) = item.text
                        && !text.trim().is_empty()
                    {
                        texts.push(text);
                    }
                }
            }
        }
    }

    Ok(texts)
}

/// Chunk text into smaller pieces.
fn chunk_text(text: &str, chunk_size: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<char> = text.chars().collect();

    if chars.is_empty() {
        return chunks;
    }

    let mut start = 0;
    while start < chars.len() {
        let end = std::cmp::min(start + chunk_size, chars.len());
        let chunk: String = chars[start..end].iter().collect();

        // Only keep chunks with meaningful content
        let trimmed = chunk.trim();
        if trimmed.len() >= 50 {
            chunks.push(trimmed.to_string());
        }

        start = end;
    }

    chunks
}

/// Generate embeddings for all chunks.
fn embed_chunks(model: &Model2VecModel, chunks: &[String]) -> Result<Vec<Vec<f32>>> {
    let mut all_embeddings = Vec::with_capacity(chunks.len());

    let pb = ProgressBar::new(chunks.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
            .ok()
            .unwrap_or_else(ProgressStyle::default_bar)
            .progress_chars("#>-"),
    );

    // Process in batches
    for batch in chunks.chunks(BATCH_SIZE) {
        if is_interrupted() {
            pb.finish_with_message("Interrupted");
            break;
        }

        let batch_refs: Vec<&str> = batch.iter().map(String::as_str).collect();
        let embeddings = model
            .embed(&batch_refs)
            .context("Failed to generate embeddings")?;

        all_embeddings.extend(embeddings);
        pb.inc(batch.len() as u64);
    }

    pb.finish_with_message("Done");
    Ok(all_embeddings)
}

/// Build HNSW index from embeddings.
fn build_hnsw_index(embeddings: &[Vec<f32>], dim: usize) -> Result<HnswGraph<CosineDistance>> {
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

/// Select query embeddings from the dataset.
fn select_query_embeddings(embeddings: &[Vec<f32>], count: usize) -> Vec<Vec<f32>> {
    let step = if embeddings.len() > count {
        embeddings.len() / count
    } else {
        1
    };

    let mut queries = Vec::with_capacity(count);
    let mut idx = 0;

    while queries.len() < count && idx < embeddings.len() {
        queries.push(embeddings[idx].clone());
        idx += step;
    }

    // If we need more, repeat from start
    while queries.len() < count {
        queries.push(embeddings[queries.len() % embeddings.len()].clone());
    }

    queries
}

/// Run search benchmark with different ef values.
fn run_search_benchmark(
    hnsw: &HnswGraph<CosineDistance>,
    query_embeddings: &[Vec<f32>],
    k: usize,
) -> Result<()> {
    let ef_values = [50, 100, 200, 400];

    for ef in ef_values {
        if is_interrupted() {
            println!("Interrupted");
            return Ok(());
        }

        let start = Instant::now();
        let mut total_results = 0usize;

        for query in query_embeddings {
            if is_interrupted() {
                println!("Interrupted");
                return Ok(());
            }

            if let Ok(results) = hnsw.search(query, k, ef) {
                total_results += results.len();
            }
        }

        let elapsed = start.elapsed();
        let num_queries = query_embeddings.len();
        let qps = num_queries as f64 / elapsed.as_secs_f64();
        let avg_latency_us = elapsed.as_micros() as f64 / num_queries as f64;
        let avg_latency_ms = avg_latency_us / 1000.0;

        println!(
            "  ef={:3}: {:8.1} QPS, {:6.2}ms avg, {:.1} avg results",
            ef,
            qps,
            avg_latency_ms,
            total_results as f64 / num_queries as f64
        );
    }

    // Pure search timing (ef=200, warm cache)
    println!("\nPure search (ef=200, warmed):");
    let warmup_queries = std::cmp::min(10, query_embeddings.len());
    for query in query_embeddings.iter().take(warmup_queries) {
        let _ = hnsw.search(query, k, 200);
    }

    let start = Instant::now();
    for query in query_embeddings {
        let _ = hnsw.search(query, k, 200);
    }
    let elapsed = start.elapsed();
    let num_queries = query_embeddings.len();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let avg_latency_ms = total_ms / num_queries as f64;
    let qps = num_queries as f64 / elapsed.as_secs_f64();

    println!("  Total time:   {:.2}ms", total_ms);
    println!("  Avg latency:  {:.3}ms", avg_latency_ms);
    println!("  QPS:          {:.0}", qps);

    Ok(())
}
