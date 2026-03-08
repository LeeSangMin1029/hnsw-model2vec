//! `dupes` command — find duplicate code chunks.
//!
//! Three modes:
//! - **Token Jaccard** (default): MinHash fingerprint-based near-duplicate detection.
//!   Compares actual code body tokens (unigrams + bigrams). Catches Type-1~3 clones.
//! - **AST hash** (`--ast`): structural clones ignoring identifier names (Type-1/2).
//! - **Embedding** (`--embed`): all-pairs cosine similarity (Type-3/4, slower).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;
use v_hnsw_core::{PayloadStore, PayloadValue, VectorStore};
use v_hnsw_graph::distance::dot_product;
use v_hnsw_storage::StorageEngine;

use crate::chunk_code::extract;

/// Run the dupes command.
pub fn run(
    db: std::path::PathBuf,
    threshold: f32,
    exclude_tests: bool,
    k: usize,
    json: bool,
    embed_mode: bool,
    ast_mode: bool,
    min_lines: usize,
) -> Result<()> {
    let engine = StorageEngine::open(&db)
        .with_context(|| format!("failed to open database at {}", db.display()))?;

    let pstore = engine.payload_store();

    if embed_mode {
        run_embedding(&engine, pstore, threshold, exclude_tests, k, json, &db, min_lines)
    } else if ast_mode {
        run_hash_groups(&engine, pstore, exclude_tests, k, json, &db, "ast_hash", min_lines)
    } else {
        run_minhash(&engine, pstore, threshold, exclude_tests, k, json, &db, min_lines)
    }
}

// ---------------------------------------------------------------------------
// MinHash Jaccard mode — token-based near-duplicate detection (default)
// ---------------------------------------------------------------------------

fn run_minhash(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    threshold: f32,
    exclude_tests: bool,
    k: usize,
    json: bool,
    db: &Path,
    min_lines: usize,
) -> Result<()> {
    let vstore = engine.vector_store();
    let ids: Vec<u64> = vstore.id_map().keys().copied().collect();

    // Load MinHash signatures
    let mut entries: Vec<(u64, Vec<u64>)> = Vec::new();
    let mut no_minhash = 0u32;

    for &id in &ids {
        if exclude_tests && is_test_chunk(pstore, id) {
            continue;
        }
        if min_lines > 0 && chunk_lines(pstore, id) < min_lines {
            continue;
        }
        if let Some(sig) = get_minhash(pstore, id) {
            entries.push((id, sig));
        } else {
            no_minhash += 1;
        }
    }

    if no_minhash > 0 {
        eprintln!("{no_minhash} chunks without minhash (re-index with `v-hnsw add`).");
    }

    let n = entries.len();
    if n < 2 {
        println!("Not enough chunks with minhash ({n}).");
        return Ok(());
    }

    let threshold_f64 = f64::from(threshold);
    eprintln!("Comparing {n} chunks (Jaccard threshold={threshold:.2})...");

    // All-pairs Jaccard estimation via MinHash (parallelized)
    let refs = &entries;
    let mut pairs: Vec<DupePair> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (id_a, sig_a) = &refs[i];
            ((i + 1)..n).filter_map(move |j| {
                let (id_b, sig_b) = &refs[j];
                let sim = extract::jaccard_from_minhash(sig_a, sig_b);
                #[expect(clippy::cast_possible_truncation)]
                (sim >= threshold_f64).then_some(DupePair {
                    id_a: *id_a,
                    id_b: *id_b,
                    similarity: sim as f32,
                })
            })
        })
        .collect();

    pairs.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(k);

    if json {
        print_pairs_json(&pairs, pstore);
    } else {
        print_pairs_text(&pairs, pstore, db);
    }

    Ok(())
}

fn get_minhash(pstore: &impl PayloadStore, id: u64) -> Option<Vec<u64>> {
    let payload = pstore.get_payload(id).ok()??;
    match payload.custom.get("minhash")? {
        PayloadValue::String(hex) => extract::minhash_from_hex(hex),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Hash-based mode — groups by ast_hash
// ---------------------------------------------------------------------------

fn run_hash_groups(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    exclude_tests: bool,
    k: usize,
    json: bool,
    db: &Path,
    hash_key: &str,
    min_lines: usize,
) -> Result<()> {
    let vstore = engine.vector_store();
    let ids: Vec<u64> = vstore.id_map().keys().copied().collect();

    let mut hash_groups: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut no_hash_count = 0u32;

    for &id in &ids {
        if exclude_tests && is_test_chunk(pstore, id) {
            continue;
        }
        if min_lines > 0 && chunk_lines(pstore, id) < min_lines {
            continue;
        }
        if let Some(hash) = get_hash(pstore, id, hash_key) {
            hash_groups.entry(hash).or_default().push(id);
        } else {
            no_hash_count += 1;
        }
    }

    let mut clone_groups: Vec<(u64, Vec<u64>)> = hash_groups
        .into_iter()
        .filter(|(_, ids)| ids.len() > 1)
        .collect();

    clone_groups.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()));
    clone_groups.truncate(k);

    if no_hash_count > 0 {
        eprintln!("{no_hash_count} chunks without {hash_key} (re-index with `v-hnsw add`).");
    }

    if json {
        print_groups_json(&clone_groups, pstore);
    } else {
        print_groups_text(&clone_groups, pstore, db);
    }

    Ok(())
}

fn get_hash(pstore: &impl PayloadStore, id: u64, key: &str) -> Option<u64> {
    let payload = pstore.get_payload(id).ok()??;
    match payload.custom.get(key)? {
        #[expect(clippy::cast_sign_loss, reason = "hash bits reinterpreted")]
        PayloadValue::Integer(v) => Some(*v as u64),
        _ => None,
    }
}

fn print_groups_text(groups: &[(u64, Vec<u64>)], pstore: &impl PayloadStore, db: &Path) {
    if groups.is_empty() {
        println!("No clones found.");
        return;
    }
    println!("{} clone groups found:\n", groups.len());
    for (i, (hash, ids)) in groups.iter().enumerate() {
        println!(
            "  Group {} ({} clones, hash={hash:016x}):",
            i + 1,
            ids.len()
        );
        for id in ids {
            println!("    - {}", label(pstore, *id));
        }
        println!();
    }
    let db_name = db.file_name().and_then(|n| n.to_str()).unwrap_or("db");
    eprintln!("Tip: v-hnsw gather {db_name} <symbol> --depth 1  to inspect.");
}

fn print_groups_json(groups: &[(u64, Vec<u64>)], pstore: &impl PayloadStore) {
    print!("{{\"groups\":[");
    for (i, (hash, ids)) in groups.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("{{\"hash\":\"{hash:016x}\",\"members\":[");
        for (j, id) in ids.iter().enumerate() {
            if j > 0 {
                print!(",");
            }
            let l = label(pstore, *id).replace('"', "\\\"");
            print!("\"{l}\"");
        }
        print!("]}}");
    }
    println!("]}}");
}

// ---------------------------------------------------------------------------
// Embedding mode — semantic similarity (slower, catches Type-3/4)
// ---------------------------------------------------------------------------

struct DupePair {
    id_a: u64,
    id_b: u64,
    similarity: f32,
}

fn run_embedding(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    threshold: f32,
    exclude_tests: bool,
    k: usize,
    json: bool,
    db: &Path,
    min_lines: usize,
) -> Result<()> {
    let vstore = engine.vector_store();

    let mut vectors: Vec<(u64, &[f32])> = vstore
        .id_map()
        .keys()
        .filter_map(|&id| vstore.get(id).ok().map(|v| (id, v)))
        .collect();

    if exclude_tests {
        vectors.retain(|(id, _)| !is_test_chunk(pstore, *id));
    }
    if min_lines > 0 {
        vectors.retain(|(id, _)| chunk_lines(pstore, *id) >= min_lines);
    }

    let n = vectors.len();
    if n < 2 {
        println!("Not enough vectors for comparison ({n}).");
        return Ok(());
    }

    eprintln!("Comparing {n} vectors (threshold={threshold:.2})...");

    let vecs = &vectors;
    let mut pairs: Vec<DupePair> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (id_a, vec_a) = vecs[i];
            ((i + 1)..n).filter_map(move |j| {
                let (id_b, vec_b) = vecs[j];
                let sim = dot_product(vec_a, vec_b);
                (sim >= threshold).then_some(DupePair {
                    id_a,
                    id_b,
                    similarity: sim,
                })
            })
        })
        .collect();

    pairs.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(k);

    if json {
        print_pairs_json(&pairs, pstore);
    } else {
        print_pairs_text(&pairs, pstore, db);
    }

    Ok(())
}

fn print_pairs_text(pairs: &[DupePair], pstore: &impl PayloadStore, db: &Path) {
    if pairs.is_empty() {
        println!("No duplicates found above threshold.");
        return;
    }
    println!("{} duplicate pairs found:\n", pairs.len());
    for (i, p) in pairs.iter().enumerate() {
        println!(
            "  {: >3}. [{:.2}]  {} ↔ {}",
            i + 1,
            p.similarity,
            label(pstore, p.id_a),
            label(pstore, p.id_b),
        );
    }
    let db_name = db.file_name().and_then(|n| n.to_str()).unwrap_or("db");
    eprintln!("\nTip: v-hnsw gather {db_name} <symbol> --depth 1  to inspect.");
}

fn print_pairs_json(pairs: &[DupePair], pstore: &impl PayloadStore) {
    print!("{{\"pairs\":[");
    for (i, p) in pairs.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        let la = label(pstore, p.id_a).replace('"', "\\\"");
        let lb = label(pstore, p.id_b).replace('"', "\\\"");
        print!(
            "{{\"a\":\"{la}\",\"b\":\"{lb}\",\"sim\":{:.4}}}",
            p.similarity
        );
    }
    println!("]}}");
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn is_test_chunk(pstore: &impl PayloadStore, id: u64) -> bool {
    let Some(payload) = pstore.get_payload(id).ok().flatten() else {
        return false;
    };
    let src = &payload.source;
    if src.contains("/tests/")
        || src.contains("\\tests\\")
        || src.contains("/test/")
        || src.contains("\\test\\")
        || src.ends_with("_test.rs")
        || src.ends_with("_test.go")
    {
        return true;
    }
    if let Ok(Some(text)) = pstore.get_text(id) {
        let first_line = text.lines().next().unwrap_or("");
        if first_line.contains("test_") || first_line.contains("Test") {
            return true;
        }
    }
    false
}

/// Get the number of lines in a chunk from its custom fields.
fn chunk_lines(pstore: &impl PayloadStore, id: u64) -> usize {
    let Some(payload) = pstore.get_payload(id).ok().flatten() else {
        return 0;
    };
    let start = match payload.custom.get("start_line") {
        Some(PayloadValue::Integer(v)) => *v,
        _ => return 0,
    };
    let end = match payload.custom.get("end_line") {
        Some(PayloadValue::Integer(v)) => *v,
        _ => return 0,
    };
    #[expect(clippy::cast_sign_loss)]
    let lines = (end - start + 1) as usize;
    lines
}

fn label(pstore: &impl PayloadStore, id: u64) -> String {
    let Some(text) = pstore.get_text(id).ok().flatten() else {
        return format!("id:{id}");
    };
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let file = lines
        .next()
        .unwrap_or("")
        .strip_prefix("File: ")
        .unwrap_or("");
    let name = first
        .strip_prefix("[function] ")
        .or_else(|| first.strip_prefix("[impl] "))
        .or_else(|| first.strip_prefix("[struct] "))
        .unwrap_or(first);
    if file.is_empty() {
        name.to_string()
    } else {
        format!("{name}  ({file})")
    }
}
