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

    // Filter overlapping ranges in the same file (parent/child chunks)
    pairs.retain(|p| !chunks_overlap(pstore, p.id_a, p.id_b));

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

    // Remove overlapping chunks within each group (keep the larger span)
    for ids in hash_groups.values_mut() {
        if ids.len() > 1 {
            let mut i = 0;
            while i < ids.len() {
                let mut j = i + 1;
                while j < ids.len() {
                    if chunks_overlap(pstore, ids[i], ids[j]) {
                        // Keep the one with more lines, drop the other
                        if chunk_lines(pstore, ids[i]) >= chunk_lines(pstore, ids[j]) {
                            ids.swap_remove(j);
                        } else {
                            ids.swap_remove(i);
                            j = i + 1; // restart inner loop for new ids[i]
                            continue;
                        }
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
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

    // Collect all labels and file paths
    let all_labels: Vec<Vec<(usize, u64, ChunkLabel)>> = groups
        .iter()
        .enumerate()
        .map(|(gi, (hash, ids))| {
            ids.iter()
                .map(|&id| (gi + 1, *hash, parse_label(pstore, id)))
                .collect()
        })
        .collect();

    // Gather all file paths for common-prefix stripping
    let mut all_files: Vec<String> = all_labels
        .iter()
        .flat_map(|g| g.iter().map(|(_, _, cl)| cl.file.clone()))
        .collect();
    strip_common_prefix(&mut all_files);

    // Re-assemble with stripped paths, grouped by file
    let mut by_file: Vec<(String, Vec<(usize, u64, String)>)> = Vec::new();
    let mut file_index: HashMap<String, usize> = HashMap::new();
    let mut fi = 0;

    for group in &all_labels {
        for (group_num, hash, cl) in group {
            let key = if all_files[fi].is_empty() {
                "(unknown)".to_owned()
            } else {
                all_files[fi].clone()
            };
            fi += 1;
            let idx = if let Some(&i) = file_index.get(&key) {
                i
            } else {
                let i = by_file.len();
                file_index.insert(key.clone(), i);
                by_file.push((key, Vec::new()));
                i
            };
            by_file[idx].1.push((*group_num, *hash, cl.name.clone()));
        }
    }

    by_file.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (file, entries) in &by_file {
        println!("  {} ({} clones)", file, entries.len());
        for (group_num, hash, name) in entries {
            println!("    G{group_num} [{hash:016x}]  {name}");
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

    // Filter overlapping ranges in the same file (parent/child chunks)
    pairs.retain(|p| !chunks_overlap(pstore, p.id_a, p.id_b));

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

    // Collect all labels
    let labels: Vec<(ChunkLabel, ChunkLabel)> = pairs
        .iter()
        .map(|p| (parse_label(pstore, p.id_a), parse_label(pstore, p.id_b)))
        .collect();

    // Gather all file paths for common-prefix stripping
    let mut all_files: Vec<String> = labels
        .iter()
        .flat_map(|(a, b)| [a.file.clone(), b.file.clone()])
        .collect();
    strip_common_prefix(&mut all_files);

    // Re-pair stripped paths (2 per pair: file_a, file_b)
    let stripped: Vec<(&str, &str)> = all_files
        .chunks_exact(2)
        .map(|c| (c[0].as_str(), c[1].as_str()))
        .collect();

    // Group by file path of side A
    let mut by_file: Vec<(String, Vec<(f32, String, String, String)>)> = Vec::new();
    let mut file_index: HashMap<String, usize> = HashMap::new();

    for (i, p) in pairs.iter().enumerate() {
        let (file_a, file_b) = stripped[i];
        let key = if file_a.is_empty() {
            "(unknown)".to_owned()
        } else {
            file_a.to_owned()
        };
        let idx = if let Some(&j) = file_index.get(&key) {
            j
        } else {
            let j = by_file.len();
            file_index.insert(key.clone(), j);
            by_file.push((key, Vec::new()));
            j
        };
        by_file[idx].1.push((
            p.similarity,
            labels[i].0.name.clone(),
            labels[i].1.name.clone(),
            file_b.to_owned(),
        ));
    }

    // Sort groups by number of pairs descending
    by_file.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

    for (file, entries) in &by_file {
        println!("  {} ({} pairs)", file, entries.len());
        for (sim, name_a, name_b, file_b) in entries {
            if file_b.is_empty() {
                println!("    [{sim:.2}]  {name_a} ↔ {name_b}");
            } else {
                println!("    [{sim:.2}]  {name_a} ↔ {name_b}  ({file_b})");
            }
        }
        println!();
    }

    let db_name = db.file_name().and_then(|n| n.to_str()).unwrap_or("db");
    eprintln!("Tip: v-hnsw gather {db_name} <symbol> --depth 1  to inspect.");
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

/// Check if two chunks overlap in the same file (parent/child relationship).
fn chunks_overlap(pstore: &impl PayloadStore, id_a: u64, id_b: u64) -> bool {
    let (Some(pa), Some(pb)) = (
        pstore.get_payload(id_a).ok().flatten(),
        pstore.get_payload(id_b).ok().flatten(),
    ) else {
        return false;
    };
    // Different files → no overlap
    if pa.source != pb.source {
        return false;
    }
    let (Some(PayloadValue::Integer(sa)), Some(PayloadValue::Integer(ea))) =
        (pa.custom.get("start_line"), pa.custom.get("end_line"))
    else {
        return false;
    };
    let (Some(PayloadValue::Integer(sb)), Some(PayloadValue::Integer(eb))) =
        (pb.custom.get("start_line"), pb.custom.get("end_line"))
    else {
        return false;
    };
    // Overlapping ranges: NOT (ea < sb || eb < sa)
    *sa <= *eb && *sb <= *ea
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

/// Find the longest common directory prefix among file paths, then strip it.
fn strip_common_prefix(paths: &mut [String]) {
    let non_empty: Vec<&str> = paths.iter().filter(|p| !p.is_empty()).map(String::as_str).collect();
    if non_empty.len() < 2 {
        return;
    }
    // Normalize separators to '/'
    for p in paths.iter_mut() {
        *p = p.replace('\\', "/");
    }
    // Find common prefix up to last shared '/'
    let first = paths.iter().find(|p| !p.is_empty()).cloned().unwrap_or_default();
    let mut prefix_len = first.len();
    for p in paths.iter().filter(|p| !p.is_empty()) {
        prefix_len = prefix_len.min(
            first.bytes()
                .zip(p.bytes())
                .take_while(|(a, b)| a == b)
                .count(),
        );
    }
    // Trim to last '/' boundary
    if let Some(pos) = first[..prefix_len].rfind('/') {
        prefix_len = pos + 1;
    } else {
        prefix_len = 0;
    }
    if prefix_len > 0 {
        for p in paths.iter_mut() {
            if p.len() >= prefix_len {
                *p = p[prefix_len..].to_owned();
            }
        }
    }
}

/// Parsed label with separated name and file path.
struct ChunkLabel {
    name: String,
    file: String,
}

impl ChunkLabel {
    fn display(&self) -> String {
        if self.file.is_empty() {
            self.name.clone()
        } else {
            format!("{}  ({})", self.name, self.file)
        }
    }
}

fn parse_label(pstore: &impl PayloadStore, id: u64) -> ChunkLabel {
    let Some(text) = pstore.get_text(id).ok().flatten() else {
        return ChunkLabel {
            name: format!("id:{id}"),
            file: String::new(),
        };
    };
    let mut lines = text.lines();
    let first = lines.next().unwrap_or("");
    let file = lines
        .next()
        .unwrap_or("")
        .strip_prefix("File: ")
        .unwrap_or("")
        .to_owned();
    let name = first
        .strip_prefix("[function] ")
        .or_else(|| first.strip_prefix("[impl] "))
        .or_else(|| first.strip_prefix("[struct] "))
        .unwrap_or(first)
        .to_owned();
    ChunkLabel { name, file }
}

fn label(pstore: &impl PayloadStore, id: u64) -> String {
    parse_label(pstore, id).display()
}
