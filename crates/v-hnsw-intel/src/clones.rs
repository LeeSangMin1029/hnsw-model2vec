//! Clone detection algorithms — finds duplicate code chunks in a database.
//!
//! Three detection signals:
//! - **AST hash**: structural clones ignoring identifier names (Type-1/2)
//! - **MinHash Jaccard**: token-based near-duplicate detection (Type-1~3)
//! - **HNSW embedding**: semantic similarity via vector ANN (Type-3/4)
//!
//! Two execution modes:
//! - Single-signal fast path (one signal only, user threshold)
//! - Unified pipeline (all signals: Filter → Verify, weighted scoring)

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{Context, Result};
use rayon::prelude::*;
use v_hnsw_code::extract;
use v_hnsw_core::{PayloadStore, PayloadValue, VectorStore};
use v_hnsw_graph::distance::dot_product;
use v_hnsw_graph::{HnswGraph, NormalizedCosineDistance};
use v_hnsw_storage::StorageEngine;

// ── Configuration ────────────────────────────────────────────────────────

/// Which detection stages to enable.
pub struct RunStages {
    pub ast: bool,
    pub minhash: bool,
    pub embed: bool,
}

// ── Result types ─────────────────────────────────────────────────────────

/// A pair of duplicate chunks with similarity score.
pub struct DupePair {
    pub id_a: u64,
    pub id_b: u64,
    pub similarity: f32,
}

/// Result of the unified multi-signal pipeline.
pub struct UnifiedDupePair {
    pub id_a: u64,
    pub id_b: u64,
    pub score: f32,
    pub jaccard: f32,
    pub cosine: f32,
    pub ast_match: bool,
}

impl UnifiedDupePair {
    /// Build a tag string like "AST", "Token", "Embed", "Token+Embed", etc.
    pub fn tag(&self) -> String {
        let mut parts = Vec::new();
        if self.ast_match {
            parts.push("AST");
        }
        if self.jaccard >= 0.5 {
            parts.push("Token");
        }
        if self.cosine >= 0.5 {
            parts.push("Embed");
        }
        if parts.is_empty() {
            parts.push("Weak");
        }
        parts.join("+")
    }
}

/// A pair of sub-blocks from different chunks that share the same AST hash.
pub struct SubBlockClone {
    pub chunk_id_a: u64,
    pub chunk_id_b: u64,
    pub block_a_start: usize,
    pub block_a_end: usize,
    pub block_b_start: usize,
    pub block_b_end: usize,
    pub body_match: bool,
}

/// Combined results from the clone detection pipeline.
pub struct CloneResults {
    pub simple_pairs: Vec<DupePair>,
    pub unified_pairs: Vec<UnifiedDupePair>,
    pub sub_block_clones: Vec<SubBlockClone>,
    /// Hash-based groups: (hash, Vec<member_ids>).
    pub hash_groups: Vec<(u64, Vec<u64>)>,
}

// ── Candidate collection ─────────────────────────────────────────────────

/// Collect candidate IDs from the vector store, applying test/line filters.
pub fn collect_filtered_ids(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    exclude_tests: bool,
    min_lines: usize,
) -> Vec<u64> {
    let vstore = engine.vector_store();
    let mut ids: Vec<u64> = vstore.id_map().keys().copied().collect();
    if exclude_tests {
        ids.retain(|id| !is_test_chunk(pstore, *id));
    }
    if min_lines > 0 {
        ids.retain(|id| chunk_lines(pstore, *id) >= min_lines);
    }
    ids
}

// ── Single-signal: AST hash groups ───────────────────────────────────────

/// Find clone groups by AST hash, with overlap removal.
///
/// Returns groups sorted by size descending, truncated to `k`.
pub fn find_hash_groups(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    hash_key: &str,
    k: usize,
) -> Vec<(u64, Vec<u64>)> {
    let mut hash_groups = collect_hash_groups(pstore, candidate_ids, hash_key);

    // Remove overlapping chunks within each group (keep the larger span)
    for ids in hash_groups.values_mut() {
        if ids.len() > 1 {
            remove_overlapping_chunks(pstore, ids);
        }
    }

    let mut clone_groups: Vec<(u64, Vec<u64>)> = hash_groups
        .into_iter()
        .filter(|(_, ids)| ids.len() > 1)
        .collect();

    clone_groups.sort_unstable_by(|a, b| b.1.len().cmp(&a.1.len()));
    clone_groups.truncate(k);
    clone_groups
}

// ── Single-signal: MinHash Jaccard ───────────────────────────────────────

/// Find duplicate pairs by MinHash Jaccard similarity.
///
/// Returns pairs above `threshold`, overlap-filtered, sorted desc, truncated to `k`.
pub fn find_minhash_pairs(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    threshold: f32,
    k: usize,
) -> Vec<DupePair> {
    let entries = collect_minhash_entries(pstore, candidate_ids);

    if entries.len() < 2 {
        return Vec::new();
    }

    let mut pairs = minhash_all_pairs(&entries, f64::from(threshold));
    finalize_pairs(&mut pairs, pstore, k);
    pairs
}

// ── Single-signal: HNSW embedding ────────────────────────────────────────

/// Find duplicate pairs by HNSW k-ANN cosine similarity.
///
/// Returns pairs above `threshold`, overlap-filtered, sorted desc, truncated to `k`.
pub fn find_embed_pairs(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    threshold: f32,
    k: usize,
    db: &Path,
) -> Result<Vec<DupePair>> {
    let hnsw_path = db.join("hnsw.bin");
    if !hnsw_path.exists() {
        anyhow::bail!(
            "HNSW index not found at {}. Run 'v-hnsw add' to index.",
            hnsw_path.display()
        );
    }

    let search_k = (k * 3).max(20);
    let raw_pairs = hnsw_similar_pairs(engine, candidate_ids, search_k, threshold, db)?;
    let mut pairs: Vec<DupePair> = raw_pairs
        .into_iter()
        .map(|(id_a, id_b, similarity)| DupePair {
            id_a,
            id_b,
            similarity,
        })
        .collect();

    finalize_pairs(&mut pairs, pstore, k);
    Ok(pairs)
}

// ── Unified multi-signal pipeline ────────────────────────────────────────

/// Run the full unified pipeline: Filter (AST+MinHash+HNSW) → Verify → Score.
///
/// Returns `(unified_pairs, sub_block_clones)`.
pub fn run_unified_pipeline(
    engine: &StorageEngine,
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    threshold: f32,
    k: usize,
    db: &Path,
    stages: &RunStages,
) -> Result<(Vec<UnifiedDupePair>, Vec<SubBlockClone>)> {
    let vstore = engine.vector_store();

    // Stage 1: Filter (collect candidate pairs)
    let mut candidates: HashSet<(u64, u64)> = HashSet::new();

    if stages.ast {
        stage1_ast_hash(pstore, candidate_ids, &mut candidates);
    }
    if stages.minhash {
        stage1_minhash(pstore, candidate_ids, &mut candidates);
    }
    if stages.embed {
        stage1_hnsw(engine, candidate_ids, &mut candidates, db)?;
    }

    eprintln!("Stage 1: {} candidate pairs", candidates.len());

    if candidates.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    // Stage 2: Verify (compute all signals for each candidate)
    let minhash_map: HashMap<u64, Vec<u64>> = candidate_ids
        .iter()
        .filter_map(|&id| get_minhash(pstore, id).map(|sig| (id, sig)))
        .collect();

    let ast_map: HashMap<u64, u64> = candidate_ids
        .iter()
        .filter_map(|&id| get_hash(pstore, id, "ast_hash").map(|h| (id, h)))
        .collect();

    let candidate_vec: Vec<(u64, u64)> = candidates.into_iter().collect();

    let mut pairs: Vec<UnifiedDupePair> = candidate_vec
        .iter()
        .filter_map(|&(id_a, id_b)| {
            if chunks_overlap(pstore, id_a, id_b) {
                return None;
            }

            let jaccard = match (minhash_map.get(&id_a), minhash_map.get(&id_b)) {
                (Some(sig_a), Some(sig_b)) => {
                    #[expect(clippy::cast_possible_truncation)]
                    let j = extract::jaccard_from_minhash(sig_a, sig_b) as f32;
                    j
                }
                _ => 0.0,
            };

            let cosine = match (vstore.get(id_a), vstore.get(id_b)) {
                (Ok(va), Ok(vb)) => dot_product(va, vb),
                _ => 0.0,
            };

            let ast_match = match (ast_map.get(&id_a), ast_map.get(&id_b)) {
                (Some(ha), Some(hb)) => ha == hb,
                _ => false,
            };

            let score = if ast_match {
                1.0_f32.max(jaccard.max(cosine))
            } else {
                jaccard.max(cosine)
            };

            (score >= threshold).then_some(UnifiedDupePair {
                id_a,
                id_b,
                score,
                jaccard,
                cosine,
                ast_match,
            })
        })
        .collect();

    pairs.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(k);

    // Sub-block clone detection
    let sub_clones = find_sub_block_clones(pstore, candidate_ids);

    Ok((pairs, sub_clones))
}

// ── HNSW neighbor search (shared) ────────────────────────────────────────

/// Search HNSW neighbors and return deduplicated similar pairs above threshold.
fn hnsw_similar_pairs(
    engine: &StorageEngine,
    candidate_ids: &[u64],
    search_k: usize,
    threshold: f32,
    db: &Path,
) -> Result<Vec<(u64, u64, f32)>> {
    let hnsw_path = db.join("hnsw.bin");
    let hnsw: HnswGraph<NormalizedCosineDistance> =
        HnswGraph::load(&hnsw_path, NormalizedCosineDistance)
            .with_context(|| format!("failed to load HNSW index at {}", hnsw_path.display()))?;
    let vstore = engine.vector_store();

    let n = candidate_ids.len();
    let search_k = search_k.min(n);
    let ef = 200;

    let candidate_set: HashSet<u64> = candidate_ids.iter().copied().collect();
    let mut seen = HashSet::<(u64, u64)>::new();
    let mut result = Vec::new();

    for &id in candidate_ids {
        let Ok(query_vec) = vstore.get(id) else {
            continue;
        };
        let Ok(neighbors) = hnsw.search_ext(vstore, query_vec, search_k, ef) else {
            continue;
        };

        for &(nid, _distance) in &neighbors {
            if nid == id || !candidate_set.contains(&nid) {
                continue;
            }
            let pair_key = if id < nid { (id, nid) } else { (nid, id) };
            if !seen.insert(pair_key) {
                continue;
            }
            let Ok(neighbor_vec) = vstore.get(nid) else {
                continue;
            };
            let sim = dot_product(query_vec, neighbor_vec);
            if sim >= threshold {
                result.push((pair_key.0, pair_key.1, sim));
            }
        }
    }

    Ok(result)
}

// ── Pipeline stages ──────────────────────────────────────────────────────

/// Stage 1a: Group by AST hash → all intra-group pairs become candidates.
fn stage1_ast_hash(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    candidates: &mut HashSet<(u64, u64)>,
) {
    let hash_groups = collect_hash_groups(pstore, candidate_ids, "ast_hash");

    let mut ast_pairs = 0usize;
    for ids in hash_groups.values() {
        if ids.len() > 1 {
            for i in 0..ids.len() {
                for j in (i + 1)..ids.len() {
                    let pair = if ids[i] < ids[j] {
                        (ids[i], ids[j])
                    } else {
                        (ids[j], ids[i])
                    };
                    candidates.insert(pair);
                    ast_pairs += 1;
                }
            }
        }
    }
    eprintln!("  AST hash: {ast_pairs} candidate pairs");
}

/// Stage 1b: MinHash Jaccard with low threshold (0.3) for broad candidate collection.
fn stage1_minhash(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    candidates: &mut HashSet<(u64, u64)>,
) {
    let entries = collect_minhash_entries(pstore, candidate_ids);

    let n = entries.len();
    if n < 2 {
        eprintln!("  MinHash: not enough chunks ({n})");
        return;
    }

    let pairs = minhash_all_pairs(&entries, 0.3);
    let minhash_pairs: Vec<(u64, u64)> = pairs
        .into_iter()
        .map(|p| {
            if p.id_a < p.id_b {
                (p.id_a, p.id_b)
            } else {
                (p.id_b, p.id_a)
            }
        })
        .collect();

    eprintln!(
        "  MinHash: {} candidate pairs (threshold=0.30)",
        minhash_pairs.len()
    );
    candidates.extend(minhash_pairs);
}

/// Stage 1c: HNSW k-ANN search with low threshold for broad candidate collection.
fn stage1_hnsw(
    engine: &StorageEngine,
    candidate_ids: &[u64],
    candidates: &mut HashSet<(u64, u64)>,
    db: &Path,
) -> Result<()> {
    let hnsw_path = db.join("hnsw.bin");
    if !hnsw_path.exists() {
        eprintln!("  HNSW: index not found, skipping embedding stage");
        return Ok(());
    }

    let raw_pairs = hnsw_similar_pairs(engine, candidate_ids, 20, 0.5, db)?;
    let mut hnsw_pairs = 0usize;
    for (id_a, id_b, _sim) in raw_pairs {
        if candidates.insert((id_a, id_b)) {
            hnsw_pairs += 1;
        }
    }

    eprintln!("  HNSW: {hnsw_pairs} new candidate pairs (threshold=0.50)");
    Ok(())
}

// ── Sub-block clone detection ────────────────────────────────────────────

/// Parse `sub_block_hashes` payload field: `["<hex_ast_hash>:<start>-<end>", ...]`
fn parse_sub_block_entries(pstore: &impl PayloadStore, id: u64) -> Vec<(u64, usize, usize)> {
    let Some(payload) = pstore.get_payload(id).ok().flatten() else {
        return Vec::new();
    };
    let Some(PayloadValue::StringList(hashes)) = payload.custom.get("sub_block_hashes") else {
        return Vec::new();
    };
    hashes
        .iter()
        .filter_map(|s| {
            let (hash_hex, range) = s.split_once(':')?;
            let (start_s, end_s) = range.split_once('-')?;
            let hash = u64::from_str_radix(hash_hex, 16).ok()?;
            let start = start_s.parse().ok()?;
            let end = end_s.parse().ok()?;
            Some((hash, start, end))
        })
        .collect()
}

/// Find sub-block clones across all candidate chunks.
fn find_sub_block_clones(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
) -> Vec<SubBlockClone> {
    let mut hash_groups: HashMap<u64, Vec<(u64, usize, usize)>> = HashMap::new();

    for &id in candidate_ids {
        for (ast_hash, start, end) in parse_sub_block_entries(pstore, id) {
            hash_groups
                .entry(ast_hash)
                .or_default()
                .push((id, start, end));
        }
    }

    let mut clones = Vec::new();
    for entries in hash_groups.values() {
        if entries.len() < 2 {
            continue;
        }
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                let (id_a, sa, ea) = entries[i];
                let (id_b, sb, eb) = entries[j];
                if id_a == id_b {
                    continue;
                }
                if same_file(pstore, id_a, id_b) && ranges_overlap(sa, ea, sb, eb) {
                    continue;
                }
                clones.push(SubBlockClone {
                    chunk_id_a: id_a,
                    chunk_id_b: id_b,
                    block_a_start: sa,
                    block_a_end: ea,
                    block_b_start: sb,
                    block_b_end: eb,
                    body_match: false,
                });
            }
        }
    }

    clones.sort_by(|a, b| {
        let size_a = a.block_a_end.saturating_sub(a.block_a_start);
        let size_b = b.block_a_end.saturating_sub(b.block_a_start);
        size_b.cmp(&size_a)
    });
    clones.truncate(50);
    clones
}

// ── Shared helpers ───────────────────────────────────────────────────────

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
pub fn chunks_overlap(pstore: &impl PayloadStore, id_a: u64, id_b: u64) -> bool {
    let (Some(pa), Some(pb)) = (
        pstore.get_payload(id_a).ok().flatten(),
        pstore.get_payload(id_b).ok().flatten(),
    ) else {
        return false;
    };
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
    *sa <= *eb && *sb <= *ea
}

/// Get the number of lines in a chunk from its custom fields.
pub fn chunk_lines(pstore: &impl PayloadStore, id: u64) -> usize {
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

fn get_hash(pstore: &impl PayloadStore, id: u64, key: &str) -> Option<u64> {
    let payload = pstore.get_payload(id).ok()??;
    match payload.custom.get(key)? {
        #[expect(clippy::cast_sign_loss, reason = "hash bits reinterpreted")]
        PayloadValue::Integer(v) => Some(*v as u64),
        _ => None,
    }
}

fn get_minhash(pstore: &impl PayloadStore, id: u64) -> Option<Vec<u64>> {
    let payload = pstore.get_payload(id).ok()??;
    match payload.custom.get("minhash")? {
        PayloadValue::String(hex) => extract::minhash_from_hex(hex),
        _ => None,
    }
}

fn collect_hash_groups(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
    hash_key: &str,
) -> HashMap<u64, Vec<u64>> {
    let mut hash_groups: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut no_hash = 0u32;

    for &id in candidate_ids {
        if let Some(hash) = get_hash(pstore, id, hash_key) {
            hash_groups.entry(hash).or_default().push(id);
        } else {
            no_hash += 1;
        }
    }

    if no_hash > 0 {
        eprintln!("  {hash_key}: {no_hash} chunks without {hash_key}");
    }
    hash_groups
}

fn collect_minhash_entries(
    pstore: &impl PayloadStore,
    candidate_ids: &[u64],
) -> Vec<(u64, Vec<u64>)> {
    let mut entries: Vec<(u64, Vec<u64>)> = Vec::new();
    let mut no_minhash = 0u32;

    for &id in candidate_ids {
        if let Some(sig) = get_minhash(pstore, id) {
            entries.push((id, sig));
        } else {
            no_minhash += 1;
        }
    }

    if no_minhash > 0 {
        eprintln!("  MinHash: {no_minhash} chunks without minhash");
    }
    entries
}

/// All-pairs MinHash Jaccard comparison (parallelized).
fn minhash_all_pairs(entries: &[(u64, Vec<u64>)], threshold: f64) -> Vec<DupePair> {
    let n = entries.len();
    (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            let (id_a, sig_a) = &entries[i];
            ((i + 1)..n).filter_map(move |j| {
                let (id_b, sig_b) = &entries[j];
                let sim = extract::jaccard_from_minhash(sig_a, sig_b);
                #[expect(clippy::cast_possible_truncation)]
                (sim >= threshold).then_some(DupePair {
                    id_a: *id_a,
                    id_b: *id_b,
                    similarity: sim as f32,
                })
            })
        })
        .collect()
}

/// Sort pairs by similarity descending, filter overlapping chunks, truncate to top `k`.
fn finalize_pairs(pairs: &mut Vec<DupePair>, pstore: &impl PayloadStore, k: usize) {
    pairs.retain(|p| !chunks_overlap(pstore, p.id_a, p.id_b));
    pairs.sort_unstable_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs.truncate(k);
}

/// Remove overlapping chunks within a group (keep the larger span).
fn remove_overlapping_chunks(pstore: &impl PayloadStore, ids: &mut Vec<u64>) {
    let mut i = 0;
    while i < ids.len() {
        let mut j = i + 1;
        while j < ids.len() {
            if chunks_overlap(pstore, ids[i], ids[j]) {
                if chunk_lines(pstore, ids[i]) >= chunk_lines(pstore, ids[j]) {
                    ids.swap_remove(j);
                } else {
                    ids.swap_remove(i);
                    j = i + 1;
                    continue;
                }
            } else {
                j += 1;
            }
        }
        i += 1;
    }
}

fn same_file(pstore: &impl PayloadStore, id_a: u64, id_b: u64) -> bool {
    let file_a = pstore
        .get_payload(id_a)
        .ok()
        .flatten()
        .map(|p| p.source.clone());
    let file_b = pstore
        .get_payload(id_b)
        .ok()
        .flatten()
        .map(|p| p.source.clone());
    file_a.is_some() && file_a == file_b
}

fn ranges_overlap(s1: usize, e1: usize, s2: usize, e2: usize) -> bool {
    s1 <= e2 && s2 <= e1
}
