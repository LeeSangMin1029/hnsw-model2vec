//! `dupes` command — find duplicate code chunks.
//!
//! Three modes:
//! - **Token Jaccard** (default): MinHash fingerprint-based near-duplicate detection.
//!   Compares actual code body tokens (unigrams + bigrams). Catches Type-1~3 clones.
//! - **AST hash** (`--ast`): structural clones ignoring identifier names (Type-1/2).
//! - **All** (`--all`): unified Filter→Verify pipeline combining AST + MinHash signals.
//!
//! Detection algorithms live in [`v_code_intel::clones`]; this module provides
//! CLI argument handling and output formatting only.

// serde_json::to_string only fails on non-string map keys, which we don't use.
#![expect(clippy::expect_used)]

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;
use v_hnsw_core::PayloadStore;
use v_code_intel::clones::{
    self, DupePair, RunStages, SubBlockClone, UnifiedDupePair,
};
use v_hnsw_storage::StorageEngine;

// ── JSON output types ────────────────────────────────────────────────────

#[derive(Serialize)]
struct PairsOutput {
    pairs: Vec<PairJson>,
}

#[derive(Serialize)]
struct PairJson {
    a: String,
    b: String,
    sim: f32,
}

#[derive(Serialize)]
struct UnifiedPairJson {
    a: String,
    b: String,
    score: f32,
    jaccard: f32,
    ast_match: bool,
    tag: String,
}

#[derive(Serialize)]
struct UnifiedPairsOutput {
    pairs: Vec<UnifiedPairJson>,
}

#[derive(Serialize)]
struct GroupJson {
    hash: String,
    members: Vec<String>,
}

#[derive(Serialize)]
struct GroupsOutput {
    groups: Vec<GroupJson>,
}

#[derive(Serialize)]
struct SubBlockJson {
    a: String,
    a_lines: [usize; 2],
    b: String,
    b_lines: [usize; 2],
    body_match: bool,
}

#[derive(Serialize)]
struct SubBlocksOutput {
    sub_block_clones: Vec<SubBlockJson>,
}

// ── CLI entry point ──────────────────────────────────────────────────────

/// Configuration for the dupes command.
pub struct DupesConfig {
    pub db: std::path::PathBuf,
    pub threshold: f32,
    pub exclude_tests: bool,
    pub k: usize,
    pub json: bool,
    pub ast_mode: bool,
    pub all_mode: bool,
    pub min_lines: usize,
    pub min_sub_lines: usize,
}

/// Run the dupes command.
pub fn run(cfg: DupesConfig) -> Result<()> {
    let DupesConfig {
        db, threshold, exclude_tests, k, json,
        ast_mode, all_mode, min_lines, min_sub_lines,
    } = cfg;
    let engine = StorageEngine::open(&db)
        .with_context(|| format!("failed to open database at {}", db.display()))?;
    let pstore = engine.payload_store();
    let candidate_ids = clones::collect_filtered_ids(&engine, pstore, exclude_tests, min_lines);

    let n = candidate_ids.len();
    if n < 2 {
        println!("Not enough chunks for comparison ({n}).");
        return Ok(());
    }

    if ast_mode {
        let groups = clones::find_hash_groups(pstore, &candidate_ids, "ast_hash", k);
        return if json {
            print_groups_json(&groups, pstore);
            Ok(())
        } else {
            print_groups_text(&groups, pstore, &db);
            Ok(())
        };
    }

    let stages = if all_mode {
        RunStages { ast: true, minhash: true }
    } else {
        RunStages { ast: false, minhash: true }
    };

    let is_unified = stages.ast && stages.minhash;

    // Single-signal fast path (MinHash only)
    if !is_unified {
        eprintln!("Comparing {n} chunks (Jaccard threshold={threshold:.2})...");
        let pairs = clones::find_minhash_pairs(pstore, &candidate_ids, threshold, k);

        return if json {
            print_pairs_json(&pairs, pstore);
            Ok(())
        } else {
            print_pairs_text(&pairs, pstore, &db);
            Ok(())
        };
    }

    // Unified multi-signal pipeline
    eprintln!("Unified pipeline: {n} chunks");

    let (unified_pairs, sub_clones) =
        clones::run_unified_pipeline(&engine, pstore, &candidate_ids, threshold, k, &stages, min_sub_lines)?;

    if unified_pairs.is_empty() {
        println!("No duplicates found.");
    } else if json {
        print_unified_json(&unified_pairs, pstore);
    } else {
        print_pairs_text(&unified_pairs, pstore, &db);
    }

    if !sub_clones.is_empty() {
        let capped: Vec<_> = sub_clones.into_iter().take(k).collect();
        if json {
            print_sub_block_json(&capped, pstore);
        } else {
            print_sub_block_text(&capped, pstore);
        }
    }

    Ok(())
}

// ── Output formatting ────────────────────────────────────────────────────

/// Common interface for duplicate pair types, enabling shared output logic.
trait DupePairLike {
    fn id_a(&self) -> u64;
    fn id_b(&self) -> u64;
    fn display_score(&self) -> f32;
    fn display_tag(&self) -> String {
        String::new()
    }
}

impl DupePairLike for DupePair {
    fn id_a(&self) -> u64 { self.id_a }
    fn id_b(&self) -> u64 { self.id_b }
    fn display_score(&self) -> f32 { self.similarity }
}

impl DupePairLike for UnifiedDupePair {
    fn id_a(&self) -> u64 { self.id_a }
    fn id_b(&self) -> u64 { self.id_b }
    fn display_score(&self) -> f32 { self.score }
    fn display_tag(&self) -> String { self.tag() }
}

/// An entry in a file-grouped duplicate listing.
struct GroupEntry {
    pair_index: usize,
    name_a: String,
    name_b: String,
    file_b: String,
}

// ── Label parsing ────────────────────────────────────────────────────────

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

// ── Path prefix stripping ────────────────────────────────────────────────

fn strip_common_prefix(paths: &mut [String]) {
    let non_empty: Vec<&str> = paths.iter().filter(|p| !p.is_empty()).map(String::as_str).collect();
    if non_empty.len() < 2 {
        return;
    }
    for p in paths.iter_mut() {
        *p = p.replace('\\', "/");
    }
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

// ── File grouping ────────────────────────────────────────────────────────

fn group_by_file(
    ids: &[(u64, u64)],
    pstore: &impl PayloadStore,
) -> Vec<(String, Vec<GroupEntry>)> {
    let labels: Vec<(ChunkLabel, ChunkLabel)> = ids
        .iter()
        .map(|&(a, b)| (parse_label(pstore, a), parse_label(pstore, b)))
        .collect();

    let mut all_files: Vec<String> = labels
        .iter()
        .flat_map(|(a, b)| [a.file.clone(), b.file.clone()])
        .collect();
    strip_common_prefix(&mut all_files);

    let stripped: Vec<(&str, &str)> = all_files
        .chunks_exact(2)
        .map(|c| (c[0].as_str(), c[1].as_str()))
        .collect();

    let mut by_file: Vec<(String, Vec<GroupEntry>)> = Vec::new();
    let mut file_index: HashMap<String, usize> = HashMap::new();

    for (i, _) in ids.iter().enumerate() {
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
        by_file[idx].1.push(GroupEntry {
            pair_index: i,
            name_a: labels[i].0.name.clone(),
            name_b: labels[i].1.name.clone(),
            file_b: file_b.to_owned(),
        });
    }

    by_file.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
    by_file
}

// ── Pair output (text / JSON) ────────────────────────────────────────────

fn print_pairs_text(
    pairs: &[impl DupePairLike],
    pstore: &impl PayloadStore,
    db: &Path,
) {
    if pairs.is_empty() {
        println!("No duplicates found above threshold.");
        return;
    }
    println!("{} duplicate pairs found:\n", pairs.len());

    let ids: Vec<(u64, u64)> = pairs.iter().map(|p| (p.id_a(), p.id_b())).collect();
    let groups = group_by_file(&ids, pstore);

    for (file, entries) in &groups {
        println!("  {} ({} pairs)", file, entries.len());
        for e in entries {
            let p = &pairs[e.pair_index];
            let score = p.display_score();
            let tag = p.display_tag();
            if tag.is_empty() {
                if e.file_b.is_empty() {
                    println!("    [{score:.2}]  {} \u{2194} {}", e.name_a, e.name_b);
                } else {
                    println!(
                        "    [{score:.2}]  {} \u{2194} {}  ({})",
                        e.name_a, e.name_b, e.file_b
                    );
                }
            } else {
                let tag_padded = format!("{tag:<12}");
                if e.file_b.is_empty() {
                    println!(
                        "    [{score:.2}] {tag_padded} {} \u{2194} {}",
                        e.name_a, e.name_b
                    );
                } else {
                    println!(
                        "    [{score:.2}] {tag_padded} {} \u{2194} {}  ({})",
                        e.name_a, e.name_b, e.file_b
                    );
                }
            }
        }
        println!();
    }

    let db_name = db.file_name().and_then(|n| n.to_str()).unwrap_or("db");
    eprintln!("Tip: v-hnsw gather {db_name} <symbol> --depth 1  to inspect.");
}

fn print_pairs_json(pairs: &[DupePair], pstore: &impl PayloadStore) {
    let output = PairsOutput {
        pairs: pairs.iter().map(|p| PairJson {
            a: label(pstore, p.id_a),
            b: label(pstore, p.id_b),
            sim: p.similarity,
        }).collect(),
    };
    println!("{}", serde_json::to_string(&output).expect("JSON serialize"));
}

fn print_unified_json(pairs: &[UnifiedDupePair], pstore: &impl PayloadStore) {
    let output = UnifiedPairsOutput {
        pairs: pairs.iter().map(|p| UnifiedPairJson {
            a: label(pstore, p.id_a),
            b: label(pstore, p.id_b),
            score: p.score,
            jaccard: p.jaccard,
            ast_match: p.ast_match,
            tag: p.tag(),
        }).collect(),
    };
    println!("{}", serde_json::to_string(&output).expect("JSON serialize"));
}

// ── Group output (AST hash mode) ─────────────────────────────────────────

fn print_groups_text(groups: &[(u64, Vec<u64>)], pstore: &impl PayloadStore, db: &Path) {
    if groups.is_empty() {
        println!("No clones found.");
        return;
    }
    println!("{} clone groups found:\n", groups.len());

    let all_labels: Vec<Vec<(usize, u64, ChunkLabel)>> = groups
        .iter()
        .enumerate()
        .map(|(gi, (hash, ids))| {
            ids.iter()
                .map(|&id| (gi + 1, *hash, parse_label(pstore, id)))
                .collect()
        })
        .collect();

    let mut all_files: Vec<String> = all_labels
        .iter()
        .flat_map(|g| g.iter().map(|(_, _, cl)| cl.file.clone()))
        .collect();
    strip_common_prefix(&mut all_files);

    type FileGroup = (String, Vec<(usize, u64, String)>);
    let mut by_file: Vec<FileGroup> = Vec::new();
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
    let output = GroupsOutput {
        groups: groups.iter().map(|(hash, ids)| GroupJson {
            hash: format!("{hash:016x}"),
            members: ids.iter().map(|&id| label(pstore, id)).collect(),
        }).collect(),
    };
    println!("{}", serde_json::to_string(&output).expect("JSON serialize"));
}

// ── Sub-block output ─────────────────────────────────────────────────────

fn print_sub_block_text(clones: &[SubBlockClone], pstore: &impl PayloadStore) {
    println!("\n{} sub-block clones (intra-function):\n", clones.len());
    for c in clones {
        let la = label(pstore, c.chunk_id_a);
        let lb = label(pstore, c.chunk_id_b);
        let lines_a = format!("L{}-{}", c.block_a_start + 1, c.block_a_end + 1);
        let lines_b = format!("L{}-{}", c.block_b_start + 1, c.block_b_end + 1);
        let body_tag = if c.body_match { " [exact]" } else { "" };
        println!("    {la} ({lines_a}) \u{2194} {lb} ({lines_b}){body_tag}");
    }
    println!();
}

fn print_sub_block_json(clones: &[SubBlockClone], pstore: &impl PayloadStore) {
    let output = SubBlocksOutput {
        sub_block_clones: clones.iter().map(|c| SubBlockJson {
            a: label(pstore, c.chunk_id_a),
            a_lines: [c.block_a_start + 1, c.block_a_end + 1],
            b: label(pstore, c.chunk_id_b),
            b_lines: [c.block_b_start + 1, c.block_b_end + 1],
            body_match: c.body_match,
        }).collect(),
    };
    println!("{}", serde_json::to_string(&output).expect("JSON serialize"));
}
