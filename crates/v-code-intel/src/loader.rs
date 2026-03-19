//! Load code chunks from a database with bincode caching.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use v_hnsw_core::{PayloadStore, PayloadValue};
use v_hnsw_storage::StorageEngine;

use crate::parse::{self, ParsedChunk};
use crate::rustdoc::RustdocTypes;

/// Cache format version for `chunks.bin` — bump when `ParsedChunk` layout changes.
const CHUNKS_CACHE_VERSION: u8 = 1;

/// Load all code chunks from the database, using a bincode cache.
pub fn load_chunks(path: &Path) -> Result<Vec<ParsedChunk>> {
    let cache = cache_path(path);
    // Use payload.dat mtime (not directory mtime) — directory mtime
    // doesn't update on Windows when files inside are modified.
    let db_mtime = fs::metadata(path.join("payload.dat"))
        .and_then(|m| m.modified())
        .ok();

    // Try cache hit: version prefix byte + bincode payload.
    if let Some(db_t) = db_mtime
        && let Ok(cache_meta) = fs::metadata(&cache)
        && let Ok(cache_t) = cache_meta.modified()
        && cache_t >= db_t
        && let Ok(bytes) = fs::read(&cache)
        && bytes.first() == Some(&CHUNKS_CACHE_VERSION)
    {
        let config = bincode::config::standard();
        if let Ok((chunks, _)) =
            bincode::decode_from_slice::<Vec<ParsedChunk>, _>(&bytes[1..], config)
        {
            eprintln!("  chunks.bin cache hit: {} chunks from {:.1}MB", chunks.len(), bytes.len() as f64 / 1_048_576.0);
            return Ok(chunks);
        }
    }

    // Cache miss — load from DB and save
    eprintln!("  chunks.bin cache miss — loading from DB");
    let chunks = load_chunks_from_db(path)?;
    save_chunks_cache(&cache, &chunks);
    Ok(chunks)
}

/// Load chunks directly from the database (no cache).
pub fn load_chunks_from_db(path: &Path) -> Result<Vec<ParsedChunk>> {
    let engine = StorageEngine::open(path)
        .with_context(|| format!("failed to open database at {}", path.display()))?;

    let vector_store = engine.vector_store();
    let payload_store = engine.payload_store();
    let mut ids: Vec<u64> = vector_store.id_map().keys().copied().collect();
    // Sort by ID for deterministic chunk ordering (HashMap iteration is random).
    ids.sort_unstable();

    let mut chunks = Vec::new();
    for id in ids {
        if let Ok(Some(text)) = payload_store.get_text(id)
            && let Some(mut chunk) = parse::parse_chunk(&text)
        {
            if let Ok(Some(payload)) = payload_store.get_payload(id)
                && let Some(PayloadValue::StringList(imports)) = payload.custom.get("imports") {
                    chunk.imports.clone_from(imports);
                }
            chunks.push(chunk);
        }
    }
    Ok(chunks)
}

/// Save parsed chunks to the bincode cache file.
///
/// Called internally after DB load, and externally by `v-code add`
/// to pre-build the cache (avoids expensive text re-parsing on first verify).
pub fn save_chunks_cache(path: &Path, chunks: &[ParsedChunk]) {
    let config = bincode::config::standard();
    // Prepend version byte, then encode chunks.
    let mut bytes = vec![CHUNKS_CACHE_VERSION];
    if let Ok(chunk_bytes) = bincode::encode_to_vec(chunks, config) {
        bytes.extend_from_slice(&chunk_bytes);
        let _ = fs::write(path, bytes);
    }
}

/// Path to the chunks.bin cache file for a given database.
pub fn cache_path(db: &Path) -> PathBuf {
    db.join("cache").join("chunks.bin")
}

/// Inject synthetic chunks for macro-generated methods discovered by rustdoc.
///
/// When a macro like `define_chunker!(CppCodeChunker, ...)` generates methods,
/// tree-sitter can't see the expanded code, so the DB only has the template
/// (e.g. `CCodeChunker::chunk`). Rustdoc, however, lists all methods including
/// macro-generated ones. This function creates minimal synthetic chunks for
/// each `owner::method` that rustdoc knows about but tree-sitter missed.
///
/// Returns the number of synthetic chunks added.
fn inject_rustdoc_synthetic_chunks(chunks: &mut Vec<ParsedChunk>, rustdoc: &RustdocTypes, extern_methods: &HashSet<String>) -> usize {
    // Build a set of existing `owner::method` keys (lowercase).
    let mut existing: HashSet<String> = HashSet::new();
    // Also collect owner struct/type locations: owner_lower → (file, lines).
    let mut owner_locations: std::collections::HashMap<String, (String, Option<(usize, usize)>)> =
        std::collections::HashMap::new();

    for c in chunks.iter() {
        let lower = c.name.to_lowercase();
        existing.insert(lower.clone());
        // Strip generics: "foo<t>::bar" → "foo::bar"
        let stripped = crate::graph::strip_generics_from_key(&lower);
        if stripped != lower {
            existing.insert(stripped);
        }
        // Owner::method alias: "mod::owner::method" → "owner::method"
        if let Some((prefix, method)) = lower.rsplit_once("::") {
            if let Some(owner_leaf) = prefix.rsplit_once("::").map(|p| p.1) {
                existing.insert(format!("{owner_leaf}::{method}"));
            }
        }
        // Track struct/enum/type locations for synthetic chunk file attribution.
        if matches!(c.kind.as_str(), "struct" | "enum" | "type" | "trait") {
            let leaf = c.name.rsplit("::").next().unwrap_or(&c.name).to_lowercase();
            owner_locations.entry(leaf).or_insert_with(|| (c.file.clone(), c.lines));
        }
    }

    // Build method→[existing_owner] index from existing chunks.
    // Only create synthetic chunks for methods that already have at least one
    // tree-sitter-discovered owner — this means the method is a real project
    // method, not something only rustdoc knows about (e.g. derived traits).
    let mut method_existing_owners: std::collections::HashMap<String, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        if c.kind != "function" {
            continue;
        }
        let lower = c.name.to_lowercase();
        if let Some(method) = lower.rsplit_once("::").map(|p| p.1) {
            method_existing_owners.entry(method.to_owned()).or_default().push(i);
        }
    }

    let mut added = 0;
    for (method, owners) in &rustdoc.method_owner {
        // Skip methods that exist in std/deps AND where rustdoc doesn't
        // know more owners than tree-sitter — no macro-generated gap to fill.
        // If rustdoc has MORE owners than tree-sitter, the extras are likely
        // macro-generated (e.g. define_chunker!) and worth synthesizing.
        if extern_methods.contains(method.as_str()) {
            let ts_count = method_existing_owners.get(method.as_str())
                .map_or(0, |v| v.len());
            if owners.len() <= ts_count {
                continue;
            }
        }
        // Only add synthetic chunks if tree-sitter already found this method
        // on at least one type. This prevents creating synthetics for derived
        // trait methods (build, default, clone) that would pollute resolution.
        let Some(template_indices) = method_existing_owners.get(method.as_str()) else {
            continue;
        };
        let template_idx = template_indices[0];

        for owner in owners {
            let qualified = format!("{owner}::{method}");
            if existing.contains(&qualified) {
                continue;
            }
            let template = &chunks[template_idx];
            // Use owner struct location if available, else template's file.
            let (file, lines) = owner_locations.get(owner)
                .cloned()
                .unwrap_or_else(|| (template.file.clone(), template.lines));
            let sig = template.signature.clone();
            let return_type = template.return_type.clone();

            // Rustdoc stores lowercase; capitalize the owner for display.
            let display_name = format!("{}::{method}", capitalize_first(owner));

            chunks.push(ParsedChunk {
                kind: "function".to_owned(),
                name: display_name,
                file,
                lines,
                signature: sig,
                calls: Vec::new(),
                call_lines: Vec::new(),
                types: Vec::new(),
                imports: Vec::new(),
                string_args: Vec::new(),
                param_flows: Vec::new(),
                param_types: Vec::new(),
                field_types: Vec::new(),
                local_types: Vec::new(),
                let_call_bindings: Vec::new(),
                return_type,
                field_accesses: Vec::new(),
                enum_variants: Vec::new(),
            });
            existing.insert(qualified);
            added += 1;
        }
    }
    added
}

/// Capitalize the first character of a string.
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().chain(chars).collect(),
    }
}

/// Result of a daemon graph build attempt.
pub enum DaemonBuildResult {
    /// Daemon returned a completed graph.
    Ready(Box<crate::graph::CallGraph>),
    /// Daemon is building asynchronously — don't cache the tree-sitter fallback.
    Building,
    /// Daemon not available.
    Unavailable,
}

/// Optional daemon hooks for graph building.
///
/// Callers that have access to `v-daemon` can provide these to enable
/// daemon-assisted graph builds (rustdoc + tree-sitter, no LSP).
pub struct DaemonHooks {
    /// Try to build graph via running daemon.
    pub try_graph_build: fn(&Path) -> DaemonBuildResult,
    /// Spawn daemon in background for next invocation.
    pub spawn: fn(&Path),
}

/// Load graph from cache or build from chunks.
///
/// Resolution strategy (in order):
/// 1. Graph cache hit → return immediately
/// 2. Daemon running → delegate `graph/build`
/// 3. Tree-sitter + rustdoc heuristic fallback
pub fn load_or_build_graph(
    db: &Path,
    daemon: Option<&DaemonHooks>,
) -> Result<crate::graph::CallGraph> {
    let (g, _) = load_or_build_graph_with_chunks(db, daemon)?;
    Ok(g)
}

/// Load or build the call graph, also returning the parsed chunks if they were loaded.
///
/// Returns `(graph, Some(chunks))` when chunks were loaded for graph building,
/// or `(graph, None)` when the graph was loaded from cache (chunks not needed).
/// Callers that need both graph and chunks can avoid double-loading.
pub fn load_or_build_graph_with_chunks(
    db: &Path,
    daemon: Option<&DaemonHooks>,
) -> Result<(crate::graph::CallGraph, Option<Vec<ParsedChunk>>)> {
    if let Some(g) = crate::graph::CallGraph::load(db) {
        return Ok((g, None));
    }

    // Try daemon first.
    let mut daemon_building = false;
    if let Some(hooks) = daemon {
        match (hooks.try_graph_build)(db) {
            DaemonBuildResult::Ready(g) => return Ok((*g, None)),
            DaemonBuildResult::Building => daemon_building = true,
            DaemonBuildResult::Unavailable => (hooks.spawn)(db),
        }
    }

    let mut chunks = load_chunks(db)?;

    // Try loading cached type info for enrichment.
    let rustdoc = crate::rustdoc::load_cached(db);

    // Load extern index — try cache first, build if unavailable.
    let extern_index = crate::extern_types::ExternMethodIndex::try_load_cached(db)
        .or_else(|| Some(crate::extern_types::ExternMethodIndex::build(db)));

    // Inject synthetic chunks for macro-generated methods discovered by rustdoc.
    // Must come after extern index load so we can filter out std/deps methods.
    if let Some(ref rdoc) = rustdoc {
        let extern_methods = extern_index.as_ref()
            .map(|ext| ext.all_method_set())
            .unwrap_or_default();
        let count = inject_rustdoc_synthetic_chunks(&mut chunks, rdoc, &extern_methods);
        if count > 0 {
            eprintln!("  rustdoc synthetic chunks: {count}");
        }
    }

    let g = crate::graph::CallGraph::build_full(
        &chunks, rustdoc.as_ref(), extern_index.as_ref(),
    );

    // Don't persist tree-sitter fallback when daemon is building —
    // daemon will save the accurate graph.bin when done.
    if !daemon_building {
        let _ = g.save(db);
    }
    Ok((g, Some(chunks)))
}
