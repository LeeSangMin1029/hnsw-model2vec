//! Code intelligence (v-code) daemon handlers: graph/build + ra/collect-types.
//!
//! Builds call graphs using tree-sitter + RA call hierarchy.
//! Collects hover-based type information using the daemon's resident RA instance.

use std::path::PathBuf;
use std::sync::Mutex;

/// Serializes concurrent `graph/build` requests so only one runs at a time.
static GRAPH_LOCK: Mutex<()> = Mutex::new(());

pub fn handle_graph_build(
    params: serde_json::Value,
    _ra: Option<&v_lsp::instance::RaInstance>,
) -> anyhow::Result<serde_json::Value> {
    let _guard = GRAPH_LOCK
        .lock()
        .map_err(|e| anyhow::anyhow!("graph lock poisoned: {e}"))?;

    #[derive(serde::Deserialize)]
    struct GraphBuildParams {
        db: String,
    }

    let p: GraphBuildParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid graph/build params: {e}"))?;
    let db_path = PathBuf::from(&p.db);
    let db = db_path
        .canonicalize()
        .unwrap_or_else(|_| db_path.clone());

    let chunks = v_code_intel::loader::load_chunks(&db)?;
    let graph = v_code_intel::graph::CallGraph::build_full(&chunks);
    let _ = graph.save(&db);

    eprintln!(
        "[daemon] Graph built: {} nodes",
        graph.len(),
    );

    Ok(serde_json::json!({
        "status": "ok",
        "nodes": graph.len(),
    }))
}

/// Collect hover-based type information using the daemon's resident RA instance.
///
/// Loads chunks from the DB cache, runs `collect_types_via_ra`, and returns
/// serialized `LspTypes`. This avoids RA re-spawn (~20s) on incremental adds.
pub fn handle_collect_types(
    params: serde_json::Value,
    ra: Option<&v_lsp::instance::RaInstance>,
) -> anyhow::Result<serde_json::Value> {
    let ra = ra.ok_or_else(|| anyhow::anyhow!("RA not available"))?;

    #[derive(serde::Deserialize)]
    struct CollectTypesParams {
        db: String,
    }

    let p: CollectTypesParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid ra/collect-types params: {e}"))?;
    let db_path = PathBuf::from(&p.db);
    let db = db_path
        .canonicalize()
        .unwrap_or_else(|_| db_path.clone());

    let chunks = v_code_intel::loader::load_chunks(&db)?;
    let lsp_types = v_code_intel::lsp_client::collect_types_via_ra(&chunks, ra);

    let receiver_count: usize = lsp_types.receiver_types.values().map(|m| m.len()).sum();
    eprintln!(
        "[daemon] collect-types: {} receivers, {} return_types",
        receiver_count,
        lsp_types.return_types.len(),
    );

    Ok(serde_json::to_value(lsp_types)?)
}

/// Extract code chunks from files using the daemon's RA instance.
///
/// Replaces tree-sitter chunking: uses `file_structure`, `outgoing_calls`,
/// `discover_tests_in_file`, and source text parsing.
pub fn handle_chunk_files(
    params: serde_json::Value,
    ra: Option<&v_lsp::instance::RaInstance>,
) -> anyhow::Result<serde_json::Value> {
    let ra = ra.ok_or_else(|| anyhow::anyhow!("RA not available — daemon starting?"))?;

    #[derive(serde::Deserialize)]
    struct ChunkParams {
        files: Vec<String>,
    }

    let p: ChunkParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid code/chunk params: {e}"))?;

    let t0 = std::time::Instant::now();
    let chunks = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ra.chunk_files(&p.files)
    })).map_err(|e| anyhow::anyhow!("chunk_files panicked: {:?}", e))?;
    eprintln!(
        "[daemon] code/chunk: {} files → {} chunks ({:.1}ms)",
        p.files.len(), chunks.len(),
        t0.elapsed().as_secs_f64() * 1000.0,
    );

    Ok(serde_json::to_value(&chunks)?)
}
