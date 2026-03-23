//! Code intelligence (v-code) daemon handlers: graph/build.
//!
//! Builds call graphs using MIR-based edge resolution.

use std::path::PathBuf;
use std::sync::Mutex;

/// Serializes concurrent `graph/build` requests so only one runs at a time.
static GRAPH_LOCK: Mutex<()> = Mutex::new(());

pub fn handle_graph_build(
    params: serde_json::Value,
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
    let graph = v_code_intel::graph::CallGraph::build(&chunks);
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
