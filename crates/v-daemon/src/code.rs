//! Code intelligence (v-code) daemon handler: graph/build.
//!
//! Builds call graphs using tree-sitter + rustdoc type enrichment.

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
    let project_root = v_code_intel::helpers::find_project_root(&db)
        .unwrap_or_else(|| PathBuf::from("."));

    // Try loading cached rustdoc types, or generate if not available.
    let rustdoc = v_code_intel::rustdoc::load_cached(&db)
        .or_else(|| {
            let types = v_code_intel::rustdoc::generate_and_load(&project_root)?;
            v_code_intel::rustdoc::save_to_cache(&db, &project_root);
            Some(types)
        });

    let graph = v_code_intel::graph::CallGraph::build_with_rustdoc(&chunks, rustdoc.as_ref());
    let _ = graph.save(&db);

    let has_rustdoc = rustdoc.is_some();
    eprintln!(
        "[daemon] Graph built: {} nodes, rustdoc={}",
        graph.len(),
        has_rustdoc,
    );

    Ok(serde_json::json!({
        "status": "ok",
        "nodes": graph.len(),
        "rustdoc": has_rustdoc,
    }))
}
