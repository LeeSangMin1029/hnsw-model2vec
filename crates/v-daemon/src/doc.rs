//! Document (v-hnsw) daemon handler: update.
//!
//! Delegates the actual update work to the `v-hnsw update` CLI subprocess,
//! avoiding a cyclic dependency on `v-hnsw-cli`. The daemon contributes its
//! already-loaded embedding model by keeping the subprocess from having to
//! reload it (the subprocess skips daemon delegation since we set
//! `V_HNSW_NO_DAEMON=1`).

use std::path::PathBuf;

use crate::state::DaemonState;

pub fn handle_update(
    params: serde_json::Value,
    state: &mut DaemonState,
) -> anyhow::Result<serde_json::Value> {
    #[derive(serde::Deserialize)]
    struct UpdateParams {
        db: String,
        input: String,
        #[serde(default)]
        exclude: Vec<String>,
    }

    let p: UpdateParams = serde_json::from_value(params)
        .map_err(|e| anyhow::anyhow!("Invalid update params: {e}"))?;
    let db_path = PathBuf::from(&p.db);
    let t0 = std::time::Instant::now();

    // Evict the DB from daemon cache before subprocess mutates it.
    let key = state.evict_db(&db_path)?;

    // Build v-hnsw update command.
    // Look for `v-hnsw` next to current exe, then in PATH.
    let hnsw_name = if cfg!(windows) { "v-hnsw.exe" } else { "v-hnsw" };
    let hnsw_exe = std::env::current_exe()
        .ok()
        .map(|e| e.with_file_name(hnsw_name))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(hnsw_name));

    let mut args = vec![
        "update".to_string(),
        key.to_string_lossy().to_string(),
        p.input.clone(),
    ];
    for ex in &p.exclude {
        args.push("--exclude".to_string());
        args.push(ex.clone());
    }

    let output = std::process::Command::new(&hnsw_exe)
        .args(&args)
        // Prevent the subprocess from trying to delegate back to daemon.
        .env("V_HNSW_NO_DAEMON", "1")
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run v-hnsw update: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("v-hnsw update failed (exit {}): {stderr}", output.status);
    }

    // Reload the updated DB into daemon cache.
    state.reload(&key)?;

    let elapsed_ms = t0.elapsed().as_millis();
    eprintln!("[daemon] Update complete via subprocess ({elapsed_ms}ms)");

    Ok(serde_json::json!({
        "status": "ok",
        "elapsed_ms": elapsed_ms,
    }))
}
