//! Generate rustdoc JSON and rebuild graph cache with compiler type info.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Run `cargo rustdoc` for all workspace lib crates, cache results, and rebuild graph.
pub fn run(db: PathBuf) -> Result<()> {
    let project_root = v_code_intel::helpers::find_project_root(&db)
        .context("Cannot find project root (Cargo.toml) from DB path")?;

    let start = std::time::Instant::now();

    // Generate rustdoc JSON (with per-crate progress + timeout).
    let types = v_code_intel::rustdoc::generate_and_load(&project_root)
        .context("rustdoc generation failed — is nightly toolchain installed?")?;

    eprintln!(
        "[rustdoc] {} fn returns, {} method owners, {} field types",
        types.fn_return_types.len(),
        types.method_owner.len(),
        types.field_types.len(),
    );

    // Cache JSON files into DB for future `add` runs.
    v_code_intel::rustdoc::save_to_cache(&db, &project_root);

    // Rebuild graph.bin.
    let chunks = v_code_intel::loader::load_chunks(&db)?;
    let graph = v_code_intel::graph::CallGraph::build_full(&chunks);
    let _ = graph.save(&db);

    eprintln!(
        "[rustdoc] Done in {:.1}s — graph rebuilt with compiler type info.",
        start.elapsed().as_secs_f64()
    );

    Ok(())
}
