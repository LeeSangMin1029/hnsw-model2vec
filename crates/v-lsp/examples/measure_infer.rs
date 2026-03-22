//! Measure per-function infer_body memory cost.
//!
//! Tests our hypothesis: 1-2 file incremental outgoing_calls
//! does NOT cause memory explosion.
//!
//! Usage: cargo run --example measure_infer --release

/// Memory info: (WSS_MB, Commit_MB, Peak_WSS_MB)
fn mem_mb() -> (f64, f64, f64) {
    #[cfg(windows)]
    {
        use std::mem::MaybeUninit;
        #[repr(C)]
        struct Pmc {
            cb: u32, pf: u32, peak_wss: usize, wss: usize,
            qpp: usize, qp: usize, qnpp: usize, qnp: usize,
            pfu: usize, ppfu: usize,
        }
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn K32GetProcessMemoryInfo(h: *mut std::ffi::c_void, p: *mut Pmc, cb: u32) -> i32;
        }
        unsafe {
            let mut pmc = MaybeUninit::<Pmc>::zeroed().assume_init();
            pmc.cb = std::mem::size_of::<Pmc>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc, pmc.cb) != 0 {
                let mb = 1024.0 * 1024.0;
                return (pmc.wss as f64 / mb, pmc.pfu as f64 / mb, pmc.peak_wss as f64 / mb);
            }
        }
        (0.0, 0.0, 0.0)
    }
    #[cfg(not(windows))]
    { (0.0, 0.0, 0.0) }
}

fn rss_mb() -> f64 { mem_mb().0 }

struct Measurement {
    label: String,
    wss_before: f64,
    wss_after: f64,
    commit_before: f64,
    commit_after: f64,
    elapsed_ms: f64,
    callees: usize,
}

fn measure_outgoing(
    ra: &v_lsp::instance::RaInstance,
    file: &str,
    line: u32,
    col: u32,
    label: &str,
) -> Measurement {
    let (wss_before, commit_before, _) = mem_mb();
    let t = std::time::Instant::now();
    let callees = match ra.outgoing_calls(file, line, col) {
        Ok(calls) => calls.len(),
        Err(e) => {
            eprintln!("  ERR {label}: {e}");
            0
        }
    };
    let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (wss_after, commit_after, _) = mem_mb();
    Measurement {
        label: label.to_owned(),
        wss_before, wss_after,
        commit_before, commit_after,
        elapsed_ms, callees,
    }
}

fn print_measurement(m: &Measurement) {
    let wss_d = m.wss_after - m.wss_before;
    let commit_d = m.commit_after - m.commit_before;
    println!(
        "  {:<42} {:>6.1}ms  WSS:{:>+7.1}MB  Commit:{:>+7.1}MB  (now {:.0}MB)  callees={}",
        m.label, m.elapsed_ms, wss_d, commit_d, m.commit_after, m.callees,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root_str = args.get(1).map(|s| s.as_str()).unwrap_or(".");
    println!("=== infer_body Memory Measurement ===");
    println!("  project: {root_str}\n");

    // ── Phase 1: Load RA ──
    let (wss0, commit0, _) = mem_mb();
    println!("[Phase 1] Loading RA workspace...");
    let t = std::time::Instant::now();
    let root = std::path::Path::new(root_str);
    let mut ra = match v_lsp::instance::RaInstance::spawn(root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to load RA: {e}");
            std::process::exit(1);
        }
    };
    let (wss1, commit1, peak1) = mem_mb();
    println!(
        "  RA loaded: {:.1}s\n  WSS: {:.0}MB → {:.0}MB (+{:.0}MB)\n  Commit: {:.0}MB → {:.0}MB (+{:.0}MB)\n  Peak WSS: {:.0}MB\n",
        t.elapsed().as_secs_f64(), wss0, wss1, wss1 - wss0, commit0, commit1, commit1 - commit0, peak1,
    );

    // ── Select targets based on project ──
    let is_qdrant = root_str.contains("qdrant");

    // (file, line_0based, col, label)
    let (light, medium, heavy, incr_file, incr_line): (Vec<_>, Vec<_>, Vec<_>, &str, u32) = if is_qdrant {
        (
            vec![
                // small pure functions
                ("lib/segment/src/index/hnsw_index/build_cache.rs", 13, 11, "BuildCacheKey::new (3 lines)"),
                ("lib/segment/src/index/hnsw_index/config.rs", 55, 11, "HnswGraphConfig::get_config_path (3 lines)"),
                ("lib/segment/src/index/hnsw_index/entry_points.rs", 34, 11, "EntryPoints::new (5 lines)"),
            ],
            vec![
                // medium: collection ops
                ("lib/segment/src/index/hnsw_index/config.rs", 34, 11, "HnswGraphConfig::new (20 lines)"),
                ("lib/segment/src/index/hnsw_index/config.rs", 59, 11, "HnswGraphConfig::load (4 lines)"),
            ],
            vec![
                // heavy: lots of deps, async, complex types
                ("lib/collection/src/collection/collection_ops.rs", 33, 18, "update_params_from_diff (async, heavy)"),
                ("lib/collection/src/collection/collection_ops.rs", 210, 18, "handle_replica_changes (async, 80+ lines)"),
                ("lib/segment/src/index/hnsw_index/hnsw.rs", 135, 11, "HnswIndex::open (heavy, 100+ lines)"),
            ],
            "lib/segment/src/index/hnsw_index/config.rs", 34,
        )
    } else {
        (
            vec![
                ("crates/v-code-intel/src/graph.rs", 18, 7, "current_rss_mb (14 lines, ext=0)"),
                ("crates/v-code-intel/src/index_tables.rs", 209, 3, "owner_leaf (5 lines, ext=0)"),
                ("crates/v-code-intel/src/index_tables.rs", 106, 7, "extract_leaf_type (21 lines, ext=0)"),
            ],
            vec![
                ("crates/v-code-intel/src/graph.rs", 180, 11, "CallGraph::build_full (43 lines, ext=3)"),
                ("crates/v-code-intel/src/lsp_client.rs", 41, 7, "collect_types_via_ra_filtered (64 lines, ext=5)"),
            ],
            vec![
                ("crates/v-daemon/src/server.rs", 15, 7, "server::run (165 lines, ext=9)"),
                ("crates/v-code/src/commands/ingest.rs", 53, 7, "chunk_via_daemon (76 lines, ext=8)"),
                ("crates/v-lsp/src/instance.rs", 220, 11, "hover_batch_parallel (71 lines, ext=7)"),
            ],
            "crates/v-code-intel/src/graph.rs", 180,
        )
    };

    // ── Phase 2: Light ──
    println!("[Phase 2] Light functions");
    for (file, line, col, label) in &light {
        let m = measure_outgoing(&ra, file, *line, *col, label);
        print_measurement(&m);
    }

    println!("\n[Phase 3] GC after light");
    let (w0, c0, _) = mem_mb();
    ra.garbage_collect();
    let (w1, c1, _) = mem_mb();
    println!("  GC: WSS {:.0}>{:.0} ({:+.0})  Commit {:.0}>{:.0} ({:+.0})\n", w0, w1, w1-w0, c0, c1, c1-c0);

    // ── Phase 4: Medium ──
    println!("[Phase 4] Medium functions");
    for (file, line, col, label) in &medium {
        let m = measure_outgoing(&ra, file, *line, *col, label);
        print_measurement(&m);
    }

    println!("\n[Phase 5] GC after medium");
    let (w0, c0, _) = mem_mb();
    ra.garbage_collect();
    let (w1, c1, _) = mem_mb();
    println!("  GC: WSS {:.0}>{:.0} ({:+.0})  Commit {:.0}>{:.0} ({:+.0})\n", w0, w1, w1-w0, c0, c1, c1-c0);

    // ── Phase 6: Heavy ──
    println!("[Phase 6] Heavy functions");
    for (file, line, col, label) in &heavy {
        let m = measure_outgoing(&ra, file, *line, *col, label);
        print_measurement(&m);
    }

    println!("\n[Phase 7] GC after heavy");
    let (w0, c0, _) = mem_mb();
    ra.garbage_collect();
    let (w1, c1, _) = mem_mb();
    println!("  GC: WSS {:.0}>{:.0} ({:+.0})  Commit {:.0}>{:.0} ({:+.0})\n", w0, w1, w1-w0, c0, c1, c1-c0);

    // ── Phase 8: Cache hit ──
    println!("[Phase 8] Re-call heavy (cache hit?)");
    for (file, line, col, label) in &heavy {
        let m = measure_outgoing(&ra, file, *line, *col, &format!("CACHED: {label}"));
        print_measurement(&m);
    }

    // ── Phase 9: Incremental ──
    println!("\n[Phase 9] Incremental: apply_change + outgoing_calls");
    let test_path = if is_qdrant {
        std::path::Path::new(root_str).join(incr_file)
    } else {
        std::path::PathBuf::from(incr_file)
    };
    let test_file_str = incr_file;
    let (_, commit_pre, _) = mem_mb();

    let content = std::fs::read_to_string(&test_path).unwrap_or_default();
    let modified = format!("{content}\n// trivial change for measurement");

    let t = std::time::Instant::now();
    ra.update_file(test_file_str, modified);
    let apply_ms = t.elapsed().as_secs_f64() * 1000.0;
    let (_, commit_post, _) = mem_mb();
    println!("  apply_change: {:.1}ms, Commit: {:.0}MB > {:.0}MB ({:+.0}MB)", apply_ms, commit_pre, commit_post, commit_post - commit_pre);

    let m = measure_outgoing(&ra, test_file_str, incr_line, 11, "incremental outgoing_calls");
    print_measurement(&m);

    ra.update_file(test_file_str, content);

    // ── Summary ──
    println!("\n[Phase 10] Final GC");
    let (w0, c0, _) = mem_mb();
    ra.garbage_collect();
    let (w1, c1, _) = mem_mb();
    println!("  GC: WSS {:.0}>{:.0} ({:+.0})  Commit {:.0}>{:.0} ({:+.0})", w0, w1, w1-w0, c0, c1, c1-c0);

    let (wss_final, commit_final, peak_final) = mem_mb();
    println!("\n=== Summary ===");
    println!("  Baseline Commit: {:.0}MB", commit1);
    println!("  Final Commit:    {:.0}MB", commit_final);
    println!("  Commit delta:    {:+.0}MB", commit_final - commit1);
    println!("  Peak WSS:        {:.0}MB", peak_final);
    println!("  Final WSS:       {:.0}MB", wss_final);
}
