//! Measure chunk_files on entire project — simulates v-code add.
//!
//! Usage: cargo run --example measure_chunk_all --release -- [project_path]

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

fn walkdir(root: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut result = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy();
                if name != "target" && name != ".git" && name != "node_modules" {
                    stack.push(path);
                }
            } else {
                result.push(path);
            }
        }
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root_str = args.get(1).map(|s| s.as_str()).unwrap_or(".");
    println!("=== chunk_files Full Project Measurement ===");
    println!("  project: {root_str}\n");

    // Load RA
    let (_, c0, _) = mem_mb();
    println!("[1] Loading RA...");
    let t = std::time::Instant::now();
    let root = std::path::Path::new(root_str);
    let mut ra = match v_lsp::instance::RaInstance::spawn(root) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed: {e}");
            std::process::exit(1);
        }
    };
    let (_, c1, _) = mem_mb();
    println!("  RA loaded: {:.1}s, Commit: {:.0}MB → {:.0}MB (+{:.0}MB)\n",
        t.elapsed().as_secs_f64(), c0, c1, c1 - c0);

    // Collect all .rs files by walking the project directory
    let mut file_map_keys: Vec<String> = Vec::new();
    for entry in walkdir(root) {
        let rel = entry.strip_prefix(root).unwrap_or(&entry);
        let rel_str = rel.to_string_lossy().replace('\\', "/");
        if rel_str.ends_with(".rs")
            && !rel_str.contains("/target/")
            && !rel_str.contains("/.git/")
            && (rel_str.starts_with("lib/") || rel_str.starts_with("src/")
                || rel_str.starts_with("crates/") || rel_str.starts_with("tests/")
                || rel_str.starts_with("benches/"))
        {
            file_map_keys.push(rel_str);
        }
    }
    file_map_keys.sort();

    println!("[2] Found {} project .rs files\n", file_map_keys.len());

    // chunk_files — this is what v-code add does
    let (w_pre, c_pre, _) = mem_mb();
    println!("[3] Running chunk_files on ALL {} files...", file_map_keys.len());
    let t = std::time::Instant::now();
    let chunks = ra.chunk_files(&file_map_keys);
    let elapsed = t.elapsed().as_secs_f64();
    let (w_post, c_post, peak) = mem_mb();

    println!("  Done: {:.1}s, {} chunks", elapsed, chunks.len());
    println!("  WSS:    {:.0}MB → {:.0}MB ({:+.0}MB)", w_pre, w_post, w_post - w_pre);
    println!("  Commit: {:.0}MB → {:.0}MB ({:+.0}MB)", c_pre, c_post, c_post - c_pre);
    println!("  Peak WSS: {:.0}MB\n", peak);

    // GC
    println!("[4] GC after chunk_files");
    let (_, c_pre_gc, _) = mem_mb();
    ra.garbage_collect();
    let (_, c_post_gc, _) = mem_mb();
    println!("  Commit: {:.0}MB → {:.0}MB ({:+.0}MB)\n", c_pre_gc, c_post_gc, c_post_gc - c_pre_gc);

    // Summary
    let (wss_final, commit_final, peak_final) = mem_mb();
    println!("=== Summary ===");
    println!("  Files:         {}", file_map_keys.len());
    println!("  Chunks:        {}", chunks.len());
    println!("  Time:          {:.1}s", elapsed);
    println!("  Baseline:      {:.0}MB commit", c1);
    println!("  Final:         {:.0}MB commit", commit_final);
    println!("  Delta:         {:+.0}MB", commit_final - c1);
    println!("  Peak WSS:      {:.0}MB", peak_final);

    // Count calls extracted
    let total_calls: usize = chunks.iter().map(|c| c.calls.len()).sum();
    let fns = chunks.iter().filter(|c| c.kind == "function").count();
    println!("  Functions:     {}", fns);
    println!("  Total calls:   {}", total_calls);
}
