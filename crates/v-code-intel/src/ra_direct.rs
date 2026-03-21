//! Call graph edge resolution via in-process RA call hierarchy API.
//!
//! Uses both `outgoing_calls` (caller→callees) and `incoming_calls`
//! (callee→callers) to maximize edge coverage. The union of both
//! directions catches edges that either direction alone might miss.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::parse::ParsedChunk;

/// Result of RA call hierarchy resolution.
pub struct RaEdges {
    /// (src_chunk_idx, tgt_chunk_idx, call_site_line_1based)
    pub edges: Vec<(usize, usize, u32)>,
}

/// Build call graph edges using RA's call hierarchy API (both directions).
pub fn resolve_via_ra(chunks: &[ParsedChunk]) -> RaEdges {
    let t0 = Instant::now();

    let workspace_root =
        std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));

    let ra = match v_lsp::instance::RaInstance::spawn(&workspace_root) {
        Ok(ra) => {
            eprintln!(
                "    [ra] in-process instance loaded ({:.1}s)",
                t0.elapsed().as_secs_f64()
            );
            ra
        }
        Err(e) => {
            eprintln!("    [ra] in-process spawn failed: {e}");
            return RaEdges { edges: Vec::new() };
        }
    };

    let file_line_to_chunk = build_file_line_map(chunks);
    let mut edge_set: HashSet<(usize, usize, u32)> = HashSet::new();

    // ── Pass 1: outgoing_calls (caller → callees) ──────────────────
    let t_out = Instant::now();
    let mut out_total = 0usize;
    let mut out_resolved = 0usize;
    let mut out_external = 0usize;
    let mut out_errors = 0usize;
    let mut out_fn_count = 0usize;

    for (src, chunk) in chunks.iter().enumerate() {
        if chunk.kind != "function" {
            continue;
        }
        let Some((start_line, _)) = chunk.lines else {
            continue;
        };
        let col = find_fn_name_col(chunk);

        let callees = match ra.outgoing_calls(&chunk.file, start_line.saturating_sub(1) as u32, col) {
            Ok(c) => c,
            Err(_) => {
                out_errors += 1;
                continue;
            }
        };

        out_fn_count += 1;
        out_total += callees.len();

        for callee in &callees {
            if callee.file.is_empty() {
                out_external += 1;
                continue;
            }
            if let Some(tgt) =
                find_narrowest_chunk(&file_line_to_chunk, &callee.file, callee.line + 1)
            {
                if tgt != src {
                    let call_line = callee.call_site_lines.first().copied().unwrap_or(0);
                    edge_set.insert((src, tgt, call_line));
                    out_resolved += 1;
                }
            } else {
                out_external += 1;
            }
        }
    }

    eprintln!(
        "    [ra] outgoing: {} fns, {} callees, {} resolved, {} ext, {} err ({:.0}ms)",
        out_fn_count, out_total, out_resolved, out_external, out_errors,
        t_out.elapsed().as_secs_f64() * 1000.0,
    );

    // ── Pass 2: incoming_calls (callee → callers) ──────────────────
    let t_in = Instant::now();
    let mut in_total = 0usize;
    let mut in_resolved = 0usize;
    let mut in_external = 0usize;
    let mut in_errors = 0usize;
    let mut in_fn_count = 0usize;

    for (tgt, chunk) in chunks.iter().enumerate() {
        if chunk.kind != "function" {
            continue;
        }
        let Some((start_line, _)) = chunk.lines else {
            continue;
        };
        let col = find_fn_name_col(chunk);

        let callers = match ra.incoming_calls(&chunk.file, start_line.saturating_sub(1) as u32, col) {
            Ok(c) => c,
            Err(_) => {
                in_errors += 1;
                continue;
            }
        };

        in_fn_count += 1;
        in_total += callers.len();

        for caller in &callers {
            if caller.file.is_empty() {
                in_external += 1;
                continue;
            }
            if let Some(src) =
                find_narrowest_chunk(&file_line_to_chunk, &caller.file, caller.line + 1)
            {
                if src != tgt {
                    let call_line = caller.call_site_lines.first().copied().unwrap_or(0);
                    edge_set.insert((src, tgt, call_line));
                    in_resolved += 1;
                }
            } else {
                in_external += 1;
            }
        }
    }

    eprintln!(
        "    [ra] incoming: {} fns, {} callers, {} resolved, {} ext, {} err ({:.0}ms)",
        in_fn_count, in_total, in_resolved, in_external, in_errors,
        t_in.elapsed().as_secs_f64() * 1000.0,
    );

    let edges: Vec<(usize, usize, u32)> = edge_set.into_iter().collect();
    eprintln!(
        "    [ra] total unique edges: {} ({:.1}s total)",
        edges.len(),
        t0.elapsed().as_secs_f64(),
    );

    RaEdges { edges }
}

/// Build reverse map: file → [(start_line, end_line, chunk_idx)].
fn build_file_line_map(chunks: &[ParsedChunk]) -> HashMap<String, Vec<(u32, u32, usize)>> {
    let mut map: HashMap<String, Vec<(u32, u32, usize)>> = HashMap::new();
    for (idx, chunk) in chunks.iter().enumerate() {
        if let Some((start, end)) = chunk.lines {
            map.entry(chunk.file.clone())
                .or_default()
                .push((start as u32, end as u32, idx));
        }
    }
    map
}

/// Find the chunk index whose range contains `line`, preferring the narrowest range.
fn find_narrowest_chunk(
    map: &HashMap<String, Vec<(u32, u32, usize)>>,
    file: &str,
    line: u32,
) -> Option<usize> {
    let ranges = map.get(file)?;
    ranges
        .iter()
        .filter(|(s, e, _)| line >= *s && line <= *e)
        .min_by_key(|(s, e, _)| e - s)
        .map(|(_, _, idx)| *idx)
}

/// Find the column offset of the function name on the chunk's start line.
fn find_fn_name_col(chunk: &ParsedChunk) -> u32 {
    let leaf = chunk.name.rsplit("::").next().unwrap_or(&chunk.name);

    if let Some(sig) = &chunk.signature {
        let pattern = format!("fn {leaf}");
        if let Some(pos) = sig.find(&pattern) {
            return (pos + 3) as u32; // skip "fn "
        }
    }
    7
}
