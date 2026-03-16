//! Call-site accuracy verification.
//!
//! Loads the call graph once and checks every callee edge's call-site line
//! against the source file to verify the callee name actually appears there.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use v_hnsw_cli::commands::db_config::DbConfig;

/// Verify call-site accuracy for all function chunks in the database.
pub fn run(db: PathBuf) -> Result<()> {
    let start = std::time::Instant::now();
    let graph = super::intel::load_or_build_graph(&db)?;

    // Resolve project root from DB config (input_path) for source file lookup.
    let project_root = DbConfig::load(&db)
        .ok()
        .and_then(|c| c.input_path)
        .map(PathBuf::from);

    // Count how many chunks share each short name (for ambiguity detection).
    let mut name_counts: HashMap<&str, usize> = HashMap::new();
    for (i, kind) in graph.kinds.iter().enumerate() {
        if kind == "function" {
            let short = graph.names[i].rsplit("::").next().unwrap_or(&graph.names[i]);
            *name_counts.entry(short).or_default() += 1;
        }
    }

    let mut file_cache: HashMap<String, Vec<String>> = HashMap::new();

    let mut total = 0usize;
    let mut ok_total = 0usize;
    let mut wrong_total = 0usize;
    let mut ambig_total = 0usize;
    let mut ambig_wrong = 0usize;

    let mut wrong_unique: Vec<(String, Vec<String>)> = Vec::new();
    let mut wrong_ambig: Vec<(String, Vec<String>)> = Vec::new();

    let n = graph.names.len();
    for i in 0..n {
        if graph.kinds[i] != "function" {
            continue;
        }

        let short_name = graph.names[i].rsplit("::").next().unwrap_or(&graph.names[i]);
        let is_ambiguous = name_counts.get(short_name).copied().unwrap_or(0) > 1;

        let call_sites = &graph.call_sites[i];
        let mut errors = Vec::new();

        for &(callee_idx, call_line) in call_sites {
            if call_line == 0 {
                continue;
            }
            let ci = callee_idx as usize;
            if ci >= n {
                continue;
            }
            if graph.kinds[ci] != "function" {
                continue;
            }

            let callee_short = graph.names[ci]
                .rsplit("::")
                .next()
                .unwrap_or(&graph.names[ci]);

            let resolved = resolve_path(&graph.files[i], project_root.as_deref());
            let lines = file_cache
                .entry(resolved.clone())
                .or_insert_with(|| load_lines(&resolved));

            total += 1;
            if is_ambiguous {
                ambig_total += 1;
            }

            let ln = call_line as usize;
            if ln == 0 || ln > lines.len() {
                wrong_total += 1;
                if is_ambiguous {
                    ambig_wrong += 1;
                }
                errors.push(format!(
                    "    {} → L{}: OUT OF RANGE",
                    graph.names[ci], call_line
                ));
                continue;
            }

            let src = &lines[ln - 1];
            if src.contains(callee_short) {
                ok_total += 1;
            } else {
                wrong_total += 1;
                if is_ambiguous {
                    ambig_wrong += 1;
                }
                let truncated: String = src.trim().chars().take(70).collect();
                errors.push(format!(
                    "    {} → L{}: '{}'",
                    graph.names[ci], call_line, truncated
                ));
            }
        }

        if !errors.is_empty() {
            let label = format!(
                "{}:{}) {}",
                graph.files[i],
                graph.lines[i].map_or(0, |l| l.0),
                graph.names[i],
            );
            if is_ambiguous {
                wrong_ambig.push((label, errors));
            } else {
                wrong_unique.push((label, errors));
            }
        }
    }

    // Summary.
    let uniq_total = total - ambig_total;
    let uniq_ok = ok_total - (ambig_total - ambig_wrong);
    let uniq_wrong = wrong_total - ambig_wrong;

    println!("{}", "=".repeat(60));
    println!("  Call-site line verification (callee edges only)");
    println!("{}", "=".repeat(60));
    if total > 0 {
        let pct = ok_total as f64 / total as f64 * 100.0;
        println!("  All edges:    {total}  correct={ok_total} ({pct:.1}%)  wrong={wrong_total}");
    }
    if uniq_total > 0 {
        let pct = uniq_ok as f64 / uniq_total as f64 * 100.0;
        println!("  Unique names: {uniq_total}  correct={uniq_ok} ({pct:.1}%)  wrong={uniq_wrong}");
    }
    println!("  Ambiguous:    {ambig_total}  wrong={ambig_wrong} (same-name functions, unreliable)");
    println!("  Elapsed:      {:.1}ms", start.elapsed().as_secs_f64() * 1000.0);

    wrong_unique.sort_by(|a, b| a.0.cmp(&b.0));
    wrong_ambig.sort_by(|a, b| a.0.cmp(&b.0));

    if !wrong_unique.is_empty() {
        println!(
            "\n--- Wrong lines: unique names ({} functions) ---",
            wrong_unique.len()
        );
        for (label, errs) in &wrong_unique {
            println!("\n  {label}:");
            for e in errs {
                println!("  {e}");
            }
        }
    }
    if !wrong_ambig.is_empty() {
        println!(
            "\n--- Wrong lines: ambiguous names ({} functions, unreliable) ---",
            wrong_ambig.len()
        );
        for (label, errs) in &wrong_ambig {
            println!("\n  {label}:");
            for e in errs {
                println!("  {e}");
            }
        }
    }

    Ok(())
}

/// Resolve a relative chunk path to an absolute path using the project root.
fn resolve_path(rel: &str, project_root: Option<&Path>) -> String {
    if let Some(root) = project_root {
        let full = root.join(rel);
        if full.exists() {
            return full.to_string_lossy().into_owned();
        }
    }
    // Fallback: try relative from cwd.
    rel.to_owned()
}

fn load_lines(path: &str) -> Vec<String> {
    std::fs::read_to_string(path)
        .map(|s| s.lines().map(String::from).collect())
        .unwrap_or_default()
}
