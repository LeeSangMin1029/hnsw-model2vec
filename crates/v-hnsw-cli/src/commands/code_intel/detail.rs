//! `v-hnsw detail` — view and manage reasoning entries for code symbols.
//!
//! ## HOW TO EXTEND
//! - Add `--export` flag to dump all reasons as a single JSON array.
//! - Add `--import` flag to bulk-load reasons from a JSON file.

use anyhow::Result;
use std::path::PathBuf;

use v_hnsw_intel::reason::{
    self, HistoryItem, ReasonEntry, RejectedAlternative,
};

/// Parameters for the detail command (parsed from CLI).
pub struct DetailParams {
    pub db: PathBuf,
    pub symbol: String,
    pub add: Option<String>,
    pub decision: Option<String>,
    pub why: Option<String>,
    pub constraint: Option<String>,
    pub rejected: Option<String>,
    pub failure: Option<String>,
    pub fix: Option<String>,
    pub root_cause: Option<String>,
    pub reject_reason: Option<String>,
    pub reject_condition: Option<String>,
    pub resolve: bool,
    pub invalidate: Option<String>,
    pub show_all: bool,
    pub delete: bool,
    /// Source file path for location tracking (#1).
    pub file_path: Option<String>,
    /// Line range "start:end" for location tracking (#1).
    pub line_range: Option<String>,
    /// Related symbol for cross-referencing (#6).
    pub relate: Option<String>,
}

/// Run the detail command.
pub fn run_detail(params: DetailParams) -> Result<()> {
    let DetailParams {
        db,
        symbol,
        add,
        decision,
        why,
        constraint,
        rejected,
        failure,
        fix,
        root_cause,
        reject_reason,
        reject_condition,
        resolve,
        invalidate,
        show_all,
        delete,
        file_path,
        line_range,
        relate,
    } = params;

    // Delete mode
    if delete {
        return run_delete(&db, &symbol);
    }

    // Resolve / invalidate modes
    if resolve {
        return run_resolve(&db, &symbol);
    }
    if let Some(ref inv_reason) = invalidate {
        return run_invalidate(&db, &symbol, inv_reason);
    }

    // Check if any mutation flag is set
    let has_mutation = add.is_some()
        || decision.is_some()
        || why.is_some()
        || constraint.is_some()
        || rejected.is_some()
        || failure.is_some()
        || fix.is_some()
        || file_path.is_some()
        || line_range.is_some()
        || relate.is_some();

    if has_mutation {
        run_mutate(
            &db, &symbol, add, decision, why, constraint, rejected,
            failure, fix, root_cause, reject_reason, reject_condition,
            file_path, line_range, relate,
        )
    } else {
        run_show_filtered(&db, &symbol, show_all)
    }
}

/// Show reasoning for a symbol, optionally filtering resolved failures.
fn run_show_filtered(db: &std::path::Path, symbol: &str, show_all: bool) -> Result<()> {
    match reason::load_reason(db, symbol)? {
        Some(entry) => {
            print_entry(&entry, show_all);
            // Show bidirectional related references (#6)
            let related = reason::find_related_reasons(db, symbol)?;
            if !related.is_empty() {
                println!("  Referenced by:");
                for r in &related {
                    println!("    <- {} ({})", r.symbol, reason::one_line_summary(r));
                }
                println!();
            }
        }
        None => println!("No reasoning found for \"{symbol}\"."),
    }
    Ok(())
}

/// Mark the last unresolved failure as resolved.
fn run_resolve(db: &std::path::Path, symbol: &str) -> Result<()> {
    let mut entry = reason::load_reason(db, symbol)?
        .ok_or_else(|| anyhow::anyhow!("No reasoning found for \"{symbol}\""))?;

    if reason::resolve_last_failure(&mut entry) {
        reason::save_reason(db, &entry)?;
        println!("Resolved last failure for \"{symbol}\".");
    } else {
        println!("No unresolved failure found for \"{symbol}\".");
    }
    Ok(())
}

/// Invalidate the last unresolved failure with a reason.
fn run_invalidate(db: &std::path::Path, symbol: &str, inv_reason: &str) -> Result<()> {
    let mut entry = reason::load_reason(db, symbol)?
        .ok_or_else(|| anyhow::anyhow!("No reasoning found for \"{symbol}\""))?;

    if reason::invalidate_last_failure(&mut entry, inv_reason) {
        reason::save_reason(db, &entry)?;
        println!("Invalidated last failure for \"{symbol}\".");
    } else {
        println!("No unresolved failure found for \"{symbol}\".");
    }
    Ok(())
}

/// Delete reasoning for a symbol.
fn run_delete(db: &std::path::Path, symbol: &str) -> Result<()> {
    if reason::delete_reason(db, symbol)? {
        println!("Deleted reasoning for \"{symbol}\".");
    } else {
        println!("No reasoning found for \"{symbol}\".");
    }
    Ok(())
}

/// Parse a line range string "start:end" into `(usize, usize)`.
fn parse_line_range(s: &str) -> Option<(usize, usize)> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 2 {
        if let (Ok(start), Ok(end)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
            return Some((start, end));
        }
    }
    None
}

/// Add or update reasoning for a symbol.
#[expect(clippy::too_many_arguments)]
fn run_mutate(
    db: &std::path::Path,
    symbol: &str,
    add: Option<String>,
    decision: Option<String>,
    why: Option<String>,
    constraint: Option<String>,
    rejected: Option<String>,
    failure: Option<String>,
    fix: Option<String>,
    root_cause: Option<String>,
    reject_reason: Option<String>,
    reject_condition: Option<String>,
    file_path: Option<String>,
    line_range: Option<String>,
    relate: Option<String>,
) -> Result<()> {
    let mut entry = reason::load_reason(db, symbol)?
        .unwrap_or_else(|| ReasonEntry {
            symbol: symbol.to_owned(),
            decision: None,
            why: None,
            constraints: Vec::new(),
            rejected: Vec::new(),
            history: Vec::new(),
            file_path: None,
            line_range: None,
            related_symbols: Vec::new(),
        });

    let today = reason::today();
    let commit = reason::get_git_head();
    let mut changes = Vec::new();

    // --file-path / --line-range: update location info (#1)
    if let Some(ref fp) = file_path {
        entry.file_path = Some(fp.clone());
        changes.push(format!("set file_path: {fp}"));
    }
    if let Some(ref lr) = line_range {
        if let Some(range) = parse_line_range(lr) {
            entry.line_range = Some(range);
            changes.push(format!("set line_range: {}-{}", range.0, range.1));
        } else {
            changes.push(format!("invalid line_range format (expected start:end): {lr}"));
        }
    }

    // --relate: add related symbol (#6) with bidirectional reference
    if let Some(ref rel) = relate {
        if !entry.related_symbols.contains(rel) {
            entry.related_symbols.push(rel.clone());
            changes.push(format!("added related symbol: {rel}"));

            // Bidirectional: also update the related symbol's entry
            if let Ok(Some(mut related_entry)) = reason::load_reason(db, rel) {
                if !related_entry.related_symbols.contains(&symbol.to_owned()) {
                    related_entry.related_symbols.push(symbol.to_owned());
                    let _ = reason::save_reason(db, &related_entry);
                }
            }
        } else {
            changes.push(format!("related symbol already exists: {rel}"));
        }
    }

    // --add: general note
    if let Some(ref note) = add {
        entry.history.push(HistoryItem {
            action: "note".to_owned(),
            date: today.clone(),
            note: Some(note.clone()),
            failure: None,
            fix: None,
            root_cause: None,
            resolved: false,
            commit: commit.clone(),
        });
        changes.push(format!("added note: {note}"));
    }

    // --decision / --why: set or update design decision
    if let Some(ref d) = decision {
        entry.decision = Some(d.clone());
        changes.push(format!("set decision: {d}"));
    }
    if let Some(ref w) = why {
        entry.why = Some(w.clone());
        changes.push(format!("set why: {w}"));
    }
    if decision.is_some() || why.is_some() {
        let is_first = entry.history.is_empty()
            || !entry.history.iter().any(|h| h.action == "create" || h.action == "decision");
        entry.history.push(HistoryItem {
            action: if is_first { "create" } else { "decision" }.to_owned(),
            date: today.clone(),
            note: decision.as_ref().map(|d| {
                if let Some(ref w) = why {
                    format!("{d} -- {w}")
                } else {
                    d.clone()
                }
            }),
            failure: None,
            fix: None,
            root_cause: None,
            resolved: false,
            commit: commit.clone(),
        });
    }

    // --constraint: append
    if let Some(ref c) = constraint {
        if !entry.constraints.contains(c) {
            entry.constraints.push(c.clone());
            changes.push(format!("added constraint: {c}"));
            entry.history.push(HistoryItem {
                action: "constraint".to_owned(),
                date: today.clone(),
                note: Some(c.clone()),
                failure: None,
                fix: None,
                root_cause: None,
                resolved: false,
                commit: commit.clone(),
            });
        } else {
            changes.push(format!("constraint already exists: {c}"));
        }
    }

    // --rejected: append (with structured reason/condition)
    if let Some(ref r) = rejected {
        let alt = RejectedAlternative {
            approach: r.clone(),
            reason: reject_reason.clone(),
            condition: reject_condition.clone(),
        };
        if !entry.rejected.iter().any(|existing| existing.approach == alt.approach) {
            entry.rejected.push(alt);
            changes.push(format!("added rejected alternative: {r}"));
            entry.history.push(HistoryItem {
                action: "rejected".to_owned(),
                date: today.clone(),
                note: Some(r.clone()),
                failure: None,
                fix: None,
                root_cause: None,
                resolved: false,
                commit: commit.clone(),
            });
        } else {
            changes.push(format!("rejected alternative already exists: {r}"));
        }
    }

    // --failure / --fix: failure-fix record (with optional root_cause)
    if failure.is_some() || fix.is_some() {
        entry.history.push(HistoryItem {
            action: "modify".to_owned(),
            date: today,
            note: None,
            failure: failure.clone(),
            fix: fix.clone(),
            root_cause: root_cause.clone(),
            resolved: false,
            commit,
        });
        if let Some(ref f) = failure {
            changes.push(format!("recorded failure: {f}"));
        }
        if let Some(ref f) = fix {
            changes.push(format!("recorded fix: {f}"));
        }
        if let Some(ref rc) = root_cause {
            changes.push(format!("root cause: {rc}"));
        }
    }

    reason::save_reason(db, &entry)?;

    println!("Updated \"{symbol}\":");
    for c in &changes {
        println!("  + {c}");
    }
    Ok(())
}

/// List all reasoning entries in the database.
#[cfg(test)]
pub fn run_list(db: &std::path::Path) -> Result<Vec<ReasonEntry>> {
    reason::list_reasons(db)
}

/// Pretty-print a reason entry, optionally filtering resolved history items.
fn print_entry(entry: &ReasonEntry, show_all: bool) {
    println!("=== {} ===\n", entry.symbol);

    if let Some(ref d) = entry.decision {
        println!("  Decision: {d}");
    }
    if let Some(ref w) = entry.why {
        println!("  Why:      {w}");
    }

    // Location info (#1)
    if let Some(ref fp) = entry.file_path {
        if let Some((start, end)) = entry.line_range {
            println!("  Location: {fp}:{start}-{end}");
        } else {
            println!("  Location: {fp}");
        }
    }

    if !entry.constraints.is_empty() {
        println!("\n  Constraints:");
        for c in &entry.constraints {
            println!("    - {c}");
        }
    }

    if !entry.rejected.is_empty() {
        println!("\n  Rejected alternatives:");
        for r in &entry.rejected {
            let mut line = format!("    x {}", r.approach);
            if let Some(ref reason) = r.reason {
                line.push_str(&format!(" -- {reason}"));
            }
            if let Some(ref cond) = r.condition {
                line.push_str(&format!(" (when: {cond})"));
            }
            println!("{line}");
        }
    }

    // Related symbols (#6)
    if !entry.related_symbols.is_empty() {
        println!("\n  Related symbols:");
        for s in &entry.related_symbols {
            println!("    ~ {s}");
        }
    }

    if !entry.history.is_empty() {
        println!("\n  History:");
        for h in &entry.history {
            // Skip resolved items unless --all is set
            if h.resolved && !show_all {
                continue;
            }

            let mut line = format!("    [{date}] {action}", date = h.date, action = h.action);
            if h.resolved {
                line.push_str(" (resolved)");
            }
            // Show commit hash (#5)
            if let Some(ref c) = h.commit {
                line.push_str(&format!(" @{c}"));
            }
            if let Some(ref note) = h.note {
                line.push_str(&format!(" - {note}"));
            }
            if let Some(ref failure) = h.failure {
                line.push_str(&format!(" | failure: {failure}"));
            }
            if let Some(ref fix) = h.fix {
                line.push_str(&format!(" | fix: {fix}"));
            }
            if let Some(ref rc) = h.root_cause {
                line.push_str(&format!(" | root_cause: {rc}"));
            }
            println!("{line}");
        }
    }
    println!();
}

