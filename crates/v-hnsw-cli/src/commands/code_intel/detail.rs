//! `v-hnsw detail` — view and manage reasoning entries for code symbols.
//!
//! ## HOW TO EXTEND
//! - Add `--export` flag to dump all reasons as a single JSON array.
//! - Add `--import` flag to bulk-load reasons from a JSON file.

use anyhow::Result;
use std::path::PathBuf;

use super::reason::{
    self, HistoryItem, ReasonEntry,
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
    pub delete: bool,
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
        delete,
    } = params;

    // Delete mode
    if delete {
        return run_delete(&db, &symbol);
    }

    // Check if any mutation flag is set
    let has_mutation = add.is_some()
        || decision.is_some()
        || why.is_some()
        || constraint.is_some()
        || rejected.is_some()
        || failure.is_some()
        || fix.is_some();

    if has_mutation {
        run_mutate(&db, &symbol, add, decision, why, constraint, rejected, failure, fix)
    } else {
        run_show(&db, &symbol)
    }
}

/// Show reasoning for a symbol.
fn run_show(db: &PathBuf, symbol: &str) -> Result<()> {
    match reason::load_reason(db, symbol)? {
        Some(entry) => print_entry(&entry),
        None => println!("No reasoning found for \"{symbol}\"."),
    }
    Ok(())
}

/// Delete reasoning for a symbol.
fn run_delete(db: &PathBuf, symbol: &str) -> Result<()> {
    if reason::delete_reason(db, symbol)? {
        println!("Deleted reasoning for \"{symbol}\".");
    } else {
        println!("No reasoning found for \"{symbol}\".");
    }
    Ok(())
}

/// Add or update reasoning for a symbol.
#[expect(clippy::too_many_arguments)]
fn run_mutate(
    db: &PathBuf,
    symbol: &str,
    add: Option<String>,
    decision: Option<String>,
    why: Option<String>,
    constraint: Option<String>,
    rejected: Option<String>,
    failure: Option<String>,
    fix: Option<String>,
) -> Result<()> {
    let mut entry = reason::load_reason(db, symbol)?
        .unwrap_or_else(|| ReasonEntry {
            symbol: symbol.to_owned(),
            decision: None,
            why: None,
            constraints: Vec::new(),
            rejected: Vec::new(),
            history: Vec::new(),
        });

    let today = reason::today();
    let mut changes = Vec::new();

    // --add: general note
    if let Some(ref note) = add {
        entry.history.push(HistoryItem {
            action: "note".to_owned(),
            date: today.clone(),
            note: Some(note.clone()),
            failure: None,
            fix: None,
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
            });
        } else {
            changes.push(format!("constraint already exists: {c}"));
        }
    }

    // --rejected: append
    if let Some(ref r) = rejected {
        if !entry.rejected.contains(r) {
            entry.rejected.push(r.clone());
            changes.push(format!("added rejected alternative: {r}"));
            entry.history.push(HistoryItem {
                action: "rejected".to_owned(),
                date: today.clone(),
                note: Some(r.clone()),
                failure: None,
                fix: None,
            });
        } else {
            changes.push(format!("rejected alternative already exists: {r}"));
        }
    }

    // --failure / --fix: failure-fix record
    if failure.is_some() || fix.is_some() {
        entry.history.push(HistoryItem {
            action: "modify".to_owned(),
            date: today,
            note: None,
            failure: failure.clone(),
            fix: fix.clone(),
        });
        if let Some(ref f) = failure {
            changes.push(format!("recorded failure: {f}"));
        }
        if let Some(ref f) = fix {
            changes.push(format!("recorded fix: {f}"));
        }
    }

    reason::save_reason(db, &entry)?;

    println!("Updated \"{symbol}\":");
    for c in &changes {
        println!("  + {c}");
    }
    Ok(())
}

/// Pretty-print a reason entry.
fn print_entry(entry: &ReasonEntry) {
    println!("=== {} ===\n", entry.symbol);

    if let Some(ref d) = entry.decision {
        println!("  Decision: {d}");
    }
    if let Some(ref w) = entry.why {
        println!("  Why:      {w}");
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
            println!("    x {r}");
        }
    }

    if !entry.history.is_empty() {
        println!("\n  History:");
        for h in &entry.history {
            let mut line = format!("    [{date}] {action}", date = h.date, action = h.action);
            if let Some(ref note) = h.note {
                line.push_str(&format!(" - {note}"));
            }
            if let Some(ref failure) = h.failure {
                line.push_str(&format!(" | failure: {failure}"));
            }
            if let Some(ref fix) = h.fix {
                line.push_str(&format!(" | fix: {fix}"));
            }
            println!("{line}");
        }
    }
    println!();
}
