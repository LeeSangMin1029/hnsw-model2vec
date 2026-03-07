//! Reasoning storage — design decisions and history for code symbols.
//!
//! Stores `ReasonEntry` as JSON files inside `<db>/reasons/<hash>.json`.
//! Human-editable, no separate indexing needed.
//!
//! ## HOW TO EXTEND
//! - Add new fields to `ReasonEntry` (serde will handle missing fields via `default`).
//! - Add query helpers (e.g., search by constraint keyword).

use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// A reasoning entry attached to a code symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasonEntry {
    /// Fully qualified symbol name (e.g., `DagState::update_status`).
    pub symbol: String,
    /// Design decision summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    /// Why this decision was made.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
    /// Active constraints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub constraints: Vec<String>,
    /// Rejected alternatives.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rejected: Vec<String>,
    /// Chronological history of changes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<HistoryItem>,
}

/// A single history entry recording an action on the symbol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryItem {
    /// Action type: create, modify, note, etc.
    pub action: String,
    /// Date string (YYYY-MM-DD).
    pub date: String,
    /// Free-form note.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// What failed (for modify actions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    /// How it was fixed (for modify actions).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
}

// ── Path helpers ──────────────────────────────────────────────────────────

fn reasons_dir(db: &Path) -> PathBuf {
    db.join("reasons")
}

fn symbol_hash(symbol: &str) -> String {
    let mut hasher = DefaultHasher::new();
    symbol.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

fn reason_path(db: &Path, symbol: &str) -> PathBuf {
    reasons_dir(db).join(format!("{}.json", symbol_hash(symbol)))
}

// ── Public API ────────────────────────────────────────────────────────────

/// Save a reason entry to disk.
pub fn save_reason(db: &Path, entry: &ReasonEntry) -> Result<()> {
    let dir = reasons_dir(db);
    fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create reasons dir: {}", dir.display()))?;

    let path = reason_path(db, &entry.symbol);
    let json = serde_json::to_string_pretty(entry)
        .context("failed to serialize reason entry")?;
    fs::write(&path, json)
        .with_context(|| format!("failed to write reason file: {}", path.display()))?;
    Ok(())
}

/// Load a reason entry by symbol name. Returns `None` if not found.
pub fn load_reason(db: &Path, symbol: &str) -> Result<Option<ReasonEntry>> {
    let path = reason_path(db, symbol);
    if !path.exists() {
        return Ok(None);
    }
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read reason file: {}", path.display()))?;
    let entry: ReasonEntry = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse reason file: {}", path.display()))?;
    Ok(Some(entry))
}

/// Delete a reason entry by symbol name. Returns true if deleted.
pub fn delete_reason(db: &Path, symbol: &str) -> Result<bool> {
    let path = reason_path(db, symbol);
    if path.exists() {
        fs::remove_file(&path)
            .with_context(|| format!("failed to delete reason file: {}", path.display()))?;
        Ok(true)
    } else {
        Ok(false)
    }
}

/// List all reason entries in the database.
pub fn list_reasons(db: &Path) -> Result<Vec<ReasonEntry>> {
    let dir = reasons_dir(db);
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = Vec::new();
    for item in fs::read_dir(&dir)
        .with_context(|| format!("failed to read reasons dir: {}", dir.display()))?
    {
        let item = item?;
        let path = item.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            if let Ok(content) = fs::read_to_string(&path) {
                if let Ok(entry) = serde_json::from_str::<ReasonEntry>(&content) {
                    entries.push(entry);
                }
            }
        }
    }

    entries.sort_by(|a, b| a.symbol.cmp(&b.symbol));
    Ok(entries)
}

/// Get today's date as YYYY-MM-DD string.
pub fn today() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    // Simple date calculation (no chrono dependency needed)
    let days = now / 86400;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Format a one-line summary of a reason entry for inline display.
pub fn one_line_summary(entry: &ReasonEntry) -> String {
    if let Some(ref decision) = entry.decision {
        if let Some(ref why) = entry.why {
            format!("{decision} -- {why}")
        } else {
            decision.clone()
        }
    } else if let Some(ref why) = entry.why {
        why.clone()
    } else if !entry.constraints.is_empty() {
        format!("constraints: {}", entry.constraints.join("; "))
    } else {
        "reason recorded".to_owned()
    }
}
