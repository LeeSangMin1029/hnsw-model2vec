//! Symbolic code editing — replace, insert, delete by symbol name.
//!
//! Uses the v-code index DB to locate symbols, then edits the source file directly.
//! Doc comments and attributes above the symbol are automatically included in the range.
//! Also provides line-based editing: `insert_at`, `delete_lines`, `replace_lines`, and `create_file`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use v_code_intel::loader::load_chunks;
use v_code_intel::helpers::find_project_root;
use v_code_intel::parse::ParsedChunk;

// ── Symbol location ─────────────────────────────────────────────────────

/// Resolved symbol location in the current source file.
struct SymbolLocation {
    /// Absolute path to the source file.
    abs_path: PathBuf,
    /// Relative path (from DB) for display.
    rel_path: String,
    /// 0-based start line (inclusive).
    start_line: usize,
    /// 0-based end line (inclusive).
    end_line: usize,
}

/// Find a symbol in the DB and resolve its current location in the source file.
///
/// `symbol` is matched against chunk names (exact or `::suffix` match).
/// `file_hint` narrows the search to chunks whose file path ends with the given suffix.
///
/// Uses a HashMap index for O(1) lookup instead of linear scan over all chunks.
fn locate_symbol(db: &Path, symbol: &str, file_hint: Option<&str>) -> Result<SymbolLocation> {
    let chunks = load_chunks(db)?;

    // Build name → indices map for fast lookup.
    let mut name_index: std::collections::HashMap<&str, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        // Index by full name
        name_index.entry(&c.name).or_default().push(i);
        // Index by last segment (after ::) for suffix matching
        if let Some(suffix) = c.name.rsplit("::").next() {
            name_index.entry(suffix).or_default().push(i);
        }
    }

    // O(1) lookup by symbol name
    let candidate_indices = name_index.get(symbol).cloned().unwrap_or_default();

    let candidates: Vec<&ParsedChunk> = candidate_indices
        .into_iter()
        .map(|i| &chunks[i])
        .filter(|c| {
            // Verify the match is exact or ::suffix (index may have collisions from last-segment)
            let name_match = c.name == symbol || c.name.ends_with(&format!("::{symbol}"));
            let file_match = file_hint.is_none_or(|f| c.file.ends_with(f));
            name_match && file_match
        })
        .collect();

    if candidates.is_empty() {
        bail!(
            "Symbol '{symbol}' not found{}",
            file_hint.map_or(String::new(), |f| format!(" in file matching '{f}'"))
        );
    }

    if candidates.len() > 1 {
        let locations: Vec<String> = candidates
            .iter()
            .map(|c| {
                let lines = c
                    .lines
                    .map_or("?".to_owned(), |(s, e)| format!("{s}-{e}"));
                format!("  {} [{}] {}:{lines}", c.name, c.kind, c.file)
            })
            .collect();
        bail!(
            "Ambiguous symbol '{symbol}' — {} matches found. Use --file to narrow:\n{}",
            candidates.len(),
            locations.join("\n")
        );
    }

    let chunk = candidates[0];
    let (start_1, end_1) = chunk
        .lines
        .context("Symbol has no line range in DB (re-index with `v-code add`)")?;

    // Resolve absolute path: prefer CWD over DB parent for worktree support.
    // In worktrees, DB may point to main repo but CWD is the worktree.
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let db_parent = db.parent().unwrap_or_else(|| Path::new("."));
    let db_parent = if db_parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        db_parent
    };

    // Try CWD first (worktree), then DB parent (main repo)
    let chunk_path = PathBuf::from(&chunk.file);
    let abs_path = if chunk_path.is_absolute() {
        v_hnsw_core::strip_unc_prefix_path(&chunk_path)
    } else {
        let cwd_path = cwd.join(&chunk.file);
        if cwd_path.exists() {
            cwd_path
        } else {
            let db_root = db_parent
                .canonicalize()
                .unwrap_or_else(|_| db_parent.to_path_buf());
            let db_root = v_hnsw_core::strip_unc_prefix_path(&db_root);
            db_root.join(&chunk.file)
        }
    };
    let project_root = if cwd.join(&chunk.file).exists() {
        cwd.clone()
    } else {
        db_parent.canonicalize().unwrap_or_else(|_| db_parent.to_path_buf())
    };
    let project_root = v_hnsw_core::strip_unc_prefix_path(&project_root);
    if !abs_path.exists() {
        bail!(
            "Source file not found: {} (resolved to {})",
            chunk.file,
            abs_path.display()
        );
    }
    // Compute relative path for display.
    let norm_root = project_root.to_string_lossy().replace('\\', "/");
    let norm_file = chunk.file.replace('\\', "/");
    let norm_file = v_hnsw_core::strip_unc_prefix(&norm_file);
    let rel_display = norm_file
        .strip_prefix(norm_root.as_str())
        .and_then(|s| s.strip_prefix('/'))
        .unwrap_or(norm_file)
        .to_owned();

    // Convert 1-based (DB) to 0-based.
    let mut start_line = start_1.saturating_sub(1);
    let end_line = end_1.saturating_sub(1);

    // Sanity check: verify the file has enough lines.
    let content = std::fs::read_to_string(&abs_path)
        .with_context(|| format!("Failed to read {}", abs_path.display()))?;
    let lines: Vec<&str> = content.lines().collect();
    if end_line >= lines.len() {
        bail!(
            "DB range L{start_1}-{end_1} exceeds file length ({} lines). \
             File may have changed — run `v-code add` to re-index.",
            lines.len()
        );
    }

    // Extend start upward to include leading doc comments and attributes.
    // This ensures `replace` captures the full definition including docs.
    while start_line > 0 {
        let prev = lines[start_line - 1].trim();
        if prev.starts_with("///")
            || prev.starts_with("//!")
            || prev.starts_with("#[")
            || prev.starts_with("#![")
            || prev.starts_with("/** ")
            || prev.starts_with("* ")
            || prev == "*/"
            // Python/JS/TS decorators and docstrings
            || prev.starts_with('@')
            || prev.starts_with("\"\"\"")
            || prev.starts_with("'''")
            // Go doc comments
            || prev.starts_with("//")
                && start_line >= 2
                && lines.get(start_line.saturating_sub(2))
                    .is_some_and(|l| l.trim().starts_with("//"))
        {
            start_line -= 1;
        } else {
            break;
        }
    }

    Ok(SymbolLocation {
        abs_path,
        rel_path: rel_display,
        start_line,
        end_line,
    })
}

// ── Edit operations ─────────────────────────────────────────────────────

/// Locate a symbol and load the source file in one step.
///
/// Returns lines and metadata needed by all symbol-editing commands.
fn load_symbol_edit(db: &Path, symbol: &str, file: Option<&str>) -> Result<(String, SymbolLocation)> {
    let loc = locate_symbol(db, symbol, file)?;
    let content = std::fs::read_to_string(&loc.abs_path)
        .with_context(|| format!("Failed to read {}", loc.abs_path.display()))?;
    Ok((content, loc))
}

/// Write lines back to a file, preserving the original trailing-newline style.
fn write_lines(path: &Path, lines: &[&str], trailing_nl: bool) -> Result<()> {
    let output = if trailing_nl {
        lines.join("\n") + "\n"
    } else {
        lines.join("\n")
    };
    std::fs::write(path, output)?;
    Ok(())
}

/// Replace the body of a symbol with new content.
pub fn replace(db: PathBuf, symbol: String, file: Option<String>, body: String) -> Result<()> {
    let (content, loc) = load_symbol_edit(&db, &symbol, file.as_deref())?;
    let lines: Vec<&str> = content.lines().collect();

    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len());
    result.extend_from_slice(&lines[..loc.start_line]);
    result.extend_from_slice(&body_lines);
    if loc.end_line + 1 < lines.len() {
        result.extend_from_slice(&lines[loc.end_line + 1..]);
    }

    write_lines(&loc.abs_path, &result, content.ends_with('\n'))?;

    let new_end = loc.start_line + body_lines.len();
    eprintln!(
        "Replaced {} (L{}-{} → L{}-{}) in {}",
        symbol,
        loc.start_line + 1,
        loc.end_line + 1,
        loc.start_line + 1,
        new_end,
        loc.rel_path
    );
    // Print replaced content so agents can verify without a separate Read.
    print_numbered_range(&body_lines, loc.start_line + 1);
    Ok(())
}

/// Insert content after a symbol.
pub fn insert_after(
    db: PathBuf,
    symbol: String,
    file: Option<String>,
    body: String,
) -> Result<()> {
    let (content, loc) = load_symbol_edit(&db, &symbol, file.as_deref())?;
    let lines: Vec<&str> = content.lines().collect();

    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + body_lines.len() + 1);
    result.extend_from_slice(&lines[..=loc.end_line]);
    result.push("");
    result.extend_from_slice(&body_lines);
    if loc.end_line + 1 < lines.len() {
        result.extend_from_slice(&lines[loc.end_line + 1..]);
    }

    write_lines(&loc.abs_path, &result, content.ends_with('\n'))?;

    let insert_start = loc.end_line + 2; // after blank line
    eprintln!(
        "Inserted after {} (after L{}) in {}",
        symbol,
        loc.end_line + 1,
        loc.rel_path
    );
    print_numbered_range(&body_lines, insert_start);
    Ok(())
}

/// Insert content before a symbol.
pub fn insert_before(
    db: PathBuf,
    symbol: String,
    file: Option<String>,
    body: String,
) -> Result<()> {
    let (content, loc) = load_symbol_edit(&db, &symbol, file.as_deref())?;
    let lines: Vec<&str> = content.lines().collect();

    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + body_lines.len() + 1);
    result.extend_from_slice(&lines[..loc.start_line]);
    result.extend_from_slice(&body_lines);
    result.push("");
    result.extend_from_slice(&lines[loc.start_line..]);

    write_lines(&loc.abs_path, &result, content.ends_with('\n'))?;

    eprintln!(
        "Inserted before {} (before L{}) in {}",
        symbol,
        loc.start_line + 1,
        loc.rel_path
    );
    print_numbered_range(&body_lines, loc.start_line + 1);
    Ok(())
}

/// Delete a symbol from the source file.
pub fn delete_symbol(db: PathBuf, symbol: String, file: Option<String>) -> Result<()> {
    let (content, loc) = load_symbol_edit(&db, &symbol, file.as_deref())?;
    let lines: Vec<&str> = content.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len());
    result.extend_from_slice(&lines[..loc.start_line]);

    // Skip blank lines immediately after the deleted symbol to avoid double-spacing.
    let mut after = loc.end_line + 1;
    while after < lines.len() && lines[after].trim().is_empty() {
        after += 1;
    }
    if after < lines.len() {
        result.extend_from_slice(&lines[after..]);
    }

    write_lines(&loc.abs_path, &result, content.ends_with('\n'))?;

    eprintln!(
        "Deleted {} (L{}-{}) from {}",
        symbol,
        loc.start_line + 1,
        loc.end_line + 1,
        loc.rel_path
    );
    Ok(())
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Print lines with 1-based line numbers to stdout, so agents can verify edits.
fn print_numbered_range(lines: &[&str], start_1based: usize) {
    for (i, line) in lines.iter().enumerate() {
        println!("{:>4}│ {line}", start_1based + i);
    }
}

// ── Line-based editing ──────────────────────────────────────────────────

/// Resolve a project-relative file path — prefer CWD (worktree) over DB root.
fn resolve_file_path(db: &Path, file: &str) -> Result<(PathBuf, String)> {
    // Try CWD first (supports worktrees)
    let cwd = std::env::current_dir().unwrap_or_default();
    let cwd_path = cwd.join(file);
    if cwd_path.exists() {
        return Ok((cwd_path, file.to_string()));
    }
    // Fall back to DB-relative
    let root = find_project_root(db)
        .context("Cannot determine project root from DB path")?;
    let abs_path = root.join(file);
    if !abs_path.exists() {
        bail!("File not found: {} (resolved to {})", file, abs_path.display());
    }
    Ok((abs_path, file.to_string()))
}

/// Insert content at a specific 1-based line number (before that line).
pub fn insert_at(db: PathBuf, file: String, line: usize, body: String) -> Result<()> {
    if line == 0 {
        bail!("--line must be >= 1 (1-based)");
    }

    let (abs_path, rel_path) = resolve_file_path(&db, &file)?;
    let content = std::fs::read_to_string(&abs_path)?;
    let trailing_nl = content.ends_with('\n');
    let lines: Vec<&str> = content.lines().collect();

    // line is 1-based; convert to 0-based insertion index.
    // Allow inserting at line == len+1 (appending at end).
    let idx = line - 1;
    if idx > lines.len() {
        bail!(
            "--line {} is past end of file ({} lines)",
            line,
            lines.len()
        );
    }

    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + body_lines.len());
    result.extend_from_slice(&lines[..idx]);
    result.extend_from_slice(&body_lines);
    if idx < lines.len() {
        result.extend_from_slice(&lines[idx..]);
    }

    write_lines(&abs_path, &result, trailing_nl)?;

    eprintln!(
        "Inserted {} line(s) at L{} in {}",
        body_lines.len(),
        line,
        rel_path,
    );
    print_numbered_range(&body_lines, line);
    Ok(())
}

/// Delete a range of lines (1-based, inclusive) from a file.
pub fn delete_lines(db: PathBuf, file: String, start: usize, end: usize) -> Result<()> {
    if start == 0 || end == 0 {
        bail!("--start and --end must be >= 1 (1-based)");
    }
    if start > end {
        bail!("--start ({start}) must be <= --end ({end})");
    }

    let (abs_path, rel_path) = resolve_file_path(&db, &file)?;
    let content = std::fs::read_to_string(&abs_path)?;
    let trailing_nl = content.ends_with('\n');
    let lines: Vec<&str> = content.lines().collect();

    if end > lines.len() {
        bail!(
            "--end {} is past end of file ({} lines)",
            end,
            lines.len()
        );
    }

    let start_idx = start - 1; // 0-based inclusive
    let end_idx = end; // 0-based exclusive (end is inclusive, so +1)

    let mut result: Vec<&str> = Vec::with_capacity(lines.len());
    result.extend_from_slice(&lines[..start_idx]);

    // Skip trailing blank lines after deleted range (same pattern as delete_symbol).
    let mut after = end_idx;
    while after < lines.len() && lines[after].trim().is_empty() {
        after += 1;
    }
    if after < lines.len() {
        result.extend_from_slice(&lines[after..]);
    }

    write_lines(&abs_path, &result, trailing_nl)?;

    eprintln!(
        "Deleted L{}-{} ({} line(s)) from {}",
        start,
        end,
        end - start + 1,
        rel_path,
    );
    Ok(())
}

/// Replace a range of lines (1-based, inclusive) with new content.
pub fn replace_lines(
    db: PathBuf,
    file: String,
    start: usize,
    end: usize,
    body: String,
) -> Result<()> {
    if start == 0 || end == 0 {
        bail!("--start and --end must be >= 1 (1-based)");
    }
    if start > end {
        bail!("--start ({start}) must be <= --end ({end})");
    }

    let (abs_path, rel_path) = resolve_file_path(&db, &file)?;
    let content = std::fs::read_to_string(&abs_path)?;
    let trailing_nl = content.ends_with('\n');
    let lines: Vec<&str> = content.lines().collect();

    if end > lines.len() {
        bail!(
            "--end {} is past end of file ({} lines)",
            end,
            lines.len()
        );
    }

    let start_idx = start - 1; // 0-based inclusive
    let end_idx = end; // 0-based exclusive

    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + body_lines.len());
    result.extend_from_slice(&lines[..start_idx]);
    result.extend_from_slice(&body_lines);
    if end_idx < lines.len() {
        result.extend_from_slice(&lines[end_idx..]);
    }

    write_lines(&abs_path, &result, trailing_nl)?;

    let new_end = start + body_lines.len().saturating_sub(1);
    eprintln!(
        "Replaced L{}-{} -> L{}-{} ({} line(s)) in {}",
        start,
        end,
        start,
        new_end,
        body_lines.len(),
        rel_path,
    );
    print_numbered_range(&body_lines, start);
    Ok(())
}

/// Create a new file at a project-relative path.
///
/// Fails if the file already exists. Parent directories are created automatically.
pub fn create_file(db: PathBuf, file: String, body: String) -> Result<()> {
    let root = find_project_root(&db)
        .context("Cannot determine project root from DB path")?;
    let abs_path = root.join(&file);

    if abs_path.exists() {
        bail!(
            "File already exists: {} (use replace-lines to edit)",
            abs_path.display()
        );
    }

    if let Some(parent) = abs_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directories for {}", parent.display()))?;
    }

    let body = body.trim_end();
    let content = if body.is_empty() {
        String::new()
    } else {
        format!("{body}\n")
    };
    std::fs::write(&abs_path, &content)
        .with_context(|| format!("Failed to write file: {}", abs_path.display()))?;

    let body_lines: Vec<&str> = body.lines().collect();
    eprintln!("Created {} ({} line(s))", file, body_lines.len());
    print_numbered_range(&body_lines, 1);
    Ok(())
}

