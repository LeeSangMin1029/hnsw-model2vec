//! Symbolic code editing — replace, insert, delete by symbol name.
//!
//! Uses the v-code index DB to locate symbols, then edits the source file directly.
//! Doc comments and attributes above the symbol are automatically included in the range.
//! Also provides line-based editing: `insert_at`, `delete_lines`, `replace_lines`, and `create_file`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use v_code_intel::loader::load_chunks;
use v_code_intel::lsp::find_project_root;
use v_code_intel::parse::CodeChunk;

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
/// `symbol` is matched against chunk names (substring match, same as `v-code symbols`).
/// `file_hint` narrows the search to chunks whose file path ends with the given suffix.
fn locate_symbol(db: &Path, symbol: &str, file_hint: Option<&str>) -> Result<SymbolLocation> {
    let chunks = load_chunks(db)?;

    // Find matching chunks.
    let candidates: Vec<&CodeChunk> = chunks
        .iter()
        .filter(|c| {
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

    // Resolve absolute path: DB stores relative paths from the project root.
    // The DB directory is typically at the project root.
    let db_parent = db.parent().unwrap_or_else(|| Path::new("."));
    let db_parent = if db_parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        db_parent
    };
    let project_root = db_parent
        .canonicalize()
        .unwrap_or_else(|_| db_parent.to_path_buf());
    // Strip Windows UNC prefix (\\?\) that canonicalize() adds.
    let project_root = strip_unc_prefix(&project_root);

    // chunk.file may be absolute (with UNC prefix) or relative.
    let chunk_path = PathBuf::from(&chunk.file);
    let abs_path = if chunk_path.is_absolute() {
        strip_unc_prefix(&chunk_path)
    } else {
        project_root.join(&chunk.file)
    };
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
    let norm_file = norm_file.strip_prefix("//?/").unwrap_or(&norm_file);
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

/// Replace the body of a symbol with new content.
pub fn replace(db: PathBuf, symbol: String, file: Option<String>, body: String) -> Result<()> {
    let loc = locate_symbol(&db, &symbol, file.as_deref())?;
    let content = std::fs::read_to_string(&loc.abs_path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len());
    result.extend_from_slice(&lines[..loc.start_line]);

    // Insert new body lines.
    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();
    result.extend_from_slice(&body_lines);

    if loc.end_line + 1 < lines.len() {
        result.extend_from_slice(&lines[loc.end_line + 1..]);
    }

    let output = if content.ends_with('\n') {
        result.join("\n") + "\n"
    } else {
        result.join("\n")
    };
    std::fs::write(&loc.abs_path, output)?;

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
    let loc = locate_symbol(&db, &symbol, file.as_deref())?;
    let content = std::fs::read_to_string(&loc.abs_path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + 10);
    // Keep everything up to and including the symbol.
    result.extend_from_slice(&lines[..=loc.end_line]);

    // Insert blank line separator + new body.
    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();
    // Add empty line between symbol end and inserted content.
    result.push("");
    result.extend_from_slice(&body_lines);

    if loc.end_line + 1 < lines.len() {
        result.extend_from_slice(&lines[loc.end_line + 1..]);
    }

    let output = if content.ends_with('\n') {
        result.join("\n") + "\n"
    } else {
        result.join("\n")
    };
    std::fs::write(&loc.abs_path, output)?;

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
    let loc = locate_symbol(&db, &symbol, file.as_deref())?;
    let content = std::fs::read_to_string(&loc.abs_path)?;
    let lines: Vec<&str> = content.lines().collect();

    let mut result: Vec<&str> = Vec::with_capacity(lines.len() + 10);
    result.extend_from_slice(&lines[..loc.start_line]);

    // Insert new body + blank line separator.
    let body = body.trim_end();
    let body_lines: Vec<&str> = body.lines().collect();
    result.extend_from_slice(&body_lines);
    result.push("");

    result.extend_from_slice(&lines[loc.start_line..]);

    let output = if content.ends_with('\n') {
        result.join("\n") + "\n"
    } else {
        result.join("\n")
    };
    std::fs::write(&loc.abs_path, output)?;

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
    let loc = locate_symbol(&db, &symbol, file.as_deref())?;
    let content = std::fs::read_to_string(&loc.abs_path)?;
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

    let output = if content.ends_with('\n') {
        result.join("\n") + "\n"
    } else {
        result.join("\n")
    };
    std::fs::write(&loc.abs_path, output)?;

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

/// Resolve a project-relative file path to an absolute path using the DB location.
fn resolve_file_path(db: &Path, file: &str) -> Result<(PathBuf, String)> {
    let root = find_project_root(db)
        .context("Cannot determine project root from DB path")?;
    let abs_path = root.join(file);
    if !abs_path.exists() {
        bail!("File not found: {} (resolved to {})", file, abs_path.display());
    }
    Ok((abs_path, file.to_string()))
}

/// Write lines back to a file, preserving the original trailing-newline style.
fn write_lines(path: &Path, lines: &[&str], had_trailing_newline: bool) -> Result<()> {
    let output = if had_trailing_newline {
        lines.join("\n") + "\n"
    } else {
        lines.join("\n")
    };
    std::fs::write(path, output)?;
    Ok(())
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

fn strip_unc_prefix(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if let Some(stripped) = s.strip_prefix(r"\\?\")
        .or_else(|| s.strip_prefix("//?/"))
    {
        PathBuf::from(stripped)
    } else {
        path.to_path_buf()
    }
}
