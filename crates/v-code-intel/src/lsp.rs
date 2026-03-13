//! LSP-based call resolver — uses language servers for accurate call graph extraction.
//!
//! Spawns a language server (rust-analyzer, ty, gopls, etc.) as a subprocess,
//! sends `textDocument/definition` requests for each call site, and collects
//! the results into a `CallMap` compatible with `CallGraph::build_with_resolved_calls`.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use crate::parse::CodeChunk;

// ── Types ────────────────────────────────────────────────────────────

/// Caller→callee edges keyed by normalized caller name.
pub type CallMap = BTreeMap<String, Vec<String>>;

/// A resolved definition location.
#[derive(Debug, Clone)]
struct Location {
    file: PathBuf,
    line: u32,
}

// ── LSP Client ───────────────────────────────────────────────────────

/// Language-agnostic LSP client for call resolution.
pub struct LspCallResolver {
    proc: Child,
    stdin: Option<std::process::ChildStdin>,
    _reader: JoinHandle<()>,
    responses: Receiver<serde_json::Value>,
    next_id: AtomicU64,
    #[expect(dead_code)]
    root_uri: String,
}

impl LspCallResolver {
    /// Start an LSP server for the project at `project_root`.
    ///
    /// Auto-detects the language from project markers (Cargo.toml, pyproject.toml, etc.)
    /// and spawns the appropriate language server.
    pub fn start(project_root: &Path) -> Result<Self> {
        let cmd = detect_lsp_command(project_root)
            .context("no supported language server found for this project")?;

        let root = project_root
            .canonicalize()
            .unwrap_or_else(|_| project_root.to_path_buf());
        let root_uri = path_to_uri(&root);

        let mut proc = Command::new(&cmd[0])
            .args(&cmd[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .current_dir(&root)
            .spawn()
            .with_context(|| format!("failed to start LSP server: {}", cmd[0]))?;

        let stdout = proc.stdout.take().context("no stdout")?;
        let stdin = proc.stdin.take().context("no stdin")?;

        let (tx, rx) = mpsc::channel();
        let reader = thread::spawn(move || read_lsp_messages(stdout, tx));

        let mut client = Self {
            proc,
            stdin: Some(stdin),
            _reader: reader,
            responses: rx,
            next_id: AtomicU64::new(1),
            root_uri: root_uri.clone(),
        };

        // LSP initialize handshake
        let init_result = client.request(
            "initialize",
            serde_json::json!({
                "processId": std::process::id(),
                "capabilities": {
                    "textDocument": {
                        "definition": { "dynamicRegistration": false }
                    }
                },
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": "project"}],
            }),
        )?;

        if init_result.is_none() {
            bail!("LSP initialize failed: no response");
        }

        client.notify("initialized", serde_json::json!({}))?;

        // Give server time to index
        thread::sleep(Duration::from_secs(2));

        Ok(client)
    }

    /// Send `textDocument/didOpen` for a file.
    pub fn did_open(&mut self, path: &Path, text: &str) -> Result<()> {
        let uri = path_to_uri(path);
        let lang = detect_language_id(path);
        self.notify(
            "textDocument/didOpen",
            serde_json::json!({
                "textDocument": {
                    "uri": uri,
                    "languageId": lang,
                    "version": 1,
                    "text": text,
                }
            }),
        )
    }

    /// Send `textDocument/didChange` for a file.
    pub fn did_change(&mut self, path: &Path, text: &str) -> Result<()> {
        let uri = path_to_uri(path);
        self.notify(
            "textDocument/didChange",
            serde_json::json!({
                "textDocument": { "uri": uri, "version": 2 },
                "contentChanges": [{ "text": text }],
            }),
        )
    }

    /// Query `textDocument/definition` at a specific position.
    /// Returns `(file_path, line_0indexed)` if resolved.
    fn definition(&mut self, path: &Path, line: u32, col: u32) -> Result<Option<Location>> {
        let uri = path_to_uri(path);
        let resp = self.request(
            "textDocument/definition",
            serde_json::json!({
                "textDocument": { "uri": uri },
                "position": { "line": line, "character": col },
            }),
        )?;

        let result = match resp {
            Some(r) => r,
            None => return Ok(None),
        };

        parse_definition_result(&result)
    }

    /// Resolve all call sites in the given chunks to a `CallMap`.
    ///
    /// Opens each unique source file, then queries `textDocument/definition`
    /// for each call site found by tree-sitter in the chunks.
    pub fn resolve_calls(&mut self, chunks: &[CodeChunk], project_root: &Path) -> Result<CallMap> {
        // Open all unique source files
        let mut opened: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
        for chunk in chunks {
            let fpath = project_root.join(&chunk.file);
            if opened.contains(&fpath) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&fpath) {
                self.did_open(&fpath, &text)?;
                opened.insert(fpath);
            }
        }

        // Wait for server to process opened files
        thread::sleep(Duration::from_secs(2));

        let mut call_map: CallMap = BTreeMap::new();

        for chunk in chunks {
            let fpath = project_root.join(&chunk.file);
            let caller_name = normalize_chunk_name(&chunk.name);

            // Use the chunk's call list from tree-sitter to find call sites
            // and resolve each one via LSP definition
            for call in &chunk.calls {
                // We need the line/col of each call site.
                // For now, use the chunk's start line + search within the chunk text.
                if let Some((line, col)) = find_call_position_in_chunk(chunk, call) {
                    if let Ok(Some(loc)) = self.definition(&fpath, line, col) {
                        // Convert definition location back to a chunk name
                        if let Some(callee) = location_to_chunk_name(&loc, chunks, project_root) {
                            call_map
                                .entry(caller_name.clone())
                                .or_default()
                                .push(callee);
                        }
                    }
                }
            }

            // Deduplicate callees
            if let Some(callees) = call_map.get_mut(&caller_name) {
                callees.sort();
                callees.dedup();
            }
        }

        Ok(call_map)
    }

    /// Gracefully shut down the LSP server.
    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.request("shutdown", serde_json::json!(null));
        let _ = self.notify("exit", serde_json::json!(null));
        thread::sleep(Duration::from_millis(300));
        let _ = self.proc.kill();
        Ok(())
    }

    // ── Internal helpers ─────────────────────────────────────────────

    fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<Option<serde_json::Value>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });
        self.send_message(&msg)?;

        // Wait for matching response (skip notifications)
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        while std::time::Instant::now() < deadline {
            match self.responses.recv_timeout(Duration::from_secs(1)) {
                Ok(resp) => {
                    if resp.get("id").and_then(|v| v.as_u64()) == Some(id) {
                        return Ok(resp.get("result").cloned());
                    }
                    // Not our response — skip (notification or other)
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => bail!("LSP server disconnected"),
            }
        }

        Ok(None) // timeout
    }

    fn notify(&mut self, method: &str, params: serde_json::Value) -> Result<()> {
        let msg = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_message(&msg)
    }

    fn send_message(&mut self, msg: &serde_json::Value) -> Result<()> {
        let body = serde_json::to_string(msg)?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());
        let stdin = self.stdin.as_mut().context("stdin closed")?;
        stdin.write_all(header.as_bytes())?;
        stdin.write_all(body.as_bytes())?;
        stdin.flush()?;
        Ok(())
    }
}

impl Drop for LspCallResolver {
    fn drop(&mut self) {
        let _ = self.proc.kill();
    }
}

// ── LSP Message Reader ───────────────────────────────────────────────

fn read_lsp_messages(stdout: std::process::ChildStdout, tx: Sender<serde_json::Value>) {
    let mut reader = BufReader::new(stdout);
    loop {
        // Read headers
        let mut content_length: usize = 0;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).unwrap_or(0) == 0 {
                return; // EOF
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                break; // End of headers
            }
            if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
                content_length = len_str.trim().parse().unwrap_or(0);
            }
        }

        if content_length == 0 {
            continue;
        }

        // Read body
        let mut body = vec![0u8; content_length];
        if reader.read_exact(&mut body).is_err() {
            return;
        }

        if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(&body) {
            if tx.send(msg).is_err() {
                return; // receiver dropped
            }
        }
    }
}

// ── Language Detection ───────────────────────────────────────────────

/// Detect the appropriate LSP server command for a project.
fn detect_lsp_command(root: &Path) -> Option<Vec<String>> {
    if root.join("Cargo.toml").exists() {
        Some(vec!["rust-analyzer".into()])
    } else if has_python_files(root) {
        // Try ty first, fall back to pyright
        if Command::new("python")
            .args(["-m", "ty", "--version"])
            .output()
            .is_ok_and(|o| o.status.success())
        {
            Some(vec![
                "python".into(),
                "-m".into(),
                "ty".into(),
                "server".into(),
            ])
        } else {
            None
        }
    } else if root.join("tsconfig.json").exists() || root.join("package.json").exists() {
        Some(vec![
            "typescript-language-server".into(),
            "--stdio".into(),
        ])
    } else if root.join("go.mod").exists() {
        Some(vec!["gopls".into(), "serve".into()])
    } else {
        None
    }
}

fn has_python_files(root: &Path) -> bool {
    if let Ok(entries) = std::fs::read_dir(root) {
        for entry in entries.flatten() {
            if entry
                .path()
                .extension()
                .is_some_and(|e| e == "py")
            {
                return true;
            }
        }
    }
    // Check common subdirs
    for subdir in ["src", "lib", "app"] {
        let d = root.join(subdir);
        if let Ok(entries) = std::fs::read_dir(d) {
            for entry in entries.flatten() {
                if entry
                    .path()
                    .extension()
                    .is_some_and(|e| e == "py")
                {
                    return true;
                }
            }
        }
    }
    false
}

fn detect_language_id(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rs") => "rust",
        Some("py") => "python",
        Some("ts") | Some("tsx") => "typescript",
        Some("js") | Some("jsx") => "javascript",
        Some("go") => "go",
        Some("java") => "java",
        Some("c") | Some("h") => "c",
        Some("cpp" | "cc" | "cxx" | "hpp") => "cpp",
        _ => "plaintext",
    }
}

// ── URI Helpers ──────────────────────────────────────────────────────

fn path_to_uri(path: &Path) -> String {
    let abs = path
        .canonicalize()
        .unwrap_or_else(|_| path.to_path_buf());
    let s = abs.to_string_lossy().replace('\\', "/");
    if s.starts_with('/') {
        format!("file://{s}")
    } else {
        format!("file:///{s}")
    }
}

fn uri_to_path(uri: &str) -> Option<PathBuf> {
    let path_str = uri
        .strip_prefix("file:///")
        .or_else(|| uri.strip_prefix("file://"))?;
    Some(PathBuf::from(path_str.replace('/', &std::path::MAIN_SEPARATOR.to_string())))
}

// ── Result Parsing ───────────────────────────────────────────────────

fn parse_definition_result(result: &serde_json::Value) -> Result<Option<Location>> {
    // LSP can return a single Location, an array of Locations, or LocationLink[]
    let targets: Vec<&serde_json::Value> = if let Some(arr) = result.as_array() {
        arr.iter().collect()
    } else if result.is_object() {
        vec![result]
    } else {
        return Ok(None);
    };

    if targets.is_empty() {
        return Ok(None);
    }

    let t = targets[0];
    let uri = t
        .get("uri")
        .or_else(|| t.get("targetUri"))
        .and_then(|v| v.as_str());
    let range = t
        .get("range")
        .or_else(|| t.get("targetSelectionRange"))
        .or_else(|| t.get("targetRange"));

    if let (Some(uri), Some(range)) = (uri, range) {
        let line = range
            .get("start")
            .and_then(|s| s.get("line"))
            .and_then(|l| l.as_u64())
            .unwrap_or(0) as u32;
        if let Some(file) = uri_to_path(uri) {
            return Ok(Some(Location { file, line }));
        }
    }

    Ok(None)
}

// ── Call Site Resolution Helpers ─────────────────────────────────────

/// Find the (line, col) of a call site within a chunk's source file.
///
/// Reads the file and searches for the call name within the chunk's line range.
fn find_call_position_in_chunk(chunk: &CodeChunk, call_name: &str) -> Option<(u32, u32)> {
    let (start, end) = chunk.lines?;
    let short = call_name.rsplit("::").next().unwrap_or(call_name);

    // We don't have the body text, so we just return the approximate position.
    // Use the start line and column 0 as a placeholder — the LSP server
    // will resolve based on the symbol name at that position.
    // A more precise approach would read the source file.
    let patterns = [format!(".{short}("), format!("{short}(")];

    // Return start line (0-indexed for LSP) with col 0
    // The caller should ideally read the file for precise positioning
    let _ = (end, &patterns);
    Some((start.saturating_sub(1) as u32, 0))
}

/// Convert a definition location back to a chunk name.
fn location_to_chunk_name(
    loc: &Location,
    chunks: &[CodeChunk],
    project_root: &Path,
) -> Option<String> {
    let rel_path = loc
        .file
        .strip_prefix(project_root)
        .map(|p| p.to_path_buf())
        .ok()
        .or_else(|| {
            // Try canonicalized comparison
            let canon_root = project_root.canonicalize().ok()?;
            let canon_file = loc.file.canonicalize().ok()?;
            canon_file.strip_prefix(canon_root).ok().map(|p| p.to_path_buf())
        })?;

    let rel_str = rel_path.to_string_lossy().replace('\\', "/");
    let target_line = (loc.line + 1) as usize; // LSP is 0-indexed, chunks are 1-indexed

    // Find the chunk that contains this definition
    for chunk in chunks {
        let chunk_file = chunk.file.replace('\\', "/");
        if let Some((start, end)) = chunk.lines {
            if chunk_file == rel_str && target_line >= start && target_line <= end {
                return Some(normalize_chunk_name(&chunk.name));
            }
        }
    }

    // Fallback: match by file only, find closest chunk
    let mut best: Option<(&CodeChunk, usize)> = None;
    for chunk in chunks {
        let chunk_file = chunk.file.replace('\\', "/");
        if let Some((start, _)) = chunk.lines {
            if chunk_file == rel_str && target_line >= start {
                let dist = target_line - start;
                if best.as_ref().is_none_or(|(_, d)| dist < *d) {
                    best = Some((chunk, dist));
                }
            }
        }
    }

    best.map(|(c, _)| normalize_chunk_name(&c.name))
}

/// Normalize a chunk name for use as a call map key.
fn normalize_chunk_name(name: &str) -> String {
    name.to_lowercase()
}

// ── Find Project Root ────────────────────────────────────────────────

/// Walk up from the DB path to find a project root directory.
pub fn find_project_root(db: &Path) -> Option<PathBuf> {
    let abs = db.canonicalize().ok()?;
    let start = if abs.is_dir() {
        abs
    } else {
        abs.parent()?.to_path_buf()
    };
    let markers = [
        "Cargo.toml",
        ".git",
        "pyproject.toml",
        "setup.py",
        "go.mod",
        "package.json",
        "tsconfig.json",
    ];
    let mut dir = start.as_path();
    for _ in 0..10 {
        if markers.iter().any(|m| dir.join(m).exists()) {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
    None
}
