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
    /// Whether the server has completed initial indexing.
    ready: bool,
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
            ready: false,
        };

        // LSP initialize handshake
        let init_result = client.request(
            "initialize",
            serde_json::json!({
                "processId": std::process::id(),
                "capabilities": {
                    "textDocument": {
                        "definition": { "dynamicRegistration": false }
                    },
                    "window": {
                        "workDoneProgress": true
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

        // Wait for server to become ready (progress-based or timeout fallback).
        client.wait_for_ready();

        Ok(client)
    }

    /// Wait for the LSP server to finish initial indexing.
    ///
    /// Monitors `$/progress` notifications for "end" tokens. Falls back
    /// to a short timeout if no progress is received.
    fn wait_for_ready(&mut self) {
        let start = std::time::Instant::now();
        let deadline = start + Duration::from_secs(120);
        let mut saw_progress = false;
        let mut completed_sequences = 0u32;
        let mut active_tokens: std::collections::HashSet<String> = std::collections::HashSet::new();
        // Grace period: after all tokens drain, wait for new progress to start.
        // rust-analyzer emits multiple sequences: Fetching → Loading → Indexing.
        let grace = Duration::from_millis(2000);

        eprintln!("[lsp] Waiting for server readiness...");

        while std::time::Instant::now() < deadline {
            let timeout = if active_tokens.is_empty() && saw_progress {
                grace // Short wait for next progress sequence
            } else {
                Duration::from_millis(500)
            };

            match self.responses.recv_timeout(timeout) {
                Ok(msg) => {
                    if msg.get("method").and_then(|m| m.as_str()) == Some("$/progress") {
                        if let Some(params) = msg.get("params") {
                            let token = params
                                .get("token")
                                .and_then(|t| t.as_str().map(String::from)
                                    .or_else(|| t.as_u64().map(|n| n.to_string())))
                                .unwrap_or_default();
                            let value = params.get("value");

                            if let Some(kind) = value.and_then(|v| v.get("kind")).and_then(|k| k.as_str()) {
                                match kind {
                                    "begin" => {
                                        saw_progress = true;
                                        active_tokens.insert(token);
                                        if let Some(title) = value.and_then(|v| v.get("title")).and_then(|t| t.as_str()) {
                                            eprintln!("[lsp] {title}...");
                                        }
                                    }
                                    "end" => {
                                        active_tokens.remove(&token);
                                        if active_tokens.is_empty() {
                                            completed_sequences += 1;
                                            eprintln!("[lsp] Progress sequence {completed_sequences} done ({:.1}s)",
                                                start.elapsed().as_secs_f64());
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if saw_progress && active_tokens.is_empty() {
                        // Grace period expired, no new progress — server is ready.
                        eprintln!("[lsp] Server ready ({:.1}s, {} sequences)",
                            start.elapsed().as_secs_f64(), completed_sequences);
                        self.ready = true;
                        return;
                    }
                    // No progress seen after 5s — server might not support it.
                    if !saw_progress && start.elapsed() > Duration::from_secs(5) {
                        eprintln!("[lsp] No progress support, using 5s fallback");
                        self.ready = true;
                        return;
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }

        eprintln!("[lsp] Readiness timeout ({:.1}s) — proceeding anyway",
            start.elapsed().as_secs_f64());
        self.ready = true;
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

    /// Resolve all call sites in the given chunks to a `CallMap`.
    ///
    /// Opens each unique source file, then queries `textDocument/definition`
    /// for each call site found by tree-sitter in the chunks.
    pub fn resolve_calls(&mut self, chunks: &[CodeChunk], project_root: &Path) -> Result<CallMap> {
        let all_files: std::collections::HashSet<&str> =
            chunks.iter().map(|c| c.file.as_str()).collect();
        self.resolve_calls_for_files(chunks, project_root, &all_files)
    }

    /// Resolve call sites only for chunks whose files are in `target_files`.
    ///
    /// All chunks are still needed for callee name resolution, but only
    /// chunks in `target_files` will have their calls queried via LSP.
    pub fn resolve_calls_for_files(
        &mut self,
        chunks: &[CodeChunk],
        project_root: &Path,
        target_files: &std::collections::HashSet<&str>,
    ) -> Result<CallMap> {
        let t0 = std::time::Instant::now();
        // Open target source files and cache their contents for position lookup
        let mut file_cache: std::collections::HashMap<PathBuf, Vec<String>> =
            std::collections::HashMap::new();
        for file in target_files {
            let fpath = project_root.join(file);
            if file_cache.contains_key(&fpath) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(&fpath) {
                self.did_open(&fpath, &text)?;
                let lines: Vec<String> = text.lines().map(String::from).collect();
                file_cache.insert(fpath, lines);
            }
        }

        eprintln!("[lsp] didOpen {} files in {:.1}s", file_cache.len(), t0.elapsed().as_secs_f64());
        let t1 = std::time::Instant::now();
        // Sequential definition resolution.
        let mut call_map: CallMap = BTreeMap::new();
        let mut total = 0u32;
        let mut received = 0u32;

        for chunk in chunks {
            if !target_files.contains(chunk.file.as_str()) {
                continue;
            }

            let fpath = project_root.join(&chunk.file);
            let caller_name = normalize_chunk_name(&chunk.name);
            let file_lines = file_cache.get(&fpath);

            for call in &chunk.calls {
                if let Some((line, col)) =
                    find_call_position_in_chunk(chunk, call, file_lines.map(Vec::as_slice))
                {
                    total += 1;
                    if let Some(loc) = self.definition(&fpath, line, col) {
                        if let Some(callee) = location_to_chunk_name(&loc, chunks, project_root) {
                            received += 1;
                            call_map
                                .entry(caller_name.clone())
                                .or_default()
                                .push(callee);
                        }
                    }
                }
            }
        }

        // Deduplicate callees.
        for callees in call_map.values_mut() {
            callees.sort();
            callees.dedup();
        }

        eprintln!("[lsp] definitions resolved in {:.1}s", t1.elapsed().as_secs_f64());
        eprintln!("[graph] LSP resolved {} caller entries ({} files, {}/{} queries answered)",
            call_map.len(), target_files.len(), received, total);
        Ok(call_map)
    }

    /// Send a `textDocument/definition` request and parse the result.
    fn definition(&mut self, path: &Path, line: u32, col: u32) -> Option<Location> {
        let uri = path_to_uri(path);
        let result = self
            .request(
                "textDocument/definition",
                serde_json::json!({
                    "textDocument": { "uri": uri },
                    "position": { "line": line, "character": col },
                }),
            )
            .ok()??;
        parse_definition_result(&result).ok().flatten()
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

        if let Ok(msg) = serde_json::from_slice::<serde_json::Value>(&body)
            && tx.send(msg).is_err() {
                return; // receiver dropped
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
    // Check project markers first (fast)
    for marker in ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt"] {
        if root.join(marker).exists() {
            return true;
        }
    }
    // Fall back to scanning for .py files
    for dir in [root.to_path_buf(), root.join("src"), root.join("lib"), root.join("app")] {
        if let Ok(entries) = std::fs::read_dir(dir)
            && entries
                .flatten()
                .any(|e| e.path().extension().is_some_and(|ext| ext == "py"))
            {
                return true;
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
    let mut s = abs.to_string_lossy().replace('\\', "/");
    // Strip Windows extended-length prefix (\\?\)
    if let Some(stripped) = s.strip_prefix("//?/") {
        s = stripped.to_string();
    }
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
    Some(PathBuf::from(path_str.replace('/', std::path::MAIN_SEPARATOR_STR)))
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
/// Searches the source lines within the chunk's line range for the call pattern
/// (e.g. `.method(` or `func(`). Returns 0-indexed (line, col) for LSP.
fn find_call_position_in_chunk(
    chunk: &CodeChunk,
    call_name: &str,
    file_lines: Option<&[String]>,
) -> Option<(u32, u32)> {
    let (start, end) = chunk.lines?;
    let short = call_name.rsplit("::").next().unwrap_or(call_name);
    let patterns = [format!(".{short}("), format!("{short}("), short.to_string()];

    if let Some(lines) = file_lines {
        // Search within chunk's line range (1-indexed → 0-indexed)
        let from = start.saturating_sub(1).min(lines.len());
        let to = end.min(lines.len());

        if from >= to {
            return Some((start.saturating_sub(1) as u32, 0));
        }

        for (i, line) in lines[from..to].iter().enumerate() {
            for pat in &patterns {
                if let Some(col) = line.find(pat.as_str()) {
                    let actual_col = if pat.starts_with('.') {
                        col + 1 // point at method name, not the dot
                    } else {
                        col
                    };
                    return Some(((from + i) as u32, actual_col as u32));
                }
            }
        }
    }

    // Fallback: return chunk start line, col 0
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

    let mut rel_str = rel_path.to_string_lossy().replace('\\', "/");
    // Strip leading "./" prefix if present
    if let Some(stripped) = rel_str.strip_prefix("./") {
        rel_str = stripped.to_string();
    }
    let target_line = (loc.line + 1) as usize; // LSP is 0-indexed, chunks are 1-indexed

    // Find the chunk that contains this definition
    for chunk in chunks {
        let chunk_file = chunk.file.replace('\\', "/").trim_start_matches("./").to_string();
        if let Some((start, end)) = chunk.lines
            && chunk_file == rel_str && target_line >= start && target_line <= end {
                return Some(normalize_chunk_name(&chunk.name));
            }
    }

    // Fallback: match by file only, find closest chunk
    let mut best: Option<(&CodeChunk, usize)> = None;
    for chunk in chunks {
        let chunk_file = chunk.file.replace('\\', "/").trim_start_matches("./").to_string();
        if let Some((start, _)) = chunk.lines
            && chunk_file == rel_str && target_line >= start {
                let dist = target_line - start;
                if best.as_ref().is_none_or(|(_, d)| dist < *d) {
                    best = Some((chunk, dist));
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
