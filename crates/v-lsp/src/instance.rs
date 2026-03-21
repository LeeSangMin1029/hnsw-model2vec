//! In-process rust-analyzer instance — load workspace into `RootDatabase`,
//! query hover for type resolution. No external process, no TCP/LSP overhead.
//!
//! The `RaInstance` owns the RA database in memory and can be shared across
//! multiple callers (v-code, agents) via the mmap IPC layer in `shm.rs`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ra_ap_ide::{
    Analysis, AnalysisHost, CallHierarchyConfig, CallItem, FileId, FilePosition, FileRange,
    FindAllRefsConfig, GotoDefinitionConfig, HoverConfig, HoverDocFormat, LineCol, LineIndex,
    NavigationTarget, Query, SubstTyLen, TextSize,
};
use ra_ap_load_cargo::{LoadCargoConfig, ProcMacroServerChoice, load_workspace_at};
use ra_ap_project_model::CargoConfig;
use ra_ap_vfs::Vfs;

use crate::error::{LspError, Result};

// ── resolve_call diagnostic counters ──────────────────────────────────
use std::sync::atomic::{AtomicUsize, Ordering as AtOrd};

static RC_TOTAL: AtomicUsize = AtomicUsize::new(0);
static RC_OK: AtomicUsize = AtomicUsize::new(0);
static RC_FILE_MISS: AtomicUsize = AtomicUsize::new(0);
static RC_LINE_ZERO: AtomicUsize = AtomicUsize::new(0);
static RC_PATTERN_MISS: AtomicUsize = AtomicUsize::new(0);
static RC_OFFSET_MISS: AtomicUsize = AtomicUsize::new(0);
static RC_GOTO_MISS: AtomicUsize = AtomicUsize::new(0);
static RC_NAV_MISS: AtomicUsize = AtomicUsize::new(0);
static RC_EMPTY_NAV: AtomicUsize = AtomicUsize::new(0);

use std::sync::Mutex;
use std::collections::HashMap as StdHashMap;

static EMPTY_NAV_FILES: Mutex<Option<StdHashMap<String, usize>>> = Mutex::new(None);

fn record_empty_nav_file(file: &str) {
    let mut guard = EMPTY_NAV_FILES.lock().unwrap_or_else(|e| e.into_inner());
    let map = guard.get_or_insert_with(StdHashMap::new);
    *map.entry(file.to_owned()).or_default() += 1;
}

/// Get top files contributing to empty_nav, sorted by count descending.
pub fn empty_nav_file_stats() -> Vec<(String, usize)> {
    let guard = EMPTY_NAV_FILES.lock().unwrap_or_else(|e| e.into_inner());
    let mut result: Vec<_> = guard.as_ref().map_or_else(Vec::new, |m| m.iter().map(|(k,v)| (k.clone(), *v)).collect());
    result.sort_by(|a, b| b.1.cmp(&a.1));
    result
}

/// Reset all resolve_call counters and return their previous values.
pub fn resolve_call_stats() -> ResolveCallStats {
    ResolveCallStats {
        total: RC_TOTAL.swap(0, AtOrd::Relaxed),
        ok: RC_OK.swap(0, AtOrd::Relaxed),
        file_miss: RC_FILE_MISS.swap(0, AtOrd::Relaxed),
        line_zero: RC_LINE_ZERO.swap(0, AtOrd::Relaxed),
        pattern_miss: RC_PATTERN_MISS.swap(0, AtOrd::Relaxed),
        offset_miss: RC_OFFSET_MISS.swap(0, AtOrd::Relaxed),
        goto_miss: RC_GOTO_MISS.swap(0, AtOrd::Relaxed),
        nav_miss: RC_NAV_MISS.swap(0, AtOrd::Relaxed),
        empty_nav: RC_EMPTY_NAV.swap(0, AtOrd::Relaxed),
    }
}

/// Peek at resolve_call counters without resetting.
pub fn resolve_call_stats_peek() -> ResolveCallStats {
    ResolveCallStats {
        total: RC_TOTAL.load(AtOrd::Relaxed),
        ok: RC_OK.load(AtOrd::Relaxed),
        file_miss: RC_FILE_MISS.load(AtOrd::Relaxed),
        line_zero: RC_LINE_ZERO.load(AtOrd::Relaxed),
        pattern_miss: RC_PATTERN_MISS.load(AtOrd::Relaxed),
        offset_miss: RC_OFFSET_MISS.load(AtOrd::Relaxed),
        goto_miss: RC_GOTO_MISS.load(AtOrd::Relaxed),
        nav_miss: RC_NAV_MISS.load(AtOrd::Relaxed),
        empty_nav: RC_EMPTY_NAV.load(AtOrd::Relaxed),
    }
}

/// Diagnostic counters from resolve_call.
#[derive(Debug, serde::Serialize)]
pub struct ResolveCallStats {
    pub total: usize,
    pub ok: usize,
    pub file_miss: usize,
    pub line_zero: usize,
    pub pattern_miss: usize,
    pub offset_miss: usize,
    pub goto_miss: usize,
    pub nav_miss: usize,
    pub empty_nav: usize,
}

/// A live rust-analyzer instance holding the project database in memory.
pub struct RaInstance {
    host: AnalysisHost,
    analysis: Analysis,
    file_map: HashMap<String, FileInfo>,
    reverse_file_map: HashMap<FileId, String>,
    workspace_root: PathBuf,
}

/// A caller or callee with its source location.
#[derive(Debug, Clone)]
pub struct CallInfo {
    pub name: String,
    pub file: String,
    pub line: u32,
    pub col: u32,
    /// Call site lines (1-based) — where this callee is invoked from.
    pub call_site_lines: Vec<u32>,
}

/// A symbol location (definition, reference, etc.).
#[derive(Debug, Clone)]
pub struct SymbolLocation {
    pub name: String,
    pub file: String,
    pub full_range_start_line: u32,
    pub full_range_end_line: u32,
    pub focus_line: Option<u32>,
    pub focus_col: Option<u32>,
    pub kind: Option<String>,
    pub container: Option<String>,
}

/// Per-file info needed for hover queries.
struct FileInfo {
    file_id: FileId,
    line_index: LineIndex,
    source: String,
}

impl RaInstance {
    /// Load a workspace into the RA database (in-process, no external process).
    pub fn spawn(workspace_root: &Path) -> Result<Self> {
        let t0 = Instant::now();

        let mut cargo_config = CargoConfig::default();
        cargo_config.sysroot = Some(ra_ap_project_model::RustLibSource::Discover);
        cargo_config.all_targets = true;
        cargo_config.set_test = true;
        let load_config = LoadCargoConfig {
            load_out_dirs_from_check: true,
            with_proc_macro_server: ProcMacroServerChoice::Sysroot,
            prefill_caches: false,
            proc_macro_processes: 2,
        };

        let cargo_toml = workspace_root.join("Cargo.toml");
        let (db, vfs, _) = match load_workspace_at(
            &cargo_toml,
            &cargo_config,
            &load_config,
            &|msg| eprintln!("    [ra] {msg}"),
        ) {
            Ok(r) => r,
            Err(e) => {
                return Err(LspError::Protocol(format!(
                    "failed to load workspace: {e}"
                )));
            }
        };

        let host = AnalysisHost::with_database(db);
        let analysis = host.analysis();

        eprintln!(
            "    [ra] workspace loaded in {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        // Prime caches: trigger type inference for all crates so goto_definition works.
        let t_prime = Instant::now();
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        eprintln!("    [ra] priming caches ({num_threads} threads)...");
        match analysis.parallel_prime_caches(num_threads, |_progress| {}) {
            Ok(()) => eprintln!(
                "    [ra] caches primed in {:.1}s",
                t_prime.elapsed().as_secs_f64()
            ),
            Err(e) => eprintln!("    [ra] prime_caches cancelled: {e:?}"),
        }

        let (file_map, reverse_file_map) = build_file_map(&analysis, &vfs, workspace_root);
        eprintln!("    [ra] file map: {} files", file_map.len());

        Ok(Self {
            host,
            analysis,
            file_map,
            reverse_file_map,
            workspace_root: workspace_root.to_path_buf(),
        })
    }

    /// Hover at a file + line/col position. Returns the hover markup text.
    pub fn hover(
        &self,
        file_uri: &str,
        line: u32,
        character: u32,
    ) -> Result<Option<String>> {
        let file_key = uri_to_relative(file_uri, &self.workspace_root);
        let Some(fi) = self.file_map.get(&file_key) else {
            return Ok(None);
        };

        let line_col = LineCol {
            line,
            col: character,
        };
        let Some(offset) = fi.line_index.offset(line_col) else {
            return Ok(None);
        };
        let range = ra_ap_ide::TextRange::new(
            offset,
            offset + ra_ap_ide::TextSize::from(1u32),
        );

        let config = hover_config();
        let file_range = FileRange {
            file_id: fi.file_id,
            range,
        };

        match self.analysis.hover(&config, file_range) {
            Ok(Some(hover)) => {
                let markup = hover.info.markup.as_str();
                Ok(Some(markup.to_owned()))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(LspError::Protocol(format!("hover error: {e}"))),
        }
    }

    /// Parallel batch hover: creates N Analysis snapshots and distributes queries across threads.
    ///
    /// Each query is (file_uri, line, col). Returns Vec<Option<String>> in same order.
    /// Falls back to sequential if thread spawn fails.
    pub fn hover_batch_parallel(
        &self,
        queries: &[(String, u32, u32)],
    ) -> Vec<Option<String>> {
        if queries.is_empty() {
            return Vec::new();
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(4);

        // Pre-resolve file lookups on the main thread (file_map is not Send).
        struct ResolvedQuery {
            idx: usize,
            file_id: FileId,
            offset: ra_ap_ide::TextSize,
        }

        let mut resolved: Vec<ResolvedQuery> = Vec::with_capacity(queries.len());
        let mut results: Vec<Option<String>> = vec![None; queries.len()];

        for (idx, (file_uri, line, col)) in queries.iter().enumerate() {
            let file_key = uri_to_relative(file_uri, &self.workspace_root);
            let Some(fi) = self.file_map.get(&file_key) else { continue };
            let line_col = LineCol { line: *line, col: *col };
            let Some(offset) = fi.line_index.offset(line_col) else { continue };
            resolved.push(ResolvedQuery { idx, file_id: fi.file_id, offset });
        }

        if resolved.is_empty() {
            return results;
        }

        // Split into chunks and process in parallel with separate Analysis snapshots.
        let chunk_size = (resolved.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[ResolvedQuery]> = resolved.chunks(chunk_size).collect();

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(chunks.len());

            for chunk in &chunks {
                let analysis = self.host.analysis();
                let chunk = *chunk;
                handles.push(s.spawn(move || {
                    let config = hover_config();
                    let mut partial: Vec<(usize, Option<String>)> = Vec::with_capacity(chunk.len());
                    for q in chunk {
                        let range = ra_ap_ide::TextRange::new(
                            q.offset,
                            q.offset + ra_ap_ide::TextSize::from(1u32),
                        );
                        let file_range = FileRange { file_id: q.file_id, range };
                        let result = match analysis.hover(&config, file_range) {
                            Ok(Some(hover)) => Some(hover.info.markup.as_str().to_owned()),
                            _ => None,
                        };
                        partial.push((q.idx, result));
                    }
                    partial
                }));
            }

            for handle in handles {
                if let Ok(partial) = handle.join() {
                    for (idx, result) in partial {
                        results[idx] = result;
                    }
                }
            }
        });

        results
    }

    /// Hover on a variable name within a function to get its type.
    ///
    /// Searches for `let {var_name}` in the source starting from `fn_start_line`
    /// and hovers on the variable to extract its type.
    pub fn hover_on_var(
        &self,
        chunk_file: &str,
        var_name: &str,
        fn_start_line: u32,
    ) -> Option<String> {
        let file_key = normalize_chunk_file(chunk_file);
        let fi = self.file_map.get(&file_key)?;

        let search = format!("let {var_name}");
        let start_offset = line_offset(&fi.source, fn_start_line)?;
        let search_area = &fi.source[start_offset..];
        let pos = search_area.find(&search)?;
        let var_offset = start_offset + pos + 4; // skip "let "

        let line_col = fi.line_index.line_col(
            ra_ap_ide::TextSize::from(var_offset as u32),
        );
        let offset = fi.line_index.offset(line_col)?;
        let range = ra_ap_ide::TextRange::new(
            offset,
            offset + ra_ap_ide::TextSize::from(var_name.len() as u32),
        );

        let config = hover_config();
        let file_range = FileRange {
            file_id: fi.file_id,
            range,
        };

        let hover = self.analysis.hover(&config, file_range).ok()??;
        extract_type_from_hover_result(&hover.info)
    }

    /// Hover on the receiver (char before `.method(`) to get its type.
    pub fn hover_on_receiver(
        &self,
        chunk_file: &str,
        method_name: &str,
        call_line_1based: u32,
    ) -> Option<String> {
        let file_key = normalize_chunk_file(chunk_file);
        let fi = self.file_map.get(&file_key)?;

        let line_0based = (call_line_1based - 1) as usize;
        let lines: Vec<&str> = fi.source.lines().collect();

        let pattern = format!(".{method_name}(");
        let pattern_turbo = format!(".{method_name}::<");

        let (target_line, dot_col) = find_pattern_in_lines(&lines, line_0based, &pattern)
            .or_else(|| find_pattern_in_lines(&lines, line_0based, &pattern_turbo))?;

        if dot_col == 0 {
            return None;
        }
        let hover_col = dot_col - 1;

        let line_col = LineCol {
            line: target_line as u32,
            col: hover_col,
        };
        let offset = fi.line_index.offset(line_col)?;
        let range = ra_ap_ide::TextRange::new(
            offset,
            offset + ra_ap_ide::TextSize::from(1u32),
        );

        let config = hover_config();
        let file_range = FileRange {
            file_id: fi.file_id,
            range,
        };

        let hover_result = self.analysis.hover(&config, file_range).ok()?;
        if hover_result.is_none() {
            let line_text = lines.get(target_line).unwrap_or(&"");
            let hover_char = line_text
                .as_bytes()
                .get(hover_col as usize)
                .map(|b| *b as char)
                .unwrap_or('?');
            eprintln!(
                "      [hover-null] line {}:{} char='{}' src='{}'  method={method_name}",
                target_line + 1,
                hover_col,
                hover_char,
                line_text.trim().chars().take(60).collect::<String>()
            );
            return None;
        }
        extract_type_from_hover_result(&hover_result?.info)
    }

    /// Resolve a call to its definition location via goto_definition.
    ///
    /// For `receiver.method()` calls, pass the method name; for bare calls, the function name.
    /// Returns (definition_file, definition_line_1based) on success.
    pub fn resolve_call(
        &self,
        chunk_file: &str,
        call_name: &str,
        call_line_1based: u32,
    ) -> Option<(String, u32)> {
        RC_TOTAL.fetch_add(1, AtOrd::Relaxed);

        let file_key = normalize_chunk_file(chunk_file);
        let fi = match self.file_map.get(&file_key) {
            Some(fi) => fi,
            None => {
                let miss = RC_FILE_MISS.fetch_add(1, AtOrd::Relaxed);
                if miss < 3 {
                    let sample_keys: Vec<_> = self.file_map.keys().take(5).collect();
                    eprintln!("    [ra-dbg] file miss: key={file_key:?}, sample_keys={sample_keys:?}");
                }
                return None;
            }
        };

        let line_0based = match call_line_1based.checked_sub(1) {
            Some(l) => l as usize,
            None => {
                RC_LINE_ZERO.fetch_add(1, AtOrd::Relaxed);
                return None;
            }
        };
        let lines: Vec<&str> = fi.source.lines().collect();

        // Extract the actual method/function name to position cursor on.
        // For "receiver.method" → leaf is "method", for "Type::method" → leaf is "method".
        let leaf_name = call_name.rsplit('.').next().unwrap_or(call_name);
        // For qualified names like "Instant::now", we need to find "now" specifically.
        let method_name = leaf_name.rsplit("::").next().unwrap_or(leaf_name);

        // Patterns to search: .method(, .method::<, Type::method(, method(
        let dot_pattern = format!(".{method_name}(");
        let dot_turbo = format!(".{method_name}::<");
        let qualified_pattern = format!("::{method_name}(");
        let bare_pattern = format!("{method_name}(");

        let (target_line, name_col) = if let Some((l, c)) =
            find_pattern_in_lines(&lines, line_0based, &dot_pattern)
        {
            (l, c as usize + 1) // skip the dot
        } else if let Some((l, c)) =
            find_pattern_in_lines(&lines, line_0based, &dot_turbo)
        {
            (l, c as usize + 1)
        } else if let Some((l, c)) =
            find_pattern_in_lines(&lines, line_0based, &qualified_pattern)
        {
            (l, c as usize + 2) // skip "::"
        } else if let Some((l, c)) =
            find_pattern_in_lines(&lines, line_0based, &bare_pattern)
        {
            (l, c as usize)
        } else {
            let miss = RC_PATTERN_MISS.fetch_add(1, AtOrd::Relaxed);
            if miss < 3 {
                let ctx = if line_0based < lines.len() { lines[line_0based] } else { "<OOB>" };
                eprintln!("    [ra-dbg] pattern miss: leaf={leaf_name:?}, line={call_line_1based}, src={ctx:?}");
            }
            return None;
        };

        let line_col = LineCol {
            line: target_line as u32,
            col: name_col as u32,
        };
        let offset = match fi.line_index.offset(line_col) {
            Some(o) => o,
            None => {
                RC_OFFSET_MISS.fetch_add(1, AtOrd::Relaxed);
                return None;
            }
        };
        let pos = FilePosition {
            file_id: fi.file_id,
            offset,
        };
        let config = GotoDefinitionConfig {
            minicore: Default::default(),
        };
        let result = match self.analysis.goto_definition(pos, &config) {
            Ok(Some(r)) => r,
            _ => {
                let miss = RC_GOTO_MISS.fetch_add(1, AtOrd::Relaxed);
                if miss < 3 {
                    eprintln!("    [ra-dbg] goto_def miss: file={file_key}, call={call_name}, line={call_line_1based}, offset={offset:?}");
                }
                return None;
            }
        };

        for nav in &result.info {
            if let Some(def_file) = self.reverse_file_map.get(&nav.file_id) {
                if let Some(fi_def) = self.file_map.get(def_file) {
                    let start = nav.focus_range.unwrap_or(nav.full_range).start();
                    let def_line = fi_def.line_index.line_col(start).line + 1;
                    RC_OK.fetch_add(1, AtOrd::Relaxed);
                    return Some((def_file.clone(), def_line));
                }
            }
        }

        if result.info.is_empty() {
            // goto_definition succeeded but returned empty results.
            // Try: maybe offset is on the dot instead of the name.
            // Re-try with offset+1 to land inside the identifier.
            let retry_pos = FilePosition {
                file_id: fi.file_id,
                offset: offset + TextSize::from(1u32),
            };
            let retry = self.analysis.goto_definition(retry_pos, &config);
            if let Ok(Some(r2)) = &retry {
                if !r2.info.is_empty() {
                    for nav in &r2.info {
                        if let Some(def_file) = self.reverse_file_map.get(&nav.file_id) {
                            if let Some(fi_def) = self.file_map.get(def_file) {
                                let start = nav.focus_range.unwrap_or(nav.full_range).start();
                                let def_line = fi_def.line_index.line_col(start).line + 1;
                                RC_OK.fetch_add(1, AtOrd::Relaxed);
                                return Some((def_file.clone(), def_line));
                            }
                        }
                    }
                    // External.
                    RC_NAV_MISS.fetch_add(1, AtOrd::Relaxed);
                    RC_OK.fetch_add(1, AtOrd::Relaxed);
                    return Some(("__external__".to_owned(), 0));
                }
            }

            RC_EMPTY_NAV.fetch_add(1, AtOrd::Relaxed);
            record_empty_nav_file(&file_key);
            return None;
        }

        // goto_definition succeeded but target is outside project (std/deps).
        RC_NAV_MISS.fetch_add(1, AtOrd::Relaxed);
        RC_OK.fetch_add(1, AtOrd::Relaxed);
        Some(("__external__".to_owned(), 0))
    }

    pub fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    // ── Call Hierarchy API ──────────────────────────────────────────

    /// Find all callers of the symbol at `file:line:col` (incoming calls).
    pub fn incoming_calls(
        &self,
        file: &str,
        line: u32,
        col: u32,
    ) -> Result<Vec<CallInfo>> {
        let pos = self.file_position(file, line, col)?;
        let config = call_hierarchy_config();
        let items = self
            .analysis
            .incoming_calls(&config, pos)
            .map_err(|e| LspError::Protocol(format!("incoming_calls cancelled: {e}")))?
            .unwrap_or_default();
        Ok(items.into_iter().map(|ci| self.call_item_to_info(&ci)).collect())
    }

    /// Find all callees from the symbol at `file:line:col` (outgoing calls).
    pub fn outgoing_calls(
        &self,
        file: &str,
        line: u32,
        col: u32,
    ) -> Result<Vec<CallInfo>> {
        let pos = self.file_position(file, line, col)?;
        let config = call_hierarchy_config();
        let items = self
            .analysis
            .outgoing_calls(&config, pos)
            .map_err(|e| LspError::Protocol(format!("outgoing_calls cancelled: {e}")))?
            .unwrap_or_default();
        Ok(items.into_iter().map(|ci| self.call_item_to_info(&ci)).collect())
    }

    // ── Navigation API ──────────────────────────────────────────────

    /// Go to definition of the symbol at `file:line:col`.
    pub fn goto_definition(
        &self,
        file: &str,
        line: u32,
        col: u32,
    ) -> Result<Vec<SymbolLocation>> {
        let pos = self.file_position(file, line, col)?;
        let config = GotoDefinitionConfig {
            minicore: Default::default(),
        };
        let result = self
            .analysis
            .goto_definition(pos, &config)
            .map_err(|e| LspError::Protocol(format!("goto_definition cancelled: {e}")))?;
        Ok(result
            .map(|ri| ri.info.iter().map(|nav| self.nav_to_location(nav)).collect())
            .unwrap_or_default())
    }

    /// Find all references of the symbol at `file:line:col`.
    pub fn find_references(
        &self,
        file: &str,
        line: u32,
        col: u32,
    ) -> Result<Vec<SymbolLocation>> {
        let pos = self.file_position(file, line, col)?;
        let config = FindAllRefsConfig {
            search_scope: None,
            minicore: Default::default(),
        };
        let results = self
            .analysis
            .find_all_refs(pos, &config)
            .map_err(|e| LspError::Protocol(format!("find_all_refs cancelled: {e}")))?
            .unwrap_or_default();
        let mut locs = Vec::new();
        for result in &results {
            if let Some(decl) = &result.declaration {
                locs.push(self.nav_to_location(&decl.nav));
            }
            for (&file_id, ranges) in &result.references {
                if let Some(rel_path) = self.reverse_file_map.get(&file_id) {
                    if let Some(fi) = self.file_map.get(rel_path) {
                        for (range, _category) in ranges {
                            let start = fi.line_index.line_col(range.start());
                            locs.push(SymbolLocation {
                                name: String::new(),
                                file: rel_path.clone(),
                                full_range_start_line: start.line,
                                full_range_end_line: fi.line_index.line_col(range.end()).line,
                                focus_line: Some(start.line),
                                focus_col: Some(start.col),
                                kind: None,
                                container: None,
                            });
                        }
                    }
                }
            }
        }
        Ok(locs)
    }

    /// Search workspace symbols by name.
    pub fn workspace_symbols(&self, query_str: &str, limit: usize) -> Result<Vec<SymbolLocation>> {
        let query = Query::new(query_str.to_owned());
        let navs = self
            .analysis
            .symbol_search(query, limit)
            .map_err(|e| LspError::Protocol(format!("symbol_search cancelled: {e}")))?;
        Ok(navs.iter().map(|nav| self.nav_to_location(nav)).collect())
    }

    // ── Position resolution ─────────────────────────────────────────

    /// Resolve a symbol name to a FilePosition by searching the file map.
    pub fn resolve_symbol(&self, symbol_name: &str) -> Option<(String, u32, u32)> {
        let name_lower = symbol_name.to_lowercase();
        let patterns: Vec<String> = if name_lower.contains("::") {
            let method = name_lower.rsplit("::").next().unwrap_or(&name_lower);
            vec![format!("fn {method}("), format!("fn {method}<")]
        } else {
            vec![format!("fn {name_lower}("), format!("fn {name_lower}<")]
        };

        for (rel_path, fi) in &self.file_map {
            let src_lower = fi.source.to_lowercase();
            for pat in &patterns {
                if let Some(pos) = src_lower.find(pat.as_str()) {
                    let fn_offset = pos + 3; // skip "fn "
                    let lc = fi.line_index.line_col(TextSize::from(fn_offset as u32));
                    return Some((rel_path.clone(), lc.line, lc.col));
                }
            }
        }
        None
    }

    // ── Internal helpers ────────────────────────────────────────────

    fn file_position(&self, file: &str, line: u32, col: u32) -> Result<FilePosition> {
        let file_key = normalize_chunk_file(file);
        let fi = self
            .file_map
            .get(&file_key)
            .ok_or_else(|| LspError::Protocol(format!("file not found: {file_key}")))?;
        let line_col = LineCol { line, col };
        let offset = fi
            .line_index
            .offset(line_col)
            .ok_or_else(|| LspError::Protocol(format!("invalid position {line}:{col}")))?;
        Ok(FilePosition {
            file_id: fi.file_id,
            offset,
        })
    }

    fn call_item_to_info(&self, item: &CallItem) -> CallInfo {
        let nav = &item.target;
        let file = self
            .reverse_file_map
            .get(&nav.file_id)
            .cloned()
            .unwrap_or_default();
        let (line, col) = self.nav_focus_line_col(nav);
        // Extract call site lines from ranges.
        let call_site_lines: Vec<u32> = item
            .ranges
            .iter()
            .filter_map(|fr| {
                let caller_file = self.reverse_file_map.get(&fr.file_id)?;
                let fi = self.file_map.get(caller_file)?;
                Some(fi.line_index.line_col(fr.range.start()).line + 1)
            })
            .collect();
        CallInfo {
            name: nav.name.to_string(),
            file,
            line,
            col,
            call_site_lines,
        }
    }

    fn nav_to_location(&self, nav: &NavigationTarget) -> SymbolLocation {
        let file = self
            .reverse_file_map
            .get(&nav.file_id)
            .cloned()
            .unwrap_or_default();
        let (focus_line, focus_col) = if let Some(fi) = self.file_map.get(&file) {
            if let Some(focus) = nav.focus_range {
                let lc = fi.line_index.line_col(focus.start());
                (Some(lc.line), Some(lc.col))
            } else {
                let lc = fi.line_index.line_col(nav.full_range.start());
                (Some(lc.line), Some(lc.col))
            }
        } else {
            (None, None)
        };
        let (start_line, end_line) = if let Some(fi) = self.file_map.get(&file) {
            (
                fi.line_index.line_col(nav.full_range.start()).line,
                fi.line_index.line_col(nav.full_range.end()).line,
            )
        } else {
            (0, 0)
        };
        SymbolLocation {
            name: nav.name.to_string(),
            file,
            full_range_start_line: start_line,
            full_range_end_line: end_line,
            focus_line,
            focus_col,
            kind: nav.kind.map(|k| format!("{k:?}")),
            container: nav.container_name.as_ref().map(|c| c.to_string()),
        }
    }

    fn nav_focus_line_col(&self, nav: &NavigationTarget) -> (u32, u32) {
        let file = self
            .reverse_file_map
            .get(&nav.file_id)
            .cloned()
            .unwrap_or_default();
        if let Some(fi) = self.file_map.get(&file) {
            let offset = nav.focus_range.map_or(nav.full_range.start(), |r| r.start());
            let lc = fi.line_index.line_col(offset);
            (lc.line, lc.col)
        } else {
            (0, 0)
        }
    }
}

// ── Internal helpers ────────────────────────────────────────────────

fn call_hierarchy_config() -> CallHierarchyConfig<'static> {
    CallHierarchyConfig {
        exclude_tests: false,
        minicore: Default::default(),
    }
}

fn hover_config() -> HoverConfig<'static> {
    HoverConfig {
        links_in_hover: false,
        memory_layout: None,
        documentation: false,
        keywords: false,
        format: HoverDocFormat::PlainText,
        max_trait_assoc_items_count: None,
        max_fields_count: None,
        max_enum_variants_count: None,
        max_subst_ty_len: SubstTyLen::LimitTo(40),
        show_drop_glue: false,
        minicore: Default::default(),
    }
}

/// Build mapping from normalized relative file paths to FileInfo,
/// and a reverse map from FileId to relative path.
fn build_file_map(
    analysis: &Analysis,
    vfs: &Vfs,
    project_root: &Path,
) -> (HashMap<String, FileInfo>, HashMap<FileId, String>) {
    let mut map = HashMap::new();
    let mut reverse = HashMap::new();
    for (file_id, vfs_path) in vfs.iter() {
        let Some(path_str) = vfs_path.as_path().map(|p| p.to_string()) else {
            continue;
        };
        let Ok(contents) = analysis.file_text(file_id) else {
            continue;
        };
        let line_index = LineIndex::new(&contents);

        let normalized = normalize_vfs_path(&path_str, project_root);
        reverse.insert(file_id, normalized.clone());
        map.insert(
            normalized,
            FileInfo {
                file_id,
                line_index,
                source: contents.to_string(),
            },
        );
    }
    (map, reverse)
}

/// Normalize a VFS absolute path to a relative path matching chunk file paths.
fn normalize_vfs_path(vfs_path: &str, project_root: &Path) -> String {
    let normalized = vfs_path.replace('\\', "/");

    // Canonicalize the root to match VFS absolute paths.
    let canon_root = project_root
        .canonicalize()
        .unwrap_or_else(|_| project_root.to_path_buf());
    let mut root_str = canon_root.to_string_lossy().replace('\\', "/");
    // Strip Windows extended-length prefix (\\?\).
    if let Some(stripped) = root_str.strip_prefix("//?/") {
        root_str = stripped.to_owned();
    }
    // Ensure no trailing slash.
    if root_str.ends_with('/') {
        root_str.pop();
    }

    if let Some(rel) = normalized.strip_prefix(&root_str) {
        let rel = rel.trim_start_matches('/');
        if let Some((crate_part, rest)) = rel.split_once('/') {
            let crate_name = crate_part.strip_suffix("-stub").unwrap_or(crate_part);
            if crate_name == "crates" {
                return rel.to_owned();
            }
            return format!("crates/{crate_name}/{rest}");
        }
        return rel.to_owned();
    }
    normalized
}

/// Normalize chunk file path for lookup.
fn normalize_chunk_file(chunk_file: &str) -> String {
    chunk_file.replace('\\', "/")
}

/// Convert a file:// URI to a relative path.
fn uri_to_relative(uri: &str, project_root: &Path) -> String {
    let path = uri
        .strip_prefix("file:///")
        .unwrap_or(uri)
        .replace('\\', "/");
    let root = project_root.to_string_lossy().replace('\\', "/");
    if let Some(rel) = path.strip_prefix(&root) {
        rel.trim_start_matches('/').to_owned()
    } else {
        path
    }
}

/// Extract type string from hover result markup.
fn extract_type_from_hover_result(hover: &ra_ap_ide::HoverResult) -> Option<String> {
    let markup = hover.markup.as_str();
    for line in markup.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("```") || trimmed.starts_with("---") {
            continue;
        }
        if let Some(after_colon) = trimmed.split_once(':') {
            let type_str = after_colon.1.trim();
            if !type_str.is_empty() {
                return Some(type_str.to_owned());
            }
        }
        return Some(trimmed.to_owned());
    }
    None
}

/// Find a pattern in source lines around `start`, checking ±5 lines (forward first).
fn find_pattern_in_lines(lines: &[&str], start: usize, pattern: &str) -> Option<(usize, u32)> {
    // Check exact line first, then expand outward.
    for offset in 0..=5 {
        let line_idx = start + offset;
        if line_idx >= lines.len() {
            break;
        }
        if let Some(col) = lines[line_idx].find(pattern) {
            return Some((line_idx, col as u32));
        }
    }
    // Check backward.
    for offset in 1..=5 {
        if offset > start {
            break;
        }
        let line_idx = start - offset;
        if let Some(col) = lines[line_idx].find(pattern) {
            return Some((line_idx, col as u32));
        }
    }
    None
}

/// Get byte offset for the start of a line (1-based).
fn line_offset(source: &str, line_1based: u32) -> Option<usize> {
    let target = (line_1based - 1) as usize;
    let mut current_line = 0;
    for (i, ch) in source.char_indices() {
        if current_line == target {
            return Some(i);
        }
        if ch == '\n' {
            current_line += 1;
        }
    }
    if current_line == target {
        Some(source.len())
    } else {
        None
    }
}
