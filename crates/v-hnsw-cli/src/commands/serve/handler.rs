//! TCP client connection handler.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::daemon::DaemonState;
use super::{
    CodeIntelParams, EmbedParams, JsonRpcError, JsonRpcRequest, JsonRpcResponse, SearchParams,
    UpdateParams,
};

/// Handle a single client connection.
pub(crate) fn handle_client(
    stream: TcpStream,
    state: &mut DaemonState,
    last_activity: &mut Instant,
) -> Result<()> {
    // Generous timeouts: update can take minutes for large repos
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(300)))?;

    let mut reader = BufReader::new(&stream);
    let mut writer = &stream;

    let mut line = String::new();
    reader.read_line(&mut line)?;

    *last_activity = Instant::now();

    let request: JsonRpcRequest = serde_json::from_str(&line)
        .with_context(|| format!("Failed to parse request: {}", line.trim()))?;

    let response = match request.method.as_str() {
        "search" => {
            let params: SearchParams =
                serde_json::from_value(request.params).context("Invalid search params")?;
            let db_path = PathBuf::from(&params.db);

            match state.search(&db_path, &params.query, params.k, params.tags) {
                Ok(result) => JsonRpcResponse {
                    id: request.id,
                    result: Some(serde_json::to_value(result)?),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -1,
                        message: e.to_string(),
                    }),
                },
            }
        }
        "ping" => JsonRpcResponse {
            id: request.id,
            result: Some(serde_json::json!({"status": "ok"})),
            error: None,
        },
        "reload" => {
            let db_path: String = request.params
                .get("db").and_then(|d| d.as_str())
                .unwrap_or("").to_string();
            match state.reload(&PathBuf::from(&db_path)) {
                Ok(()) => JsonRpcResponse {
                    id: request.id,
                    result: Some(serde_json::json!({"status": "reloaded"})),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -2,
                        message: format!("Reload failed: {}", e),
                    }),
                },
            }
        }
        "embed" => {
            let params: EmbedParams =
                serde_json::from_value(request.params).context("Invalid embed params")?;

            match state.embed(&params.texts.iter().map(|s| s.as_str()).collect::<Vec<_>>()) {
                Ok(embeddings) => JsonRpcResponse {
                    id: request.id,
                    result: Some(serde_json::json!({"embeddings": embeddings})),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -3,
                        message: format!("Embed failed: {}", e),
                    }),
                },
            }
        }
        "update" => {
            let params: UpdateParams =
                serde_json::from_value(request.params).context("Invalid update params")?;
            let db_path = PathBuf::from(&params.db);
            let input_path = PathBuf::from(&params.input);

            match state.update(&db_path, &input_path, &params.exclude) {
                Ok(stats) => JsonRpcResponse {
                    id: request.id,
                    result: Some(serde_json::to_value(stats)?),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -4,
                        message: format!("Update failed: {e}"),
                    }),
                },
            }
        }
        "shutdown" => {
            let response = JsonRpcResponse {
                id: request.id,
                result: Some(serde_json::json!({"status": "shutting_down"})),
                error: None,
            };
            let response_json = serde_json::to_string(&response)?;
            writeln!(writer, "{}", response_json)?;
            writer.flush()?;
            anyhow::bail!("Shutdown requested");
        }
        // ── Code-intel methods ────────────────────────────────────────
        "stats" | "def" | "refs" | "symbols" | "gather" | "impact" | "trace" | "detail" => {
            let params: CodeIntelParams =
                serde_json::from_value(request.params).context("Invalid code-intel params")?;
            match handle_code_intel(&request.method, &params) {
                Ok(value) => JsonRpcResponse {
                    id: request.id,
                    result: Some(value),
                    error: None,
                },
                Err(e) => JsonRpcResponse {
                    id: request.id,
                    result: None,
                    error: Some(JsonRpcError {
                        code: -10,
                        message: format!("{}: {e}", request.method),
                    }),
                },
            }
        }
        _ => JsonRpcResponse {
            id: request.id,
            result: None,
            error: Some(JsonRpcError {
                code: -32601,
                message: format!("Unknown method: {}", request.method),
            }),
        },
    };

    let response_json = serde_json::to_string(&response)?;
    writeln!(writer, "{response_json}")?;
    writer.flush()?;

    Ok(())
}

// ── Code-intel dispatch ──────────────────────────────────────────────────

use crate::commands::code_intel;

/// Dispatch a code-intel method and return the result as JSON.
fn handle_code_intel(method: &str, params: &CodeIntelParams) -> Result<serde_json::Value> {
    let db = PathBuf::from(&params.db);

    match method {
        "stats" => ci_stats(&db),
        "def" => {
            let name = params.name.as_deref()
                .context("missing 'name' param for def")?;
            ci_def(&db, name)
        }
        "refs" => {
            let name = params.name.as_deref()
                .context("missing 'name' param for refs")?;
            ci_refs(&db, name)
        }
        "symbols" => ci_symbols(&db, params.name.as_deref(), params.kind.as_deref()),
        "gather" => {
            let symbol = params.symbol.as_deref()
                .context("missing 'symbol' param for gather")?;
            ci_gather(&db, symbol, params.depth.unwrap_or(2), params.k, params.include_tests)
        }
        "impact" => {
            let symbol = params.symbol.as_deref()
                .context("missing 'symbol' param for impact")?;
            ci_impact(&db, symbol, params.depth.unwrap_or(2), params.include_tests)
        }
        "trace" => {
            let from = params.from.as_deref()
                .context("missing 'from' param for trace")?;
            let to = params.to.as_deref()
                .context("missing 'to' param for trace")?;
            ci_trace(&db, from, to)
        }
        "detail" => {
            let symbol = params.symbol.as_deref()
                .or(params.name.as_deref())
                .context("missing 'symbol' or 'name' param for detail")?;
            ci_detail(&db, symbol)
        }
        _ => anyhow::bail!("unknown code-intel method: {method}"),
    }
}

// ── Individual code-intel handlers ───────────────────────────────────────

fn ci_stats(db: &Path) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let stats = build_stats_map(&chunks);
    Ok(stats)
}

fn ci_def(db: &Path, name: &str) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let name_lower = name.to_lowercase();
    let matches: Vec<&code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
        c.name.to_lowercase() == name_lower
            || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
    }).collect();
    Ok(grouped_json(&matches))
}

fn ci_refs(db: &Path, name: &str) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let name_lower = name.to_lowercase();
    let refs: Vec<(&code_intel::parse::CodeChunk, Vec<&str>)> = chunks.iter().filter_map(|c| {
        let mut via = Vec::new();
        if c.calls.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
            via.push("calls");
        }
        if c.types.iter().any(|s| s.to_lowercase().contains(&name_lower)) {
            via.push("types");
        }
        if c.signature.as_ref().is_some_and(|s| s.to_lowercase().contains(&name_lower)) {
            via.push("signature");
        }
        if c.name.to_lowercase().contains(&name_lower) {
            via.push("name");
        }
        if via.is_empty() { None } else { Some((c, via)) }
    }).collect();
    Ok(grouped_json_refs(&refs))
}

fn ci_symbols(db: &Path, name: Option<&str>, kind: Option<&str>) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let filtered: Vec<&code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
        if let Some(n) = name
            && !c.name.to_lowercase().contains(&n.to_lowercase()) { return false; }
        if let Some(k) = kind
            && c.kind.to_lowercase() != k.to_lowercase() { return false; }
        true
    }).collect();
    Ok(grouped_json(&filtered))
}

fn ci_gather(db: &Path, symbol: &str, depth: u32, k: usize, include_tests: bool) -> Result<serde_json::Value> {
    let graph = code_intel::context::load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    // Use gather's compute function via JSON format
    // Re-implement inline to avoid stdout capture
    let forward = gather_bfs_forward(&graph, &seeds, depth, include_tests);
    let reverse = gather_bfs_reverse(&graph, &seeds, depth, include_tests);
    let mut entries = merge_gather(forward, reverse);
    entries.truncate(k);
    Ok(build_gather_json(&graph, &entries))
}

fn ci_impact(db: &Path, symbol: &str, depth: u32, include_tests: bool) -> Result<serde_json::Value> {
    let graph = code_intel::context::load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    let entries = impact_bfs_reverse(&graph, &seeds, depth, include_tests);
    Ok(build_impact_json(&graph, &entries))
}

fn ci_trace(db: &Path, from: &str, to: &str) -> Result<serde_json::Value> {
    let graph = code_intel::context::load_or_build_graph(db)?;
    let sources = graph.resolve(from);
    let targets = graph.resolve(to);
    if sources.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{from}\"")}));
    }
    if targets.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{to}\"")}));
    }
    match trace_bfs(&graph, &sources, &targets) {
        Some(path) => Ok(build_trace_json(&graph, &path)),
        None => Ok(serde_json::json!({"path": null, "hops": null})),
    }
}

fn ci_detail(db: &Path, symbol: &str) -> Result<serde_json::Value> {
    match code_intel::reason::load_reason(db, symbol)? {
        Some(entry) => Ok(serde_json::to_value(entry)?),
        None => Ok(serde_json::json!({"symbol": symbol, "found": false})),
    }
}

// ── Stats helper ─────────────────────────────────────────────────────────

fn build_stats_map(chunks: &[code_intel::parse::CodeChunk]) -> serde_json::Value {
    use std::collections::BTreeMap;
    let mut stats: BTreeMap<String, [usize; 4]> = BTreeMap::new();
    for c in chunks {
        let crate_name = extract_crate_name(&c.file);
        let row = stats.entry(crate_name).or_insert([0; 4]);
        let is_test = c.file.contains("/tests/") || c.name.starts_with("test_");
        match (c.kind.as_str(), is_test) {
            ("function", false) => row[0] += 1,
            ("function", true) => row[1] += 1,
            ("struct", _) => row[2] += 1,
            ("enum", _) => row[3] += 1,
            _ => {}
        }
    }
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("p=prod_fn,t=test_fn,s=struct,e=enum"));
    for (name, row) in &stats {
        map.insert(name.clone(), serde_json::json!({"p":row[0],"t":row[1],"s":row[2],"e":row[3]}));
    }
    serde_json::Value::Object(map)
}

fn extract_crate_name(path: &str) -> String {
    if let Some(start) = path.find("crates/") {
        let rest = &path[start + 7..];
        if let Some(slash) = rest.find('/') {
            return rest[..slash].to_owned();
        }
    }
    "(root)".to_owned()
}

// ── Grouped JSON builders ────────────────────────────────────────────────

fn relative_path(path: &str) -> &str {
    if let Some(idx) = path.find("crates/") {
        &path[idx..]
    } else if let Some(idx) = path.find("src/") {
        &path[idx..]
    } else {
        path
    }
}

fn lines_str(c: &code_intel::parse::CodeChunk) -> String {
    if let Some((start, end)) = c.lines {
        format!("{start}-{end}")
    } else {
        String::new()
    }
}

fn grouped_json(chunks: &[&code_intel::parse::CodeChunk]) -> serde_json::Value {
    use std::collections::BTreeMap;
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("f=file,l=lines,k=kind,n=name"));
    let mut groups: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for c in chunks {
        let path = relative_path(&c.file);
        groups.entry(path).or_default().push(serde_json::json!({
            "l": lines_str(c),
            "k": &c.kind,
            "n": &c.name,
        }));
    }
    for (path, items) in groups {
        map.insert(path.to_owned(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(map)
}

fn grouped_json_refs(refs: &[(&code_intel::parse::CodeChunk, Vec<&str>)]) -> serde_json::Value {
    use std::collections::BTreeMap;
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("f=file,l=lines,k=kind,n=name,v=via"));
    let mut groups: BTreeMap<&str, Vec<serde_json::Value>> = BTreeMap::new();
    for (c, via) in refs {
        let path = relative_path(&c.file);
        groups.entry(path).or_default().push(serde_json::json!({
            "l": lines_str(c),
            "k": &c.kind,
            "n": &c.name,
            "v": via,
        }));
    }
    for (path, items) in groups {
        map.insert(path.to_owned(), serde_json::Value::Array(items));
    }
    serde_json::Value::Object(map)
}

// ── Graph-based BFS (gather/impact/trace) ────────────────────────────────

use crate::commands::code_intel::graph::CallGraph;

struct GatherEntry {
    idx: u32,
    depth: u32,
    score: f64,
    direction: &'static str,
}

fn gather_bfs_forward(graph: &CallGraph, seeds: &[u32], max_depth: u32, include_tests: bool) -> Vec<GatherEntry> {
    use std::collections::VecDeque;
    let mut visited = vec![false; graph.len()];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    let mut results = Vec::new();
    for &seed in seeds {
        if (seed as usize) < graph.len() && !visited[seed as usize] {
            visited[seed as usize] = true;
            queue.push_back((seed, 0));
        }
    }
    while let Some((idx, depth)) = queue.pop_front() {
        let is_test = graph.is_test[idx as usize];
        if !include_tests && is_test { continue; }
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };
        results.push(GatherEntry { idx, depth, score, direction: "callee" });
        if depth < max_depth {
            for &callee in &graph.callees[idx as usize] {
                if !visited[callee as usize] {
                    visited[callee as usize] = true;
                    queue.push_back((callee, depth + 1));
                }
            }
        }
    }
    results
}

fn gather_bfs_reverse(graph: &CallGraph, seeds: &[u32], max_depth: u32, include_tests: bool) -> Vec<GatherEntry> {
    use std::collections::VecDeque;
    let mut visited = vec![false; graph.len()];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    let mut results = Vec::new();
    for &seed in seeds {
        if (seed as usize) < graph.len() && !visited[seed as usize] {
            visited[seed as usize] = true;
            queue.push_back((seed, 0));
        }
    }
    while let Some((idx, depth)) = queue.pop_front() {
        let is_test = graph.is_test[idx as usize];
        if !include_tests && is_test { continue; }
        let score = (1.0 / f64::from(depth + 1)) * if is_test { 0.1 } else { 1.0 };
        results.push(GatherEntry { idx, depth, score, direction: "caller" });
        if depth < max_depth {
            for &caller in &graph.callers[idx as usize] {
                if !visited[caller as usize] {
                    visited[caller as usize] = true;
                    queue.push_back((caller, depth + 1));
                }
            }
        }
    }
    results
}

fn merge_gather(forward: Vec<GatherEntry>, reverse: Vec<GatherEntry>) -> Vec<GatherEntry> {
    use std::collections::BTreeMap;
    let mut best: BTreeMap<u32, GatherEntry> = BTreeMap::new();
    for entry in forward.into_iter().chain(reverse) {
        best.entry(entry.idx)
            .and_modify(|existing| {
                if entry.score > existing.score {
                    *existing = GatherEntry {
                        idx: entry.idx,
                        depth: entry.depth,
                        score: entry.score,
                        direction: entry.direction,
                    };
                }
            })
            .or_insert(entry);
    }
    let mut results: Vec<GatherEntry> = best.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

fn build_gather_json(graph: &CallGraph, entries: &[GatherEntry]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("f=file,l=lines,k=kind,n=name,d=depth,sc=score,dir=direction,t=test"));
    let items: Vec<serde_json::Value> = entries.iter().map(|e| {
        let i = e.idx as usize;
        serde_json::json!({
            "f": relative_path(&graph.files[i]),
            "l": graph_lines_str(graph.lines[i]),
            "k": &graph.kinds[i],
            "n": &graph.names[i],
            "d": e.depth,
            "sc": format!("{:.2}", e.score),
            "dir": e.direction,
            "t": graph.is_test[i],
        })
    }).collect();
    map.insert("results".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

struct ImpactEntry {
    idx: u32,
    depth: u32,
    is_test: bool,
}

fn impact_bfs_reverse(graph: &CallGraph, seeds: &[u32], max_depth: u32, include_tests: bool) -> Vec<ImpactEntry> {
    use std::collections::VecDeque;
    let mut visited = vec![false; graph.len()];
    let mut queue: VecDeque<(u32, u32)> = VecDeque::new();
    let mut results = Vec::new();
    for &seed in seeds {
        if (seed as usize) < graph.len() && !visited[seed as usize] {
            visited[seed as usize] = true;
            queue.push_back((seed, 0));
        }
    }
    while let Some((idx, depth)) = queue.pop_front() {
        let is_test = graph.is_test[idx as usize];
        results.push(ImpactEntry { idx, depth, is_test });
        if depth < max_depth {
            for &caller in &graph.callers[idx as usize] {
                if !visited[caller as usize] {
                    visited[caller as usize] = true;
                    queue.push_back((caller, depth + 1));
                }
            }
        }
    }
    if !include_tests {
        results.retain(|e| !e.is_test);
    }
    results
}

fn build_impact_json(graph: &CallGraph, entries: &[ImpactEntry]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("f=file,l=lines,k=kind,n=name,d=depth,t=test"));
    let prod_count = entries.iter().filter(|e| e.depth > 0 && !e.is_test).count();
    let test_count = entries.iter().filter(|e| e.depth > 0 && e.is_test).count();
    map.insert("prod_callers".to_owned(), serde_json::json!(prod_count));
    map.insert("test_callers".to_owned(), serde_json::json!(test_count));
    let items: Vec<serde_json::Value> = entries.iter().map(|e| {
        let i = e.idx as usize;
        serde_json::json!({
            "f": relative_path(&graph.files[i]),
            "l": graph_lines_str(graph.lines[i]),
            "k": &graph.kinds[i],
            "n": &graph.names[i],
            "d": e.depth,
            "t": e.is_test,
        })
    }).collect();
    map.insert("results".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

fn trace_bfs(graph: &CallGraph, sources: &[u32], targets: &[u32]) -> Option<Vec<u32>> {
    use std::collections::VecDeque;
    let len = graph.len();
    let mut visited = vec![false; len];
    let mut parent: Vec<Option<u32>> = vec![None; len];
    let mut queue: VecDeque<u32> = VecDeque::new();
    let mut is_target = vec![false; len];
    for &t in targets {
        if (t as usize) < len { is_target[t as usize] = true; }
    }
    for &s in sources {
        if (s as usize) < len && !visited[s as usize] {
            visited[s as usize] = true;
            queue.push_back(s);
        }
    }
    while let Some(idx) = queue.pop_front() {
        if is_target[idx as usize] {
            let mut path = vec![idx];
            let mut current = idx;
            while let Some(p) = parent[current as usize] {
                path.push(p);
                current = p;
            }
            path.reverse();
            return Some(path);
        }
        for &callee in &graph.callees[idx as usize] {
            if !visited[callee as usize] {
                visited[callee as usize] = true;
                parent[callee as usize] = Some(idx);
                queue.push_back(callee);
            }
        }
    }
    None
}

fn build_trace_json(graph: &CallGraph, path: &[u32]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    map.insert("_s".to_owned(), serde_json::json!("f=file,l=lines,k=kind,n=name,t=test"));
    map.insert("hops".to_owned(), serde_json::json!(path.len() - 1));
    let items: Vec<serde_json::Value> = path.iter().map(|&idx| {
        let i = idx as usize;
        serde_json::json!({
            "f": relative_path(&graph.files[i]),
            "l": graph_lines_str(graph.lines[i]),
            "k": &graph.kinds[i],
            "n": &graph.names[i],
            "t": graph.is_test[i],
        })
    }).collect();
    map.insert("path".to_owned(), serde_json::Value::Array(items));
    serde_json::Value::Object(map)
}

fn graph_lines_str(lines: Option<(usize, usize)>) -> String {
    if let Some((s, e)) = lines {
        format!("{s}-{e}")
    } else {
        String::new()
    }
}
