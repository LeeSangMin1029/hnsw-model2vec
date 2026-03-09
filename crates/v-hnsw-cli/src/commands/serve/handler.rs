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
// Reuse shared helpers from code_intel module instead of reimplementing.

fn ci_stats(db: &Path) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let stats = code_intel::build_stats(&chunks);
    Ok(code_intel::stats_to_json(&stats))
}

fn ci_def(db: &Path, name: &str) -> Result<serde_json::Value> {
    let chunks = code_intel::load_chunks(db)?;
    let name_lower = name.to_lowercase();
    let matches: Vec<&code_intel::parse::CodeChunk> = chunks.iter().filter(|c| {
        c.name.to_lowercase() == name_lower
            || c.name.to_lowercase().ends_with(&format!("::{name_lower}"))
    }).collect();
    Ok(code_intel::grouped_json(&matches))
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
    Ok(code_intel::grouped_json_refs(&refs))
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
    Ok(code_intel::grouped_json(&filtered))
}

fn ci_gather(db: &Path, symbol: &str, depth: u32, k: usize, include_tests: bool) -> Result<serde_json::Value> {
    use code_intel::gather::{bfs_directed, merge_entries, Direction};

    let graph = code_intel::context::load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    let forward = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Forward);
    let reverse = bfs_directed(&graph, &seeds, depth, include_tests, Direction::Reverse);
    let mut entries = merge_entries(forward, reverse);
    entries.truncate(k);
    Ok(code_intel::build_bfs_json(&graph, &entries))
}

fn ci_impact(db: &Path, symbol: &str, depth: u32, include_tests: bool) -> Result<serde_json::Value> {
    use code_intel::impact::bfs_reverse;

    let graph = code_intel::context::load_or_build_graph(db)?;
    let seeds = graph.resolve(symbol);
    if seeds.is_empty() {
        return Ok(serde_json::json!({"results": [], "message": format!("No symbol found matching \"{symbol}\"")}));
    }
    let all_entries = bfs_reverse(&graph, &seeds, depth);
    let entries: Vec<_> = if include_tests {
        all_entries
    } else {
        all_entries.into_iter().filter(|e| !e.is_test).collect()
    };
    Ok(code_intel::build_bfs_json(&graph, &entries))
}

fn ci_trace(db: &Path, from: &str, to: &str) -> Result<serde_json::Value> {
    use code_intel::trace::{bfs_shortest_path, build_json};

    let graph = code_intel::context::load_or_build_graph(db)?;
    let sources = graph.resolve(from);
    let targets = graph.resolve(to);
    if sources.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{from}\"")}));
    }
    if targets.is_empty() {
        return Ok(serde_json::json!({"path": null, "message": format!("No symbol found matching \"{to}\"")}));
    }
    match bfs_shortest_path(&graph, &sources, &targets) {
        Some(path) => Ok(build_json(&graph, &path)),
        None => Ok(serde_json::json!({"path": null, "hops": null})),
    }
}

fn ci_detail(db: &Path, symbol: &str) -> Result<serde_json::Value> {
    match code_intel::reason::load_reason(db, symbol)? {
        Some(entry) => Ok(serde_json::to_value(entry)?),
        None => Ok(serde_json::json!({"symbol": symbol, "found": false})),
    }
}

