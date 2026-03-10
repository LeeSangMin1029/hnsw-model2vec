//! TCP client connection handler.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
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
        "stats" => code_intel::stats_as_json(&db),
        "def" => {
            let name = params.name.as_deref()
                .context("missing 'name' param for def")?;
            code_intel::def_as_json(&db, name)
        }
        "refs" => {
            let name = params.name.as_deref()
                .context("missing 'name' param for refs")?;
            code_intel::refs_as_json(&db, name)
        }
        "symbols" => code_intel::symbols_as_json(&db, params.name.as_deref(), params.kind.as_deref()),
        "gather" => {
            let symbol = params.symbol.as_deref()
                .context("missing 'symbol' param for gather")?;
            code_intel::gather_as_json(&db, symbol, params.depth.unwrap_or(2), params.k, params.include_tests)
        }
        "impact" => {
            let symbol = params.symbol.as_deref()
                .context("missing 'symbol' param for impact")?;
            code_intel::impact_as_json(&db, symbol, params.depth.unwrap_or(2), params.include_tests)
        }
        "trace" => {
            let from = params.from.as_deref()
                .context("missing 'from' param for trace")?;
            let to = params.to.as_deref()
                .context("missing 'to' param for trace")?;
            code_intel::trace_as_json(&db, from, to)
        }
        "detail" => {
            let symbol = params.symbol.as_deref()
                .or(params.name.as_deref())
                .context("missing 'symbol' or 'name' param for detail")?;
            code_intel::detail_as_json(&db, symbol)
        }
        _ => anyhow::bail!("unknown code-intel method: {method}"),
    }
}

