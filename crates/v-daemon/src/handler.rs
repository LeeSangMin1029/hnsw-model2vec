//! JSON-RPC request handler — dispatches built-in + extension methods.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::state::DaemonState;

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    id: u64,
    method: String,
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

#[derive(Debug, Deserialize)]
struct SearchParams {
    db: String,
    query: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    tags: Vec<String>,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Deserialize)]
struct EmbedParams {
    texts: Vec<String>,
}

/// Handle a single client connection.
pub fn handle_client(
    stream: TcpStream,
    state: &mut DaemonState,
    last_activity: &mut Instant,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(300)))?;

    let mut reader = BufReader::new(&stream);
    let mut writer = &stream;

    let mut line = String::new();
    reader.read_line(&mut line)?;

    *last_activity = Instant::now();

    let request: JsonRpcRequest = serde_json::from_str(&line)
        .with_context(|| format!("Failed to parse request: {}", line.trim()))?;

    let response = dispatch(&request, state, &mut writer)?;

    let response_json = serde_json::to_string(&response)?;
    writeln!(writer, "{response_json}")?;
    writer.flush()?;

    Ok(())
}

fn dispatch(
    request: &JsonRpcRequest,
    state: &mut DaemonState,
    writer: &mut &TcpStream,
) -> Result<JsonRpcResponse> {
    let id = request.id;

    match request.method.as_str() {
        "search" => {
            let params: SearchParams =
                serde_json::from_value(request.params.clone()).context("Invalid search params")?;
            let db_path = PathBuf::from(&params.db);
            match state.search(&db_path, &params.query, params.k, params.tags) {
                Ok(result) => Ok(ok_response(id, serde_json::to_value(result)?)),
                Err(e) => Ok(err_response(id, -1, e.to_string())),
            }
        }
        "ping" => Ok(ok_response(id, serde_json::json!({"status": "ok"}))),
        "reload" => {
            let db_path: String = request.params
                .get("db").and_then(|d| d.as_str())
                .unwrap_or("").to_string();
            match state.reload(&PathBuf::from(&db_path)) {
                Ok(()) => Ok(ok_response(id, serde_json::json!({"status": "reloaded"}))),
                Err(e) => Ok(err_response(id, -2, format!("Reload failed: {e}"))),
            }
        }
        "embed" => {
            let params: EmbedParams =
                serde_json::from_value(request.params.clone()).context("Invalid embed params")?;
            match state.embed(&params.texts.iter().map(|s| s.as_str()).collect::<Vec<_>>()) {
                Ok(embeddings) => Ok(ok_response(id, serde_json::json!({"embeddings": embeddings}))),
                Err(e) => Ok(err_response(id, -3, format!("Embed failed: {e}"))),
            }
        }
        "shutdown" => {
            let response = ok_response(id, serde_json::json!({"status": "shutting_down"}));
            let response_json = serde_json::to_string(&response)?;
            writeln!(writer, "{response_json}")?;
            writer.flush()?;
            anyhow::bail!("Shutdown requested");
        }
        // Extension methods: document + code
        "update" => match crate::doc::handle_update(request.params.clone(), state) {
            Ok(v) => Ok(ok_response(id, v)),
            Err(e) => Ok(err_response(id, -4, e.to_string())),
        },
        "graph/build" => match crate::code::handle_graph_build(request.params.clone(), state.ra.as_ref()) {
            Ok(v) => Ok(ok_response(id, v)),
            Err(e) => Ok(err_response(id, -4, e.to_string())),
        },
        "ra/collect-types" => match crate::code::handle_collect_types(request.params.clone(), state.ra.as_ref()) {
            Ok(v) => Ok(ok_response(id, v)),
            Err(e) => Ok(err_response(id, -5, e.to_string())),
        },
        "code/chunk" => match crate::code::handle_chunk_files(request.params.clone(), state.ra.as_ref()) {
            Ok(v) => Ok(ok_response(id, v)),
            Err(e) => Ok(err_response(id, -6, e.to_string())),
        },
        method => Ok(err_response(id, -32601, format!("Unknown method: {method}"))),
    }
}

fn ok_response(id: u64, result: serde_json::Value) -> JsonRpcResponse {
    JsonRpcResponse { id, result: Some(result), error: None }
}

fn err_response(id: u64, code: i32, message: String) -> JsonRpcResponse {
    JsonRpcResponse { id, result: None, error: Some(JsonRpcError { code, message }) }
}
