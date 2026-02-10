//! TCP client connection handler.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use super::daemon::DaemonState;
use super::{EmbedParams, JsonRpcError, JsonRpcRequest, JsonRpcResponse, SearchParams};

/// Handle a single client connection.
pub(crate) fn handle_client(
    stream: TcpStream,
    state: &mut DaemonState,
    last_activity: &mut Instant,
) -> Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;

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
    writeln!(writer, "{}", response_json)?;
    writer.flush()?;

    Ok(())
}
