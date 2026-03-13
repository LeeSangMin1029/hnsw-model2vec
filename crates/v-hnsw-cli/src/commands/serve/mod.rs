//! Serve command - Daemon server for fast embedding search.
//!
//! Keeps the embedding model loaded in memory to avoid repeated model loading.
//! Uses TCP socket for cross-platform compatibility.
//!
//! Extension methods (e.g. "update") are registered via `MethodHandler` callback
//! so the core daemon has no compile-time dependency on domain-specific modules.

pub mod daemon;
mod handler;

#[cfg(test)]
mod tests;

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::is_interrupted;
use daemon::DaemonState;

pub use handler::MethodHandler;

/// JSON-RPC request.
#[derive(Debug, Deserialize)]
pub(crate) struct JsonRpcRequest {
    pub id: u64,
    pub method: String,
    pub params: serde_json::Value,
}

/// JSON-RPC response.
#[derive(Debug, Serialize)]
pub(crate) struct JsonRpcResponse {
    pub id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

/// JSON-RPC error.
#[derive(Debug, Serialize)]
pub(crate) struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

/// Search request parameters.
#[derive(Debug, Deserialize)]
pub(crate) struct SearchParams {
    pub db: String,
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub tags: Vec<String>,
}

fn default_k() -> usize {
    10
}

/// Embed request parameters.
#[derive(Debug, Deserialize)]
pub(crate) struct EmbedParams {
    pub texts: Vec<String>,
}

/// Read the daemon port from global cache file.
pub fn read_port_file() -> Option<u16> {
    std::fs::read_to_string(super::common::cache_file("v-hnsw.port"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Write daemon state files (PID + port).
fn write_daemon_files(port: u16) -> Result<()> {
    let write = |name: &str, val: &str| -> Result<()> {
        let p = super::common::cache_file(name);
        std::fs::write(&p, val)
            .with_context(|| format!("Failed to write {}", p.display()))
    };
    write("v-hnsw.pid", &std::process::id().to_string())?;
    write("v-hnsw.port", &port.to_string())
}

/// Delete PID and port files.
fn cleanup_files() {
    for name in ["v-hnsw.pid", "v-hnsw.port"] {
        let _ = std::fs::remove_file(super::common::cache_file(name));
    }
}

/// Check if daemon is already running.
pub fn is_daemon_running() -> bool {
    if let Some(port) = read_port_file()
        && let Ok(addr) = format!("127.0.0.1:{port}").parse()
    {
        TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok()
    } else {
        false
    }
}

/// Run the daemon server.
///
/// `extra_handler` allows callers to register domain-specific RPC methods
/// (e.g. v-hnsw registers "update", v-code does not).
pub fn run(
    db_path: Option<PathBuf>,
    port: u16,
    timeout_secs: u64,
    background: bool,
    extra_handler: Option<MethodHandler>,
) -> Result<()> {
    if is_daemon_running()
        && let Some(existing_port) = read_port_file()
    {
        eprintln!("[daemon] Already running on port {existing_port}");
        return Ok(());
    }

    // Daemonize if requested
    if background {
        spawn_background(db_path.as_deref(), port, timeout_secs)?;
        return Ok(());
    }

    // Load state, optionally preloading one database
    let initial = db_path.as_deref()
        .map(|p| p.canonicalize())
        .transpose()
        .context("Database not found")?;
    let mut state = DaemonState::new(initial.as_deref())?;

    // Bind to port — no fallback to :0 to prevent duplicate daemons
    let listener = match TcpListener::bind(format!("127.0.0.1:{port}")) {
        Ok(l) => l,
        Err(_) if is_daemon_running() => {
            eprintln!("[daemon] Another daemon already owns port {port}, exiting");
            return Ok(());
        }
        Err(e) => {
            anyhow::bail!("Failed to bind to port {port}: {e} (is another program using it?)");
        }
    };

    let actual_port = listener.local_addr()?.port();

    write_daemon_files(actual_port)?;

    listener.set_nonblocking(true)?;

    tracing::info!(port = actual_port, "Daemon listening");
    eprintln!("[daemon] Listening on 127.0.0.1:{actual_port}");
    if timeout_secs == 0 {
        eprintln!("[daemon] Persistent mode (no idle timeout)");
    } else {
        eprintln!("[daemon] Idle timeout: {timeout_secs}s");
    }
    eprintln!("[daemon] Ready for connections");

    let mut last_activity = Instant::now();

    loop {
        if is_interrupted() {
            eprintln!("\n[daemon] Received shutdown signal");
            break;
        }

        if timeout_secs > 0 && last_activity.elapsed() > Duration::from_secs(timeout_secs) {
            eprintln!("[daemon] Idle timeout reached, shutting down");
            break;
        }

        // Unload embedding model if idle to free ~263 MB
        state.maybe_unload_model();
        // Evict idle databases to free mmap memory
        state.maybe_evict_databases();
        // Evict idle LSP servers to free subprocess memory
        #[cfg(feature = "code-intel")]
        state.maybe_evict_lsp();

        match listener.accept() {
            Ok((stream, addr)) => {
                eprintln!("[daemon] Connection from {addr}");
                if let Err(e) =
                    handler::handle_client(stream, &mut state, &mut last_activity, extra_handler)
                {
                    if e.to_string().contains("Shutdown requested") {
                        break;
                    }
                    eprintln!("[daemon] Client error: {e}");
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("[daemon] Accept error: {e}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    eprintln!("[daemon] Saving query cache...");
    if let Err(e) = state.save_cache() {
        eprintln!("[daemon] Failed to save query cache: {e}");
    }

    eprintln!("[daemon] Cleaning up...");
    cleanup_files();
    eprintln!("[daemon] Shutdown complete");

    Ok(())
}

/// Spawn daemon process in background.
fn spawn_background(db_path: Option<&Path>, port: u16, timeout_secs: u64) -> Result<()> {
    let port_str = port.to_string();
    let timeout_str = timeout_secs.to_string();
    let mut args: Vec<&str> = vec!["serve", "--port", &port_str, "--timeout", &timeout_str];

    let path_owned;
    if let Some(p) = db_path {
        path_owned = p.to_str().context("Non-UTF8 path")?.to_string();
        args.push(&path_owned);
    }

    super::common::spawn_detached(&args)?;
    std::thread::sleep(Duration::from_millis(500));
    Ok(())
}

/// Send a JSON-RPC request to the running daemon and return the result.
pub fn daemon_rpc(
    method: &str,
    params: serde_json::Value,
    read_timeout_secs: u64,
) -> Result<serde_json::Value> {
    let port = read_port_file()
        .ok_or_else(|| anyhow::anyhow!("Daemon not running (no port file)"))?;

    let addr: std::net::SocketAddr = format!("127.0.0.1:{port}")
        .parse()
        .context("Failed to parse socket address")?;
    let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(2))
        .context("Failed to connect to daemon")?;

    stream.set_read_timeout(Some(Duration::from_secs(read_timeout_secs)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    let request = serde_json::json!({"id": 0, "method": method, "params": params});
    writeln!(stream, "{}", serde_json::to_string(&request)?)?;
    stream.flush()?;

    let mut reader = BufReader::new(&stream);
    let mut response_line = String::new();
    reader.read_line(&mut response_line)?;

    let response: serde_json::Value = serde_json::from_str(&response_line)
        .context("Failed to parse daemon response")?;

    if let Some(err) = response.get("error") {
        anyhow::bail!("Daemon {method} failed: {err}");
    }

    response
        .get("result")
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("Empty response from daemon"))
}

/// Notify running daemon to reload indexes for a specific database.
pub fn notify_daemon_reload(db_path: &Path) -> Result<()> {
    let canonical = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    let db_str = canonical.to_str().unwrap_or("");
    let result = daemon_rpc("reload", serde_json::json!({"db": db_str}), 30)?;

    if result.get("status").and_then(|s| s.as_str()) == Some("reloaded") {
        Ok(())
    } else {
        anyhow::bail!("Daemon reload failed: {result}")
    }
}
