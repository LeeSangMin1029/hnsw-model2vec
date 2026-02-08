//! Serve command - Daemon server for fast embedding search.
//!
//! Keeps the embedding model loaded in memory to avoid repeated model loading.
//! Uses TCP socket for cross-platform compatibility.

mod daemon;
mod handler;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::is_interrupted;
use daemon::DaemonState;

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

/// Hash a path to create a unique identifier.
fn hash_path(path: &Path) -> String {
    let mut hasher = DefaultHasher::new();
    path.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

/// Get the PID file path for a database.
pub fn get_pid_file(db_path: &Path) -> PathBuf {
    let hash = hash_path(db_path);
    let cache_dir = super::common::cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();
    cache_dir.join(format!("{}.pid", hash))
}

/// Get the port file path for a database.
pub fn get_port_file(db_path: &Path) -> PathBuf {
    let hash = hash_path(db_path);
    let cache_dir = super::common::cache_dir();
    std::fs::create_dir_all(&cache_dir).ok();
    cache_dir.join(format!("{}.port", hash))
}

/// Read the port from port file.
pub fn read_port_file(db_path: &Path) -> Option<u16> {
    let port_file = get_port_file(db_path);
    std::fs::read_to_string(&port_file)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Write the port to port file.
fn write_port_file(db_path: &Path, port: u16) -> Result<()> {
    let port_file = get_port_file(db_path);
    std::fs::write(&port_file, port.to_string())
        .with_context(|| format!("Failed to write port file: {}", port_file.display()))
}

/// Write PID file.
fn write_pid_file(db_path: &Path) -> Result<()> {
    let pid_file = get_pid_file(db_path);
    let pid = std::process::id();
    std::fs::write(&pid_file, pid.to_string())
        .with_context(|| format!("Failed to write PID file: {}", pid_file.display()))
}

/// Delete PID and port files.
fn cleanup_files(db_path: &Path) {
    let pid_file = get_pid_file(db_path);
    let port_file = get_port_file(db_path);
    let _ = std::fs::remove_file(&pid_file);
    let _ = std::fs::remove_file(&port_file);
}

/// Check if daemon is already running for this database.
pub fn is_daemon_running(db_path: &Path) -> bool {
    if let Some(port) = read_port_file(db_path)
        && let Ok(addr) = format!("127.0.0.1:{}", port).parse()
    {
        TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok()
    } else {
        false
    }
}

/// Run the daemon server.
pub fn run(db_path: PathBuf, port: u16, timeout_secs: u64, background: bool) -> Result<()> {
    let db_path = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    if is_daemon_running(&db_path)
        && let Some(existing_port) = read_port_file(&db_path)
    {
        eprintln!("[daemon] Already running on port {}", existing_port);
        return Ok(());
    }

    // Daemonize if requested
    if background {
        spawn_background(&db_path, port, timeout_secs)?;
        return Ok(());
    }

    // Load state (model, indexes)
    let mut state = DaemonState::new(&db_path)?;

    // Bind to port
    let listener = TcpListener::bind(format!("127.0.0.1:{}", port))
        .or_else(|_| TcpListener::bind("127.0.0.1:0"))
        .context("Failed to bind to any port")?;

    let actual_port = listener.local_addr()?.port();

    write_pid_file(&db_path)?;
    write_port_file(&db_path, actual_port)?;

    listener.set_nonblocking(true)?;

    tracing::info!(port = actual_port, "Daemon listening");
    eprintln!("[daemon] Listening on 127.0.0.1:{}", actual_port);
    eprintln!("[daemon] Idle timeout: {}s", timeout_secs);
    eprintln!("[daemon] PID file: {}", get_pid_file(&db_path).display());
    eprintln!("[daemon] Ready for connections");

    let mut last_activity = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);

    loop {
        if is_interrupted() {
            eprintln!("\n[daemon] Received shutdown signal");
            break;
        }

        if last_activity.elapsed() > timeout {
            eprintln!("[daemon] Idle timeout reached, shutting down");
            break;
        }

        match listener.accept() {
            Ok((stream, addr)) => {
                eprintln!("[daemon] Connection from {}", addr);
                if let Err(e) =
                    handler::handle_client(stream, &mut state, &db_path, &mut last_activity)
                {
                    if e.to_string().contains("Shutdown requested") {
                        break;
                    }
                    eprintln!("[daemon] Client error: {}", e);
                }
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => {
                eprintln!("[daemon] Accept error: {}", e);
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }

    eprintln!("[daemon] Cleaning up...");
    cleanup_files(&db_path);
    eprintln!("[daemon] Shutdown complete");

    Ok(())
}

/// Spawn daemon process in background.
fn spawn_background(db_path: &Path, port: u16, timeout_secs: u64) -> Result<()> {
    let exe = std::env::current_exe()?;
    let path_str = db_path.to_str().context("Non-UTF8 path")?;

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x00000200;
        const DETACHED_PROCESS: u32 = 0x00000008;

        std::process::Command::new(exe)
            .args([
                "serve",
                path_str,
                "--port",
                &port.to_string(),
                "--timeout",
                &timeout_secs.to_string(),
            ])
            .creation_flags(CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS)
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn background daemon")?;
    }

    #[cfg(not(windows))]
    {
        std::process::Command::new(exe)
            .args([
                "serve",
                path_str,
                "--port",
                &port.to_string(),
                "--timeout",
                &timeout_secs.to_string(),
            ])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .context("Failed to spawn background daemon")?;
    }

    std::thread::sleep(Duration::from_millis(500));
    Ok(())
}

/// Notify running daemon to reload indexes from disk.
pub fn notify_daemon_reload(db_path: &Path) -> Result<()> {
    let canonical = db_path
        .canonicalize()
        .with_context(|| format!("Database not found: {}", db_path.display()))?;

    let port = read_port_file(&canonical)
        .ok_or_else(|| anyhow::anyhow!("Daemon not running (no port file)"))?;

    let addr = format!("127.0.0.1:{}", port)
        .parse()
        .context("Failed to parse socket address")?;
    let mut stream = TcpStream::connect_timeout(
        &addr,
        Duration::from_secs(2),
    )
    .context("Failed to connect to daemon")?;

    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;

    let request = serde_json::json!({"id": 0, "method": "reload", "params": {}});
    writeln!(stream, "{}", serde_json::to_string(&request)?)?;
    stream.flush()?;

    let mut reader = BufReader::new(&stream);
    let mut response = String::new();
    reader.read_line(&mut response)?;

    if response.contains("reloaded") {
        Ok(())
    } else {
        anyhow::bail!("Daemon reload failed: {}", response.trim())
    }
}
