//! Daemon client utilities — communicate with the running v-daemon process.
//!
//! Provides port discovery, RPC communication, and daemon auto-start.
//! Lives in v-hnsw-storage so both v-hnsw-cli and v-daemon can use it
//! without circular dependencies.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

/// Platform-aware path to the daemon port file.
pub fn port_path() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local)
                .join("v-hnsw")
                .join("cache")
                .join("v-daemon.port");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(cache).join("v-hnsw").join("v-daemon.port");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home)
                .join(".cache")
                .join("v-hnsw")
                .join("v-daemon.port");
        }
    }
    std::env::temp_dir().join("v-hnsw").join("v-daemon.port")
}

/// Read the daemon port from the port file.
pub fn read_port() -> Option<u16> {
    std::fs::read_to_string(port_path())
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Check if the daemon is running and accepting connections.
pub fn is_running() -> bool {
    if let Some(port) = read_port() {
        let addr = format!("127.0.0.1:{port}");
        if let Ok(addr) = addr.parse() {
            return TcpStream::connect_timeout(&addr, Duration::from_millis(100)).is_ok();
        }
    }
    false
}

/// Send a JSON-RPC request to the running daemon and return the result.
pub fn daemon_rpc(
    method: &str,
    params: serde_json::Value,
    read_timeout_secs: u64,
) -> Result<serde_json::Value> {
    let port = read_port()
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

/// Send a JSON-RPC request to the daemon without waiting for a response.
///
/// Spawns a background thread that connects, sends the request, and reads
/// the response — but the caller returns immediately (fire-and-forget).
/// Send a JSON-RPC request to the daemon without waiting for a response.
///
/// Connects and sends the request on the current thread (so the message
/// reaches the daemon before the process exits), but does NOT read the
/// response — the caller returns immediately after flushing.
pub fn daemon_rpc_fire_and_forget(method: &str, params: serde_json::Value) {
    let Some(port) = read_port() else { return };
    let Ok(addr) = format!("127.0.0.1:{port}").parse() else { return };
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_secs(2)) else { return };

    let request = serde_json::json!({"id": 0, "method": method, "params": params});
    let Ok(json) = serde_json::to_string(&request) else { return };
    let _ = writeln!(stream, "{json}");
    let _ = stream.flush();
    // Don't read response — daemon will process asynchronously.
}

/// Notify the running daemon to reload indexes for a database.
pub fn notify_reload(db_path: &Path) -> Result<()> {
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

/// Spawn daemon and wait until ready (max 10s).
///
/// Returns `true` if the daemon became reachable within the deadline.
pub fn spawn_daemon_and_wait(db: &Path) -> bool {
    spawn_daemon(db);
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if is_running() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    false
}

/// Ensure the daemon is running with the current binary version.
///
/// If the binary changed since the daemon started, shuts down the old
/// daemon and spawns a fresh one. Returns `true` if a daemon is ready.
pub fn ensure_daemon(db: &Path) -> bool {
    if is_running() && !is_binary_stale() {
        return true;
    }
    if is_running() && is_binary_stale() {
        let _ = daemon_rpc("shutdown", serde_json::json!(null), 5);
        std::thread::sleep(Duration::from_millis(500));
    }
    spawn_daemon_and_wait(db)
}

/// Check whether the current executable is newer than the running daemon.
fn is_binary_stale() -> bool {
    let mtime_path = port_path()
        .parent()
        .map(|d| d.join("v-daemon.mtime"))
        .unwrap_or_else(|| std::env::temp_dir().join("v-hnsw").join("v-daemon.mtime"));

    let Ok(stored) = std::fs::read_to_string(&mtime_path) else {
        return true;
    };
    let Ok(stored_ts) = stored.trim().parse::<i64>() else {
        return true;
    };

    let Ok(exe) = std::env::current_exe() else {
        return true;
    };
    let Ok(meta) = exe.metadata() else {
        return true;
    };
    let Ok(modified) = meta.modified() else {
        return true;
    };

    let current_ts = modified
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);

    current_ts != stored_ts
}

/// Auto-start `v-daemon` in background if not already running.
///
/// Looks for `v-daemon` next to the current executable, then in PATH.
pub fn spawn_daemon(db: &Path) {
    if is_running() {
        return;
    }

    let db_str = db.to_string_lossy();
    let daemon_name = if cfg!(windows) {
        "v-daemon.exe"
    } else {
        "v-daemon"
    };

    let daemon_exe = std::env::current_exe()
        .ok()
        .map(|e| e.with_file_name(daemon_name))
        .filter(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from(daemon_name));

    let mut cmd = std::process::Command::new(&daemon_exe);
    cmd.args(["--db", &db_str, "--background"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000200 | 0x08000000);
    }

    let _ = cmd.spawn();
}
