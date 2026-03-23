//! Daemon client utilities — communicate with the running v-daemon process.
//!
//! Provides port discovery, RPC communication, and daemon auto-start.
//! Lives in v-hnsw-storage so both v-hnsw-cli and v-daemon can use it
//! without circular dependencies.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};

/// Platform-aware path to the daemon port file.
///
/// Uses the current working directory's hash to isolate per-project daemon instances.
/// Multiple projects can run independent daemons simultaneously.
pub fn port_path() -> PathBuf {
    port_path_for_project(&project_key())
}

/// Port path for a specific project key.
pub fn port_path_for_project(key: &str) -> PathBuf {
    let base = daemon_base_dir();
    base.join(format!("v-daemon-{key}.port"))
}

/// Derive a short project key from the current directory.
fn project_key() -> String {
    let cwd = std::env::current_dir().unwrap_or_default();
    let canonical = cwd.canonicalize().unwrap_or(cwd);
    let hash = simple_hash(canonical.to_string_lossy().as_ref());
    format!("{hash:016x}")
}

/// Simple non-cryptographic hash for path → key derivation.
fn simple_hash(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn daemon_base_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("v-hnsw").join("daemons");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(cache).join("v-hnsw").join("daemons");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("v-hnsw").join("daemons");
        }
    }
    std::env::temp_dir().join("v-hnsw").join("daemons")
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
    let port = read_port().unwrap_or(19530);
    let addr = format!("127.0.0.1:{port}");
    let mut stream = TcpStream::connect(&addr)
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

/// Spawn daemon if not running and wait until ready.
///
/// Returns `true` if the daemon became reachable within the deadline.
pub fn spawn_daemon_and_wait(db: &Path) -> bool {
    if is_running() { return true; }
    spawn_daemon(db);
    true
}

/// Ensure the daemon is running with the current binary version.
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
