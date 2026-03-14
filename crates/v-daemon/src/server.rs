//! TCP server loop — listens for JSON-RPC connections and dispatches to handlers.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::interrupt::is_interrupted;
use crate::state::DaemonState;

/// Run the daemon server.
pub fn run(
    db_path: Option<PathBuf>,
    port: u16,
    timeout_secs: u64,
    background: bool,
) -> Result<()> {
    if crate::client::is_running()
        && let Some(existing_port) = crate::client::read_port()
    {
        eprintln!("[daemon] Already running on port {existing_port}");
        return Ok(());
    }

    if background {
        return spawn_background(db_path.as_deref(), port, timeout_secs);
    }

    let initial = db_path
        .as_deref()
        .map(|p| p.canonicalize())
        .transpose()
        .context("Database not found")?;
    let mut state = DaemonState::new(initial.as_deref())?;

    let listener = match TcpListener::bind(format!("127.0.0.1:{port}")) {
        Ok(l) => l,
        Err(_) if crate::client::is_running() => {
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

        state.maybe_unload_model();
        state.maybe_evict_databases();
        state.maybe_evict_lsp();

        match listener.accept() {
            Ok((stream, addr)) => {
                eprintln!("[daemon] Connection from {addr}");
                if let Err(e) =
                    crate::handler::handle_client(stream, &mut state, &mut last_activity)
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

// ── Port/PID file management ────────────────────────────────────────────

fn port_dir() -> PathBuf {
    let p = crate::client::port_path();
    p.parent().map(|d| d.to_path_buf()).unwrap_or_else(std::env::temp_dir)
}

fn write_daemon_files(port: u16) -> Result<()> {
    let dir = port_dir();
    std::fs::create_dir_all(&dir).ok();
    let write = |name: &str, val: &str| -> Result<()> {
        let p = dir.join(name);
        std::fs::write(&p, val)
            .with_context(|| format!("Failed to write {}", p.display()))
    };
    write("v-daemon.pid", &std::process::id().to_string())?;
    write("v-daemon.port", &port.to_string())
}

fn cleanup_files() {
    let dir = port_dir();
    for name in ["v-daemon.pid", "v-daemon.port"] {
        let _ = std::fs::remove_file(dir.join(name));
    }
}

// ── Background spawning ─────────────────────────────────────────────────

fn spawn_background(db_path: Option<&Path>, port: u16, timeout_secs: u64) -> Result<()> {
    let port_str = port.to_string();
    let timeout_str = timeout_secs.to_string();

    let exe = std::env::current_exe()?;
    let mut args: Vec<&str> = vec!["--port", &port_str, "--timeout", &timeout_str];

    let path_owned;
    if let Some(p) = db_path {
        path_owned = p.to_str().context("Non-UTF8 path")?.to_string();
        args.push("--db");
        args.push(&path_owned);
    }

    let mut cmd = std::process::Command::new(&exe);
    cmd.args(&args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x00000200 | 0x08000000);
    }

    cmd.spawn().context("Failed to spawn background daemon")?;
    std::thread::sleep(Duration::from_millis(500));
    Ok(())
}
