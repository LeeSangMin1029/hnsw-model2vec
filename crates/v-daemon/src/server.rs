//! TCP server loop — listens for JSON-RPC connections and dispatches to handlers.

use std::fs::File;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use fs2::FileExt;

use crate::interrupt::is_interrupted;
use crate::state::DaemonState;
use crate::watcher::FileWatcher;

/// Run the daemon server.
pub fn run(
    db_path: Option<PathBuf>,
    port: u16,
    timeout_secs: u64,
    background: bool,
) -> Result<()> {
    if background {
        // Background mode: check before spawning (no lock needed here,
        // the spawned process will acquire the lock itself).
        if crate::client::is_running()
            && let Some(existing_port) = crate::client::read_port()
        {
            eprintln!("[daemon] Already running on port {existing_port}");
            return Ok(());
        }
        return spawn_background(db_path.as_deref(), port, timeout_secs);
    }

    // Acquire an exclusive lockfile to prevent multiple daemon instances.
    // The lock is held for the lifetime of `_lock_file`; the OS releases it on process exit.
    let lock_path = port_dir().join("v-daemon.lock");
    std::fs::create_dir_all(lock_path.parent().unwrap_or_else(|| Path::new("."))).ok();
    let lock_file =
        File::create(&lock_path).with_context(|| format!("Cannot create {}", lock_path.display()))?;
    if lock_file.try_lock_exclusive().is_err() {
        eprintln!("[daemon] Already running (lockfile held by another process)");
        return Ok(());
    }
    // Keep `_lock_file` alive so the lock persists until shutdown.
    let _lock_file = lock_file;

    // On Windows, create a Job Object so that all child processes (rust-analyzer,
    // proc-macro-server) are automatically killed when the daemon exits — even
    // on crashes where Drop destructors don't run.
    #[cfg(windows)]
    let _job = setup_job_object();

    let initial = db_path
        .as_deref()
        .map(|p| p.canonicalize())
        .transpose()
        .context("Database not found")?;
    let mut state = DaemonState::new(initial.as_deref())?;

    // Bind port FIRST — OS backlog accepts connections while RA loads.
    // Client connect() succeeds immediately, request waits in socket buffer.
    // After RA loads, daemon accept()s and processes queued requests.
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

    eprintln!("[daemon] Listening on 127.0.0.1:{actual_port}");
    if timeout_secs == 0 {
        eprintln!("[daemon] Persistent mode (no idle timeout)");
    } else {
        eprintln!("[daemon] Idle timeout: {timeout_secs}s");
    }

    // Load RA (blocks, but client connections queue in OS backlog).
    let workspace_root = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    match v_lsp::instance::RaInstance::spawn(&workspace_root) {
        Ok(ra) => {
            eprintln!("[daemon] RA instance loaded");
            state.ra = Some(ra);
        }
        Err(e) => eprintln!("[daemon] RA spawn failed: {e}"),
    }

    eprintln!("[daemon] Ready");

    let mut last_activity = Instant::now();

    // Start file watcher for the project root (if a DB is configured).
    let mut watcher = db_path
        .as_deref()
        .and_then(|db| {
            let db_canon = db.canonicalize().ok()?;
            let project_root = v_code_intel::helpers::find_project_root(&db_canon)?;
            eprintln!("[daemon] Project root: {}", project_root.display());
            FileWatcher::new(&[project_root], db_canon)
        });

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

        // Poll file watcher for source changes → update RA + invalidate graph cache.
        if let Some(ref mut w) = watcher {
            let changed = w.poll_changes();
            if !changed.is_empty() {
                eprintln!("[watcher] {} file(s) changed", changed.len());
                if let Some(ref mut ra) = state.ra {
                    let updates: Vec<(String, String)> = changed.iter()
                        .filter_map(|path| {
                            let content = std::fs::read_to_string(path).ok()?;
                            let abs = path.canonicalize().unwrap_or_else(|_| path.clone());
                            let abs_str = v_hnsw_core::strip_unc_prefix(&abs.to_string_lossy())
                                .replace('\\', "/");
                            let root = ra.workspace_root().to_string_lossy().replace('\\', "/");
                            let root = v_hnsw_core::strip_unc_prefix(&root);
                            let rel = abs_str.strip_prefix(root)
                                .and_then(|s| s.strip_prefix('/'))
                                .unwrap_or(&abs_str);
                            Some((rel.to_owned(), content))
                        })
                        .collect();
                    if !updates.is_empty() {
                        eprintln!("[watcher] updating {} file(s) in RA...", updates.len());
                        let n = ra.update_files(&updates);
                        eprintln!("[watcher] RA update done ({n} files)");
                    }
                }
                w.invalidate_graph_cache();
            }
        }

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
    write("v-daemon.port", &port.to_string())?;

    // Store current exe mtime so clients can detect stale binaries.
    if let Ok(exe) = std::env::current_exe()
        && let Ok(meta) = exe.metadata()
            && let Ok(modified) = meta.modified()
                && let Ok(dur) = modified.duration_since(std::time::UNIX_EPOCH) {
                    let _ = write("v-daemon.mtime", &dur.as_secs().to_string());
                }

    Ok(())
}

fn cleanup_files() {
    let dir = port_dir();
    for name in ["v-daemon.pid", "v-daemon.port", "v-daemon.mtime", "v-daemon.lock"] {
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

// ── Job Object (Windows) ────────────────────────────────────────────

/// Create a Windows Job Object with `KILL_ON_JOB_CLOSE` and assign the
/// current process to it. When the daemon exits (including crashes), the
/// OS automatically terminates all child processes (rust-analyzer, etc.).
///
/// Returns the job handle wrapped in a struct that closes it on drop.
/// If Job Object creation fails, returns `None` — the daemon still runs,
/// just without automatic child cleanup.
#[cfg(windows)]
#[expect(unsafe_code, reason = "Windows Job Object API requires FFI calls")]
fn setup_job_object() -> Option<JobObjectGuard> {
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    unsafe {
        let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if job.is_null() {
            eprintln!("[daemon] Failed to create Job Object");
            return None;
        }

        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;

        let ok = SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &info as *const _ as *const _,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        );
        if ok == 0 {
            eprintln!("[daemon] Failed to set Job Object limits");
            CloseHandle(job);
            return None;
        }

        // Assign current process to the job.
        let current = windows_sys::Win32::System::Threading::GetCurrentProcess();
        let ok = AssignProcessToJobObject(job, current);
        if ok == 0 {
            eprintln!("[daemon] Failed to assign process to Job Object");
            CloseHandle(job);
            return None;
        }

        eprintln!("[daemon] Job Object active — child processes will auto-terminate on exit");
        Some(JobObjectGuard(job))
    }
}

#[cfg(windows)]
struct JobObjectGuard(windows_sys::Win32::Foundation::HANDLE);

#[cfg(windows)]
#[expect(unsafe_code, reason = "CloseHandle FFI call")]
impl Drop for JobObjectGuard {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}
