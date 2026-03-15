//! File watcher — monitors source files and invalidates graph cache on changes.
//!
//! Uses `notify` crate for cross-platform filesystem events.
//! Debounces rapid changes (2 seconds) and filters to source file extensions only.
//! On change: invalidates graph cache + triggers automatic re-indexing via `v-code add`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};

/// Source file extensions to watch.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "js", "go", "c", "cpp", "java", "tsx", "jsx",
];

/// Directories to skip entirely.
const IGNORED_DIRS: &[&str] = &[
    ".git", "target", "node_modules", "__pycache__", ".mypy_cache",
    "dist", "build", ".next", ".nuxt",
];

/// Debounce window in seconds.
const DEBOUNCE_SECS: u64 = 2;

/// Guards against concurrent auto-reindex runs.
static REINDEX_RUNNING: AtomicBool = AtomicBool::new(false);

/// Watches project directories for source file changes.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<PathBuf>,
    pending: HashSet<PathBuf>,
    last_event: Option<Instant>,
    db_path: PathBuf,
    input_path: Option<PathBuf>,
}

impl FileWatcher {
    /// Start watching the given directories recursively.
    ///
    /// `input_path` is the source root for `v-code add` (read from `DbConfig`).
    /// Returns `None` if watcher creation fails (non-fatal — daemon runs without watching).
    pub fn new(watch_dirs: &[PathBuf], db_path: PathBuf, input_path: Option<PathBuf>) -> Option<Self> {
        let (tx, rx) = mpsc::channel();

        let sender = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| {
            let Ok(event) = res else { return };

            // Only care about create/modify/remove events.
            if !matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) {
                return;
            }

            for path in event.paths {
                if is_source_file(&path) && !is_in_ignored_dir(&path) {
                    // Channel send failure is non-fatal (receiver dropped).
                    let _ = sender.send(path);
                }
            }
        })
        .ok()?;

        for dir in watch_dirs {
            if dir.is_dir()
                && let Err(e) = watcher.watch(dir, RecursiveMode::Recursive)
            {
                eprintln!("[watcher] Failed to watch {}: {e}", dir.display());
            }
        }

        eprintln!(
            "[watcher] Watching {} director{} for source changes",
            watch_dirs.len(),
            if watch_dirs.len() == 1 { "y" } else { "ies" }
        );

        Some(Self {
            _watcher: watcher,
            rx,
            pending: HashSet::new(),
            last_event: None,
            db_path,
            input_path,
        })
    }

    /// Poll for debounced changes. Returns the list of changed source files
    /// if the debounce window has elapsed since the last event.
    ///
    /// Call this from the main loop (~100ms intervals).
    pub fn poll_changes(&mut self) -> Vec<PathBuf> {
        // Drain all pending events from the channel.
        while let Ok(path) = self.rx.try_recv() {
            self.pending.insert(path);
            self.last_event = Some(Instant::now());
        }

        // If we have pending changes and the debounce window has passed, flush them.
        if !self.pending.is_empty()
            && let Some(last) = self.last_event
            && last.elapsed() >= Duration::from_secs(DEBOUNCE_SECS)
        {
            let changed: Vec<PathBuf> = self.pending.drain().collect();
            self.last_event = None;
            return changed;
        }

        Vec::new()
    }

    /// Delete the graph cache file for the associated DB.
    pub fn invalidate_graph_cache(&self) {
        let cache_path = self.db_path.join("cache").join("graph.bin");
        if cache_path.exists() {
            match std::fs::remove_file(&cache_path) {
                Ok(()) => eprintln!("[watcher] Graph cache invalidated: {}", cache_path.display()),
                Err(e) => eprintln!("[watcher] Failed to remove graph cache: {e}"),
            }
        }
    }

    /// Trigger automatic re-indexing via `v-code add` subprocess.
    ///
    /// Spawns `v-code add <db> <input>` in the background. Only one reindex
    /// runs at a time — subsequent requests are skipped until the current one finishes.
    pub fn auto_reindex(&self) {
        let Some(ref input) = self.input_path else {
            eprintln!("[watcher] No input_path configured — skipping auto-reindex");
            return;
        };

        // Guard: only one reindex at a time.
        if REINDEX_RUNNING.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
            eprintln!("[watcher] Auto-reindex already in progress, skipping");
            return;
        }

        let db = self.db_path.clone();
        let input = input.clone();

        std::thread::spawn(move || {
            eprintln!("[watcher] Starting auto-reindex: v-code add {} {}", db.display(), input.display());
            let result = std::process::Command::new("v-code")
                .args(["add"])
                .arg(&db)
                .arg(&input)
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .output();

            match result {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    if output.status.success() {
                        eprintln!("[watcher] Auto-reindex completed successfully");
                        for line in stdout.lines().chain(stderr.lines()) {
                            if !line.is_empty() {
                                eprintln!("[watcher]   {line}");
                            }
                        }
                    } else {
                        eprintln!("[watcher] Auto-reindex failed (exit: {})", output.status);
                        for line in stderr.lines() {
                            eprintln!("[watcher]   {line}");
                        }
                    }
                }
                Err(e) => eprintln!("[watcher] Failed to spawn v-code: {e}"),
            }

            REINDEX_RUNNING.store(false, Ordering::SeqCst);
        });
    }
}

/// Check if a path has a source file extension.
fn is_source_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| SOURCE_EXTENSIONS.contains(&ext))
}

/// Check if a path is inside an ignored directory.
fn is_in_ignored_dir(path: &Path) -> bool {
    path.components().any(|c| {
        c.as_os_str()
            .to_str()
            .is_some_and(|s| IGNORED_DIRS.contains(&s))
    })
}
