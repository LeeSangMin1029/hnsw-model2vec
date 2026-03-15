//! File watcher — monitors source files and invalidates graph cache on changes.
//!
//! Uses `notify` crate for cross-platform filesystem events.
//! No debounce — body_hash early cutoff makes frequent triggers cheap.
//! On change: invalidates graph cache + triggers `v-code add` (body_hash early cutoff).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Guards against concurrent auto-reindex runs.
static REINDEX_RUNNING: AtomicBool = AtomicBool::new(false);

/// Watches project directories for source file changes.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<PathBuf>,
    pending: HashSet<PathBuf>,
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
            db_path,
            input_path,
        })
    }

    /// Poll for changes. Returns changed source files immediately.
    ///
    /// No debounce — `REINDEX_RUNNING` guard prevents concurrent runs
    /// and body_hash early cutoff skips unchanged symbols.
    pub fn poll_changes(&mut self) -> Vec<PathBuf> {
        while let Ok(path) = self.rx.try_recv() {
            self.pending.insert(path);
        }

        if !self.pending.is_empty() {
            return self.pending.drain().collect();
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

    /// Trigger automatic re-indexing in-process via `v_code::commands::add::run`.
    ///
    /// Runs on a background thread. Only one reindex runs at a time —
    /// subsequent requests are skipped until the current one finishes.
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
            eprintln!("[watcher] Starting in-process reindex: {} → {}", input.display(), db.display());
            match v_code::commands::add::run(db, input, &[]) {
                Ok(()) => eprintln!("[watcher] In-process reindex completed"),
                Err(e) => eprintln!("[watcher] In-process reindex failed: {e}"),
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
