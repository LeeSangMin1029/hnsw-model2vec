//! File watcher — monitors source files and invalidates graph cache on changes.
//!
//! Uses `notify` crate for cross-platform filesystem events.
//! Debounces rapid changes (2 seconds) and filters to source file extensions only.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
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

/// Watches project directories for source file changes.
pub struct FileWatcher {
    _watcher: RecommendedWatcher,
    rx: mpsc::Receiver<PathBuf>,
    pending: HashSet<PathBuf>,
    last_event: Option<Instant>,
    db_path: PathBuf,
}

impl FileWatcher {
    /// Start watching the given directories recursively.
    ///
    /// Returns `None` if watcher creation fails (non-fatal — daemon runs without watching).
    pub fn new(watch_dirs: &[PathBuf], db_path: PathBuf) -> Option<Self> {
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
