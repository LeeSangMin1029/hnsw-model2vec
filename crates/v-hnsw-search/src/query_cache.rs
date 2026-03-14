//! LRU cache for query embeddings, with disk persistence.
//!
//! Stores query text → embedding vector mappings to skip model inference
//! on repeated queries. Persisted to disk so cache survives daemon restarts.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use lru::LruCache;

const QUERY_CACHE_MAX: usize = 1000;
const QUERY_CACHE_FILE: &str = "query_cache.bin";

/// Platform-aware cache directory for v-hnsw.
fn cache_dir() -> PathBuf {
    #[cfg(target_os = "windows")]
    {
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            return PathBuf::from(local).join("v-hnsw").join("cache");
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(cache) = std::env::var("XDG_CACHE_HOME") {
            return PathBuf::from(cache).join("v-hnsw");
        }
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(".cache").join("v-hnsw");
        }
    }
    std::env::temp_dir().join("v-hnsw")
}

/// LRU cache for query embeddings, with disk persistence.
pub struct QueryCache {
    cache: LruCache<String, Vec<f32>>,
    db_path: PathBuf,
}

/// On-disk format: Vec of (query, embedding) pairs in LRU order.
#[derive(serde::Serialize, serde::Deserialize)]
struct CacheEntry {
    query: String,
    embedding: Vec<f32>,
}

impl QueryCache {
    /// Global cache (not tied to a specific DB).
    pub fn global() -> Self {
        Self::load(&cache_dir())
    }

    /// Load cache from disk, or create empty if not found.
    pub fn load(db_path: &Path) -> Self {
        let cap = NonZeroUsize::new(QUERY_CACHE_MAX)
            .unwrap_or(NonZeroUsize::MIN);
        let mut cache = LruCache::new(cap);

        let cache_path = db_path.join(QUERY_CACHE_FILE);
        if let Ok(data) = std::fs::read(&cache_path)
            && let Ok((entries, _)) = bincode::serde::decode_from_slice::<Vec<CacheEntry>, _>(
                &data,
                bincode::config::standard(),
            )
        {
            // Insert in reverse to preserve LRU order (most recent last)
            for entry in entries.into_iter().rev() {
                cache.put(entry.query, entry.embedding);
            }
            tracing::debug!(count = cache.len(), "Query cache loaded");
        }

        Self {
            cache,
            db_path: db_path.to_path_buf(),
        }
    }

    /// Get cached embedding for a query (promotes to most-recently-used).
    pub fn get(&mut self, query: &str) -> Option<&Vec<f32>> {
        self.cache.get(query)
    }

    /// Insert a query→embedding mapping.
    pub fn insert(&mut self, query: String, embedding: Vec<f32>) {
        self.cache.put(query, embedding);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Save cache to disk.
    pub fn save(&self) -> Result<()> {
        let entries: Vec<CacheEntry> = self.cache.iter()
            .map(|(q, e)| CacheEntry {
                query: q.clone(),
                embedding: e.clone(),
            })
            .collect();

        let encoded = bincode::serde::encode_to_vec(&entries, bincode::config::standard())
            .map_err(|e| anyhow::anyhow!("Failed to encode query cache: {e}"))?;

        let cache_path = self.db_path.join(QUERY_CACHE_FILE);
        std::fs::write(&cache_path, encoded)
            .with_context(|| format!("Failed to write query cache to {}", cache_path.display()))?;

        tracing::debug!(count = self.len(), "Query cache saved");
        Ok(())
    }
}
