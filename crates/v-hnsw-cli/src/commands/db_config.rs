//! Database configuration (shared between doc and code).

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Database metadata stored in config.json.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbConfig {
    /// Database format version.
    pub version: u32,
    /// Vector dimension.
    pub dim: usize,
    /// Distance metric name.
    pub metric: String,
    /// HNSW M parameter.
    pub m: usize,
    /// HNSW ef_construction parameter.
    pub ef_construction: usize,
    /// Whether Korean tokenizer is enabled.
    pub korean: bool,
    /// Embedding model used (for vsearch auto-detection).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_model: Option<String>,
    /// Content type: "code", "markdown", or "mixed".
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// Original input path used during `add` (for `update` default).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_path: Option<String>,
}

fn default_content_type() -> String {
    "mixed".to_owned()
}

impl DbConfig {
    pub const CURRENT_VERSION: u32 = 1;

    /// Load config from database path.
    pub fn load(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let data = std::fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config: {}", config_path.display()))?;
        let config: DbConfig = serde_json::from_str(&data)
            .with_context(|| "failed to parse config.json")?;
        Ok(config)
    }

    /// Save config to database path.
    pub fn save(&self, path: &Path) -> Result<()> {
        let config_path = path.join("config.json");
        let data = serde_json::to_string_pretty(self)
            .with_context(|| "failed to serialize config")?;
        std::fs::write(&config_path, data)
            .with_context(|| format!("failed to write config: {}", config_path.display()))?;
        Ok(())
    }
}
