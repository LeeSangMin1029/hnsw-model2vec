//! Create command - Initialize a new v-hnsw database.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use v_hnsw_storage::{StorageConfig, StorageEngine};

use crate::cli::MetricType;

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

/// Run the create command.
pub fn run(
    path: PathBuf,
    dim: usize,
    metric: MetricType,
    m: usize,
    ef: usize,
    korean: bool,
) -> Result<()> {
    // Check if database already exists
    if path.exists() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            anyhow::bail!(
                "Database already exists at {}. Use a different path or delete the existing database.",
                path.display()
            );
        }
    }

    // Validate parameters
    if dim == 0 {
        anyhow::bail!("Dimension must be greater than 0");
    }
    if m < 2 {
        anyhow::bail!("M parameter must be at least 2");
    }
    if ef < m {
        anyhow::bail!("ef_construction should be at least M ({m})");
    }

    // Create storage
    let storage_config = StorageConfig {
        dim,
        initial_capacity: 10_000,
        checkpoint_threshold: 1000,
    };

    let _engine = StorageEngine::create(&path, storage_config)
        .with_context(|| format!("failed to create storage at {}", path.display()))?;

    // Save database config
    let metric_name = match metric {
        MetricType::Cosine => "cosine",
        MetricType::L2 => "l2",
        MetricType::Dot => "dot",
    };

    let db_config = DbConfig {
        version: DbConfig::CURRENT_VERSION,
        dim,
        metric: metric_name.to_string(),
        m,
        ef_construction: ef,
        korean,
        embed_model: None,  // Set later when inserting with --embed
        content_type: default_content_type(),
    };
    db_config.save(&path)?;

    // Print success message
    println!("Database created successfully:");
    println!("  Path:       {}", path.display());
    println!("  Dimension:  {dim}");
    println!("  Metric:     {metric_name}");
    println!("  M:          {m}");
    println!("  ef:         {ef}");
    println!("  Korean:     {}", if korean { "enabled" } else { "disabled" });

    Ok(())
}
