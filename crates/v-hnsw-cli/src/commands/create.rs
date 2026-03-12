//! Create command - Initialize a new v-hnsw database.

use std::path::PathBuf;

use anyhow::{Context, Result};
use v_hnsw_graph::HnswConfig;
use v_hnsw_storage::{StorageConfig, StorageEngine};

use crate::cli::MetricType;

pub use super::db_config::DbConfig;

impl DbConfig {
    /// Build an `HnswConfig` from this database config.
    pub fn to_hnsw_config(&self) -> Result<HnswConfig> {
        HnswConfig::builder()
            .dim(self.dim)
            .m(self.m)
            .ef_construction(self.ef_construction)
            .build()
            .with_context(|| "Failed to create HNSW config")
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
    if path.exists() {
        let config_path = path.join("config.json");
        if config_path.exists() {
            anyhow::bail!(
                "Database already exists at {}. Use a different path or delete the existing database.",
                path.display()
            );
        }
    }

    if dim == 0 {
        anyhow::bail!("Dimension must be greater than 0");
    }
    if m < 2 {
        anyhow::bail!("M parameter must be at least 2");
    }
    if ef < m {
        anyhow::bail!("ef_construction should be at least M ({m})");
    }

    let storage_config = StorageConfig {
        dim,
        initial_capacity: 10_000,
        checkpoint_threshold: 1000,
    };

    let _engine = StorageEngine::create(&path, storage_config)
        .with_context(|| format!("failed to create storage at {}", path.display()))?;

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
        embed_model: None,
        content_type: "mixed".to_owned(),
        input_path: None,
    };
    db_config.save(&path)?;

    println!("Database created: {} (dim={dim}, metric={metric_name})", path.display());

    Ok(())
}
