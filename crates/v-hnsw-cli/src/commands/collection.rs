//! Collection management commands.

use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;
use v_hnsw_storage::{CollectionInfo, Manifest};

use crate::MetricType;

/// Run the collection create subcommand.
pub fn create(
    path: PathBuf,
    name: String,
    dim: Option<usize>,
    metric: MetricType,
) -> Result<()> {
    // Load or create manifest
    let manifest_path = path.join("manifest.json");
    let mut manifest = Manifest::load(&manifest_path)?;

    // Check if collection already exists
    if manifest.get_collection(&name).is_some() {
        anyhow::bail!("Collection '{}' already exists", name);
    }

    // Validate dimension
    let dimension = dim.context("Dimension (-d) is required for creating a collection")?;

    // Create collection directory
    let collection_dir = path.join(&name);
    fs::create_dir_all(&collection_dir)
        .with_context(|| format!("Failed to create collection directory: {:?}", collection_dir))?;

    // Add to manifest
    let created_at = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("System time error: {}", e))?
        .as_secs();

    let info = CollectionInfo {
        name: name.clone(),
        dim: dimension,
        metric: format!("{:?}", metric).to_lowercase(),
        created_at,
        count: 0,
    };

    manifest.add_collection(info)?;
    manifest.save(&manifest_path)?;

    println!("Created collection '{}' with dim={}, metric={:?}", name, dimension, metric);
    Ok(())
}

/// Run the collection list subcommand.
pub fn list(path: PathBuf) -> Result<()> {
    let manifest_path = path.join("manifest.json");
    let manifest = Manifest::load(&manifest_path)?;

    let collections = manifest.list_collections();
    if collections.is_empty() {
        println!("No collections found.");
        return Ok(());
    }

    println!("Collections:");
    for name in collections {
        if let Some(info) = manifest.get_collection(name) {
            println!(
                "  {} (dim={}, metric={}, count={})",
                info.name, info.dim, info.metric, info.count
            );
        }
    }

    Ok(())
}

/// Run the collection delete subcommand.
pub fn delete(path: PathBuf, name: String) -> Result<()> {
    let manifest_path = path.join("manifest.json");
    let mut manifest = Manifest::load(&manifest_path)?;

    // Check if collection exists
    if manifest.get_collection(&name).is_none() {
        anyhow::bail!("Collection '{}' not found", name);
    }

    // Remove from manifest
    manifest.remove_collection(&name)?;
    manifest.save(&manifest_path)?;

    // Remove collection directory
    let collection_dir = path.join(&name);
    if collection_dir.exists() {
        fs::remove_dir_all(&collection_dir)
            .with_context(|| format!("Failed to remove collection directory: {:?}", collection_dir))?;
    }

    println!("Deleted collection '{}'", name);
    Ok(())
}

/// Run the collection rename subcommand.
pub fn rename(path: PathBuf, old: String, new: String) -> Result<()> {
    let manifest_path = path.join("manifest.json");
    let mut manifest = Manifest::load(&manifest_path)?;

    // Check if old collection exists
    let old_info = manifest.get_collection(&old)
        .context(format!("Collection '{}' not found", old))?
        .clone();

    // Check if new name is already taken
    if manifest.get_collection(&new).is_some() {
        anyhow::bail!("Collection '{}' already exists", new);
    }

    // Create new collection info with updated name
    let new_info = CollectionInfo {
        name: new.clone(),
        dim: old_info.dim,
        metric: old_info.metric,
        created_at: old_info.created_at,
        count: old_info.count,
    };

    // Update manifest
    manifest.remove_collection(&old)?;
    manifest.add_collection(new_info)?;
    manifest.save(&manifest_path)?;

    // Rename directory
    let old_dir = path.join(&old);
    let new_dir = path.join(&new);
    if old_dir.exists() {
        fs::rename(&old_dir, &new_dir)
            .with_context(|| format!("Failed to rename directory from {:?} to {:?}", old_dir, new_dir))?;
    }

    println!("Renamed collection '{}' to '{}'", old, new);
    Ok(())
}

/// Run the collection info subcommand.
pub fn info(path: PathBuf, name: String) -> Result<()> {
    let manifest_path = path.join("manifest.json");
    let manifest = Manifest::load(&manifest_path)?;

    let info = manifest.get_collection(&name)
        .context(format!("Collection '{}' not found", name))?;

    println!("Collection: {}", info.name);
    println!("  Dimension: {}", info.dim);
    println!("  Metric: {}", info.metric);
    println!("  Created: {}", info.created_at);
    println!("  Point Count: {}", info.count);

    Ok(())
}

/// Dispatch collection subcommands.
pub fn run(path: PathBuf, action: CollectionAction) -> Result<()> {
    match action {
        CollectionAction::Create { name, dim, metric } => create(path, name, dim, metric),
        CollectionAction::List => list(path),
        CollectionAction::Delete { name } => delete(path, name),
        CollectionAction::Rename { old, new } => rename(path, old, new),
        CollectionAction::Info { name } => info(path, name),
    }
}

/// Collection subcommand actions.
#[derive(Debug, clap::Subcommand)]
pub enum CollectionAction {
    /// Create a new collection.
    Create {
        /// Name of the collection.
        name: String,
        /// Vector dimension.
        #[arg(short = 'd', long)]
        dim: Option<usize>,
        /// Distance metric.
        #[arg(long, default_value = "cosine")]
        metric: MetricType,
    },
    /// List all collections.
    List,
    /// Delete a collection.
    Delete {
        /// Name of the collection to delete.
        name: String,
    },
    /// Rename a collection.
    Rename {
        /// Current name.
        old: String,
        /// New name.
        new: String,
    },
    /// Show collection information.
    Info {
        /// Name of the collection.
        name: String,
    },
}
