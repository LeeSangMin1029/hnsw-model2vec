//! Tests for collection.rs — CollectionAction parsing and Manifest-based operations.

use clap::Parser;
use tempfile::TempDir;
use v_hnsw_storage::{CollectionInfo, Manifest};

use crate::cli::Cli;
use crate::commands::collection;

// ── CollectionAction subcommand parsing ──

#[test]
fn parse_collection_create() {
    let cli = Cli::try_parse_from([
        "v-hnsw", "collection", "root/", "create", "my_col", "-d", "128",
    ])
    .unwrap();
    match cli.command {
        crate::cli::Commands::Collection { action, .. } => {
            assert!(matches!(
                action,
                collection::CollectionAction::Create { .. }
            ));
        }
        _ => panic!("expected Collection command"),
    }
}

#[test]
fn parse_collection_list() {
    let cli = Cli::try_parse_from(["v-hnsw", "collection", "root/", "list"]).unwrap();
    match cli.command {
        crate::cli::Commands::Collection { action, .. } => {
            assert!(matches!(action, collection::CollectionAction::List));
        }
        _ => panic!("expected Collection command"),
    }
}

#[test]
fn parse_collection_delete() {
    let cli = Cli::try_parse_from(["v-hnsw", "collection", "root/", "delete", "old_col"]).unwrap();
    match cli.command {
        crate::cli::Commands::Collection { action, .. } => {
            assert!(matches!(
                action,
                collection::CollectionAction::Delete { .. }
            ));
        }
        _ => panic!("expected Collection command"),
    }
}

#[test]
fn parse_collection_rename() {
    let cli =
        Cli::try_parse_from(["v-hnsw", "collection", "root/", "rename", "old", "new"]).unwrap();
    match cli.command {
        crate::cli::Commands::Collection { action, .. } => {
            assert!(matches!(
                action,
                collection::CollectionAction::Rename { .. }
            ));
        }
        _ => panic!("expected Collection command"),
    }
}

#[test]
fn parse_collection_info() {
    let cli =
        Cli::try_parse_from(["v-hnsw", "collection", "root/", "info", "my_col"]).unwrap();
    match cli.command {
        crate::cli::Commands::Collection { action, .. } => {
            assert!(matches!(
                action,
                collection::CollectionAction::Info { .. }
            ));
        }
        _ => panic!("expected Collection command"),
    }
}

// ── Manifest-based create / list / delete / rename ──

fn make_collection_info(name: &str) -> CollectionInfo {
    CollectionInfo {
        name: name.to_string(),
        dim: 128,
        metric: "cosine".to_string(),
        created_at: 1_700_000_000,
        count: 0,
    }
}

#[test]
fn manifest_create_and_list() {
    let dir = TempDir::new().unwrap();
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = Manifest::load(&manifest_path).unwrap();
    assert!(manifest.list_collections().is_empty());

    manifest.add_collection(make_collection_info("col_a")).unwrap();
    manifest.save(&manifest_path).unwrap();

    // Reload and verify
    let manifest = Manifest::load(&manifest_path).unwrap();
    let names = manifest.list_collections();
    assert_eq!(names.len(), 1);
    assert!(names.contains(&"col_a"));
}

#[test]
fn manifest_delete_collection() {
    let dir = TempDir::new().unwrap();
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = Manifest::load(&manifest_path).unwrap();
    manifest.add_collection(make_collection_info("to_delete")).unwrap();
    manifest.save(&manifest_path).unwrap();

    // Delete
    let mut manifest = Manifest::load(&manifest_path).unwrap();
    manifest.remove_collection("to_delete").unwrap();
    manifest.save(&manifest_path).unwrap();

    let manifest = Manifest::load(&manifest_path).unwrap();
    assert!(manifest.list_collections().is_empty());
    assert!(manifest.get_collection("to_delete").is_none());
}

#[test]
fn manifest_rename_collection() {
    let dir = TempDir::new().unwrap();
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = Manifest::load(&manifest_path).unwrap();
    manifest.add_collection(make_collection_info("old_name")).unwrap();
    manifest.save(&manifest_path).unwrap();

    // Rename by remove + add
    let mut manifest = Manifest::load(&manifest_path).unwrap();
    let old_info = manifest.get_collection("old_name").unwrap().clone();
    manifest.remove_collection("old_name").unwrap();

    let new_info = CollectionInfo {
        name: "new_name".to_string(),
        ..old_info
    };
    manifest.add_collection(new_info).unwrap();
    manifest.save(&manifest_path).unwrap();

    let manifest = Manifest::load(&manifest_path).unwrap();
    assert!(manifest.get_collection("old_name").is_none());
    assert!(manifest.get_collection("new_name").is_some());
    assert_eq!(manifest.get_collection("new_name").unwrap().dim, 128);
}

#[test]
fn manifest_duplicate_add_fails() {
    let dir = TempDir::new().unwrap();
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = Manifest::load(&manifest_path).unwrap();
    manifest.add_collection(make_collection_info("dup")).unwrap();
    let result = manifest.add_collection(make_collection_info("dup"));
    assert!(result.is_err());
}

#[test]
fn manifest_collection_info_fields() {
    let dir = TempDir::new().unwrap();
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = Manifest::load(&manifest_path).unwrap();
    let info = CollectionInfo {
        name: "test_col".to_string(),
        dim: 256,
        metric: "l2".to_string(),
        created_at: 1_234_567_890,
        count: 42,
    };
    manifest.add_collection(info).unwrap();
    manifest.save(&manifest_path).unwrap();

    let manifest = Manifest::load(&manifest_path).unwrap();
    let loaded = manifest.get_collection("test_col").unwrap();
    assert_eq!(loaded.dim, 256);
    assert_eq!(loaded.metric, "l2");
    assert_eq!(loaded.created_at, 1_234_567_890);
    assert_eq!(loaded.count, 42);
}
