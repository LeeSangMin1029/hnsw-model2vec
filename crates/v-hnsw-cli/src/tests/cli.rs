//! Tests for cli.rs — clap argument parsing.

use clap::Parser;

use crate::cli::{Cli, Commands, MetricType};

// ── MetricType ValueEnum parsing ──

#[test]
fn metric_type_parse_cosine() {
    let cli = Cli::try_parse_from(["v-hnsw", "create", "test.db", "-d", "128", "--metric", "cosine"]).unwrap();
    match cli.command {
        Commands::Create { metric, .. } => assert!(matches!(metric, MetricType::Cosine)),
        _ => panic!("expected Create command"),
    }
}

#[test]
fn metric_type_parse_l2() {
    let cli = Cli::try_parse_from(["v-hnsw", "create", "test.db", "-d", "128", "--metric", "l2"]).unwrap();
    match cli.command {
        Commands::Create { metric, .. } => assert!(matches!(metric, MetricType::L2)),
        _ => panic!("expected Create command"),
    }
}

#[test]
fn metric_type_parse_dot() {
    let cli = Cli::try_parse_from(["v-hnsw", "create", "test.db", "-d", "128", "--metric", "dot"]).unwrap();
    match cli.command {
        Commands::Create { metric, .. } => assert!(matches!(metric, MetricType::Dot)),
        _ => panic!("expected Create command"),
    }
}

#[test]
fn metric_type_default_is_cosine() {
    let cli = Cli::try_parse_from(["v-hnsw", "create", "test.db", "-d", "128"]).unwrap();
    match cli.command {
        Commands::Create { metric, .. } => assert!(matches!(metric, MetricType::Cosine)),
        _ => panic!("expected Create command"),
    }
}

// ── Cli::try_parse_from — valid commands ──

#[test]
fn parse_create_command() {
    let cli = Cli::try_parse_from([
        "v-hnsw", "create", "test.db", "-d", "256", "-m", "32", "-e", "400", "--korean",
    ])
    .unwrap();
    match cli.command {
        Commands::Create { path, dim, m, ef, korean, .. } => {
            assert_eq!(path.to_str().unwrap(), "test.db");
            assert_eq!(dim, 256);
            assert_eq!(m, 32);
            assert_eq!(ef, 400);
            assert!(korean);
        }
        _ => panic!("expected Create command"),
    }
}

#[test]
fn parse_info_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "info", "my.db"]).unwrap();
    assert!(matches!(cli.command, Commands::Info { .. }));
}

#[test]
fn parse_find_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "find", "my.db", "hello world", "-k", "5"]).unwrap();
    match cli.command {
        Commands::Find { db, query, k, .. } => {
            assert_eq!(db.to_str().unwrap(), "my.db");
            assert_eq!(query.as_deref(), Some("hello world"));
            assert_eq!(k, 5);
        }
        _ => panic!("expected Find command"),
    }
}

#[test]
fn parse_get_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "get", "my.db", "1", "2", "3"]).unwrap();
    match cli.command {
        Commands::Get { ids, .. } => assert_eq!(ids, vec![1, 2, 3]),
        _ => panic!("expected Get command"),
    }
}

#[test]
fn parse_export_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "export", "my.db", "-o", "out.jsonl"]).unwrap();
    match cli.command {
        Commands::Export { output, .. } => assert_eq!(output.to_str().unwrap(), "out.jsonl"),
        _ => panic!("expected Export command"),
    }
}

#[test]
fn parse_delete_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "delete", "my.db", "--id", "42"]).unwrap();
    match cli.command {
        Commands::Delete { id, .. } => assert_eq!(id, 42),
        _ => panic!("expected Delete command"),
    }
}

#[test]
fn parse_add_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "add", "my.db", "./docs"]).unwrap();
    assert!(matches!(cli.command, Commands::Add { .. }));
}

#[test]
fn parse_build_index_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "build-index", "my.db"]).unwrap();
    assert!(matches!(cli.command, Commands::BuildIndex { .. }));
}

#[test]
fn parse_serve_command() {
    let cli = Cli::try_parse_from(["v-hnsw", "serve", "--port", "8080"]).unwrap();
    match cli.command {
        Commands::Serve { port, .. } => assert_eq!(port, 8080),
        _ => panic!("expected Serve command"),
    }
}

// ── Error cases ──

#[test]
fn parse_no_args_fails() {
    assert!(Cli::try_parse_from(["v-hnsw"]).is_err());
}

#[test]
fn parse_unknown_command_fails() {
    assert!(Cli::try_parse_from(["v-hnsw", "nonexistent"]).is_err());
}

#[test]
fn parse_create_missing_dim_fails() {
    assert!(Cli::try_parse_from(["v-hnsw", "create", "test.db"]).is_err());
}

#[test]
fn parse_invalid_metric_fails() {
    assert!(Cli::try_parse_from([
        "v-hnsw", "create", "test.db", "-d", "128", "--metric", "hamming",
    ])
    .is_err());
}

#[test]
fn parse_get_no_ids_fails() {
    assert!(Cli::try_parse_from(["v-hnsw", "get", "my.db"]).is_err());
}

// ── Commands enum variants ──

#[test]
fn commands_collection_variant() {
    let cli = Cli::try_parse_from(["v-hnsw", "collection", "root/", "list"]).unwrap();
    assert!(matches!(cli.command, Commands::Collection { .. }));
}

#[test]
fn commands_bench_variant() {
    let cli = Cli::try_parse_from(["v-hnsw", "bench", "my.db"]).unwrap();
    match cli.command {
        Commands::Bench { queries, k, .. } => {
            assert_eq!(queries, 100); // default
            assert_eq!(k, 10); // default
        }
        _ => panic!("expected Bench command"),
    }
}

#[test]
fn commands_update_variant() {
    let cli = Cli::try_parse_from(["v-hnsw", "update", "my.db", "./src"]).unwrap();
    assert!(matches!(cli.command, Commands::Update { .. }));
}

