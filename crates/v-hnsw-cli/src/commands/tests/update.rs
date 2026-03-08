use crate::commands::update::{is_supported_text_file, UpdateStats};

// ── is_supported_text_file ───────────────────────────────────────

#[test]
fn supported_text_ts() {
    assert!(is_supported_text_file("ts"));
}

#[test]
fn supported_text_tsx() {
    assert!(is_supported_text_file("tsx"));
}

#[test]
fn supported_text_py() {
    assert!(is_supported_text_file("py"));
}

#[test]
fn supported_text_go() {
    assert!(is_supported_text_file("go"));
}

#[test]
fn supported_text_java() {
    assert!(is_supported_text_file("java"));
}

#[test]
fn supported_text_toml() {
    assert!(is_supported_text_file("toml"));
}

#[test]
fn supported_text_yaml() {
    assert!(is_supported_text_file("yaml"));
    assert!(is_supported_text_file("yml"));
}

#[test]
fn supported_text_json() {
    assert!(is_supported_text_file("json"));
}

#[test]
fn supported_text_c_family() {
    assert!(is_supported_text_file("c"));
    assert!(is_supported_text_file("cpp"));
    assert!(is_supported_text_file("h"));
    assert!(is_supported_text_file("hpp"));
}

#[test]
fn supported_text_frontend() {
    assert!(is_supported_text_file("js"));
    assert!(is_supported_text_file("jsx"));
    assert!(is_supported_text_file("svelte"));
    assert!(is_supported_text_file("vue"));
}

#[test]
fn unsupported_text_rs() {
    // .rs is handled by tree-sitter code chunking, not plain text
    assert!(!is_supported_text_file("rs"));
}

#[test]
fn unsupported_text_md() {
    // Markdown has its own chunker
    assert!(!is_supported_text_file("md"));
}

#[test]
fn unsupported_text_binary() {
    assert!(!is_supported_text_file("bin"));
    assert!(!is_supported_text_file("exe"));
    assert!(!is_supported_text_file("png"));
}

#[test]
fn unsupported_text_empty() {
    assert!(!is_supported_text_file(""));
}

// ── UpdateStats serialization ────────────────────────────────────

#[test]
fn update_stats_default() {
    let stats = UpdateStats::default();
    assert_eq!(stats.new, 0);
    assert_eq!(stats.modified, 0);
    assert_eq!(stats.deleted, 0);
    assert_eq!(stats.unchanged, 0);
    assert_eq!(stats.hash_skipped, 0);
}

#[test]
fn update_stats_roundtrip() {
    let stats = UpdateStats {
        new: 3,
        modified: 2,
        deleted: 1,
        unchanged: 10,
        hash_skipped: 5,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let deserialized: UpdateStats = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.new, 3);
    assert_eq!(deserialized.modified, 2);
    assert_eq!(deserialized.deleted, 1);
    assert_eq!(deserialized.unchanged, 10);
    assert_eq!(deserialized.hash_skipped, 5);
}
