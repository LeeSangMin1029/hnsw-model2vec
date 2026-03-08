use crate::commands::code_intel;
use crate::commands::code_intel::{
    extract_crate_name, relative_path, lines_str, grouped_json,
    stats_to_json, build_stats, format_lines_str_opt,
};

// ── extract_crate_name ───────────────────────────────────────────

#[test]
fn extract_crate_name_normal() {
    assert_eq!(extract_crate_name("crates/v-hnsw-cli/src/main.rs"), "v-hnsw-cli");
}

#[test]
fn extract_crate_name_nested() {
    assert_eq!(
        extract_crate_name("crates/v-hnsw-storage/src/engine.rs"),
        "v-hnsw-storage"
    );
}

#[test]
fn extract_crate_name_no_crates_prefix() {
    assert_eq!(extract_crate_name("src/lib.rs"), "(root)");
}

#[test]
fn extract_crate_name_just_crates() {
    // "crates/" followed by something without a slash
    assert_eq!(extract_crate_name("crates/foo"), "(root)");
}

// ── relative_path ────────────────────────────────────────────────

#[test]
fn relative_path_with_crates() {
    assert_eq!(relative_path("/home/user/project/crates/foo/src/lib.rs"), "crates/foo/src/lib.rs");
}

#[test]
fn relative_path_with_src() {
    assert_eq!(relative_path("/home/user/project/src/main.rs"), "src/main.rs");
}

#[test]
fn relative_path_neither() {
    assert_eq!(relative_path("lib.rs"), "lib.rs");
}

#[test]
fn relative_path_crates_preferred_over_src() {
    // "crates/" appears before "src/", so crates/ wins
    let result = relative_path("/root/crates/foo/src/bar.rs");
    assert_eq!(result, "crates/foo/src/bar.rs");
}

// ── format_lines_str_opt ─────────────────────────────────────────

#[test]
fn format_lines_str_opt_some() {
    assert_eq!(format_lines_str_opt(Some((10, 20))), "10-20");
}

#[test]
fn format_lines_str_opt_none() {
    assert_eq!(format_lines_str_opt(None), "");
}

#[test]
fn format_lines_str_opt_same_line() {
    assert_eq!(format_lines_str_opt(Some((5, 5))), "5-5");
}

// ── lines_str (CodeChunk version) ────────────────────────────────

#[test]
fn lines_str_with_range() {
    let chunk = code_intel::parse::CodeChunk {
        name: "test".to_string(),
        kind: "function".to_string(),
        file: "test.rs".to_string(),
        lines: Some((1, 10)),
        signature: None,
        calls: vec![],
        types: vec![],
    };
    assert_eq!(lines_str(&chunk), "1-10");
}

#[test]
fn lines_str_without_range() {
    let chunk = code_intel::parse::CodeChunk {
        name: "test".to_string(),
        kind: "struct".to_string(),
        file: "test.rs".to_string(),
        lines: None,
        signature: None,
        calls: vec![],
        types: vec![],
    };
    assert_eq!(lines_str(&chunk), "");
}

// ── stats_to_json ────────────────────────────────────────────────

#[test]
fn stats_to_json_empty() {
    let stats = build_stats(&[]);
    let json = stats_to_json(&stats);
    let obj = json.as_object().unwrap();
    // Should always have the _s legend key
    assert!(obj.contains_key("_s"));
    assert_eq!(obj.len(), 1);
}

#[test]
fn stats_to_json_counts_functions() {
    let chunks = vec![
        code_intel::parse::CodeChunk {
            name: "my_func".to_string(),
            kind: "function".to_string(),
            file: "crates/foo/src/lib.rs".to_string(),
            lines: Some((1, 10)),
            signature: None,
            calls: vec![],
            types: vec![],
        },
        code_intel::parse::CodeChunk {
            name: "test_something".to_string(),
            kind: "function".to_string(),
            file: "crates/foo/src/tests/test.rs".to_string(),
            lines: Some((1, 5)),
            signature: None,
            calls: vec![],
            types: vec![],
        },
    ];
    let stats = build_stats(&chunks);
    let json = stats_to_json(&stats);
    let foo = json.get("foo").unwrap();
    assert_eq!(foo["p"], 1, "should have 1 prod function");
    assert_eq!(foo["t"], 1, "should have 1 test function");
}

#[test]
fn stats_to_json_structs_and_enums() {
    let chunks = vec![
        code_intel::parse::CodeChunk {
            name: "MyStruct".to_string(),
            kind: "struct".to_string(),
            file: "crates/bar/src/types.rs".to_string(),
            lines: None,
            signature: None,
            calls: vec![],
            types: vec![],
        },
        code_intel::parse::CodeChunk {
            name: "MyEnum".to_string(),
            kind: "enum".to_string(),
            file: "crates/bar/src/types.rs".to_string(),
            lines: None,
            signature: None,
            calls: vec![],
            types: vec![],
        },
    ];
    let stats = build_stats(&chunks);
    let json = stats_to_json(&stats);
    let bar = json.get("bar").unwrap();
    assert_eq!(bar["s"], 1, "should have 1 struct");
    assert_eq!(bar["e"], 1, "should have 1 enum");
}

// ── grouped_json ─────────────────────────────────────────────────

#[test]
fn grouped_json_groups_by_file() {
    let c1 = code_intel::parse::CodeChunk {
        name: "func_a".to_string(),
        kind: "function".to_string(),
        file: "crates/x/src/lib.rs".to_string(),
        lines: Some((1, 5)),
        signature: None,
        calls: vec![],
        types: vec![],
    };
    let c2 = code_intel::parse::CodeChunk {
        name: "func_b".to_string(),
        kind: "function".to_string(),
        file: "crates/x/src/lib.rs".to_string(),
        lines: Some((10, 20)),
        signature: None,
        calls: vec![],
        types: vec![],
    };
    let chunks: Vec<&code_intel::parse::CodeChunk> = vec![&c1, &c2];
    let json = grouped_json(&chunks);
    let obj = json.as_object().unwrap();
    // Both chunks in same file, so one file key + _s
    assert_eq!(obj.len(), 2);
    let entries = obj.get("crates/x/src/lib.rs").unwrap().as_array().unwrap();
    assert_eq!(entries.len(), 2);
}
