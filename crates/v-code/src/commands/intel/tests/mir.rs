use std::path::Path;

use v_code_intel::graph::CallGraph;
use v_code_intel::mir::{build_mir_call_map, parse_mir};
use v_code_intel::parse::CodeChunk;

const SAMPLE_MIR: &str = r#"
fn graph::CallGraph::build(_1: &[CodeChunk]) -> Self {
    let mut _0: CallGraph;
    let _2: BTreeMap<String, u32>;

    bb0: {
        _2 = <BTreeMap<String, u32> as Default>::default() -> [return: bb1, unwind continue];
    }

    bb1: {
        _5 = graph::resolve(copy _3, &_2) -> [return: bb2, unwind continue];
    }

    bb2: {
        _8 = graph::is_test_chunk(copy _7) -> [return: bb3, unwind continue];
    }
}

fn graph::resolve(_1: &str, _2: &BTreeMap<String, u32>) -> Option<u32> {
    bb0: {
        _3 = <BTreeMap<String, u32>>::get(copy _2, copy _1) -> [return: bb1, unwind continue];
    }
}

fn graph::<impl at crates\v-code-intel\src\graph.rs:14:1: 14:35>::save(_1: &CallGraph, _2: &Path) -> Result<(), anyhow::Error> {
    bb0: {
        _5 = graph::CallGraph::build(copy _4) -> [return: bb1, unwind continue];
    }
}

fn graph::is_test_chunk(_1: &CodeChunk) -> bool {
    bb0: {
        _0 = const false;
        return;
    }
}

fn helpers::{closure#0}(_1: &mut {closure@src/helpers.rs:10:14: 10:22}, _2: &str) -> bool {
    bb0: {
        return;
    }
}
"#;

#[test]
fn parse_mir_extracts_functions() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    // Closures should be skipped.
    assert!(
        fns.iter().all(|f| !f.name.contains("closure")),
        "closures should be filtered: {fns:?}"
    );
    // Should have at least build, resolve, save, is_test_chunk.
    assert!(fns.len() >= 4, "expected >= 4 functions, got {}", fns.len());
}

#[test]
fn parse_mir_extracts_calls() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    let build_fn = fns.iter().find(|f| f.name.contains("build")).unwrap();
    // build calls resolve and is_test_chunk.
    assert!(
        build_fn.calls.iter().any(|c| c.contains("resolve")),
        "build should call resolve: {:?}",
        build_fn.calls
    );
    assert!(
        build_fn.calls.iter().any(|c| c.contains("is_test_chunk")),
        "build should call is_test_chunk: {:?}",
        build_fn.calls
    );
}

#[test]
fn parse_mir_strips_crate_prefix() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    // No function name should start with "v_code_intel::".
    for f in &fns {
        assert!(
            !f.name.starts_with("v_code_intel::"),
            "crate prefix not stripped: {}",
            f.name
        );
    }
}

#[test]
fn parse_mir_filters_external_calls() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    for f in &fns {
        for call in &f.calls {
            assert!(
                !call.starts_with("std::") && !call.starts_with("core::"),
                "external call not filtered: {call}"
            );
        }
    }
}

#[test]
fn parse_mir_resolves_impl_at() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    // `<impl at ...>::save` should become `graph::save`.
    let save_fn = fns.iter().find(|f| f.name.contains("save"));
    assert!(save_fn.is_some(), "save function not found: {fns:?}");
    assert!(
        !save_fn.unwrap().name.contains("<impl at"),
        "impl at not resolved: {}",
        save_fn.unwrap().name
    );
}

#[test]
fn build_mir_call_map_groups_by_caller() {
    let fns = parse_mir(SAMPLE_MIR, "v_code_intel", Path::new("."));
    let map = build_mir_call_map(&fns);
    // is_test_chunk has no calls → not in map.
    for (name, calls) in &map {
        assert!(!calls.is_empty(), "{name} has empty calls");
    }
}

#[test]
fn parse_mir_handles_trait_as_syntax() {
    let mir = r#"
fn distance::L2Distance::compute(_1: &L2Distance, _2: &[f32]) -> f32 {
    bb0: {
        _3 = <dyn VectorStore as VectorStore>::get(copy _1, copy _2) -> [return: bb1, unwind continue];
        _4 = <Vec<f32> as Index<usize>>::index(copy _3, copy _4) -> [return: bb2, unwind continue];
    }
}
"#;
    let fns = parse_mir(mir, "v_hnsw_graph", Path::new("."));
    let compute = fns.iter().find(|f| f.name.contains("compute")).unwrap();
    // `<dyn VectorStore as VectorStore>::get` → external? No, it's workspace.
    // But the VectorStore::get pattern should be normalized.
    for call in &compute.calls {
        assert!(
            !call.contains(" as "),
            "trait as syntax not resolved: {call}"
        );
    }
}

// ── build_with_resolved_calls integration ──────────────────────────────────────

fn chunk(name: &str, file: &str, calls: &[&str]) -> CodeChunk {
    CodeChunk {
        kind: "function".to_owned(),
        name: name.to_owned(),
        file: file.to_owned(),
        lines: Some((1, 10)),
        signature: Some(format!("fn {name}()")),
        calls: calls.iter().map(|s| s.to_string()).collect(),
        types: vec![],
        imports: vec![],
    }
}

#[test]
fn build_with_resolved_calls_resolves_calls() {
    // tree-sitter chunks: A calls "self.run" which won't resolve without MIR.
    let chunks = vec![
        chunk("mod_a::Foo::process", "src/a.rs", &["self.run"]),
        chunk("mod_b::Bar::run", "src/b.rs", &[]),
        chunk("mod_a::Foo::run", "src/a.rs", &[]),
    ];

    // MIR says: Foo::process calls Foo::run (not Bar::run).
    let mir_text = r#"
fn mod_a::Foo::process(_1: &Foo) -> () {
    bb0: {
        _2 = mod_a::Foo::run(copy _1) -> [return: bb1, unwind continue];
    }
}
"#;
    let fns = parse_mir(mir_text, "test_crate", Path::new("."));
    let mir_map = build_mir_call_map(&fns);

    let graph = CallGraph::build_with_resolved_calls(&chunks, &mir_map);
    // Foo::process → Foo::run (index 2), NOT Bar::run (index 1).
    assert_eq!(graph.callees[0], vec![2], "MIR should resolve to Foo::run");
}

#[test]
fn build_with_resolved_calls_no_fallback_for_rust_files() {
    // Rust files with no MIR data should NOT fall back to tree-sitter.
    let chunks = vec![
        chunk("mod_a::Alpha", "src/a.rs", &["mod_b::Beta"]),
        chunk("mod_b::Beta", "src/b.rs", &[]),
    ];

    let mir_map = build_mir_call_map(&[]);
    let graph = CallGraph::build_with_resolved_calls(&chunks, &mir_map);
    // MIR is authoritative for Rust — no fallback edges.
    assert_eq!(graph.callees[0], vec![] as Vec<u32>, "no fallback for .rs files");
}

#[test]
fn build_with_resolved_calls_falls_back_for_non_rust() {
    // Non-Rust files should still use tree-sitter fallback.
    let chunks = vec![
        chunk("mod_a::Alpha", "src/a.ts", &["mod_b::Beta"]),
        chunk("mod_b::Beta", "src/b.ts", &[]),
    ];

    let mir_map = build_mir_call_map(&[]);
    let graph = CallGraph::build_with_resolved_calls(&chunks, &mir_map);
    assert_eq!(graph.callees[0], vec![1], "fallback should work for .ts files");
}
