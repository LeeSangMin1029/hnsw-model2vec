use crate::commands::intel::graph::CallGraph;
use crate::commands::intel::parse::CodeChunk;

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

// ── CallGraph::build ─────────────────────────────────────────────────

#[test]
fn build_empty() {
    let graph = CallGraph::build(&[]);
    assert_eq!(graph.len(), 0);
    assert!(graph.is_empty());
}

#[test]
fn build_records_metadata() {
    let chunks = vec![chunk("Foo::bar", "src/foo.rs", &[])];
    let graph = CallGraph::build(&chunks);

    assert_eq!(graph.len(), 1);
    assert_eq!(graph.names[0], "Foo::bar");
    assert_eq!(graph.files[0], "src/foo.rs");
    assert_eq!(graph.kinds[0], "function");
    assert_eq!(graph.lines[0], Some((1, 10)));
    assert_eq!(graph.signatures[0].as_deref(), Some("fn Foo::bar()"));
}

#[test]
fn build_creates_callee_and_caller_edges() {
    let chunks = vec![
        chunk("A::run", "src/a.rs", &["B::exec"]),
        chunk("B::exec", "src/b.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    // A calls B
    assert_eq!(graph.callees[0], vec![1]);
    // B is called by A
    assert_eq!(graph.callers[1], vec![0]);
    // B calls nothing
    assert!(graph.callees[1].is_empty());
    // A has no callers
    assert!(graph.callers[0].is_empty());
}

#[test]
fn build_ignores_self_calls() {
    let chunks = vec![chunk("A::run", "src/a.rs", &["A::run"])];
    let graph = CallGraph::build(&chunks);

    assert!(graph.callees[0].is_empty());
    assert!(graph.callers[0].is_empty());
}

#[test]
fn build_deduplicates_edges() {
    let chunks = vec![
        chunk("A", "src/a.rs", &["B", "B", "B"]),
        chunk("B", "src/b.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    assert_eq!(graph.callees[0].len(), 1);
    assert_eq!(graph.callers[1].len(), 1);
}

#[test]
fn build_resolves_short_names() {
    let chunks = vec![
        {
            let mut c = chunk("mod_a::Alpha", "src/a.rs", &["Beta"]);
            c.imports = vec!["use mod_b::Beta;".to_owned()];
            c
        },
        chunk("mod_b::Beta", "src/b.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    // "Beta" should resolve to mod_b::Beta via import
    assert_eq!(graph.callees[0], vec![1]);
}

// ── CallGraph::resolve ───────────────────────────────────────────────

#[test]
fn resolve_exact() {
    let chunks = vec![
        chunk("foo::bar", "src/foo.rs", &[]),
        chunk("baz::qux", "src/baz.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    let results = graph.resolve("foo::bar");
    assert_eq!(results, vec![0]);
}

#[test]
fn resolve_case_insensitive() {
    let chunks = vec![chunk("Foo::Bar", "src/foo.rs", &[])];
    let graph = CallGraph::build(&chunks);

    let results = graph.resolve("foo::bar");
    assert_eq!(results, vec![0]);
}

#[test]
fn resolve_suffix_fallback() {
    let chunks = vec![chunk("very::long::path::run", "src/a.rs", &[])];
    let graph = CallGraph::build(&chunks);

    // Exact match fails, should match via ::run suffix
    let results = graph.resolve("run");
    assert_eq!(results, vec![0]);
}

#[test]
fn resolve_no_match() {
    let chunks = vec![chunk("Foo::bar", "src/foo.rs", &[])];
    let graph = CallGraph::build(&chunks);

    let results = graph.resolve("nonexistent");
    assert!(results.is_empty());
}

// ── is_test detection ────────────────────────────────────────────────

#[test]
fn is_test_detection() {
    let chunks = vec![
        chunk("test_something", "src/lib.rs", &[]),
        chunk("run", "src/tests/foo.rs", &[]),
        chunk("normal", "src/lib.rs", &[]),
        chunk("also_test", "src/test_helpers.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    assert!(graph.is_test[0], "test_ prefix");
    assert!(graph.is_test[1], "/tests/ in path");
    assert!(!graph.is_test[2], "normal function");
    assert!(graph.is_test[3], "/test_ in path");
}

// ── name_index sorted ────────────────────────────────────────────────

#[test]
fn name_index_is_sorted() {
    let chunks = vec![
        chunk("Zebra", "src/z.rs", &[]),
        chunk("Alpha", "src/a.rs", &[]),
        chunk("Middle", "src/m.rs", &[]),
    ];
    let graph = CallGraph::build(&chunks);

    let names: Vec<&str> = graph.name_index.iter().map(|(n, _)| n.as_str()).collect();
    let mut sorted = names.clone();
    sorted.sort();
    assert_eq!(names, sorted);
}
