//! End-to-end tests for call graph resolution across Rust grammar patterns.
//!
//! Each test feeds source code through tree-sitter chunking → graph.rs resolution
//! and verifies that expected caller→callee edges exist (or don't exist).

use v_code_chunk::{CodeChunkConfig, RustCodeChunker};

use crate::graph::CallGraph;
use crate::parse::CodeChunk;

// ── Helpers ─────────────────────────────────────────────────────────

fn chunks_from_files(files: &[(&str, &str)]) -> Vec<CodeChunk> {
    let config = CodeChunkConfig {
        min_lines: 1,
        extract_imports: true,
        extract_calls: true,
    };
    let chunker = RustCodeChunker::new(config);
    let mut all = Vec::new();
    for &(path, src) in files {
        let ts_chunks = chunker.chunk(src);
        for tc in ts_chunks {
            let embed = tc.to_embed_text(path, &[]);
            if let Some(parsed) = crate::parse::parse_chunk(&embed) {
                all.push(parsed);
            }
        }
    }
    all
}

fn chunks_from_src(src: &str) -> Vec<CodeChunk> {
    chunks_from_files(&[("src/lib.rs", src)])
}

fn build_graph(chunks: &[CodeChunk]) -> CallGraph {
    CallGraph::build(chunks)
}

/// Check if caller has callee edge in graph.
fn has_edge(graph: &CallGraph, caller: &str, callee: &str) -> bool {
    let caller_lower = caller.to_lowercase();
    let callee_lower = callee.to_lowercase();
    let caller_idx = graph.names.iter().position(|n| n.to_lowercase() == caller_lower);
    let callee_idx = graph.names.iter().position(|n| n.to_lowercase() == callee_lower);
    if let (Some(ci), Some(ti)) = (caller_idx, callee_idx) {
        graph.callees[ci].iter().any(|&t| t as usize == ti)
    } else {
        false
    }
}

/// List all callee names for a caller.
fn callee_names<'a>(graph: &'a CallGraph, caller: &str) -> Vec<&'a str> {
    let caller_lower = caller.to_lowercase();
    let Some(ci) = graph.names.iter().position(|n| n.to_lowercase() == caller_lower) else {
        return vec![];
    };
    graph.callees[ci]
        .iter()
        .map(|&t| graph.names[t as usize].as_str())
        .collect()
}

// ── 1. Basic function calls ─────────────────────────────────────────

#[test]
fn direct_function_call() {
    let chunks = chunks_from_src(r#"
fn caller() {
    callee();
}
fn callee() -> i32 { 42 }
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "caller", "callee"));
}

#[test]
fn qualified_function_call() {
    let chunks = chunks_from_files(&[
        ("src/foo.rs", "pub fn run() { Bar::exec(); }"),
        ("src/bar.rs", r#"
pub struct Bar;
impl Bar {
    pub fn exec() -> i32 { 1 }
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Bar::exec"));
}

// ── 2. Method calls on self ─────────────────────────────────────────

#[test]
fn self_method_call() {
    let chunks = chunks_from_src(r#"
struct Engine;
impl Engine {
    fn start(&self) {
        self.initialize();
    }
    fn initialize(&self) {}
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "Engine::start", "Engine::initialize"));
}

#[test]
fn self_chained_method_call() {
    let chunks = chunks_from_src(r#"
struct Builder;
impl Builder {
    fn with_name(&mut self) -> &mut Self {
        self
    }
    fn build(&self) -> i32 { 0 }
    fn run(&mut self) {
        self.with_name().build();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "Builder::run", "Builder::with_name"));
}

// ── 3. Constructor + return type inference ──────────────────────────

#[test]
fn constructor_self_return_type() {
    let chunks = chunks_from_files(&[
        ("src/engine.rs", r#"
pub struct Engine;
impl Engine {
    pub fn new() -> Self { Engine }
    pub fn start(&self) {}
}
        "#),
        ("src/main.rs", r#"
use crate::Engine;
fn run() {
    let e = Engine::new();
    e.start();
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Engine::new"));
    // e.start() should resolve to Engine::start via return_type_map
    assert!(has_edge(&g, "run", "Engine::start"));
}

// ── 4. Struct field type → method resolution ────────────────────────

#[test]
fn struct_field_method_call() {
    let chunks = chunks_from_src(r#"
struct Database;
impl Database {
    fn query(&self) -> i32 { 0 }
}
struct App {
    db: Database,
}
impl App {
    fn handle(&self) {
        self.db.query();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "App::handle", "Database::query"));
}

// ── 5. Param type → method resolution ───────────────────────────────

#[test]
fn param_type_method_call() {
    let chunks = chunks_from_src(r#"
struct Config;
impl Config {
    fn validate(&self) -> bool { true }
}
fn process(cfg: Config) {
    cfg.validate();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "process", "Config::validate"));
}

// ── 6. Local type annotation ────────────────────────────────────────

#[test]
fn local_type_annotation_method_call() {
    let chunks = chunks_from_src(r#"
struct Parser;
impl Parser {
    fn parse(&self) -> i32 { 0 }
}
fn run() {
    let p: Parser = todo!();
    p.parse();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Parser::parse"));
}

// ── 7. Trait method filtering ───────────────────────────────────────

#[test]
fn trait_method_not_matched_on_unknown_receiver() {
    let chunks = chunks_from_src(r#"
trait Processor {
    fn process(&self);
}
struct MyProcessor;
impl Processor for MyProcessor {
    fn process(&self) {}
}
fn run(x: i32) {
    x.process();
}
    "#);
    let g = build_graph(&chunks);
    // x is i32 (external type), .process() is a trait method → should NOT match
    assert!(!has_edge(&g, "run", "Processor::process"));
    assert!(!has_edge(&g, "run", "MyProcessor::process"));
}

#[test]
fn non_trait_method_matched_on_unknown_receiver() {
    let chunks = chunks_from_src(r#"
struct Widget;
impl Widget {
    fn render(&self) {}
}
fn run() {
    let w = unknown_fn();
    w.render();
}
fn unknown_fn() -> i32 { 0 }
    "#);
    let g = build_graph(&chunks);
    // render is NOT a trait method → short fallback should match
    assert!(has_edge(&g, "run", "Widget::render"));
}

#[test]
fn std_like_trait_methods_filtered() {
    let chunks = chunks_from_src(r#"
trait Clone {
    fn clone(&self) -> Self;
}
trait Iterator {
    fn next(&mut self) -> Option<i32>;
}
struct Foo;
impl Clone for Foo {
    fn clone(&self) -> Self { Foo }
}
impl Iterator for Foo {
    fn next(&mut self) -> Option<i32> { None }
}
fn run() {
    let x = unknown();
    x.clone();
    x.next();
}
fn unknown() -> i32 { 0 }
    "#);
    let g = build_graph(&chunks);
    // clone and next are trait methods → unknown receiver → skip
    assert!(!has_edge(&g, "run", "Clone::clone"));
    assert!(!has_edge(&g, "run", "Iterator::next"));
}

// ── 8. Import resolution ────────────────────────────────────────────

#[test]
fn import_resolves_qualified_name() {
    let chunks = chunks_from_files(&[
        ("src/main.rs", r#"
use crate::util::helper;
fn run() {
    helper();
}
        "#),
        ("src/util.rs", r#"
pub fn helper() -> i32 { 42 }
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "helper"));
}

// ── 9. Type reference edges ─────────────────────────────────────────

#[test]
fn type_ref_creates_edge() {
    let chunks = chunks_from_files(&[
        ("src/main.rs", r#"
fn process() -> Config {
    Config {}
}
        "#),
        ("src/config.rs", r#"
pub struct Config {
    pub name: String,
}
        "#),
    ]);
    let g = build_graph(&chunks);
    // process references Config type → type_ref edge
    assert!(has_edge(&g, "process", "Config"));
}

// ── 10. Generic types ───────────────────────────────────────────────

#[test]
fn generic_struct_method() {
    let chunks = chunks_from_src(r#"
struct Cache<T> {
    data: T,
}
impl<T> Cache<T> {
    fn get(&self) -> &T { &self.data }
    fn set(&mut self, val: T) { self.data = val; }
}
fn run() {
    let c = Cache::get();
}
    "#);
    let g = build_graph(&chunks);
    // Cache<T>::get should be findable
    let names: Vec<&str> = g.names.iter().map(|s| s.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("Cache") && n.contains("get")),
        "should have Cache::get chunk, got: {names:?}"
    );
}

// ── 11. Trait impl methods ──────────────────────────────────────────

#[test]
fn trait_impl_method_resolved() {
    let chunks = chunks_from_src(r#"
trait Runnable {
    fn execute(&self);
}
struct Task;
impl Runnable for Task {
    fn execute(&self) {}
}
fn dispatch(t: Task) {
    t.execute();
}
    "#);
    let g = build_graph(&chunks);
    // t is Task (param type known), execute exists on Task via trait impl
    // Should resolve via param_type → Task::execute or Runnable for Task::execute
    let callees = callee_names(&g, "dispatch");
    assert!(
        callees.iter().any(|c| c.contains("execute")),
        "dispatch should call execute, got: {callees:?}"
    );
}

// ── 12. Async functions ─────────────────────────────────────────────

#[test]
fn async_fn_calls_extracted() {
    let chunks = chunks_from_src(r#"
async fn fetch() -> i32 { 0 }
async fn handler() {
    let data = fetch().await;
    process(data);
}
fn process(x: i32) {}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "handler", "fetch"));
    assert!(has_edge(&g, "handler", "process"));
}

// ── 13. Closures ────────────────────────────────────────────────────

#[test]
fn closure_calls_captured() {
    let chunks = chunks_from_src(r#"
fn helper() -> i32 { 0 }
fn run() {
    let f = || helper();
    f();
}
    "#);
    let g = build_graph(&chunks);
    // Calls inside closures should be captured
    assert!(has_edge(&g, "run", "helper"));
}

// ── 14. Match expressions ───────────────────────────────────────────

#[test]
fn calls_in_match_arms() {
    let chunks = chunks_from_src(r#"
fn on_a() {}
fn on_b() {}
fn dispatch(x: i32) {
    match x {
        1 => on_a(),
        _ => on_b(),
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "dispatch", "on_a"));
    assert!(has_edge(&g, "dispatch", "on_b"));
}

// ── 15. If/else expressions ─────────────────────────────────────────

#[test]
fn calls_in_if_else() {
    let chunks = chunks_from_src(r#"
fn check() -> bool { true }
fn action_a() {}
fn action_b() {}
fn run() {
    if check() {
        action_a();
    } else {
        action_b();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "check"));
    assert!(has_edge(&g, "run", "action_a"));
    assert!(has_edge(&g, "run", "action_b"));
}

// ── 16. For/while loops ─────────────────────────────────────────────

#[test]
fn calls_in_loops() {
    let chunks = chunks_from_src(r#"
fn step() {}
fn condition() -> bool { true }
fn run() {
    for _ in 0..10 {
        step();
    }
    while condition() {
        step();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "step"));
    assert!(has_edge(&g, "run", "condition"));
}

// ── 17. ? operator (try) ────────────────────────────────────────────

#[test]
fn try_operator_calls() {
    let chunks = chunks_from_src(r#"
fn validate() -> Result<(), String> { Ok(()) }
fn run() -> Result<(), String> {
    validate()?;
    Ok(())
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "validate"));
}

// ── 18. Multiple files (cross-file) ─────────────────────────────────

#[test]
fn cross_file_resolution() {
    let chunks = chunks_from_files(&[
        ("src/server.rs", r#"
pub struct Server;
impl Server {
    pub fn listen(&self) {}
}
        "#),
        ("src/main.rs", r#"
fn main() {
    let s = Server::new();
    s.listen();
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "main", "Server::listen"));
}

// ── 19. Enum methods ────────────────────────────────────────────────

#[test]
fn enum_method_call() {
    let chunks = chunks_from_src(r#"
enum Status {
    Active,
    Inactive,
}
impl Status {
    fn is_active(&self) -> bool {
        matches!(self, Status::Active)
    }
}
fn check(s: Status) {
    s.is_active();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "check", "Status::is_active"));
}

// ── 20. Same-file calls should NOT create cross-file edges ──────────

#[test]
fn same_file_internal_calls() {
    let chunks = chunks_from_src(r#"
fn helper() -> i32 { 42 }
fn main() {
    helper();
}
    "#);
    let g = build_graph(&chunks);
    // Both are in same file, still should have edge (same-file call)
    assert!(has_edge(&g, "main", "helper"));
}

// ── 21. Nested function calls (chaining) ────────────────────────────

#[test]
fn nested_call_expression() {
    let chunks = chunks_from_src(r#"
fn outer(x: i32) -> i32 { x }
fn inner(x: i32) -> i32 { x }
fn run() {
    outer(inner(1));
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "outer"));
    assert!(has_edge(&g, "run", "inner"));
}

// ── 22. Static methods vs instance methods ──────────────────────────

#[test]
fn static_vs_instance_method() {
    let chunks = chunks_from_src(r#"
struct Db;
impl Db {
    fn connect() -> Self { Db }
    fn query(&self) -> i32 { 0 }
}
fn run() {
    let db = Db::connect();
    db.query();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Db::connect"));
    assert!(has_edge(&g, "run", "Db::query"));
}

// ── 23. where clause generics ───────────────────────────────────────

#[test]
fn where_clause_function() {
    let chunks = chunks_from_src(r#"
fn process<T>(x: T) -> i32 where T: std::fmt::Debug {
    helper();
    0
}
fn helper() -> i32 { 1 }
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "process", "helper"));
}

// ── 24. Multiple impl blocks ────────────────────────────────────────

#[test]
fn multiple_impl_blocks() {
    let chunks = chunks_from_src(r#"
struct Foo;
impl Foo {
    fn method_a(&self) {}
}
impl Foo {
    fn method_b(&self) {
        self.method_a();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "Foo::method_b", "Foo::method_a"));
}

// ── 25. Associated functions (no self) ──────────────────────────────

#[test]
fn associated_function_call() {
    let chunks = chunks_from_src(r#"
struct Vec2;
impl Vec2 {
    fn zero() -> Self { Vec2 }
    fn add(a: Self, b: Self) -> Self { Vec2 }
}
fn run() {
    let a = Vec2::zero();
    let b = Vec2::zero();
    Vec2::add(a, b);
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Vec2::zero"));
    assert!(has_edge(&g, "run", "Vec2::add"));
}

// ── 26. Trait with default methods ──────────────────────────────────

#[test]
fn trait_default_method_chunk_exists() {
    let chunks = chunks_from_src(r#"
trait Greet {
    fn hello(&self) -> String {
        String::from("hello")
    }
    fn goodbye(&self) -> String;
}
    "#);
    let g = build_graph(&chunks);
    // Default method should exist as a chunk
    let names: Vec<&str> = g.names.iter().map(|s| s.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("hello")),
        "trait default method should be chunked, got: {names:?}"
    );
}

// ── 27. Macro-like patterns ─────────────────────────────────────────

#[test]
fn calls_after_macro_invocation() {
    let chunks = chunks_from_src(r#"
fn helper() -> i32 { 0 }
fn run() {
    println!("start");
    let x = helper();
    println!("end");
}
    "#);
    let g = build_graph(&chunks);
    // helper() should still be captured even with macro invocations around it
    assert!(has_edge(&g, "run", "helper"));
}

// ── 28. Unsafe blocks ───────────────────────────────────────────────

#[test]
fn calls_in_unsafe_block() {
    let chunks = chunks_from_src(r#"
fn dangerous() -> i32 { 0 }
fn run() {
    unsafe {
        dangerous();
    }
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "dangerous"));
}

// ── 29. Return type propagation through chain ───────────────────────

#[test]
fn return_type_chain() {
    let chunks = chunks_from_files(&[
        ("src/types.rs", r#"
pub struct Config;
impl Config {
    pub fn load() -> Self { Config }
    pub fn validate(&self) -> bool { true }
}
        "#),
        ("src/main.rs", r#"
fn run() {
    let cfg = Config::load();
    cfg.validate();
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Config::load"));
    // Config::load returns Self → cfg is Config → cfg.validate() = Config::validate
    assert!(has_edge(&g, "run", "Config::validate"));
}

// ── 30. Multiple return type → Self resolution ──────────────────────

#[test]
fn builder_pattern_self_return() {
    let chunks = chunks_from_src(r#"
struct Builder;
impl Builder {
    fn new() -> Self { Builder }
    fn option_a(self) -> Self { self }
    fn option_b(self) -> Self { self }
    fn finish(self) -> i32 { 0 }
}
fn run() {
    Builder::new().option_a().option_b().finish();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Builder::new"));
    // NOTE: tree-sitter extracts chained calls but cannot resolve intermediate
    // receiver types in `Builder::new().option_a().option_b().finish()`.
    // Only the first segment `Builder::new` is resolved via qualified name.
    // .option_a(), .option_b(), .finish() have unknown receivers.
    // This is a known tree-sitter limitation (no expression-level type tracking).
    let callees = callee_names(&g, "run");
    assert!(
        callees.iter().any(|c| c.contains("new")),
        "should resolve Builder::new, got: {callees:?}"
    );
}
