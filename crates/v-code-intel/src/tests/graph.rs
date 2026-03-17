//! End-to-end tests for call graph resolution across Rust grammar patterns.
//!
//! Each test feeds source code through tree-sitter chunking → graph.rs resolution
//! and verifies that expected caller→callee edges exist (or don't exist).

use v_code_chunk::{CodeChunkConfig, RustCodeChunker};

use crate::graph::CallGraph;
use crate::parse::ParsedChunk;

// ── Helpers ─────────────────────────────────────────────────────────

fn chunks_from_files(files: &[(&str, &str)]) -> Vec<ParsedChunk> {
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

fn chunks_from_src(src: &str) -> Vec<ParsedChunk> {
    chunks_from_files(&[("src/lib.rs", src)])
}

fn build_graph(chunks: &[ParsedChunk]) -> CallGraph {
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
fn unknown_receiver_dot_method_not_resolved() {
    // `.method()` on unknown receiver should NOT resolve via short fallback
    // to avoid false positives (e.g., `.get()` → Sq8VectorStore::get).
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
    assert!(!has_edge(&g, "run", "Widget::render"));
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

// ── 31. Inferred local type enables method resolution ────────────────

#[test]
fn inferred_local_type_from_constructor() {
    let chunks = chunks_from_src(r#"
struct Engine { dim: usize }
impl Engine {
    fn new() -> Self { Engine { dim: 0 } }
    fn process(&self) -> usize { self.dim }
}
fn run() {
    let engine = Engine::new();
    engine.process();
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Engine::new"));
    assert!(
        has_edge(&g, "run", "Engine::process"),
        "should resolve engine.process() via inferred type from Engine::new(), callees: {:?}",
        callee_names(&g, "run")
    );
}

#[test]
fn inferred_local_type_from_struct_literal() {
    let chunks = chunks_from_src(r#"
struct Opts { verbose: bool }
impl Opts {
    fn apply(&self) {}
}
fn setup() {
    let opts = Opts { verbose: true };
    opts.apply();
}
    "#);
    let g = build_graph(&chunks);
    assert!(
        has_edge(&g, "setup", "Opts::apply"),
        "should resolve opts.apply() via struct literal type, callees: {:?}",
        callee_names(&g, "setup")
    );
}

#[test]
fn inferred_local_type_with_try_operator() {
    let chunks = chunks_from_src(r#"
struct Connection {}
impl Connection {
    fn open(path: &str) -> Result<Self, Error> { todo!() }
    fn query(&self, sql: &str) {}
}
fn run() -> Result<(), Error> {
    let conn = Connection::open("db.sqlite")?;
    conn.query("SELECT 1");
    Ok(())
}
    "#);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "Connection::open"));
    assert!(
        has_edge(&g, "run", "Connection::query"),
        "should resolve conn.query() via inferred type through ?, callees: {:?}",
        callee_names(&g, "run")
    );
}

#[test]
fn inferred_type_does_not_false_positive_on_lowercase_fn() {
    // `let result = compute();` — compute is not PascalCase, should NOT infer type
    let chunks = chunks_from_src(r#"
struct Widget {}
impl Widget {
    fn render(&self) {}
}
fn compute() -> i32 { 42 }
fn run() {
    let result = compute();
    result.render();
}
    "#);
    let g = build_graph(&chunks);
    // result.render() should NOT resolve to Widget::render because compute() is not Type::method()
    assert!(
        !has_edge(&g, "run", "Widget::render"),
        "should NOT resolve result.render() to Widget::render, callees: {:?}",
        callee_names(&g, "run")
    );
}

// ── 35. Enum variant constructor should not match function ───────────

#[test]
fn enum_variant_does_not_match_function() {
    let chunks = chunks_from_src(r#"
enum MyError {
    Embed(String),
}
fn embed(text: &str) -> Vec<f32> { vec![] }
fn run() {
    let err = MyError::Embed("fail".into());
}
    "#);
    let g = build_graph(&chunks);
    // MyError::Embed is an enum variant, not a function call.
    // Should NOT resolve to the standalone `embed` function.
    assert!(
        !has_edge(&g, "run", "embed"),
        "enum variant MyError::Embed should NOT match function embed, callees: {:?}",
        callee_names(&g, "run")
    );
}

// ══════════════════════════════════════════════════════════════════════
// Coverage audit: test every Rust call pattern for precision AND recall
// ══════════════════════════════════════════════════════════════════════

/// Helper: assert edge exists (recall check).
fn assert_edge(g: &CallGraph, caller: &str, callee: &str) {
    assert!(
        has_edge(g, caller, callee),
        "[RECALL] expected {caller} → {callee}, got: {:?}",
        callee_names(g, caller)
    );
}

/// Helper: assert edge does NOT exist (precision check).
fn assert_no_edge(g: &CallGraph, caller: &str, callee: &str) {
    assert!(
        !has_edge(g, caller, callee),
        "[PRECISION] unexpected {caller} → {callee}",
    );
}

// ── P1. trait object dispatch ────────────────────────────────────────

#[test]
fn trait_object_dispatch() {
    let chunks = chunks_from_src(r#"
trait Processor {
    fn process(&self) -> i32;
}
struct FastProcessor;
impl Processor for FastProcessor {
    fn process(&self) -> i32 { 42 }
}
fn run(p: &dyn Processor) {
    p.process();
}
    "#);
    let g = build_graph(&chunks);
    // dyn Processor → can we resolve p.process() to Processor::process or FastProcessor::process?
    let callees = callee_names(&g, "run");
    let has_any = callees.iter().any(|c| c.to_lowercase().contains("process"));
    assert!(has_any, "[RECALL] trait object p.process() not resolved, got: {callees:?}");
}

// ── P1b. dyn Trait with lifetime ──────────────────────────────────────

#[test]
fn dyn_trait_with_lifetime_dispatch() {
    let chunks = chunks_from_src(r#"
trait HirDatabase {
    fn generic_params(&self) -> Vec<String>;
}
struct RealDb;
impl HirDatabase for RealDb {
    fn generic_params(&self) -> Vec<String> { vec![] }
}
fn analyze<'db>(db: &'db dyn HirDatabase) {
    db.generic_params();
}
    "#);
    let g = build_graph(&chunks);
    let callees = callee_names(&g, "analyze");
    let has_generic_params = callees.iter().any(|c| c.to_lowercase().contains("generic_params"));
    assert!(has_generic_params, "[RECALL] dyn trait with lifetime db.generic_params() not resolved, got: {callees:?}");
}

// ── P2. generic bound method ─────────────────────────────────────────

#[test]
fn generic_bound_method_call() {
    let chunks = chunks_from_src(r#"
trait Encoder {
    fn encode(&self, text: &str) -> Vec<u8>;
}
struct Utf8Encoder;
impl Encoder for Utf8Encoder {
    fn encode(&self, text: &str) -> Vec<u8> { text.as_bytes().to_vec() }
}
fn process<E: Encoder>(enc: &E, text: &str) -> Vec<u8> {
    enc.encode(text)
}
    "#);
    let g = build_graph(&chunks);
    let callees = callee_names(&g, "process");
    let has_encode = callees.iter().any(|c| c.to_lowercase().contains("encode"));
    assert!(has_encode, "[RECALL] generic enc.encode() not resolved, got: {callees:?}");
}

// ── P3. closure internal calls ───────────────────────────────────────

#[test]
fn closure_internal_call() {
    let chunks = chunks_from_src(r#"
fn transform(x: i32) -> i32 { x * 2 }
fn run() {
    let items = vec![1, 2, 3];
    let result: Vec<i32> = items.iter().map(|x| transform(*x)).collect();
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "transform");
}

// ── P4. multi-hop method chain ───────────────────────────────────────

#[test]
fn multi_hop_method_chain() {
    let chunks = chunks_from_src(r#"
struct Builder { val: i32 }
impl Builder {
    fn new() -> Self { Builder { val: 0 } }
    fn with_val(mut self, v: i32) -> Self { self.val = v; self }
    fn build(self) -> Product { Product { val: self.val } }
}
struct Product { val: i32 }
impl Product {
    fn run(&self) -> i32 { self.val }
}
fn main_fn() {
    let p = Builder::new().with_val(42).build();
    p.run();
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "main_fn", "Builder::new");
    // Can we resolve chain intermediate methods?
    let callees = callee_names(&g, "main_fn");
    let has_with_val = callees.iter().any(|c| c.to_lowercase().contains("with_val"));
    let has_build = callees.iter().any(|c| c.to_lowercase().contains("build"));
    let has_run = callees.iter().any(|c| c.to_lowercase().contains("run"));
    // Record what's resolved vs not
    eprintln!("[chain] with_val={has_with_val} build={has_build} run={has_run} callees={callees:?}");
    // At minimum, Builder::new should resolve
    assert_edge(&g, "main_fn", "Builder::new");
}

// ── P5. macro calls ─────────────────────────────────────────────────

#[test]
fn macro_internal_function_call() {
    // Functions called inside macro invocations
    let chunks = chunks_from_src(r#"
fn helper() -> String { "ok".to_owned() }
fn run() {
    println!("{}", helper());
}
    "#);
    let g = build_graph(&chunks);
    // tree-sitter does NOT parse inside macro invocations → known limitation.
    let callees = callee_names(&g, "run");
    let has_helper = callees.iter().any(|c| c.to_lowercase().contains("helper"));
    eprintln!("[macro] helper={has_helper} callees={callees:?}");
    // Known gap: macro arguments are opaque to tree-sitter.
    assert!(!has_helper, "[KNOWN GAP] macro internal calls are not detected");
}

// ── P6. async/await ──────────────────────────────────────────────────

#[test]
fn async_await_call() {
    let chunks = chunks_from_src(r#"
struct Client;
impl Client {
    async fn fetch(&self, url: &str) -> String { url.to_owned() }
}
async fn run(client: &Client) {
    let result = client.fetch("http://example.com").await;
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "Client::fetch");
}

// ── P7. operator overload ────────────────────────────────────────────

#[test]
fn operator_overload_detection() {
    let chunks = chunks_from_src(r#"
use std::ops::Add;
struct Vec2 { x: f32, y: f32 }
impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, rhs: Vec2) -> Vec2 { Vec2 { x: self.x + rhs.x, y: self.y + rhs.y } }
}
fn run() {
    let a = Vec2 { x: 1.0, y: 2.0 };
    let b = Vec2 { x: 3.0, y: 4.0 };
    let c = a + b;
}
    "#);
    let g = build_graph(&chunks);
    let callees = callee_names(&g, "run");
    let has_add = callees.iter().any(|c| c.to_lowercase().contains("add"));
    eprintln!("[operator] add={has_add} callees={callees:?}");
    // Operator overloads are invisible to tree-sitter: `a + b` is not a call node
}

// ── P8. re-export chain ──────────────────────────────────────────────

#[test]
fn reexport_resolution() {
    let chunks = chunks_from_files(&[
        ("src/types.rs", r#"
pub struct Config { pub val: i32 }
impl Config {
    pub fn load() -> Self { Config { val: 0 } }
    pub fn validate(&self) -> bool { self.val > 0 }
}
        "#),
        ("src/lib.rs", r#"
pub use crate::types::Config;
        "#),
        ("src/main.rs", r#"
use crate::types::Config;
fn run() {
    let c = Config::load();
    c.validate();
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "Config::load");
    assert_edge(&g, "run", "Config::validate");
}

// ── P9. impl trait return ────────────────────────────────────────────

#[test]
fn impl_trait_return_type() {
    let chunks = chunks_from_src(r#"
trait Filter {
    fn apply(&self, input: &str) -> String;
}
struct Upper;
impl Filter for Upper {
    fn apply(&self, input: &str) -> String { input.to_uppercase() }
}
fn make_filter() -> impl Filter { Upper }
fn run() {
    let f = make_filter();
    f.apply("hello");
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "make_filter");
    // impl Trait return → can we resolve f.apply()?
    let callees = callee_names(&g, "run");
    let has_apply = callees.iter().any(|c| c.to_lowercase().contains("apply"));
    eprintln!("[impl_trait] apply={has_apply} callees={callees:?}");
}

// ── P10. where clause bound ──────────────────────────────────────────

#[test]
fn where_clause_method_call() {
    let chunks = chunks_from_src(r#"
trait Serializer {
    fn serialize(&self) -> Vec<u8>;
}
struct JsonSerializer;
impl Serializer for JsonSerializer {
    fn serialize(&self) -> Vec<u8> { vec![123] }
}
fn save<S>(s: &S) where S: Serializer {
    let data = s.serialize();
}
    "#);
    let g = build_graph(&chunks);
    let callees = callee_names(&g, "save");
    let has_serialize = callees.iter().any(|c| c.to_lowercase().contains("serialize"));
    eprintln!("[where] serialize={has_serialize} callees={callees:?}");
}

// ── P11. if-let / match pattern ──────────────────────────────────────

#[test]
fn if_let_method_call() {
    let chunks = chunks_from_src(r#"
struct Store;
impl Store {
    fn get(&self, id: u64) -> Option<String> { None }
}
fn process(id: u64) -> String { id.to_string() }
fn run(store: &Store) {
    if let Some(val) = store.get(42) {
        process(42);
    }
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "Store::get");
    assert_edge(&g, "run", "process");
}

// ── P12. nested function ─────────────────────────────────────────────

#[test]
fn nested_function_call() {
    let chunks = chunks_from_src(r#"
fn outer() {
    fn inner() -> i32 { 42 }
    let x = inner();
}
    "#);
    let g = build_graph(&chunks);
    // Inner function is defined inside outer — can we resolve?
    let callees = callee_names(&g, "outer");
    let has_inner = callees.iter().any(|c| c.to_lowercase().contains("inner"));
    eprintln!("[nested] inner={has_inner} callees={callees:?}");
}

// ── P13. turbofish syntax ────────────────────────────────────────────

#[test]
fn turbofish_call() {
    let chunks = chunks_from_src(r#"
fn parse_number(s: &str) -> i32 { 0 }
fn run() {
    let n = "42".parse::<i32>().unwrap();
    let m = parse_number("42");
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "parse_number");
}

// ── P14. associated function (not method) ────────────────────────────

#[test]
fn associated_function_no_self() {
    let chunks = chunks_from_src(r#"
struct Registry;
impl Registry {
    fn instance() -> Self { Registry }
    fn count() -> usize { 0 }
}
fn run() {
    let r = Registry::instance();
    let c = Registry::count();
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "Registry::instance");
    assert_edge(&g, "run", "Registry::count");
}

// ── P15. Default trait method ────────────────────────────────────────

#[test]
fn default_trait_method() {
    let chunks = chunks_from_src(r#"
trait Configurable {
    fn name(&self) -> &str;
    fn display(&self) -> String {
        format!("Config: {}", self.name())
    }
}
struct App;
impl Configurable for App {
    fn name(&self) -> &str { "app" }
}
fn run(app: &App) {
    app.display();
}
    "#);
    let g = build_graph(&chunks);
    let callees = callee_names(&g, "run");
    let has_display = callees.iter().any(|c| c.to_lowercase().contains("display"));
    eprintln!("[default_trait] display={has_display} callees={callees:?}");
}

// ── P16. tuple struct constructor ────────────────────────────────────

#[test]
fn tuple_struct_constructor_vs_function() {
    let chunks = chunks_from_src(r#"
struct Wrapper(i32);
impl Wrapper {
    fn value(&self) -> i32 { self.0 }
}
fn run() {
    let w = Wrapper(42);
    w.value();
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "Wrapper::value");
}

// ── P17. method call on function return ──────────────────────────────

#[test]
fn method_on_function_return() {
    let chunks = chunks_from_src(r#"
struct Config { val: i32 }
impl Config {
    fn load() -> Self { Config { val: 0 } }
    fn validate(&self) -> bool { self.val > 0 }
}
fn get_config() -> Config { Config::load() }
fn run() {
    let c = get_config();
    c.validate();
}
    "#);
    let g = build_graph(&chunks);
    assert_edge(&g, "run", "get_config");
    // get_config() returns Config but it's a lowercase function, not Type::method()
    // → c's type is NOT inferred → c.validate() resolution depends on heuristics
    let callees = callee_names(&g, "run");
    let has_validate = callees.iter().any(|c| c.to_lowercase().contains("validate"));
    eprintln!("[fn_return] validate={has_validate} callees={callees:?}");
}

// ── Field access index ───────────────────────────────────────────────

#[test]
fn field_access_index_self_field() {
    let chunks = chunks_from_src(r#"
struct App {
    db: Database,
    name: String,
}
impl App {
    fn run(&self) -> String {
        let n = self.name;
        self.db.query();
        n
    }
}
struct Database;
impl Database {
    fn query(&self) -> i32 { 0 }
}
    "#);
    let g = build_graph(&chunks);
    // self.db is a field access (receiver in self.db.query())
    let db_accessors = g.find_field_access("app::db");
    assert!(
        !db_accessors.is_empty(),
        "self.db should be in field_access_index as app::db, index: {:?}",
        g.field_access_index
    );
    let accessor_names: Vec<&str> = db_accessors.iter()
        .map(|&i| g.names[i as usize].as_str())
        .collect();
    assert!(
        accessor_names.iter().any(|n| n.contains("run")),
        "App::run should access self.db, got: {accessor_names:?}"
    );
    // self.name is a field access (let n = self.name)
    let name_accessors = g.find_field_access("app::name");
    assert!(
        !name_accessors.is_empty(),
        "self.name should be in field_access_index as app::name, index: {:?}",
        g.field_access_index
    );
}

#[test]
fn field_access_index_param_field() {
    let chunks = chunks_from_src(r#"
struct Config {
    verbose: bool,
    output: String,
}
fn process(cfg: Config) -> bool {
    let v = cfg.verbose;
    let o = cfg.output;
    v
}
    "#);
    let g = build_graph(&chunks);
    let verbose_accessors = g.find_field_access("config::verbose");
    assert!(
        !verbose_accessors.is_empty(),
        "cfg.verbose should be in field_access_index as config::verbose, index: {:?}",
        g.field_access_index
    );
    let output_accessors = g.find_field_access("config::output");
    assert!(
        !output_accessors.is_empty(),
        "cfg.output should be in field_access_index as config::output, index: {:?}",
        g.field_access_index
    );
}

#[test]
fn field_access_index_type_lookup() {
    let chunks = chunks_from_src(r#"
struct Payload {
    source: String,
    tags: Vec<String>,
}
fn check_source(p: Payload) {
    let s = p.source;
}
fn check_tags(p: Payload) {
    let t = p.tags;
}
    "#);
    let g = build_graph(&chunks);
    let entries = g.find_field_accesses_for_type("payload");
    assert!(
        entries.len() >= 2,
        "should find at least source and tags field accesses, got: {entries:?}"
    );
    let field_names: Vec<&str> = entries.iter().map(|(f, _)| *f).collect();
    assert!(field_names.contains(&"source"), "should have source: {field_names:?}");
    assert!(field_names.contains(&"tags"), "should have tags: {field_names:?}");
}

// ══════════════════════════════════════════════════════════════════════
// Pure function tests: extract_leaf_type, extract_generic_bounds,
// owning_type, is_test_path, is_test_chunk
// ══════════════════════════════════════════════════════════════════════

use crate::graph::{extract_leaf_type, extract_generic_bounds, owning_type, is_test_path};

// ── extract_leaf_type ────────────────────────────────────────────────

#[test]
fn leaf_type_simple() {
    assert_eq!(extract_leaf_type("string"), "string");
    assert_eq!(extract_leaf_type("i32"), "i32");
}

#[test]
fn leaf_type_reference() {
    assert_eq!(extract_leaf_type("&foo"), "foo");
    assert_eq!(extract_leaf_type("&mut bar"), "bar");
}

#[test]
fn leaf_type_dyn_impl() {
    assert_eq!(extract_leaf_type("dyn processor"), "processor");
    assert_eq!(extract_leaf_type("impl filter"), "filter");
}

#[test]
fn leaf_type_ref_dyn_trait() {
    // &dyn HirDatabase → hirdatabase
    assert_eq!(extract_leaf_type("&dyn hirdatabase"), "hirdatabase");
    // &mut dyn Iterator → iterator
    assert_eq!(extract_leaf_type("&mut dyn iterator"), "iterator");
}

#[test]
fn leaf_type_lifetime_dyn_trait() {
    // &'db dyn Trait → trait
    assert_eq!(extract_leaf_type("&'db dyn trait"), "trait");
    // &'a mut dyn Processor → processor
    assert_eq!(extract_leaf_type("&'a mut dyn processor"), "processor");
    // &'static dyn Foo → foo
    assert_eq!(extract_leaf_type("&'static dyn foo"), "foo");
}

#[test]
fn leaf_type_generic() {
    assert_eq!(extract_leaf_type("hashmap<string, i32>"), "hashmap");
}

#[test]
fn leaf_type_unwraps_option() {
    assert_eq!(extract_leaf_type("option<config>"), "config");
}

#[test]
fn leaf_type_unwraps_result() {
    assert_eq!(extract_leaf_type("result<foo, error>"), "foo");
}

#[test]
fn leaf_type_unwraps_box() {
    assert_eq!(extract_leaf_type("box<widget>"), "widget");
}

#[test]
fn leaf_type_unwraps_vec() {
    assert_eq!(extract_leaf_type("vec<item>"), "item");
}

#[test]
fn leaf_type_unwraps_arc() {
    assert_eq!(extract_leaf_type("arc<engine>"), "engine");
}

#[test]
fn leaf_type_ref_option() {
    assert_eq!(extract_leaf_type("&option<foo>"), "foo");
}

#[test]
fn leaf_type_self() {
    assert_eq!(extract_leaf_type("Self"), "Self");
}

// ── extract_generic_bounds ───────────────────────────────────────────

#[test]
fn generic_bounds_single() {
    let bounds = extract_generic_bounds("fn foo<T: Search>(x: T)");
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0], ("t".to_owned(), "search".to_owned()));
}

#[test]
fn generic_bounds_multiple() {
    let bounds = extract_generic_bounds("fn foo<T: Search, U: Clone>(x: T, y: U)");
    assert_eq!(bounds.len(), 2);
    assert!(bounds.contains(&("t".to_owned(), "search".to_owned())));
    assert!(bounds.contains(&("u".to_owned(), "clone".to_owned())));
}

#[test]
fn generic_bounds_multi_trait_first_only() {
    let bounds = extract_generic_bounds("fn foo<T: Search + Clone + Display>(x: T)");
    assert_eq!(bounds.len(), 1);
    // Only the first trait bound is extracted
    assert_eq!(bounds[0], ("t".to_owned(), "search".to_owned()));
}

#[test]
fn generic_bounds_where_clause() {
    // where clause is only parsed when <> generics are also present
    let bounds = extract_generic_bounds("fn foo<T>(x: T) where T: Serializer {");
    assert!(
        bounds.iter().any(|(t, b)| t == "t" && b == "serializer"),
        "should extract where clause bound, got: {bounds:?}"
    );
}

#[test]
fn generic_bounds_where_clause_no_angle_brackets() {
    // Without <>, extract_generic_bounds returns empty (known limitation)
    let bounds = extract_generic_bounds("fn foo(x: T) where T: Serializer {");
    assert!(bounds.is_empty(), "no <> means no extraction: {bounds:?}");
}

#[test]
fn generic_bounds_nested_generics() {
    let bounds = extract_generic_bounds("fn foo<T: Iterator<Item = i32>>(x: T)");
    assert_eq!(bounds.len(), 1);
    assert_eq!(bounds[0].0, "t");
    assert_eq!(bounds[0].1, "iterator");
}

#[test]
fn generic_bounds_no_generics() {
    let bounds = extract_generic_bounds("fn foo(x: i32)");
    assert!(bounds.is_empty());
}

// ── owning_type ──────────────────────────────────────────────────────

#[test]
fn owning_type_simple() {
    assert_eq!(owning_type("Foo::bar"), Some("foo".to_owned()));
}

#[test]
fn owning_type_nested() {
    assert_eq!(owning_type("module::Foo::bar"), Some("foo".to_owned()));
}

#[test]
fn owning_type_generic() {
    assert_eq!(owning_type("Foo<T>::bar"), Some("foo".to_owned()));
}

#[test]
fn owning_type_no_qualifier() {
    assert_eq!(owning_type("bar"), None);
}

// ── is_test_path ─────────────────────────────────────────────────────

#[test]
fn is_test_detection() {
    assert!(is_test_path("src/tests/foo.rs"));
    assert!(is_test_path("src/test/bar.rs"));
    assert!(is_test_path("parser_test.rs"));
    assert!(is_test_path("parser_test.go"));
    assert!(is_test_path("src/test_helpers.rs"));
    assert!(!is_test_path("src/lib.rs"));
    assert!(!is_test_path("src/main.rs"));
}

// ── CallGraph.resolve ────────────────────────────────────────────────

#[test]
fn resolve_by_exact_name() {
    let chunks = chunks_from_src(r#"
fn alpha() {}
fn beta() { alpha(); }
    "#);
    let g = build_graph(&chunks);
    let indices = g.resolve("alpha");
    assert!(!indices.is_empty(), "should resolve 'alpha'");
    let found_name = &g.names[indices[0] as usize];
    assert_eq!(found_name.to_lowercase(), "alpha");
}

#[test]
fn resolve_case_insensitive() {
    let chunks = chunks_from_src(r#"
struct MyEngine;
impl MyEngine {
    fn run(&self) {}
}
    "#);
    let g = build_graph(&chunks);
    let indices = g.resolve("myengine::run");
    assert!(!indices.is_empty(), "should resolve case-insensitive");
}

#[test]
fn resolve_nonexistent_returns_empty() {
    let chunks = chunks_from_src("fn foo() {}");
    let g = build_graph(&chunks);
    let indices = g.resolve("nonexistent");
    assert!(indices.is_empty());
}

// ── CallGraph.find_string ────────────────────────────────────────────

#[test]
fn find_string_matches_literal() {
    let chunks = chunks_from_src(r#"
fn setup() {
    open("config.json");
}
fn open(path: &str) {}
    "#);
    let g = build_graph(&chunks);
    let matches = g.find_string("config.json");
    assert!(
        !matches.is_empty(),
        "should find 'config.json' in string index"
    );
}

#[test]
fn find_string_no_match() {
    let chunks = chunks_from_src(r#"
fn setup() {
    open("data.db");
}
fn open(path: &str) {}
    "#);
    let g = build_graph(&chunks);
    let matches = g.find_string("nonexistent");
    assert!(matches.is_empty());
}

// ── CallGraph.call_site_line ─────────────────────────────────────────

#[test]
fn call_site_line_returns_line() {
    let chunks = chunks_from_src(r#"
fn helper() -> i32 { 0 }
fn caller() {
    helper();
}
    "#);
    let g = build_graph(&chunks);
    let caller_idx = g.resolve("caller");
    let helper_idx = g.resolve("helper");
    assert!(!caller_idx.is_empty());
    assert!(!helper_idx.is_empty());
    let line = g.call_site_line(caller_idx[0], helper_idx[0]);
    // line should be > 0 (1-based) since the call exists
    assert!(line > 0, "call_site_line should return the source line");
}

// ── CallGraph callers (reverse edges) ────────────────────────────────

#[test]
fn callers_populated() {
    let chunks = chunks_from_src(r#"
fn target() -> i32 { 42 }
fn caller_a() { target(); }
fn caller_b() { target(); }
    "#);
    let g = build_graph(&chunks);
    let target_idx = g.resolve("target");
    assert!(!target_idx.is_empty());
    let ti = target_idx[0] as usize;
    let caller_names: Vec<&str> = g.callers[ti]
        .iter()
        .map(|&i| g.names[i as usize].as_str())
        .collect();
    assert!(
        caller_names.iter().any(|n| n.to_lowercase().contains("caller_a")),
        "target should have caller_a, got: {caller_names:?}"
    );
    assert!(
        caller_names.iter().any(|n| n.to_lowercase().contains("caller_b")),
        "target should have caller_b, got: {caller_names:?}"
    );
}

// ── is_test flag ─────────────────────────────────────────────────────

#[test]
fn is_test_flag_for_test_file() {
    let chunks = chunks_from_files(&[
        ("src/lib.rs", "fn prod() {}"),
        ("src/tests/foo.rs", "fn test_it() { prod(); }"),
    ]);
    let g = build_graph(&chunks);
    let prod_idx = g.resolve("prod");
    let test_idx = g.resolve("test_it");
    assert!(!prod_idx.is_empty());
    assert!(!test_idx.is_empty());
    assert!(!g.is_test[prod_idx[0] as usize], "prod should not be test");
    assert!(g.is_test[test_idx[0] as usize], "test_it should be test");
}

// ── trait_impls ──────────────────────────────────────────────────────

#[test]
fn trait_impls_populated() {
    let chunks = chunks_from_src(r#"
trait Processor {
    fn process(&self);
}
struct FastProc;
impl Processor for FastProc {
    fn process(&self) {}
}
    "#);
    let g = build_graph(&chunks);
    // Find the trait chunk
    let trait_idx = g.names.iter().position(|n| n.to_lowercase() == "processor");
    if let Some(ti) = trait_idx {
        let impls = &g.trait_impls[ti];
        assert!(
            !impls.is_empty(),
            "Processor trait should have impl entries, trait_impls: {:?}",
            g.trait_impls.iter().enumerate()
                .filter(|(_, v)| !v.is_empty())
                .collect::<Vec<_>>()
        );
    }
}

// ── Graph len/is_empty ───────────────────────────────────────────────

#[test]
fn graph_len_and_is_empty() {
    let empty: Vec<ParsedChunk> = vec![];
    let g_empty = build_graph(&empty);
    assert_eq!(g_empty.len(), 0);
    assert!(g_empty.is_empty());

    let chunks = chunks_from_src("fn foo() {}");
    let g = build_graph(&chunks);
    assert!(g.len() > 0);
    assert!(!g.is_empty());
}

// ── Import resolution across files ───────────────────────────────────

#[test]
fn import_group_resolution() {
    let chunks = chunks_from_files(&[
        ("src/main.rs", r#"
use crate::util::{helper, process};
fn run() {
    helper();
    process();
}
        "#),
        ("src/util.rs", r#"
pub fn helper() -> i32 { 42 }
pub fn process() -> i32 { 0 }
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "run", "helper"), "should resolve grouped import helper");
    assert!(has_edge(&g, "run", "process"), "should resolve grouped import process");
}

// ── Mixed: struct field type + method call + import ──────────────────

#[test]
fn complex_field_type_with_import() {
    let chunks = chunks_from_files(&[
        ("src/app.rs", r#"
use crate::db::Database;
struct App {
    db: Database,
}
impl App {
    fn handle(&self) {
        self.db.query();
    }
}
        "#),
        ("src/db.rs", r#"
pub struct Database;
impl Database {
    pub fn query(&self) -> i32 { 0 }
}
        "#),
    ]);
    let g = build_graph(&chunks);
    assert!(has_edge(&g, "App::handle", "Database::query"));
}

// ── name_index: binary search lookup ─────────────────────────────────

#[test]
fn name_index_sorted() {
    let chunks = chunks_from_src(r#"
fn alpha() {}
fn beta() {}
fn gamma() {}
    "#);
    let g = build_graph(&chunks);
    // name_index should be sorted
    let keys: Vec<&str> = g.name_index.iter().map(|(k, _)| k.as_str()).collect();
    let mut sorted = keys.clone();
    sorted.sort();
    assert_eq!(keys, sorted, "name_index should be sorted for binary search");
}
