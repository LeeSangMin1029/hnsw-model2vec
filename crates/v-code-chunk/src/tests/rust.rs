//! Tests for tree-sitter Rust code chunker.

use crate::{CodeChunkConfig, RustCodeChunker};
use super::fixtures::SAMPLE_RUST;

#[test]
fn chunks_basic_rust_file() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    assert!(!chunks.is_empty(), "should produce at least one chunk");

    let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.contains(&"process_payment"),
        "should extract process_payment function, got: {names:?}"
    );
    assert!(
        names.contains(&"PaymentIntent"),
        "should extract PaymentIntent struct, got: {names:?}"
    );
}

#[test]
fn extracts_imports() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks.iter().find(|c| c.name == "process_payment");
    assert!(func_chunk.is_some(), "should find process_payment");

    let imports = &func_chunk.unwrap().imports;
    assert!(
        imports.iter().any(|i| i.contains("HashMap")),
        "should include HashMap import, got: {imports:?}"
    );
}

#[test]
fn extracts_function_calls() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    assert!(
        func_chunk.calls.iter().any(|c| c.contains("validate_amount")),
        "should detect validate_amount call, got: {:?}",
        func_chunk.calls
    );
}

#[test]
fn extracts_doc_comments() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    assert!(
        func_chunk.doc_comment.as_ref().is_some_and(|d| d.contains("Process a payment")),
        "should extract doc comment, got: {:?}",
        func_chunk.doc_comment
    );
}

#[test]
fn extracts_impl_methods() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();
    assert!(
        names.iter().any(|n| n.contains("PaymentIntent::new")),
        "should extract impl methods with qualified name, got: {names:?}"
    );
}

#[test]
fn embed_text_format() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let embed = func_chunk.to_embed_text("src/payment.rs", &[]);
    assert!(embed.contains("[function]"), "should include kind tag");
    assert!(embed.contains("src/payment.rs"), "should include file path");
    assert!(embed.contains("Process a payment"), "should include doc comment");
}

#[test]
fn custom_fields_complete() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let custom = func_chunk.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
    assert!(custom.contains_key("calls"));
    assert!(custom.contains_key("signature"));
}

#[test]
fn embed_text_includes_called_by() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let called_by = vec!["main".to_owned(), "handle_request".to_owned()];
    let embed = func_chunk.to_embed_text("src/payment.rs", &called_by);
    assert!(
        embed.contains("Called by: main, handle_request"),
        "should include called_by in embed text, got: {embed}"
    );
}

#[test]
fn embed_text_omits_empty_called_by() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let embed = func_chunk.to_embed_text("src/payment.rs", &[]);
    assert!(
        !embed.contains("Called by"),
        "should not include Called by when empty, got: {embed}"
    );
}

#[test]
fn custom_fields_include_called_by() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let called_by = vec!["main".to_owned()];
    let custom = func_chunk.to_custom_fields(&called_by);
    assert!(
        custom.contains_key("called_by"),
        "should include called_by in custom fields"
    );
}

#[test]
fn custom_fields_omit_empty_called_by() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let custom = func_chunk.to_custom_fields(&[]);
    assert!(
        !custom.contains_key("called_by"),
        "should not include called_by when empty"
    );
}

#[test]
fn extracts_type_refs() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    assert!(
        func_chunk.type_refs.contains(&"PaymentIntent".to_owned()),
        "should extract PaymentIntent type ref, got: {:?}",
        func_chunk.type_refs
    );
    assert!(
        func_chunk.type_refs.contains(&"Result".to_owned()),
        "should extract Result type ref, got: {:?}",
        func_chunk.type_refs
    );
    assert!(
        func_chunk.type_refs.contains(&"Error".to_owned()),
        "should extract Error type ref, got: {:?}",
        func_chunk.type_refs
    );
}

#[test]
fn extracts_param_types() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    assert!(
        func_chunk
            .param_types
            .contains(&("amount".to_owned(), "f64".to_owned())),
        "should extract (amount, f64), got: {:?}",
        func_chunk.param_types
    );
    assert!(
        func_chunk
            .param_types
            .contains(&("currency".to_owned(), "&str".to_owned())),
        "should extract (currency, &str), got: {:?}",
        func_chunk.param_types
    );
}

#[test]
fn extracts_return_type() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    assert!(
        func_chunk.return_type.is_some(),
        "should extract return type"
    );
    let ret = func_chunk.return_type.as_ref().unwrap();
    assert!(
        ret.contains("Result"),
        "return type should contain Result, got: {ret}"
    );
}

#[test]
fn impl_method_has_return_type() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("PaymentIntent::new"))
        .expect("should find PaymentIntent::new");

    assert!(
        method.return_type.is_some(),
        "impl method should have return type, got: {:?}",
        method.return_type
    );
    assert_eq!(
        method.return_type.as_deref(),
        Some("Self"),
        "PaymentIntent::new should return Self"
    );
}

#[test]
fn embed_text_includes_type_info() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let embed = func_chunk.to_embed_text("src/payment.rs", &[]);
    assert!(
        embed.contains("Types:"),
        "embed text should include Types section, got: {embed}"
    );
    assert!(
        embed.contains("Params:"),
        "embed text should include Params section, got: {embed}"
    );
}

#[test]
fn custom_fields_include_type_refs() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func_chunk = chunks
        .iter()
        .find(|c| c.name == "process_payment")
        .expect("should find process_payment");

    let custom = func_chunk.to_custom_fields(&[]);
    assert!(
        custom.contains_key("type_refs"),
        "should include type_refs in custom fields"
    );
    assert!(
        custom.contains_key("return_type"),
        "should include return_type in custom fields"
    );
}

#[test]
fn self_method_call_extracted() {
    let config = CodeChunkConfig { min_lines: 1, extract_calls: true, ..CodeChunkConfig::default() };
    let chunker = RustCodeChunker::new(config);
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_SELF_CALLS);

    let do_work = chunks.iter().find(|c| c.name.contains("do_work"))
        .expect("should find do_work method");

    assert!(
        do_work.calls.iter().any(|c| c == "self.helper"),
        "do_work should extract self.helper call, got: {:?}", do_work.calls
    );
}

#[test]
fn multiline_call_normalized() {
    let config = CodeChunkConfig { min_lines: 1, extract_calls: true, ..CodeChunkConfig::default() };
    let chunker = RustCodeChunker::new(config);
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_SELF_CALLS);

    let mc = chunks.iter().find(|c| c.name.contains("multiline_call"))
        .expect("should find multiline_call method");

    // "self\n            .helper()" should be normalized to "self.helper"
    assert!(
        mc.calls.iter().any(|c| c == "self.helper"),
        "multiline self.helper should be normalized, got: {:?}", mc.calls
    );
    // after_call should also be extracted (not lost due to multiline)
    assert!(
        mc.calls.iter().any(|c| c == "after_call"),
        "after_call should not be lost after multiline call, got: {:?}", mc.calls
    );
}

// ── Edge case tests ─────────────────────────────────────────────────

#[test]
fn empty_source_produces_no_chunks() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::EMPTY_SOURCE);
    assert!(chunks.is_empty(), "empty source should produce no chunks");
}

#[test]
fn whitespace_only_source_produces_no_chunks() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::WHITESPACE_ONLY);
    assert!(chunks.is_empty(), "whitespace-only source should produce no chunks");
}

#[test]
fn single_line_fn_below_min_lines_excluded() {
    let config = CodeChunkConfig {
        min_lines: 2,
        ..CodeChunkConfig::default()
    };
    let chunker = RustCodeChunker::new(config);
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_SINGLE_LINE_FN);
    assert!(
        !chunks.iter().any(|c| c.name == "one_liner"),
        "single-line function should be excluded when min_lines=2"
    );
}

#[test]
fn single_line_fn_included_with_min_lines_1() {
    let config = CodeChunkConfig {
        min_lines: 1,
        ..CodeChunkConfig::default()
    };
    let chunker = RustCodeChunker::new(config);
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_SINGLE_LINE_FN);
    assert!(
        chunks.iter().any(|c| c.name == "one_liner"),
        "single-line function should be included when min_lines=1"
    );
}

#[test]
fn nested_module_extracted() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_NESTED);
    assert!(
        chunks.iter().any(|c| c.name == "outer"),
        "should extract module 'outer', got names: {:?}",
        chunks.iter().map(|c| &c.name).collect::<Vec<_>>()
    );
}

#[test]
fn deeply_nested_impl_extracts_methods() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_DEEPLY_NESTED_IMPL);
    let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();

    assert!(
        names.iter().any(|n| n.contains("Outer::method_a")),
        "should extract method_a, got: {names:?}"
    );
    assert!(
        names.iter().any(|n| n.contains("Outer::method_b")),
        "should extract method_b, got: {names:?}"
    );
}

#[test]
fn trait_impl_has_qualified_name() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(super::fixtures::SAMPLE_RUST_DEEPLY_NESTED_IMPL);
    let names: Vec<&str> = chunks.iter().map(|c| c.name.as_str()).collect();

    // impl Display for Outer should have "Display for Outer" or similar
    assert!(
        names.iter().any(|n| n.contains("Display") && n.contains("Outer")),
        "trait impl should have qualified name with trait and type, got: {names:?}"
    );

    // Generic trait impl: impl<T: Clone> MyTrait for Vec<T>
    assert!(
        names.iter().any(|n| n.contains("MyTrait") && n.contains("Vec")),
        "generic trait impl should have qualified name, got: {names:?}"
    );
}

#[test]
fn syntax_error_does_not_panic() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    // Malformed Rust source: tree-sitter should handle gracefully
    let chunks = chunker.chunk("fn broken( { }");
    // Should not panic; may produce 0 or some partial chunks
    let _ = chunks;
}

#[test]
fn config_disable_imports_and_calls() {
    let config = CodeChunkConfig {
        min_lines: 2,
        extract_imports: false,
        extract_calls: false,
    };
    let chunker = RustCodeChunker::new(config);
    let chunks = chunker.chunk(SAMPLE_RUST);

    let func = chunks.iter().find(|c| c.name == "process_payment");
    assert!(func.is_some());
    let func = func.unwrap();
    assert!(func.imports.is_empty(), "imports should be empty when disabled");
    assert!(func.calls.is_empty(), "calls should be empty when disabled");
}

#[test]
fn enum_chunk_extracted() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);
    assert!(
        chunks.iter().any(|c| c.name == "PaymentStatus"),
        "should extract enum PaymentStatus"
    );
}

#[test]
fn struct_visibility_is_pub() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);
    let s = chunks.iter().find(|c| c.name == "PaymentIntent").unwrap();
    assert_eq!(s.visibility, "pub", "PaymentIntent should have pub visibility");
}

#[test]
fn struct_signature_contains_fields() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);
    let s = chunks.iter().find(|c| c.name == "PaymentIntent").unwrap();
    let sig = s.signature.as_deref().unwrap_or("");
    assert!(!sig.is_empty(), "struct signature should contain fields, got empty. kind={:?}", s.kind);
    assert!(sig.contains("amount"), "should contain 'amount' field: {sig}");
}

#[test]
fn private_enum_has_empty_visibility() {
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_RUST);
    let e = chunks.iter().find(|c| c.name == "PaymentStatus").unwrap();
    assert_eq!(e.visibility, "", "PaymentStatus should have empty visibility (private)");
}

#[test]
fn infer_local_types_from_constructor_calls() {
    let code = r#"
struct Engine { dim: usize }
impl Engine {
    fn new() -> Self { Engine { dim: 0 } }
    fn dim(&self) -> usize { self.dim }
}
struct Config { val: u32 }
fn run() {
    let engine = Engine::new();
    let cfg: Config = Config { val: 1 };
    let items = Vec::new();
    let name = String::from("hello");
    engine.dim();
}
"#;
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(code);
    let run_fn = chunks.iter().find(|c| c.name == "run").expect("should find run");

    // `let engine = Engine::new()` → ("engine", "Engine")
    assert!(
        run_fn.local_types.contains(&("engine".to_owned(), "Engine".to_owned())),
        "should infer engine: Engine from Engine::new(), got: {:?}",
        run_fn.local_types
    );

    // `let cfg: Config = ...` → ("cfg", "Config") — explicit annotation still works
    assert!(
        run_fn.local_types.contains(&("cfg".to_owned(), "Config".to_owned())),
        "should extract cfg: Config from annotation, got: {:?}",
        run_fn.local_types
    );

    // `let items = Vec::new()` → ("items", "Vec")
    assert!(
        run_fn.local_types.contains(&("items".to_owned(), "Vec".to_owned())),
        "should infer items: Vec from Vec::new(), got: {:?}",
        run_fn.local_types
    );

    // `let name = String::from("hello")` → ("name", "String")
    assert!(
        run_fn.local_types.contains(&("name".to_owned(), "String".to_owned())),
        "should infer name: String from String::from(), got: {:?}",
        run_fn.local_types
    );
}

#[test]
fn infer_local_types_struct_literal() {
    let code = r#"
struct Opts { verbose: bool }
fn build() {
    let opts = Opts { verbose: true };
}
"#;
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(code);
    let build_fn = chunks.iter().find(|c| c.name == "build").expect("should find build");

    assert!(
        build_fn.local_types.contains(&("opts".to_owned(), "Opts".to_owned())),
        "should infer opts: Opts from struct literal, got: {:?}",
        build_fn.local_types
    );
}

#[test]
fn infer_local_types_try_expression() {
    let code = r#"
struct Connection {}
impl Connection {
    fn open(path: &str) -> Result<Self, Error> { todo!() }
}
fn connect() -> Result<(), Error> {
    let conn = Connection::open("db.sqlite")?;
    Ok(())
}
"#;
    let chunker = RustCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(code);
    let func = chunks.iter().find(|c| c.name == "connect").expect("should find connect");

    assert!(
        func.local_types.contains(&("conn".to_owned(), "Connection".to_owned())),
        "should infer conn: Connection through try expr (?), got: {:?}",
        func.local_types
    );
}
