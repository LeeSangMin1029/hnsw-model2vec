//! Tests for tree-sitter Rust code chunker.

use super::{CodeChunkConfig, RustCodeChunker};

const SAMPLE_RUST: &str = r#"
use std::collections::HashMap;
use crate::types::Payment;

/// Process a payment through the gateway.
///
/// Validates amount and creates a payment intent.
pub fn process_payment(amount: f64, currency: &str) -> Result<PaymentIntent, Error> {
    validate_amount(amount)?;
    let intent = stripe::create_intent(amount, currency)?;
    db::insert(&intent)?;
    Ok(intent)
}

/// Payment data structure.
#[derive(Debug, Clone)]
pub struct PaymentIntent {
    pub id: String,
    pub amount: f64,
    pub currency: String,
}

impl PaymentIntent {
    /// Create a new payment intent.
    pub fn new(id: String, amount: f64, currency: String) -> Self {
        Self { id, amount, currency }
    }

    /// Check if the payment is valid.
    fn is_valid(&self) -> bool {
        self.amount > 0.0
    }
}

enum PaymentStatus {
    Pending,
    Completed,
    Failed,
}
"#;

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
