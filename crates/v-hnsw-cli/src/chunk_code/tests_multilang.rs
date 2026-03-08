//! Tests for multi-language tree-sitter code chunkers.
//!
//! Validates that TypeScript, Python, Go, Java, and C/C++ parsers
//! extract the same CodeChunk fields as the existing Rust parser.

use super::{CodeChunkConfig, CodeNodeKind, is_supported_code_file};

// ---------------------------------------------------------------------------
// Helper: assert a chunk with the given name exists and return it
// ---------------------------------------------------------------------------
macro_rules! find_chunk {
    ($chunks:expr, $name:expr) => {
        $chunks
            .iter()
            .find(|c| c.name == $name)
            .unwrap_or_else(|| {
                let names: Vec<&str> = $chunks.iter().map(|c| c.name.as_str()).collect();
                panic!(
                    "expected chunk named {:?}, found: {:?}",
                    $name, names
                );
            })
    };
}

macro_rules! has_chunk {
    ($chunks:expr, $name:expr) => {
        $chunks.iter().any(|c| c.name == $name)
    };
}

// ===========================================================================
// is_supported_code_file — extension registry
// ===========================================================================

#[test]
fn supported_extensions_include_all_languages() {
    let extensions = ["rs", "ts", "tsx", "py", "go", "java", "c", "cpp", "h", "hpp"];
    for ext in extensions {
        assert!(
            is_supported_code_file(ext),
            "extension {ext:?} should be supported"
        );
    }
}

#[test]
fn unsupported_extensions_rejected() {
    let extensions = ["txt", "md", "json", "toml", "yaml", "html", "css"];
    for ext in extensions {
        assert!(
            !is_supported_code_file(ext),
            "extension {ext:?} should NOT be supported"
        );
    }
}

// ===========================================================================
// TypeScript
// ===========================================================================

const SAMPLE_TS: &str = r#"
import { Request, Response } from 'express';
import { UserService } from './services';

/** Process an incoming HTTP request. */
export function handleRequest(req: Request, res: Response): Promise<void> {
    const user = UserService.findById(req.params.id);
    validate(user);
    res.json(user);
}

/** A user data transfer object. */
export interface UserDTO {
    id: string;
    name: string;
    email: string;
}

export class UserController {
    private service: UserService;

    constructor(service: UserService) {
        this.service = service;
    }

    /** Get a user by ID. */
    async getUser(id: string): Promise<UserDTO> {
        return this.service.findById(id);
    }

    /** Delete a user. */
    async deleteUser(id: string): Promise<void> {
        await this.service.delete(id);
    }
}

export enum Role {
    Admin = 'admin',
    User = 'user',
    Guest = 'guest',
}
"#;

#[test]
fn ts_extracts_function() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert_eq!(func.kind, CodeNodeKind::Function);
    assert_eq!(func.visibility, "export");
}

#[test]
fn ts_extracts_interface() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "UserDTO"), "should extract interface UserDTO");
}

#[test]
fn ts_extracts_class_and_methods() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "UserController"), "should extract class");
    assert!(
        chunks.iter().any(|c| c.name.contains("UserController") && c.name.contains("getUser")),
        "should extract class methods with qualified name"
    );
}

#[test]
fn ts_extracts_enum() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    assert!(has_chunk!(chunks, "Role"), "should extract enum Role");
}

#[test]
fn ts_extracts_calls() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.calls.iter().any(|c| c.contains("validate")),
        "should detect validate call, got: {:?}",
        func.calls
    );
}

#[test]
fn ts_extracts_doc_comments() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("incoming HTTP request")),
        "should extract JSDoc comment, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn ts_embed_text_and_custom_fields() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");

    let embed = func.to_embed_text("src/controller.ts", &[]);
    assert!(embed.contains("[function]"), "embed text should include kind");
    assert!(embed.contains("src/controller.ts"), "embed text should include file path");

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ===========================================================================
// Python
// ===========================================================================

const SAMPLE_PY: &str = r#"
import os
from typing import Optional, List
from dataclasses import dataclass

def process_data(items: List[str], limit: int = 10) -> Optional[dict]:
    """Process a list of data items.

    Args:
        items: The input items to process.
        limit: Maximum number of items.

    Returns:
        A dictionary of processed results.
    """
    validated = validate_items(items)
    return aggregate(validated, limit)

@dataclass
class DataResult:
    """Result of data processing."""
    count: int
    items: List[str]
    status: str = "pending"

    def is_complete(self) -> bool:
        """Check if processing is complete."""
        return self.status == "complete"

    def summary(self) -> str:
        """Get a summary string."""
        return f"{self.count} items: {self.status}"

class DataProcessor:
    """Processes data with configuration."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, data: List[str]) -> DataResult:
        """Execute the processing pipeline."""
        result = process_data(data, self.config.get("limit", 10))
        return DataResult(count=len(data), items=data)

def _private_helper(x: int) -> int:
    return x * 2
"#;

#[test]
fn py_extracts_function() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn py_extracts_class() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(has_chunk!(chunks, "DataResult"), "should extract DataResult class");
    assert!(has_chunk!(chunks, "DataProcessor"), "should extract DataProcessor class");
}

#[test]
fn py_extracts_methods() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(
        chunks.iter().any(|c| c.name.contains("DataResult") && c.name.contains("is_complete")),
        "should extract methods with qualified name"
    );
}

#[test]
fn py_extracts_calls() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.calls.iter().any(|c| c.contains("validate_items")),
        "should detect validate_items call, got: {:?}",
        func.calls
    );
}

#[test]
fn py_extracts_docstrings() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("Process a list")),
        "should extract docstring, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn py_embed_text_and_custom_fields() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");

    let embed = func.to_embed_text("src/processor.py", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("src/processor.py"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ===========================================================================
// Go
// ===========================================================================

const SAMPLE_GO: &str = r#"
package main

import (
    "fmt"
    "net/http"
)

// HandleRequest processes an incoming HTTP request.
func HandleRequest(w http.ResponseWriter, r *http.Request) error {
    data := ParseBody(r)
    if err := Validate(data); err != nil {
        return err
    }
    fmt.Fprintf(w, "OK")
    return nil
}

// RequestData holds parsed request information.
type RequestData struct {
    Method  string
    Path    string
    Headers map[string]string
}

// String returns a string representation.
func (rd *RequestData) String() string {
    return fmt.Sprintf("%s %s", rd.Method, rd.Path)
}

// Validate checks if the data is valid.
func (rd *RequestData) Validate() error {
    if rd.Method == "" {
        return fmt.Errorf("empty method")
    }
    return nil
}

// Handler is an interface for request handlers.
type Handler interface {
    ServeHTTP(w http.ResponseWriter, r *http.Request)
    Name() string
}

// Status represents processing status.
type Status int

const (
    StatusPending Status = iota
    StatusDone
    StatusFailed
)
"#;

#[test]
fn go_extracts_function() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn go_extracts_struct() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(has_chunk!(chunks, "RequestData"), "should extract RequestData struct");
}

#[test]
fn go_extracts_methods() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(
        chunks.iter().any(|c| c.name.contains("RequestData") && c.name.contains("String")),
        "should extract method with receiver-qualified name"
    );
}

#[test]
fn go_extracts_interface() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    assert!(has_chunk!(chunks, "Handler"), "should extract interface Handler");
}

#[test]
fn go_extracts_calls() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.calls.iter().any(|c| c.contains("ParseBody") || c.contains("Validate")),
        "should detect function calls, got: {:?}",
        func.calls
    );
}

#[test]
fn go_extracts_doc_comments() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("processes an incoming")),
        "should extract Go doc comment, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn go_embed_text_and_custom_fields() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");

    let embed = func.to_embed_text("pkg/handler.go", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("pkg/handler.go"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ===========================================================================
// Java
// ===========================================================================

const SAMPLE_JAVA: &str = r#"
package com.example.service;

import java.util.List;
import java.util.Optional;

/**
 * Processes payment transactions.
 */
public class PaymentService {

    private final PaymentGateway gateway;

    public PaymentService(PaymentGateway gateway) {
        this.gateway = gateway;
    }

    /**
     * Process a payment with the given amount.
     */
    public PaymentResult processPayment(double amount, String currency) {
        validate(amount);
        PaymentIntent intent = gateway.createIntent(amount, currency);
        return new PaymentResult(intent.getId(), "success");
    }

    private void validate(double amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException("Invalid amount");
        }
    }
}

/**
 * Result of a payment operation.
 */
public class PaymentResult {
    private final String id;
    private final String status;

    public PaymentResult(String id, String status) {
        this.id = id;
        this.status = status;
    }

    public String getId() {
        return id;
    }
}

public interface PaymentGateway {
    PaymentIntent createIntent(double amount, String currency);
    void cancelIntent(String intentId);
}

public enum PaymentStatus {
    PENDING,
    COMPLETED,
    FAILED,
    REFUNDED
}
"#;

#[test]
fn java_extracts_class() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentService"), "should extract PaymentService class");
    assert!(has_chunk!(chunks, "PaymentResult"), "should extract PaymentResult class");
}

#[test]
fn java_extracts_methods() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(
        chunks.iter().any(|c| c.name.contains("PaymentService") && c.name.contains("processPayment")),
        "should extract methods with qualified name"
    );
}

#[test]
fn java_extracts_interface() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentGateway"), "should extract interface");
}

#[test]
fn java_extracts_enum() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    assert!(has_chunk!(chunks, "PaymentStatus"), "should extract enum");
}

#[test]
fn java_extracts_calls() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("processPayment"))
        .expect("should find processPayment");

    assert!(
        method.calls.iter().any(|c| c.contains("validate")),
        "should detect validate call, got: {:?}",
        method.calls
    );
}

#[test]
fn java_extracts_doc_comments() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let class = find_chunk!(chunks, "PaymentService");
    assert!(
        class.doc_comment.as_ref().is_some_and(|d| d.contains("payment transactions")),
        "should extract Javadoc comment, got: {:?}",
        class.doc_comment
    );
}

#[test]
fn java_embed_text_and_custom_fields() {
    let chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_JAVA);

    let class = find_chunk!(chunks, "PaymentService");

    let embed = class.to_embed_text("src/PaymentService.java", &[]);
    assert!(embed.contains("src/PaymentService.java"));

    let custom = class.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ===========================================================================
// C / C++
// ===========================================================================

const SAMPLE_C: &str = r#"
#include <stdio.h>
#include <stdlib.h>

/**
 * Process a data buffer.
 */
int process_buffer(const char *data, size_t len) {
    if (data == NULL) return -1;
    validate_input(data, len);
    return transform(data, len);
}

typedef struct {
    int x;
    int y;
    char name[64];
} Point;

enum Color {
    RED,
    GREEN,
    BLUE
};

static void helper_func(int n) {
    printf("helper: %d\n", n);
}
"#;

const SAMPLE_CPP: &str = r#"
#include <string>
#include <vector>
#include <memory>

/**
 * Represents a graph node.
 */
class GraphNode {
public:
    GraphNode(int id, std::string label)
        : id_(id), label_(std::move(label)) {}

    /** Get the node ID. */
    int getId() const { return id_; }

    /** Get the label. */
    std::string getLabel() const { return label_; }

    /** Add a neighbor node. */
    void addNeighbor(std::shared_ptr<GraphNode> node) {
        neighbors_.push_back(node);
        updateIndex();
    }

private:
    int id_;
    std::string label_;
    std::vector<std::shared_ptr<GraphNode>> neighbors_;

    void updateIndex() {
        // internal bookkeeping
    }
};

/** Build a graph from raw data. */
std::vector<GraphNode> buildGraph(const std::vector<int>& ids) {
    std::vector<GraphNode> nodes;
    for (auto id : ids) {
        nodes.emplace_back(id, std::to_string(id));
    }
    return nodes;
}

enum class NodeType {
    Source,
    Sink,
    Intermediate
};

struct EdgeWeight {
    double weight;
    bool directed;
};
"#;

#[test]
fn c_extracts_function() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert_eq!(func.kind, CodeNodeKind::Function);
}

#[test]
fn c_extracts_struct() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    assert!(
        chunks.iter().any(|c| c.name == "Point" && c.kind == CodeNodeKind::Struct),
        "should extract typedef struct as named struct"
    );
}

#[test]
fn c_extracts_enum() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    assert!(has_chunk!(chunks, "Color"), "should extract enum Color");
}

#[test]
fn c_extracts_calls() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert!(
        func.calls.iter().any(|c| c.contains("validate_input")),
        "should detect validate_input call, got: {:?}",
        func.calls
    );
}

#[test]
fn c_extracts_doc_comments() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");
    assert!(
        func.doc_comment.as_ref().is_some_and(|d| d.contains("data buffer")),
        "should extract C block comment as doc, got: {:?}",
        func.doc_comment
    );
}

#[test]
fn cpp_extracts_class() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "GraphNode"), "should extract class GraphNode");
}

#[test]
fn cpp_extracts_methods() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(
        chunks.iter().any(|c| c.name.contains("GraphNode") && c.name.contains("addNeighbor")),
        "should extract class methods with qualified name"
    );
}

#[test]
fn cpp_extracts_free_function() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "buildGraph"), "should extract free function");
}

#[test]
fn cpp_extracts_enum_class() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "NodeType"), "should extract enum class");
}

#[test]
fn cpp_extracts_struct() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    assert!(has_chunk!(chunks, "EdgeWeight"), "should extract struct");
}

#[test]
fn cpp_extracts_calls() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    let method = chunks
        .iter()
        .find(|c| c.name.contains("addNeighbor"))
        .expect("should find addNeighbor");

    assert!(
        method.calls.iter().any(|c| c.contains("updateIndex") || c.contains("push_back")),
        "should detect calls in method body, got: {:?}",
        method.calls
    );
}

#[test]
fn cpp_embed_text_and_custom_fields() {
    let chunker = super::CppCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_CPP);

    let class = find_chunk!(chunks, "GraphNode");

    let embed = class.to_embed_text("src/graph.cpp", &[]);
    assert!(embed.contains("src/graph.cpp"));

    let custom = class.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

#[test]
fn c_embed_text_and_custom_fields() {
    let chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_C);

    let func = find_chunk!(chunks, "process_buffer");

    let embed = func.to_embed_text("src/buffer.c", &[]);
    assert!(embed.contains("[function]"));
    assert!(embed.contains("src/buffer.c"));

    let custom = func.to_custom_fields(&[]);
    assert!(custom.contains_key("kind"));
    assert!(custom.contains_key("name"));
}

// ===========================================================================
// Cross-language: all chunkers produce valid CodeChunk fields
// ===========================================================================

#[test]
fn all_chunks_have_valid_line_ranges() {
    let ts_chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let py_chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let go_chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let java_chunker = super::JavaCodeChunker::new(CodeChunkConfig::default());
    let c_chunker = super::CCodeChunker::new(CodeChunkConfig::default());
    let cpp_chunker = super::CppCodeChunker::new(CodeChunkConfig::default());

    let all_chunks: Vec<(&str, Vec<super::CodeChunk>)> = vec![
        ("ts", ts_chunker.chunk(SAMPLE_TS)),
        ("py", py_chunker.chunk(SAMPLE_PY)),
        ("go", go_chunker.chunk(SAMPLE_GO)),
        ("java", java_chunker.chunk(SAMPLE_JAVA)),
        ("c", c_chunker.chunk(SAMPLE_C)),
        ("cpp", cpp_chunker.chunk(SAMPLE_CPP)),
    ];

    for (lang, chunks) in &all_chunks {
        assert!(!chunks.is_empty(), "{lang} should produce chunks");
        for chunk in chunks {
            assert!(
                chunk.end_line >= chunk.start_line,
                "{lang}: {}: end_line ({}) < start_line ({})",
                chunk.name,
                chunk.end_line,
                chunk.start_line
            );
            assert!(
                chunk.end_byte > chunk.start_byte,
                "{lang}: {}: end_byte ({}) <= start_byte ({})",
                chunk.name,
                chunk.end_byte,
                chunk.start_byte
            );
            assert!(
                !chunk.name.is_empty(),
                "{lang}: chunk at line {} has empty name",
                chunk.start_line
            );
            assert!(
                !chunk.text.is_empty(),
                "{lang}: {}: chunk has empty text",
                chunk.name
            );
        }
    }
}

#[test]
fn all_chunkers_set_chunk_index_sequentially() {
    let ts_chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let py_chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());

    for (label, chunks) in [
        ("ts", ts_chunker.chunk(SAMPLE_TS)),
        ("py", py_chunker.chunk(SAMPLE_PY)),
    ] {
        let indices: Vec<usize> = chunks.iter().map(|c| c.chunk_index).collect();
        for (i, idx) in indices.iter().enumerate() {
            assert_eq!(
                *idx, i,
                "{label}: chunk_index should be sequential, expected {i} at position {i}, got {idx}"
            );
        }
    }
}

// ===========================================================================
// Update command: language detection for file extensions
// ===========================================================================

#[test]
fn language_tag_for_extension() {
    // This tests the expected lang_for_extension helper that update.rs should use
    let cases = [
        ("rs", "rust"),
        ("ts", "typescript"),
        ("tsx", "typescript"),
        ("py", "python"),
        ("go", "go"),
        ("java", "java"),
        ("c", "c"),
        ("h", "c"),
        ("cpp", "cpp"),
        ("hpp", "cpp"),
    ];

    for (ext, expected_lang) in cases {
        let lang = super::lang_for_extension(ext);
        assert_eq!(
            lang,
            Some(expected_lang),
            "extension {ext:?} should map to lang {expected_lang:?}"
        );
    }
}

#[test]
fn lang_for_unknown_extension_returns_none() {
    assert_eq!(super::lang_for_extension("txt"), None);
    assert_eq!(super::lang_for_extension("md"), None);
    assert_eq!(super::lang_for_extension("json"), None);
}

// ===========================================================================
// chunk_for_language dispatcher
// ===========================================================================

#[test]
fn dispatcher_routes_to_correct_chunker() {
    // Rust should work (already implemented)
    let rs_chunks = super::chunk_for_language("rs", SAMPLE_RUST_MINI);
    assert!(rs_chunks.is_some(), "rs extension should be supported");
    let rs_chunks = rs_chunks.unwrap();
    assert!(!rs_chunks.is_empty(), "Rust dispatcher should produce chunks");
    assert!(
        rs_chunks.iter().any(|c| c.name == "hello"),
        "should find hello function via dispatcher"
    );
}

const SAMPLE_RUST_MINI: &str = r#"
pub fn hello() -> String {
    String::from("hi")
}
"#;

#[test]
fn dispatcher_returns_none_for_unsupported() {
    assert!(super::chunk_for_language("txt", "hello").is_none());
    assert!(super::chunk_for_language("md", "# heading").is_none());
    assert!(super::chunk_for_language("json", "{}").is_none());
}

#[test]
fn dispatcher_routes_ts_extension() {
    let chunks = super::chunk_for_language("ts", SAMPLE_TS);
    assert!(chunks.is_some(), "ts extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "TS dispatcher should produce chunks");
}

#[test]
fn dispatcher_routes_tsx_extension() {
    let chunks = super::chunk_for_language("tsx", SAMPLE_TS);
    assert!(chunks.is_some(), "tsx extension should be supported");
}

#[test]
fn dispatcher_routes_py_extension() {
    let chunks = super::chunk_for_language("py", SAMPLE_PY);
    assert!(chunks.is_some(), "py extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "Python dispatcher should produce chunks");
}

#[test]
fn dispatcher_routes_go_extension() {
    let chunks = super::chunk_for_language("go", SAMPLE_GO);
    assert!(chunks.is_some(), "go extension should be supported");
    let chunks = chunks.unwrap();
    assert!(!chunks.is_empty(), "Go dispatcher should produce chunks");
}

// ===========================================================================
// TypeScript: additional edge cases
// ===========================================================================

#[test]
fn ts_extracts_imports() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    // At least one chunk should have imports populated
    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "TS chunks should include file-level imports");
}

#[test]
fn ts_function_has_param_types() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "req"),
        "should extract req param, got: {:?}",
        func.param_types
    );
}

#[test]
fn ts_function_has_return_type() {
    let chunker = super::TypeScriptCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_TS);

    let func = find_chunk!(chunks, "handleRequest");
    assert!(
        func.return_type.as_ref().is_some_and(|r| r.contains("Promise")),
        "should extract Promise return type, got: {:?}",
        func.return_type
    );
}

// ===========================================================================
// Python: additional edge cases
// ===========================================================================

#[test]
fn py_extracts_private_function() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    assert!(
        has_chunk!(chunks, "_private_helper"),
        "should extract private helper function"
    );
}

#[test]
fn py_function_has_param_types() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "items"),
        "should extract items param, got: {:?}",
        func.param_types
    );
}

#[test]
fn py_function_has_return_type() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let func = find_chunk!(chunks, "process_data");
    assert!(
        func.return_type.is_some(),
        "should extract return type annotation, got: {:?}",
        func.return_type
    );
}

#[test]
fn py_extracts_imports() {
    let chunker = super::PythonCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_PY);

    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "Python chunks should include file-level imports");
}

// ===========================================================================
// Go: additional edge cases
// ===========================================================================

#[test]
fn go_extracts_imports() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let has_imports = chunks.iter().any(|c| !c.imports.is_empty());
    assert!(has_imports, "Go chunks should include file-level imports");
}

#[test]
fn go_function_has_param_types() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.param_types.iter().any(|(n, _)| n == "w" || n == "r"),
        "should extract Go params, got: {:?}",
        func.param_types
    );
}

#[test]
fn go_function_has_return_type() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    let func = find_chunk!(chunks, "HandleRequest");
    assert!(
        func.return_type.as_ref().is_some_and(|r| r.contains("error")),
        "should extract error return type, got: {:?}",
        func.return_type
    );
}

#[test]
fn go_extracts_type_alias() {
    let chunker = super::GoCodeChunker::new(CodeChunkConfig::default());
    let chunks = chunker.chunk(SAMPLE_GO);

    // `type Status int` is a type declaration
    assert!(
        chunks.iter().any(|c| c.name == "Status"),
        "should extract type alias Status"
    );
}
