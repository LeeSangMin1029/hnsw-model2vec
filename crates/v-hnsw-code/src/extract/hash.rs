//! AST hashing, body hashing, sub-block splitting, and MinHash signatures.

use std::hash::{Hash, Hasher};

use crate::SubBlock;

// ---------------------------------------------------------------------------
// AST structural hash — for code clone detection
// ---------------------------------------------------------------------------

/// Compute a structural hash of a tree-sitter node subtree.
///
/// Identifier/literal *values* are ignored (normalized), so two functions
/// that differ only in variable names produce the same hash (Type-2 clones).
/// The hash captures: node kinds, child structure, and operator tokens.
pub fn ast_structure_hash(node: &tree_sitter::Node, src: &[u8]) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    hash_node_recursive(node, src, &mut hasher);
    hasher.finish()
}

fn hash_node_recursive(node: &tree_sitter::Node, src: &[u8], h: &mut impl Hasher) {
    let kind = node.kind();
    kind.hash(h);

    // For operators and keywords, include the actual text (e.g., "+", "return").
    // For identifiers and literals, hash only the *kind* (already done above)
    // so that renamed variables / different constants still match.
    if !is_identifier_or_literal(kind) && node.child_count() == 0 {
        if let Ok(text) = node.utf8_text(src) {
            text.hash(h);
        }
    }

    // Hash child count to distinguish `f(a)` from `f(a, b)`
    node.child_count().hash(h);

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        hash_node_recursive(&child, src, h);
    }
}

/// Node kinds whose text values should be normalized (not hashed).
fn is_identifier_or_literal(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "type_identifier"
            | "field_identifier"
            | "property_identifier"
            | "shorthand_property_identifier"
            | "string_literal"
            | "string"
            | "string_content"
            | "template_string"
            | "number"
            | "integer_literal"
            | "float_literal"
            | "boolean_literal"
            | "char_literal"
            | "comment"
            | "line_comment"
            | "block_comment"
            | "interpreted_string_literal"
            | "raw_string_literal"
            | "rune_literal"
    )
}

/// Hash the actual code body text with normalization.
///
/// Unlike `ast_structure_hash` (which ignores identifiers), this preserves
/// variable/function names but strips comments and normalizes whitespace.
/// This catches truly duplicated logic — `fn len()` and `fn dim()` will
/// differ because identifiers are preserved.
pub fn body_hash(text: &str) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip empty lines and single-line comments
        if trimmed.is_empty()
            || trimmed.starts_with("//")
            || trimmed.starts_with('#')
            || trimmed.starts_with("///")
        {
            continue;
        }
        // Strip inline comments
        let code = trimmed
            .find("//")
            .map_or(trimmed, |pos| trimmed[..pos].trim_end());
        if !code.is_empty() {
            code.hash(&mut hasher);
        }
    }
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Sub-block splitting — for intra-function fine-grained clone detection
// ---------------------------------------------------------------------------

/// Control structure node kinds that serve as block split boundaries.
///
/// Inspired by Tamer (ISSTA 2023): splitting ASTs at control structure
/// boundaries produces sub-blocks that can be compared independently.
const BLOCK_SPLIT_KINDS: &[&str] = &[
    // Rust
    "if_expression",
    "for_expression",
    "while_expression",
    "loop_expression",
    "match_expression",
    // C-family / JS / TS / Java / Go
    "if_statement",
    "for_statement",
    "for_in_statement",
    "for_of_statement",
    "while_statement",
    "do_statement",
    "switch_statement",
    "switch_expression",
    "try_statement",
    "try_expression",
];

/// Minimum lines for a sub-block to be included.
const MIN_SUB_BLOCK_LINES: usize = 3;

/// Split a function AST node into sub-blocks at control structure boundaries.
///
/// Each returned `SubBlock` has its own AST hash and body hash, enabling
/// detection of cloned logic *within* and *across* functions.
pub fn split_into_blocks(
    node: &tree_sitter::Node,
    src: &[u8],
) -> Vec<SubBlock> {
    let mut blocks = Vec::new();
    collect_blocks(node, src, &mut blocks);
    blocks
}

/// Recursively collect sub-blocks from control structure boundaries.
fn collect_blocks(
    node: &tree_sitter::Node,
    src: &[u8],
    out: &mut Vec<SubBlock>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if BLOCK_SPLIT_KINDS.contains(&child.kind()) {
            let start_line = child.start_position().row;
            let end_line = child.end_position().row;
            if end_line.saturating_sub(start_line) + 1 >= MIN_SUB_BLOCK_LINES {
                let ast_h = ast_structure_hash(&child, src);
                let text = child.utf8_text(src).unwrap_or("");
                let body_h = body_hash(text);
                out.push(SubBlock {
                    start_byte: child.start_byte(),
                    end_byte: child.end_byte(),
                    start_line,
                    end_line,
                    ast_hash: ast_h,
                    body_hash: body_h,
                });
            }
        }
        // Recurse into children to find nested control structures
        collect_blocks(&child, src, out);
    }
}

// ---------------------------------------------------------------------------
// Token-based MinHash — for near-duplicate (Type-3) clone detection
// ---------------------------------------------------------------------------

/// Tokenize code body for clone detection.
///
/// Strips comments, normalizes number literals to `$N`, keeps identifiers
/// and keywords. Returns unigram tokens.
pub fn code_tokens(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut in_block_comment = false;

    for line in text.lines() {
        let trimmed = line.trim();

        // Block comment tracking
        if in_block_comment {
            if let Some(pos) = trimmed.find("*/") {
                let rest = &trimmed[pos + 2..];
                in_block_comment = false;
                tokenize_line(rest, &mut tokens);
            }
            continue;
        }
        if trimmed.starts_with("/*") {
            if !trimmed.contains("*/") {
                in_block_comment = true;
            }
            continue;
        }

        // Skip line comments
        if trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        // Strip inline comments
        let code = trimmed.find("//").map_or(trimmed, |pos| &trimmed[..pos]);
        tokenize_line(code, &mut tokens);
    }
    tokens
}

fn tokenize_line(code: &str, tokens: &mut Vec<String>) {
    for word in code.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if word.is_empty() {
            continue;
        }
        if word.chars().all(|c| c.is_ascii_digit()) {
            tokens.push("$N".to_owned());
        } else {
            tokens.push(word.to_owned());
        }
    }
}

/// Compute MinHash signature from tokens (unigrams + bigrams).
///
/// Returns `k` minimum hash values, one per hash function.
/// Jaccard similarity ≈ fraction of matching positions between two signatures.
pub fn minhash_signature(tokens: &[String], k: usize) -> Vec<u64> {
    // Build feature set: unigrams + bigrams for sequence sensitivity
    let n_features = tokens.len() + tokens.len().saturating_sub(1);
    let mut features = Vec::with_capacity(n_features);
    for t in tokens {
        features.push(t.as_str());
    }

    // Bigrams stored as owned strings
    let bigrams: Vec<String> = tokens
        .windows(2)
        .map(|w| format!("{}_{}", w[0], w[1]))
        .collect();
    for b in &bigrams {
        features.push(b.as_str());
    }

    (0..k)
        .map(|seed| {
            features
                .iter()
                .map(|feature| {
                    let mut hasher = std::hash::DefaultHasher::new();
                    (seed as u64).hash(&mut hasher);
                    feature.hash(&mut hasher);
                    hasher.finish()
                })
                .min()
                .unwrap_or(u64::MAX)
        })
        .collect()
}

/// Estimate Jaccard similarity from two MinHash signatures.
pub fn jaccard_from_minhash(a: &[u64], b: &[u64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let matches = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    matches as f64 / a.len() as f64
}

/// Encode MinHash signature as compact hex string.
pub fn minhash_to_hex(sig: &[u64]) -> String {
    let mut hex = String::with_capacity(sig.len() * 16);
    for h in sig {
        use std::fmt::Write;
        let _ = write!(hex, "{h:016x}");
    }
    hex
}

/// Decode MinHash signature from hex string.
pub fn minhash_from_hex(hex: &str) -> Option<Vec<u64>> {
    if hex.len() % 16 != 0 {
        return None;
    }
    let k = hex.len() / 16;
    let mut sig = Vec::with_capacity(k);
    for i in 0..k {
        let chunk = &hex[i * 16..(i + 1) * 16];
        sig.push(u64::from_str_radix(chunk, 16).ok()?);
    }
    Some(sig)
}

/// Number of MinHash functions to use.
pub const MINHASH_K: usize = 64;

/// Parse source code with the given tree-sitter language and extract imports.
///
/// Returns `None` if the language cannot be set or parsing fails.
pub fn parse_source(
    language: tree_sitter::Language,
    source: &str,
    extract_imports: bool,
    import_kinds: &[&str],
) -> Option<super::ParsedSource> {
    let mut parser = tree_sitter::Parser::new();
    if parser.set_language(&language).is_err() {
        return None;
    }
    let tree = parser.parse(source, None)?;
    let root = tree.root_node();
    let src = source.as_bytes();
    let imports = if extract_imports {
        super::lang::extract_imports_by_kind(&root, src, import_kinds)
    } else {
        Vec::new()
    };
    Some(super::ParsedSource { tree, imports })
}
