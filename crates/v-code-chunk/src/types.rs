use std::collections::HashMap;
use std::fmt;

use v_hnsw_core::PayloadValue;

use crate::extract;

/// A sub-block within a code chunk, split at control structure boundaries.
///
/// Used for fine-grained (intra-function) clone detection: two functions may
/// not be duplicates overall, but share identical internal blocks.
#[derive(Debug, Clone)]
pub struct SubBlock {
    /// Byte offset in source file.
    pub start_byte: usize,
    pub end_byte: usize,
    /// Line numbers (0-based).
    pub start_line: usize,
    pub end_line: usize,
    /// Structural AST hash (identifiers normalized).
    pub ast_hash: u64,
    /// Normalized body text hash.
    pub body_hash: u64,
}

impl fmt::Display for SubBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "L{}-{}", self.start_line + 1, self.end_line + 1)
    }
}

/// Configuration for code chunking.
#[derive(Debug, Clone)]
pub struct CodeChunkConfig {
    /// Minimum lines for a chunk to be included.
    pub min_lines: usize,
    /// Extract file-level `use` statements.
    pub extract_imports: bool,
    /// Extract function calls from bodies.
    pub extract_calls: bool,
}

impl Default for CodeChunkConfig {
    fn default() -> Self {
        Self {
            min_lines: 2,
            extract_imports: true,
            extract_calls: true,
        }
    }
}

/// Kind of code node extracted by tree-sitter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeNodeKind {
    Function,
    Struct,
    Enum,
    Impl,
    Trait,
    TypeAlias,
    Const,
    Static,
    Module,
    MacroDefinition,
    Class,
    Interface,
}

impl CodeNodeKind {
    /// String label for payload storage.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Function => "function",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Impl => "impl",
            Self::Trait => "trait",
            Self::TypeAlias => "type_alias",
            Self::Const => "const",
            Self::Static => "static",
            Self::Module => "module",
            Self::MacroDefinition => "macro",
            Self::Class => "class",
            Self::Interface => "interface",
        }
    }
}

/// A semantic code chunk extracted via tree-sitter.
#[derive(Debug, Clone)]
pub struct CodeChunk {
    /// The raw source code text.
    pub text: String,
    /// Kind of code node.
    pub kind: CodeNodeKind,
    /// Symbol name (e.g., `process_payment`, `PaymentIntent`).
    pub name: String,
    /// Function signature (params + return), if applicable.
    pub signature: Option<String>,
    /// Doc comment text, if any.
    pub doc_comment: Option<String>,
    /// Visibility: `"pub"`, `"pub(crate)"`, `""`.
    pub visibility: String,
    /// Start line (0-based).
    pub start_line: usize,
    /// End line (0-based).
    pub end_line: usize,
    /// Byte offsets in source.
    pub start_byte: usize,
    pub end_byte: usize,
    /// Sequential chunk index within the file.
    pub chunk_index: usize,
    /// File-level import statements.
    pub imports: Vec<String>,
    /// Function calls within the body.
    pub calls: Vec<String>,
    /// Source line (0-based) of each call in `calls` (parallel array).
    pub call_lines: Vec<u32>,
    /// Type names referenced in signature and body.
    pub type_refs: Vec<String>,
    /// Parameter name-type pairs (e.g., `[("amount", "f64")]`).
    pub param_types: Vec<(String, String)>,
    /// Return type string (e.g., `"Result<Vec<Item>>"`).
    pub return_type: Option<String>,
    /// Structural AST hash for clone detection (0 = not computed).
    pub ast_hash: u64,
    /// Normalized body text hash for exact-logic clone detection (0 = not computed).
    pub body_hash: u64,
    /// Sub-blocks split at control structure boundaries for fine-grained clone detection.
    pub sub_blocks: Vec<SubBlock>,
    /// String literal arguments found in function calls: `(callee, value, line, arg_pos)`.
    pub string_args: Vec<(String, String, u32, u8)>,
    /// Parameter-to-callee argument flows: `(param_name, param_pos, callee, callee_arg, line)`.
    pub param_flows: Vec<(String, u8, String, u8, u32)>,
}

impl CodeChunk {
    /// Build text optimized for embedding (semantic search).
    ///
    /// Includes doc comment, signature, calls, called_by — not the full body.
    /// Use `called_by` to inject reverse-reference data from cross-file analysis.
    pub fn to_embed_text(&self, file_path: &str, called_by: &[String]) -> String {
        let mut parts = Vec::new();

        // [kind] visibility name
        let vis = if self.visibility.is_empty() {
            String::new()
        } else {
            format!("{} ", self.visibility)
        };
        parts.push(format!("[{}] {vis}{}", self.kind.as_str(), self.name));

        // File location
        parts.push(format!(
            "File: {file_path}:{}-{}",
            self.start_line + 1,
            self.end_line + 1
        ));

        // Doc comment
        if let Some(ref doc) = self.doc_comment {
            parts.push(doc.clone());
        }

        // Signature
        if let Some(ref sig) = self.signature {
            parts.push(format!("Signature: {sig}"));
        }

        // Parameter types
        if !self.param_types.is_empty() {
            let params: Vec<String> = self
                .param_types
                .iter()
                .map(|(n, t)| format!("{n}: {t}"))
                .collect();
            parts.push(format!("Params: {}", params.join(", ")));
        }

        // Type references
        if !self.type_refs.is_empty() {
            parts.push(format!("Types: {}", self.type_refs.join(", ")));
        }

        // Calls (with source line annotations: name@line, 1-based)
        if !self.calls.is_empty() {
            let annotated: Vec<String> = self.calls.iter().enumerate().map(|(i, c)| {
                if let Some(&line) = self.call_lines.get(i) {
                    format!("{c}@{}", line + 1) // 0-based → 1-based
                } else {
                    c.clone()
                }
            }).collect();
            parts.push(format!("Calls: {}", annotated.join(", ")));
        }

        // String args
        if !self.string_args.is_empty() {
            let items: Vec<String> = self.string_args.iter()
                .map(|(callee, value, _, _)| format!("{callee}(\"{value}\")"))
                .collect();
            parts.push(format!("Strings: {}", items.join(", ")));
        }

        // Parameter flows
        if !self.param_flows.is_empty() {
            let items: Vec<String> = self
                .param_flows
                .iter()
                .map(|(pname, _, callee, _, _)| format!("{pname}\u{2192}{callee}"))
                .collect();
            parts.push(format!("Flows: {}", items.join(", ")));
        }

        // Called by (reverse references)
        if !called_by.is_empty() {
            parts.push(format!("Called by: {}", called_by.join(", ")));
        }

        parts.join("\n")
    }

    /// Convert code metadata to `payload.custom` fields.
    ///
    /// Pass `called_by` from cross-file reverse-reference analysis.
    pub fn to_custom_fields(&self, called_by: &[String]) -> HashMap<String, PayloadValue> {
        let mut custom = HashMap::new();

        custom.insert(
            "kind".to_owned(),
            PayloadValue::String(self.kind.as_str().to_owned()),
        );
        custom.insert(
            "name".to_owned(),
            PayloadValue::String(self.name.clone()),
        );
        custom.insert(
            "visibility".to_owned(),
            PayloadValue::String(self.visibility.clone()),
        );
        custom.insert(
            "start_line".to_owned(),
            PayloadValue::Integer(i64::try_from(self.start_line).unwrap_or(0)),
        );
        custom.insert(
            "end_line".to_owned(),
            PayloadValue::Integer(i64::try_from(self.end_line).unwrap_or(0)),
        );

        if let Some(ref sig) = self.signature {
            custom.insert("signature".to_owned(), PayloadValue::String(sig.clone()));
        }
        if let Some(ref doc) = self.doc_comment {
            custom.insert("doc".to_owned(), PayloadValue::String(doc.clone()));
        }
        if !self.calls.is_empty() {
            custom.insert(
                "calls".to_owned(),
                PayloadValue::StringList(self.calls.clone()),
            );
        }
        if !self.imports.is_empty() {
            custom.insert(
                "imports".to_owned(),
                PayloadValue::StringList(self.imports.clone()),
            );
        }
        if !called_by.is_empty() {
            custom.insert(
                "called_by".to_owned(),
                PayloadValue::StringList(called_by.to_vec()),
            );
        }
        if !self.type_refs.is_empty() {
            custom.insert(
                "type_refs".to_owned(),
                PayloadValue::StringList(self.type_refs.clone()),
            );
        }
        if let Some(ref ret) = self.return_type {
            custom.insert("return_type".to_owned(), PayloadValue::String(ret.clone()));
        }
        if self.ast_hash != 0 {
            #[expect(clippy::cast_possible_wrap, reason = "hash bits reinterpreted")]
            custom.insert(
                "ast_hash".to_owned(),
                PayloadValue::Integer(self.ast_hash as i64),
            );
        }
        if self.body_hash != 0 {
            #[expect(clippy::cast_possible_wrap, reason = "hash bits reinterpreted")]
            custom.insert(
                "body_hash".to_owned(),
                PayloadValue::Integer(self.body_hash as i64),
            );
        }

        // String args
        if !self.string_args.is_empty() {
            let encoded: Vec<String> = self.string_args.iter()
                .map(|(callee, value, line, pos)| format!("{callee}\t{value}\t{line}\t{pos}"))
                .collect();
            custom.insert(
                "string_args".to_owned(),
                PayloadValue::StringList(encoded),
            );
        }

        // Parameter flows
        if !self.param_flows.is_empty() {
            let encoded: Vec<String> = self
                .param_flows
                .iter()
                .map(|(pname, ppos, callee, carg, line)| {
                    format!("{pname}\t{ppos}\t{callee}\t{carg}\t{line}")
                })
                .collect();
            custom.insert(
                "param_flows".to_owned(),
                PayloadValue::StringList(encoded),
            );
        }

        // Sub-block AST hashes for fine-grained clone detection
        if !self.sub_blocks.is_empty() {
            let hashes: Vec<String> = self.sub_blocks.iter()
                .map(|sb| format!("{:x}:{}-{}", sb.ast_hash, sb.start_line, sb.end_line))
                .collect();
            custom.insert(
                "sub_block_hashes".to_owned(),
                PayloadValue::StringList(hashes),
            );
        }

        // MinHash token fingerprint for near-duplicate detection
        let tokens = extract::code_tokens(&self.text);
        if tokens.len() >= 10 {
            let sig = extract::minhash_signature(&tokens, extract::MINHASH_K);
            custom.insert(
                "minhash".to_owned(),
                PayloadValue::String(extract::minhash_to_hex(&sig)),
            );
        }

        custom
    }
}
