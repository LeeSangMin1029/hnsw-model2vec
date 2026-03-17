use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(CCodeChunker);

const C_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[("function_definition", CodeNodeKind::Function)],
    extract_name_fn: extract::extract_c_func_name,
    extract_vis_fn: extract::extract_c_visibility,
    extract_params_fn: extract::extract_c_params,
    extract_return_fn: extract::extract_c_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &[],
    type_chunk_kinds: &[
        ("struct_specifier", CodeNodeKind::Struct),
        ("enum_specifier", CodeNodeKind::Enum),
    ],
    method_parent_kinds: &[],
    wrapper_kind: None,
};

impl CCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let Some(parsed) = extract::parse_source(
            tree_sitter_c::LANGUAGE.into(),
            source,
            self.config.extract_imports,
            &["preproc_include"],
        ) else {
            return Vec::new();
        };

        let mut chunks = extract::chunk_standard(&self.config, &C_EXTRACTORS, &parsed, source);

        // Handle C typedefs (not covered by standard kind_map/type_chunk_kinds)
        let root = parsed.tree.root_node();
        let src = source.as_bytes();
        let mut cursor = root.walk();
        for child in root.children(&mut cursor) {
            if child.kind() == "type_definition" {
                let doc = extract::extract_block_doc_comment_before(&root, &child, src);
                if let Some(mut chunk) = self.c_typedef_to_chunk(&child, src, &parsed.imports, chunks.len()) {
                    chunk.doc_comment = doc;
                    chunks.push(chunk);
                }
            }
        }

        chunks
    }

    fn c_typedef_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let line_count = text.lines().count();
        if line_count < self.config.min_lines {
            return None;
        }

        let mut name = String::new();
        let mut has_struct = false;
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "struct_specifier" {
                has_struct = true;
            }
            if child.kind() == "type_identifier" {
                name = child.utf8_text(src).unwrap_or_default().to_owned();
            }
        }

        if name.is_empty() {
            return None;
        }

        let kind = if has_struct {
            CodeNodeKind::Struct
        } else {
            CodeNodeKind::TypeAlias
        };

        Some(CodeChunk {
            text,
            kind,
            name,
            signature: None,
            doc_comment: None,
            visibility: String::new(),
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls: Vec::new(),
            call_lines: Vec::new(),
            type_refs: extract::collect_sorted_unique(node, src, extract::walk_for_type_ids),
            param_types: Vec::new(),
            field_types: Vec::new(),
            local_types: Vec::new(),
            let_call_bindings: Vec::new(),
            field_accesses: Vec::new(),
            enum_variants: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0, sub_blocks: Vec::new(), string_args: Vec::new(), param_flows: Vec::new(),
        })
    }
}
