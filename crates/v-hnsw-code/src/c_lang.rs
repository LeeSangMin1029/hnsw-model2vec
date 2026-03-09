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
        let root = parsed.tree.root_node();
        let src = source.as_bytes();
        let imports = parsed.imports;

        let mut chunks = Vec::new();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            let doc = extract::extract_block_doc_comment_before(&root, &child, src);

            match child.kind() {
                "function_definition" => {
                    if let Some(mut chunk) = extract::build_chunk(
                        &self.config, &C_EXTRACTORS, &child, src, &imports, chunks.len(),
                    ) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "struct_specifier" => {
                    if let Some(chunk) = extract::simple_type_chunk(
                        &child, src, CodeNodeKind::Struct, doc, &imports, chunks.len(), self.config.min_lines,
                    ) {
                        chunks.push(chunk);
                    }
                }
                "enum_specifier" => {
                    if let Some(chunk) = extract::simple_type_chunk(
                        &child, src, CodeNodeKind::Enum, doc, &imports, chunks.len(), self.config.min_lines,
                    ) {
                        chunks.push(chunk);
                    }
                }
                "type_definition" => {
                    if let Some(mut chunk) = self.c_typedef_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                _ => {}
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
            type_refs: extract::extract_type_refs(node, src),
            param_types: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0,
        })
    }
}
