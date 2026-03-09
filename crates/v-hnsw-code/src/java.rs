use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(JavaCodeChunker);

const JAVA_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("class_declaration", CodeNodeKind::Class),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_java_visibility,
    extract_params_fn: extract::extract_java_params,
    extract_return_fn: extract::extract_java_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["method_declaration", "constructor_declaration"],
};

impl JavaCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let mut parser = tree_sitter::Parser::new();
        let language: tree_sitter::Language = tree_sitter_java::LANGUAGE.into();
        if parser.set_language(&language).is_err() {
            return Vec::new();
        }

        let Some(tree) = parser.parse(source, None) else {
            return Vec::new();
        };

        let root = tree.root_node();
        let src = source.as_bytes();

        let imports = if self.config.extract_imports {
            extract::extract_imports_by_kind(&root, src, &["import_declaration"])
        } else {
            Vec::new()
        };

        let mut chunks = Vec::new();
        self.walk_java_declarations(&root, src, &imports, &mut chunks);

        chunks
    }

    fn walk_java_declarations(
        &self,
        parent: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        chunks: &mut Vec<CodeChunk>,
    ) {
        let mut cursor = parent.walk();
        for child in parent.children(&mut cursor) {
            let doc = extract::extract_block_doc_comment_before(parent, &child, src);

            match child.kind() {
                "class_declaration" => {
                    if let Some(mut chunk) = self.java_class_to_chunk(&child, src, imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                    let name = extract::extract_name(&child, src);
                    if let Some(body) = child.child_by_field_name("body") {
                        extract::extract_methods(
                            &self.config, &JAVA_EXTRACTORS, &child, src, imports, &name, chunks,
                        );
                        // Java class body is "class_body", not the node itself.
                        // extract_methods looks for child_by_field_name("body") internally.
                        let _ = body; // used via extract_methods
                    }
                }
                "interface_declaration" => {
                    if let Some(mut chunk) = self.java_type_decl_to_chunk(&child, src, imports, chunks.len(), CodeNodeKind::Interface) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "enum_declaration" => {
                    if let Some(mut chunk) = self.java_type_decl_to_chunk(&child, src, imports, chunks.len(), CodeNodeKind::Enum) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                _ => {}
            }
        }
    }

    fn java_class_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let name = extract::extract_name(node, src);
        let visibility = extract::extract_java_visibility(node, src);
        let type_refs = extract::extract_type_refs(node, src);

        Some(CodeChunk {
            text,
            kind: CodeNodeKind::Class,
            name,
            signature: None,
            doc_comment: None,
            visibility,
            start_line: node.start_position().row,
            end_line: node.end_position().row,
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            chunk_index: index,
            imports: imports.to_vec(),
            calls: Vec::new(),
            type_refs,
            param_types: Vec::new(),
            return_type: None, ast_hash: 0, body_hash: 0,
        })
    }

    fn java_type_decl_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
        kind: CodeNodeKind,
    ) -> Option<CodeChunk> {
        let text = node.utf8_text(src).ok()?.to_owned();
        let name = extract::extract_name(node, src);
        let visibility = extract::extract_java_visibility(node, src);

        Some(CodeChunk {
            text,
            kind,
            name,
            signature: None,
            doc_comment: None,
            visibility,
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
