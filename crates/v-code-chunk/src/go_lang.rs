use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(GoCodeChunker);

const GO_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_declaration", CodeNodeKind::Function),
        ("method_declaration", CodeNodeKind::Function),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_go_visibility_from_node,
    extract_params_fn: extract::extract_go_params,
    extract_return_fn: extract::extract_go_return_type,
    extract_doc_fn: extract::extract_go_doc_comment_before,
    method_kinds: &[],
    type_chunk_kinds: &[],
    method_parent_kinds: &[],
    wrapper_kind: None,
};

impl GoCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let Some(parsed) = extract::parse_source(
            tree_sitter_go::LANGUAGE.into(),
            source,
            self.config.extract_imports,
            &["import_declaration"],
        ) else {
            return Vec::new();
        };

        let mut chunks = extract::chunk_standard(&self.config, &GO_EXTRACTORS, &parsed, source);

        let root = parsed.tree.root_node();
        let src = source.as_bytes();
        let mut cursor = root.walk();

        for child in root.children(&mut cursor) {
            match child.kind() {
                // Fix method names to include receiver type (e.g. `RequestData::String`)
                "method_declaration" => {
                    let receiver_type = extract_go_receiver_type(&child, src);
                    if let Some(recv) = receiver_type {
                        let raw_name = extract::extract_name(&child, src);
                        let full_name = format!("{recv}::{raw_name}");
                        if let Some(chunk) = chunks.iter_mut().find(|c| c.start_byte == child.start_byte()) {
                            chunk.name = full_name;
                        }
                    }
                }
                // Handle type declarations (struct, interface, type alias)
                "type_declaration" => {
                    let doc = extract::extract_go_doc_comment_before(&root, &child, src);
                    self.extract_go_type_decl(&child, src, &parsed.imports, &mut chunks, doc);
                }
                _ => {}
            }
        }

        chunks
    }

    fn extract_go_type_decl(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        chunks: &mut Vec<CodeChunk>,
        doc: Option<String>,
    ) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() != "type_spec" {
                continue;
            }

            let name = extract::extract_name(&child, src);

            let type_node = child.child_by_field_name("type");
            let kind = match type_node.map(|n| n.kind()) {
                Some("struct_type") => CodeNodeKind::Struct,
                Some("interface_type") => CodeNodeKind::Trait,
                _ => CodeNodeKind::TypeAlias,
            };

            let full_text = node.utf8_text(src).unwrap_or_default().to_owned();
            let line_count = full_text.lines().count();

            if line_count < self.config.min_lines && kind != CodeNodeKind::TypeAlias {
                continue;
            }

            let visibility = extract::extract_go_visibility(&name);

            chunks.push(CodeChunk {
                text: full_text,
                kind,
                name,
                signature: None,
                doc_comment: doc.clone(),
                visibility,
                start_line: node.start_position().row,
                end_line: node.end_position().row,
                start_byte: node.start_byte(),
                end_byte: node.end_byte(),
                chunk_index: chunks.len(),
                imports: imports.to_vec(),
                calls: Vec::new(),
                call_lines: Vec::new(),
                type_refs: extract::collect_sorted_unique(&child, src, extract::walk_for_type_ids),
                param_types: Vec::new(),
                return_type: None, ast_hash: 0, body_hash: 0, sub_blocks: Vec::new(), string_args: Vec::new(), param_flows: Vec::new(),
            });
        }
    }
}

/// Extract the receiver type from a Go method declaration.
fn extract_go_receiver_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    let receiver = node.child_by_field_name("receiver")?;
    let mut cursor = receiver.walk();
    for param in receiver.children(&mut cursor) {
        if param.kind() == "parameter_declaration" {
            return param.child_by_field_name("type")
                .and_then(|t| {
                    let text = t.utf8_text(src).ok()?;
                    Some(text.trim_start_matches('*').to_owned())
                });
        }
    }
    None
}
