use super::{CodeChunk, CodeNodeKind, extract};

super::define_chunker!(CppCodeChunker);

const CPP_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[("function_definition", CodeNodeKind::Function)],
    extract_name_fn: extract::extract_c_func_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_c_params,
    extract_return_fn: extract::extract_c_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["function_definition"],
};

impl CppCodeChunker {
    pub fn chunk(&self, source: &str) -> Vec<CodeChunk> {
        let Some(parsed) = extract::parse_source(
            tree_sitter_cpp::LANGUAGE.into(),
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
                        &self.config, &CPP_EXTRACTORS, &child, src, &imports, chunks.len(),
                    ) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                }
                "class_specifier" => {
                    if let Some(mut chunk) = self.cpp_class_to_chunk(&child, src, &imports, chunks.len()) {
                        chunk.doc_comment = doc;
                        chunks.push(chunk);
                    }
                    let name = extract::extract_name(&child, src);
                    extract::extract_methods(
                        &self.config, &CPP_EXTRACTORS, &child, src, &imports, &name, &mut chunks,
                    );
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
                _ => {}
            }
        }

        chunks
    }

    fn cpp_class_to_chunk(
        &self,
        node: &tree_sitter::Node,
        src: &[u8],
        imports: &[String],
        index: usize,
    ) -> Option<CodeChunk> {
        let mut chunk = extract::simple_type_chunk(
            node, src, CodeNodeKind::Class, None, imports, index, 0,
        )?;
        // Class kind always needs type_refs (simple_type_chunk only extracts for Struct)
        chunk.type_refs = extract::extract_type_refs(node, src);
        Some(chunk)
    }
}
