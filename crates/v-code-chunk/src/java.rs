use super::{CodeNodeKind, extract};

const JAVA_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_java_visibility,
    extract_params_fn: extract::extract_java_params,
    extract_return_fn: extract::extract_c_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["method_declaration", "constructor_declaration"],
    type_chunk_kinds: &[
        ("class_declaration", CodeNodeKind::Class),
        ("interface_declaration", CodeNodeKind::Interface),
        ("enum_declaration", CodeNodeKind::Enum),
    ],
    method_parent_kinds: &["class_declaration"],
    wrapper_kind: None,
};

super::define_chunker!(JavaCodeChunker, tree_sitter_java::LANGUAGE, &["import_declaration"], &JAVA_EXTRACTORS);
