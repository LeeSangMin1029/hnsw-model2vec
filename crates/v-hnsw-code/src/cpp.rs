use super::{CodeNodeKind, extract};

const CPP_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[("function_definition", CodeNodeKind::Function)],
    extract_name_fn: extract::extract_c_func_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_c_params,
    extract_return_fn: extract::extract_c_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["function_definition"],
    type_chunk_kinds: &[
        ("struct_specifier", CodeNodeKind::Struct),
        ("enum_specifier", CodeNodeKind::Enum),
        ("class_specifier", CodeNodeKind::Class),
    ],
    method_parent_kinds: &["class_specifier"],
    wrapper_kind: None,
};

super::define_chunker!(CppCodeChunker, tree_sitter_cpp::LANGUAGE, &["preproc_include"], &CPP_EXTRACTORS);
