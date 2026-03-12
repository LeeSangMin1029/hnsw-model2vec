use super::{CodeNodeKind, extract};

const RUST_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_item", CodeNodeKind::Function),
        ("struct_item", CodeNodeKind::Struct),
        ("enum_item", CodeNodeKind::Enum),
        ("impl_item", CodeNodeKind::Impl),
        ("trait_item", CodeNodeKind::Trait),
        ("type_item", CodeNodeKind::TypeAlias),
        ("const_item", CodeNodeKind::Const),
        ("static_item", CodeNodeKind::Static),
        ("mod_item", CodeNodeKind::Module),
        ("macro_definition", CodeNodeKind::MacroDefinition),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::extract_visibility,
    extract_params_fn: extract::extract_param_types,
    extract_return_fn: extract::extract_return_type,
    extract_doc_fn: extract::extract_doc_comment_before,
    method_kinds: &["function_item"],
    type_chunk_kinds: &[],
    method_parent_kinds: &["impl_item", "trait_item"],
    wrapper_kind: None,
};

super::define_chunker!(RustCodeChunker, tree_sitter_rust::LANGUAGE, &["use_declaration"], &RUST_EXTRACTORS);
