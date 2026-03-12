use super::{CodeNodeKind, extract};

pub(crate) const TS_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_declaration", CodeNodeKind::Function),
        ("class_declaration", CodeNodeKind::Class),
        ("interface_declaration", CodeNodeKind::Interface),
        ("enum_declaration", CodeNodeKind::Enum),
        ("type_alias_declaration", CodeNodeKind::TypeAlias),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_ts_params,
    extract_return_fn: extract::extract_ts_return_type,
    extract_doc_fn: extract::extract_block_doc_comment_before,
    method_kinds: &["method_definition"],
    type_chunk_kinds: &[],
    method_parent_kinds: &["class_declaration"],
    wrapper_kind: Some(("export_statement", "export")),
};

super::define_chunker!(TypeScriptCodeChunker, tree_sitter_typescript::LANGUAGE_TYPESCRIPT, &["import_statement"], &TS_EXTRACTORS);
