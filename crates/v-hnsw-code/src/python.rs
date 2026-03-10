use super::{CodeNodeKind, extract};

const PY_EXTRACTORS: extract::LangExtractors = extract::LangExtractors {
    kind_map: &[
        ("function_definition", CodeNodeKind::Function),
        ("class_definition", CodeNodeKind::Class),
    ],
    extract_name_fn: extract::extract_name,
    extract_vis_fn: extract::no_visibility,
    extract_params_fn: extract::extract_py_params,
    extract_return_fn: extract::extract_python_return_type,
    extract_doc_fn: extract::extract_python_doc_wrapper,
    method_kinds: &["function_definition"],
    type_chunk_kinds: &[],
    method_parent_kinds: &["class_definition"],
    wrapper_kind: Some(("decorated_definition", "")),
};

super::define_chunker!(PythonCodeChunker, tree_sitter_python::LANGUAGE, &["import_statement", "import_from_statement"], &PY_EXTRACTORS);
