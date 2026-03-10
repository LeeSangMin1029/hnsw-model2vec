use super::typescript::TS_EXTRACTORS;

super::define_chunker!(JavaScriptCodeChunker, tree_sitter_typescript::LANGUAGE_TSX, &["import_statement"], &TS_EXTRACTORS);
