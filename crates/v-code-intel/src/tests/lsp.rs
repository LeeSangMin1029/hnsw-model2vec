use crate::lsp::normalize_chunk_name;

#[test]
fn normalize_chunk_name_lowercase() {
    assert_eq!(normalize_chunk_name("MyStruct"), "mystruct");
}

#[test]
fn normalize_chunk_name_already_lowercase() {
    assert_eq!(normalize_chunk_name("foo_bar"), "foo_bar");
}

#[test]
fn normalize_chunk_name_mixed_case_with_colons() {
    assert_eq!(normalize_chunk_name("Foo::BarBaz"), "foo::barbaz");
}

#[test]
fn normalize_chunk_name_empty() {
    assert_eq!(normalize_chunk_name(""), "");
}

#[test]
fn normalize_chunk_name_uppercase() {
    assert_eq!(normalize_chunk_name("ALL_CAPS"), "all_caps");
}
