//! Lightweight impl-block parser for external type method extraction.
//!
//! Parses Rust source files to extract `(type_name, method_name)` pairs
//! from `impl Type { fn method() }` blocks. Skips function bodies entirely
//! for maximum speed. Used to build an external type→method index from
//! std library and cargo dependency sources.

use tree_sitter::{Language, Parser};

/// Extract `(type_name, method_name)` pairs from Rust source code.
///
/// Scans all `impl_item` nodes for their type and method names.
/// Handles both inherent impls (`impl Vec<T> { fn len() }`) and
/// trait impls (`impl Iterator for MyIter { fn next() }`).
///
/// For trait impls, returns the concrete type name (after `for`).
pub fn extract_impl_methods(src: &[u8]) -> Vec<(String, String)> {
    let lang: Language = tree_sitter_rust::LANGUAGE.into();
    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(src, None) else {
        return Vec::new();
    };
    let mut results = Vec::new();
    collect_impl_methods(&tree.root_node(), src, &mut results);
    results
}

fn collect_impl_methods(
    node: &tree_sitter::Node,
    src: &[u8],
    results: &mut Vec<(String, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "impl_item" {
            if let Some(type_name) = extract_impl_type(&child, src) {
                collect_methods_in_impl(&child, src, &type_name, results);
            }
        } else {
            // Recurse into modules, but not into function bodies
            let kind = child.kind();
            if kind == "mod_item" || kind == "source_file" || kind == "declaration_list" {
                collect_impl_methods(&child, src, results);
            }
        }
    }
}

/// Extract the concrete type name from an `impl_item` node.
///
/// - Inherent impl: `impl Vec<T> { ... }` → `"vec"`
/// - Trait impl: `impl Display for MyStruct { ... }` → `"mystruct"`
fn extract_impl_type(node: &tree_sitter::Node, src: &[u8]) -> Option<String> {
    // Look for the type in the impl header.
    // For "impl Trait for Type", we want the type after "for".
    // For "impl Type", we want the type directly.
    let mut has_for = false;
    let mut type_node = None;
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        let kind = child.kind();
        if kind == "declaration_list" {
            break;
        }
        if child.utf8_text(src).ok() == Some("for") {
            has_for = true;
            continue;
        }
        // Type nodes: type_identifier, scoped_type_identifier, generic_type, primitive_type
        if matches!(
            kind,
            "type_identifier" | "scoped_type_identifier" | "generic_type" | "primitive_type"
        ) {
            if has_for {
                // After "for" → this is the concrete type
                type_node = Some(child);
                break;
            }
            type_node = Some(child);
        }
    }

    let node = type_node?;
    // Extract the leaf type name (strip generics and path prefix)
    let text = node.utf8_text(src).ok()?;
    let leaf = extract_type_leaf(text);
    if leaf.is_empty() {
        return None;
    }
    Some(leaf.to_lowercase())
}

/// Extract method names from an impl block's declaration_list.
fn collect_methods_in_impl(
    impl_node: &tree_sitter::Node,
    src: &[u8],
    type_name: &str,
    results: &mut Vec<(String, String)>,
) {
    let mut cursor = impl_node.walk();
    for child in impl_node.children(&mut cursor) {
        if child.kind() == "declaration_list" {
            let mut inner = child.walk();
            for item in child.children(&mut inner) {
                if item.kind() == "function_item"
                    && let Some(name_node) = item.child_by_field_name("name")
                        && let Ok(method_name) = name_node.utf8_text(src) {
                            results.push((
                                type_name.to_owned(),
                                method_name.to_lowercase(),
                            ));
                        }
            }
        }
    }
}

/// Extract the leaf type name from a type string.
///
/// `"std::collections::HashMap<K, V>"` → `"HashMap"`
/// `"Vec<T>"` → `"Vec"`
/// `"MyStruct"` → `"MyStruct"`
fn extract_type_leaf(ty: &str) -> &str {
    // Strip generics
    let ty = ty.split('<').next().unwrap_or(ty);
    // Take last path segment
    ty.rsplit("::").next().unwrap_or(ty).trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inherent_impl() {
        let src = b"impl MyStruct { fn foo(&self) {} fn bar() {} }";
        let methods = extract_impl_methods(src);
        assert_eq!(methods.len(), 2);
        assert_eq!(methods[0], ("mystruct".to_owned(), "foo".to_owned()));
        assert_eq!(methods[1], ("mystruct".to_owned(), "bar".to_owned()));
    }

    #[test]
    fn trait_impl_uses_concrete_type() {
        let src = b"impl Display for MyStruct { fn fmt(&self) {} }";
        let methods = extract_impl_methods(src);
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0], ("mystruct".to_owned(), "fmt".to_owned()));
    }

    #[test]
    fn generic_impl() {
        let src = b"impl<T> Vec<T> { fn len(&self) -> usize { 0 } fn push(&mut self, _: T) {} }";
        let methods = extract_impl_methods(src);
        assert_eq!(methods.len(), 2);
        assert_eq!(methods[0].0, "vec");
        assert_eq!(methods[0].1, "len");
        assert_eq!(methods[1].1, "push");
    }

    #[test]
    fn nested_in_module() {
        let src = b"mod inner { impl Foo { fn baz() {} } }";
        let methods = extract_impl_methods(src);
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0], ("foo".to_owned(), "baz".to_owned()));
    }

    #[test]
    fn empty_impl() {
        let src = b"impl EmptyStruct {}";
        let methods = extract_impl_methods(src);
        assert!(methods.is_empty());
    }

    #[test]
    fn scoped_type() {
        let src = b"impl std::fmt::Display for MyType { fn fmt(&self) {} }";
        let methods = extract_impl_methods(src);
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].0, "mytype");
    }
}
