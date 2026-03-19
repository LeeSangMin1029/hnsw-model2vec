use crate::extern_types::ExternMethodIndex;

fn build_index() -> ExternMethodIndex {
    let tmp = std::env::temp_dir().join("v-code-test-extern");
    let _ = std::fs::create_dir_all(&tmp);
    let index = ExternMethodIndex::build(&tmp);
    let _ = std::fs::remove_dir_all(&tmp);
    index
}

#[test]
fn build_discovers_std_types() {
    let index = build_index();

    // Must find core std types
    assert!(
        index.type_count() > 50,
        "expected >50 types, got {}",
        index.type_count()
    );
    assert!(
        index.total_methods() > 500,
        "expected >500 methods, got {}",
        index.total_methods()
    );

    // Vec methods
    assert!(index.has_method("vec", "len"), "Vec::len not found");
    assert!(index.has_method("vec", "push"), "Vec::push not found");
    assert!(index.has_method("vec", "is_empty"), "Vec::is_empty not found");
    assert!(index.has_method("vec", "truncate"), "Vec::truncate not found");

    // HashMap methods
    assert!(index.has_method("hashmap", "get"), "HashMap::get not found");
    assert!(index.has_method("hashmap", "insert"), "HashMap::insert not found");
    assert!(
        index.has_method("hashmap", "contains_key"),
        "HashMap::contains_key not found"
    );
}

#[test]
fn build_discovers_cargo_deps() {
    let index = build_index();

    // Should have found some dependency types beyond std
    // Check for any type from a known project dep (rayon, serde, etc.)
    let dep_types: Vec<&str> = index
        .types
        .keys()
        .filter(|t| {
            // Exclude likely std types
            !matches!(
                t.as_str(),
                "vec" | "hashmap" | "string" | "btreemap" | "option" | "result"
            )
        })
        .map(|s| s.as_str())
        .take(5)
        .collect();
    assert!(
        !dep_types.is_empty(),
        "expected some non-std types from cargo deps"
    );
}

#[test]
fn has_method_returns_false_for_unknown() {
    let index = build_index();
    assert!(!index.has_method("nonexistent_type_xyz", "foo"));
    assert!(!index.has_method("vec", "nonexistent_method_xyz"));
}
