use crate::rustdoc::RustdocTypes;

/// Minimal rustdoc JSON fixture for testing.
fn minimal_json() -> String {
    r#"{
  "root": 0,
  "crate_version": "0.1.0",
  "includes_private": true,
  "index": {
    "1": {
      "id": 1, "crate_id": 0, "name": "MyStruct",
      "inner": {
        "struct": {
          "kind": { "plain": { "fields": [10, 11], "has_stripped_fields": false } },
          "generics": { "params": [], "where_predicates": [] },
          "impls": [100]
        }
      }
    },
    "10": {
      "id": 10, "crate_id": 0, "name": "name",
      "inner": { "struct_field": { "primitive": "str" } }
    },
    "11": {
      "id": 11, "crate_id": 0, "name": "db",
      "inner": { "struct_field": { "resolved_path": { "path": "crate::Database", "id": 2, "args": null } } }
    },
    "2": {
      "id": 2, "crate_id": 0, "name": "Database",
      "inner": {
        "struct": {
          "kind": { "plain": { "fields": [], "has_stripped_fields": false } },
          "generics": { "params": [], "where_predicates": [] },
          "impls": [101]
        }
      }
    },
    "100": {
      "id": 100, "crate_id": 0,
      "inner": {
        "impl": {
          "is_unsafe": false, "generics": { "params": [], "where_predicates": [] },
          "provided_trait_methods": [], "trait": null, "blanket_impl": null,
          "for": { "resolved_path": { "path": "crate::MyStruct", "id": 1, "args": null } },
          "items": [200, 201]
        }
      }
    },
    "101": {
      "id": 101, "crate_id": 0,
      "inner": {
        "impl": {
          "is_unsafe": false, "generics": { "params": [], "where_predicates": [] },
          "provided_trait_methods": [], "trait": null, "blanket_impl": null,
          "for": { "resolved_path": { "path": "crate::Database", "id": 2, "args": null } },
          "items": [202]
        }
      }
    },
    "200": {
      "id": 200, "crate_id": 0, "name": "new",
      "inner": {
        "function": {
          "sig": {
            "inputs": [],
            "output": { "generic": "Self" }
          },
          "generics": { "params": [], "where_predicates": [] },
          "has_body": true
        }
      }
    },
    "201": {
      "id": 201, "crate_id": 0, "name": "connect",
      "inner": {
        "function": {
          "sig": {
            "inputs": [["self", { "generic": "Self" }]],
            "output": { "resolved_path": { "path": "crate::Database", "id": 2, "args": null } }
          },
          "generics": { "params": [], "where_predicates": [] },
          "has_body": true
        }
      }
    },
    "202": {
      "id": 202, "crate_id": 0, "name": "query",
      "inner": {
        "function": {
          "sig": {
            "inputs": [
              ["self", { "generic": "Self" }],
              ["sql", { "borrowed_ref": { "lifetime": null, "is_mutable": false, "type": { "primitive": "str" } } }]
            ],
            "output": { "resolved_path": { "path": "Vec", "id": 999, "args": null } }
          },
          "generics": { "params": [], "where_predicates": [] },
          "has_body": true
        }
      }
    },
    "300": {
      "id": 300, "crate_id": 0, "name": "free_fn",
      "inner": {
        "function": {
          "sig": {
            "inputs": [],
            "output": { "resolved_path": { "path": "std::path::PathBuf", "id": 500, "args": null } }
          },
          "generics": { "params": [], "where_predicates": [] },
          "has_body": true
        }
      }
    }
  },
  "paths": {},
  "external_crates": {}
}"#
    .to_owned()
}

#[test]
fn parse_fn_return_types() {
    let types = RustdocTypes::from_str(&minimal_json()).unwrap();

    // MyStruct::new returns Self → resolved to "mystruct"
    assert_eq!(types.fn_return_types.get("mystruct::new").map(String::as_str), Some("mystruct"));

    // MyStruct::connect returns Database
    assert_eq!(types.fn_return_types.get("mystruct::connect").map(String::as_str), Some("database"));

    // Database::query returns Vec
    assert_eq!(types.fn_return_types.get("database::query").map(String::as_str), Some("vec"));

    // free_fn returns PathBuf
    assert_eq!(types.fn_return_types.get("free_fn").map(String::as_str), Some("pathbuf"));
}

#[test]
fn parse_method_owners() {
    let types = RustdocTypes::from_str(&minimal_json()).unwrap();

    // "new" is owned by MyStruct
    assert!(types.method_owner.get("new").unwrap().contains(&"mystruct".to_owned()));

    // "connect" is owned by MyStruct
    assert!(types.method_owner.get("connect").unwrap().contains(&"mystruct".to_owned()));

    // "query" is owned by Database
    assert!(types.method_owner.get("query").unwrap().contains(&"database".to_owned()));
}

#[test]
fn parse_struct_fields() {
    let types = RustdocTypes::from_str(&minimal_json()).unwrap();

    assert_eq!(types.field_type("mystruct", "name"), Some("str"));
    assert_eq!(types.field_type("mystruct", "db"), Some("database"));
    assert_eq!(types.field_type("mystruct", "nonexistent"), None);
}

#[test]
fn blanket_traits_excluded() {
    let types = RustdocTypes::from_str(&minimal_json()).unwrap();

    // "from", "clone", etc. should not appear in method_owner
    assert!(types.method_owner.get("from").is_none());
    assert!(types.method_owner.get("clone").is_none());
}

#[test]
fn return_type_self_resolution() {
    let types = RustdocTypes::from_str(&minimal_json()).unwrap();

    // new() returns Self → should be resolved to owner type "mystruct"
    let ret = types.return_type("mystruct::new");
    assert_eq!(ret, Some("mystruct"));
}
