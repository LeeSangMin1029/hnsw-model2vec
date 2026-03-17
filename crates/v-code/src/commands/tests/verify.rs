//! Unit tests for `verify` module — short_name, categorize_miss.

use super::{categorize_miss, short_name};

// ── short_name ───────────────────────────────────────────────────────

#[test]
fn short_name_qualified() {
    assert_eq!(short_name("Foo::bar"), "bar");
}

#[test]
fn short_name_deeply_qualified() {
    assert_eq!(short_name("std::collections::HashMap::new"), "new");
}

#[test]
fn short_name_unqualified() {
    assert_eq!(short_name("baz"), "baz");
}

#[test]
fn short_name_empty() {
    assert_eq!(short_name(""), "");
}

// ── categorize_miss ──────────────────────────────────────────────────

#[test]
fn categorize_self_method() {
    assert_eq!(categorize_miss("self.process"), "self.method");
}

#[test]
fn categorize_self_field_method() {
    assert_eq!(categorize_miss("self.engine.start"), "self.field.method");
}

#[test]
fn categorize_qualified() {
    assert_eq!(categorize_miss("Config::load"), "Type::method (qualified)");
}

#[test]
fn categorize_receiver_method() {
    assert_eq!(categorize_miss("db.query"), "receiver.method");
}

#[test]
fn categorize_bare_function() {
    assert_eq!(categorize_miss("process"), "bare function");
}
