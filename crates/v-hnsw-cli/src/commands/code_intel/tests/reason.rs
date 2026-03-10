use crate::commands::code_intel::reason::{
    self, HistoryItem, ReasonEntry, RejectedAlternative,
};

fn make_entry(symbol: &str) -> ReasonEntry {
    ReasonEntry {
        symbol: symbol.to_owned(),
        decision: Some("use BTreeMap".to_owned()),
        why: Some("ordered iteration needed".to_owned()),
        constraints: vec!["must be deterministic".to_owned()],
        rejected: vec![RejectedAlternative {
            approach: "HashMap — non-deterministic".to_owned(),
            reason: None,
            condition: None,
        }],
        history: vec![HistoryItem {
            action: "create".to_owned(),
            date: "2025-01-01".to_owned(),
            note: Some("initial decision".to_owned()),
            failure: None,
            fix: None,
            root_cause: None,
            resolved: false,
            commit: None,
        }],
        file_path: None,
        line_range: None,
        related_symbols: Vec::new(),
    }
}

// ── one_line_summary ─────────────────────────────────────────────────

#[test]
fn summary_with_decision_and_why() {
    let entry = make_entry("Foo::bar");
    let s = reason::one_line_summary(&entry);
    assert_eq!(s, "use BTreeMap -- ordered iteration needed");
}

#[test]
fn summary_decision_only() {
    let mut entry = make_entry("Foo::bar");
    entry.why = None;
    let s = reason::one_line_summary(&entry);
    assert_eq!(s, "use BTreeMap");
}

#[test]
fn summary_why_only() {
    let mut entry = make_entry("Foo::bar");
    entry.decision = None;
    let s = reason::one_line_summary(&entry);
    assert_eq!(s, "ordered iteration needed");
}

#[test]
fn summary_constraints_only() {
    let mut entry = make_entry("Foo::bar");
    entry.decision = None;
    entry.why = None;
    let s = reason::one_line_summary(&entry);
    assert_eq!(s, "constraints: must be deterministic");
}

#[test]
fn summary_empty_fallback() {
    let entry = ReasonEntry {
        symbol: "X".to_owned(),
        decision: None,
        why: None,
        constraints: vec![],
        rejected: vec![],
        history: vec![],
        file_path: None,
        line_range: None,
        related_symbols: Vec::new(),
    };
    assert_eq!(reason::one_line_summary(&entry), "reason recorded");
}

// ── save / load / delete / list (tempdir) ────────────────────────────

#[test]
fn save_and_load_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path();
    let entry = make_entry("MyStruct::do_thing");

    reason::save_reason(db, &entry).unwrap();

    let loaded = reason::load_reason(db, "MyStruct::do_thing")
        .unwrap()
        .expect("entry should exist");

    assert_eq!(loaded.symbol, "MyStruct::do_thing");
    assert_eq!(loaded.decision.as_deref(), Some("use BTreeMap"));
    assert_eq!(loaded.constraints.len(), 1);
    assert_eq!(loaded.history.len(), 1);
}

#[test]
fn load_missing_returns_none() {
    let dir = tempfile::tempdir().unwrap();
    let result = reason::load_reason(dir.path(), "nonexistent").unwrap();
    assert!(result.is_none());
}

#[test]
fn delete_existing_returns_true() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path();
    let entry = make_entry("A::b");
    reason::save_reason(db, &entry).unwrap();

    assert!(reason::delete_reason(db, "A::b").unwrap());
    assert!(reason::load_reason(db, "A::b").unwrap().is_none());
}

#[test]
fn delete_missing_returns_false() {
    let dir = tempfile::tempdir().unwrap();
    assert!(!reason::delete_reason(dir.path(), "nope").unwrap());
}

#[test]
fn list_reasons_empty_dir() {
    let dir = tempfile::tempdir().unwrap();
    let list = reason::list_reasons(dir.path()).unwrap();
    assert!(list.is_empty());
}

#[test]
fn list_reasons_returns_sorted() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path();

    let mut e1 = make_entry("Zebra::run");
    e1.decision = Some("z-decision".to_owned());
    let mut e2 = make_entry("Alpha::go");
    e2.decision = Some("a-decision".to_owned());

    reason::save_reason(db, &e1).unwrap();
    reason::save_reason(db, &e2).unwrap();

    let list = reason::list_reasons(db).unwrap();
    assert_eq!(list.len(), 2);
    assert_eq!(list[0].symbol, "Alpha::go");
    assert_eq!(list[1].symbol, "Zebra::run");
}

// ── serde round-trip (missing optional fields) ───────────────────────

#[test]
fn serde_missing_optional_fields() {
    let json = r#"{"symbol":"X"}"#;
    let entry: ReasonEntry = serde_json::from_str(json).unwrap();
    assert_eq!(entry.symbol, "X");
    assert!(entry.decision.is_none());
    assert!(entry.why.is_none());
    assert!(entry.constraints.is_empty());
    assert!(entry.rejected.is_empty());
    assert!(entry.history.is_empty());
}

// ── today() returns valid format ─────────────────────────────────────

#[test]
fn today_returns_valid_date_format() {
    let t = reason::today();
    // Should match YYYY-MM-DD
    assert_eq!(t.len(), 10);
    assert_eq!(&t[4..5], "-");
    assert_eq!(&t[7..8], "-");
    let year: u32 = t[..4].parse().unwrap();
    let month: u32 = t[5..7].parse().unwrap();
    let day: u32 = t[8..10].parse().unwrap();
    assert!(year >= 2024);
    assert!((1..=12).contains(&month));
    assert!((1..=31).contains(&day));
}
