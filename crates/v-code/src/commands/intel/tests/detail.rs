use crate::commands::intel::detail::*;
use crate::commands::intel::reason::{self, HistoryItem, ReasonEntry, RejectedAlternative};

use std::path::Path;
use tempfile::TempDir;

fn make_entry(symbol: &str, decision: Option<&str>) -> ReasonEntry {
    ReasonEntry {
        symbol: symbol.to_owned(),
        decision: decision.map(|s| s.to_owned()),
        why: None,
        constraints: Vec::new(),
        rejected: Vec::new(),
        history: Vec::new(),
        file_path: None,
        line_range: None,
        related_symbols: Vec::new(),
    }
}

fn save(db: &Path, entry: &ReasonEntry) {
    reason::save_reason(db, entry).unwrap();
}

/// Helper to build `DetailParams` with sensible defaults.
fn params(db: &Path, symbol: &str) -> DetailParams {
    DetailParams {
        db: db.to_path_buf(),
        symbol: symbol.to_owned(),
        add: None,
        decision: None,
        why: None,
        constraint: None,
        rejected: None,
        failure: None,
        fix: None,
        root_cause: None,
        reject_reason: None,
        reject_condition: None,
        resolve: false,
        invalidate: None,
        show_all: false,
        delete: false,
        file_path: None,
        line_range: None,
        relate: None,
    }
}

// ── list tests ──────────────────────────────────────────────────

#[test]
fn list_empty_db() {
    let tmp = TempDir::new().unwrap();
    let entries = reason::list_reasons(tmp.path()).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn list_single_entry() {
    let tmp = TempDir::new().unwrap();
    let e = make_entry("Foo::bar", Some("use Arc"));
    save(tmp.path(), &e);

    let entries = reason::list_reasons(tmp.path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].symbol, "Foo::bar");
    assert_eq!(entries[0].decision.as_deref(), Some("use Arc"));
}

#[test]
fn list_multiple_entries_sorted() {
    let tmp = TempDir::new().unwrap();
    save(tmp.path(), &make_entry("Zebra::run", Some("fast path")));
    save(tmp.path(), &make_entry("Alpha::init", Some("lazy init")));
    save(tmp.path(), &make_entry("Mid::process", None));

    let entries = reason::list_reasons(tmp.path()).unwrap();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].symbol, "Alpha::init");
    assert_eq!(entries[1].symbol, "Mid::process");
    assert_eq!(entries[2].symbol, "Zebra::run");
}

#[test]
fn list_after_delete_excludes_deleted() {
    let tmp = TempDir::new().unwrap();
    save(tmp.path(), &make_entry("A", Some("keep")));
    save(tmp.path(), &make_entry("B", Some("remove")));

    reason::delete_reason(tmp.path(), "B").unwrap();

    let entries = reason::list_reasons(tmp.path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].symbol, "A");
}

// ── existing detail behaviour ───────────────────────────────────

#[test]
fn show_missing_symbol() {
    let tmp = TempDir::new().unwrap();
    let result = run_detail(params(tmp.path(), "NonExistent"));
    assert!(result.is_ok());
}

#[test]
fn mutate_creates_entry() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Foo");
    p.decision = Some("use HashMap".to_owned());
    p.why = Some("O(1) lookup".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Foo").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use HashMap"));
    assert_eq!(loaded.why.as_deref(), Some("O(1) lookup"));
}

#[test]
fn mutate_add_constraint() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Bar");
    p.constraint = Some("must be Send".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Bar").unwrap().unwrap();
    assert_eq!(loaded.constraints, vec!["must be Send"]);
}

#[test]
fn mutate_duplicate_constraint_not_added() {
    let tmp = TempDir::new().unwrap();
    let sym = "Dup";
    for _ in 0..2 {
        let mut p = params(tmp.path(), sym);
        p.constraint = Some("unique".to_owned());
        run_detail(p).unwrap();
    }
    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.constraints.len(), 1);
}

#[test]
fn delete_existing_entry() {
    let tmp = TempDir::new().unwrap();
    save(tmp.path(), &make_entry("Del", Some("bye")));

    let mut p = params(tmp.path(), "Del");
    p.delete = true;
    run_detail(p).unwrap();

    assert!(reason::load_reason(tmp.path(), "Del").unwrap().is_none());
}

#[test]
fn delete_missing_symbol_succeeds() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Ghost");
    p.delete = true;
    let result = run_detail(p);
    assert!(result.is_ok());
}

#[test]
fn list_preserves_full_entry_data() {
    let tmp = TempDir::new().unwrap();
    let mut e = make_entry("Rich", Some("complex"));
    e.why = Some("performance".to_owned());
    e.constraints = vec!["no alloc".to_owned()];
    e.rejected = vec![RejectedAlternative {
        approach: "BTreeMap".to_owned(),
        reason: None,
        condition: None,
    }];
    e.history = vec![HistoryItem {
        action: "create".to_owned(),
        date: "2026-01-01".to_owned(),
        note: Some("initial".to_owned()),
        failure: None,
        fix: None,
        root_cause: None,
        resolved: false,
        commit: None,
    }];
    save(tmp.path(), &e);

    let entries = reason::list_reasons(tmp.path()).unwrap();
    assert_eq!(entries.len(), 1);
    let got = &entries[0];
    assert_eq!(got.decision.as_deref(), Some("complex"));
    assert_eq!(got.why.as_deref(), Some("performance"));
    assert_eq!(got.constraints, vec!["no alloc"]);
    assert_eq!(got.rejected.len(), 1);
    assert_eq!(got.rejected[0].approach, "BTreeMap");
    assert_eq!(got.history.len(), 1);
    assert_eq!(got.history[0].action, "create");
}

// ── Mutation combination tests ──────────────────────────────────────

#[test]
fn mutate_decision_and_constraint_together() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Combo");
    p.decision = Some("use BTreeMap".to_owned());
    p.why = Some("sorted keys".to_owned());
    p.constraint = Some("must be deterministic".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Combo").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use BTreeMap"));
    assert_eq!(loaded.why.as_deref(), Some("sorted keys"));
    assert_eq!(loaded.constraints, vec!["must be deterministic"]);
    assert!(loaded.history.len() >= 2);
}

#[test]
fn mutate_all_flags_at_once() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Everything");
    p.add = Some("initial note".to_owned());
    p.decision = Some("use Arc".to_owned());
    p.why = Some("thread safety".to_owned());
    p.constraint = Some("must be Send+Sync".to_owned());
    p.rejected = Some("Rc".to_owned());
    p.failure = Some("deadlock under load".to_owned());
    p.fix = Some("use try_lock".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Everything").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use Arc"));
    assert_eq!(loaded.why.as_deref(), Some("thread safety"));
    assert_eq!(loaded.constraints, vec!["must be Send+Sync"]);
    assert_eq!(loaded.rejected.len(), 1);
    assert_eq!(loaded.rejected[0].approach, "Rc");

    let actions: Vec<&str> = loaded.history.iter().map(|h| h.action.as_str()).collect();
    assert!(actions.contains(&"note"), "should have note action");
    assert!(
        actions.contains(&"create") || actions.contains(&"decision"),
        "should have create or decision action"
    );
    assert!(actions.contains(&"constraint"), "should have constraint action");
    assert!(actions.contains(&"rejected"), "should have rejected action");
    assert!(actions.contains(&"modify"), "should have modify action for failure/fix");

    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert_eq!(modify.failure.as_deref(), Some("deadlock under load"));
    assert_eq!(modify.fix.as_deref(), Some("use try_lock"));
}

#[test]
fn mutate_duplicate_rejected_not_added() {
    let tmp = TempDir::new().unwrap();
    let sym = "DupRej";
    for _ in 0..2 {
        let mut p = params(tmp.path(), sym);
        p.rejected = Some("LinkedList".to_owned());
        run_detail(p).unwrap();
    }
    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.rejected.len(), 1, "duplicate rejected should not be added");
}

#[test]
fn mutate_add_note_only() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "NoteOnly");
    p.add = Some("just a note".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "NoteOnly").unwrap().unwrap();
    assert!(loaded.decision.is_none(), "no decision should be set");
    assert_eq!(loaded.history.len(), 1);
    assert_eq!(loaded.history[0].action, "note");
    assert_eq!(loaded.history[0].note.as_deref(), Some("just a note"));
}

#[test]
fn mutate_failure_only_without_fix() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "FailOnly");
    p.failure = Some("OOM on large input".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "FailOnly").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert_eq!(modify.failure.as_deref(), Some("OOM on large input"));
    assert!(modify.fix.is_none(), "fix should be None when not provided");
}

#[test]
fn mutate_fix_only_without_failure() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "FixOnly");
    p.fix = Some("added retry logic".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "FixOnly").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert!(modify.failure.is_none());
    assert_eq!(modify.fix.as_deref(), Some("added retry logic"));
}

#[test]
fn mutate_update_existing_decision() {
    let tmp = TempDir::new().unwrap();
    let mut p1 = params(tmp.path(), "Update");
    p1.decision = Some("use HashMap".to_owned());
    p1.why = Some("fast lookup".to_owned());
    run_detail(p1).unwrap();

    let mut p2 = params(tmp.path(), "Update");
    p2.decision = Some("use BTreeMap".to_owned());
    p2.why = Some("need sorted iteration".to_owned());
    run_detail(p2).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Update").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use BTreeMap"), "decision should be overwritten");
    assert_eq!(loaded.why.as_deref(), Some("need sorted iteration"), "why should be overwritten");
    assert!(loaded.history.len() >= 2);
}

#[test]
fn delete_takes_priority_over_mutations() {
    let tmp = TempDir::new().unwrap();
    save(tmp.path(), &make_entry("Priority", Some("initial")));

    let mut p = params(tmp.path(), "Priority");
    p.add = Some("this should be ignored".to_owned());
    p.decision = Some("also ignored".to_owned());
    p.delete = true;
    run_detail(p).unwrap();

    assert!(
        reason::load_reason(tmp.path(), "Priority").unwrap().is_none(),
        "delete should take priority over mutations"
    );
}

#[test]
fn multiple_constraints_accumulated() {
    let tmp = TempDir::new().unwrap();
    let sym = "MultiC";
    let constraints = ["must be Send", "must be Sync", "no unsafe"];
    for c in &constraints {
        let mut p = params(tmp.path(), sym);
        p.constraint = Some(c.to_string());
        run_detail(p).unwrap();
    }

    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.constraints.len(), 3);
    for c in &constraints {
        assert!(loaded.constraints.contains(&c.to_string()));
    }
}

// ── #2 Wrong Attribution: root_cause ────────────────────────────────

#[test]
fn failure_with_root_cause() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "Caller");
    p.failure = Some("timeout under load".to_owned());
    p.root_cause = Some("Callee::slow_fn".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Caller").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert_eq!(modify.failure.as_deref(), Some("timeout under load"));
    assert_eq!(modify.root_cause.as_deref(), Some("Callee::slow_fn"));
}

// ── #4 Resolved failures ───────────────────────────────────────────

#[test]
fn resolve_marks_last_failure() {
    let tmp = TempDir::new().unwrap();
    // Create entry with a failure
    let mut p = params(tmp.path(), "ResolveSym");
    p.failure = Some("crash on empty input".to_owned());
    run_detail(p).unwrap();

    // Resolve it
    let mut p2 = params(tmp.path(), "ResolveSym");
    p2.resolve = true;
    run_detail(p2).unwrap();

    let loaded = reason::load_reason(tmp.path(), "ResolveSym").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert!(modify.resolved, "failure should be resolved");
}

#[test]
fn resolve_no_failure_is_noop() {
    let tmp = TempDir::new().unwrap();
    // Create entry without failure
    let mut p = params(tmp.path(), "NoFail");
    p.add = Some("just a note".to_owned());
    run_detail(p).unwrap();

    // Try to resolve — should succeed but not crash
    let mut p2 = params(tmp.path(), "NoFail");
    p2.resolve = true;
    run_detail(p2).unwrap();

    let loaded = reason::load_reason(tmp.path(), "NoFail").unwrap().unwrap();
    // No resolved items since there was no failure
    assert!(loaded.history.iter().all(|h| !h.resolved));
}

// ── #7 Invalidated failures ────────────────────────────────────────

#[test]
fn invalidate_marks_and_tags_failure() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "InvSym");
    p.failure = Some("wrong diagnosis".to_owned());
    run_detail(p).unwrap();

    let mut p2 = params(tmp.path(), "InvSym");
    p2.invalidate = Some("was actually a test flake".to_owned());
    run_detail(p2).unwrap();

    let loaded = reason::load_reason(tmp.path(), "InvSym").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert!(modify.resolved, "invalidated should be resolved");
    assert!(
        modify.note.as_deref().unwrap().contains("[INVALIDATED: was actually a test flake]"),
        "note should contain invalidation tag"
    );
}

// ── #9 Structured rejected alternatives ────────────────────────────

#[test]
fn rejected_with_reason_and_condition() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "StructRej");
    p.rejected = Some("async_trait".to_owned());
    p.reject_reason = Some("dynamic dispatch overhead".to_owned());
    p.reject_condition = Some("hot path only".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "StructRej").unwrap().unwrap();
    assert_eq!(loaded.rejected.len(), 1);
    assert_eq!(loaded.rejected[0].approach, "async_trait");
    assert_eq!(loaded.rejected[0].reason.as_deref(), Some("dynamic dispatch overhead"));
    assert_eq!(loaded.rejected[0].condition.as_deref(), Some("hot path only"));
}

#[test]
fn legacy_plain_string_rejected_deserialization() {
    // Simulate legacy JSON with plain string rejected alternatives
    let tmp = TempDir::new().unwrap();
    let legacy_json = r#"{
        "symbol": "Legacy",
        "rejected": ["BTreeMap", "LinkedList"],
        "constraints": [],
        "history": []
    }"#;
    let dir = tmp.path().join("reasons");
    std::fs::create_dir_all(&dir).unwrap();

    // Compute hash the same way as reason_path
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    "Legacy".hash(&mut hasher);
    let hash = format!("{:x}", hasher.finish());
    std::fs::write(dir.join(format!("{hash}.json")), legacy_json).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Legacy").unwrap().unwrap();
    assert_eq!(loaded.rejected.len(), 2);
    assert_eq!(loaded.rejected[0].approach, "BTreeMap");
    assert!(loaded.rejected[0].reason.is_none());
    assert_eq!(loaded.rejected[1].approach, "LinkedList");
}

// ── #8 File locking (basic smoke test) ─────────────────────────────

#[test]
fn concurrent_saves_do_not_corrupt() {
    let tmp = TempDir::new().unwrap();
    let db = tmp.path().to_path_buf();

    // Sequential saves should work fine (basic lock acquire/release)
    for i in 0..5 {
        let mut entry = make_entry("ConcSym", Some("v1"));
        entry.constraints.push(format!("constraint-{i}"));
        reason::save_reason(&db, &entry).unwrap();
    }

    let loaded = reason::load_reason(&db, "ConcSym").unwrap().unwrap();
    assert_eq!(loaded.symbol, "ConcSym");
}

// ── serde compatibility: new fields default ────────────────────────

#[test]
fn new_history_fields_default_on_old_json() {
    let tmp = TempDir::new().unwrap();
    // Old-format JSON without root_cause/resolved
    let old_json = r#"{
        "symbol": "OldFmt",
        "history": [
            {"action": "modify", "date": "2026-01-01", "failure": "old bug"}
        ]
    }"#;
    let dir = tmp.path().join("reasons");
    std::fs::create_dir_all(&dir).unwrap();

    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    "OldFmt".hash(&mut hasher);
    let hash = format!("{:x}", hasher.finish());
    std::fs::write(dir.join(format!("{hash}.json")), old_json).unwrap();

    let loaded = reason::load_reason(tmp.path(), "OldFmt").unwrap().unwrap();
    assert_eq!(loaded.history.len(), 1);
    assert!(loaded.history[0].root_cause.is_none(), "root_cause should default to None");
    assert!(!loaded.history[0].resolved, "resolved should default to false");
    assert!(loaded.history[0].commit.is_none(), "commit should default to None");
    // New fields (#1, #6) should also default
    assert!(loaded.file_path.is_none(), "file_path should default to None");
    assert!(loaded.line_range.is_none(), "line_range should default to None");
    assert!(loaded.related_symbols.is_empty(), "related_symbols should default to empty");
}

// ── #1 Location tracking and rename resilience ─────────────────────

#[test]
fn set_file_path_and_line_range() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "LocSym");
    p.decision = Some("use Vec".to_owned());
    p.file_path = Some("src/lib.rs".to_owned());
    p.line_range = Some("10:25".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "LocSym").unwrap().unwrap();
    assert_eq!(loaded.file_path.as_deref(), Some("src/lib.rs"));
    assert_eq!(loaded.line_range, Some((10, 25)));
}

#[test]
fn fallback_load_by_location() {
    let tmp = TempDir::new().unwrap();

    // Create entry with location info under old symbol name
    let mut p = params(tmp.path(), "OldName::foo");
    p.decision = Some("important decision".to_owned());
    p.file_path = Some("src/mod.rs".to_owned());
    p.line_range = Some("42:50".to_owned());
    run_detail(p).unwrap();

    // Try to load with new symbol name — should fail normally
    let direct = reason::load_reason(tmp.path(), "NewName::foo").unwrap();
    assert!(direct.is_none(), "direct lookup should fail for renamed symbol");

    // Fallback by file:line should find it
    let fallback = reason::load_reason_with_fallback(
        tmp.path(), "NewName::foo", Some("src/mod.rs"), Some(42),
    ).unwrap();
    assert!(fallback.is_some(), "fallback should find entry by file:line");
    assert_eq!(fallback.unwrap().symbol, "OldName::foo");
}

// ── #5 Git commit in history ───────────────────────────────────────

#[test]
fn history_records_commit_hash() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "CommitSym");
    p.add = Some("test note".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "CommitSym").unwrap().unwrap();
    assert_eq!(loaded.history.len(), 1);
    // In a git repo, commit should be Some; in CI without git it may be None.
    // We just verify the field exists and is handled correctly.
    // Since we're running in a git repo, it should be Some.
    if let Some(ref commit) = loaded.history[0].commit {
        assert!(!commit.is_empty(), "commit hash should not be empty");
        // Short hash is typically 7-12 chars
        assert!(commit.len() <= 12, "should be a short hash");
    }
}

// ── #6 Related symbols ─────────────────────────────────────────────

#[test]
fn add_related_symbol() {
    let tmp = TempDir::new().unwrap();
    let mut p = params(tmp.path(), "SymA");
    p.decision = Some("shared state".to_owned());
    p.relate = Some("SymB".to_owned());
    run_detail(p).unwrap();

    let loaded = reason::load_reason(tmp.path(), "SymA").unwrap().unwrap();
    assert_eq!(loaded.related_symbols, vec!["SymB"]);
}

#[test]
fn related_symbol_bidirectional() {
    let tmp = TempDir::new().unwrap();

    // Create SymB first
    let mut pb = params(tmp.path(), "SymB");
    pb.decision = Some("owned by SymA".to_owned());
    run_detail(pb).unwrap();

    // Add SymB as related to SymA
    let mut pa = params(tmp.path(), "SymA");
    pa.decision = Some("uses SymB".to_owned());
    pa.relate = Some("SymB".to_owned());
    run_detail(pa).unwrap();

    // SymA should list SymB
    let a = reason::load_reason(tmp.path(), "SymA").unwrap().unwrap();
    assert!(a.related_symbols.contains(&"SymB".to_owned()));

    // SymB should also list SymA (bidirectional)
    let b = reason::load_reason(tmp.path(), "SymB").unwrap().unwrap();
    assert!(b.related_symbols.contains(&"SymA".to_owned()));
}

#[test]
fn duplicate_related_symbol_not_added() {
    let tmp = TempDir::new().unwrap();
    for _ in 0..2 {
        let mut p = params(tmp.path(), "DupRel");
        p.relate = Some("Other".to_owned());
        run_detail(p).unwrap();
    }
    let loaded = reason::load_reason(tmp.path(), "DupRel").unwrap().unwrap();
    assert_eq!(loaded.related_symbols.len(), 1);
}

#[test]
fn find_related_reasons_cross_reference() {
    let tmp = TempDir::new().unwrap();

    // Create entries that reference each other
    let mut e1 = make_entry("Alpha", Some("a-decision"));
    e1.related_symbols = vec!["Beta".to_owned()];
    save(tmp.path(), &e1);

    let mut e2 = make_entry("Beta", Some("b-decision"));
    e2.related_symbols = vec!["Alpha".to_owned()];
    save(tmp.path(), &e2);

    // find_related_reasons for "Alpha" should return Beta
    let refs = reason::find_related_reasons(tmp.path(), "Alpha").unwrap();
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].symbol, "Beta");
}
