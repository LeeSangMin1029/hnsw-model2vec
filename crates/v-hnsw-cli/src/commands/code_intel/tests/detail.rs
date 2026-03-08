use crate::commands::code_intel::detail::*;
use crate::commands::code_intel::reason::{self, HistoryItem, ReasonEntry};

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
    }
}

fn save(db: &Path, entry: &ReasonEntry) {
    reason::save_reason(db, entry).unwrap();
}

// ── list tests ──────────────────────────────────────────────────

#[test]
fn list_empty_db() {
    let tmp = TempDir::new().unwrap();
    let entries = run_list(tmp.path()).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn list_single_entry() {
    let tmp = TempDir::new().unwrap();
    let e = make_entry("Foo::bar", Some("use Arc"));
    save(tmp.path(), &e);

    let entries = run_list(tmp.path()).unwrap();
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

    let entries = run_list(tmp.path()).unwrap();
    assert_eq!(entries.len(), 3);
    // list_reasons sorts by symbol name
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

    let entries = run_list(tmp.path()).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].symbol, "A");
}

// ── existing detail behaviour ───────────────────────────────────

#[test]
fn show_missing_symbol() {
    let tmp = TempDir::new().unwrap();
    // run_detail with no mutation flags on missing symbol should succeed
    let result = run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "NonExistent".to_owned(),
        add: None, decision: None, why: None,
        constraint: None, rejected: None,
        failure: None, fix: None, delete: false,
    });
    assert!(result.is_ok());
}

#[test]
fn mutate_creates_entry() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Foo".to_owned(),
        add: None,
        decision: Some("use HashMap".to_owned()),
        why: Some("O(1) lookup".to_owned()),
        constraint: None, rejected: None,
        failure: None, fix: None, delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Foo").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use HashMap"));
    assert_eq!(loaded.why.as_deref(), Some("O(1) lookup"));
}

#[test]
fn mutate_add_constraint() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Bar".to_owned(),
        add: None, decision: None, why: None,
        constraint: Some("must be Send".to_owned()),
        rejected: None, failure: None, fix: None, delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Bar").unwrap().unwrap();
    assert_eq!(loaded.constraints, vec!["must be Send"]);
}

#[test]
fn mutate_duplicate_constraint_not_added() {
    let tmp = TempDir::new().unwrap();
    let sym = "Dup";
    for _ in 0..2 {
        run_detail(DetailParams {
            db: tmp.path().to_path_buf(),
            symbol: sym.to_owned(),
            add: None, decision: None, why: None,
            constraint: Some("unique".to_owned()),
            rejected: None, failure: None, fix: None, delete: false,
        }).unwrap();
    }
    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.constraints.len(), 1);
}

#[test]
fn delete_existing_entry() {
    let tmp = TempDir::new().unwrap();
    save(tmp.path(), &make_entry("Del", Some("bye")));

    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Del".to_owned(),
        add: None, decision: None, why: None,
        constraint: None, rejected: None,
        failure: None, fix: None, delete: true,
    }).unwrap();

    assert!(reason::load_reason(tmp.path(), "Del").unwrap().is_none());
}

#[test]
fn delete_missing_symbol_succeeds() {
    let tmp = TempDir::new().unwrap();
    let result = run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Ghost".to_owned(),
        add: None, decision: None, why: None,
        constraint: None, rejected: None,
        failure: None, fix: None, delete: true,
    });
    assert!(result.is_ok());
}

#[test]
fn list_preserves_full_entry_data() {
    let tmp = TempDir::new().unwrap();
    let mut e = make_entry("Rich", Some("complex"));
    e.why = Some("performance".to_owned());
    e.constraints = vec!["no alloc".to_owned()];
    e.rejected = vec!["BTreeMap".to_owned()];
    e.history = vec![HistoryItem {
        action: "create".to_owned(),
        date: "2026-01-01".to_owned(),
        note: Some("initial".to_owned()),
        failure: None,
        fix: None,
    }];
    save(tmp.path(), &e);

    let entries = run_list(tmp.path()).unwrap();
    assert_eq!(entries.len(), 1);
    let got = &entries[0];
    assert_eq!(got.decision.as_deref(), Some("complex"));
    assert_eq!(got.why.as_deref(), Some("performance"));
    assert_eq!(got.constraints, vec!["no alloc"]);
    assert_eq!(got.rejected, vec!["BTreeMap"]);
    assert_eq!(got.history.len(), 1);
    assert_eq!(got.history[0].action, "create");
}

// ── Mutation combination tests ──────────────────────────────────────

#[test]
fn mutate_decision_and_constraint_together() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Combo".to_owned(),
        add: None,
        decision: Some("use BTreeMap".to_owned()),
        why: Some("sorted keys".to_owned()),
        constraint: Some("must be deterministic".to_owned()),
        rejected: None,
        failure: None,
        fix: None,
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Combo").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use BTreeMap"));
    assert_eq!(loaded.why.as_deref(), Some("sorted keys"));
    assert_eq!(loaded.constraints, vec!["must be deterministic"]);
    // Should have at least 2 history entries: decision + constraint
    assert!(loaded.history.len() >= 2);
}

#[test]
fn mutate_all_flags_at_once() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Everything".to_owned(),
        add: Some("initial note".to_owned()),
        decision: Some("use Arc".to_owned()),
        why: Some("thread safety".to_owned()),
        constraint: Some("must be Send+Sync".to_owned()),
        rejected: Some("Rc".to_owned()),
        failure: Some("deadlock under load".to_owned()),
        fix: Some("use try_lock".to_owned()),
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Everything").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use Arc"));
    assert_eq!(loaded.why.as_deref(), Some("thread safety"));
    assert_eq!(loaded.constraints, vec!["must be Send+Sync"]);
    assert_eq!(loaded.rejected, vec!["Rc"]);

    // Check history has all actions
    let actions: Vec<&str> = loaded.history.iter().map(|h| h.action.as_str()).collect();
    assert!(actions.contains(&"note"), "should have note action");
    assert!(
        actions.contains(&"create") || actions.contains(&"decision"),
        "should have create or decision action"
    );
    assert!(actions.contains(&"constraint"), "should have constraint action");
    assert!(actions.contains(&"rejected"), "should have rejected action");
    assert!(actions.contains(&"modify"), "should have modify action for failure/fix");

    // Check failure/fix recorded in modify entry
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert_eq!(modify.failure.as_deref(), Some("deadlock under load"));
    assert_eq!(modify.fix.as_deref(), Some("use try_lock"));
}

#[test]
fn mutate_duplicate_rejected_not_added() {
    let tmp = TempDir::new().unwrap();
    let sym = "DupRej";
    for _ in 0..2 {
        run_detail(DetailParams {
            db: tmp.path().to_path_buf(),
            symbol: sym.to_owned(),
            add: None,
            decision: None,
            why: None,
            constraint: None,
            rejected: Some("LinkedList".to_owned()),
            failure: None,
            fix: None,
            delete: false,
        }).unwrap();
    }
    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.rejected.len(), 1, "duplicate rejected should not be added");
}

#[test]
fn mutate_add_note_only() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "NoteOnly".to_owned(),
        add: Some("just a note".to_owned()),
        decision: None,
        why: None,
        constraint: None,
        rejected: None,
        failure: None,
        fix: None,
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "NoteOnly").unwrap().unwrap();
    assert!(loaded.decision.is_none(), "no decision should be set");
    assert_eq!(loaded.history.len(), 1);
    assert_eq!(loaded.history[0].action, "note");
    assert_eq!(loaded.history[0].note.as_deref(), Some("just a note"));
}

#[test]
fn mutate_failure_only_without_fix() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "FailOnly".to_owned(),
        add: None,
        decision: None,
        why: None,
        constraint: None,
        rejected: None,
        failure: Some("OOM on large input".to_owned()),
        fix: None,
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "FailOnly").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert_eq!(modify.failure.as_deref(), Some("OOM on large input"));
    assert!(modify.fix.is_none(), "fix should be None when not provided");
}

#[test]
fn mutate_fix_only_without_failure() {
    let tmp = TempDir::new().unwrap();
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "FixOnly".to_owned(),
        add: None,
        decision: None,
        why: None,
        constraint: None,
        rejected: None,
        failure: None,
        fix: Some("added retry logic".to_owned()),
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "FixOnly").unwrap().unwrap();
    let modify = loaded.history.iter().find(|h| h.action == "modify").unwrap();
    assert!(modify.failure.is_none());
    assert_eq!(modify.fix.as_deref(), Some("added retry logic"));
}

#[test]
fn mutate_update_existing_decision() {
    let tmp = TempDir::new().unwrap();
    // Create initial entry
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Update".to_owned(),
        add: None,
        decision: Some("use HashMap".to_owned()),
        why: Some("fast lookup".to_owned()),
        constraint: None,
        rejected: None,
        failure: None,
        fix: None,
        delete: false,
    }).unwrap();

    // Update decision
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Update".to_owned(),
        add: None,
        decision: Some("use BTreeMap".to_owned()),
        why: Some("need sorted iteration".to_owned()),
        constraint: None,
        rejected: None,
        failure: None,
        fix: None,
        delete: false,
    }).unwrap();

    let loaded = reason::load_reason(tmp.path(), "Update").unwrap().unwrap();
    assert_eq!(loaded.decision.as_deref(), Some("use BTreeMap"), "decision should be overwritten");
    assert_eq!(loaded.why.as_deref(), Some("need sorted iteration"), "why should be overwritten");
    // Should have at least 2 history entries
    assert!(loaded.history.len() >= 2);
}

#[test]
fn delete_takes_priority_over_mutations() {
    let tmp = TempDir::new().unwrap();
    // Create an entry first
    save(tmp.path(), &make_entry("Priority", Some("initial")));

    // Delete flag + mutation flags — delete should win
    run_detail(DetailParams {
        db: tmp.path().to_path_buf(),
        symbol: "Priority".to_owned(),
        add: Some("this should be ignored".to_owned()),
        decision: Some("also ignored".to_owned()),
        why: None,
        constraint: None,
        rejected: None,
        failure: None,
        fix: None,
        delete: true,
    }).unwrap();

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
        run_detail(DetailParams {
            db: tmp.path().to_path_buf(),
            symbol: sym.to_owned(),
            add: None,
            decision: None,
            why: None,
            constraint: Some(c.to_string()),
            rejected: None,
            failure: None,
            fix: None,
            delete: false,
        }).unwrap();
    }

    let loaded = reason::load_reason(tmp.path(), sym).unwrap().unwrap();
    assert_eq!(loaded.constraints.len(), 3);
    for c in &constraints {
        assert!(loaded.constraints.contains(&c.to_string()));
    }
}
