use crate::commands::update::UpdateStats;

// ── UpdateStats serialization ────────────────────────────────────

#[test]
fn update_stats_default() {
    let stats = UpdateStats::default();
    assert_eq!(stats.new, 0);
    assert_eq!(stats.modified, 0);
    assert_eq!(stats.deleted, 0);
    assert_eq!(stats.unchanged, 0);
    assert_eq!(stats.hash_skipped, 0);
}

#[test]
fn update_stats_roundtrip() {
    let stats = UpdateStats {
        new: 3,
        modified: 2,
        deleted: 1,
        unchanged: 10,
        hash_skipped: 5,
    };
    let json = serde_json::to_string(&stats).unwrap();
    let deserialized: UpdateStats = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.new, 3);
    assert_eq!(deserialized.modified, 2);
    assert_eq!(deserialized.deleted, 1);
    assert_eq!(deserialized.unchanged, 10);
    assert_eq!(deserialized.hash_skipped, 5);
}
