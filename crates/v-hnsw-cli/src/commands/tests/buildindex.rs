use std::collections::HashMap;
use crate::commands::buildindex::sorted_ids_by_slot;

#[test]
fn sorted_ids_by_slot_empty() {
    let map = HashMap::new();
    let result = sorted_ids_by_slot(&map);
    assert!(result.is_empty());
}

#[test]
fn sorted_ids_by_slot_single() {
    let mut map = HashMap::new();
    map.insert(42u64, 0u32);
    let result = sorted_ids_by_slot(&map);
    assert_eq!(result, vec![42]);
}

#[test]
fn sorted_ids_by_slot_ordering() {
    let mut map = HashMap::new();
    // IDs with slots in reverse order
    map.insert(100, 3);
    map.insert(200, 1);
    map.insert(300, 0);
    map.insert(400, 2);
    let result = sorted_ids_by_slot(&map);
    // Should be sorted by slot: 300(slot=0), 200(slot=1), 400(slot=2), 100(slot=3)
    assert_eq!(result, vec![300, 200, 400, 100]);
}

#[test]
fn sorted_ids_by_slot_sequential() {
    let mut map = HashMap::new();
    map.insert(1, 0);
    map.insert(2, 1);
    map.insert(3, 2);
    let result = sorted_ids_by_slot(&map);
    assert_eq!(result, vec![1, 2, 3]);
}
