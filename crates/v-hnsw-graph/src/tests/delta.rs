use v_hnsw_core::PointId;
use crate::delta::{DeltaNeighbors, encode_varint, decode_varint};

#[test]
fn test_empty() {
    let dn = DeltaNeighbors::new();
    assert!(dn.is_empty());
    assert_eq!(dn.len(), 0);
    assert_eq!(dn.decode(), Vec::<PointId>::new());
}

#[test]
fn test_single() {
    let dn = DeltaNeighbors::from_ids(&[42]);
    assert_eq!(dn.len(), 1);
    assert!(!dn.is_empty());
    assert_eq!(dn.decode(), vec![42]);
}

#[test]
fn test_multiple_sorted() {
    let ids = vec![10, 20, 30, 40, 50];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 5);
    assert_eq!(dn.decode(), ids);
}

#[test]
fn test_multiple_unsorted() {
    let ids = vec![50, 10, 30, 20, 40];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 5);
    assert_eq!(dn.decode(), vec![10, 20, 30, 40, 50]);
}

#[test]
fn test_duplicates() {
    let ids = vec![10, 20, 10, 30, 20];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 3);
    assert_eq!(dn.decode(), vec![10, 20, 30]);
}

#[test]
fn test_push() {
    let mut dn = DeltaNeighbors::from_ids(&[10, 30]);
    dn.push(20);
    assert_eq!(dn.len(), 3);
    assert_eq!(dn.decode(), vec![10, 20, 30]);
}

#[test]
fn test_contains() {
    let dn = DeltaNeighbors::from_ids(&[10, 20, 30, 40, 50]);
    assert!(dn.contains(10));
    assert!(dn.contains(30));
    assert!(dn.contains(50));
    assert!(!dn.contains(15));
    assert!(!dn.contains(0));
    assert!(!dn.contains(60));
}

#[test]
fn test_contains_empty() {
    let dn = DeltaNeighbors::new();
    assert!(!dn.contains(1));
}

#[test]
fn test_large_ids() {
    let ids = vec![
        1_000_000_000,
        2_000_000_000,
        u64::MAX - 1000,
        u64::MAX - 500,
        u64::MAX,
    ];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 5);
    assert_eq!(dn.decode(), ids);
}

#[test]
fn test_memory_savings() {
    // Sequential IDs: deltas are 1, which encode as a single byte each.
    let ids: Vec<PointId> = (0..100).collect();
    let dn = DeltaNeighbors::from_ids(&ids);
    let raw_size = 8 * ids.len(); // 800 bytes for Vec<u64>
    let encoded_size = dn.memory_bytes();
    assert!(
        encoded_size < raw_size,
        "encoded {encoded_size} should be < raw {raw_size}"
    );
}

#[test]
fn test_push_duplicate() {
    let mut dn = DeltaNeighbors::from_ids(&[10, 20, 30]);
    dn.push(20); // duplicate — should be deduplicated
    assert_eq!(dn.len(), 3);
    assert_eq!(dn.decode(), vec![10, 20, 30]);
}

#[test]
fn test_varint_roundtrip() {
    // Test various varint sizes (1-byte, 2-byte, multi-byte).
    let values: Vec<u64> = vec![0, 1, 127, 128, 16383, 16384, u64::MAX];
    for &v in &values {
        let mut buf = Vec::new();
        encode_varint(&mut buf, v);
        let mut pos = 0;
        let decoded = decode_varint(&buf, &mut pos);
        assert_eq!(decoded, v, "varint roundtrip failed for {v}");
        assert_eq!(pos, buf.len(), "not all bytes consumed for {v}");
    }
}

#[test]
fn test_from_ids_empty_slice() {
    let dn = DeltaNeighbors::from_ids(&[]);
    assert!(dn.is_empty());
    assert_eq!(dn.len(), 0);
    assert_eq!(dn.decode(), Vec::<PointId>::new());
}

#[test]
fn test_from_ids_zero() {
    let dn = DeltaNeighbors::from_ids(&[0]);
    assert_eq!(dn.len(), 1);
    assert_eq!(dn.decode(), vec![0]);
    assert!(dn.contains(0));
}

#[test]
fn test_push_to_empty() {
    let mut dn = DeltaNeighbors::new();
    dn.push(42);
    assert_eq!(dn.len(), 1);
    assert_eq!(dn.decode(), vec![42]);
    assert!(dn.contains(42));
}

#[test]
fn test_consecutive_ids() {
    let ids: Vec<PointId> = (0..50).collect();
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 50);
    assert_eq!(dn.decode(), ids);
    // Consecutive IDs should compress very well (deltas are all 1)
    assert!(dn.memory_bytes() < 8 * 50);
}

#[test]
fn test_contains_boundary_first_and_last() {
    let dn = DeltaNeighbors::from_ids(&[5, 10, 15]);
    assert!(dn.contains(5));   // first
    assert!(dn.contains(15));  // last
    assert!(!dn.contains(4));  // before first
    assert!(!dn.contains(16)); // after last
    assert!(!dn.contains(7));  // between first and second
}

#[test]
fn test_all_same_ids() {
    let ids = vec![42, 42, 42, 42, 42];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 1); // deduplicated
    assert_eq!(dn.decode(), vec![42]);
}

#[test]
fn test_two_adjacent_ids() {
    let dn = DeltaNeighbors::from_ids(&[100, 101]);
    assert_eq!(dn.len(), 2);
    assert_eq!(dn.decode(), vec![100, 101]);
    assert!(dn.contains(100));
    assert!(dn.contains(101));
    assert!(!dn.contains(99));
    assert!(!dn.contains(102));
}

#[test]
fn test_varint_one_byte_max() {
    // 127 is the max value for a single-byte varint
    let mut buf = Vec::new();
    encode_varint(&mut buf, 127);
    assert_eq!(buf.len(), 1);
    let mut pos = 0;
    assert_eq!(decode_varint(&buf, &mut pos), 127);
}

#[test]
fn test_varint_two_byte_min() {
    // 128 requires 2 bytes in LEB128
    let mut buf = Vec::new();
    encode_varint(&mut buf, 128);
    assert_eq!(buf.len(), 2);
    let mut pos = 0;
    assert_eq!(decode_varint(&buf, &mut pos), 128);
}

#[test]
fn test_varint_zero() {
    let mut buf = Vec::new();
    encode_varint(&mut buf, 0);
    assert_eq!(buf.len(), 1);
    let mut pos = 0;
    assert_eq!(decode_varint(&buf, &mut pos), 0);
}

#[test]
fn test_decode_varint_out_of_bounds() {
    let data: Vec<u8> = vec![];
    let mut pos = 0;
    let result = decode_varint(&data, &mut pos);
    assert_eq!(result, 0); // returns 0 when out of bounds
}

#[test]
fn test_push_multiple_maintains_sorted_order() {
    let mut dn = DeltaNeighbors::from_ids(&[50]);
    dn.push(10);
    dn.push(30);
    dn.push(70);
    assert_eq!(dn.decode(), vec![10, 30, 50, 70]);
}

#[test]
fn test_large_gap_between_ids() {
    // Test with very large gaps between IDs
    let ids = vec![1, 1_000_000, 2_000_000_000];
    let dn = DeltaNeighbors::from_ids(&ids);
    assert_eq!(dn.len(), 3);
    assert_eq!(dn.decode(), ids);
    assert!(dn.contains(1));
    assert!(dn.contains(1_000_000));
    assert!(dn.contains(2_000_000_000));
    assert!(!dn.contains(2));
}
