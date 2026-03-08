use crate::{data_dir, home_dir, ko_dic_dir, read_le_u64, storage_err, VhnswError};

// ---------------------------------------------------------------------------
// read_le_u64
// ---------------------------------------------------------------------------

#[test]
fn test_read_le_u64_valid() {
    let value: u64 = 0x0102_0304_0506_0708;
    let bytes = value.to_le_bytes();
    assert_eq!(read_le_u64(&bytes, 0), Some(value));
}

#[test]
fn test_read_le_u64_with_offset() {
    let mut data = vec![0xAA, 0xBB]; // 2-byte prefix
    let value: u64 = 42;
    data.extend_from_slice(&value.to_le_bytes());
    assert_eq!(read_le_u64(&data, 2), Some(42));
}

#[test]
fn test_read_le_u64_too_short() {
    let data = [0u8; 7];
    assert_eq!(read_le_u64(&data, 0), None);
}

#[test]
fn test_read_le_u64_offset_out_of_bounds() {
    let data = [0u8; 8];
    assert_eq!(read_le_u64(&data, 1), None); // only 7 bytes left
}

#[test]
fn test_read_le_u64_empty() {
    assert_eq!(read_le_u64(&[], 0), None);
}

#[test]
fn test_read_le_u64_zero_value() {
    let bytes = 0u64.to_le_bytes();
    assert_eq!(read_le_u64(&bytes, 0), Some(0));
}

#[test]
fn test_read_le_u64_max_value() {
    let bytes = u64::MAX.to_le_bytes();
    assert_eq!(read_le_u64(&bytes, 0), Some(u64::MAX));
}

#[test]
fn test_read_le_u64_exact_boundary() {
    // Exactly 8 bytes: offset 0 should work, offset 1 should fail.
    let data = [0xFF; 8];
    assert!(read_le_u64(&data, 0).is_some());
    assert!(read_le_u64(&data, 1).is_none());
}

#[test]
fn test_read_le_u64_multiple_values() {
    // Two u64 values back to back
    let mut data = Vec::new();
    data.extend_from_slice(&100u64.to_le_bytes());
    data.extend_from_slice(&200u64.to_le_bytes());
    assert_eq!(read_le_u64(&data, 0), Some(100));
    assert_eq!(read_le_u64(&data, 8), Some(200));
}

// ---------------------------------------------------------------------------
// home_dir / data_dir / ko_dic_dir
// ---------------------------------------------------------------------------

#[test]
fn test_home_dir_returns_some() {
    assert!(home_dir().is_some());
}

#[test]
fn test_data_dir_ends_with_v_hnsw() {
    let dir = data_dir();
    assert!(dir.ends_with(".v-hnsw"));
}

#[test]
fn test_ko_dic_dir_ends_with_ko_dic() {
    let dir = ko_dic_dir();
    assert!(dir.ends_with("ko-dic"));
    assert!(dir.to_string_lossy().contains(".v-hnsw"));
}

#[test]
fn test_data_dir_is_child_of_home() {
    if let Some(home) = home_dir() {
        let data = data_dir();
        assert!(data.starts_with(&home));
    }
}

#[test]
fn test_ko_dic_dir_is_child_of_data_dir() {
    let data = data_dir();
    let ko = ko_dic_dir();
    assert!(ko.starts_with(&data));
}

// ---------------------------------------------------------------------------
// storage_err helper
// ---------------------------------------------------------------------------

#[test]
fn test_storage_err() {
    let err = storage_err("disk full");
    match err {
        VhnswError::Storage(io_err) => {
            assert_eq!(io_err.to_string(), "disk full");
        }
        other => panic!("expected Storage variant, got: {other}"),
    }
}

#[test]
fn test_storage_err_empty_message() {
    let err = storage_err("");
    match err {
        VhnswError::Storage(io_err) => {
            assert_eq!(io_err.to_string(), "");
        }
        other => panic!("expected Storage variant, got: {other}"),
    }
}
