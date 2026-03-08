//! Tests for the fvecs/bvecs binary format reader.

use std::io::Write;

use crate::commands::readers::fvecs::FvecsReader;
use crate::commands::readers::VectorReader;

/// Write a single fvecs record: [dim: u32_le][dim x f32_le].
fn write_fvecs_record(buf: &mut Vec<u8>, values: &[f32]) {
    let dim = values.len() as u32;
    buf.extend_from_slice(&dim.to_le_bytes());
    for v in values {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

/// Write a single bvecs record: [dim: u32_le][dim x u8].
fn write_bvecs_record(buf: &mut Vec<u8>, values: &[u8]) {
    let dim = values.len() as u32;
    buf.extend_from_slice(&dim.to_le_bytes());
    buf.extend_from_slice(values);
}

// ---------------------------------------------------------------------------
// FvecsReader::open
// ---------------------------------------------------------------------------

#[test]
fn open_fvecs_single_record() {
    let mut data = Vec::new();
    write_fvecs_record(&mut data, &[1.0, 2.0, 3.0]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    assert_eq!(reader.count().unwrap(), 1);
}

#[test]
fn open_fvecs_multiple_records() {
    let mut data = Vec::new();
    write_fvecs_record(&mut data, &[1.0, 2.0]);
    write_fvecs_record(&mut data, &[3.0, 4.0]);
    write_fvecs_record(&mut data, &[5.0, 6.0]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    assert_eq!(reader.count().unwrap(), 3);
}

#[test]
fn open_bvecs_single_record() {
    let mut data = Vec::new();
    write_bvecs_record(&mut data, &[10, 20, 30, 40]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".bvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    assert_eq!(reader.count().unwrap(), 1);
}

#[test]
fn open_rejects_unsupported_extension() {
    let tmp = tempfile::Builder::new()
        .suffix(".csv")
        .tempfile()
        .unwrap();
    let result = FvecsReader::open(tmp.path());
    assert!(result.is_err());
    let msg = format!("{}", result.err().unwrap());
    assert!(msg.contains("csv"), "error should mention the bad extension: {msg}");
}

#[test]
fn open_rejects_empty_file() {
    let tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    // File is 0 bytes
    let result = FvecsReader::open(tmp.path());
    assert!(result.is_err());
}

#[test]
fn open_rejects_truncated_file() {
    // File has dim=3 but only 2 floats (12 bytes payload instead of 12)
    let mut data = Vec::new();
    data.extend_from_slice(&3u32.to_le_bytes()); // dim = 3
    data.extend_from_slice(&1.0f32.to_le_bytes());
    data.extend_from_slice(&2.0f32.to_le_bytes());
    // Missing third float => file_len not multiple of record_bytes

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let result = FvecsReader::open(tmp.path());
    assert!(result.is_err(), "truncated file should be rejected");
}

// ---------------------------------------------------------------------------
// FvecsReader::records iteration
// ---------------------------------------------------------------------------

#[test]
fn fvecs_records_returns_correct_vectors() {
    let mut data = Vec::new();
    write_fvecs_record(&mut data, &[1.5, 2.5, 3.5]);
    write_fvecs_record(&mut data, &[4.0, 5.0, 6.0]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 2);
    assert_eq!(records[0].id, 0);
    assert_eq!(records[0].vector, vec![1.5, 2.5, 3.5]);
    assert_eq!(records[1].id, 1);
    assert_eq!(records[1].vector, vec![4.0, 5.0, 6.0]);
    // fvecs records have no text/source/tags
    assert!(records[0].text.is_none());
    assert!(records[0].source.is_none());
    assert!(records[0].tags.is_none());
}

#[test]
fn bvecs_records_returns_f32_promoted_values() {
    let mut data = Vec::new();
    write_bvecs_record(&mut data, &[0, 128, 255]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".bvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();

    assert_eq!(records.len(), 1);
    assert_eq!(records[0].vector, vec![0.0, 128.0, 255.0]);
}

#[test]
fn fvecs_empty_iteration() {
    // A valid fvecs file with 1 record, but let's check that iteration
    // over a file with exactly 0 records after dim check would fail at open.
    // Instead, test that iterating a 1-record file works and stops cleanly.
    let mut data = Vec::new();
    write_fvecs_record(&mut data, &[42.0]);

    let mut tmp = tempfile::Builder::new()
        .suffix(".fvecs")
        .tempfile()
        .unwrap();
    tmp.write_all(&data).unwrap();

    let mut reader = FvecsReader::open(tmp.path()).unwrap();
    let records: Vec<_> = reader.records().collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].vector, vec![42.0]);
}
