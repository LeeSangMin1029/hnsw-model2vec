use crate::wal::*;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use v_hnsw_core::{Payload, Result};

fn wal_payload() -> Payload {
    Payload {
        source: "test.md".to_string(),
        tags: vec![],
        created_at: 0,
        source_modified_at: 0,
        chunk_index: 0,
        chunk_total: 1,
        custom: HashMap::new(),
    }
}

#[test]
fn test_wal_create_and_append() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("wal_test_create");
    let _ = fs::remove_dir_all(&temp_dir);

    let mut wal = Wal::create(&temp_dir)?;

    let payload = wal_payload();

    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0, 2.0],
        payload: payload.clone(),
        text: "test".to_string(),
    })?;

    wal.append(&WalRecord::Remove { id: 1 })?;

    assert_eq!(wal.pending_count(), 2);

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn test_wal_replay() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("wal_test_replay");
    let _ = fs::remove_dir_all(&temp_dir);

    let mut wal = Wal::create(&temp_dir)?;

    let payload = wal_payload();

    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0, 2.0],
        payload: payload.clone(),
        text: "test1".to_string(),
    })?;

    wal.checkpoint(1, 1)?;

    wal.append(&WalRecord::Insert {
        id: 2,
        vector: vec![3.0, 4.0],
        payload: payload.clone(),
        text: "test2".to_string(),
    })?;

    drop(wal);

    // Reopen and replay
    let wal = Wal::open(&temp_dir)?;
    let records = wal.replay()?;

    // Should only have records after checkpoint
    assert_eq!(records.len(), 1);
    matches!(&records[0], WalRecord::Insert { id: 2, .. });

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[test]
fn test_wal_incomplete_batch() -> Result<()> {
    let temp_dir = std::env::temp_dir().join("wal_test_batch");
    let _ = fs::remove_dir_all(&temp_dir);

    let mut wal = Wal::create(&temp_dir)?;

    let payload = wal_payload();

    // Complete batch
    wal.append(&WalRecord::BatchBegin { batch_id: 1 })?;
    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0],
        payload: payload.clone(),
        text: "complete".to_string(),
    })?;
    wal.append(&WalRecord::BatchEnd { batch_id: 1 })?;

    // Incomplete batch (no end)
    wal.append(&WalRecord::BatchBegin { batch_id: 2 })?;
    wal.append(&WalRecord::Insert {
        id: 2,
        vector: vec![2.0],
        payload: payload.clone(),
        text: "incomplete".to_string(),
    })?;

    drop(wal);

    // Replay should only include the complete batch
    let wal = Wal::open(&temp_dir)?;
    let records = wal.replay()?;

    // Should have: BatchBegin(1), Insert(1), BatchEnd(1)
    assert_eq!(records.len(), 3);

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

// --- New edge-case tests ---

#[test]
fn test_wal_empty_replay() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_empty");

    let wal = Wal::create(&wal_dir)?;
    let records = wal.replay()?;
    assert!(records.is_empty(), "empty WAL should replay zero records");
    assert_eq!(wal.pending_count(), 0);
    Ok(())
}

#[test]
fn test_wal_corrupted_crc_stops_replay() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_corrupt_crc");

    // Write a valid record then corrupt the file
    {
        let mut wal = Wal::create(&wal_dir)?;
        wal.append(&WalRecord::Insert {
            id: 1,
            vector: vec![1.0],
            payload: wal_payload(),
            text: "good".to_string(),
        })?;
        wal.append(&WalRecord::Insert {
            id: 2,
            vector: vec![2.0],
            payload: wal_payload(),
            text: "will be corrupted".to_string(),
        })?;
        drop(wal);
    }

    // Corrupt: read the segment, find the length of the first record, then corrupt the second
    let seg_path = wal_dir.join("wal-000000.log");
    let mut data = fs::read(&seg_path)?;
    // First record: [crc32:4][length:4][data:length]
    // Read the first record's data length to skip past it
    let first_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
    let second_record_start = 4 + 4 + first_len; // start of second record's CRC
    // Flip a byte in the second record's CRC to make it invalid
    if second_record_start < data.len() {
        data[second_record_start] ^= 0xFF;
    }
    fs::write(&seg_path, &data)?;

    let wal = Wal::open(&wal_dir)?;
    let records = wal.replay()?;

    // Should recover only the first valid record (corrupt CRC stops reading)
    assert_eq!(records.len(), 1);
    assert!(matches!(&records[0], WalRecord::Insert { id: 1, .. }));
    Ok(())
}

#[test]
fn test_wal_truncated_record_is_ignored() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_truncated");

    {
        let mut wal = Wal::create(&wal_dir)?;
        wal.append(&WalRecord::Insert {
            id: 10,
            vector: vec![1.0, 2.0, 3.0],
            payload: wal_payload(),
            text: "valid".to_string(),
        })?;
        drop(wal);
    }

    // Append a partial/truncated record header at the end
    let seg_path = wal_dir.join("wal-000000.log");
    let mut f = fs::OpenOptions::new().append(true).open(&seg_path)?;
    f.write_all(&[0xDE, 0xAD])?; // only 2 bytes, not enough for a CRC header
    drop(f);

    let wal = Wal::open(&wal_dir)?;
    let records = wal.replay()?;
    assert_eq!(records.len(), 1);
    assert!(matches!(&records[0], WalRecord::Insert { id: 10, .. }));
    Ok(())
}

#[test]
fn test_wal_multiple_checkpoints_only_last_matters() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_multi_ckpt");

    let mut wal = Wal::create(&wal_dir)?;
    let p = wal_payload();

    // Insert A, checkpoint, Insert B, checkpoint, Insert C
    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0],
        payload: p.clone(),
        text: "A".into(),
    })?;
    wal.checkpoint(1, 1)?;

    wal.append(&WalRecord::Insert {
        id: 2,
        vector: vec![2.0],
        payload: p.clone(),
        text: "B".into(),
    })?;
    wal.checkpoint(2, 2)?;

    wal.append(&WalRecord::Insert {
        id: 3,
        vector: vec![3.0],
        payload: p.clone(),
        text: "C".into(),
    })?;

    drop(wal);

    let wal = Wal::open(&wal_dir)?;
    let records = wal.replay()?;

    // Only records after the LAST checkpoint should be returned
    assert_eq!(records.len(), 1);
    assert!(matches!(&records[0], WalRecord::Insert { id: 3, .. }));
    Ok(())
}

#[test]
fn test_wal_truncate_clears_all() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_truncate");

    let mut wal = Wal::create(&wal_dir)?;
    let p = wal_payload();

    for i in 0..10 {
        wal.append(&WalRecord::Insert {
            id: i,
            vector: vec![i as f32],
            payload: p.clone(),
            text: format!("rec{i}"),
        })?;
    }

    wal.truncate()?;

    let records = wal.replay()?;
    assert!(records.is_empty(), "truncate should clear all records");
    assert_eq!(wal.pending_count(), 0);
    Ok(())
}

#[test]
fn test_wal_append_batch_empty() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_batch_empty");

    let mut wal = Wal::create(&wal_dir)?;

    // Appending an empty batch should be a no-op
    wal.append_batch(&[])?;
    assert_eq!(wal.pending_count(), 0);

    let records = wal.replay()?;
    assert!(records.is_empty());
    Ok(())
}

#[test]
fn test_wal_append_batch_multiple_records() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_batch_multi");

    let mut wal = Wal::create(&wal_dir)?;
    let p = wal_payload();

    let records_to_write: Vec<WalRecord> = (0..5)
        .map(|i| WalRecord::Insert {
            id: i,
            vector: vec![i as f32; 4],
            payload: p.clone(),
            text: format!("batch_item_{i}"),
        })
        .collect();

    wal.append_batch(&records_to_write)?;
    assert_eq!(wal.pending_count(), 5);

    drop(wal);

    let wal = Wal::open(&wal_dir)?;
    let replayed = wal.replay()?;
    assert_eq!(replayed.len(), 5);
    Ok(())
}

#[test]
fn test_wal_open_nonexistent_dir_returns_error() {
    let result = Wal::open("/tmp/v_hnsw_test_nonexistent_wal_dir_42424242");
    assert!(result.is_err());
}

#[test]
fn test_wal_purge_old_segments() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_purge");

    let mut wal = Wal::create(&wal_dir)?;
    let p = wal_payload();

    // Write some data to segment 0
    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0],
        payload: p.clone(),
        text: "s0".into(),
    })?;

    // Purge old segments (segment_number is 0, so nothing older to purge)
    wal.purge_old_segments()?;

    // Segment 0 file should still exist
    assert!(wal_dir.join("wal-000000.log").exists());
    Ok(())
}

#[test]
fn test_wal_checkpoint_resets_pending_count() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_ckpt_count");

    let mut wal = Wal::create(&wal_dir)?;
    let p = wal_payload();

    wal.append(&WalRecord::Insert {
        id: 1,
        vector: vec![1.0],
        payload: p.clone(),
        text: "x".into(),
    })?;
    wal.append(&WalRecord::Insert {
        id: 2,
        vector: vec![2.0],
        payload: p.clone(),
        text: "y".into(),
    })?;

    assert_eq!(wal.pending_count(), 2);
    wal.checkpoint(1, 2)?;
    assert_eq!(wal.pending_count(), 0);
    Ok(())
}

#[test]
fn test_wal_large_payload_roundtrip() -> Result<()> {
    let temp = tempfile::tempdir().unwrap();
    let wal_dir = temp.path().join("wal_large");

    let mut wal = Wal::create(&wal_dir)?;

    let big_text = "x".repeat(100_000);
    let big_vector: Vec<f32> = (0..1024).map(|i| i as f32 * 0.001).collect();
    let p = wal_payload();

    wal.append(&WalRecord::Insert {
        id: 42,
        vector: big_vector.clone(),
        payload: p,
        text: big_text.clone(),
    })?;

    drop(wal);

    let wal = Wal::open(&wal_dir)?;
    let records = wal.replay()?;
    assert_eq!(records.len(), 1);
    if let WalRecord::Insert { id, vector, text, .. } = &records[0] {
        assert_eq!(*id, 42);
        assert_eq!(vector.len(), 1024);
        assert_eq!(text.len(), 100_000);
    } else {
        panic!("expected Insert record");
    }
    Ok(())
}
