use crate::shm::ShmRing;

#[test]
fn roundtrip_write_read() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.shm");

    let mut ring = ShmRing::create_with_params(&path, 4, 1024).expect("create");

    let msg = b"hello world";
    ring.write(msg).expect("write");

    let data = ring.try_read().expect("read").expect("some");
    assert_eq!(data, msg);
}

#[test]
fn multiple_messages() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.shm");

    let mut ring = ShmRing::create_with_params(&path, 8, 4096).expect("create");

    for i in 0..5u32 {
        let msg = format!("message {i}");
        ring.write(msg.as_bytes()).expect("write");
    }

    for i in 0..5u32 {
        let data = ring.try_read().expect("read").expect("some");
        let expected = format!("message {i}");
        assert_eq!(data, expected.as_bytes());
    }

    // No more messages.
    assert!(ring.try_read().expect("read").is_none());
}

#[test]
fn reopen_existing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.shm");

    {
        let mut ring = ShmRing::create_with_params(&path, 4, 1024).expect("create");
        ring.write(b"persistent").expect("write");
    }

    // Reopen.
    let mut ring = ShmRing::open(&path).expect("open");
    let data = ring.try_read().expect("read").expect("some");
    assert_eq!(data, b"persistent");
}

#[test]
fn empty_read_returns_none() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test.shm");

    let mut ring = ShmRing::create_with_params(&path, 4, 1024).expect("create");
    assert!(ring.try_read().expect("read").is_none());
}
