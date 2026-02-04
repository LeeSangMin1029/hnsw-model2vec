//! Delta-encoded neighbor lists for compact storage.

use v_hnsw_core::PointId;

/// Delta-encoded neighbor list.
///
/// Stores sorted PointIds compactly: the first ID as a varint, then deltas
/// between consecutive sorted IDs as varints. For typical HNSW graphs where
/// neighbor IDs are moderately spaced, this saves ~60-75% memory compared
/// to `Vec<PointId>` (u64).
#[derive(Debug, Clone, Default)]
pub(crate) struct DeltaNeighbors {
    /// LEB128-encoded data: `[first_id_varint, delta1_varint, delta2_varint, ...]`
    data: Vec<u8>,
    /// Number of stored neighbor IDs.
    count: u16,
}

/// Encode a `u64` value as an unsigned LEB128 varint, appending bytes to `buf`.
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        }
        buf.push(byte | 0x80);
    }
}

/// Decode one unsigned LEB128 varint starting at `data[*pos]`.
///
/// Advances `*pos` past the consumed bytes. Returns 0 if `pos` is out of
/// bounds (callers rely on `count` to avoid calling this in that situation).
fn decode_varint(data: &[u8], pos: &mut usize) -> u64 {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    loop {
        if *pos >= data.len() {
            break;
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 {
            break;
        }
    }
    result
}

#[allow(dead_code)]
impl DeltaNeighbors {
    /// Create an empty `DeltaNeighbors`.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            count: 0,
        }
    }

    /// Build from a slice of IDs. The IDs are sorted and deduplicated internally.
    pub fn from_ids(ids: &[PointId]) -> Self {
        if ids.is_empty() {
            return Self::new();
        }

        let mut sorted: Vec<PointId> = ids.to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        let mut data = Vec::with_capacity(sorted.len() * 2); // rough estimate
        let mut prev = sorted[0];
        encode_varint(&mut data, prev);

        for &id in &sorted[1..] {
            encode_varint(&mut data, id - prev);
            prev = id;
        }

        Self {
            data,
            count: sorted.len() as u16,
        }
    }

    /// Decode all stored IDs into a `Vec<PointId>`.
    pub fn decode(&self) -> Vec<PointId> {
        if self.count == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(self.count as usize);
        let mut pos: usize = 0;

        let first = decode_varint(&self.data, &mut pos);
        result.push(first);

        let mut current = first;
        for _ in 1..self.count {
            let delta = decode_varint(&self.data, &mut pos);
            current += delta;
            result.push(current);
        }

        result
    }

    /// Number of stored neighbor IDs.
    pub fn len(&self) -> usize {
        self.count as usize
    }

    /// Returns `true` if there are no stored IDs.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Add one ID. Decodes, inserts, re-encodes.
    ///
    /// This is acceptable because pushes are rare compared to reads in HNSW.
    pub fn push(&mut self, id: PointId) {
        let mut ids = self.decode();
        ids.push(id);
        *self = Self::from_ids(&ids);
    }

    /// Check whether `id` is present in the encoded neighbor list.
    pub fn contains(&self, id: PointId) -> bool {
        if self.count == 0 {
            return false;
        }

        let mut pos: usize = 0;
        let first = decode_varint(&self.data, &mut pos);
        if first == id {
            return true;
        }

        let mut current = first;
        for _ in 1..self.count {
            let delta = decode_varint(&self.data, &mut pos);
            current += delta;
            if current == id {
                return true;
            }
            // IDs are sorted — early exit if we passed the target.
            if current > id {
                return false;
            }
        }

        false
    }

    /// Actual heap memory used: encoded bytes + 2 bytes for the count field.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
