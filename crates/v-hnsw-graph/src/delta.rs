//! Delta-encoded neighbor lists for compact storage.

use v_hnsw_core::PointId;

/// Delta-encoded neighbor list.
///
/// Stores sorted PointIds compactly: the first ID as a varint, then deltas
/// between consecutive sorted IDs as varints. For typical HNSW graphs where
/// neighbor IDs are moderately spaced, this saves ~60-75% memory compared
/// to `Vec<PointId>` (u64).
#[derive(Debug, Clone, Default, bincode::Encode, bincode::Decode)]
pub(crate) struct DeltaNeighbors {
    /// LEB128-encoded data: `[first_id_varint, delta1_varint, delta2_varint, ...]`
    data: Vec<u8>,
    /// Number of stored neighbor IDs.
    count: u16,
}

/// Encode a `u64` value as an unsigned LEB128 varint, appending bytes to `buf`.
pub(crate) fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
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
pub(crate) fn decode_varint(data: &[u8], pos: &mut usize) -> u64 {
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
