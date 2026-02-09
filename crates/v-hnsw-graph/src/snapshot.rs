//! HNSW snapshot: mmap-ready flat file for zero-copy search.
//!
//! Layout (all sections 8-byte aligned):
//! - Header (128B): magic, version, graph metadata (16 × u64)
//! - Lookup Table (N × 24B): sorted by point_id for binary search
//! - Neighbor Data (variable): per-node per-layer `[count: u64, ids: [u64]]`

use std::io::{BufWriter, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError};

use crate::config::HnswConfig;
use crate::graph::HnswGraph;
use crate::search::{search_with_store, NodeGraph};

const MAGIC: u64 = 0x484E_5357_534E_4150; // "HNSWSNAP"
const VERSION: u64 = 1;
const HEADER_SLOTS: usize = 16; // 16 × 8 = 128 bytes

/// Lookup entry: 24 bytes (3 × u64), always 8-byte aligned.
///
/// Sorted by `point_id` for O(log N) binary search (~16 steps for 54K nodes).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LookupEntry {
    point_id: u64,
    /// Byte offset from the start of the neighbor data section.
    data_offset: u64,
    /// Bits `[0:7]` = max_layer (`LayerId`). Bit 32 = deleted flag.
    max_layer_and_flags: u64,
}

/// HNSW graph snapshot backed by a memory-mapped file.
///
/// Provides zero-copy neighbor access via [`NodeGraph`] trait,
/// enabling `search_with_store` without heap-loading the graph.
pub struct HnswSnapshot {
    mmap: memmap2::Mmap,
    num_nodes: usize,
    count: usize,
    lookup_offset: usize,
    data_offset: usize,
    entry_point: Option<PointId>,
    max_layer: LayerId,
    config: HnswConfig,
}

fn storage_err(msg: &str) -> VhnswError {
    VhnswError::Storage(std::io::Error::other(msg))
}

impl HnswSnapshot {
    /// Write an [`HnswGraph`] as a flat snapshot file.
    ///
    /// Nodes are sorted by `point_id` so the snapshot supports binary-search lookup.
    pub fn save<D: DistanceMetric>(graph: &HnswGraph<D>, path: &Path) -> v_hnsw_core::Result<()> {
        let file = std::fs::File::create(path).map_err(VhnswError::Storage)?;
        let mut w = BufWriter::new(file);

        // Sort nodes by point_id for binary-search lookup
        let mut nodes: Vec<_> = graph.nodes.iter().collect();
        nodes.sort_unstable_by_key(|&(&id, _)| id);

        // Build neighbor data blob + lookup entries
        let mut data = Vec::new();
        let mut entries = Vec::with_capacity(nodes.len());

        for &(&id, ref node) in &nodes {
            let offset = data.len() as u64;
            let flags = (node.max_layer as u64) | if node.deleted { 1u64 << 32 } else { 0 };
            entries.push(LookupEntry {
                point_id: id,
                data_offset: offset,
                max_layer_and_flags: flags,
            });
            for layer in 0..=node.max_layer {
                let nbrs = node.neighbors_at(layer);
                data.extend_from_slice(&(nbrs.len() as u64).to_le_bytes());
                for &nid in nbrs {
                    data.extend_from_slice(&nid.to_le_bytes());
                }
            }
        }

        // Header: 16 × u64 = 128 bytes
        let c = &graph.config;
        let header: [u64; HEADER_SLOTS] = [
            MAGIC,
            VERSION,
            nodes.len() as u64,
            graph.entry_point.unwrap_or(0),
            u64::from(graph.entry_point.is_some()),
            graph.max_layer as u64,
            graph.count as u64,
            graph.rng_state,
            c.dim as u64,
            c.m as u64,
            c.m0 as u64,
            c.ef_construction as u64,
            c.max_elements as u64,
            c.ml.to_bits(),
            0, 0, // reserved
        ];

        w.write_all(bytemuck::cast_slice(&header)).map_err(VhnswError::Storage)?;
        w.write_all(bytemuck::cast_slice(&entries)).map_err(VhnswError::Storage)?;
        w.write_all(&data).map_err(VhnswError::Storage)?;
        w.flush().map_err(VhnswError::Storage)?;
        Ok(())
    }

    /// Open a snapshot file via memory mapping.
    pub fn open(path: &Path) -> v_hnsw_core::Result<Self> {
        let file = std::fs::File::open(path).map_err(VhnswError::Storage)?;
        // SAFETY: read-only mmap; file format is validated below.
        #[allow(unsafe_code)]
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(VhnswError::Storage)?;

        let header_bytes = HEADER_SLOTS * 8;
        if mmap.len() < header_bytes {
            return Err(storage_err("snapshot file too small"));
        }

        // Copy header values before moving mmap into struct
        let h: [u64; HEADER_SLOTS] = *bytemuck::try_from_bytes(&mmap[..header_bytes])
            .map_err(|_| storage_err("header alignment error"))?;

        if h[0] != MAGIC {
            return Err(storage_err("invalid snapshot magic"));
        }
        if h[1] != VERSION {
            return Err(storage_err("unsupported snapshot version"));
        }

        let num_nodes = h[2] as usize;
        let lookup_offset = header_bytes;
        let data_offset = lookup_offset + num_nodes * std::mem::size_of::<LookupEntry>();

        Ok(Self {
            mmap,
            num_nodes,
            count: h[6] as usize,
            lookup_offset,
            data_offset,
            entry_point: if h[4] != 0 { Some(h[3]) } else { None },
            max_layer: h[5] as LayerId,
            config: HnswConfig {
                dim: h[8] as usize,
                m: h[9] as usize,
                m0: h[10] as usize,
                ef_construction: h[11] as usize,
                max_elements: h[12] as usize,
                ml: f64::from_bits(h[13]),
            },
        })
    }

    /// Search the snapshot using an external vector store and distance metric.
    pub fn search_ext<D: DistanceMetric>(
        &self,
        distance: &D,
        store: &dyn VectorStore,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        search_with_store(
            self, store, distance, &self.config,
            self.entry_point, self.max_layer, query, k, ef,
        )
    }

    /// Number of live (non-deleted) nodes.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Get the snapshot configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    /// Get the entry point.
    pub fn entry_point(&self) -> Option<PointId> {
        self.entry_point
    }

    fn lookup_table(&self) -> &[LookupEntry] {
        let end = self.lookup_offset + self.num_nodes * std::mem::size_of::<LookupEntry>();
        bytemuck::try_cast_slice(&self.mmap[self.lookup_offset..end]).unwrap_or(&[])
    }

    fn find_entry(&self, id: PointId) -> Option<&LookupEntry> {
        let table = self.lookup_table();
        table.binary_search_by_key(&id, |e| e.point_id).ok().map(|i| &table[i])
    }
}

impl NodeGraph for HnswSnapshot {
    fn neighbors(&self, id: PointId, layer: LayerId) -> Option<&[PointId]> {
        let entry = self.find_entry(id)?;
        let node_max = (entry.max_layer_and_flags & 0xFF) as LayerId;
        if layer > node_max {
            return Some(&[]);
        }

        let mut off = self.data_offset + entry.data_offset as usize;

        // Skip preceding layers
        for _ in 0..layer {
            let count = read_u64(&self.mmap, off)? as usize;
            off += 8 + count * 8;
        }

        // Read target layer
        let count = read_u64(&self.mmap, off)? as usize;
        if count == 0 {
            return Some(&[]);
        }
        let nbr_bytes = self.mmap.get(off + 8..off + 8 + count * 8)?;
        bytemuck::try_cast_slice(nbr_bytes).ok()
    }

    fn is_deleted(&self, id: PointId) -> bool {
        self.find_entry(id).is_none_or(|e| (e.max_layer_and_flags >> 32) & 1 != 0)
    }
}

/// Read a little-endian u64 from a byte slice at the given offset.
fn read_u64(data: &[u8], offset: usize) -> Option<u64> {
    let bytes: [u8; 8] = data.get(offset..offset + 8)?.try_into().ok()?;
    Some(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::L2Distance;
    use v_hnsw_core::VectorIndex;

    fn test_vector(id: u64, dim: usize) -> Vec<f32> {
        (0..dim).map(|j| (id as f32 * 0.1 + j as f32 * 0.3).sin()).collect()
    }

    #[test]
    fn test_snapshot_roundtrip() -> v_hnsw_core::Result<()> {
        let dim = 16;
        let config = HnswConfig::builder().dim(dim).m(8).build()?;
        let mut graph = HnswGraph::with_seed(config, L2Distance, 42);
        for i in 0..100 {
            graph.insert(i, &test_vector(i, dim))?;
        }

        let path = std::env::temp_dir().join("test_hnsw_snapshot.snap");
        HnswSnapshot::save(&graph, &path)?;
        let snap = HnswSnapshot::open(&path)?;

        assert_eq!(snap.len(), graph.len());
        assert_eq!(snap.entry_point(), graph.entry_point());
        assert_eq!(snap.config().dim, dim);

        // Verify neighbor access matches
        for (&id, node) in &graph.nodes {
            for layer in 0..=node.max_layer {
                let graph_nbrs = node.neighbors_at(layer);
                let snap_nbrs = snap.neighbors(id, layer);
                assert_eq!(
                    snap_nbrs.map(|s| s.to_vec()),
                    Some(graph_nbrs.to_vec()),
                    "mismatch at node {} layer {}",
                    id,
                    layer
                );
            }
            assert_eq!(snap.is_deleted(id), node.deleted);
        }

        // Search comparison
        let query = test_vector(50, dim);
        let graph_results = graph.search(&query, 10, 50)?;
        let snap_results = snap.search_ext(&L2Distance, &graph.store, &query, 10, 50)?;

        assert_eq!(graph_results.len(), snap_results.len());
        for (g, s) in graph_results.iter().zip(snap_results.iter()) {
            assert_eq!(g.0, s.0);
            assert!((g.1 - s.1).abs() < 1e-6);
        }

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn test_snapshot_empty_graph() -> v_hnsw_core::Result<()> {
        let config = HnswConfig::builder().dim(4).build()?;
        let graph = HnswGraph::new(config, L2Distance);

        let path = std::env::temp_dir().join("test_hnsw_snapshot_empty.snap");
        HnswSnapshot::save(&graph, &path)?;
        let snap = HnswSnapshot::open(&path)?;

        assert_eq!(snap.len(), 0);
        assert_eq!(snap.entry_point(), None);

        let results = snap.search_ext(&L2Distance, &graph.store, &[1.0, 2.0, 3.0, 4.0], 5, 50)?;
        assert!(results.is_empty());

        let _ = std::fs::remove_file(&path);
        Ok(())
    }

    #[test]
    fn test_snapshot_deleted_node() -> v_hnsw_core::Result<()> {
        let dim = 4;
        let config = HnswConfig::builder().dim(dim).build()?;
        let mut graph = HnswGraph::with_seed(config, L2Distance, 42);

        graph.insert(1, &[1.0, 0.0, 0.0, 0.0])?;
        graph.insert(2, &[0.0, 1.0, 0.0, 0.0])?;
        graph.delete(1)?;

        let path = std::env::temp_dir().join("test_hnsw_snapshot_deleted.snap");
        HnswSnapshot::save(&graph, &path)?;
        let snap = HnswSnapshot::open(&path)?;

        assert_eq!(snap.len(), 1); // only 1 live node
        assert!(snap.is_deleted(1));
        assert!(!snap.is_deleted(2));

        // Search near deleted point: should only return point 2
        let results = snap.search_ext(&L2Distance, &graph.store, &[1.0, 0.0, 0.0, 0.0], 5, 50)?;
        for (id, _) in &results {
            assert_ne!(*id, 1);
        }

        let _ = std::fs::remove_file(&path);
        Ok(())
    }
}
