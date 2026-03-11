//! HNSW snapshot: mmap-ready flat file for zero-copy search.
//!
//! Layout (all sections 8-byte aligned):
//! - Header (128B): magic, version, graph metadata (16 × u64)
//! - Lookup Table (N × 24B): sorted by point_id for binary search
//! - Neighbor Data (variable): per-node per-layer `[count: u64, ids: [u64]]`

use std::io::{BufWriter, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError, storage_err, read_le_u64};

use crate::config::HnswConfig;
use crate::graph::HnswGraph;
use crate::search::{search_with_store, search_two_stage, DistanceComputer, NodeGraph};

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

        for &(&id, node) in &nodes {
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

    /// Two-stage search: approximate distance for traversal, exact for rescore.
    ///
    /// Use with SQ8: pass an SQ8 `DistanceComputer` as `approx` and f32 as `exact`.
    pub fn search_two_stage(
        &self,
        approx: &dyn DistanceComputer,
        exact: &dyn DistanceComputer,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
        search_two_stage(
            self, approx, exact, &self.config,
            self.entry_point, self.max_layer, query, k, ef,
        )
    }

    /// Number of live (non-deleted) nodes.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if there are no live nodes.
    pub fn is_empty(&self) -> bool {
        self.count == 0
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
            let count = read_le_u64(&self.mmap, off)? as usize;
            off += 8 + count * 8;
        }

        // Read target layer
        let count = read_le_u64(&self.mmap, off)? as usize;
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
