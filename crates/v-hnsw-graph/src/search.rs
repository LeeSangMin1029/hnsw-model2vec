//! HNSW search algorithms.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use rustc_hash::FxHashSet;

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError};
use crate::distance::prefetch_vector;

use crate::graph::HnswGraph;
use crate::node::Node;

/// Abstract graph node access for search.
///
/// Implemented by `HashMap<PointId, Node>` (heap graph) and `HnswSnapshot` (mmap).
pub trait NodeGraph {
    /// Get the neighbor list at a given layer. Returns `None` if node doesn't exist.
    fn neighbors(&self, id: PointId, layer: LayerId) -> Option<&[PointId]>;
    /// Check if a node is deleted (or doesn't exist).
    fn is_deleted(&self, id: PointId) -> bool;
}

impl NodeGraph for HashMap<PointId, Node> {
    fn neighbors(&self, id: PointId, layer: LayerId) -> Option<&[PointId]> {
        self.get(&id).map(|n| n.neighbors_at(layer))
    }
    fn is_deleted(&self, id: PointId) -> bool {
        self.get(&id).is_none_or(|n| n.deleted)
    }
}

/// Reusable search buffers to avoid per-query allocations.
struct SearchBuffer {
    visited: FxHashSet<PointId>,
    candidates: BinaryHeap<Reverse<HeapEntry>>,
    results: BinaryHeap<HeapEntry>,
}

impl SearchBuffer {
    fn new() -> Self {
        Self {
            visited: FxHashSet::default(),
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.results.clear();
    }
}

thread_local! {
    static SEARCH_BUF: RefCell<SearchBuffer> = RefCell::new(SearchBuffer::new());
}

/// Ordered entry for the search heaps.
///
/// Implements `Ord` by distance (total ordering via `f32` comparison).
#[derive(Clone, Copy, PartialEq)]
struct HeapEntry {
    id: PointId,
    dist: f32,
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Primary: by distance. NaN sorts last (treated as infinity).
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Distance computer: abstracts vector lookup + distance computation.
///
/// Used by generic search functions to decouple graph traversal from
/// the underlying storage format (f32, SQ8, etc.).
pub trait DistanceComputer {
    /// Compute distance from `query` to the vector identified by `id`.
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32>;

    /// Prefetch vector data for the given id to hide memory latency.
    /// Default is a no-op; override for storage backends that support prefetch.
    fn prefetch(&self, _id: PointId) {}
}

/// Adapter: wraps `VectorStore + DistanceMetric` into a `DistanceComputer`.
pub(crate) struct StoreDc<'a, D> {
    pub store: &'a dyn VectorStore,
    pub metric: &'a D,
}

impl<D: DistanceMetric> DistanceComputer for StoreDc<'_, D> {
    #[inline]
    fn distance(&self, query: &[f32], id: PointId) -> v_hnsw_core::Result<f32> {
        let vec = self.store.get(id)?;
        Ok(self.metric.distance(query, vec))
    }

    #[inline]
    fn prefetch(&self, id: PointId) {
        if let Ok(vec) = self.store.get(id) {
            prefetch_vector(vec);
        }
    }
}

/// Top-level HNSW search using the graph's internal vector store.
///
/// Returns up to `k` nearest neighbors sorted by ascending distance.
pub(crate) fn search<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    query: &[f32],
    k: usize,
    ef: usize,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    search_with_store(&graph.nodes, &graph.store, &graph.distance, &graph.config,
        graph.entry_point, graph.max_layer, query, k, ef)
}

/// Top-level HNSW search using an external vector store.
///
/// Allows searching without copying vectors into the graph (e.g. mmap direct read).
pub(crate) fn search_ext<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    store: &dyn VectorStore,
    query: &[f32],
    k: usize,
    ef: usize,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    search_with_store(&graph.nodes, store, &graph.distance, &graph.config,
        graph.entry_point, graph.max_layer, query, k, ef)
}

/// Greedy descent from entry point through upper layers, returning the layer-0 entry.
///
/// Shared by single-stage and two-stage search paths.
fn descend_to_layer0<N: NodeGraph>(
    nodes: &N,
    dc: &dyn DistanceComputer,
    config: &crate::config::HnswConfig,
    entry_point: Option<PointId>,
    max_layer: LayerId,
    query: &[f32],
) -> v_hnsw_core::Result<Option<(PointId, f32)>> {
    if query.len() != config.dim {
        return Err(VhnswError::DimensionMismatch {
            expected: config.dim,
            got: query.len(),
        });
    }

    let entry_id = match entry_point {
        Some(id) => id,
        None => return Ok(None),
    };

    let mut cur_dist = dc.distance(query, entry_id)?;
    let mut cur_id = entry_id;

    if max_layer > 0 {
        for layer_idx in (1..=max_layer).rev() {
            let layer = layer_idx as LayerId;
            let (new_id, new_dist) = greedy_closest_dc(nodes, dc, query, cur_id, cur_dist, layer)?;
            cur_id = new_id;
            cur_dist = new_dist;
        }
    }

    Ok(Some((cur_id, cur_dist)))
}

/// Sort by distance ascending and truncate to top-k.
fn sort_truncate(results: &mut Vec<(PointId, f32)>, k: usize) {
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
}

/// Internal search implementation, generic over node access.
#[expect(clippy::too_many_arguments)]
pub(crate) fn search_with_store<D: DistanceMetric, N: NodeGraph>(
    nodes: &N,
    store: &dyn VectorStore,
    distance: &D,
    config: &crate::config::HnswConfig,
    entry_point: Option<PointId>,
    max_layer: LayerId,
    query: &[f32],
    k: usize,
    ef: usize,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    let dc = StoreDc { store, metric: distance };
    let Some((cur_id, cur_dist)) = descend_to_layer0(nodes, &dc, config, entry_point, max_layer, query)? else {
        return Ok(Vec::new());
    };

    let effective_ef = ef.max(k);
    let entry_points = vec![(cur_id, cur_dist)];
    let mut results = search_layer_dc(nodes, &dc, query, &entry_points, effective_ef, 0)?;

    sort_truncate(&mut results, k);
    Ok(results)
}

/// Two-stage HNSW search with approximate distance for traversal and exact rescore.
///
/// 1. Greedy descent (upper layers) + layer-0 beam search using `approx_dc`.
/// 2. Rescore top-ef candidates using `exact_dc` for accurate final ranking.
/// 3. Return top-k results.
#[expect(clippy::too_many_arguments)]
pub fn search_two_stage<N: NodeGraph>(
    nodes: &N,
    approx_dc: &dyn DistanceComputer,
    exact_dc: &dyn DistanceComputer,
    config: &crate::config::HnswConfig,
    entry_point: Option<PointId>,
    max_layer: LayerId,
    query: &[f32],
    k: usize,
    ef: usize,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    let Some((cur_id, cur_d)) = descend_to_layer0(nodes, approx_dc, config, entry_point, max_layer, query)? else {
        return Ok(Vec::new());
    };

    let effective_ef = ef.max(k);
    let entry_points = vec![(cur_id, cur_d)];
    let approx_results = search_layer_dc(nodes, approx_dc, query, &entry_points, effective_ef, 0)?;

    // Rescore all candidates with exact distance
    let mut rescored: Vec<(PointId, f32)> = Vec::with_capacity(approx_results.len());
    for (id, _approx_dist) in &approx_results {
        if let Ok(exact_dist) = exact_dc.distance(query, *id) {
            rescored.push((*id, exact_dist));
        }
    }

    sort_truncate(&mut rescored, k);
    Ok(rescored)
}

/// Greedy descent using a `DistanceComputer`.
fn greedy_closest_dc<N: NodeGraph>(
    nodes: &N,
    dc: &dyn DistanceComputer,
    query: &[f32],
    mut cur_id: PointId,
    mut cur_dist: f32,
    layer: LayerId,
) -> v_hnsw_core::Result<(PointId, f32)> {
    loop {
        let mut changed = false;
        let neighbors = match nodes.neighbors(cur_id, layer) {
            Some(n) => n,
            None => break,
        };

        for &neighbor_id in neighbors {
            if nodes.is_deleted(neighbor_id) {
                continue;
            }
            let dist = dc.distance(query, neighbor_id)?;
            if dist < cur_dist {
                cur_dist = dist;
                cur_id = neighbor_id;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
    Ok((cur_id, cur_dist))
}

/// Search within a single layer using a `DistanceComputer`.
///
/// Uses a min-heap for candidates (closest first) and a max-heap for results.
/// Applies software prefetch via `DistanceComputer::prefetch` to hide memory latency.
/// Reuses thread-local buffers to avoid per-query allocations.
fn search_layer_dc<N: NodeGraph>(
    nodes: &N,
    dc: &dyn DistanceComputer,
    query: &[f32],
    entry_points: &[(PointId, f32)],
    ef: usize,
    layer: LayerId,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    SEARCH_BUF.with_borrow_mut(|buf| {
        buf.clear();

        for &(id, dist) in entry_points {
            if buf.visited.insert(id) {
                buf.candidates.push(Reverse(HeapEntry { id, dist }));
                if !nodes.is_deleted(id) {
                    buf.results.push(HeapEntry { id, dist });
                }
            }
        }

        while let Some(Reverse(closest)) = buf.candidates.pop() {
            if let Some(worst) = buf.results.peek()
                && closest.dist > worst.dist
                && buf.results.len() >= ef
            {
                break;
            }

            let neighbor_ids = match nodes.neighbors(closest.id, layer) {
                Some(n) => n,
                None => continue,
            };

            // Traverse neighbors with software prefetch (lookahead = 4)
            const PREFETCH_AHEAD: usize = 4;
            for (i, &neighbor_id) in neighbor_ids.iter().enumerate() {
                if !buf.visited.insert(neighbor_id) {
                    continue;
                }
                if nodes.is_deleted(neighbor_id) {
                    continue;
                }

                // Prefetch ahead to hide memory latency
                if i + PREFETCH_AHEAD < neighbor_ids.len() {
                    let ahead_id = neighbor_ids[i + PREFETCH_AHEAD];
                    if !buf.visited.contains(&ahead_id) {
                        dc.prefetch(ahead_id);
                    }
                }

                let dist = dc.distance(query, neighbor_id)?;

                let should_add = if buf.results.len() < ef {
                    true
                } else if let Some(worst) = buf.results.peek() {
                    dist < worst.dist
                } else {
                    true
                };

                if should_add {
                    buf.candidates.push(Reverse(HeapEntry {
                        id: neighbor_id,
                        dist,
                    }));
                    buf.results.push(HeapEntry {
                        id: neighbor_id,
                        dist,
                    });
                    if buf.results.len() > ef {
                        buf.results.pop();
                    }
                }
            }
        }

        let result_vec: Vec<(PointId, f32)> =
            buf.results.drain().map(|e| (e.id, e.dist)).collect();
        Ok(result_vec)
    })
}

/// Greedy descent: find the single closest node at a given layer.
///
/// Shared by both search and insert paths.
/// Delegates to `greedy_closest_dc` via `StoreDc` adapter.
pub(crate) fn greedy_closest<D: DistanceMetric, N: NodeGraph>(
    nodes: &N,
    store: &dyn VectorStore,
    distance: &D,
    query: &[f32],
    cur_id: PointId,
    cur_dist: f32,
    layer: LayerId,
) -> v_hnsw_core::Result<(PointId, f32)> {
    let dc = StoreDc { store, metric: distance };
    greedy_closest_dc(nodes, &dc, query, cur_id, cur_dist, layer)
}

/// Search within a single layer using beam search.
///
/// Delegates to `search_layer_dc` via `StoreDc` adapter (with prefetch support).
pub(crate) fn search_layer<D: DistanceMetric, N: NodeGraph>(
    nodes: &N,
    store: &dyn VectorStore,
    distance: &D,
    query: &[f32],
    entry_points: &[(PointId, f32)],
    ef: usize,
    layer: LayerId,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    let dc = StoreDc { store, metric: distance };
    search_layer_dc(nodes, &dc, query, entry_points, ef, layer)
}
