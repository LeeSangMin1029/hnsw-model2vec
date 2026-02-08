//! HNSW search algorithms.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError};
use crate::distance::prefetch_read;

use crate::graph::HnswGraph;

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

/// Top-level HNSW search: traverse from entry point down to layer 0.
///
/// Returns up to `k` nearest neighbors sorted by ascending distance.
pub(crate) fn search<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    query: &[f32],
    k: usize,
    ef: usize,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    if query.len() != graph.config.dim {
        return Err(VhnswError::DimensionMismatch {
            expected: graph.config.dim,
            got: query.len(),
        });
    }

    let entry_id = match graph.entry_point {
        Some(id) => id,
        None => return Ok(Vec::new()),
    };

    let entry_vec = graph.store.get(entry_id)?;
    let mut cur_dist = graph.distance.distance(query, entry_vec);
    let mut cur_id = entry_id;

    // Greedy descent from top layer down to layer 1
    let max_layer = graph.max_layer;
    if max_layer > 0 {
        for layer_idx in (1..=max_layer).rev() {
            let layer = layer_idx as LayerId;
            let (new_id, new_dist) = greedy_closest(graph, query, cur_id, cur_dist, layer)?;
            cur_id = new_id;
            cur_dist = new_dist;
        }
    }

    // Search at layer 0 with full ef
    let effective_ef = ef.max(k);
    let entry_points = vec![(cur_id, cur_dist)];
    let results = search_layer(graph, query, &entry_points, effective_ef, 0)?;

    // Take top-k sorted by distance ascending
    let mut sorted = results;
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);

    Ok(sorted)
}

/// Greedy descent: find the single closest node at a given layer.
fn greedy_closest<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    query: &[f32],
    mut cur_id: PointId,
    mut cur_dist: f32,
    layer: LayerId,
) -> v_hnsw_core::Result<(PointId, f32)> {
    loop {
        let mut changed = false;
        let neighbors = match graph.nodes.get(&cur_id) {
            Some(node) => node.neighbors_at(layer),
            None => break,
        };

        for &neighbor_id in &neighbors {
            let node = match graph.nodes.get(&neighbor_id) {
                Some(n) => n,
                None => continue,
            };
            if node.deleted {
                continue;
            }
            let vec = graph.store.get(neighbor_id)?;
            let dist = graph.distance.distance(query, vec);
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

/// Search within a single layer using beam search.
///
/// Uses a min-heap for candidates (to explore closest first) and a max-heap
/// for results (to efficiently evict the farthest when full).
///
/// Applies software prefetch for the next neighbor's vector while computing
/// distance for the current one.
pub(crate) fn search_layer<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    query: &[f32],
    entry_points: &[(PointId, f32)],
    ef: usize,
    layer: LayerId,
) -> v_hnsw_core::Result<Vec<(PointId, f32)>> {
    let mut visited = HashSet::new();

    // Min-heap of candidates (closest first via Reverse)
    let mut candidates: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();

    // Max-heap of results (farthest first, capped at ef)
    let mut results: BinaryHeap<HeapEntry> = BinaryHeap::new();

    // Initialize with entry points
    for &(id, dist) in entry_points {
        if visited.insert(id) {
            candidates.push(Reverse(HeapEntry { id, dist }));
            // Only add non-deleted nodes to results
            let is_deleted = graph.nodes.get(&id).is_some_and(|n| n.deleted);
            if !is_deleted {
                results.push(HeapEntry { id, dist });
            }
        }
    }

    while let Some(Reverse(closest)) = candidates.pop() {
        // If the closest candidate is farther than the worst result and we have
        // enough results, we can stop.
        if let Some(worst) = results.peek()
            && closest.dist > worst.dist
            && results.len() >= ef
        {
            break;
        }

        // Get neighbor list for this candidate at the given layer
        let neighbor_ids: Vec<PointId> = match graph.nodes.get(&closest.id) {
            Some(node) => node.neighbors_at(layer),
            None => continue,
        };

        // Traverse neighbors with software prefetch
        for (i, &neighbor_id) in neighbor_ids.iter().enumerate() {
            if !visited.insert(neighbor_id) {
                continue;
            }

            // Check if this node is deleted
            let is_deleted = graph
                .nodes
                .get(&neighbor_id)
                .is_none_or(|n| n.deleted);
            if is_deleted {
                continue;
            }

            // Prefetch the NEXT neighbor's vector while we work on the current one
            if i + 1 < neighbor_ids.len() {
                let next_id = neighbor_ids[i + 1];
                if let Ok(next_vec) = graph.store.get(next_id) {
                    prefetch_read(next_vec.as_ptr());
                }
            }

            let vec = graph.store.get(neighbor_id)?;
            let dist = graph.distance.distance(query, vec);

            // Check if this neighbor should be added
            let should_add = if results.len() < ef {
                true
            } else if let Some(worst) = results.peek() {
                dist < worst.dist
            } else {
                true
            };

            if should_add {
                candidates.push(Reverse(HeapEntry {
                    id: neighbor_id,
                    dist,
                }));
                results.push(HeapEntry {
                    id: neighbor_id,
                    dist,
                });
                if results.len() > ef {
                    results.pop(); // Remove the farthest
                }
            }
        }
    }

    // Collect results
    let result_vec: Vec<(PointId, f32)> = results.into_iter().map(|e| (e.id, e.dist)).collect();
    Ok(result_vec)
}
