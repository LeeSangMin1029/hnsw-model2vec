//! HNSW insert algorithm.

use std::collections::HashMap;

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError};

use crate::config::HnswConfig;
use crate::graph::HnswGraph;
use crate::node::Node;
use crate::search::{greedy_closest, search_layer};
use crate::select::select_neighbors;

/// Xorshift64 PRNG. Returns the next pseudo-random u64.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Generate a random f64 in [0, 1) from the PRNG state.
fn random_float(state: &mut u64) -> f64 {
    (xorshift64(state) as f64) / (u64::MAX as f64)
}

/// Validate dimension and capacity before insertion.
fn validate_insert<D: DistanceMetric>(
    graph: &HnswGraph<D>,
    vector_len: usize,
) -> v_hnsw_core::Result<()> {
    if vector_len != graph.config.dim {
        return Err(VhnswError::DimensionMismatch {
            expected: graph.config.dim,
            got: vector_len,
        });
    }
    if graph.count >= graph.config.max_elements {
        return Err(VhnswError::IndexFull {
            capacity: graph.config.max_elements,
        });
    }
    Ok(())
}

/// Insert a point into the HNSW graph, storing the vector in the internal store.
pub(crate) fn insert<D: DistanceMetric>(
    graph: &mut HnswGraph<D>,
    id: PointId,
    vector: &[f32],
) -> v_hnsw_core::Result<()> {
    validate_insert(graph, vector.len())?;
    graph.store.insert(id, vector)?;

    insert_core(
        &mut graph.nodes, &graph.store, &graph.distance, &graph.config,
        &mut graph.entry_point, &mut graph.max_layer, &mut graph.count,
        &mut graph.rng_state, id,
    )
}

/// Insert a point using an external vector store (no vector copy).
///
/// The vector must already exist in `store` at the given `id`.
/// Used by buildindex to read directly from mmap without copying.
pub(crate) fn insert_with_store<D: DistanceMetric>(
    graph: &mut HnswGraph<D>,
    store: &dyn VectorStore,
    id: PointId,
) -> v_hnsw_core::Result<()> {
    let vector = store.get(id)?;
    validate_insert(graph, vector.len())?;

    insert_core(
        &mut graph.nodes, store, &graph.distance, &graph.config,
        &mut graph.entry_point, &mut graph.max_layer, &mut graph.count,
        &mut graph.rng_state, id,
    )
}

/// Core HNSW insertion with decomposed field references.
///
/// Takes individual graph fields to allow the caller to split borrows
/// (e.g. `&graph.store` + `&mut graph.nodes` from the same `HnswGraph`).
///
/// Algorithm:
/// 1. Assign a random layer
/// 2. Greedy descent from top to the assigned layer + 1
/// 3. For each layer from min(assigned, max) down to 0:
///    search, select neighbors, set bidirectional connections
/// 4. Update entry point if new node has higher layer
#[allow(clippy::too_many_arguments)]
fn insert_core<D: DistanceMetric>(
    nodes: &mut HashMap<PointId, Node>,
    store: &dyn VectorStore,
    distance: &D,
    config: &HnswConfig,
    entry_point: &mut Option<PointId>,
    max_layer: &mut LayerId,
    count: &mut usize,
    rng_state: &mut u64,
    id: PointId,
) -> v_hnsw_core::Result<()> {
    let vector = store.get(id)?;

    // Assign random layer: floor(-ln(rand) * ml), capped at 255
    let r = random_float(rng_state).max(1e-15);
    let new_layer_f = (-r.ln() * config.ml).floor();
    let new_layer: LayerId = if new_layer_f >= 255.0 { 255 } else { new_layer_f as u8 };

    let node = Node::new(id, new_layer);
    nodes.insert(id, node);

    // First node: set as entry point and return
    if entry_point.is_none() {
        *entry_point = Some(id);
        *max_layer = new_layer;
        *count += 1;
        return Ok(());
    }

    let entry_id = entry_point.ok_or(VhnswError::PointNotFound(0))?;
    let entry_vec = store.get(entry_id)?;
    let mut cur_dist = distance.distance(vector, entry_vec);
    let mut cur_id = entry_id;
    let current_max_layer = *max_layer;

    // Phase 1: Greedy descent from top layer down to new_layer + 1
    if current_max_layer > new_layer {
        for layer_idx in ((new_layer as u16 + 1)..=(current_max_layer as u16)).rev() {
            let layer = layer_idx as LayerId;
            let (next_id, next_dist) = greedy_closest(
                nodes, store, distance, vector, cur_id, cur_dist, layer,
            )?;
            cur_id = next_id;
            cur_dist = next_dist;
        }
    }

    // Phase 2: Insert at each layer from min(new_layer, current_max_layer) down to 0
    let top_insert_layer = new_layer.min(current_max_layer);
    let mut ep = vec![(cur_id, cur_dist)];

    for layer_idx in (0..=top_insert_layer as u16).rev() {
        let layer = layer_idx as LayerId;
        let max_neighbors = if layer == 0 { config.m0 } else { config.m };

        let candidates = search_layer(
            nodes, store, distance, vector, &ep, config.ef_construction, layer,
        )?;

        let selected = select_neighbors(&candidates, max_neighbors);

        prune_backward_connections(nodes, store, distance, id, &selected, layer, max_neighbors);

        if let Some(node) = nodes.get_mut(&id) {
            node.set_neighbors(layer, selected);
        }

        ep = candidates;
    }

    // Phase 3: Update entry point if new node has a higher layer
    if new_layer > current_max_layer {
        *entry_point = Some(id);
        *max_layer = new_layer;
    }

    *count += 1;
    Ok(())
}

/// Set backward connections from selected neighbors to the new node, pruning if needed.
fn prune_backward_connections<D: DistanceMetric>(
    nodes: &mut HashMap<PointId, Node>,
    store: &dyn VectorStore,
    distance: &D,
    new_id: PointId,
    selected: &[PointId],
    layer: LayerId,
    max_neighbors: usize,
) {
    for &neighbor_id in selected {
        let needs_pruning = if let Some(neighbor_node) = nodes.get_mut(&neighbor_id) {
            neighbor_node.add_neighbor(layer, new_id);
            neighbor_node.neighbors_at(layer).len() > max_neighbors
        } else {
            false
        };

        if needs_pruning {
            let neighbor_list = match nodes.get(&neighbor_id) {
                Some(n) => n.neighbors_at(layer).to_vec(),
                None => continue,
            };

            if let Ok(neighbor_vec) = store.get(neighbor_id) {
                let mut scored: Vec<(PointId, f32)> = Vec::with_capacity(neighbor_list.len());
                for &nid in &neighbor_list {
                    if let Ok(nv) = store.get(nid) {
                        let d = distance.distance(neighbor_vec, nv);
                        scored.push((nid, d));
                    }
                }
                let pruned = select_neighbors(&scored, max_neighbors);
                if let Some(nn) = nodes.get_mut(&neighbor_id) {
                    nn.set_neighbors(layer, pruned);
                }
            }
        }
    }
}
