//! HNSW insert algorithm.

use v_hnsw_core::{DistanceMetric, LayerId, PointId, VectorStore, VhnswError};

use crate::graph::HnswGraph;
use crate::node::Node;
use crate::search::search_layer;
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

/// Insert a point into the HNSW graph.
///
/// Follows the standard HNSW insertion algorithm:
/// 1. Store the vector
/// 2. Assign a random layer
/// 3. Greedy descent from top to the assigned layer + 1
/// 4. For each layer from min(assigned, max) down to 0:
///    search, select neighbors, set bidirectional connections
/// 5. Update entry point if new node has higher layer
pub(crate) fn insert<D: DistanceMetric>(
    graph: &mut HnswGraph<D>,
    id: PointId,
    vector: &[f32],
) -> v_hnsw_core::Result<()> {
    // Validate dimension
    if vector.len() != graph.config.dim {
        return Err(VhnswError::DimensionMismatch {
            expected: graph.config.dim,
            got: vector.len(),
        });
    }

    // Check capacity
    if graph.count >= graph.config.max_elements {
        return Err(VhnswError::IndexFull {
            capacity: graph.config.max_elements,
        });
    }

    // Store the vector
    graph.store.insert(id, vector)?;

    // Assign random layer: floor(-ln(rand) * ml), capped at 255
    let r = random_float(&mut graph.rng_state);
    // Clamp r to avoid ln(0). Use a small epsilon.
    let r = r.max(1e-15);
    let new_layer_f = (-r.ln() * graph.config.ml).floor();
    let new_layer: LayerId = if new_layer_f >= 255.0 {
        255
    } else {
        new_layer_f as u8
    };

    // Create the node
    let node = Node::new(id, new_layer);
    graph.nodes.insert(id, node);

    // If this is the first node, set as entry point and return
    if graph.entry_point.is_none() {
        graph.entry_point = Some(id);
        graph.max_layer = new_layer;
        graph.count += 1;
        return Ok(());
    }

    // Get current entry point
    let entry_id = graph
        .entry_point
        .ok_or(VhnswError::PointNotFound(0))?;

    let entry_vec = graph.store.get(entry_id)?;
    let mut cur_dist = graph.distance.distance(vector, entry_vec);
    let mut cur_id = entry_id;

    let current_max_layer = graph.max_layer;

    // Phase 1: Greedy descent from top layer down to new_layer + 1
    if current_max_layer > new_layer {
        for layer_idx in ((new_layer as u16 + 1)..=(current_max_layer as u16)).rev() {
            let layer = layer_idx as LayerId;
            let (next_id, next_dist) = greedy_closest_insert(graph, vector, cur_id, cur_dist, layer)?;
            cur_id = next_id;
            cur_dist = next_dist;
        }
    }

    // Phase 2: Insert at each layer from min(new_layer, current_max_layer) down to 0
    let top_insert_layer = new_layer.min(current_max_layer);

    // Entry points for search_layer
    let mut ep = vec![(cur_id, cur_dist)];

    for layer_idx in (0..=top_insert_layer as u16).rev() {
        let layer = layer_idx as LayerId;

        // Determine max neighbors for this layer
        let max_neighbors = if layer == 0 {
            graph.config.m0
        } else {
            graph.config.m
        };

        // Search for nearest neighbors at this layer
        let candidates = search_layer(graph, vector, &ep, graph.config.ef_construction, layer)?;

        // Select best neighbors
        let selected = select_neighbors(&candidates, max_neighbors);

        // Set forward connections from the new node
        if let Some(node) = graph.nodes.get_mut(&id) {
            node.set_neighbors(layer, selected.clone());
        }

        // Set backward connections and prune if needed
        for &neighbor_id in &selected {
            if let Some(neighbor_node) = graph.nodes.get_mut(&neighbor_id) {
                neighbor_node.add_neighbor(layer, id);

                // Check if neighbor exceeds max connections
                let neighbor_count = neighbor_node.neighbors_at(layer).len();
                if neighbor_count > max_neighbors {
                    // Need to prune: compute distances from neighbor to all its neighbors
                    let neighbor_list: Vec<PointId> = neighbor_node.neighbors_at(layer);
                    let neighbor_vec_result = graph.store.get(neighbor_id);
                    if let Ok(neighbor_vec) = neighbor_vec_result {
                        let neighbor_vec_owned: Vec<f32> = neighbor_vec.to_vec();
                        let mut scored: Vec<(PointId, f32)> = Vec::with_capacity(neighbor_list.len());
                        for &nid in &neighbor_list {
                            if let Ok(nv) = graph.store.get(nid) {
                                let d = graph.distance.distance(&neighbor_vec_owned, nv);
                                scored.push((nid, d));
                            }
                        }
                        let pruned = select_neighbors(&scored, max_neighbors);
                        if let Some(nn) = graph.nodes.get_mut(&neighbor_id) {
                            nn.set_neighbors(layer, pruned);
                        }
                    }
                }
            }
        }

        // Update entry points for next (lower) layer: use the candidates found
        ep = candidates;
    }

    // Phase 3: Update entry point if new node has a higher layer
    if new_layer > current_max_layer {
        graph.entry_point = Some(id);
        graph.max_layer = new_layer;
    }

    graph.count += 1;
    Ok(())
}

/// Greedy search for the single closest node at a given layer (used during insert).
fn greedy_closest_insert<D: DistanceMetric>(
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
