//! HNSW graph node representation.

use v_hnsw_core::{LayerId, PointId};

use crate::delta::DeltaNeighbors;

/// A node in the HNSW graph.
///
/// Each node stores its point ID, the maximum layer it exists on,
/// its neighbor lists per layer, and a deletion tombstone flag.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// The point identifier for this node.
    #[allow(dead_code)]
    pub id: PointId,
    /// The highest layer this node exists on.
    #[allow(dead_code)]
    pub max_layer: LayerId,
    /// Delta-encoded neighbor lists indexed by layer.
    pub neighbors: Vec<DeltaNeighbors>,
    /// Tombstone flag for lazy deletion.
    pub deleted: bool,
}

impl Node {
    /// Create a new node at the given layer with empty neighbor lists.
    pub fn new(id: PointId, max_layer: LayerId) -> Self {
        let layer_count = max_layer as usize + 1;
        let neighbors = vec![DeltaNeighbors::new(); layer_count];
        Self {
            id,
            max_layer,
            neighbors,
            deleted: false,
        }
    }

    /// Get the neighbor list at the given layer.
    ///
    /// Returns an empty vec if the layer is above this node's max layer.
    pub fn neighbors_at(&self, layer: LayerId) -> Vec<PointId> {
        let idx = layer as usize;
        if idx < self.neighbors.len() {
            self.neighbors[idx].decode()
        } else {
            Vec::new()
        }
    }

    /// Replace the entire neighbor list at the given layer.
    ///
    /// Does nothing if the layer is above this node's max layer.
    pub fn set_neighbors(&mut self, layer: LayerId, ids: Vec<PointId>) {
        let idx = layer as usize;
        if idx < self.neighbors.len() {
            self.neighbors[idx] = DeltaNeighbors::from_ids(&ids);
        }
    }

    /// Append a neighbor to the given layer.
    ///
    /// Does nothing if the layer is above this node's max layer.
    pub fn add_neighbor(&mut self, layer: LayerId, id: PointId) {
        let idx = layer as usize;
        if idx < self.neighbors.len() {
            self.neighbors[idx].push(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node::new(42, 3);
        assert_eq!(node.id, 42);
        assert_eq!(node.max_layer, 3);
        assert_eq!(node.neighbors.len(), 4);
        assert!(!node.deleted);
        for layer in 0..=3 {
            assert!(node.neighbors_at(layer).is_empty());
        }
    }

    #[test]
    fn test_neighbors_at_out_of_range() {
        let node = Node::new(1, 2);
        assert!(node.neighbors_at(5).is_empty());
    }

    #[test]
    fn test_set_neighbors() {
        let mut node = Node::new(1, 1);
        node.set_neighbors(0, vec![10, 20, 30]);
        let n = node.neighbors_at(0);
        assert_eq!(n, vec![10, 20, 30]);
        assert!(node.neighbors_at(1).is_empty());
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = Node::new(1, 0);
        node.add_neighbor(0, 100);
        node.add_neighbor(0, 200);
        let n = node.neighbors_at(0);
        assert_eq!(n, vec![100, 200]);
    }

    #[test]
    fn test_add_neighbor_out_of_range() {
        let mut node = Node::new(1, 0);
        node.add_neighbor(5, 100); // should be a no-op
        assert!(node.neighbors_at(0).is_empty());
    }
}
