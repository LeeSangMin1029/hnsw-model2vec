//! HNSW graph node representation.

use v_hnsw_core::{LayerId, PointId};

use crate::delta::DeltaNeighbors;

/// A node in the HNSW graph.
///
/// Each node stores its point ID, the maximum layer it exists on,
/// its neighbor lists per layer, and a deletion tombstone flag.
///
/// At runtime, neighbors are stored as plain `Vec<PointId>` per layer
/// for O(1) access. Delta encoding is only applied during serialization.
#[derive(Debug, Clone)]
pub(crate) struct Node {
    /// The point identifier for this node.
    pub id: PointId,
    /// The highest layer this node exists on.
    pub max_layer: LayerId,
    /// Neighbor lists indexed by layer (decoded, for fast runtime access).
    pub neighbors: Vec<Vec<PointId>>,
    /// Tombstone flag for lazy deletion.
    pub deleted: bool,
}

/// Serialized form of a node, using delta-encoded neighbor lists.
#[derive(Debug, Clone, bincode::Encode, bincode::Decode)]
pub(crate) struct NodeSerialized {
    pub id: PointId,
    pub max_layer: LayerId,
    pub neighbors: Vec<DeltaNeighbors>,
    pub deleted: bool,
}

impl From<&Node> for NodeSerialized {
    fn from(node: &Node) -> Self {
        Self {
            id: node.id,
            max_layer: node.max_layer,
            neighbors: node
                .neighbors
                .iter()
                .map(|ids| DeltaNeighbors::from_ids(ids))
                .collect(),
            deleted: node.deleted,
        }
    }
}

impl From<NodeSerialized> for Node {
    fn from(s: NodeSerialized) -> Self {
        Self {
            id: s.id,
            max_layer: s.max_layer,
            neighbors: s.neighbors.iter().map(|dn| dn.decode()).collect(),
            deleted: s.deleted,
        }
    }
}

impl Node {
    /// Create a new node at the given layer with empty neighbor lists.
    pub fn new(id: PointId, max_layer: LayerId) -> Self {
        let layer_count = max_layer as usize + 1;
        let neighbors = vec![Vec::new(); layer_count];
        Self {
            id,
            max_layer,
            neighbors,
            deleted: false,
        }
    }

    /// Get the neighbor list at the given layer.
    ///
    /// Returns an empty slice if the layer is above this node's max layer.
    pub fn neighbors_at(&self, layer: LayerId) -> &[PointId] {
        let idx = layer as usize;
        if idx < self.neighbors.len() {
            &self.neighbors[idx]
        } else {
            &[]
        }
    }

    /// Replace the entire neighbor list at the given layer.
    ///
    /// Does nothing if the layer is above this node's max layer.
    pub fn set_neighbors(&mut self, layer: LayerId, ids: Vec<PointId>) {
        let idx = layer as usize;
        if idx < self.neighbors.len() {
            self.neighbors[idx] = ids;
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
        assert_eq!(n, &[10, 20, 30]);
        assert!(node.neighbors_at(1).is_empty());
    }

    #[test]
    fn test_add_neighbor() {
        let mut node = Node::new(1, 0);
        node.add_neighbor(0, 100);
        node.add_neighbor(0, 200);
        let n = node.neighbors_at(0);
        assert_eq!(n, &[100, 200]);
    }

    #[test]
    fn test_add_neighbor_out_of_range() {
        let mut node = Node::new(1, 0);
        node.add_neighbor(5, 100); // should be a no-op
        assert!(node.neighbors_at(0).is_empty());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut node = Node::new(42, 2);
        node.set_neighbors(0, vec![10, 20, 30]);
        node.set_neighbors(1, vec![100, 200]);

        let serialized = NodeSerialized::from(&node);
        let restored = Node::from(serialized);

        assert_eq!(restored.id, node.id);
        assert_eq!(restored.max_layer, node.max_layer);
        assert_eq!(restored.deleted, node.deleted);
        // Delta encoding sorts, so compare sorted
        let mut n0 = restored.neighbors_at(0).to_vec();
        n0.sort();
        assert_eq!(n0, vec![10, 20, 30]);
        let mut n1 = restored.neighbors_at(1).to_vec();
        n1.sort();
        assert_eq!(n1, vec![100, 200]);
    }
}
