use crate::node::{Node, NodeSerialized};

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

#[test]
fn test_node_layer_zero() {
    let node = Node::new(1, 0);
    assert_eq!(node.max_layer, 0);
    assert_eq!(node.neighbors.len(), 1); // only layer 0
    assert!(node.neighbors_at(0).is_empty());
    assert!(node.neighbors_at(1).is_empty()); // out of range returns empty
}

#[test]
fn test_set_neighbors_out_of_range() {
    let mut node = Node::new(1, 1);
    node.set_neighbors(5, vec![10, 20]); // should be a no-op
    assert!(node.neighbors_at(0).is_empty());
    assert!(node.neighbors_at(1).is_empty());
}

#[test]
fn test_set_neighbors_overwrites() {
    let mut node = Node::new(1, 0);
    node.set_neighbors(0, vec![10, 20]);
    assert_eq!(node.neighbors_at(0), &[10, 20]);
    node.set_neighbors(0, vec![30, 40, 50]);
    assert_eq!(node.neighbors_at(0), &[30, 40, 50]);
}

#[test]
fn test_serialization_roundtrip_deleted() {
    let mut node = Node::new(99, 1);
    node.set_neighbors(0, vec![1, 2, 3]);
    node.deleted = true;

    let serialized = NodeSerialized::from(&node);
    let restored = Node::from(serialized);

    assert_eq!(restored.id, 99);
    assert!(restored.deleted);
}

#[test]
fn test_serialization_roundtrip_empty_neighbors() {
    let node = Node::new(5, 3);
    let serialized = NodeSerialized::from(&node);
    let restored = Node::from(serialized);

    assert_eq!(restored.id, 5);
    assert_eq!(restored.max_layer, 3);
    for layer in 0..=3 {
        assert!(restored.neighbors_at(layer).is_empty());
    }
}

#[test]
fn test_node_high_layer() {
    let node = Node::new(1, 10);
    assert_eq!(node.max_layer, 10);
    assert_eq!(node.neighbors.len(), 11); // layers 0..=10
    assert!(node.neighbors_at(10).is_empty());
    assert!(node.neighbors_at(11).is_empty()); // out of range
}

#[test]
fn test_add_neighbor_multiple_layers() {
    let mut node = Node::new(1, 2);
    node.add_neighbor(0, 10);
    node.add_neighbor(1, 20);
    node.add_neighbor(2, 30);
    assert_eq!(node.neighbors_at(0), &[10]);
    assert_eq!(node.neighbors_at(1), &[20]);
    assert_eq!(node.neighbors_at(2), &[30]);
}
