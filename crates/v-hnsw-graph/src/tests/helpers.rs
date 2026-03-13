/// Generate a deterministic test vector for the given point id and dimension.
pub fn test_vector(point_id: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| (point_id as f32 * 0.1 + j as f32 * 0.3).sin())
        .collect()
}
