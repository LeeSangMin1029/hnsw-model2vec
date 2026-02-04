//! Search result types.

use v_hnsw_core::PointId;

/// A single search result with score and optional associated data.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Unique point identifier.
    pub id: PointId,
    /// Search score (lower = more similar for distance metrics).
    pub score: f32,
    /// The vector data, if requested.
    pub vector: Option<Vec<f32>>,
    /// Associated text, if stored.
    pub text: Option<String>,
}

impl SearchResult {
    /// Create a new search result with just id and score.
    pub fn new(id: PointId, score: f32) -> Self {
        Self {
            id,
            score,
            vector: None,
            text: None,
        }
    }

    /// Create a search result with all fields.
    pub fn with_data(
        id: PointId,
        score: f32,
        vector: Option<Vec<f32>>,
        text: Option<String>,
    ) -> Self {
        Self {
            id,
            score,
            vector,
            text,
        }
    }
}

impl From<(PointId, f32)> for SearchResult {
    fn from((id, score): (PointId, f32)) -> Self {
        SearchResult::new(id, score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result_new() {
        let result = SearchResult::new(42, 0.5);
        assert_eq!(result.id, 42);
        assert!((result.score - 0.5).abs() < f32::EPSILON);
        assert!(result.vector.is_none());
        assert!(result.text.is_none());
    }

    #[test]
    fn test_search_result_with_data() {
        let vec = vec![1.0, 2.0, 3.0];
        let text = "hello world".to_string();
        let result = SearchResult::with_data(42, 0.5, Some(vec.clone()), Some(text.clone()));
        assert_eq!(result.id, 42);
        assert_eq!(result.vector, Some(vec));
        assert_eq!(result.text, Some(text));
    }

    #[test]
    fn test_from_tuple() {
        let result: SearchResult = (123, 0.75).into();
        assert_eq!(result.id, 123);
        assert!((result.score - 0.75).abs() < f32::EPSILON);
    }
}
