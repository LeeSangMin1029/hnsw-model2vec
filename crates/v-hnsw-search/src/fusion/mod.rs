//! Result fusion algorithms.
//!
//! Combines ranked lists from multiple retrieval methods.

mod convex;
mod rrf;

pub use convex::ConvexFusion;
pub use rrf::RrfFusion;
