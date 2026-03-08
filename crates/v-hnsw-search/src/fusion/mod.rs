//! Result fusion algorithms.
//!
//! Combines ranked lists from multiple retrieval methods.

mod convex;

#[cfg(test)]
mod tests;

pub use convex::ConvexFusion;
