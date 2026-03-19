use std::hash::{DefaultHasher, Hasher};

fn main() {
    // Hash graph.rs so that CACHE_HASH changes whenever resolve logic changes.
    // This auto-invalidates the graph cache without manual CACHE_VERSION bumps.
    let graph_src = std::fs::read("src/graph.rs").expect("cannot read src/graph.rs");
    let mut hasher = DefaultHasher::new();
    hasher.write(&graph_src);
    let hash = hasher.finish();
    println!("cargo::rustc-env=GRAPH_SOURCE_HASH={hash:016x}");
    println!("cargo::rerun-if-changed=src/graph.rs");
}
