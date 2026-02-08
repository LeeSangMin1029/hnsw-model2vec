use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use v_hnsw_core::{DistanceMetric, PointId, VectorIndex};
use v_hnsw_graph::{HnswConfig, HnswGraph, L2Distance};

/// Generate a reproducible vector using a simple seed-based function
fn generate_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i + seed) as f32 * 0.1).sin())
        .collect()
}

/// Build an index with N vectors of given dimension
fn build_index(n: usize, dim: usize, seed: u64) -> HnswGraph<L2Distance> {
    let config = HnswConfig::builder()
        .dim(dim)
        .max_elements(n)
        .m(16)
        .ef_construction(100)
        .build()
        .expect("valid config");

    let mut graph = HnswGraph::with_seed(config, L2Distance, seed);

    for i in 0..n {
        let vector = generate_vector(dim, i);
        graph.insert(i as PointId, &vector).expect("insert failed");
    }

    graph
}

/// Benchmark single query latency at different scales
fn bench_search_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_latency");

    let configs = [
        (1000, 128, "1K_dim128"),
        (1000, 384, "1K_dim384"),
        (10000, 128, "10K_dim128"),
        (10000, 384, "10K_dim384"),
        (100000, 128, "100K_dim128"),
    ];

    for (size, dim, name) in configs {
        let graph = build_index(size, dim, 42);
        let query = generate_vector(dim, size + 1);

        group.bench_with_input(BenchmarkId::from_parameter(name), &(graph, query), |b, (g, q)| {
            b.iter(|| {
                let results = g.search(black_box(q), black_box(10), black_box(100));
                black_box(results)
            })
        });
    }

    group.finish();
}

/// Benchmark queries per second (throughput)
fn bench_search_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_qps");
    group.throughput(Throughput::Elements(1000));

    let configs = [(10000, 128, "10K_dim128"), (10000, 384, "10K_dim384")];

    for (size, dim, name) in configs {
        let graph = build_index(size, dim, 42);

        // Generate 1000 queries
        let queries: Vec<Vec<f32>> = (0..1000).map(|i| generate_vector(dim, size + i)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(graph, queries),
            |b, (g, qs)| {
                b.iter(|| {
                    for q in qs {
                        let _ = black_box(g.search(black_box(q), black_box(10), black_box(100)));
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark index building time
fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");

    let configs = [(1000, 128, "1K_dim128"), (10000, 128, "10K_dim128")];

    for (size, dim, name) in configs {
        group.bench_with_input(BenchmarkId::from_parameter(name), &(size, dim), |b, &(n, d)| {
            b.iter(|| {
                let graph = build_index(black_box(n), black_box(d), 42);
                black_box(graph);
            })
        });
    }

    group.finish();
}

/// Benchmark recall@10 vs brute-force ground truth
fn bench_recall(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_recall");

    let size = 10000;
    let dim = 128;
    let num_queries = 100;

    // Build index
    let graph = build_index(size, dim, 42);

    // Generate queries
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|i| generate_vector(dim, size + i))
        .collect();

    // Compute ground truth using brute force
    let vectors: Vec<Vec<f32>> = (0..size).map(|i| generate_vector(dim, i)).collect();

    let ground_truth: Vec<Vec<PointId>> = queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(PointId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dist = L2Distance.distance(query, v);
                    (i as PointId, dist)
                })
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(10).map(|(id, _)| *id).collect()
        })
        .collect();

    // Benchmark different ef_search values
    for ef in [50, 100, 200] {
        group.bench_with_input(
            BenchmarkId::new("ef_search", ef),
            &ef,
            |b, &ef_val| {
                b.iter(|| {
                    let mut total_recall = 0.0;
                    for (i, query) in queries.iter().enumerate() {
                        let results = graph.search(black_box(query), black_box(10), black_box(ef_val));
                        if let Ok(results) = results {
                            let retrieved: Vec<PointId> = results.iter().map(|(id, _)| *id).collect();
                            let true_neighbors = &ground_truth[i];
                            let matches = retrieved
                                .iter()
                                .filter(|id| true_neighbors.contains(id))
                                .count();
                            total_recall += matches as f32 / 10.0;
                        }
                    }
                    black_box(total_recall / queries.len() as f32);
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_search_latency,
    bench_search_throughput,
    bench_insert,
    bench_recall
);
criterion_main!(benches);
