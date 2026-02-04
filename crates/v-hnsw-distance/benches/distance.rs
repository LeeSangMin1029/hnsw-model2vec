use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use v_hnsw_distance::{CosineDistance, DotProductDistance, L2Distance};

use v_hnsw_core::DistanceMetric;

const DIMENSIONS: &[usize] = &[128, 384, 768, 1536];

fn generate_vector(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((i + seed) as f32 * 0.1).sin())
        .collect()
}

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_distance");

    for &dim in DIMENSIONS {
        group.throughput(Throughput::Elements(dim as u64));

        let a = generate_vector(dim, 0);
        let b = generate_vector(dim, 1);
        let metric = L2Distance;

        group.bench_function(format!("dim_{dim}"), |bench| {
            bench.iter(|| {
                criterion::black_box(metric.distance(
                    criterion::black_box(&a),
                    criterion::black_box(&b),
                ))
            })
        });
    }

    group.finish();
}

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for &dim in DIMENSIONS {
        group.throughput(Throughput::Elements(dim as u64));

        let a = generate_vector(dim, 0);
        let b = generate_vector(dim, 1);
        let metric = CosineDistance;

        group.bench_function(format!("dim_{dim}"), |bench| {
            bench.iter(|| {
                criterion::black_box(metric.distance(
                    criterion::black_box(&a),
                    criterion::black_box(&b),
                ))
            })
        });
    }

    group.finish();
}

fn bench_dot_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product_distance");

    for &dim in DIMENSIONS {
        group.throughput(Throughput::Elements(dim as u64));

        let a = generate_vector(dim, 0);
        let b = generate_vector(dim, 1);
        let metric = DotProductDistance;

        group.bench_function(format!("dim_{dim}"), |bench| {
            bench.iter(|| {
                criterion::black_box(metric.distance(
                    criterion::black_box(&a),
                    criterion::black_box(&b),
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_l2, bench_cosine, bench_dot_product);
criterion_main!(benches);
