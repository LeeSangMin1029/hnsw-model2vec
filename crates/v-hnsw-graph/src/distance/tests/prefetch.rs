use crate::distance::prefetch::{prefetch_read, prefetch_vector};

#[test]
fn prefetch_read_valid_pointer() {
    let value: u64 = 42;
    let ptr = &value as *const u64;
    // Should not panic — prefetch is a CPU hint only
    prefetch_read(ptr);
}

#[test]
fn prefetch_read_stack_array() {
    let arr = [1.0f32; 64];
    let ptr = arr.as_ptr();
    prefetch_read(ptr);
}

#[test]
fn prefetch_vector_empty_slice() {
    let data: &[f32] = &[];
    // Should return early without panic
    prefetch_vector(data);
}

#[test]
fn prefetch_vector_small_slice() {
    // 4 floats = 16 bytes, less than one cache line
    let data = [1.0f32, 2.0, 3.0, 4.0];
    prefetch_vector(&data);
}

#[test]
fn prefetch_vector_one_cache_line() {
    // 16 floats = 64 bytes = exactly one cache line
    let data = [0.0f32; 16];
    prefetch_vector(&data);
}

#[test]
fn prefetch_vector_multiple_cache_lines() {
    // 128 floats = 512 bytes = 8 cache lines, but only 4 are prefetched
    let data = [1.0f32; 128];
    prefetch_vector(&data);
}

#[test]
fn prefetch_vector_large_slice() {
    // 1024 floats = 4096 bytes, still limited to 4 cache lines prefetch
    let data = vec![0.5f32; 1024];
    prefetch_vector(&data);
}

#[test]
fn prefetch_vector_typical_embedding_dim() {
    // 256 dimensions, typical model2vec output
    let data = vec![0.1f32; 256];
    prefetch_vector(&data);
}
