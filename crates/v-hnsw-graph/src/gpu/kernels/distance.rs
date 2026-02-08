//! Batch L2 squared distance kernel for CubeCL.
//!
//! Given a single query vector Q of length `dim` and N database vectors
//! (flattened into `N * dim` floats), computes the L2 squared distance
//! from Q to every database vector in parallel on the GPU.

#![allow(unsafe_code)]

use cubecl::prelude::*;

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

/// Each workgroup item handles one database vector.
///
/// `query`    -- `[dim]` floats
/// `database` -- `[n_vectors * dim]` floats, row-major
/// `output`   -- `[n_vectors]` floats, one distance per vector
#[cube(launch_unchecked)]
fn batch_l2_squared_kernel<F: Float>(
    query: &Array<Line<F>>,
    database: &Array<Line<F>>,
    output: &mut Array<Line<F>>,
    #[comptime] dim: usize,
) {
    let vec_idx = ABSOLUTE_POS;
    if vec_idx >= output.len() {
        terminate!();
    }

    let base = vec_idx * dim;
    let mut sum = Line::new(F::new(0.0));

    for d in 0..dim {
        let diff = query[d] - database[base + d];
        sum += diff * diff;
    }

    output[vec_idx] = sum;
}

// ---------------------------------------------------------------------------
// Launch helper
// ---------------------------------------------------------------------------

/// Launch the batch L2 squared distance kernel on a CubeCL [`Runtime`].
///
/// Returns one f32 distance per database vector.
#[cfg_attr(
    not(any(feature = "cuda", feature = "wgpu")),
    allow(dead_code)
)]
pub(crate) fn launch_batch_l2_squared<R: Runtime>(
    client: &ComputeClient<R>,
    query: &[f32],
    database: &[f32],
    n_vectors: usize,
    dim: usize,
) -> v_hnsw_core::Result<Vec<f32>> {
    if n_vectors == 0 {
        return Ok(Vec::new());
    }

    // Upload data to device memory.
    let query_handle = client.create_from_slice(f32::as_bytes(query));
    let db_handle = client.create_from_slice(f32::as_bytes(database));
    let output_handle = client.empty(n_vectors * core::mem::size_of::<f32>());

    // Choose a reasonable workgroup size capped at 256.
    let workgroup_size = (n_vectors as u32).min(256);
    let num_workgroups = (n_vectors as u32).div_ceil(workgroup_size);

    unsafe {
        batch_l2_squared_kernel::launch_unchecked::<f32, R>(
            client,
            CubeCount::Static(num_workgroups, 1, 1),
            CubeDim::new_1d(workgroup_size),
            ArrayArg::from_raw_parts::<f32>(&query_handle, dim, 1),
            ArrayArg::from_raw_parts::<f32>(&db_handle, n_vectors * dim, 1),
            ArrayArg::from_raw_parts::<f32>(&output_handle, n_vectors, 1),
            dim,
        )
    }
    .map_err(super::super::error::gpu_err)?;

    // Read back results.
    let bytes = client.read_one(output_handle);
    Ok(f32::from_bytes(&bytes).to_vec())
}

