//! Software prefetch helpers for reducing cache misses during HNSW traversal.
#![allow(unsafe_code)]

/// Prefetch a memory address into L1 cache for reading.
///
/// This is a performance hint; it has no effect on correctness.
/// Used during HNSW graph traversal to prefetch the next neighbor's vector
/// while computing distance for the current neighbor.
#[inline]
pub fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: Prefetch is always safe; it's a performance hint only.
        // Invalid addresses are silently ignored by the CPU.
        unsafe {
            std::arch::x86_64::_mm_prefetch(ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // On ARM, use the prefetch intrinsic
        // NOTE: std::arch::aarch64 prefetch requires nightly.
        // On stable ARM, this is a no-op; the compiler may auto-prefetch.
        let _ = ptr;
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let _ = ptr;
    }
}

/// Prefetch a slice of f32 data for reading.
#[inline]
#[allow(dead_code)]
pub fn prefetch_vector(data: &[f32]) {
    if !data.is_empty() {
        prefetch_read(data.as_ptr());
    }
}
