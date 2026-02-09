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

/// Prefetch multiple cache lines of a vector for reading.
///
/// dim=256 → 1024 bytes → 16 cache lines (64 bytes each).
/// Prefetches the first few cache lines to cover the hot portion.
#[inline]
pub fn prefetch_vector(data: &[f32]) {
    if data.is_empty() {
        return;
    }
    let ptr = data.as_ptr() as *const u8;
    let byte_len = data.len() * 4;
    // Prefetch up to 4 cache lines (256 bytes, covers first 64 f32)
    let lines = (byte_len / 64).min(4);
    for i in 0..lines {
        prefetch_read(unsafe { ptr.add(i * 64) } as *const f32);
    }
}
