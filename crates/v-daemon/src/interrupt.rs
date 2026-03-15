//! Ctrl+C interrupt handling — delegates to v-hnsw-core.

pub use v_hnsw_core::interrupt::is_interrupted;

/// Install Ctrl+C handler that sets the interrupt flag.
pub fn install_handler() {
    if let Err(e) = ctrlc::set_handler(move || {
        v_hnsw_core::interrupt::set_interrupted();
        eprintln!("\nInterrupted. Cleaning up...");
    }) {
        eprintln!("Warning: Failed to set Ctrl+C handler: {e}");
    }
}
