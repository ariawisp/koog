// Unified GPU backend with zero-cost compile-time dispatch

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod streamlined;  // Metal implementation used by Unified backend
pub mod unified;      // Zero-cost unified backend

// CUDA backend for cross-platform support (used internally by Unified)
#[cfg(feature = "cuda")]
pub mod cuda;

// Primary exports
pub use unified::{UnifiedBackend, create_unified_backend};
// Types used by Unified backend
pub use streamlined::{EssentialDeviceInfo, StreamlinedMetrics};

// BREAKING: Do not re-export legacy trait or factory

// MIGRATION PATH:
// 1. Replace Arc<dyn StreamlinedBackendTrait> with UnifiedBackend
// 2. Replace create_streamlined_backend() with create_unified_backend()
// 3. Remove trait object allocations for zero-cost performance
