// Unified buffer abstractions for cross-platform GPU operations

pub mod unified_gpu_buffer;

// PRIMARY EXPORTS: Single unified buffer system
pub use unified_gpu_buffer::{
    GpuBuffer,
    UnifiedGpuBufferPool,
    UnifiedBufferPoolStats,
};

// MIGRATION NOTES:
// 1. Replace OptimizedBuffer with GpuBuffer
// 2. Replace OptimizedBufferPool with UnifiedGpuBufferPool  
// 3. Replace BufferView with direct GpuBuffer operations
// 4. All buffer operations now have zero-cost dispatch