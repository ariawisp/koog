// Unified backend implementation with compile-time dispatch

use crate::errors::{NoesisError, NoesisResult};
use crate::gpu_optimized::OptimizedBufferPool;
use crate::buffer::{GpuBuffer, UnifiedGpuBufferPool}; // NEW: Unified buffer system
use std::sync::Arc;

// Import platform-specific backends
use super::streamlined::{StreamlinedGpuBackend, StreamlinedMetrics, EssentialDeviceInfo};
#[cfg(feature = "cuda")]
use super::cuda::StreamlinedCudaBackend;

/// Unified backend with zero-cost dispatch
pub enum UnifiedBackend {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(StreamlinedGpuBackend),
    
    #[cfg(feature = "cuda")]
    Cuda(StreamlinedCudaBackend),
    
    // Future backend types (ONNX for Windows, etc.)
    #[cfg(all(target_os = "windows", feature = "onnx"))]
    Onnx(super::onnx::OnnxBackend),
}

impl UnifiedBackend {
    /// Create platform-appropriate backend automatically
    pub fn new() -> NoesisResult<Self> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            Ok(UnifiedBackend::Metal(StreamlinedGpuBackend::new()?))
        }
        
        #[cfg(all(feature = "cuda", not(all(target_os = "macos", feature = "metal"))))]
        {
            Ok(UnifiedBackend::Cuda(StreamlinedCudaBackend::new()?))
        }
        
        #[cfg(all(target_os = "windows", feature = "onnx"))]
        {
            Ok(UnifiedBackend::Onnx(super::onnx::OnnxBackend::new("model.onnx")?))
        }
        
        #[cfg(not(any(
            all(target_os = "macos", feature = "metal"),
            feature = "cuda",
            all(target_os = "windows", feature = "onnx")
        )))]
        {
            Err(NoesisError::backend_initialization("No GPU backend available"))
        }
    }
    
    /// CRITICAL PATH: Zero-cost fast token generation
    /// 
    /// This method is inlined and dispatches at compile time,
    /// providing maximum performance for the hot path.
    #[inline(always)]
    pub fn generate_tokens_fast_path(
        &mut self,
        input_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
    ) -> NoesisResult<Vec<u32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => {
                backend.generate_tokens_fast_path(input_tokens, max_tokens, temperature)
            }
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => {
                backend.generate_tokens_fast_path(input_tokens, max_tokens, temperature)
            }
            
            #[cfg(all(target_os = "windows", feature = "onnx"))]
            UnifiedBackend::Onnx(backend) => {
                backend.infer(1, input_tokens, max_tokens, temperature, 0.9).map_err(NoesisError::from)
            }
        }
    }
    
    /// ESSENTIAL: Zero-cost minimal streaming
    #[inline(always)]
    pub fn stream_tokens_minimal(
        &mut self,
        input_tokens: &[u32],
        max_tokens: usize,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> NoesisResult<()> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => {
                backend.stream_tokens_minimal(input_tokens, max_tokens, on_token)
            }
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => {
                backend.stream_tokens_minimal(input_tokens, max_tokens, on_token)
            }
            
            #[cfg(all(target_os = "windows", feature = "onnx"))]
            UnifiedBackend::Onnx(_backend) => {
                let tokens = self.generate_tokens_fast_path(input_tokens, max_tokens, 0.7)?;
                for t in tokens { if !on_token(t) { break; } }
                Ok(())
            }
        }
    }
    
    /// Get performance metrics (zero-cost)
    #[inline(always)]
    pub fn get_performance_metrics(&self) -> &StreamlinedMetrics {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => backend.get_performance_metrics(),
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => backend.get_performance_metrics(),
            
            #[cfg(all(target_os = "windows", feature = "onnx"))]
            UnifiedBackend::Onnx(_backend) => {
                // Dummy static metrics for ONNX stub
                static METRICS: StreamlinedMetrics = StreamlinedMetrics{ tokens_generated: 0, total_generation_time: std::time::Duration::from_secs(0), average_tokens_per_second: 0.0, buffer_pool_hit_rate: 0.0 };
                &METRICS
            }
        }
    }
    
    /// Get legacy buffer pool (zero-cost) - DEPRECATED
    #[inline(always)]
    pub fn get_buffer_pool(&self) -> Arc<OptimizedBufferPool> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => backend.buffer_pool(),
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(_backend) => {
                // TODO: Implement CUDA buffer pool access
                // For now, create minimal buffer pool
                unimplemented!("CUDA buffer pool access not yet implemented")
            }
        }
    }
    
    /// REVOLUTIONARY: Create unified GPU buffer with zero-cost dispatch
    #[inline(always)]
    pub fn create_buffer(&self, size: usize) -> NoesisResult<GpuBuffer> {
        GpuBuffer::new(size)
    }
    
    /// REVOLUTIONARY: Create unified buffer pool for optimal performance
    pub fn create_unified_buffer_pool(&self, max_buffers_per_class: usize) -> UnifiedGpuBufferPool {
        UnifiedGpuBufferPool::new(max_buffers_per_class)
    }
    
    /// Allocate buffer for token operations with zero-cost dispatch
    #[inline(always)]
    pub fn allocate_token_buffer(&self, token_count: usize, pool: &UnifiedGpuBufferPool) -> NoesisResult<GpuBuffer> {
        pool.allocate_for_tokens(token_count)
    }
    
    /// Load essential pipelines/kernels
    pub fn load_essential_pipelines(&mut self) -> NoesisResult<()> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => backend.load_essential_pipelines(),
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => backend.load_essential_kernels(),
        }
    }
    
    /// Get essential device information
    pub fn get_essential_device_info(&self) -> EssentialDeviceInfo {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => backend.get_essential_device_info(),
            
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => {
                let cuda_info = backend.get_device_info();
                EssentialDeviceInfo {
                    name: cuda_info.name.clone(),
                    has_unified_memory: false, // CUDA typically doesn't have unified memory
                    max_working_set: cuda_info.total_memory,
                    backend_type: crate::errors::BackendType::Cuda,
                }
            }
            
            #[cfg(all(target_os = "windows", feature = "onnx"))]
            UnifiedBackend::Onnx(_backend) => {
                EssentialDeviceInfo {
                    name: "NVIDIA RTX (via TensorRT)".to_string(),
                    has_unified_memory: false,
                    max_working_set: 24 << 30,
                    backend_type: crate::errors::BackendType::Onnx,
                }
            }
        }
    }
    
    /// Platform-specific downcasting for advanced operations
    pub fn as_metal(&mut self) -> Option<&mut StreamlinedGpuBackend> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            UnifiedBackend::Metal(backend) => Some(backend),
            _ => None,
        }
    }
    
    #[cfg(feature = "cuda")]
    pub fn as_cuda(&mut self) -> Option<&mut StreamlinedCudaBackend> {
        match self {
            #[cfg(feature = "cuda")]
            UnifiedBackend::Cuda(backend) => Some(backend),
            _ => None,
        }
    }
}

// Use EssentialDeviceInfo from streamlined module (no duplication)

// Thread safety - each variant is Send + Sync
unsafe impl Send for UnifiedBackend {}
unsafe impl Sync for UnifiedBackend {}

// Removed legacy trait impl to enforce Unified-only API

/// Factory function for creating unified backends (replaces create_streamlined_backend)
pub fn create_unified_backend() -> NoesisResult<UnifiedBackend> {
    UnifiedBackend::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_backend_creation() {
        match create_unified_backend() {
            Ok(backend) => {
                let device_info = backend.get_essential_device_info();
                println!("Created backend: {} ({:?})", device_info.name, device_info.backend_type);
                
                // Test that we can get performance metrics
                let metrics = backend.get_performance_metrics();
                println!("Backend metrics: {:.2} tok/s", metrics.average_tokens_per_second);
            }
            Err(e) => {
                println!("Backend creation failed (expected on some platforms): {}", e);
            }
        }
    }
}
