// Optimized Metal Compute Kernels for Sampling Acceleration
// Performance target: +60-80% improvement through custom Metal shaders
// These kernels replace the generic GPT-OSS operations with Apple Silicon optimized versions

use crate::gpu_optimized::OptimizedMetalDevice as MetalDevice;
use anyhow::{Result, Context};
use objc2_metal::{
    MTLBuffer, MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLDevice, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize, MTLStorageMode
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use std::sync::Arc;
use std::ptr;
use std::ffi::c_void;
use log::{info, debug};

/// Optimized sampling strategies for different use cases
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Greedy decoding - fastest, deterministic
    Greedy,
    /// Top-K sampling with temperature
    TopK { k: usize, temperature: f32 },
    /// Top-P (nucleus) sampling
    TopP { p: f32, temperature: f32 },
    /// Speculative sampling with draft model
    Speculative { draft_k: usize, target_p: f32 },
    /// Beam search (for highest quality)
    Beam { beam_width: usize },
}

/// Metal kernel library for optimized operations
pub struct MetalKernelLibrary {
    device: Arc<MetalDevice>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    
    // Pre-compiled kernel functions
    softmax_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    topk_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    topp_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    matmul_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    layernorm_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    rope_kernel: Retained<ProtocolObject<dyn MTLFunction>>,
    
    // Pipeline states (compiled kernels)
    softmax_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    topk_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    topp_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    matmul_pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl MetalKernelLibrary {
    /// Create kernel library with optimized Metal shaders
    pub fn new(device: Arc<MetalDevice>) -> Result<Self> {
        // Metal shader source code (embedded)
        let shader_source = include_str!("../shaders/optimized_kernels.metal");
        let source_string = NSString::from_str(shader_source);
        
        // Compile shaders
        let compile_options = unsafe {
            MTLCompileOptions::new()
        };
        
        let library = unsafe {
            device.device()
                .newLibraryWithSource_options_error(
                    &source_string,
                    Some(&compile_options),
                )
                .map_err(|e| anyhow::anyhow!("Failed to compile Metal shaders: {:?}", e))?
        };
        
        // Get kernel functions
        let softmax_name = NSString::from_str("optimized_softmax");
        let softmax_kernel = unsafe {
            library.newFunctionWithName(&softmax_name)
                .context("Failed to find softmax kernel")?
        };
        
        let topk_name = NSString::from_str("optimized_topk");
        let topk_kernel = unsafe {
            library.newFunctionWithName(&topk_name)
                .context("Failed to find topk kernel")?
        };
        
        let topp_name = NSString::from_str("optimized_topp");
        let topp_kernel = unsafe {
            library.newFunctionWithName(&topp_name)
                .context("Failed to find topp kernel")?
        };
        
        let matmul_name = NSString::from_str("optimized_matmul_tiled");
        let matmul_kernel = unsafe {
            library.newFunctionWithName(&matmul_name)
                .context("Failed to find matmul kernel")?
        };
        
        let layernorm_name = NSString::from_str("optimized_layernorm");
        let layernorm_kernel = unsafe {
            library.newFunctionWithName(&layernorm_name)
                .context("Failed to find layernorm kernel")?
        };
        
        let rope_name = NSString::from_str("optimized_rope");
        let rope_kernel = unsafe {
            library.newFunctionWithName(&rope_name)
                .context("Failed to find RoPE kernel")?
        };
        
        // Create pipeline states
        let softmax_pipeline = unsafe {
            device.device()
                .newComputePipelineStateWithFunction_error(&softmax_kernel)
                .map_err(|e| anyhow::anyhow!("Failed to create softmax pipeline: {:?}", e))?
        };
        
        let topk_pipeline = unsafe {
            device.device()
                .newComputePipelineStateWithFunction_error(&topk_kernel)
                .map_err(|e| anyhow::anyhow!("Failed to create topk pipeline: {:?}", e))?
        };
        
        let topp_pipeline = unsafe {
            device.device()
                .newComputePipelineStateWithFunction_error(&topp_kernel)
                .map_err(|e| anyhow::anyhow!("Failed to create topp pipeline: {:?}", e))?
        };
        
        let matmul_pipeline = unsafe {
            device.device()
                .newComputePipelineStateWithFunction_error(&matmul_kernel)
                .map_err(|e| anyhow::anyhow!("Failed to create matmul pipeline: {:?}", e))?
        };
        
        info!("Created Metal kernel library with optimized compute pipelines");
        
        Ok(Self {
            device,
            library,
            softmax_kernel,
            topk_kernel,
            topp_kernel,
            matmul_kernel,
            layernorm_kernel,
            rope_kernel,
            softmax_pipeline,
            topk_pipeline,
            topp_pipeline,
            matmul_pipeline,
        })
    }
    
    /// Execute optimized softmax on logits
    pub fn softmax(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        vocab_size: usize,
        temperature: f32,
    ) -> Result<()> {
        unsafe {
            command_encoder.setComputePipelineState(&self.softmax_pipeline);
            command_encoder.setBuffer_offset_atIndex(Some(logits), 0, 0);
            command_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            
            // Pass parameters
            let params = [vocab_size as u32, temperature.to_bits()];
            command_encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(params.as_ptr() as *mut std::ffi::c_void).unwrap(),
                params.len() * 4,
                2,
            );
            
            // Calculate thread groups
            let threads_per_group = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            
            let thread_groups = MTLSize {
                width: (vocab_size + 255) / 256,
                height: 1,
                depth: 1,
            };
            
            command_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                thread_groups,
                threads_per_group,
            );
        }
        
        Ok(())
    }
    
    /// Execute optimized Top-K sampling
    pub fn topk_sample(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        indices: &ProtocolObject<dyn MTLBuffer>,
        vocab_size: usize,
        k: usize,
        temperature: f32,
    ) -> Result<()> {
        unsafe {
            command_encoder.setComputePipelineState(&self.topk_pipeline);
            command_encoder.setBuffer_offset_atIndex(Some(logits), 0, 0);
            command_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            command_encoder.setBuffer_offset_atIndex(Some(indices), 0, 2);
            
            let params = [vocab_size as u32, k as u32, temperature.to_bits()];
            command_encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(params.as_ptr() as *mut c_void).unwrap(),
                params.len() * 4,
                3,
            );
            
            // Use optimized thread configuration for Top-K
            let threads_per_group = MTLSize {
                width: 32, // Warp size for efficient reduction
                height: 1,
                depth: 1,
            };
            
            let thread_groups = MTLSize {
                width: (k + 31) / 32,
                height: 1,
                depth: 1,
            };
            
            command_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                thread_groups,
                threads_per_group,
            );
        }
        
        Ok(())
    }
    
    /// Execute optimized Top-P (nucleus) sampling
    pub fn topp_sample(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits: &ProtocolObject<dyn MTLBuffer>,
        output: &ProtocolObject<dyn MTLBuffer>,
        vocab_size: usize,
        p: f32,
        temperature: f32,
    ) -> Result<()> {
        unsafe {
            command_encoder.setComputePipelineState(&self.topp_pipeline);
            command_encoder.setBuffer_offset_atIndex(Some(logits), 0, 0);
            command_encoder.setBuffer_offset_atIndex(Some(output), 0, 1);
            
            let params = [vocab_size as u32, p.to_bits(), temperature.to_bits()];
            command_encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(params.as_ptr() as *mut std::ffi::c_void).unwrap(),
                params.len() * 4,
                2,
            );
            
            // Dynamic thread configuration based on vocab size
            let threads_per_group = MTLSize {
                width: 256,
                height: 1,
                depth: 1,
            };
            
            let thread_groups = MTLSize {
                width: (vocab_size + 255) / 256,
                height: 1,
                depth: 1,
            };
            
            command_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                thread_groups,
                threads_per_group,
            );
        }
        
        Ok(())
    }
    
    /// Execute optimized tiled matrix multiplication
    /// Uses 2D tiling for optimal cache usage on Apple Silicon
    pub fn matmul_tiled(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        a: &ProtocolObject<dyn MTLBuffer>,
        b: &ProtocolObject<dyn MTLBuffer>,
        c: &ProtocolObject<dyn MTLBuffer>,
        m: usize, // rows of A
        n: usize, // cols of B
        k: usize, // cols of A, rows of B
    ) -> Result<()> {
        unsafe {
            command_encoder.setComputePipelineState(&self.matmul_pipeline);
            command_encoder.setBuffer_offset_atIndex(Some(a), 0, 0);
            command_encoder.setBuffer_offset_atIndex(Some(b), 0, 1);
            command_encoder.setBuffer_offset_atIndex(Some(c), 0, 2);
            
            let params = [m as u32, n as u32, k as u32];
            command_encoder.setBytes_length_atIndex(
                std::ptr::NonNull::new(params.as_ptr() as *mut c_void).unwrap(),
                params.len() * 4,
                3,
            );
            
            // Optimal tile size for Apple Silicon (tuned for M2 Ultra)
            const TILE_SIZE: usize = 32;
            
            let threads_per_group = MTLSize {
                width: TILE_SIZE,
                height: TILE_SIZE,
                depth: 1,
            };
            
            let thread_groups = MTLSize {
                width: (n + TILE_SIZE - 1) / TILE_SIZE,
                height: (m + TILE_SIZE - 1) / TILE_SIZE,
                depth: 1,
            };
            
            command_encoder.dispatchThreadgroups_threadsPerThreadgroup(
                thread_groups,
                threads_per_group,
            );
        }
        
        Ok(())
    }
}

/// Sampling accelerator using optimized Metal kernels
pub struct SamplingAccelerator {
    kernel_library: Arc<MetalKernelLibrary>,
    strategy: SamplingStrategy,
    
    // Pre-allocated buffers for sampling
    logits_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    probs_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    indices_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    
    vocab_size: usize,
}

impl SamplingAccelerator {
    pub fn new(
        device: Arc<MetalDevice>,
        kernel_library: Arc<MetalKernelLibrary>,
        vocab_size: usize,
        strategy: SamplingStrategy,
    ) -> Result<Self> {
        // Pre-allocate buffers
        let buffer_size = vocab_size * std::mem::size_of::<f32>();
        
        let options = MTLResourceOptions::StorageModeShared;
        
        let logits_buffer = unsafe {
            device.device()
                .newBufferWithLength_options(buffer_size, options)
                .context("Failed to allocate logits buffer")?
        };
        
        let probs_buffer = unsafe {
            device.device()
                .newBufferWithLength_options(buffer_size, options)
                .context("Failed to allocate probs buffer")?
        };
        
        let indices_buffer = unsafe {
            device.device()
                .newBufferWithLength_options(
                    vocab_size * std::mem::size_of::<u32>(),
                    options,
                )
                .context("Failed to allocate indices buffer")?
        };
        
        info!("Created sampling accelerator with strategy: {:?}", strategy);
        
        Ok(Self {
            kernel_library,
            strategy,
            logits_buffer,
            probs_buffer,
            indices_buffer,
            vocab_size,
        })
    }
    
    /// Sample next token using optimized Metal kernels
    pub fn sample(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        logits: &[f32],
    ) -> Result<u32> {
        // Copy logits to GPU buffer (this is the only copy needed)
        unsafe {
            let logits_ptr = self.logits_buffer.contents().as_ptr() as *mut f32;
            ptr::copy_nonoverlapping(logits.as_ptr(), logits_ptr, self.vocab_size);
        }
        
        // Execute sampling based on strategy
        match self.strategy {
            SamplingStrategy::Greedy => {
                // Simple argmax - can be done on CPU for small vocab
                let max_idx = logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                Ok(max_idx)
            }
            
            SamplingStrategy::TopK { k, temperature } => {
                self.kernel_library.topk_sample(
                    command_encoder,
                    &self.logits_buffer,
                    &self.probs_buffer,
                    &self.indices_buffer,
                    self.vocab_size,
                    k,
                    temperature,
                )?;
                
                // Read back sampled token
                unsafe {
                    let indices_ptr = self.indices_buffer.contents().as_ptr() as *const u32;
                    Ok(*indices_ptr)
                }
            }
            
            SamplingStrategy::TopP { p, temperature } => {
                self.kernel_library.topp_sample(
                    command_encoder,
                    &self.logits_buffer,
                    &self.probs_buffer,
                    self.vocab_size,
                    p,
                    temperature,
                )?;
                
                // Read back sampled token
                unsafe {
                    let probs_ptr = self.probs_buffer.contents().as_ptr() as *const u32;
                    Ok(*probs_ptr) // First element contains sampled token
                }
            }
            
            _ => {
                // Fallback to greedy for unimplemented strategies
                self.sample_greedy(logits)
            }
        }
    }
    
    fn sample_greedy(&self, logits: &[f32]) -> Result<u32> {
        let max_idx = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        Ok(max_idx)
    }
}

/// Speculative decoding accelerator
/// Uses a small draft model to generate candidates, verified by main model
pub struct SpeculativeDecoder {
    draft_accelerator: SamplingAccelerator,
    target_accelerator: SamplingAccelerator,
    draft_length: usize,
    acceptance_threshold: f32,
}

impl SpeculativeDecoder {
    pub fn new(
        device: Arc<MetalDevice>,
        kernel_library: Arc<MetalKernelLibrary>,
        vocab_size: usize,
        draft_length: usize,
    ) -> Result<Self> {
        let draft_accelerator = SamplingAccelerator::new(
            device.clone(),
            kernel_library.clone(),
            vocab_size,
            SamplingStrategy::TopK { k: 5, temperature: 0.8 },
        )?;
        
        let target_accelerator = SamplingAccelerator::new(
            device,
            kernel_library,
            vocab_size,
            SamplingStrategy::TopP { p: 0.9, temperature: 0.7 },
        )?;
        
        Ok(Self {
            draft_accelerator,
            target_accelerator,
            draft_length,
            acceptance_threshold: 0.8,
        })
    }
    
    /// Generate multiple tokens speculatively
    pub fn decode_speculative(
        &self,
        command_encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        draft_logits: Vec<Vec<f32>>,
        target_logits: Vec<Vec<f32>>,
    ) -> Result<Vec<u32>> {
        let mut accepted_tokens = Vec::new();
        
        // Generate draft tokens
        for (i, logits) in draft_logits.iter().enumerate() {
            let draft_token = self.draft_accelerator.sample(command_encoder, logits)?;
            
            // Verify with target model
            if i < target_logits.len() {
                let target_token = self.target_accelerator.sample(command_encoder, &target_logits[i])?;
                
                if draft_token == target_token {
                    accepted_tokens.push(draft_token);
                } else {
                    // Rejection - use target token and stop
                    accepted_tokens.push(target_token);
                    break;
                }
            }
        }
        
        Ok(accepted_tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sampling_strategies() {
        // Test different sampling configurations
        let strategies = vec![
            SamplingStrategy::Greedy,
            SamplingStrategy::TopK { k: 10, temperature: 0.8 },
            SamplingStrategy::TopP { p: 0.95, temperature: 0.7 },
            SamplingStrategy::Beam { beam_width: 4 },
        ];
        
        for strategy in strategies {
            println!("Testing strategy: {:?}", strategy);
            // Actual tests would require Metal device
        }
    }
}