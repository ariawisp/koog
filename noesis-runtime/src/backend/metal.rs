// Metal Backend Implementation
// Wraps the existing MetalInferenceEngine to implement GpuBackend trait

use super::{GpuBackend, DeviceInfo, BackendType, MemoryStats, StateCheckpoint, ComputePipeline, BufferHandle};
use crate::inference::noesis_metal::{MetalInferenceEngine, is_metal_available, get_device_info};
use crate::gpu_optimized::{OptimizedMetalDevice as MetalDevice, OptimizedBufferPool as BufferPool, OptimizedBuffer as UnifiedBuffer};
use anyhow::{Result, Context};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use log::info;
use objc2::rc::Id;
use objc2_metal::{MTLComputePipelineState, MTLDevice as MTLDeviceProtocol, MTLLibrary, MTLSize, MTLCommandQueue};
use objc2_foundation::NSString;

/// Metal backend for macOS
pub struct MetalBackend {
    device: Arc<MetalDevice>,
    buffer_pool: Arc<BufferPool>,
    inference_engine: Arc<MetalInferenceEngine>,
    
    // Cognitive compute resources (simplified for now)
    cognitive_buffers: RwLock<HashMap<String, UnifiedBuffer>>,
}

// Safety: Metal objects are thread-safe when wrapped in Arc
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl GpuBackend for MetalBackend {
    fn new(model_path: &str) -> Result<Self> where Self: Sized {
        info!("Initializing Metal backend with model: {}", model_path);
        
        // Initialize Metal device
        let device = Arc::new(MetalDevice::new()?);
        
        // Create buffer pool (1GB)
        let buffer_pool = Arc::new(BufferPool::new(&device, 1 << 30)?);
        
        // Initialize inference engine
        let inference_engine = Arc::new(MetalInferenceEngine::new(model_path)?);
        
        Ok(MetalBackend {
            device,
            buffer_pool,
            inference_engine,
            cognitive_buffers: RwLock::new(HashMap::new()),
        })
    }
    
    fn load_model(&self, model_path: &str) -> Result<u64> {
        self.inference_engine.load_model(model_path, &self.buffer_pool)
    }
    
    fn infer(
        &self,
        model_handle: u64,
        input_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        // Allocate buffer for input
        let input_buffer = self.buffer_pool.allocate(input_tokens.len() * 4)?;
        input_buffer.write(input_tokens)?;
        
        // Run inference
        let output_buffer = self.inference_engine.infer(
            model_handle,
            &input_buffer,
            max_tokens,
            temperature,
            top_p
        )?;
        
        // Read output tokens
        output_buffer.read_tokens()
    }
    
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Allocate buffer for input
        let input_buffer = self.buffer_pool.allocate(tokens.len() * 4)?;
        input_buffer.write(tokens)?;
        
        // Generate embeddings
        self.inference_engine.embed(&input_buffer)
    }
    
    fn stream_tokens(
        &self,
        model_handle: u64,
        input_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<()>
    {
        // Allocate buffer for input
        let input_buffer = self.buffer_pool.allocate(input_tokens.len() * 4)?;
        input_buffer.write(input_tokens)?;
        
        // Stream tokens using the inference engine
        // This is a simplified implementation - in production we'd stream directly
        let output_buffer = self.inference_engine.infer(
            model_handle,
            &input_buffer,
            max_tokens,
            temperature,
            top_p
        )?;
        
        let tokens = output_buffer.read_tokens()?;
        for token in tokens {
            if !on_token(token) {
                break;
            }
        }
        
        Ok(())
    }
    
    fn get_device_info(&self) -> DeviceInfo {
        let metal_info = get_device_info();
        
        DeviceInfo {
            name: self.device.name(),
            backend_type: BackendType::Metal,
            total_memory: self.device.recommended_max_working_set_size(),
            compute_capability: metal_info,
            supports_unified_memory: self.device.has_unified_memory(),
        }
    }
    
    fn is_available() -> bool where Self: Sized {
        is_metal_available()
    }
    
    fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            used_bytes: self.buffer_pool.bytes_allocated() as u64,
            total_bytes: self.buffer_pool.total_capacity() as u64,
            cached_models: 1, // TODO: Get from inference engine
        }
    }
    
    fn checkpoint(&self, _handle: u64) -> Result<StateCheckpoint> {
        // TODO: Implement Metal-specific checkpointing
        // Would save GPU buffer states
        Ok(StateCheckpoint {
            tokens: vec![],
            kv_cache: vec![],
            timestamp: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
        })
    }
    
    fn restore(&self, _checkpoint: StateCheckpoint) -> Result<u64> {
        // TODO: Implement checkpoint restoration
        Ok(1)
    }
    
    fn create_compute_pipeline(&self, kernel_name: &str) -> Result<ComputePipeline> {
        // Stub implementation - cognitive kernels integration will be completed later
        info!("Creating Metal compute pipeline: {} (stub)", kernel_name);
        
        Ok(ComputePipeline {
            id: kernel_name.to_string(),
            backend_type: BackendType::Metal,
        })
    }
    
    fn execute_compute_pipeline(
        &self,
        pipeline: &ComputePipeline,
        buffers: &[&BufferHandle],
        thread_count: usize,
    ) -> Result<()> {
        // Stub implementation - cognitive kernels execution will be completed later
        info!("Executing Metal compute pipeline: {} with {} buffers, {} threads (stub)", 
              pipeline.id, buffers.len(), thread_count);
        
        Ok(())
    }
    
    fn create_buffer(&self, size: usize, label: &str) -> Result<BufferHandle> {
        // Create actual Metal buffer through BufferPool
        let buffer = self.buffer_pool.allocate(size)?;
        
        // Store buffer for later access
        let mut cognitive_buffers = self.cognitive_buffers.write().unwrap();
        cognitive_buffers.insert(label.to_string(), buffer);
        
        Ok(BufferHandle {
            id: label.to_string(),
            size,
            backend_type: BackendType::Metal,
        })
    }
    
    fn read_buffer(&self, buffer: &BufferHandle) -> Result<Vec<u8>> {
        let cognitive_buffers = self.cognitive_buffers.read().unwrap();
        let unified_buffer = cognitive_buffers.get(&buffer.id)
            .context(format!("Buffer not found: {}", buffer.id))?;
        
        // Read from UnifiedBuffer and convert to bytes
        let tokens = unified_buffer.read_tokens()
            .map_err(|e| anyhow::anyhow!("Buffer read error: {}", e))?;
        
        // Convert u32 tokens to bytes
        let bytes: Vec<u8> = tokens.iter()
            .flat_map(|&token| token.to_ne_bytes())
            .take(buffer.size) // Limit to requested size
            .collect();
        
        Ok(bytes)
    }
    
    fn write_buffer(&self, buffer: &BufferHandle, data: &[u8]) -> Result<()> {
        let cognitive_buffers = self.cognitive_buffers.read().unwrap();
        let unified_buffer = cognitive_buffers.get(&buffer.id)
            .context(format!("Buffer not found: {}", buffer.id))?;
        
        if data.len() > buffer.size {
            return Err(anyhow::anyhow!("Data size {} exceeds buffer size {}", data.len(), buffer.size));
        }
        
        // Convert bytes to u32 tokens for UnifiedBuffer
        let mut tokens = Vec::new();
        for chunk in data.chunks(4) {
            let mut bytes = [0u8; 4];
            bytes[..chunk.len()].copy_from_slice(chunk);
            tokens.push(u32::from_ne_bytes(bytes));
        }
        
        unified_buffer.write(&tokens)
            .map_err(|e| anyhow::anyhow!("Buffer write error: {}", e))?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metal_availability() {
        let available = MetalBackend::is_available();
        println!("Metal available: {}", available);
        
        if available {
            let backend = MetalBackend::new("dummy.safetensors");
            if let Ok(backend) = backend {
                let info = backend.get_device_info();
                println!("Metal device: {}", info.name);
                println!("Unified memory: {}", info.supports_unified_memory);
            }
        }
    }
}