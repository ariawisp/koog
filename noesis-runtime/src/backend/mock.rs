// Mock GPU Backend for Testing
//
// Used in unit tests where we need a GpuBackend implementation
// but don't want to initialize actual GPU hardware

use super::{GpuBackend, DeviceInfo, BackendType, MemoryStats, StateCheckpoint, ComputePipeline, BufferHandle};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock GPU backend for testing
pub struct MockGpuBackend {
    models: Arc<Mutex<HashMap<u64, String>>>,
    buffers: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    next_model_id: Arc<Mutex<u64>>,
    next_buffer_id: Arc<Mutex<u64>>,
}

impl MockGpuBackend {
    pub fn new() -> Self {
        Self {
            models: Arc::new(Mutex::new(HashMap::new())),
            buffers: Arc::new(Mutex::new(HashMap::new())),
            next_model_id: Arc::new(Mutex::new(1)),
            next_buffer_id: Arc::new(Mutex::new(1)),
        }
    }
}

impl GpuBackend for MockGpuBackend {
    fn new(_model_path: &str) -> Result<Self> where Self: Sized {
        Ok(Self::new())
    }
    
    fn load_model(&self, model_path: &str) -> Result<u64> {
        let mut models = self.models.lock().unwrap();
        let mut next_id = self.next_model_id.lock().unwrap();
        
        let model_id = *next_id;
        *next_id += 1;
        
        models.insert(model_id, model_path.to_string());
        Ok(model_id)
    }
    
    fn infer(
        &self,
        _model_handle: u64,
        input_tokens: &[u32],
        max_tokens: usize,
        _temperature: f32,
        _top_p: f32,
    ) -> Result<Vec<u32>> {
        // Mock inference - return some tokens based on input
        let mut output = Vec::new();
        for i in 0..max_tokens.min(10) {
            let token = input_tokens.get(0).unwrap_or(&12345) + i as u32;
            output.push(token);
        }
        Ok(output)
    }
    
    fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        // Mock embeddings - return zeros
        Ok(vec![0.0; tokens.len() * 768]) // 768-dimensional embeddings
    }
    
    fn stream_tokens(
        &self,
        model_handle: u64,
        input_tokens: &[u32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<()> {
        // Mock streaming - call inference and stream results
        let tokens = self.infer(model_handle, input_tokens, max_tokens, temperature, top_p)?;
        
        for token in tokens {
            if !on_token(token) {
                break;
            }
        }
        
        Ok(())
    }
    
    fn get_device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: "Mock GPU Device".to_string(),
            backend_type: BackendType::Metal, // Arbitrary choice
            total_memory: 8 * 1024 * 1024 * 1024, // 8GB
            compute_capability: "Mock 1.0".to_string(),
            supports_unified_memory: true,
        }
    }
    
    fn is_available() -> bool where Self: Sized {
        true // Mock backend is always available
    }
    
    fn get_memory_stats(&self) -> MemoryStats {
        MemoryStats {
            used_bytes: 1024 * 1024 * 1024, // 1GB used
            total_bytes: 8 * 1024 * 1024 * 1024, // 8GB total
            cached_models: self.models.lock().unwrap().len(),
        }
    }
    
    fn checkpoint(&self, _handle: u64) -> Result<StateCheckpoint> {
        Ok(StateCheckpoint {
            tokens: vec![1, 2, 3, 4, 5], // Mock tokens
            kv_cache: vec![0u8; 1024], // Mock KV cache
            timestamp: chrono::Utc::now().timestamp(),
        })
    }
    
    fn restore(&self, _checkpoint: StateCheckpoint) -> Result<u64> {
        // Mock restore - return a new handle
        Ok(999)
    }
    
    fn create_compute_pipeline(&self, kernel_name: &str) -> Result<ComputePipeline> {
        Ok(ComputePipeline {
            id: format!("mock_{}", kernel_name),
            backend_type: BackendType::Metal,
        })
    }
    
    fn execute_compute_pipeline(
        &self,
        _pipeline: &ComputePipeline,
        _buffers: &[&BufferHandle],
        _thread_count: usize,
    ) -> Result<()> {
        // Mock execution - do nothing but succeed
        Ok(())
    }
    
    fn create_buffer(&self, size: usize, label: &str) -> Result<BufferHandle> {
        let mut next_id = self.next_buffer_id.lock().unwrap();
        let buffer_id = format!("{}_{}", label, *next_id);
        *next_id += 1;
        
        // Initialize buffer with zeros
        let mut buffers = self.buffers.lock().unwrap();
        buffers.insert(buffer_id.clone(), vec![0u8; size]);
        
        Ok(BufferHandle {
            id: buffer_id,
            size,
            backend_type: BackendType::Metal,
        })
    }
    
    fn read_buffer(&self, buffer: &BufferHandle) -> Result<Vec<u8>> {
        let buffers = self.buffers.lock().unwrap();
        
        match buffers.get(&buffer.id) {
            Some(data) => Ok(data.clone()),
            None => Ok(vec![0u8; buffer.size]), // Return zeros if buffer not found
        }
    }
    
    fn write_buffer(&self, buffer: &BufferHandle, data: &[u8]) -> Result<()> {
        let mut buffers = self.buffers.lock().unwrap();
        
        // Ensure data fits in buffer
        let write_size = data.len().min(buffer.size);
        let mut buffer_data = vec![0u8; buffer.size];
        buffer_data[..write_size].copy_from_slice(&data[..write_size]);
        
        buffers.insert(buffer.id.clone(), buffer_data);
        Ok(())
    }
}

impl Default for MockGpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mock_backend_creation() {
        let backend = MockGpuBackend::new();
        assert!(MockGpuBackend::is_available());
        
        let device_info = backend.get_device_info();
        assert_eq!(device_info.name, "Mock GPU Device");
    }
    
    #[test]
    fn test_mock_inference() {
        let backend = MockGpuBackend::new();
        let model_handle = backend.load_model("test_model").unwrap();
        
        let result = backend.infer(model_handle, &[1, 2, 3], 5, 0.8, 0.9);
        assert!(result.is_ok());
        
        let tokens = result.unwrap();
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 10);
    }
    
    #[test]
    fn test_mock_buffer_operations() {
        let backend = MockGpuBackend::new();
        
        // Create buffer
        let buffer = backend.create_buffer(1024, "test_buffer").unwrap();
        assert_eq!(buffer.size, 1024);
        
        // Write data
        let test_data = vec![1, 2, 3, 4, 5];
        backend.write_buffer(&buffer, &test_data).unwrap();
        
        // Read data back
        let read_data = backend.read_buffer(&buffer).unwrap();
        assert_eq!(read_data.len(), 1024);
        assert_eq!(&read_data[..5], &test_data);
    }
}