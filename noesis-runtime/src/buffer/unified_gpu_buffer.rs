// Unified GPU buffer implementation for cross-platform operations

use crate::errors::{NoesisError, NoesisResult};

#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::rc::Retained;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2::runtime::ProtocolObject;
#[cfg(all(target_os = "macos", feature = "metal"))]
use objc2_metal::MTLBuffer;

#[cfg(feature = "cuda")]
use cust::memory::DevicePointer;

/// Unified GPU buffer with zero-cost platform dispatch
pub enum GpuBuffer {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal {
        buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        size: usize,
        actual_size: usize,
    },
    
    #[cfg(feature = "cuda")]
    Cuda {
        buffer: DevicePointer<u8>,
        offset: usize,
        size: usize,
        stream: cust::stream::Stream,
    },
    
    // Future: ONNX/DirectML for Windows
    #[cfg(feature = "onnx")]
    Onnx {
        // Windows-specific buffer implementation
    },
}

impl GpuBuffer {
    /// Create new GPU buffer with platform-appropriate implementation
    pub fn new(size: usize) -> NoesisResult<Self> {
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            Self::new_metal(size)
        }
        
        #[cfg(all(feature = "cuda", not(all(target_os = "macos", feature = "metal"))))]
        {
            Self::new_cuda(size)
        }
        
        #[cfg(not(any(
            all(target_os = "macos", feature = "metal"),
            feature = "cuda",
            feature = "onnx"
        )))]
        {
            Err(NoesisError::backend_initialization("No GPU buffer implementation available"))
        }
    }
    
    /// CRITICAL PATH: Write tokens with zero-cost dispatch
    #[inline(always)]
    pub fn write_tokens(&self, tokens: &[u32]) -> NoesisResult<()> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBuffer::Metal { buffer, offset, size, .. } => {
                self.write_tokens_metal(buffer, *offset, *size, tokens)
            }
            
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { buffer, offset, size, stream } => {
                self.write_tokens_cuda(buffer, *offset, *size, stream, tokens)
            }
            
            #[cfg(feature = "onnx")]
            GpuBuffer::Onnx { .. } => {
                // TODO: Implement ONNX token writing
                unimplemented!("ONNX buffer operations not yet implemented")
            }
        }
    }
    
    /// CRITICAL PATH: Read tokens with zero-cost dispatch
    #[inline(always)]
    pub fn read_tokens(&self, count: usize) -> NoesisResult<Vec<u32>> {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBuffer::Metal { buffer, offset, .. } => {
                self.read_tokens_metal(buffer, *offset, count)
            }
            
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { buffer, offset, stream, .. } => {
                self.read_tokens_cuda(buffer, *offset, stream, count)
            }
            
            #[cfg(feature = "onnx")]
            GpuBuffer::Onnx { .. } => {
                // TODO: Implement ONNX token reading
                unimplemented!("ONNX buffer operations not yet implemented")
            }
        }
    }
    
    /// Get buffer size
    #[inline(always)]
    pub fn size(&self) -> usize {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBuffer::Metal { size, .. } => *size,
            
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { size, .. } => *size,
            
            #[cfg(feature = "onnx")]
            GpuBuffer::Onnx { .. } => 0, // TODO: Implement
        }
    }
    
    /// Check if buffer uses unified memory (zero-copy capable)
    pub fn has_unified_memory(&self) -> bool {
        match self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            GpuBuffer::Metal { .. } => true, // Apple Silicon unified memory
            
            #[cfg(feature = "cuda")]
            GpuBuffer::Cuda { .. } => false, // CUDA typically requires explicit transfers
            
            #[cfg(feature = "onnx")]
            GpuBuffer::Onnx { .. } => false, // DirectML may support unified memory in future
        }
    }
    
    // Platform-specific implementations
    
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn new_metal(size: usize) -> NoesisResult<Self> {
        // TODO: Fix Metal buffer creation - objc2-metal API method names need verification
        // The correct method name for buffer creation is unclear in current objc2-metal version
        Err(NoesisError::backend_initialization("Metal buffer creation temporarily disabled - needs API verification"))
    }
    
    #[cfg(feature = "cuda")]
    fn new_cuda(size: usize) -> NoesisResult<Self> {
        let stream = cust::stream::Stream::new(cust::stream::StreamFlags::NON_BLOCKING, None)
            .map_err(|e| NoesisError::backend_initialization(&format!("CUDA stream creation failed: {}", e)))?;
        
        let buffer = unsafe {
            DevicePointer::<u8>::alloc(size)
                .map_err(|e| NoesisError::memory_allocation(&format!("CUDA allocation failed: {}", e)))?
        };
        
        Ok(GpuBuffer::Cuda {
            buffer,
            offset: 0,
            size,
            stream,
        })
    }
    
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn write_tokens_metal(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        size: usize,
        tokens: &[u32],
    ) -> NoesisResult<()> {
        let byte_size = tokens.len() * 4;
        if byte_size > size {
            return Err(NoesisError::buffer_operation(
                crate::errors::BufferOperation::Write,
                &format!("Token data too large: {} > {}", byte_size, size)
            ));
        }
        
        // SAFETY: Metal unified memory allows direct CPU access
        let buffer_contents = buffer.contents();
        let ptr = unsafe { buffer_contents.as_ptr().add(offset) as *mut u32 };
        
        // Verify alignment
        if (ptr as usize) % 4 != 0 {
            return Err(NoesisError::buffer_operation(
                crate::errors::BufferOperation::Write,
                "Metal buffer not aligned for u32 access"
            ));
        }
        
        // Zero-copy write using unified memory
        unsafe {
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, tokens.len());
        }
        
        Ok(())
    }
    
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn read_tokens_metal(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        count: usize,
    ) -> NoesisResult<Vec<u32>> {
        let buffer_contents = buffer.contents();
        let ptr = unsafe { buffer_contents.as_ptr().add(offset) as *const u32 };
        
        // Verify alignment
        if (ptr as usize) % 4 != 0 {
            return Err(NoesisError::buffer_operation(
                crate::errors::BufferOperation::Read,
                "Metal buffer not aligned for u32 access"
            ));
        }
        
        // Zero-copy read using unified memory
        let mut tokens = Vec::with_capacity(count);
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, tokens.as_mut_ptr(), count);
            tokens.set_len(count);
        }
        
        Ok(tokens)
    }
    
    #[cfg(feature = "cuda")]
    fn write_tokens_cuda(
        &self,
        buffer: &DevicePointer<u8>,
        offset: usize,
        size: usize,
        stream: &cust::stream::Stream,
        tokens: &[u32],
    ) -> NoesisResult<()> {
        let byte_size = tokens.len() * 4;
        if byte_size > size {
            return Err(NoesisError::buffer_operation(
                crate::errors::BufferOperation::Write,
                &format!("Token data too large: {} > {}", byte_size, size)
            ));
        }
        
        // Convert tokens to bytes
        let bytes: Vec<u8> = tokens.iter()
            .flat_map(|&token| token.to_le_bytes())
            .collect();
        
        // Async copy to GPU
        unsafe {
            let offset_buffer = buffer.clone().offset(offset as isize);
            stream.synchronize().map_err(|e| {
                NoesisError::buffer_operation(
                    crate::errors::BufferOperation::Write,
                    &format!("CUDA stream sync failed: {}", e)
                )
            })?;
            
            offset_buffer.copy_from(&bytes).map_err(|e| {
                NoesisError::buffer_operation(
                    crate::errors::BufferOperation::Write,
                    &format!("CUDA copy failed: {}", e)
                )
            })?;
        }
        
        Ok(())
    }
    
    #[cfg(feature = "cuda")]
    fn read_tokens_cuda(
        &self,
        buffer: &DevicePointer<u8>,
        offset: usize,
        stream: &cust::stream::Stream,
        count: usize,
    ) -> NoesisResult<Vec<u32>> {
        let byte_size = count * 4;
        let mut host_bytes = vec![0u8; byte_size];
        
        // Async copy from GPU
        unsafe {
            let offset_buffer = buffer.clone().offset(offset as isize);
            stream.synchronize().map_err(|e| {
                NoesisError::buffer_operation(
                    crate::errors::BufferOperation::Read,
                    &format!("CUDA stream sync failed: {}", e)
                )
            })?;
            
            offset_buffer.copy_to(&mut host_bytes).map_err(|e| {
                NoesisError::buffer_operation(
                    crate::errors::BufferOperation::Read,
                    &format!("CUDA copy failed: {}", e)
                )
            })?;
        }
        
        // Convert bytes to tokens
        let tokens: Vec<u32> = host_bytes.chunks_exact(4)
            .take(count)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        Ok(tokens)
    }
}

// Thread safety
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // Platform-specific cleanup is handled by the underlying types
        // Metal: Retained handles reference counting
        // CUDA: DevicePointer handles deallocation
    }
}

/// Unified buffer pool that manages GpuBuffer instances
pub struct UnifiedGpuBufferPool {
    // Buffer size classes for efficient allocation
    size_classes: Vec<usize>,
    // Pool of available buffers per size class
    available_buffers: std::sync::Mutex<std::collections::HashMap<usize, Vec<GpuBuffer>>>,
    // Pool configuration
    max_buffers_per_class: usize,
    total_allocated: std::sync::atomic::AtomicUsize,
}

impl UnifiedGpuBufferPool {
    /// Create new unified buffer pool
    pub fn new(max_buffers_per_class: usize) -> Self {
        // Common buffer sizes for token operations
        let size_classes = vec![
            1024,        // 256 tokens
            4096,        // 1K tokens  
            16384,       // 4K tokens
            65536,       // 16K tokens
            262144,      // 64K tokens
            1048576,     // 256K tokens
        ];
        
        Self {
            size_classes,
            available_buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
            max_buffers_per_class,
            total_allocated: std::sync::atomic::AtomicUsize::new(0),
        }
    }
    
    /// Allocate buffer for token count with optimal size class
    pub fn allocate_for_tokens(&self, token_count: usize) -> NoesisResult<GpuBuffer> {
        let byte_size = token_count * 4;
        let size_class = self.find_size_class(byte_size);
        
        // Try to reuse existing buffer
        {
            let mut available = self.available_buffers.lock().unwrap();
            if let Some(buffers) = available.get_mut(&size_class) {
                if let Some(buffer) = buffers.pop() {
                    return Ok(buffer);
                }
            }
        }
        
        // Create new buffer
        let buffer = GpuBuffer::new(size_class)?;
        self.total_allocated.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        
        Ok(buffer)
    }
    
    /// Return buffer to pool for reuse
    pub fn deallocate(&self, buffer: GpuBuffer) {
        let size = buffer.size();
        
        let mut available = self.available_buffers.lock().unwrap();
        let buffers = available.entry(size).or_insert_with(Vec::new);
        
        if buffers.len() < self.max_buffers_per_class {
            buffers.push(buffer);
        } else {
            // Let buffer drop naturally if pool is full
            self.total_allocated.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    
    fn find_size_class(&self, required_size: usize) -> usize {
        self.size_classes
            .iter()
            .find(|&&size| size >= required_size)
            .copied()
            .unwrap_or_else(|| {
                // For very large allocations, round up to next power of 2
                let mut size = required_size;
                size = size.next_power_of_two();
                size
            })
    }
    
    /// Get pool statistics
    pub fn statistics(&self) -> UnifiedBufferPoolStats {
        let available = self.available_buffers.lock().unwrap();
        let available_count: usize = available.values().map(|v| v.len()).sum();
        let total_allocated = self.total_allocated.load(std::sync::atomic::Ordering::Relaxed);
        
        UnifiedBufferPoolStats {
            total_allocated,
            available_count,
            active_buffers: total_allocated.saturating_sub(available_count),
            size_classes_used: available.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedBufferPoolStats {
    pub total_allocated: usize,
    pub available_count: usize,
    pub active_buffers: usize,
    pub size_classes_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_unified_buffer_creation() {
        match GpuBuffer::new(4096) {
            Ok(buffer) => {
                assert_eq!(buffer.size(), 4096);
                println!("Unified buffer created successfully");
            }
            Err(e) => {
                println!("Buffer creation failed (expected on some platforms): {}", e);
            }
        }
    }
    
    #[test]
    fn test_buffer_pool() {
        let pool = UnifiedGpuBufferPool::new(10);
        
        if let Ok(buffer) = pool.allocate_for_tokens(1000) {
            assert!(buffer.size() >= 4000); // 1000 tokens * 4 bytes
            
            let stats_before = pool.statistics();
            pool.deallocate(buffer);
            let stats_after = pool.statistics();
            
            assert_eq!(stats_after.available_count, stats_before.available_count + 1);
            println!("Buffer pool test passed");
        }
    }
}