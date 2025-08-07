// PHASE 8B: HOT PATH BUFFER POOL - Eliminate allocations in critical paths
//
// PERFORMANCE CRITICAL: Pre-allocated buffers for token generation to eliminate
// all Vec::new() and Vec::with_capacity() calls in the inference hot path.
//
// TARGET: 25% performance improvement by removing allocation overhead

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use log::{info, debug};

/// Pre-allocated buffer pool for hot path operations
/// Eliminates allocation overhead in token generation loops
pub struct HotPathBufferPool {
    /// Pool of token buffers (Vec<u32>) for reuse
    token_buffers: Arc<Mutex<VecDeque<Vec<u32>>>>,
    /// Pool of byte buffers (Vec<u8>) for serialization
    byte_buffers: Arc<Mutex<VecDeque<Vec<u8>>>>,
    /// Pool of receiver buffers for async operations
    receiver_buffers: Arc<Mutex<VecDeque<Vec<tokio::sync::oneshot::Receiver<u32>>>>>,
    /// Pool of position arrays for parallel generation
    position_buffers: Arc<Mutex<VecDeque<Vec<usize>>>>,
    
    // Configuration
    max_pools_per_type: usize,
    default_token_capacity: usize,
    default_byte_capacity: usize,
}

impl HotPathBufferPool {
    /// Create buffer pool optimized for inference hot paths
    pub fn new(max_pools_per_type: usize) -> Self {
        info!("🚀 Initializing hot path buffer pool");
        info!("   Max pools per type: {}", max_pools_per_type);
        
        Self {
            token_buffers: Arc::new(Mutex::new(VecDeque::new())),
            byte_buffers: Arc::new(Mutex::new(VecDeque::new())),
            receiver_buffers: Arc::new(Mutex::new(VecDeque::new())),
            position_buffers: Arc::new(Mutex::new(VecDeque::new())),
            max_pools_per_type,
            default_token_capacity: 2048,  // Typical max token generation
            default_byte_capacity: 8192,   // 2048 tokens * 4 bytes
        }
    }
    
    /// Pre-warm the buffer pools with commonly used sizes
    pub fn prewarm(&self) {
        info!("🔥 Pre-warming buffer pools for optimal performance");
        
        // Pre-allocate token buffers with common capacities
        let token_capacities = [64, 256, 512, 1024, 2048];
        for &capacity in &token_capacities {
            for _ in 0..2 {  // 2 buffers per size
                self.return_token_buffer(Vec::with_capacity(capacity));
            }
        }
        
        // Pre-allocate byte buffers
        let byte_capacities = [256, 1024, 2048, 4096, 8192];
        for &capacity in &byte_capacities {
            for _ in 0..2 {
                self.return_byte_buffer(Vec::with_capacity(capacity));
            }
        }
        
        info!("✅ Buffer pools pre-warmed successfully");
    }
    
    /// Get a pre-allocated token buffer, avoiding Vec::new() in hot path
    pub fn get_token_buffer(&self, min_capacity: usize) -> Vec<u32> {
        let mut pool = self.token_buffers.lock().unwrap();
        
        // Try to find a buffer with sufficient capacity
        for _ in 0..pool.len() {
            if let Some(mut buffer) = pool.pop_front() {
                if buffer.capacity() >= min_capacity {
                    buffer.clear(); // Clear contents but keep capacity
                    debug!("♻️ Reused token buffer (capacity: {})", buffer.capacity());
                    return buffer;
                }
                // Put back if too small
                pool.push_back(buffer);
            } else {
                break;
            }
        }
        
        // No suitable buffer found, create new one
        let capacity = min_capacity.max(self.default_token_capacity);
        debug!("🆕 Created new token buffer (capacity: {})", capacity);
        Vec::with_capacity(capacity)
    }
    
    /// Return token buffer to pool for reuse
    pub fn return_token_buffer(&self, buffer: Vec<u32>) {
        let mut pool = self.token_buffers.lock().unwrap();
        let capacity = buffer.capacity();
        
        if pool.len() < self.max_pools_per_type && capacity > 0 {
            pool.push_back(buffer);
            debug!("🔄 Returned token buffer to pool (capacity: {})", capacity);
        }
        // Drop buffer if pool is full or buffer is empty
    }
    
    /// Get a pre-allocated byte buffer for serialization
    pub fn get_byte_buffer(&self, min_capacity: usize) -> Vec<u8> {
        let mut pool = self.byte_buffers.lock().unwrap();
        
        for _ in 0..pool.len() {
            if let Some(mut buffer) = pool.pop_front() {
                if buffer.capacity() >= min_capacity {
                    buffer.clear();
                    debug!("♻️ Reused byte buffer (capacity: {})", buffer.capacity());
                    return buffer;
                }
                pool.push_back(buffer);
            } else {
                break;
            }
        }
        
        let capacity = min_capacity.max(self.default_byte_capacity);
        debug!("🆕 Created new byte buffer (capacity: {})", capacity);
        Vec::with_capacity(capacity)
    }
    
    /// Return byte buffer to pool
    pub fn return_byte_buffer(&self, buffer: Vec<u8>) {
        let mut pool = self.byte_buffers.lock().unwrap();
        
        if pool.len() < self.max_pools_per_type && buffer.capacity() > 0 {
            pool.push_back(buffer);
            debug!("🔄 Returned byte buffer to pool");
        }
    }
    
    /// Get pre-allocated receiver buffer for parallel operations
    pub fn get_receiver_buffer(&self, size: usize) -> Vec<tokio::sync::oneshot::Receiver<u32>> {
        let mut pool = self.receiver_buffers.lock().unwrap();
        
        if let Some(mut buffer) = pool.pop_front() {
            if buffer.capacity() >= size {
                buffer.clear();
                debug!("♻️ Reused receiver buffer");
                return buffer;
            }
            pool.push_back(buffer);
        }
        
        debug!("🆕 Created new receiver buffer (capacity: {})", size);
        Vec::with_capacity(size)
    }
    
    /// Return receiver buffer to pool
    pub fn return_receiver_buffer(&self, buffer: Vec<tokio::sync::oneshot::Receiver<u32>>) {
        let mut pool = self.receiver_buffers.lock().unwrap();
        
        if pool.len() < self.max_pools_per_type {
            pool.push_back(buffer);
            debug!("🔄 Returned receiver buffer to pool");
        }
    }
    
    /// Get position array for parallel batch operations
    pub fn get_position_buffer(&self, size: usize) -> Vec<usize> {
        let mut pool = self.position_buffers.lock().unwrap();
        
        if let Some(mut buffer) = pool.pop_front() {
            if buffer.capacity() >= size {
                buffer.clear();
                // Pre-populate with positions
                for i in 0..size {
                    buffer.push(i);
                }
                debug!("♻️ Reused position buffer");
                return buffer;
            }
            pool.push_back(buffer);
        }
        
        debug!("🆕 Created new position buffer (size: {})", size);
        (0..size).collect()
    }
    
    /// Return position buffer to pool
    pub fn return_position_buffer(&self, buffer: Vec<usize>) {
        let mut pool = self.position_buffers.lock().unwrap();
        
        if pool.len() < self.max_pools_per_type {
            pool.push_back(buffer);
            debug!("🔄 Returned position buffer to pool");
        }
    }
    
    /// Get pool statistics for monitoring
    pub fn get_stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            token_buffers_available: self.token_buffers.lock().unwrap().len(),
            byte_buffers_available: self.byte_buffers.lock().unwrap().len(),
            receiver_buffers_available: self.receiver_buffers.lock().unwrap().len(),
            position_buffers_available: self.position_buffers.lock().unwrap().len(),
            max_pools_per_type: self.max_pools_per_type,
        }
    }
}

/// Statistics for buffer pool monitoring
#[derive(Debug)]
pub struct BufferPoolStats {
    pub token_buffers_available: usize,
    pub byte_buffers_available: usize,
    pub receiver_buffers_available: usize,
    pub position_buffers_available: usize,
    pub max_pools_per_type: usize,
}

impl std::fmt::Display for BufferPoolStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, 
               "BufferPool[tokens:{}/{}, bytes:{}/{}, receivers:{}/{}, positions:{}/{}]",
               self.token_buffers_available, self.max_pools_per_type,
               self.byte_buffers_available, self.max_pools_per_type,
               self.receiver_buffers_available, self.max_pools_per_type,
               self.position_buffers_available, self.max_pools_per_type
        )
    }
}

/// RAII wrapper for automatic buffer return
pub struct TokenBuffer {
    buffer: Option<Vec<u32>>,
    pool: Arc<HotPathBufferPool>,
}

impl TokenBuffer {
    pub fn new(pool: Arc<HotPathBufferPool>, min_capacity: usize) -> Self {
        let buffer = pool.get_token_buffer(min_capacity);
        Self {
            buffer: Some(buffer),
            pool,
        }
    }
    
    pub fn push(&mut self, token: u32) {
        if let Some(ref mut buffer) = self.buffer {
            buffer.push(token);
        }
    }
    
    pub fn extend_from_slice(&mut self, tokens: &[u32]) {
        if let Some(ref mut buffer) = self.buffer {
            buffer.extend_from_slice(tokens);
        }
    }
    
    pub fn len(&self) -> usize {
        self.buffer.as_ref().map_or(0, |b| b.len())
    }
    
    pub fn as_slice(&self) -> &[u32] {
        self.buffer.as_ref().map_or(&[], |b| b.as_slice())
    }
    
    pub fn take(mut self) -> Vec<u32> {
        self.buffer.take().unwrap_or_default()
    }
}

impl Drop for TokenBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.return_token_buffer(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_pool_reuse() {
        let pool = HotPathBufferPool::new(10);
        
        // Get and return buffer
        let buffer = pool.get_token_buffer(100);
        let capacity = buffer.capacity();
        pool.return_token_buffer(buffer);
        
        // Next buffer should reuse the capacity
        let buffer2 = pool.get_token_buffer(50);
        assert!(buffer2.capacity() >= capacity);
        
        let stats = pool.get_stats();
        println!("Pool stats: {}", stats);
    }
    
    #[test]
    fn test_raii_wrapper() {
        let pool = Arc::new(HotPathBufferPool::new(10));
        
        {
            let mut token_buf = TokenBuffer::new(pool.clone(), 100);
            token_buf.push(42);
            token_buf.push(1337);
            assert_eq!(token_buf.len(), 2);
        } // Buffer automatically returned to pool here
        
        let stats = pool.get_stats();
        assert!(stats.token_buffers_available > 0);
    }
}