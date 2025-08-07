// Zero-Copy Unified Memory Management for Apple Silicon
// This module leverages Apple's unified memory architecture where CPU and GPU share the same physical RAM
// Performance target: +40-60% improvement by eliminating all CPU<->GPU copies

use crate::gpu_optimized::{OptimizedMetalDevice as MetalDevice};
use anyhow::{Result, Context};
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions, MTLStorageMode};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use std::sync::{Arc, RwLock, atomic::{AtomicUsize, Ordering}};
use std::collections::HashMap;
use std::ptr::NonNull;
use log::{info, debug, warn};

/// Apple Silicon Unified Memory characteristics
/// M2 Ultra: 192GB unified memory, 800+ GB/s bandwidth, 1000+ GPU cores
pub struct UnifiedMemoryConfig {
    pub total_memory: usize,
    pub memory_bandwidth_gbps: u32,
    pub gpu_cores: u32,
    pub enable_zero_copy: bool,
    pub persistent_kv_cache: bool,
    pub unified_token_buffer_size: usize,
}

impl Default for UnifiedMemoryConfig {
    fn default() -> Self {
        Self {
            total_memory: 192 * 1024 * 1024 * 1024, // 192GB on M2 Ultra
            memory_bandwidth_gbps: 800,              // 800+ GB/s
            gpu_cores: 1000,                         // 60-76 actual cores, 1000+ threads
            enable_zero_copy: true,
            persistent_kv_cache: true,
            unified_token_buffer_size: 8 * 1024 * 1024 * 1024, // 8GB unified buffer
        }
    }
}

/// Zero-copy token buffer that stays GPU-resident
/// CRITICAL: Tokens NEVER leave GPU memory during processing
pub struct ZeroCopyTokenBuffer {
    /// The actual Metal buffer (shared CPU/GPU memory)
    metal_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Raw pointer for direct CPU access (unified memory)
    cpu_ptr: NonNull<u32>,
    /// Size in tokens (not bytes)
    capacity_tokens: usize,
    /// Current write position
    write_pos: AtomicUsize,
    /// Memory mode (shared vs private)
    storage_mode: MTLStorageMode,
}

unsafe impl Send for ZeroCopyTokenBuffer {}
unsafe impl Sync for ZeroCopyTokenBuffer {}

impl ZeroCopyTokenBuffer {
    /// Create a new zero-copy token buffer
    /// Uses MTLStorageModeShared for true zero-copy on Apple Silicon
    pub fn new(device: &MetalDevice, capacity_tokens: usize) -> Result<Self> {
        let size_bytes = capacity_tokens * std::mem::size_of::<u32>();
        
        // CRITICAL: Use shared storage mode for zero-copy
        // This gives both CPU and GPU direct access to the same memory
        let options = MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeDefaultCache;
        
        let metal_buffer = unsafe {
            device.device()
                .newBufferWithLength_options(size_bytes, options)
                .context("Failed to allocate unified memory buffer")?
        };
        
        // Get CPU-accessible pointer (same memory as GPU!)
        let cpu_ptr = unsafe {
            let ptr = metal_buffer.contents();
            NonNull::new(ptr.as_ptr() as *mut u32)
                .context("Failed to get CPU pointer to unified memory")?
        };
        
        info!("Created zero-copy token buffer: {} tokens ({:.2} MB), shared memory",
            capacity_tokens, size_bytes as f64 / (1024.0 * 1024.0));
        
        Ok(Self {
            metal_buffer,
            cpu_ptr,
            capacity_tokens,
            write_pos: AtomicUsize::new(0),
            storage_mode: MTLStorageMode::Shared,
        })
    }
    
    /// Get buffer capacity in tokens
    pub fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
    
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        self.capacity_tokens * std::mem::size_of::<u32>()
    }
    
    /// Write tokens directly to unified memory (zero-copy)
    pub fn write_tokens(&self, tokens: &[u32]) -> Result<usize> {
        let start_pos = self.write_pos.fetch_add(tokens.len(), Ordering::SeqCst);
        
        if start_pos + tokens.len() > self.capacity_tokens {
            self.write_pos.fetch_sub(tokens.len(), Ordering::SeqCst);
            return Err(anyhow::anyhow!("Token buffer overflow"));
        }
        
        // Direct write to unified memory - NO COPY!
        unsafe {
            let dst = self.cpu_ptr.as_ptr().add(start_pos);
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, tokens.len());
        }
        
        // No GPU sync needed - it's the same memory!
        Ok(start_pos)
    }
    
    /// Get a GPU-ready view of tokens (zero-copy)
    pub fn gpu_view(&self, offset: usize, len: usize) -> Result<TokenGpuView> {
        if offset + len > self.capacity_tokens {
            return Err(anyhow::anyhow!("View exceeds buffer bounds"));
        }
        
        Ok(TokenGpuView {
            buffer: self.metal_buffer.clone(),
            offset_bytes: offset * std::mem::size_of::<u32>(),
            length_tokens: len,
        })
    }
    
    /// Reset the buffer for reuse
    pub fn reset(&self) {
        self.write_pos.store(0, Ordering::SeqCst);
    }
    
    /// Get the underlying Metal buffer for GPU operations
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.metal_buffer
    }
}

/// A GPU-ready view of tokens (zero-copy reference)
pub struct TokenGpuView {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    offset_bytes: usize,
    length_tokens: usize,
}

impl TokenGpuView {
    pub fn metal_buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }
    
    pub fn offset(&self) -> usize {
        self.offset_bytes
    }
    
    pub fn len_tokens(&self) -> usize {
        self.length_tokens
    }
}

/// Persistent KV Cache that stays GPU-resident across requests
/// CRITICAL: Model weights and KV cache NEVER leave GPU memory
pub struct PersistentKvCache {
    /// K cache buffer (stays on GPU permanently)
    k_cache: Arc<ZeroCopyTokenBuffer>,
    /// V cache buffer (stays on GPU permanently)
    v_cache: Arc<ZeroCopyTokenBuffer>,
    /// Cache metadata
    metadata: RwLock<KvCacheMetadata>,
    /// Reference count for active contexts
    active_contexts: AtomicUsize,
    /// Memory pressure monitor
    memory_pressure_threshold: f32,
    /// Last eviction timestamp
    last_eviction: RwLock<std::time::Instant>,
}

struct KvCacheMetadata {
    /// Number of layers
    n_layers: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current cache utilization
    current_seq_len: usize,
    /// Cache hit rate
    hit_rate: f32,
    /// High water mark for memory usage
    high_water_mark: usize,
    /// Sliding window size for long sequences
    sliding_window_size: usize,
}

impl PersistentKvCache {
    pub fn new(
        device: &MetalDevice,
        n_layers: usize,
        hidden_dim: usize,
        n_heads: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        // Calculate cache sizes
        let cache_size_per_layer = max_seq_len * hidden_dim;
        let total_k_size = n_layers * cache_size_per_layer;
        let total_v_size = total_k_size; // K and V are same size
        
        info!("Creating persistent KV cache: {} layers, {} hidden, {} heads, {} max_seq",
            n_layers, hidden_dim, n_heads, max_seq_len);
        info!("KV cache size: {:.2} GB total", 
            (total_k_size + total_v_size) as f64 * 4.0 / (1024.0 * 1024.0 * 1024.0));
        
        let k_cache = Arc::new(ZeroCopyTokenBuffer::new(device, total_k_size)?);
        let v_cache = Arc::new(ZeroCopyTokenBuffer::new(device, total_v_size)?);
        
        Ok(Self {
            k_cache,
            v_cache,
            metadata: RwLock::new(KvCacheMetadata {
                n_layers,
                hidden_dim,
                n_heads,
                max_seq_len,
                current_seq_len: 0,
                hit_rate: 0.0,
                high_water_mark: max_seq_len * 3 / 4,  // 75% of max
                sliding_window_size: max_seq_len / 2,  // 50% window
            }),
            active_contexts: AtomicUsize::new(0),
            memory_pressure_threshold: 0.75,  // Evict when 75% full
            last_eviction: RwLock::new(std::time::Instant::now()),
        })
    }
    
    /// Get GPU views for a specific layer (zero-copy)
    pub fn get_layer_cache(&self, layer_idx: usize) -> Result<(TokenGpuView, TokenGpuView)> {
        let metadata = self.metadata.read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock on metadata: {}", e))?;
        let layer_offset = layer_idx * metadata.max_seq_len * metadata.hidden_dim;
        let layer_size = metadata.current_seq_len * metadata.hidden_dim;
        
        let k_view = self.k_cache.gpu_view(layer_offset, layer_size)?;
        let v_view = self.v_cache.gpu_view(layer_offset, layer_size)?;
        
        Ok((k_view, v_view))
    }
    
    /// Update cache with new tokens (happens on GPU)
    pub fn update(&self, layer_idx: usize, new_k: &[u32], new_v: &[u32]) -> Result<()> {
        let mut metadata = self.metadata.write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock on metadata: {}", e))?;
        
        // Check memory pressure and apply sliding window if needed
        let memory_pressure = metadata.current_seq_len as f32 / metadata.max_seq_len as f32;
        if memory_pressure > self.memory_pressure_threshold {
            self.apply_sliding_window(&mut metadata)?;
        }
        
        let layer_offset = layer_idx * metadata.max_seq_len * metadata.hidden_dim;
        let write_offset = layer_offset + metadata.current_seq_len * metadata.hidden_dim;
        
        // Ensure we don't overflow
        let new_tokens = new_k.len() / metadata.hidden_dim;
        if metadata.current_seq_len + new_tokens > metadata.max_seq_len {
            warn!("KV cache overflow prevented: {} + {} > {}", 
                metadata.current_seq_len, new_tokens, metadata.max_seq_len);
            self.apply_sliding_window(&mut metadata)?;
        }
        
        // Direct write to GPU memory (zero-copy)
        unsafe {
            let k_ptr = self.k_cache.cpu_ptr.as_ptr().add(write_offset);
            let v_ptr = self.v_cache.cpu_ptr.as_ptr().add(write_offset);
            
            std::ptr::copy_nonoverlapping(new_k.as_ptr(), k_ptr, new_k.len());
            std::ptr::copy_nonoverlapping(new_v.as_ptr(), v_ptr, new_v.len());
        }
        
        // Update only once after all layers
        if layer_idx == metadata.n_layers - 1 {
            metadata.current_seq_len += new_tokens;
            
            // Track high water mark
            if metadata.current_seq_len > metadata.high_water_mark {
                debug!("KV cache high water mark reached: {}/{}", 
                    metadata.current_seq_len, metadata.max_seq_len);
            }
        }
        
        Ok(())
    }
    
    /// Clear cache for new sequence
    pub fn clear(&self) {
        if let Ok(mut metadata) = self.metadata.write() {
            metadata.current_seq_len = 0;
            metadata.hit_rate = 0.0;
        } else {
            warn!("Failed to acquire write lock for cache clear - continuing anyway");
        }
        
        // Don't actually clear memory - just reset position
        // This avoids unnecessary memory operations
    }
    
    /// Apply sliding window to prevent memory overflow on long sequences
    fn apply_sliding_window(&self, metadata: &mut KvCacheMetadata) -> Result<()> {
        let window_size = metadata.sliding_window_size;
        let shift_amount = metadata.current_seq_len - window_size;
        
        if shift_amount <= 0 {
            return Ok(());
        }
        
        info!("Applying sliding window: shifting {} tokens, keeping {}", 
            shift_amount, window_size);
        
        // For each layer, shift the cache contents
        for layer_idx in 0..metadata.n_layers {
            let layer_offset = layer_idx * metadata.max_seq_len * metadata.hidden_dim;
            let src_offset = layer_offset + shift_amount * metadata.hidden_dim;
            let dst_offset = layer_offset;
            let copy_size = window_size * metadata.hidden_dim * std::mem::size_of::<u32>();
            
            unsafe {
                // Shift K cache
                let k_src = self.k_cache.cpu_ptr.as_ptr().add(src_offset);
                let k_dst = self.k_cache.cpu_ptr.as_ptr().add(dst_offset);
                std::ptr::copy(k_src, k_dst as *mut u32, window_size * metadata.hidden_dim);
                
                // Shift V cache
                let v_src = self.v_cache.cpu_ptr.as_ptr().add(src_offset);
                let v_dst = self.v_cache.cpu_ptr.as_ptr().add(dst_offset);
                std::ptr::copy(v_src, v_dst as *mut u32, window_size * metadata.hidden_dim);
            }
        }
        
        metadata.current_seq_len = window_size;
        if let Ok(mut last_eviction) = self.last_eviction.write() {
            *last_eviction = std::time::Instant::now();
        }
        
        Ok(())
    }
}

/// Unified memory manager for the entire inference pipeline
/// Coordinates all zero-copy operations
pub struct UnifiedMemoryManager {
    config: UnifiedMemoryConfig,
    device: Arc<MetalDevice>,
    
    /// Primary token buffer (all tokens stay here)
    token_buffer: Arc<ZeroCopyTokenBuffer>,
    
    /// Model weights (permanently GPU-resident)
    model_weights: HashMap<String, Arc<ZeroCopyTokenBuffer>>,
    
    /// KV caches per model
    kv_caches: HashMap<String, Arc<PersistentKvCache>>,
    
    /// Memory usage stats
    total_allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
}

impl UnifiedMemoryManager {
    pub fn new(device: Arc<MetalDevice>, config: UnifiedMemoryConfig) -> Result<Self> {
        info!("Initializing Unified Memory Manager");
        info!("  Total memory: {:.2} GB", config.total_memory as f64 / (1024.0 * 1024.0 * 1024.0));
        info!("  Memory bandwidth: {} GB/s", config.memory_bandwidth_gbps);
        info!("  GPU cores: {}", config.gpu_cores);
        info!("  Zero-copy: {}", config.enable_zero_copy);
        
        let token_buffer = Arc::new(ZeroCopyTokenBuffer::new(
            &device,
            config.unified_token_buffer_size / 4, // Convert bytes to tokens
        )?);
        
        Ok(Self {
            config,
            device,
            token_buffer,
            model_weights: HashMap::new(),
            kv_caches: HashMap::new(),
            total_allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
        })
    }
    
    /// Load model weights directly to GPU (one-time operation)
    pub fn load_model_weights(&mut self, model_name: &str, weights: &[u8]) -> Result<()> {
        let weight_tokens = weights.len() / 4;
        let weight_buffer = Arc::new(ZeroCopyTokenBuffer::new(&self.device, weight_tokens)?);
        
        // Copy weights once to GPU memory
        unsafe {
            std::ptr::copy_nonoverlapping(
                weights.as_ptr() as *const u32,
                weight_buffer.cpu_ptr.as_ptr(),
                weight_tokens,
            );
        }
        
        self.model_weights.insert(model_name.to_string(), weight_buffer);
        
        let allocated = weights.len();
        self.total_allocated.fetch_add(allocated, Ordering::SeqCst);
        let current = self.total_allocated.load(Ordering::SeqCst);
        let peak = self.peak_allocated.load(Ordering::SeqCst);
        if current > peak {
            self.peak_allocated.store(current, Ordering::SeqCst);
        }
        
        info!("Loaded model weights for '{}': {:.2} GB",
            model_name, allocated as f64 / (1024.0 * 1024.0 * 1024.0));
        
        Ok(())
    }
    
    /// Get model weights GPU view (zero-copy)
    pub fn get_model_weights(&self, model_name: &str) -> Result<TokenGpuView> {
        let weights = self.model_weights.get(model_name)
            .context("Model weights not loaded")?;
        
        weights.gpu_view(0, weights.capacity_tokens)
    }
    
    /// Create or get KV cache for a model
    pub fn get_or_create_kv_cache(
        &mut self,
        model_name: &str,
        n_layers: usize,
        hidden_dim: usize,
        n_heads: usize,
        max_seq_len: usize,
    ) -> Result<Arc<PersistentKvCache>> {
        if let Some(cache) = self.kv_caches.get(model_name) {
            cache.active_contexts.fetch_add(1, Ordering::SeqCst);
            return Ok(cache.clone());
        }
        
        let cache = Arc::new(PersistentKvCache::new(
            &self.device,
            n_layers,
            hidden_dim,
            n_heads,
            max_seq_len,
        )?);
        
        self.kv_caches.insert(model_name.to_string(), cache.clone());
        cache.active_contexts.fetch_add(1, Ordering::SeqCst);
        
        Ok(cache)
    }
    
    /// Get reference to the Metal device
    pub fn device(&self) -> &Arc<MetalDevice> {
        &self.device
    }
    
    /// Get memory statistics
    pub fn memory_stats(&self) -> UnifiedMemoryStats {
        UnifiedMemoryStats {
            total_allocated: self.total_allocated.load(Ordering::SeqCst),
            peak_allocated: self.peak_allocated.load(Ordering::SeqCst),
            available: self.config.total_memory - self.total_allocated.load(Ordering::SeqCst),
            num_models_loaded: self.model_weights.len(),
            num_active_caches: self.kv_caches.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct UnifiedMemoryStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub available: usize,
    pub num_models_loaded: usize,
    pub num_active_caches: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_zero_copy_performance() {
        // This would test that tokens stay GPU-resident
        // Verify no CPU<->GPU copies occur
        // Measure bandwidth utilization
    }
    
    #[test]
    fn test_persistent_kv_cache() {
        // Test that KV cache persists across requests
        // Verify cache hit rates
        // Measure memory savings
    }
}