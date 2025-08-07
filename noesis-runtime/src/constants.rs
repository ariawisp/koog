// Constants module for Noesis Runtime
// 
// Centralizes all magic numbers and configuration constants
// to improve code maintainability and documentation

/// Default inference parameters
pub mod inference {
    /// Default maximum tokens to generate per inference request
    pub const DEFAULT_MAX_TOKENS: usize = 50;
    
    /// Default temperature for inference (0.0 = deterministic, 1.0 = very random)
    pub const DEFAULT_TEMPERATURE: f32 = 0.8;
    
    /// Default top-p value for nucleus sampling
    pub const DEFAULT_TOP_P: f32 = 0.95;
    
    /// Simple response length calculation factor
    pub const RESPONSE_LENGTH_FACTOR: usize = 10;
    
    /// Minimum response tokens to generate
    pub const MIN_RESPONSE_TOKENS: usize = 1;
    
    /// Maximum response tokens for simple generation
    pub const MAX_SIMPLE_RESPONSE_TOKENS: usize = 50;
}

/// Buffer and memory management constants
pub mod buffer {
    /// Initial token buffer size in bytes (1MB)
    pub const INITIAL_TOKEN_BUFFER_SIZE: usize = 1024 * 256;
    
    /// Bytes per token (u32)
    pub const BYTES_PER_TOKEN: usize = 4;
}

/// Timing and cleanup constants
pub mod timing {
    /// Metal cleanup delay in milliseconds
    pub const METAL_CLEANUP_DELAY_MS: u64 = 5;
    
    /// Default inference timeout in milliseconds
    pub const INFERENCE_TIMEOUT_MS: u64 = 30_000; // 30 seconds
}

/// Channel and cognitive processing constants
pub mod cognitive {
    /// Default embedding dimension
    pub const DEFAULT_EMBEDDING_DIM: usize = 512;
    
    /// Maximum contradiction detection iterations
    pub const MAX_CONTRADICTION_ITERATIONS: usize = 10;
    
    /// Default channel routing timeout in milliseconds
    pub const CHANNEL_ROUTING_TIMEOUT_MS: u64 = 1000;
}

/// GPU and parallel processing constants
pub mod gpu {
    /// Default number of parallel GPU workers
    pub const DEFAULT_WORKER_COUNT: usize = 8;
    
    /// GPU buffer alignment requirement (bytes)
    pub const GPU_BUFFER_ALIGNMENT: usize = 16;
    
    /// Default GPU memory pool size
    pub const DEFAULT_GPU_POOL_SIZE: usize = 100;
}

/// Hash and performance constants
pub mod performance {
    /// Token space modulo for simple generation (GPT-style vocab size approximation)
    pub const TOKEN_VOCAB_SIZE: u32 = 50000;
    
    /// Default hash seed for reproducible behavior
    pub const DEFAULT_HASH_SEED: u64 = 0x517cc1b727220a95;
}

/// GPT-OSS Mixture of Experts constants
pub mod moe {
    /// Total number of experts in GPT-OSS architecture
    pub const TOTAL_EXPERTS: usize = 128;
    
    /// Number of active experts per token (key optimization opportunity)
    pub const ACTIVE_EXPERTS_PER_TOKEN: usize = 4;
    
    /// Expert sparsity ratio (only 3.125% of experts active)
    pub const SPARSITY_RATIO: f32 = ACTIVE_EXPERTS_PER_TOKEN as f32 / TOTAL_EXPERTS as f32;
    
    /// Expert routing threshold for cache hit optimization  
    pub const ROUTING_THRESHOLD: f32 = 0.1;
    
    /// Maximum cached expert combinations (for frequent patterns)
    pub const MAX_EXPERT_CACHE_SIZE: usize = 256;
}