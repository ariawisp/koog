// Noesis Runtime - Unified Cognitive Engine
// Where inference, memory, and structure are inseparable

// SAFETY REWRITE: Target stable Rust only - no nightly features
// All performance-critical operations now use safe, stable APIs
// 
// Removed nightly features:
// - portable_simd: Use standard operations, compiler optimizes automatically
// - allocator_api: Use Box and Vec, which are optimized
// - new_uninit: Use Vec::with_capacity for safe initialization
// - get_mut_unchecked: Removed all unchecked access
// - core_intrinsics: Replaced with standard library functions
// - All others: Either unused or have stable equivalents

// NOTE: Some unsafe code is still needed for Metal FFI bindings
// But all application logic should be safe

use anyhow::{Result, Context};
use dashmap::DashMap;
use arc_swap::ArcSwap;
use std::sync::Arc;
use tracing::info;
// JNI functions now in jni/ module
use chrono::Utc;

// Import unified error types
use crate::errors::NoesisResult;

// Import unified buffer types
use crate::buffer::unified_gpu_buffer::GpuBuffer;

// Core modules
pub mod constants;       // NEW: Centralized configuration constants
pub mod error;
pub mod errors;  // NEW: Unified error hierarchy
pub mod gpu_optimized;  // LEGACY: High-performance buffer pool (being replaced)
pub mod buffer;  // NEW: PHASE 8C Unified cross-platform buffer system
pub mod unified_buffer;  // LEGACY: Will be deprecated
pub mod unified_memory;
pub mod memory;
pub mod inference;
pub mod harmony;
pub mod backend;
pub mod streaming_output;       // Real-time terminal streaming for harmony results
pub mod streaming;              // Channel-aware token streaming with optimized routing
// Re-enabled after gpu migration
pub mod cognitive; // TOKEN-NATIVE COGNITIVE PROCESSING - All in Rust
pub mod parsing; // Text parsing module
mod safety_tests; // Safety tests for the emergency rewrite
pub mod real_benchmarks; // Real benchmarks with honest performance reporting
pub mod jni; // PHASE 8D: Extracted JNI bindings

// Re-export all JNI functions for backwards compatibility
pub use jni::*;

// Generated FlatBuffer schemas
mod events_generated;
mod noesis_request_generated; // Zero-copy FlatBuffers for requests/responses

use gpu_optimized::{OptimizedBufferPool, OptimizedMetalDevice, OptimizedBuffer, PerformanceTarget};
use buffer::{UnifiedGpuBufferPool};
use backend::{UnifiedBackend, create_unified_backend};
use memory::{TokenGraph, TemporalIndex, ChannelRouter};
use memory::channels::RoutingRules;
use inference::MetalInferenceEngine;
use harmony::{HarmonyEncoder};
// Use openai_harmony StreamableParser directly at call sites as needed
use unified_buffer::{BufferView};
use streaming_output::{StreamingOutputManager, create_minimal_streaming_output};
use std::sync::RwLock;

/// The unified Noesis Runtime - a single cognitive engine
pub struct NoesisRuntime {
    // Zero-cost compile-time dispatch backend
    unified_backend: UnifiedBackend,
    
    // High-performance buffer pool
    buffer_pool: Arc<OptimizedBufferPool>,
    
    // Unified cross-platform buffer pool
    unified_buffer_pool: Arc<UnifiedGpuBufferPool>,
    
    // Inference subsystem
    inference_engine: Arc<MetalInferenceEngine>,
    model_cache: DashMap<String, u64>,
    
    // Memory subsystem
    token_graph: Arc<RwLock<TokenGraph>>,
    temporal_index: Arc<RwLock<TemporalIndex>>,
    channel_router: Arc<ChannelRouter>,
    
    // Harmony subsystem
    encoder: Arc<HarmonyEncoder>,
    
    // Unified state
    active_context: ArcSwap<CognitiveContext>,
    checkpoints: DashMap<String, BinaryCheckpoint>,
    
    // Real-time streaming output
    streaming_output: Option<StreamingOutputManager>,
}

/// A cognitive context - the current state of reasoning
#[derive(Clone)]
pub struct CognitiveContext {
    // Current token stream (high-performance GPU buffer)
    tokens: Arc<OptimizedBuffer>,
    
    // Channel-separated views into the token stream
    analysis_view: BufferView,
    commentary_view: BufferView,
    final_view: BufferView,
    
    // Temporal position
    timestamp: i64,
    
    // Graph position
    current_node: Option<petgraph::graph::NodeIndex>,
    
    // Active model for inference
    model_handle: Option<u64>,
}

/// Binary checkpoint for fork/merge operations
pub struct BinaryCheckpoint {
    // Raw token data (can be mmap'd)
    token_data: Vec<u8>,
    
    // Graph state
    graph_snapshot: Vec<u8>,
    
    // Metadata
    timestamp: i64,
    channels: Vec<Channel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Channel {
    Analysis,
    Commentary,
    Final,
}

impl Channel {
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => Channel::Analysis,
            1 => Channel::Commentary,
            2 => Channel::Final,
            _ => Channel::Final,
        }
    }
    
    pub fn to_byte(&self) -> u8 {
        match self {
            Channel::Analysis => 0,
            Channel::Commentary => 1,
            Channel::Final => 2,
        }
    }
    
    pub fn security_level(&self) -> memory::channels::SecurityLevel {
        use memory::channels::SecurityLevel;
        match self {
            Channel::Analysis => SecurityLevel::Internal,
            Channel::Commentary => SecurityLevel::Restricted,
            Channel::Final => SecurityLevel::Public,
        }
    }
}

impl NoesisRuntime {
    /// Initialize the unified runtime with zero-cost backend
    pub fn new() -> Result<Self> {
        // Initialize unified backend with zero-cost dispatch
        let mut unified_backend = create_unified_backend()
            .context("Failed to create unified GPU backend")?;
        
        // Load essential pipelines for performance
        unified_backend.load_essential_pipelines()
            .context("Failed to load essential GPU pipelines")?;
        
        // Get buffer pool from unified backend
        let buffer_pool = unified_backend.get_buffer_pool();
        
        // Create unified buffer pool for cross-platform operations
        let unified_buffer_pool = Arc::new(unified_backend.create_unified_buffer_pool(100));
        
        // Pre-warm unified memory for optimal startup
        buffer_pool.prewarm_unified_memory()?;
        
        // Initialize Harmony encoder
        let encoder = Arc::new(HarmonyEncoder::new("o200k_harmony")?);
        
        // Initialize streamlined inference engine
        let mut inference_engine_temp = MetalInferenceEngine::new_without_gptoss()?;
        inference_engine_temp.set_harmony_encoder(encoder.clone());
        let inference_engine = Arc::new(inference_engine_temp);
        
        let model_cache = DashMap::new();
        
        // Initialize memory subsystem
        let token_graph = Arc::new(RwLock::new(TokenGraph::new()));
        let temporal_index = Arc::new(RwLock::new(TemporalIndex::new()));
        let channel_router = Arc::new(ChannelRouter::new(RoutingRules::default()));
        
        // Create initial context with optimized buffer
        let initial_tokens = buffer_pool.allocate_token_buffer(crate::constants::buffer::INITIAL_TOKEN_BUFFER_SIZE)?;
        let initial_context = CognitiveContext {
            tokens: Arc::new(initial_tokens),
            analysis_view: BufferView::new(0, 0),
            commentary_view: BufferView::new(0, 0),
            final_view: BufferView::new(0, 0),
            timestamp: Utc::now().timestamp_nanos_opt().unwrap_or(0),
            current_node: None,
            model_handle: None,
        };
        
        // Initialize real-time streaming output (optional - can fail without breaking runtime)
        let streaming_output = match create_minimal_streaming_output() {
            Ok(output_manager) => {
                eprintln!("[NoesisRuntime] ✅ Real-time harmony streaming enabled");
                Some(output_manager)
            }
            Err(e) => {
                eprintln!("[NoesisRuntime] ⚠️  Real-time streaming disabled: {}", e);
                None
            }
        };
        
        Ok(NoesisRuntime {
            unified_backend,
            buffer_pool,
            unified_buffer_pool,
            inference_engine,
            model_cache,
            token_graph,
            temporal_index,
            channel_router,
            encoder,
            active_context: ArcSwap::from_pointee(initial_context),
            checkpoints: DashMap::new(),
            streaming_output,
        })
    }
    
    /// Load a model for inference
    pub fn load_model(&self, model_path: &str) -> Result<u64> {
        // Check cache first
        if let Some(handle) = self.model_cache.get(model_path) {
            return Ok(*handle);
        }
        
        // Load model into GPU memory
        // PERFORMANCE: Use ahash for ~3x faster hashing than DefaultHasher
        use std::hash::{Hash, Hasher};
        use ahash::RandomState;
        
        let hash_state = RandomState::with_seeds(
            crate::constants::performance::DEFAULT_HASH_SEED, 
            crate::constants::performance::DEFAULT_HASH_SEED,
            crate::constants::performance::DEFAULT_HASH_SEED,
            crate::constants::performance::DEFAULT_HASH_SEED
        );
        let handle = hash_state.hash_one(model_path); // Direct hash computation
        
        info!("[NoesisRuntime] Loading model: {} (handle: {})", model_path, handle);
        
        self.model_cache.insert(model_path.to_string(), handle);
        
        // Update active context
        let ctx = self.active_context.load();
        let new_ctx = CognitiveContext {
            tokens: ctx.tokens.clone(),
            analysis_view: ctx.analysis_view.clone(),
            commentary_view: ctx.commentary_view.clone(),
            final_view: ctx.final_view.clone(),
            timestamp: ctx.timestamp,
            current_node: ctx.current_node,
            model_handle: Some(handle),
        };
        self.active_context.store(Arc::new(new_ctx));
        
        Ok(handle)
    }
    
    /// Process tokens through backend with real GPU inference
    pub fn process_tokens(&self, input_tokens: &[u32]) -> Result<Vec<u32>> {
        if input_tokens.is_empty() {
            return Ok(Vec::new());
        }
        
        // Get current active model handle
        let ctx = self.active_context.load();
        let model_handle = ctx.model_handle
            .ok_or_else(|| anyhow::anyhow!("No model loaded - call load_model() first"))?;
        
        // Create input buffer from tokens
        let input_buffer = self.buffer_pool.allocate_token_buffer(input_tokens.len())?;
        input_buffer.write_tokens(input_tokens)?;
        
        // Run real GPU inference with configured defaults
        use crate::constants::inference::*;
        
        let output_buffer = self.inference_engine.infer(
            model_handle,
            &input_buffer,
            DEFAULT_MAX_TOKENS,
            DEFAULT_TEMPERATURE,
            DEFAULT_TOP_P
        )?;
        
        // Extract generated tokens from output buffer
        let output_size = output_buffer.size() / crate::constants::buffer::BYTES_PER_TOKEN;
        let output_tokens = output_buffer.read_tokens(output_size)?;
        
        // Route through memory graph for cognitive processing
        self.route_tokens_to_memory(&output_tokens)?;
        
        Ok(output_tokens)
    }
    
    /// Route generated tokens through memory system (streamlined)
    fn route_tokens_to_memory(&self, tokens: &[u32]) -> Result<()> {
        let timestamp = Utc::now().timestamp_nanos_opt().unwrap_or(0);
        
        // Simple channel routing (no complex parsing for performance)
        let channel = Channel::Final; // Default to final channel
        
        // Add to token graph
        let buffer_data = tokens.iter()
            .flat_map(|&token| token.to_le_bytes())
            .collect::<Vec<u8>>();
        
        let node_idx = self.token_graph.write().unwrap().add_node(
            buffer_data,
            channel,
            timestamp
        )?;
        
        // Update temporal index
        self.temporal_index.write().unwrap().insert(node_idx, timestamp);
        
        Ok(())
    }
    
    /// Fork the current cognitive state for parallel exploration
    pub fn fork(&self, checkpoint_id: &str) -> Result<()> {
        let ctx = self.active_context.load();
        
        // Create binary checkpoint (zero-copy where possible)
        let checkpoint = BinaryCheckpoint {
            token_data: vec![], // TODO: implement token serialization
            graph_snapshot: self.token_graph.read().unwrap().serialize()?,
            timestamp: ctx.timestamp,
            channels: vec![Channel::Analysis, Channel::Commentary, Channel::Final],
        };
        
        self.checkpoints.insert(checkpoint_id.to_string(), checkpoint);
        Ok(())
    }
    
    /// Merge a forked state back into main
    pub fn merge(&self, checkpoint_id: &str) -> Result<()> {
        let checkpoint = self.checkpoints.get(checkpoint_id)
            .context("Checkpoint not found")?;
        
        // Find contradictions between states using GPU
        let current_embedding = self.token_graph.read().unwrap().compute_embedding()?;
        let checkpoint_graph = TokenGraph::deserialize(&checkpoint.graph_snapshot)?;
        let checkpoint_embedding = checkpoint_graph.compute_embedding()?;
        
        let contradictions = self.inference_engine.detect_contradictions(
            &current_embedding,
            &checkpoint_embedding
        )?;
        
        if !contradictions.is_empty() {
            // Resolve contradictions through inference
            self.resolve_contradictions(contradictions)?;
        }
        
        // Merge graphs
        self.token_graph.write().unwrap().merge(&checkpoint_graph)?;
        
        Ok(())
    }
    
    /// Query memory with semantic search
    pub fn query_memory(&self, query_tokens: &[u32], k: usize) -> Result<Vec<MemoryResult>> {
        // Embed query using GPU
        let query_buffer = self.buffer_pool.allocate_by_size(query_tokens.len() * 4)?;
        // query_buffer.write(query_tokens)?; // TODO: implement write method
        
        // TODO: Implement proper embedding call
        let query_embedding = vec![0.0f32; 512]; // Placeholder embedding
        
        // Search graph using GPU-accelerated k-NN
        let results = self.token_graph.read().unwrap().search_knn(&query_embedding, k)?;
        
        Ok(results.into_iter().map(|(node_idx, score)| {
            MemoryResult {
                tokens: self.token_graph.read().unwrap().get_node_tokens(node_idx).unwrap_or_default(),
                score,
                timestamp: self.temporal_index.read().unwrap().get_timestamp(node_idx).unwrap_or(0),
            }
        }).collect())
    }
    
    /// Get GPU memory statistics
    pub fn get_stats(&self) -> RuntimeStats {
        RuntimeStats {
            gpu_memory_used: 0, // TODO: implement stats
            gpu_memory_total: 0, // TODO: implement stats
            graph_nodes: self.token_graph.read().unwrap().node_count(),
            graph_edges: self.token_graph.read().unwrap().edge_count(),
            cached_models: self.model_cache.len(),
            active_parsers: 0,
        }
    }
    
    fn resolve_contradictions(&self, contradictions: Vec<Contradiction>) -> Result<()> {
        // Use inference to resolve contradictions
        // This is where the magic happens - letting the model decide
        for contradiction in contradictions {
            let resolution_prompt = self.encoder.encode_contradiction_resolution(
                &contradiction.context_a,
                &contradiction.context_b
            )?;
            
            // Create a UnifiedBuffer from the resolution bytes
            let resolution_buffer = self.buffer_pool.allocate_by_size(resolution_prompt.len())?;
            // resolution_buffer.write_bytes(&resolution_prompt)?; // TODO: implement write_bytes
            
            // TODO: Implement proper inference call
            let resolution: Vec<u32> = vec![]; // Placeholder
            
            // TODO: Apply resolution to graph (requires proper UnifiedBuffer type)
            // self.token_graph.write().unwrap().apply_resolution(contradiction.node_a, contradiction.node_b, resolution)?;
        }
        Ok(())
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct MemoryResult {
    pub tokens: Vec<u32>,
    pub score: f32,
    pub timestamp: i64,
}

pub struct RuntimeStats {
    pub gpu_memory_used: usize,
    pub gpu_memory_total: usize,
    pub graph_nodes: usize,
    pub graph_edges: usize,
    pub cached_models: usize,
    pub active_parsers: usize,
}

pub struct Contradiction {
    pub node_a: petgraph::graph::NodeIndex,
    pub node_b: petgraph::graph::NodeIndex,
    pub context_a: Vec<u32>,
    pub context_b: Vec<u32>,
}

impl NoesisRuntime {
    /// Create unified GPU buffer for zero-copy operations
    pub fn create_unified_buffer(&self, token_count: usize) -> NoesisResult<GpuBuffer> {
        self.unified_backend.allocate_token_buffer(token_count, &self.unified_buffer_pool)
    }
    
    /// Get unified buffer pool statistics
    pub fn unified_buffer_stats(&self) -> crate::buffer::UnifiedBufferPoolStats {
        self.unified_buffer_pool.statistics()
    }
}

impl Drop for NoesisRuntime {
    fn drop(&mut self) {
        eprintln!("[NoesisRuntime] Starting cleanup...");
        
        // No internal streaming parsers to clear
        // Clear checkpoints
        self.checkpoints.clear();
        
        // Clear model cache (just the handles, models are in inference_engine)
        self.model_cache.clear();
        
        // Force drop the Arc references by replacing with dummy values
        // This ensures cleanup happens in the right order
        let _ = std::mem::replace(&mut self.channel_router, Arc::new(ChannelRouter::new(RoutingRules::default())));
        let _ = std::mem::replace(&mut self.temporal_index, Arc::new(RwLock::new(TemporalIndex::new())));
        let _ = std::mem::replace(&mut self.token_graph, Arc::new(RwLock::new(TokenGraph::new())));
        
        // Small delay to ensure Metal cleanup completes
        std::thread::sleep(std::time::Duration::from_millis(crate::constants::timing::METAL_CLEANUP_DELAY_MS));
        
        // The Arc<MetalInferenceEngine> will be dropped last
        
        eprintln!("[NoesisRuntime] Cleanup completed");
    }
}

// ============== JNI Interface ==============
// All JNI functions now extracted to jni/ module for better organization

// Re-export JNI functions for backwards compatibility
pub use crate::jni::runtime::*;
pub use crate::jni::streaming::*;
pub use crate::jni::memory::*;
pub use crate::jni::utils::*;
