// Inference subsystem re-exports
pub mod noesis_metal;
pub mod kernels;
pub mod speculative;
pub mod buffer_pool;
pub mod parallel_contexts;
pub mod fused_kernels;
pub mod channel_aware_speculation;
pub mod harmony_token_recycling;
pub mod variable_effort_batching;
pub mod off_by_one_attention;

pub use noesis_metal::MetalInferenceEngine;
pub use kernels::{MetalKernelLibrary, SamplingAccelerator, SamplingStrategy, SpeculativeDecoder};
pub use speculative::{SpeculativeEngine, SpeculativeConfig};
pub use buffer_pool::{HotPathBufferPool, TokenBuffer, BufferPoolStats};
pub use parallel_contexts::{ParallelContextManager, ParallelContextHandle, ParallelContextMetrics, PoolStatus};
pub use fused_kernels::{FusedKernelManager, FusedKernelConfig, FusedKernelMetrics, FusedGenerationParams};
pub use channel_aware_speculation::{ChannelAwareSpeculativeEngine, ChannelAwareConfig};
pub use harmony_token_recycling::{HarmonyTokenRecycler, TokenRecyclingConfig, RecyclingTokenStream, TokenRecyclingExt};
pub use variable_effort_batching::{
    VariableEffortBatcher, VariableEffortConfig, ReasoningEffort, 
    EffortAwareRequest, BatchedResponse, BatchingMetricsSnapshot
};
pub use off_by_one_attention::{OffByOneAttentionOptimizer, OffByOneConfig, OffByOneMetrics};