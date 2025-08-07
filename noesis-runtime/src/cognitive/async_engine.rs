// Fully async cognitive processing layer

use crate::cognitive::{
    Channel, HarmonyChannelRouter, HarmonyChannelContext,
    ContradictionAlert, SemanticContext
};
use crate::errors::{NoesisError, NoesisResult};
use crate::gpu_optimized::OptimizedBuffer;
use crate::memory::TokenGraph;
use crate::backend::UnifiedBackend;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant};
use futures_util::stream::{Stream, StreamExt};
use tracing::{debug, info, warn, error, instrument};

/// Fully async cognitive engine
pub struct AsyncCognitiveEngine {
    // Harmony integration (async-friendly)
    harmony_router: Arc<Mutex<HarmonyChannelRouter>>,
    
    // Cross-platform GPU backend (async-wrapped)
    backend: Arc<UnifiedBackend>,
    
    // Async cognitive processors
    contradiction_processor: Arc<AsyncContradictionProcessor>,
    semantic_processor: Arc<AsyncSemanticProcessor>,
    state_manager: Arc<AsyncStateManager>,
    safety_filter: Arc<AsyncSafetyFilter>,
    
    // Token graph with async access
    token_graph: Arc<TokenGraph>,
    
    // Async performance metrics
    metrics: Arc<RwLock<AsyncEngineMetrics>>,
    
    // Async processing configuration
    config: AsyncProcessingConfig,
}

impl AsyncCognitiveEngine {
    /// Create new fully async cognitive engine
    /// 
    /// BREAKING CHANGE: Constructor is now async and returns Result
    pub async fn new(
        backend: Arc<UnifiedBackend>,
        token_graph: Arc<TokenGraph>,
        config: Option<AsyncProcessingConfig>,
    ) -> NoesisResult<Self> {
        
        info!("🚀 Initializing AsyncCognitiveEngine with full async architecture");
        let start_time = Instant::now();
        
        // Initialize harmony router (async-safe)
        let harmony_router = Arc::new(Mutex::new(HarmonyChannelRouter::new()?));
        
        // Initialize async cognitive processors
        let contradiction_processor = Arc::new(
            AsyncContradictionProcessor::new(backend.clone()).await?
        );
        let semantic_processor = Arc::new(
            AsyncSemanticProcessor::new(backend.clone()).await?
        );
        let state_manager = Arc::new(
            AsyncStateManager::new(token_graph.clone(), backend.clone()).await?
        );
        let safety_filter = Arc::new(
            AsyncSafetyFilter::new(backend.clone()).await?
        );
        
        let config = config.unwrap_or_default();
        let metrics = Arc::new(RwLock::new(AsyncEngineMetrics::new()));
        
        let init_time = start_time.elapsed();
        info!("✅ AsyncCognitiveEngine initialized in {:?}", init_time);
        
        Ok(Self {
            harmony_router,
            backend,
            contradiction_processor,
            semantic_processor,
            state_manager,
            safety_filter,
            token_graph,
            metrics,
            config,
        })
    }
    
    /// Fully async token processing with harmony integration
    /// 
    /// BREAKING CHANGE: All processing is now async with proper error handling
    /// and concurrent processing capabilities
    #[instrument(skip(self, token_buffer), fields(token = %token))]
    pub async fn process_token_async(
        &self,
        token: u32,
        token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<AsyncTokenResult> {
        
        let start_time = Instant::now();
        let mut metrics = self.metrics.write().await;
        metrics.tokens_processed += 1;
        drop(metrics);
        
        // 1. HARMONY PROCESSING (async-safe with mutex)
        let harmony_context = {
            let mut router = self.harmony_router.lock().await;
            match router.process_token(token) {
                Ok(ctx) => ctx,
                Err(e) => {
                    error!("🚨 Harmony processing failed for token {}: {}", token, e);
                    let mut metrics = self.metrics.write().await;
                    metrics.harmony_errors += 1;
                    
                    return Ok(AsyncTokenResult::harmony_error(token, e));
                }
            }
        };
        
        // 2. PARALLEL ASYNC PROCESSING
        // Run all cognitive analyses concurrently for maximum performance
        let (contradiction_result, semantic_result, safety_result) = tokio::try_join!(
            self.process_contradictions_async(token, &harmony_context, token_buffer),
            self.process_semantics_async(token, &harmony_context, token_buffer),
            self.process_safety_async(token, &harmony_context)
        )?;
        
        // 3. STATE UPDATE (async)
        self.update_state_async(token, &harmony_context, &contradiction_result, &semantic_result).await?;
        
        // 4. ACTION DETERMINATION
        let async_action = self.determine_async_action(&harmony_context, &safety_result, &contradiction_result).await;
        
        // 5. METRICS UPDATE
        let processing_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        metrics.total_processing_time += processing_time;
        metrics.channel_transitions += if harmony_context.current_channel != harmony_context.previous_channel.unwrap_or(harmony_context.current_channel) { 1 } else { 0 };
        
        debug!("✅ Token {} processed in {:?}", token, processing_time);
        
        Ok(AsyncTokenResult {
            token,
            harmony_context,
            contradiction_result,
            semantic_result,
            safety_result,
            async_action,
            processing_time,
            async_metadata: AsyncProcessingMetadata {
                concurrent_processing: true,
                harmony_validated: true,
                tokio_context: true,
            },
        })
    }
    
    /// Stream tokens for real-time processing
    /// 
    /// Async stream processing with backpressure handling
    pub fn process_token_stream<S>(&self, token_stream: S) -> impl Stream<Item = NoesisResult<AsyncTokenResult>> + '_
    where
        S: Stream<Item = (u32, OptimizedBuffer)> + Send + 'static,
    {
        token_stream
            .map(move |(token, buffer)| {
                let engine = self;
                async move {
                    engine.process_token_async(token, &buffer).await
                }
            })
            .buffer_unordered(self.config.max_concurrent_tokens) // Concurrent processing
    }
    
    /// PARALLEL ASYNC CONTRADICTION PROCESSING
    async fn process_contradictions_async(
        &self,
        token: u32,
        harmony_context: &HarmonyChannelContext,
        token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<AsyncContradictionResult> {
        
        // Direct async processing without spawn (avoid lifetime issues)
        self.contradiction_processor
            .check_contradictions_harmony_async(token, harmony_context, token_buffer)
            .await
    }
    
    /// PARALLEL ASYNC SEMANTIC PROCESSING
    async fn process_semantics_async(
        &self,
        token: u32,
        harmony_context: &HarmonyChannelContext,
        token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<AsyncSemanticResult> {
        
        // Direct async processing without spawn (avoid lifetime issues)
        self.semantic_processor
            .process_semantics_harmony_async(token, harmony_context, token_buffer)
            .await
    }
    
    /// PARALLEL ASYNC SAFETY PROCESSING
    async fn process_safety_async(
        &self,
        token: u32,
        harmony_context: &HarmonyChannelContext,
    ) -> NoesisResult<AsyncSafetyResult> {
        
        // Direct async processing without spawn (avoid lifetime issues)
        self.safety_filter
            .assess_safety_harmony_async(token, harmony_context)
            .await
    }
    
    /// ASYNC STATE UPDATE
    async fn update_state_async(
        &self,
        token: u32,
        harmony_context: &HarmonyChannelContext,
        contradiction_result: &AsyncContradictionResult,
        semantic_result: &AsyncSemanticResult,
    ) -> NoesisResult<()> {
        
        self.state_manager.update_state_harmony_async(
            token, 
            harmony_context, 
            contradiction_result, 
            semantic_result
        ).await
    }
    
    /// ASYNC ACTION DETERMINATION
    async fn determine_async_action(
        &self,
        harmony_context: &HarmonyChannelContext,
        safety_result: &AsyncSafetyResult,
        contradiction_result: &AsyncContradictionResult,
    ) -> AsyncCognitiveAction {
        
        // Priority 1: Safety (immediate response)
        if !safety_result.is_safe {
            return AsyncCognitiveAction::BlockUnsafe {
                reason: safety_result.risk_description.clone(),
                channel: harmony_context.current_channel,
                immediate: true,
            };
        }
        
        // Priority 2: Critical contradictions (may require async resolution)
        if contradiction_result.has_critical_contradictions() {
            return AsyncCognitiveAction::ResolveContradictionAsync {
                contradiction_ids: contradiction_result.get_critical_ids(),
                resolution_strategy: contradiction_result.suggested_strategy(),
                requires_user_input: false,
            };
        }
        
        // Priority 3: Normal processing based on harmony channel
        match harmony_context.current_channel {
            Channel::Analysis => AsyncCognitiveAction::ProcessInternallyAsync {
                harmony_validated: harmony_context.harmony_channel.is_some(),
                background_processing: true,
            },
            Channel::Commentary => AsyncCognitiveAction::ProcessWithVisibilityAsync {
                harmony_validated: harmony_context.harmony_channel.is_some(),
                recipient: harmony_context.recipient.clone(),
                user_notification: true,
            },
            Channel::Final => AsyncCognitiveAction::ProcessAsSafeOutputAsync {
                harmony_validated: harmony_context.harmony_channel.is_some(),
                immediate_display: true,
            },
        }
    }
    
    /// Get async processing metrics
    pub async fn get_metrics(&self) -> AsyncEngineMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Reset engine for new conversation (async)
    pub async fn reset_for_new_conversation(&self) -> NoesisResult<()> {
        // Reset all components concurrently
        tokio::try_join!(
            async { self.harmony_router.lock().await.reset() },
            self.state_manager.reset_async(),
            self.contradiction_processor.reset_async(),
            self.semantic_processor.reset_async(),
            self.safety_filter.reset_async()
        )?;
        
        // Reset metrics
        *self.metrics.write().await = AsyncEngineMetrics::new();
        
        info!("🔄 AsyncCognitiveEngine reset for new conversation");
        Ok(())
    }
    
    /// Graceful shutdown with cleanup
    pub async fn shutdown(&self) -> NoesisResult<()> {
        info!("🛑 Shutting down AsyncCognitiveEngine gracefully");
        
        // Give ongoing tasks time to complete
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Final metrics log
        let metrics = self.get_metrics().await;
        info!("📊 Final metrics: {:?}", metrics);
        
        Ok(())
    }
}

// ASYNC RESULT TYPES

#[derive(Debug, Clone)]
pub struct AsyncTokenResult {
    pub token: u32,
    pub harmony_context: HarmonyChannelContext,
    pub contradiction_result: AsyncContradictionResult,
    pub semantic_result: AsyncSemanticResult,
    pub safety_result: AsyncSafetyResult,
    pub async_action: AsyncCognitiveAction,
    pub processing_time: Duration,
    pub async_metadata: AsyncProcessingMetadata,
}

impl AsyncTokenResult {
    pub fn harmony_error(token: u32, error: NoesisError) -> Self {
        Self {
            token,
            harmony_context: HarmonyChannelContext::error_fallback(token),
            contradiction_result: AsyncContradictionResult::default(),
            semantic_result: AsyncSemanticResult::default(),
            safety_result: AsyncSafetyResult::safe_fallback(),
            async_action: AsyncCognitiveAction::ProcessAsSafeOutputAsync {
                harmony_validated: false,
                immediate_display: true,
            },
            processing_time: Duration::from_millis(0),
            async_metadata: AsyncProcessingMetadata {
                concurrent_processing: false,
                harmony_validated: false,
                tokio_context: true,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsyncContradictionResult {
    pub contradictions: Vec<ContradictionAlert>,
    pub processing_time: Duration,
    pub async_processed: bool,
}

impl AsyncContradictionResult {
    pub fn has_critical_contradictions(&self) -> bool {
        self.contradictions.iter().any(|c| c.severity > 0.8)
    }
    
    pub fn get_critical_ids(&self) -> Vec<String> {
        self.contradictions
            .iter()
            .filter(|c| c.severity > 0.8)
            .map(|c| c.contradiction_id.clone())
            .collect()
    }
    
    pub fn suggested_strategy(&self) -> ResolutionStrategy {
        if self.contradictions.len() > 3 {
            ResolutionStrategy::ForkAndMerge
        } else {
            ResolutionStrategy::LinearResolve
        }
    }
}

impl Default for AsyncContradictionResult {
    fn default() -> Self {
        Self {
            contradictions: vec![],
            processing_time: Duration::from_millis(0),
            async_processed: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsyncSemanticResult {
    pub semantic_context: SemanticContext,
    pub processing_time: Duration,
    pub async_processed: bool,
}

impl Default for AsyncSemanticResult {
    fn default() -> Self {
        Self {
            semantic_context: SemanticContext::default(),
            processing_time: Duration::from_millis(0),
            async_processed: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AsyncSafetyResult {
    pub is_safe: bool,
    pub risk_description: String,
    pub processing_time: Duration,
    pub async_processed: bool,
}

impl AsyncSafetyResult {
    pub fn safe_fallback() -> Self {
        Self {
            is_safe: true,
            risk_description: "Safe fallback - no assessment".to_string(),
            processing_time: Duration::from_millis(0),
            async_processed: false,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AsyncCognitiveAction {
    ProcessInternallyAsync {
        harmony_validated: bool,
        background_processing: bool,
    },
    ProcessWithVisibilityAsync {
        harmony_validated: bool,
        recipient: Option<String>,
        user_notification: bool,
    },
    ProcessAsSafeOutputAsync {
        harmony_validated: bool,
        immediate_display: bool,
    },
    BlockUnsafe {
        reason: String,
        channel: Channel,
        immediate: bool,
    },
    ResolveContradictionAsync {
        contradiction_ids: Vec<String>,
        resolution_strategy: ResolutionStrategy,
        requires_user_input: bool,
    },
}

#[derive(Debug, Clone)]
pub enum ResolutionStrategy {
    LinearResolve,
    ForkAndMerge,
    UserInput,
}

#[derive(Debug, Clone)]
pub struct AsyncProcessingMetadata {
    pub concurrent_processing: bool,
    pub harmony_validated: bool,
    pub tokio_context: bool,
}

// ASYNC PROCESSING CONFIGURATION

#[derive(Debug, Clone)]
pub struct AsyncProcessingConfig {
    pub max_concurrent_tokens: usize,
    pub contradiction_timeout: Duration,
    pub semantic_timeout: Duration,
    pub safety_timeout: Duration,
    pub enable_background_processing: bool,
}

impl Default for AsyncProcessingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tokens: 8,
            contradiction_timeout: Duration::from_millis(500),
            semantic_timeout: Duration::from_millis(300),
            safety_timeout: Duration::from_millis(100),
            enable_background_processing: true,
        }
    }
}

// ASYNC METRICS

#[derive(Debug, Clone)]
pub struct AsyncEngineMetrics {
    pub tokens_processed: u64,
    pub harmony_errors: u32,
    pub channel_transitions: u32,
    pub total_processing_time: Duration,
    pub concurrent_tasks_spawned: u64,
    pub average_processing_time: Duration,
}

impl AsyncEngineMetrics {
    pub fn new() -> Self {
        Self {
            tokens_processed: 0,
            harmony_errors: 0,
            channel_transitions: 0,
            total_processing_time: Duration::from_millis(0),
            concurrent_tasks_spawned: 0,
            average_processing_time: Duration::from_millis(0),
        }
    }
}

// ASYNC PROCESSOR STUBS (will be implemented as we convert each component)

pub struct AsyncContradictionProcessor {
    backend: Arc<UnifiedBackend>,
}

impl AsyncContradictionProcessor {
    pub async fn new(backend: Arc<UnifiedBackend>) -> NoesisResult<Self> {
        Ok(Self { backend })
    }
    
    pub async fn check_contradictions_harmony_async(
        &self,
        _token: u32,
        _harmony_context: &HarmonyChannelContext,
        _token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<AsyncContradictionResult> {
        // TODO: Implement async contradiction detection
        Ok(AsyncContradictionResult::default())
    }
    
    pub async fn reset_async(&self) -> NoesisResult<()> {
        Ok(())
    }
}

pub struct AsyncSemanticProcessor {
    backend: Arc<UnifiedBackend>,
}

impl AsyncSemanticProcessor {
    pub async fn new(backend: Arc<UnifiedBackend>) -> NoesisResult<Self> {
        Ok(Self { backend })
    }
    
    pub async fn process_semantics_harmony_async(
        &self,
        _token: u32,
        _harmony_context: &HarmonyChannelContext,
        _token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<AsyncSemanticResult> {
        // TODO: Implement async semantic processing
        Ok(AsyncSemanticResult::default())
    }
    
    pub async fn reset_async(&self) -> NoesisResult<()> {
        Ok(())
    }
}

pub struct AsyncStateManager {
    token_graph: Arc<TokenGraph>,
    backend: Arc<UnifiedBackend>,
}

impl AsyncStateManager {
    pub async fn new(token_graph: Arc<TokenGraph>, backend: Arc<UnifiedBackend>) -> NoesisResult<Self> {
        Ok(Self { token_graph, backend })
    }
    
    pub async fn update_state_harmony_async(
        &self,
        _token: u32,
        _harmony_context: &HarmonyChannelContext,
        _contradiction_result: &AsyncContradictionResult,
        _semantic_result: &AsyncSemanticResult,
    ) -> NoesisResult<()> {
        // TODO: Implement async state management
        Ok(())
    }
    
    pub async fn reset_async(&self) -> NoesisResult<()> {
        Ok(())
    }
}

pub struct AsyncSafetyFilter {
    backend: Arc<UnifiedBackend>,
}

impl AsyncSafetyFilter {
    pub async fn new(backend: Arc<UnifiedBackend>) -> NoesisResult<Self> {
        Ok(Self { backend })
    }
    
    pub async fn assess_safety_harmony_async(
        &self,
        _token: u32,
        _harmony_context: &HarmonyChannelContext,
    ) -> NoesisResult<AsyncSafetyResult> {
        // TODO: Implement async safety assessment
        Ok(AsyncSafetyResult::safe_fallback())
    }
    
    pub async fn reset_async(&self) -> NoesisResult<()> {
        Ok(())
    }
}

// Extension for HarmonyChannelContext to support error fallbacks
impl HarmonyChannelContext {
    pub fn error_fallback(token: u32) -> Self {
        Self {
            current_channel: Channel::Final,
            previous_channel: None,
            security_level: crate::cognitive::SecurityLevel::Public,
            user_visible: true,
            harmony_channel: None,
            content_delta: None,
            current_role: "error".to_string(),
            recipient: None,
            content_type: None,
            token_processed: token,
            parser_ready: false,
        }
    }
}
