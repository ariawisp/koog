// Token-native cognitive processing
//
// Features:
// - Real-time contradiction detection during generation
// - GPU-accelerated semantic processing 
// - Hardware-enforced channel boundaries
// - Fork/merge reasoning with zero-copy operations

pub mod channels;
pub mod harmony_channels;  // NEW: OpenAI Harmony integration
pub mod harmony_engine;    // NEW: Complete harmony-native engine
pub mod async_engine;      // NEW: Fully async cognitive layer
pub mod contradiction_detector;
pub mod semantic_processor; 
pub mod state_manager;
pub mod safety_filter;

pub use channels::{Channel, ChannelRouter, ChannelContext, RoutingDecision, SecurityContext, SecurityBoundary, SecurityLevel, ChannelSemanticHistory};
pub use harmony_channels::{HarmonyChannelRouter, HarmonyChannelContext, HarmonyRoutingDecision}; // NEW exports
pub use harmony_engine::{HarmonyCognitiveEngine, HarmonyTokenResult, HarmonyAction}; // NEW: Complete engine
pub use async_engine::{AsyncCognitiveEngine, AsyncTokenResult, AsyncCognitiveAction}; // NEW: Fully async
pub use contradiction_detector::{ContradictionDetector, ContradictionAlert, PathContradictionAnalysis, PathResolutionStrategy};
pub use semantic_processor::{SemanticProcessor, SemanticContext, SemanticShiftType};
pub use state_manager::{CognitiveStateManager, CognitiveCheckpoint};
pub use safety_filter::{SafetyFilter, SafetyAssessment};
pub use state_manager::MergeResult;

use crate::gpu_optimized::OptimizedBuffer;
use crate::memory::TokenGraph;
use crate::backend::UnifiedBackend;
use crate::errors::{NoesisError, NoesisResult, CognitiveComponent, ErrorSeverity};
use std::sync::Arc;

/// Core Cognitive Engine - Token-Native Processing
/// 
/// This is the HEART of Noesis Runtime's revolutionary capabilities.
/// All cognitive operations work directly on u32 token arrays with GPU acceleration.
pub struct CognitiveEngine {
    // Zero-cost compile-time dispatch backend
    pub(crate) backend: Arc<UnifiedBackend>,
    
    // Cognitive processing modules
    pub(crate) channel_router: ChannelRouter,
    pub(crate) contradiction_detector: ContradictionDetector,
    pub(crate) semantic_processor: SemanticProcessor,
    pub(crate) state_manager: CognitiveStateManager,
    pub(crate) safety_filter: SafetyFilter,
    
    // Token graph - unified memory and cognition
    pub(crate) token_graph: Arc<TokenGraph>,
}

impl CognitiveEngine {
    /// Initialize cognitive engine with zero-cost GPU backend
    pub fn new(
        backend: Arc<UnifiedBackend>,
        token_graph: Arc<TokenGraph>,
    ) -> NoesisResult<Self> {
        let channel_router = ChannelRouter::new(backend.clone())?;
        let contradiction_detector = ContradictionDetector::new(backend.clone())?;
        let semantic_processor = SemanticProcessor::new(backend.clone(), backend.get_buffer_pool())?;
        let state_manager = CognitiveStateManager::new(backend.clone(), backend.get_buffer_pool(), token_graph.clone())?;
        let safety_filter = SafetyFilter::new(backend.clone())?;
        
        Ok(Self {
            backend,
            channel_router,
            contradiction_detector,
            semantic_processor,
            state_manager,
            safety_filter,
            token_graph,
        })
    }
    
    /// Process token during generation with full cognitive pipeline
    /// 
    /// This is called for EVERY token as it's being generated.
    /// Returns cognitive context and any alerts/interventions needed.
    pub fn process_token_realtime(
        &mut self,
        token: u32,
        token_buffer: &OptimizedBuffer,
        generation_context: &GenerationContext,
    ) -> NoesisResult<TokenCognitiveResult> {
        
        // 1. Channel routing - determine which cognitive channel this token belongs to
        let channel_context = self.channel_router.route_token(
            token, 
            &generation_context.active_channels,
            token_buffer
        )?;
        
        // 2. Real-time contradiction detection across active reasoning paths
        let contradiction_alerts = self.contradiction_detector.check_token_contradictions(
            token,
            &channel_context,
            &generation_context.reasoning_paths,
            token_buffer
        )?;
        
        // 3. GPU-accelerated semantic processing
        // Create a temporary SemanticHistory that matches what the semantic processor expects
        let semantic_history = crate::cognitive::semantic_processor::SemanticHistory {
            recent_tokens: generation_context.semantic_history.recent_tokens.clone(),
            recent_embeddings: vec![], // TODO: maintain embeddings in generation context
            channel_transitions: generation_context.semantic_history.channel_transitions.clone(),
            semantic_shifts: vec![], // TODO: track semantic shifts
        };
        
        let semantic_context = self.semantic_processor.process_token_semantics(
            token,
            token_buffer,
            &channel_context,
            &semantic_history
        )?;
        
        // 4. Safety filtering with hardware boundaries
        let safety_assessment = self.safety_filter.assess_token_safety(
            token,
            &channel_context,
            &semantic_context,
            token_buffer
        )?;
        
        // 5. Determine cognitive action before moving values
        let cognitive_action = determine_cognitive_action(&contradiction_alerts, &safety_assessment);
        
        // 6. Update cognitive state in token graph
        self.state_manager.update_cognitive_state(
            token,
            &channel_context,
            &semantic_context,
            &contradiction_alerts
        )?;
        
        Ok(TokenCognitiveResult {
            token,
            channel_context,
            semantic_context,
            contradiction_alerts,
            safety_assessment,
            cognitive_action,
        })
    }
    
    /// Fork reasoning at decision point - zero-copy GPU operation
    pub fn fork_reasoning(&mut self, checkpoint_id: &str) -> NoesisResult<CognitiveCheckpoint> {
        self.state_manager.fork_reasoning(checkpoint_id)
    }
    
    /// Merge reasoning paths with contradiction resolution
    pub fn merge_reasoning_paths(
        &mut self,
        path_ids: &[String],
    ) -> NoesisResult<MergeResult> {
        // TODO: Get contradiction analysis across paths
        let contradictions = crate::cognitive::PathContradictionAnalysis::default();
        
        // Resolve contradictions and merge
        self.state_manager.merge_with_contradiction_resolution(path_ids, &contradictions)
    }
    
    /// Stream cognitive insights during generation
    pub fn stream_cognitive_insights(&self) -> impl Iterator<Item = CognitiveInsight> + '_ {
        self.state_manager.stream_insights()
    }
}

/// Token-level cognitive processing result
#[derive(Debug, Clone)]
pub struct TokenCognitiveResult {
    pub token: u32,
    pub channel_context: ChannelContext,
    pub semantic_context: SemanticContext,
    pub contradiction_alerts: Vec<ContradictionAlert>,
    pub safety_assessment: SafetyAssessment,
    pub cognitive_action: CognitiveAction,
}

/// Cognitive action to take based on processing
#[derive(Debug, Clone)]
pub enum CognitiveAction {
    /// Continue normal generation
    Continue,
    /// Fork reasoning due to uncertainty
    ForkReasoning { checkpoint_id: String },
    /// Intervene due to contradiction
    InterveneDueToContradiction { contradiction_id: String },
    /// Block token due to safety
    BlockTokenUnsafe { reason: String },
    /// Switch cognitive channel
    SwitchChannel { new_channel: Channel },
}

/// Context for token generation
#[derive(Debug, Clone)]
pub struct GenerationContext {
    pub active_channels: Vec<Channel>,
    pub reasoning_paths: Vec<String>,
    pub semantic_history: ChannelSemanticHistory,
    pub safety_context: SecurityContext,
}

/// Cognitive insights for real-time monitoring
#[derive(Debug, Clone)]
pub struct CognitiveInsight {
    pub timestamp: u64,
    pub insight_type: InsightType,
    pub description: String,
    pub confidence: f32,
    pub related_tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
pub enum InsightType {
    ReasoningDepthChange,
    ContradictionDetected,
    ChannelSwitch,
    SemanticShift,
    SafetyConcern,
    CognitiveLoad,
}

// BREAKING CHANGE: CognitiveError has been unified into NoesisError
// All cognitive processing now uses the unified error hierarchy
// Use crate::errors::{NoesisError, CognitiveComponent} for error construction
// 
// Migration examples:
// - CognitiveError::ChannelError(msg) → NoesisError::channel_routing(msg)
// - CognitiveError::HarmonyError(msg) → NoesisError::harmony(msg)
// - CognitiveError::GpuError(msg) → NoesisError::gpu_operation(msg, BackendType::Metal)
//
// Legacy compatibility type alias (TEMPORARY - will be removed)
pub type CognitiveError = NoesisError;

// Helper function to determine cognitive action
fn determine_cognitive_action(
    contradiction_alerts: &[ContradictionAlert],
    safety_assessment: &SafetyAssessment,
) -> CognitiveAction {
    // Safety takes highest priority
    if !safety_assessment.is_safe {
        return CognitiveAction::BlockTokenUnsafe { 
            reason: safety_assessment.risk_description.clone() 
        };
    }
    
    // Check for critical contradictions
    for alert in contradiction_alerts {
        if alert.severity >= 0.8 {
            return CognitiveAction::InterveneDueToContradiction {
                contradiction_id: alert.contradiction_id.clone(),
            };
        }
    }
    
    // Check if we should fork reasoning
    let high_uncertainty = contradiction_alerts.len() > 2;
    if high_uncertainty {
        let checkpoint_id = format!("uncertainty_{}", chrono::Utc::now().timestamp_nanos());
        return CognitiveAction::ForkReasoning { checkpoint_id };
    }
    
    CognitiveAction::Continue
}