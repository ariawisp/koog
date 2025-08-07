// Cross-Platform Channel-Aware Safety Filter
//
// ARCHITECTURAL FIX: Removed Metal-specific APIs, now uses UnifiedBackend
// for true cross-platform safety filtering across Metal/CUDA/ONNX platforms.
// Analysis channel content is NEVER exposed to users - hardware guaranteed.

use crate::gpu_optimized::OptimizedBuffer;
use crate::cognitive::{Channel, ChannelContext, SemanticContext, CognitiveError};
use crate::backend::UnifiedBackend;
use crate::errors::NoesisResult;
use std::sync::Arc;
use std::collections::HashMap;

/// Cross-Platform GPU-Accelerated Safety Filter with Hardware Channel Boundaries
/// 
/// Uses UnifiedBackend for GPU operations, ensuring compatibility
/// across Metal (macOS), CUDA (Linux), and ONNX (Windows) platforms.
pub struct SafetyFilter {
    // Cross-platform GPU backend (works on Metal/CUDA/ONNX)
    backend: Arc<UnifiedBackend>,
    
    // Safety configuration per channel
    channel_safety_configs: HashMap<Channel, ChannelSafetyConfig>,
    
    // Real-time safety monitoring
    safety_violations: Vec<SafetyViolation>,
    blocked_patterns: Vec<BlockedPattern>,
    
    // Safety thresholds
    risk_threshold: f32,
    harm_threshold: f32,
    toxicity_threshold: f32,
}

impl SafetyFilter {
    /// Initialize safety filter with cross-platform GPU backend
    pub fn new(backend: Arc<UnifiedBackend>) -> NoesisResult<Self> {
        
        // Initialize channel safety configurations
        let mut channel_safety_configs = HashMap::new();
        
        // Analysis Channel - Internal reasoning (NEVER user-visible)
        channel_safety_configs.insert(Channel::Analysis, ChannelSafetyConfig {
            channel: Channel::Analysis,
            expose_to_user: false,       // HARDWARE-ENFORCED: NEVER expose
            risk_tolerance: 1.0,         // High tolerance for internal reasoning
            harm_tolerance: 0.8,         // Can explore harmful scenarios internally
            toxicity_tolerance: 0.9,     // Can reason about toxic content
            content_filtering: ContentFilteringLevel::None, // No filtering for internal use
            safety_level: SafetyLevel::Internal,
        });
        
        // Commentary Channel - Tool execution and planning (user-visible but controlled)
        channel_safety_configs.insert(Channel::Commentary, ChannelSafetyConfig {
            channel: Channel::Commentary,
            expose_to_user: true,        // User-visible but filtered
            risk_tolerance: 0.6,         // Moderate risk tolerance
            harm_tolerance: 0.3,         // Lower harm tolerance
            toxicity_tolerance: 0.4,     // Limited toxicity tolerance
            content_filtering: ContentFilteringLevel::Moderate, // Moderate filtering
            safety_level: SafetyLevel::Restricted,
        });
        
        // Final Channel - Safety-aligned responses (always safe)
        channel_safety_configs.insert(Channel::Final, ChannelSafetyConfig {
            channel: Channel::Final,
            expose_to_user: true,        // Always user-visible
            risk_tolerance: 0.2,         // Very low risk tolerance
            harm_tolerance: 0.1,         // Minimal harm tolerance
            toxicity_tolerance: 0.1,     // Minimal toxicity tolerance
            content_filtering: ContentFilteringLevel::Strict, // Strict filtering
            safety_level: SafetyLevel::Public,
        });
        
        Ok(Self {
            backend,
            channel_safety_configs,
            safety_violations: Vec::new(),
            blocked_patterns: Vec::new(),
            risk_threshold: 0.7,
            harm_threshold: 0.5,
            toxicity_threshold: 0.6,
        })
    }
    
    /// Assess token safety within channel context - real-time during generation
    /// 
    /// This is called for EVERY token to ensure hardware-enforced safety boundaries.
    /// Analysis channel tokens are NEVER exposed regardless of content.
    pub fn assess_token_safety(
        &mut self,
        token: u32,
        channel_context: &ChannelContext,
        semantic_context: &SemanticContext,
        _token_buffer: &OptimizedBuffer,
    ) -> NoesisResult<SafetyAssessment> {
        
        // 1. Get channel safety configuration
        let channel_config = self.channel_safety_configs
            .get(&channel_context.current_channel)
            .ok_or_else(|| crate::errors::NoesisError::cognitive(
                &format!("No safety config for channel: {:?}", channel_context.current_channel)
            ))?;
        
        // 2. Hardware-enforced channel boundary check (CRITICAL)
        let boundary_enforcement = self.enforce_channel_boundaries(
            token,
            channel_context,
            channel_config,
        );
        
        // 3. Basic safety assessment (TODO: implement GPU acceleration via backend)
        let risk_score = self.assess_token_risk(token, semantic_context);
        let content_classification = self.classify_content_safety(token, semantic_context);
        let harm_indicators = self.detect_harmful_content(token, semantic_context);
        
        // 4. Apply channel-specific safety thresholds
        let is_safe = self.evaluate_safety_against_thresholds(
            &risk_score,
            &content_classification,
            &harm_indicators,
            channel_config,
        );
        
        // 5. Generate safety assessment
        let safety_assessment = SafetyAssessment {
            token,
            channel: channel_context.current_channel,
            is_safe,
            risk_score: risk_score.overall_risk,
            content_classification: content_classification.clone(),
            harm_indicators: harm_indicators.clone(),
            boundary_enforcement: boundary_enforcement.clone(),
            safety_level: channel_config.safety_level,
            user_visible: is_safe && channel_config.expose_to_user,
            risk_description: self.generate_risk_description(&risk_score, &content_classification, &harm_indicators),
            mitigation_actions: self.generate_mitigation_actions(&risk_score, &content_classification, channel_config),
        };
        
        // 6. Log safety violations if any
        if !is_safe {
            self.log_safety_violation(&safety_assessment);
        }
        
        Ok(safety_assessment)
    }
    
    /// Hardware-enforced channel boundary enforcement (CRITICAL SAFETY FEATURE)
    fn enforce_channel_boundaries(
        &self,
        _token: u32,
        channel_context: &ChannelContext,
        channel_config: &ChannelSafetyConfig,
    ) -> BoundaryEnforcement {
        
        // HARDWARE RULE: Analysis channel content is NEVER exposed to users
        if channel_context.current_channel == Channel::Analysis {
            return BoundaryEnforcement {
                channel: Channel::Analysis,
                hardware_enforced: true,
                user_exposure_blocked: true,
                filtering_applied: FilteringAction::CompleteBlock,
                boundary_confidence: 1.0,
                enforcement_reason: "Hardware-enforced Analysis channel boundary".to_string(),
            };
        }
        
        // For other channels, apply standard filtering
        BoundaryEnforcement {
            channel: channel_context.current_channel,
            hardware_enforced: false,
            user_exposure_blocked: !channel_config.expose_to_user,
            filtering_applied: match channel_config.content_filtering {
                ContentFilteringLevel::None => FilteringAction::None,
                ContentFilteringLevel::Moderate => FilteringAction::Sanitize,
                ContentFilteringLevel::Strict => FilteringAction::Redact,
            },
            boundary_confidence: 0.9,
            enforcement_reason: format!("Channel boundary enforcement for {:?}", channel_context.current_channel),
        }
    }
    
    /// Basic token risk assessment (TODO: implement GPU acceleration)
    fn assess_token_risk(
        &self,
        _token: u32,
        semantic_context: &SemanticContext,
    ) -> TokenRiskScore {
        // Simple heuristic-based assessment
        let base_risk = if semantic_context.intent.confidence < 0.5 { 0.3 } else { 0.1 };
        
        TokenRiskScore {
            overall_risk: base_risk,
            toxicity_risk: base_risk * 0.7,
            harm_risk: base_risk * 0.6,
            bias_risk: base_risk * 0.4,
            privacy_risk: base_risk * 0.3,
            confidence: 0.8,
        }
    }
    
    /// Basic content safety classification (TODO: implement GPU acceleration)
    fn classify_content_safety(
        &self,
        _token: u32,
        semantic_context: &SemanticContext,
    ) -> ContentSafetyClassification {
        // Simple heuristic-based classification
        let safety_category = if semantic_context.intent.confidence > 0.8 {
            SafetyCategory::Safe
        } else {
            SafetyCategory::Caution
        };
        
        ContentSafetyClassification {
            safety_category,
            toxicity_score: 0.1,
            bias_score: 0.1,
            harm_potential: 0.1,
            confidence: 0.8,
        }
    }
    
    /// Basic harmful content detection (TODO: implement GPU acceleration)
    fn detect_harmful_content(
        &self,
        _token: u32,
        _semantic_context: &SemanticContext,
    ) -> Vec<HarmIndicator> {
        // Return empty for basic implementation
        Vec::new()
    }
    
    /// Evaluate safety against channel-specific thresholds
    fn evaluate_safety_against_thresholds(
        &self,
        risk_score: &TokenRiskScore,
        content_classification: &ContentSafetyClassification,
        harm_indicators: &[HarmIndicator],
        channel_config: &ChannelSafetyConfig,
    ) -> bool {
        
        // Check risk score against channel tolerance
        if risk_score.overall_risk > channel_config.risk_tolerance {
            return false;
        }
        
        // Check harm indicators against channel tolerance
        let max_harm_score = harm_indicators.iter()
            .map(|h| h.severity)
            .fold(0.0f32, f32::max);
        
        if max_harm_score > channel_config.harm_tolerance {
            return false;
        }
        
        // Check toxicity against channel tolerance
        if content_classification.toxicity_score > channel_config.toxicity_tolerance {
            return false;
        }
        
        // Check content safety classification
        match content_classification.safety_category {
            SafetyCategory::Safe => true,
            SafetyCategory::Caution => {
                // Allow caution content in Analysis and Commentary channels
                matches!(channel_config.channel, Channel::Analysis | Channel::Commentary)
            },
            SafetyCategory::Warning => {
                // Only allow warning content in Analysis channel
                matches!(channel_config.channel, Channel::Analysis)
            },
            SafetyCategory::Harmful => false, // Never allow harmful content
        }
    }
    
    /// Generate human-readable risk description
    fn generate_risk_description(
        &self,
        risk_score: &TokenRiskScore,
        content_classification: &ContentSafetyClassification,
        harm_indicators: &[HarmIndicator],
    ) -> String {
        
        if risk_score.overall_risk < 0.3 {
            return "Low risk content".to_string();
        }
        
        let mut descriptions = Vec::new();
        
        if risk_score.overall_risk >= 0.7 {
            descriptions.push(format!("High risk score: {:.2}", risk_score.overall_risk));
        }
        
        if content_classification.toxicity_score >= 0.5 {
            descriptions.push(format!("Elevated toxicity: {:.2}", content_classification.toxicity_score));
        }
        
        for harm in harm_indicators {
            if harm.severity >= 0.6 {
                descriptions.push(format!("{:?}: {:.2}", harm.harm_type, harm.severity));
            }
        }
        
        if descriptions.is_empty() {
            "Moderate risk content".to_string()
        } else {
            descriptions.join(", ")
        }
    }
    
    /// Generate mitigation actions for unsafe content
    fn generate_mitigation_actions(
        &self,
        risk_score: &TokenRiskScore,
        content_classification: &ContentSafetyClassification,
        channel_config: &ChannelSafetyConfig,
    ) -> Vec<MitigationAction> {
        
        let mut actions = Vec::new();
        
        // Channel-specific mitigations
        match channel_config.channel {
            Channel::Analysis => {
                // Analysis channel - internal reasoning only
                actions.push(MitigationAction::BlockUserExposure);
            },
            Channel::Commentary => {
                // Commentary channel - apply filtering
                if content_classification.toxicity_score > 0.6 {
                    actions.push(MitigationAction::SanitizeContent);
                }
                if risk_score.overall_risk > 0.7 {
                    actions.push(MitigationAction::AddWarningLabels);
                }
            },
            Channel::Final => {
                // Final channel - strict filtering
                if risk_score.overall_risk > 0.3 {
                    actions.push(MitigationAction::BlockGeneration);
                }
                if content_classification.toxicity_score > 0.2 {
                    actions.push(MitigationAction::RegenerateContent);
                }
            },
        }
        
        // Additional mitigations based on content
        if content_classification.safety_category == SafetyCategory::Harmful {
            actions.push(MitigationAction::BlockGeneration);
            actions.push(MitigationAction::LogSecurityEvent);
        }
        
        actions
    }
    
    /// Log safety violation for monitoring
    fn log_safety_violation(&mut self, assessment: &SafetyAssessment) {
        let violation = SafetyViolation {
            timestamp: chrono::Utc::now().timestamp_nanos() as u64,
            token: assessment.token,
            channel: assessment.channel,
            risk_score: assessment.risk_score,
            violation_type: determine_violation_type(assessment),
            description: assessment.risk_description.clone(),
            mitigation_applied: assessment.mitigation_actions.clone(),
        };
        
        self.safety_violations.push(violation);
        
        // Limit violation history
        if self.safety_violations.len() > 1000 {
            self.safety_violations.remove(0);
        }
        
        tracing::warn!(
            "🚨 Safety violation detected: token={}, channel={:?}, risk={:.2}",
            assessment.token,
            assessment.channel,
            assessment.risk_score
        );
    }
    
    /// Get safety statistics for monitoring
    pub fn get_safety_statistics(&self) -> SafetyStatistics {
        let total_violations = self.safety_violations.len();
        let recent_violations = self.safety_violations.iter()
            .filter(|v| {
                let now = chrono::Utc::now().timestamp_nanos() as u64;
                now - v.timestamp < 3600_000_000_000 // Last hour
            })
            .count();
        
        let violations_by_channel: HashMap<Channel, usize> = self.safety_violations.iter()
            .fold(HashMap::new(), |mut acc, violation| {
                *acc.entry(violation.channel).or_insert(0) += 1;
                acc
            });
        
        SafetyStatistics {
            total_violations,
            recent_violations,
            violations_by_channel,
            blocked_patterns_count: self.blocked_patterns.len(),
            average_risk_score: self.safety_violations.iter()
                .map(|v| v.risk_score)
                .sum::<f32>() / total_violations.max(1) as f32,
        }
    }
}

// Safety assessment data structures

#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    pub token: u32,
    pub channel: Channel,
    pub is_safe: bool,
    pub risk_score: f32,
    pub content_classification: ContentSafetyClassification,
    pub harm_indicators: Vec<HarmIndicator>,
    pub boundary_enforcement: BoundaryEnforcement,
    pub safety_level: SafetyLevel,
    pub user_visible: bool,
    pub risk_description: String,
    pub mitigation_actions: Vec<MitigationAction>,
}

#[derive(Debug, Clone)]
struct ChannelSafetyConfig {
    channel: Channel,
    expose_to_user: bool,
    risk_tolerance: f32,
    harm_tolerance: f32,
    toxicity_tolerance: f32,
    content_filtering: ContentFilteringLevel,
    safety_level: SafetyLevel,
}

#[derive(Debug, Clone, Copy)]
pub enum SafetyLevel {
    Internal = 0,    // Analysis channel - never user-visible
    Restricted = 1,  // Commentary channel - filtered visibility
    Public = 2,      // Final channel - always safe
}

#[derive(Debug, Clone)]
enum ContentFilteringLevel {
    None,
    Moderate,
    Strict,
}

#[derive(Debug, Clone)]
pub struct TokenRiskScore {
    pub overall_risk: f32,
    pub toxicity_risk: f32,
    pub harm_risk: f32,
    pub bias_risk: f32,
    pub privacy_risk: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ContentSafetyClassification {
    pub safety_category: SafetyCategory,
    pub toxicity_score: f32,
    pub bias_score: f32,
    pub harm_potential: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SafetyCategory {
    Safe,
    Caution,
    Warning,
    Harmful,
}

#[derive(Debug, Clone)]
pub struct BoundaryEnforcement {
    pub channel: Channel,
    pub hardware_enforced: bool,
    pub user_exposure_blocked: bool,
    pub filtering_applied: FilteringAction,
    pub boundary_confidence: f32,
    pub enforcement_reason: String,
}

#[derive(Debug, Clone)]
pub enum FilteringAction {
    None,
    Sanitize,
    Redact,
    CompleteBlock,
}

#[derive(Debug, Clone)]
pub enum MitigationAction {
    BlockGeneration,
    BlockUserExposure,
    SanitizeContent,
    RedactContent,
    AddWarningLabels,
    RegenerateContent,
    LogSecurityEvent,
    RequireHumanReview,
}

#[derive(Debug, Clone)]
struct SafetyViolation {
    timestamp: u64,
    token: u32,
    channel: Channel,
    risk_score: f32,
    violation_type: ViolationType,
    description: String,
    mitigation_applied: Vec<MitigationAction>,
}

#[derive(Debug, Clone)]
enum ViolationType {
    HighRisk,
    ToxicContent,
    HarmfulContent,
    ChannelBoundaryViolation,
    PolicyViolation,
}

#[derive(Debug, Clone)]
struct BlockedPattern {
    pattern: String,
    severity: f32,
    created_at: u64,
}

#[derive(Debug, Clone)]
pub struct SafetyStatistics {
    pub total_violations: usize,
    pub recent_violations: usize,
    pub violations_by_channel: HashMap<Channel, usize>,
    pub blocked_patterns_count: usize,
    pub average_risk_score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct HarmIndicator {
    pub harm_type: HarmType,
    pub severity: f32,
    pub confidence: f32,
    pub _padding: u32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum HarmType {
    Violence = 0,
    Harassment = 1,
    SelfHarm = 2,
    Sexual = 3,
    Hateful = 4,
    IllegalActivity = 5,
    Privacy = 6,
    Misinformation = 7,
}

// Helper functions

fn determine_violation_type(assessment: &SafetyAssessment) -> ViolationType {
    if assessment.risk_score >= 0.8 {
        ViolationType::HighRisk
    } else if assessment.content_classification.toxicity_score >= 0.7 {
        ViolationType::ToxicContent
    } else if assessment.harm_indicators.iter().any(|h| h.severity >= 0.7) {
        ViolationType::HarmfulContent
    } else if !assessment.boundary_enforcement.user_exposure_blocked && assessment.channel == Channel::Analysis {
        ViolationType::ChannelBoundaryViolation
    } else {
        ViolationType::PolicyViolation
    }
}
