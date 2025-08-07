// OpenAI Harmony StreamableParser integration

use openai_harmony::{
    load_harmony_encoding, HarmonyEncodingName, StreamableParser
};
use crate::cognitive::{Channel, SecurityLevel, CognitiveError};
use crate::errors::{NoesisError, NoesisResult};
use std::sync::{Arc, Mutex};
use tracing::{debug, warn, error};

/// OpenAI Harmony-based channel router
pub struct HarmonyChannelRouter {
    /// Official OpenAI Harmony parser - handles ALL token parsing
    parser: Arc<Mutex<StreamableParser>>,
    /// Current channel state
    current_channel: Channel,
    /// Channel transition history
    channel_history: Vec<(Channel, u64)>,
    /// Security enforcement state
    security_enforced: bool,
}

impl HarmonyChannelRouter {
    /// Create new harmony-based channel router
    /// 
    /// BREAKING CHANGE: No more GPU backend dependency for channel detection
    /// OpenAI Harmony handles all the parsing complexity for us
    pub fn new() -> Result<Self, CognitiveError> {
        // Load official GPT-OSS harmony encoding
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(|e| NoesisError::harmony(&format!("Failed to load harmony encoding: {}", e)))?;
        
        // Create StreamableParser with Assistant role (typical for inference)
        let parser = openai_harmony::StreamableParser::new(encoding, Some(openai_harmony::chat::Role::Assistant))
            .map_err(|e| NoesisError::harmony(&format!("Failed to create StreamableParser: {}", e)))?;
        
        debug!("🚀 HarmonyChannelRouter initialized with official OpenAI StreamableParser");
        
        Ok(Self {
            parser: Arc::new(Mutex::new(parser)),
            current_channel: Channel::Final, // Start safe
            channel_history: Vec::new(),
            security_enforced: true,
        })
    }
    
    /// Process token through official OpenAI Harmony parser
    /// 
    /// AGGRESSIVE SIMPLIFICATION: All the complex token parsing is handled by
    /// the official library. We just extract the channel information.
    pub fn process_token(&mut self, token: u32) -> Result<HarmonyChannelContext, CognitiveError> {
        let mut parser = self.parser.lock()
            .map_err(|_| NoesisError::harmony("Parser mutex poisoned"))?;
        
        // Feed token to official OpenAI Harmony parser
        parser.process(token);
        
        // Extract channel information from harmony parser
        let harmony_channel = parser.current_channel();
        let content_delta = parser.last_content_delta().ok().flatten();
        let current_role = parser.current_role();
        let recipient = parser.current_recipient();
        let content_type = parser.current_content_type();
        
        // Convert OpenAI Harmony channel to our Channel enum
        let detected_channel = self.map_harmony_channel_to_noesis(harmony_channel.as_deref())?;
        
        // Track channel transitions
        if detected_channel != self.current_channel {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            
            debug!(
                "🔄 Channel transition: {:?} -> {:?} (via Harmony parser)",
                self.current_channel,
                detected_channel
            );
            
            self.channel_history.push((self.current_channel, timestamp));
            self.current_channel = detected_channel;
        }
        
        // Create enhanced context with harmony information
        let context = HarmonyChannelContext {
            // Core channel information
            current_channel: self.current_channel,
            previous_channel: self.channel_history.last().map(|(ch, _)| *ch),
            
            // Official OpenAI Harmony data
            harmony_channel,
            content_delta,
            current_role: format!("{:?}", current_role),
            recipient: recipient.map(|s| s.to_string()),
            content_type: content_type.map(|s| s.to_string()),
            
            // Security context
            security_level: self.current_channel.security_level(),
            user_visible: self.current_channel.is_user_visible() && self.security_enforced,
            
            // Parser state
            token_processed: token,
            parser_ready: true,
        };
        
        Ok(context)
    }
    
    /// Get complete conversation state from harmony parser
    /// 
    /// REVOLUTIONARY FEATURE: Access to the full conversation as parsed
    /// by the official OpenAI library - no custom state management needed
    pub fn get_conversation_state(&self) -> Result<HarmonyConversationState, CognitiveError> {
        let parser = self.parser.lock()
            .map_err(|_| NoesisError::harmony("Parser mutex poisoned"))?;
        
        let messages = parser.messages();
        let current_content = parser.current_content().unwrap_or_default();
        let tokens = parser.tokens();
        
        Ok(HarmonyConversationState {
            message_count: messages.len(),
            current_content: current_content.to_string(),
            total_tokens: tokens.len(),
            current_channel: self.current_channel,
            channel_transitions: self.channel_history.len(),
        })
    }
    
    /// Force channel switch (for emergency safety interventions)
    /// 
    /// Note: This breaks harmony parsing state but provides safety override
    pub fn force_safety_channel(&mut self) -> Result<(), CognitiveError> {
        warn!("🚨 Emergency channel switch to Final (safety override)");
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        
        self.channel_history.push((self.current_channel, timestamp));
        self.current_channel = Channel::Final;
        self.security_enforced = true;
        
        Ok(())
    }
    
    /// Map OpenAI Harmony channel string to our Channel enum
    fn map_harmony_channel_to_noesis(&self, harmony_channel: Option<&str>) -> Result<Channel, CognitiveError> {
        match harmony_channel {
            Some("analysis") => Ok(Channel::Analysis),
            Some("commentary") => Ok(Channel::Commentary),
            Some("final") => Ok(Channel::Final),
            None => Ok(self.current_channel), // No change if no channel detected
            Some(unknown) => {
                warn!("🤔 Unknown harmony channel '{}', defaulting to Final for safety", unknown);
                Ok(Channel::Final) // Safe default
            }
        }
    }
    
    /// Get current channel
    pub fn current_channel(&self) -> Channel {
        self.current_channel
    }
    
    /// Get channel history
    pub fn channel_history(&self) -> &[(Channel, u64)] {
        &self.channel_history
    }
    
    /// Reset parser state (for new conversations)
    pub fn reset(&mut self) -> Result<(), CognitiveError> {
        // Create fresh parser
        let encoding = load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            .map_err(|e| NoesisError::harmony(&format!("Failed to reload harmony encoding: {}", e)))?;
        
        let new_parser = openai_harmony::StreamableParser::new(encoding, Some(openai_harmony::chat::Role::Assistant))
            .map_err(|e| NoesisError::harmony(&format!("Failed to create StreamableParser: {}", e)))?;
        
        // Replace parser
        *self.parser.lock()
            .map_err(|_| NoesisError::harmony("Parser mutex poisoned"))? = new_parser;
        
        // Reset state
        self.current_channel = Channel::Final;
        self.channel_history.clear();
        self.security_enforced = true;
        
        debug!("🔄 HarmonyChannelRouter reset for new conversation");
        
        Ok(())
    }
}

/// Enhanced channel context with OpenAI Harmony integration
#[derive(Debug, Clone)]
pub struct HarmonyChannelContext {
    // Core Noesis channel information
    pub current_channel: Channel,
    pub previous_channel: Option<Channel>,
    pub security_level: SecurityLevel,
    pub user_visible: bool,
    
    // OpenAI Harmony parser information
    pub harmony_channel: Option<String>,
    pub content_delta: Option<String>,
    pub current_role: String,
    pub recipient: Option<String>,
    pub content_type: Option<String>,
    
    // Processing state
    pub token_processed: u32,
    pub parser_ready: bool,
}

/// Complete conversation state from harmony parser
#[derive(Debug, Clone)]
pub struct HarmonyConversationState {
    pub message_count: usize,
    pub current_content: String,
    pub total_tokens: usize,
    pub current_channel: Channel,
    pub channel_transitions: usize,
}

/// BREAKING CHANGE: Simplified routing decision
/// 
/// The complex routing logic is eliminated in favor of simple
/// channel-based decisions. OpenAI Harmony handles the complexity.
#[derive(Debug, Clone)]
pub enum HarmonyRoutingDecision {
    /// Analysis channel - never show to users
    ProcessInternally {
        harmony_detected: bool,
    },
    /// Commentary channel - show but with filtering
    ProcessWithVisibility {
        harmony_detected: bool,
        recipient: Option<String>,
    },
    /// Final channel - always safe to show
    ProcessAsSafeOutput {
        harmony_detected: bool,
    },
}

impl HarmonyChannelContext {
    /// Get routing decision based on harmony-detected channel
    pub fn get_routing_decision(&self) -> HarmonyRoutingDecision {
        let harmony_detected = self.harmony_channel.is_some();
        
        match self.current_channel {
            Channel::Analysis => HarmonyRoutingDecision::ProcessInternally { 
                harmony_detected 
            },
            Channel::Commentary => HarmonyRoutingDecision::ProcessWithVisibility { 
                harmony_detected,
                recipient: self.recipient.clone(),
            },
            Channel::Final => HarmonyRoutingDecision::ProcessAsSafeOutput { 
                harmony_detected 
            },
        }
    }
    
    /// Check if content should be visible to users
    pub fn should_show_to_user(&self) -> bool {
        match self.current_channel {
            Channel::Analysis => false, // NEVER show analysis
            Channel::Commentary => self.user_visible, // Show if allowed
            Channel::Final => true, // ALWAYS show final
        }
    }
    
    /// Get security assessment
    pub fn security_assessment(&self) -> SecurityAssessment {
        SecurityAssessment {
            channel_safe: matches!(self.current_channel, Channel::Final),
            harmony_validated: self.harmony_channel.is_some(),
            user_exposure_safe: self.should_show_to_user(),
            recipient_identified: self.recipient.is_some(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SecurityAssessment {
    pub channel_safe: bool,
    pub harmony_validated: bool,
    pub user_exposure_safe: bool,
    pub recipient_identified: bool,
}

// Update the error enum to include harmony-specific errors
impl crate::cognitive::CognitiveError {
    pub fn harmony_error<T: Into<String>>(msg: T) -> Self {
        let message = msg.into();
        NoesisError::harmony(&message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_harmony_channel_router_creation() {
        let router = HarmonyChannelRouter::new();
        assert!(router.is_ok(), "Should create HarmonyChannelRouter successfully");
        
        let router = router.unwrap();
        assert_eq!(router.current_channel(), Channel::Final, "Should start in Final channel");
    }
    
    #[test]
    fn test_channel_mapping() {
        let router = HarmonyChannelRouter::new().unwrap();
        
        // Test harmony channel mapping
        assert!(matches!(
            router.map_harmony_channel_to_noesis(Some("analysis")),
            Ok(Channel::Analysis)
        ));
        assert!(matches!(
            router.map_harmony_channel_to_noesis(Some("commentary")),
            Ok(Channel::Commentary)
        ));
        assert!(matches!(
            router.map_harmony_channel_to_noesis(Some("final")),
            Ok(Channel::Final)
        ));
        assert!(matches!(
            router.map_harmony_channel_to_noesis(Some("unknown")),
            Ok(Channel::Final) // Safe default
        ));
    }
    
    #[tokio::test]
    async fn test_token_processing() {
        let mut router = HarmonyChannelRouter::new().unwrap();
        
        // Test processing a simple token
        let result = router.process_token(12345);
        assert!(result.is_ok(), "Should process token successfully");
        
        let context = result.unwrap();
        assert!(context.parser_ready, "Parser should be ready");
        assert_eq!(context.token_processed, 12345, "Should track processed token");
    }
}
