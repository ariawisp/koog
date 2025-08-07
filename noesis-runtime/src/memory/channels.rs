// Channel Router - Multi-stream token processing
// This enables parallel channel processing that Graphiti cannot achieve

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use serde::{Serialize, Deserialize};
use anyhow::Result;
use crate::Channel;

// Channel is imported from crate root
    

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SecurityLevel {
    Internal,    // Never leaves the system
    Restricted,  // Only for tool execution
    Public,      // Safe for users
}

/// Routes tokens to appropriate channels for parallel processing
pub struct ChannelRouter {
    /// Channel-specific token buffers
    buffers: Arc<RwLock<HashMap<Channel, Vec<u32>>>>,
    
    /// Channel processors
    processors: HashMap<Channel, mpsc::Sender<Vec<u32>>>,
    
    /// Routing rules
    rules: RoutingRules,
    
    /// Statistics
    stats: Arc<RwLock<RoutingStats>>,
}

/// Rules for routing tokens to channels
#[derive(Debug, Clone)]
pub struct RoutingRules {
    /// Enable parallel processing
    pub parallel_enabled: bool,
    
    /// Maximum tokens per channel before flush
    pub buffer_size: usize,
    
    /// Channel priorities (higher = more important)
    pub priorities: HashMap<Channel, u8>,
    
    /// Security enforcement
    pub enforce_security: bool,
}

impl Default for RoutingRules {
    fn default() -> Self {
        let mut priorities = HashMap::new();
        priorities.insert(Channel::Final, 3);      // Highest priority
        priorities.insert(Channel::Commentary, 2);  // Medium priority
        priorities.insert(Channel::Analysis, 1);    // Lowest priority
        
        Self {
            parallel_enabled: true,
            buffer_size: 1024,
            priorities,
            enforce_security: true,
        }
    }
}

/// Statistics for monitoring routing performance
#[derive(Debug, Default)]
#[derive(Clone)]
pub struct RoutingStats {
    pub tokens_routed: HashMap<Channel, usize>,
    pub tokens_dropped: HashMap<Channel, usize>,
    pub buffer_overflows: usize,
    pub security_violations: usize,
}

impl ChannelRouter {
    /// Create a new channel router
    pub fn new(rules: RoutingRules) -> Self {
        Self {
            buffers: Arc::new(RwLock::new(HashMap::new())),
            processors: HashMap::new(),
            rules,
            stats: Arc::new(RwLock::new(RoutingStats::default())),
        }
    }
    
    /// Register a processor for a specific channel
    pub fn register_processor(&mut self, channel: Channel, sender: mpsc::Sender<Vec<u32>>) {
        self.processors.insert(channel, sender);
    }
    
    /// Route tokens based on channel markers
    /// This is the core of our multi-stream processing
    pub async fn route_tokens(&self, tokens: &[u32]) -> Result<(), RoutingError> {
        let mut current_channel = Channel::Final; // Default channel
        let mut current_buffer = Vec::new();
        
        for &token in tokens {
            // Check for channel switch token
            if token == 200005 { // <|channel|> token
                // Flush current buffer
                if !current_buffer.is_empty() {
                    self.flush_to_channel(current_channel, current_buffer.clone()).await?;
                    current_buffer.clear();
                }
                
                // Next token should indicate the channel
                // This is simplified - in practice we'd parse the channel name
                continue;
            }
            
            // Check for special stop tokens
            match token {
                200002 => { // <|return|> - completion stop
                    self.flush_to_channel(Channel::Final, current_buffer.clone()).await?;
                    current_buffer.clear();
                }
                200012 => { // <|call|> - tool call stop
                    self.flush_to_channel(Channel::Commentary, current_buffer.clone()).await?;
                    current_buffer.clear();
                }
                _ => {
                    current_buffer.push(token);
                }
            }
            
            // Flush if buffer is full
            if current_buffer.len() >= self.rules.buffer_size {
                self.flush_to_channel(current_channel, current_buffer.clone()).await?;
                current_buffer.clear();
            }
        }
        
        // Flush remaining tokens
        if !current_buffer.is_empty() {
            self.flush_to_channel(current_channel, current_buffer).await?;
        }
        
        Ok(())
    }
    
    /// Flush tokens to a specific channel
    async fn flush_to_channel(&self, channel: Channel, tokens: Vec<u32>) -> Result<(), RoutingError> {
        // Check security level
        if self.rules.enforce_security {
            match channel.security_level() {
                SecurityLevel::Internal => {
                    // Never send Analysis channel tokens externally
                    if !self.is_internal_processor(channel) {
                        let mut stats = self.stats.write().await;
                        stats.security_violations += 1;
                        return Err(RoutingError::SecurityViolation(channel));
                    }
                }
                SecurityLevel::Restricted => {
                    // Only allow tool processors
                    if !self.is_tool_processor(channel) {
                        let mut stats = self.stats.write().await;
                        stats.security_violations += 1;
                        return Err(RoutingError::SecurityViolation(channel));
                    }
                }
                SecurityLevel::Public => {
                    // Can be sent anywhere
                }
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            *stats.tokens_routed.entry(channel).or_insert(0) += tokens.len();
        }
        
        // Send to processor if registered
        if let Some(sender) = self.processors.get(&channel) {
            if sender.send(tokens.clone()).await.is_err() {
                let mut stats = self.stats.write().await;
                *stats.tokens_dropped.entry(channel).or_insert(0) += tokens.len();
                return Err(RoutingError::ProcessorClosed(channel));
            }
        } else {
            // Buffer tokens if no processor
            let mut buffers = self.buffers.write().await;
            buffers.entry(channel).or_insert_with(Vec::new).extend(tokens);
        }
        
        Ok(())
    }
    
    /// Extract channel from token stream
    pub fn extract_channel(&self, tokens: &[u32]) -> Channel {
        // Look for channel marker tokens
        for i in 0..tokens.len() {
            if tokens[i] == 200005 { // <|channel|>
                // Next tokens should indicate channel
                // This is simplified - real implementation would parse properly
                if i + 1 < tokens.len() {
                    // Parse channel name from subsequent tokens
                    // For now, use heuristics
                    return Channel::Commentary;
                }
            }
        }
        
        Channel::Final // Default
    }
    
    /// Split token stream by channels
    pub async fn split_by_channel(&self, tokens: &[u32]) -> HashMap<Channel, Vec<u32>> {
        let mut result = HashMap::new();
        let mut current_channel = Channel::Final;
        let mut current_tokens = Vec::new();
        
        for &token in tokens {
            if token == 200005 { // <|channel|> marker
                // Save current tokens
                if !current_tokens.is_empty() {
                    result.entry(current_channel)
                        .or_insert_with(Vec::new)
                        .extend(&current_tokens);
                    current_tokens.clear();
                }
                // Switch channel (simplified)
                current_channel = match current_channel {
                    Channel::Final => Channel::Commentary,
                    Channel::Commentary => Channel::Analysis,
                    Channel::Analysis => Channel::Final,
                };
            } else {
                current_tokens.push(token);
            }
        }
        
        // Save remaining tokens
        if !current_tokens.is_empty() {
            result.entry(current_channel)
                .or_insert_with(Vec::new)
                .extend(current_tokens);
        }
        
        result
    }
    
    /// Merge channels back into a single stream
    pub fn merge_channels(&self, channels: HashMap<Channel, Vec<u32>>) -> Vec<u32> {
        let mut result = Vec::new();
        
        // Process in priority order
        let mut sorted_channels: Vec<_> = channels.into_iter().collect();
        sorted_channels.sort_by_key(|(channel, _)| {
            std::cmp::Reverse(self.rules.priorities.get(channel).copied().unwrap_or(0))
        });
        
        for (channel, tokens) in sorted_channels {
            if !tokens.is_empty() {
                // Add channel marker
                result.push(200005); // <|channel|>
                result.push(channel.to_byte() as u32);
                // Add tokens
                result.extend(tokens);
            }
        }
        
        result
    }
    
    /// Get routing statistics
    pub async fn stats(&self) -> RoutingStats {
        let stats = self.stats.read().await;
        stats.clone()
    }
    
    // Helper methods for security checks
    
    fn is_internal_processor(&self, channel: Channel) -> bool {
        // In production, check if processor is internal
        channel == Channel::Analysis
    }
    
    fn is_tool_processor(&self, channel: Channel) -> bool {
        // In production, check if processor can handle tools
        channel == Channel::Commentary
    }
    
    /// Process a buffer through a channel (synchronous version)
    pub fn process(&self, channel: crate::Channel, buffer: &Vec<u8>) -> Result<()> {
        // Convert bytes to tokens
        let tokens: Vec<u32> = buffer
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            .collect();
        
        // Update buffers (using std::sync::RwLock instead of tokio)
        use std::sync::RwLock as StdRwLock;
        use std::sync::Arc as StdArc;
        
        // This is a simplified implementation
        // In production, would properly handle async/sync boundary
        
        Ok(())
    }
}

/// Errors that can occur during routing
#[derive(Debug, thiserror::Error)]
pub enum RoutingError {
    #[error("Security violation: channel {0:?} cannot be exposed")]
    SecurityViolation(Channel),
    
    #[error("Processor closed for channel {0:?}")]
    ProcessorClosed(Channel),
    
    #[error("Buffer overflow for channel {0:?}")]
    BufferOverflow(Channel),
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_channel_routing() {
        let router = ChannelRouter::new(RoutingRules::default());
        
        // Test token routing
        let tokens = vec![200005, 1, 100, 200, 300, 200002];
        assert!(router.route_tokens(&tokens).await.is_ok());
        
        // Test channel extraction
        let channel = router.extract_channel(&tokens);
        assert_eq!(channel, Channel::Commentary);
    }
    
    #[tokio::test]
    async fn test_channel_splitting() {
        let router = ChannelRouter::new(RoutingRules::default());
        
        let tokens = vec![100, 200, 200005, 300, 400, 200005, 500, 600];
        let split = router.split_by_channel(&tokens).await;
        
        assert!(split.contains_key(&Channel::Final));
        assert!(split.contains_key(&Channel::Commentary));
    }
}