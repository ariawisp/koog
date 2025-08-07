// Harmony encoder - native OpenAI o200k_harmony tokenizer
use anyhow::{Result, Context};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use openai_harmony::chat::{Message, Role, Conversation, SystemContent};
use std::sync::Arc;
use std::collections::HashMap;
use log::{info, debug};

pub struct HarmonyEncoder {
    pub encoding: Arc<HarmonyEncoding>,
}

impl HarmonyEncoder {
    pub fn new(encoding_name: &str) -> Result<Self> {
        info!("Initializing Harmony encoder with: {}", encoding_name);
        
        // Load the o200k_harmony encoding from OpenAI
        let encoding = match encoding_name {
            "o200k_harmony" | "harmony_gpt_oss" => {
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?
            },
            _ => {
                // Default to GPT-OSS harmony encoding
                info!("Unknown encoding '{}', defaulting to HarmonyGptOss", encoding_name);
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)?
            }
        };
        
        Ok(HarmonyEncoder {
            encoding: Arc::new(encoding),
        })
    }
    
    /// Parse a token buffer into channel-separated buffers
    pub fn parse_channels(&self, buffer: &[u8]) -> Result<HashMap<crate::Channel, Vec<u8>>> {
        // Convert bytes to tokens (assuming 4 bytes per u32 token)
        let tokens: Vec<u32> = buffer
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // Parse tokens using harmony parser
        let messages = self.encoding.parse_messages_from_completion_tokens(
            tokens.clone(),
            Some(Role::Assistant)
        )?;
        
        // Separate messages by channel
        let mut channel_buffers = HashMap::new();
        
        for message in messages {
            let channel = match message.channel.as_deref() {
                Some("analysis") => crate::Channel::Analysis,
                Some("commentary") => crate::Channel::Commentary,
                Some("final") | None => crate::Channel::Final,
                _ => crate::Channel::Final,
            };
            
            // Render message back to tokens
            let message_tokens = self.encoding.render(&message, None)?;
            
            // Convert tokens to bytes
            let bytes: Vec<u8> = message_tokens
                .iter()
                .flat_map(|&token| token.to_le_bytes())
                .collect();
            
            channel_buffers.entry(channel)
                .or_insert_with(Vec::new)
                .extend(bytes);
        }
        
        Ok(channel_buffers)
    }
    
    /// Encode a contradiction resolution prompt
    pub fn encode_contradiction_resolution(&self, context_a: &[u32], context_b: &[u32]) -> Result<Vec<u8>> {
        // Decode tokens to text for the prompt
        let text_a = self.decode(context_a)?;
        let text_b = self.decode(context_b)?;
        
        // Create a resolution prompt
        let prompt = format!(
            "Resolve the following contradiction:\n\nContext A: {}\n\nContext B: {}\n\nProvide a coherent resolution:",
            text_a, text_b
        );
        
        // Create a conversation for the resolution
        let conversation = Conversation::from_messages([
            Message::from_role_and_content(
                Role::System,
                SystemContent::new()
                    .with_reasoning_effort(openai_harmony::chat::ReasoningEffort::High)
            ),
            Message::from_role_and_content(Role::User, prompt),
        ]);
        
        // Render to tokens
        let tokens = self.encoding.render_conversation_for_completion(
            &conversation,
            Role::Assistant,
            None
        )?;
        
        // Convert tokens to bytes
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&token| token.to_le_bytes())
            .collect();
        
        Ok(bytes)
    }
    
    /// Encode text to tokens
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        // The encode_ordinary method doesn't exist, use the correct API
        let message = openai_harmony::chat::Message::from_role_and_content(
            openai_harmony::chat::Role::User,
            text
        );
        let tokens = self.encoding.render(&message, None)?;
        Ok(tokens)
    }
    
    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        // Use the Harmony library's tokenizer - it handles o200k_harmony properly
        // The tokenizer() method returns a CoreBPE which has decode_utf8
        Ok(self.encoding.tokenizer().decode_utf8(tokens)?)
    }
    
    /// Get stop tokens for inference
    pub fn stop_tokens(&self) -> Result<std::collections::HashSet<u32>> {
        self.encoding.stop_tokens()
    }
    
    /// Get special tokens
    pub fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            start: 200006,
            end: 200007,
            message: 200008,
            channel: 200005,
            constrain: 200003,
            return_token: 200002,
            call: 200012,
        }
    }
}

/// Special tokens used in Harmony format
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub start: u32,      // <|start|>
    pub end: u32,        // <|end|>
    pub message: u32,    // <|message|>
    pub channel: u32,    // <|channel|>
    pub constrain: u32,  // <|constrain|>
    pub return_token: u32, // <|return|>
    pub call: u32,       // <|call|>
}