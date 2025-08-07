// Unified parsing module for Noesis Runtime
// All text processing happens here in Rust for zero-copy operation

pub mod json;
pub mod markdown;
pub mod tool_response;

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Structured data parser trait
pub trait StructuredParser {
    type Output;
    
    /// Parse text into structured format
    fn parse(&self, text: &str) -> Result<Self::Output>;
    
    /// Parse directly from token stream (zero-copy)
    fn parse_tokens(&self, tokens: &[u32]) -> Result<Self::Output>;
    
    /// Pretty-print structured data
    fn pretty(&self, data: &Self::Output) -> String;
}

/// Schema definition for structured output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    pub name: String,
    pub description: String,
    pub properties: serde_json::Value,
    pub required: Vec<String>,
}

/// Parsing context with token-level operations
pub struct ParsingContext {
    // Store HarmonyEncoding and use its tokenizer() to decode on demand
    encoding: openai_harmony::HarmonyEncoding,
    buffer_pool: Arc<crate::gpu_optimized::OptimizedBufferPool>,
}

impl ParsingContext {
    pub fn new(buffer_pool: Arc<crate::gpu_optimized::OptimizedBufferPool>) -> Result<Self> {
        // Load Harmony encoding (o200k_harmony) via openai_harmony
        let enc = openai_harmony::load_harmony_encoding(openai_harmony::HarmonyEncodingName::HarmonyGptOss)?;
        Ok(Self { encoding: enc, buffer_pool })
    }
    
    /// Decode tokens to text for parsing
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        let text = self.encoding.tokenizer().decode_utf8(tokens)
            .map_err(|e| anyhow::anyhow!("Failed to decode tokens: {}", e))?;
        Ok(text)
    }
    
    /// Extract JSON from mixed content (analysis channel often has both reasoning and JSON)
    pub fn extract_json(&self, text: &str) -> Option<String> {
        // Find JSON boundaries
        let start_markers = ["{", "["];
        let end_markers = ["}", "]"];
        
        for start in start_markers {
            if let Some(start_idx) = text.find(start) {
                // Find matching end bracket
                let mut depth = 0;
                let chars: Vec<char> = text[start_idx..].chars().collect();
                let mut end_idx = start_idx;
                
                for (i, ch) in chars.iter().enumerate() {
                    match ch {
                        '{' | '[' => depth += 1,
                        '}' | ']' => {
                            depth -= 1;
                            if depth == 0 {
                                end_idx = start_idx + i + 1;
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                
                if end_idx > start_idx {
                    return Some(text[start_idx..end_idx].to_string());
                }
            }
        }
        None
    }
    
    /// Extract markdown code blocks
    pub fn extract_code_blocks(&self, text: &str) -> Vec<(Option<String>, String)> {
        let mut blocks = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut i = 0;
        
        while i < lines.len() {
            if lines[i].starts_with("```") {
                let lang = lines[i][3..].trim().to_string();
                let lang = if lang.is_empty() { None } else { Some(lang) };
                
                let mut code = Vec::new();
                i += 1;
                
                while i < lines.len() && !lines[i].starts_with("```") {
                    code.push(lines[i]);
                    i += 1;
                }
                
                if !code.is_empty() {
                    blocks.push((lang, code.join("\n")));
                }
            }
            i += 1;
        }
        
        blocks
    }
}

use std::sync::Arc;
