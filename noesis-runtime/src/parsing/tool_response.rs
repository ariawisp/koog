// Tool response parsing - unified handler for all tool output formats
// This replaces the scattered parsing logic in Kotlin

use super::json::JsonParser;
use super::markdown::MarkdownParser;
use crate::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Unified tool response parser
/// Handles various output formats from tools (JSON, Markdown, plain text)
pub struct ToolResponseParser {
    json_parser: JsonParser,
    markdown_parser: MarkdownParser,
}

impl ToolResponseParser {
    pub fn new() -> Self {
        Self {
            json_parser: JsonParser::new(),
            markdown_parser: MarkdownParser::new(),
        }
    }
    
    /// Parse tool response based on content type hint
    pub fn parse_response(&self, content: &str, content_type: Option<&str>) -> ToolResponse {
        match content_type {
            Some("json") | Some("application/json") => {
                self.parse_json_response(content)
            }
            Some("markdown") | Some("text/markdown") => {
                self.parse_markdown_response(content)
            }
            Some("error") => {
                ToolResponse::Error(ToolError {
                    message: content.to_string(),
                    code: None,
                })
            }
            _ => {
                // Auto-detect format
                self.auto_parse(content)
            }
        }
    }
    
    /// Parse JSON tool response
    fn parse_json_response(&self, content: &str) -> ToolResponse {
        match self.json_parser.parse_from_response(content) {
            Ok(value) => ToolResponse::Success(ToolResult::Json(value)),
            Err(e) => ToolResponse::Error(ToolError {
                message: format!("Failed to parse JSON: {}", e),
                code: Some("PARSE_ERROR".to_string()),
            }),
        }
    }
    
    /// Parse Markdown tool response
    fn parse_markdown_response(&self, content: &str) -> ToolResponse {
        match self.markdown_parser.parse(content) {
            Ok(markdown_content) => {
                // Extract structured data from markdown
                let mut data = ToolResult::Markdown(MarkdownResult {
                    raw: content.to_string(),
                    code_blocks: markdown_content.code_blocks.into_iter().map(|block| {
                        (block.language.unwrap_or_default(), block.content)
                    }).collect(),
                    tables: markdown_content.tables.into_iter().map(|table| {
                        TableData {
                            headers: table.headers,
                            rows: table.rows,
                        }
                    }).collect(),
                });
                
                ToolResponse::Success(data)
            }
            Err(e) => ToolResponse::Error(ToolError {
                message: format!("Failed to parse Markdown: {}", e),
                code: Some("PARSE_ERROR".to_string()),
            }),
        }
    }
    
    /// Auto-detect response format
    fn auto_parse(&self, content: &str) -> ToolResponse {
        let trimmed = content.trim();
        
        // Try JSON first (most common for tools)
        if trimmed.starts_with('{') || trimmed.starts_with('[') {
            if let Ok(value) = self.json_parser.parse_from_response(content) {
                return ToolResponse::Success(ToolResult::Json(value));
            }
        }
        
        // Check for markdown indicators
        if content.contains("```") || content.contains("###") || content.contains('|') {
            return self.parse_markdown_response(content);
        }
        
        // Default to plain text
        ToolResponse::Success(ToolResult::Text(content.to_string()))
    }
    
    /// Extract specific fields from tool response
    pub fn extract_field(&self, response: &ToolResponse, field_path: &str) -> Option<String> {
        match response {
            ToolResponse::Success(ToolResult::Json(value)) => {
                // Navigate JSON path (e.g., "data.items[0].name")
                self.navigate_json_path(value, field_path)
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        v => Some(v.to_string()),
                    })
            }
            ToolResponse::Success(ToolResult::Text(text)) => Some(text.clone()),
            _ => None,
        }
    }
    
    /// Navigate JSON using dot notation path
    fn navigate_json_path<'a>(&self, value: &'a Value, path: &str) -> Option<&'a Value> {
        let mut current = value;
        
        for segment in path.split('.') {
            // Handle array indexing
            if let Some(bracket_start) = segment.find('[') {
                let field = &segment[..bracket_start];
                let index_str = &segment[bracket_start + 1..segment.len() - 1];
                let index: usize = index_str.parse().ok()?;
                
                current = current.get(field)?.get(index)?;
            } else {
                current = current.get(segment)?;
            }
        }
        
        Some(current)
    }
}

/// Tool response types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResponse {
    Success(ToolResult),
    Error(ToolError),
    Pending,  // For async tools
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolResult {
    Json(Value),
    Text(String),
    Markdown(MarkdownResult),
    Binary(Vec<u8>),  // For future binary tool outputs
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarkdownResult {
    pub raw: String,
    pub code_blocks: Vec<(String, String)>,  // (language, content)
    pub tables: Vec<TableData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    pub message: String,
    pub code: Option<String>,
}

impl ToolResponse {
    /// Check if response is successful
    pub fn is_success(&self) -> bool {
        matches!(self, ToolResponse::Success(_))
    }
    
    /// Get error message if response is an error
    pub fn error_message(&self) -> Option<&str> {
        match self {
            ToolResponse::Error(e) => Some(&e.message),
            _ => None,
        }
    }
    
    /// Convert to JSON for serialization
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Channel-aware tool response routing
pub struct ToolResponseRouter {
    parser: ToolResponseParser,
}

impl ToolResponseRouter {
    pub fn new() -> Self {
        Self {
            parser: ToolResponseParser::new(),
        }
    }
    
    /// Route tool response to appropriate channel based on content
    pub fn route_response(&self, 
                         tool_name: &str, 
                         response: &str,
                         content_type: Option<&str>) -> (Channel, ToolResponse) {
        let parsed = self.parser.parse_response(response, content_type);
        
        // Determine channel based on response type
        let channel = match &parsed {
            ToolResponse::Error(_) => Channel::Commentary,  // Errors go to commentary
            ToolResponse::Success(result) => {
                match result {
                    ToolResult::Json(_) | ToolResult::Text(_) => {
                        // Data responses go to analysis for processing
                        Channel::Analysis
                    }
                    ToolResult::Markdown(_) => {
                        // Formatted content might go to final
                        if tool_name.contains("browser") || tool_name.contains("search") {
                            Channel::Final
                        } else {
                            Channel::Commentary
                        }
                    }
                    ToolResult::Binary(_) => Channel::Commentary,
                }
            }
            ToolResponse::Pending => Channel::Commentary,
        };
        
        (channel, parsed)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Channel {
    Final,
    Analysis,
    Commentary,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_json_tool_response() {
        let parser = ToolResponseParser::new();
        let response = r#"{"status": "success", "data": {"temperature": 22, "unit": "celsius"}}"#;
        
        match parser.parse_response(response, Some("json")) {
            ToolResponse::Success(ToolResult::Json(value)) => {
                assert_eq!(value["status"], "success");
                assert_eq!(value["data"]["temperature"], 22);
            }
            _ => panic!("Expected JSON response"),
        }
    }
    
    #[test]
    fn test_extract_field() {
        let parser = ToolResponseParser::new();
        let response = r#"{"data": {"items": [{"name": "first"}, {"name": "second"}]}}"#;
        let parsed = parser.parse_response(response, Some("json"));
        
        let field = parser.extract_field(&parsed, "data.items[0].name");
        assert_eq!(field, Some("first".to_string()));
    }
    
    #[test]
    fn test_auto_parse() {
        let parser = ToolResponseParser::new();
        
        // Should detect JSON
        let json = r#"{"key": "value"}"#;
        match parser.parse_response(json, None) {
            ToolResponse::Success(ToolResult::Json(_)) => {}
            _ => panic!("Should detect JSON"),
        }
        
        // Should detect Markdown
        let markdown = "# Header\n```python\ncode\n```";
        match parser.parse_response(markdown, None) {
            ToolResponse::Success(ToolResult::Markdown(_)) => {}
            _ => panic!("Should detect Markdown"),
        }
        
        // Should default to text
        let text = "Plain text response";
        match parser.parse_response(text, None) {
            ToolResponse::Success(ToolResult::Text(_)) => {}
            _ => panic!("Should default to text"),
        }
    }
}