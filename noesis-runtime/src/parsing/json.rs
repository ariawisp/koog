// JSON parsing for tool responses and structured output
// Replaces prompt-structure/json/JsonStructuredData.kt

use super::{StructuredParser, Schema};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, Map};
use openai_harmony;

/// JSON parser for structured LLM output
pub struct JsonParser {
    schema: Option<Schema>,
    strict: bool,
}

impl JsonParser {
    pub fn new() -> Self {
        Self {
            schema: None,
            strict: false,
        }
    }
    
    pub fn with_schema(mut self, schema: Schema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    pub fn strict_mode(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
    
    /// Validate JSON against schema
    fn validate(&self, value: &Value) -> Result<()> {
        if let Some(schema) = &self.schema {
            // Basic validation - can be enhanced with jsonschema crate
            if let Value::Object(obj) = value {
                // Check required fields
                for required in &schema.required {
                    if !obj.contains_key(required) {
                        return Err(format!("Missing required field: {}", required).into());
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Extract and parse JSON from LLM response
    pub fn parse_from_response(&self, text: &str) -> Result<Value> {
        // Try to find JSON in the response
        // LLMs often wrap JSON in markdown code blocks or mix with text
        
        // First try: direct parse
        if let Ok(value) = serde_json::from_str::<Value>(text) {
            self.validate(&value)?;
            return Ok(value);
        }
        
        // Second try: extract from code block
        if text.contains("```json") {
            if let Some(start) = text.find("```json") {
                let json_start = start + 7; // "```json".len()
                if let Some(end) = text[json_start..].find("```") {
                    let json_str = &text[json_start..json_start + end].trim();
                    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
                        self.validate(&value)?;
                        return Ok(value);
                    }
                }
            }
        }
        
        // Third try: extract any JSON object or array
        let trimmed = text.trim();
        if (trimmed.starts_with('{') && trimmed.ends_with('}')) ||
           (trimmed.starts_with('[') && trimmed.ends_with(']')) {
            if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
                self.validate(&value)?;
                return Ok(value);
            }
        }
        
        // Fourth try: find JSON boundaries in mixed content
        if let Some(json_str) = self.extract_json_from_text(text) {
            if let Ok(value) = serde_json::from_str::<Value>(&json_str) {
                self.validate(&value)?;
                return Ok(value);
            }
        }
        
        Err("No valid JSON found in response".into())
    }
    
    /// Extract JSON object or array from mixed text
    fn extract_json_from_text(&self, text: &str) -> Option<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut depth = 0;
        let mut start_idx = None;
        let mut in_string = false;
        let mut escape_next = false;
        
        for (i, &ch) in chars.iter().enumerate() {
            if escape_next {
                escape_next = false;
                continue;
            }
            
            match ch {
                '\\' if in_string => escape_next = true,
                '"' if !in_string => in_string = true,
                '"' if in_string => in_string = false,
                '{' | '[' if !in_string => {
                    if depth == 0 {
                        start_idx = Some(i);
                    }
                    depth += 1;
                }
                '}' | ']' if !in_string => {
                    depth -= 1;
                    if depth == 0 && start_idx.is_some() {
                        let json_str: String = chars[start_idx.unwrap()..=i].iter().collect();
                        return Some(json_str);
                    }
                }
                _ => {}
            }
        }
        
        None
    }
}

impl StructuredParser for JsonParser {
    type Output = Value;
    
    fn parse(&self, text: &str) -> Result<Self::Output> {
        self.parse_from_response(text)
    }
    
    fn parse_tokens(&self, tokens: &[u32]) -> Result<Self::Output> {
        // Decode tokens using Harmony tokenizer (o200k_harmony)
        let enc = openai_harmony::load_harmony_encoding(openai_harmony::HarmonyEncodingName::HarmonyGptOss)
            .map_err(|e| format!("Failed to load Harmony encoding: {}", e))?;
        let text = enc.tokenizer().decode_utf8(tokens)
            .map_err(|e| format!("Failed to decode tokens: {}", e))?;
        self.parse(&text)
    }
    
    fn pretty(&self, data: &Self::Output) -> String {
        serde_json::to_string_pretty(data).unwrap_or_else(|_| data.to_string())
    }
}

/// Tool call parser for extracting function calls from responses
pub struct ToolCallParser {
    json_parser: JsonParser,
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            json_parser: JsonParser::new(),
        }
    }
    
    /// Parse tool call from commentary channel
    pub fn parse_tool_call(&self, text: &str) -> Result<ToolCall> {
        let json = self.json_parser.parse_from_response(text)?;
        
        // Convert to tool call structure
        if let Value::Object(obj) = json {
            let name = obj.get("function")
                .or_else(|| obj.get("name"))
                .and_then(|v| v.as_str())
                .ok_or("Missing tool name")?
                .to_string();
            
            let arguments = obj.get("arguments")
                .or_else(|| obj.get("params"))
                .or_else(|| obj.get("parameters"))
                .cloned()
                .unwrap_or(Value::Object(Map::new()));
            
            Ok(ToolCall {
                name,
                arguments: serde_json::to_string(&arguments)?,
            })
        } else {
            Err("Invalid tool call format".into())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_json_from_code_block() {
        let parser = JsonParser::new();
        let text = r#"
        Here's the JSON response:
        ```json
        {
            "name": "test",
            "value": 42
        }
        ```
        "#;
        
        let result = parser.parse(text).unwrap();
        assert_eq!(result["name"], "test");
        assert_eq!(result["value"], 42);
    }
    
    #[test]
    fn test_parse_mixed_content() {
        let parser = JsonParser::new();
        let text = r#"Let me analyze this: {"status": "success", "data": [1, 2, 3]} is the result"#;
        
        let result = parser.parse(text).unwrap();
        assert_eq!(result["status"], "success");
        assert_eq!(result["data"][0], 1);
    }
    
    #[test]
    fn test_tool_call_parsing() {
        let parser = ToolCallParser::new();
        let text = r#"{"function": "get_weather", "arguments": {"location": "San Francisco"}}"#;
        
        let tool_call = parser.parse_tool_call(text).unwrap();
        assert_eq!(tool_call.name, "get_weather");
        assert!(tool_call.arguments.contains("San Francisco"));
    }
}
