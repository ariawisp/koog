use jni::JNIEnv;
use jni::objects::{JClass, JString, JIntArray};
use jni::sys::{jintArray, jlong, jstring};
use serde::{Deserialize, Serialize};
use std::panic;
use openai_harmony::{load_harmony_encoding, HarmonyEncodingName, StreamableParser};
use openai_harmony::chat::Role as HarmonyRole;

/// JSON representation of a full Harmony message with channels
#[derive(Debug, Serialize, Deserialize)]
struct JsonHarmonyMessage {
    role: String,
    content: String,
    channel: Option<String>,
    recipient: Option<String>,
    content_type: Option<String>,
    name: Option<String>,
}

/// Configuration for rendering
#[derive(Debug, Serialize, Deserialize)]
struct RenderConfig {
    max_tokens: Option<u32>,
    include_system_tokens: bool,
    include_developer_tokens: bool,
}

/// Load a Harmony encoding by name.
/// Returns a pointer to the encoding object.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_loadEncoding(
    mut env: JNIEnv,
    _class: JClass,
    name: JString,
) -> jlong {
    let name_str: String = match env.get_string(&name) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    
    let result = panic::catch_unwind(|| {
        let encoding_name = match name_str.as_str() {
            "harmony_gpt_oss" => HarmonyEncodingName::HarmonyGptOss,
            _ => panic!("Unknown encoding: {}", name_str),
        };
        
        let encoding = load_harmony_encoding(encoding_name)
            .expect("Failed to load encoding");
        
        Box::into_raw(Box::new(encoding)) as jlong
    });
    
    match result {
        Ok(ptr) => ptr,
        Err(_) => 0,
    }
}

/// Render a conversation to tokens.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_renderConversation(
    mut env: JNIEnv,
    _class: JClass,
    encoding_ptr: jlong,
    messages_json: JString,
    role: JString,
    config_json: JString,
) -> jintArray {
    let messages_str: String = match env.get_string(&messages_json) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };
    
    let role_str: String = match env.get_string(&role) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };
    
    let config_str: String = match env.get_string(&config_json) {
        Ok(s) => s.into(),
        Err(_) => "{}".to_string(),
    };
    
    let result = panic::catch_unwind(|| {
        if encoding_ptr == 0 {
            panic!("Null encoding pointer");
        }
        
        let encoding = unsafe { &*(encoding_ptr as *const HarmonyEncoding) };
        
        // Parse role
        let next_role = match role_str.as_str() {
            "assistant" => Role::Assistant,
            "user" => Role::User,
            "system" => Role::System,
            "developer" => Role::Developer,
            "tool" => Role::Tool,
            _ => panic!("Unknown role: {}", role_str),
        };
        
        // Parse config
        let config: RenderConfig = serde_json::from_str(&config_str)
            .unwrap_or(RenderConfig {
                max_tokens: None,
                include_system_tokens: true,
                include_developer_tokens: true,
            });
        
        // Parse JSON messages with full Harmony support
        let json_messages: Vec<JsonHarmonyMessage> = serde_json::from_str(&messages_str)
            .expect("Failed to parse messages JSON");
        
        // Convert to Harmony messages with channel support
        let mut messages = Vec::new();
        for json_msg in json_messages {
            let role = match json_msg.role.as_str() {
                "assistant" => Role::Assistant,
                "user" => Role::User,
                "system" => Role::System,
                "developer" => Role::Developer,
                "tool" => Role::Tool,
                _ => panic!("Unknown role in message: {}", json_msg.role),
            };
            
            // Create author with optional name
            let author = if let Some(name) = json_msg.name {
                Author::new(role, Some(name))
            } else {
                Author::from(role)
            };
            
            // Create message with full Harmony structure
            let mut message = if role == Role::System {
                // Use SystemContent for system messages
                let system_content = SystemContent::new()
                    .with_model_identity(&json_msg.content);
                Message::from_role_and_content(role, system_content)
            } else if role == Role::Developer {
                // Use DeveloperContent for developer messages
                let dev_content = DeveloperContent::new()
                    .with_instructions(&json_msg.content);
                Message::from_role_and_content(role, dev_content)
            } else {
                // Use TextContent for other messages
                Message::from_role_and_content(role, TextContent::new(&json_msg.content))
            };
            
            // Set channel if provided
            if let Some(channel) = json_msg.channel {
                message = message.with_channel(&channel);
            }
            
            // Set recipient if provided
            if let Some(recipient) = json_msg.recipient {
                message = message.with_recipient(&recipient);
            }
            
            messages.push(message);
        }
        
        let conversation = Conversation::from_messages(messages);
        
        // Render with configuration
        let tokens = encoding.render_conversation_for_completion(&conversation, next_role, None)
            .expect("Failed to render conversation");
        
        tokens
    });
    
    match result {
        Ok(tokens) => {
            let output = match env.new_int_array(tokens.len() as i32) {
                Ok(a) => a,
                Err(_) => return std::ptr::null_mut(),
            };
            
            let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            if env.set_int_array_region(&output, 0, &tokens_i32).is_err() {
                return std::ptr::null_mut();
            }
            
            output.into_raw()
        }
        Err(_) => {
            env.new_int_array(0)
                .map(|a| a.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
    }
}

/// Parse tokens back into messages.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_parseTokens(
    mut env: JNIEnv,
    _class: JClass,
    encoding_ptr: jlong,
    tokens: jintArray,
    role: JString,
) -> jstring {
    let role_str: String = match env.get_string(&role) {
        Ok(s) => s.into(),
        Err(_) => return std::ptr::null_mut(),
    };
    
    let tokens_array = unsafe { JIntArray::from_raw(tokens) };
    let tokens_len = match env.get_array_length(&tokens_array) {
        Ok(len) => len,
        Err(_) => return std::ptr::null_mut(),
    };
    
    let mut tokens_vec = vec![0i32; tokens_len as usize];
    if env.get_int_array_region(&tokens_array, 0, &mut tokens_vec).is_err() {
        return std::ptr::null_mut();
    }
    
    let result = panic::catch_unwind(|| {
        if encoding_ptr == 0 {
            panic!("Null encoding pointer");
        }
        
        let encoding = unsafe { &*(encoding_ptr as *const HarmonyEncoding) };
        
        let parse_role = match role_str.as_str() {
            "assistant" => Some(Role::Assistant),
            "user" => Some(Role::User),
            "system" => Some(Role::System),
            "developer" => Some(Role::Developer),
            "tool" => Some(Role::Tool),
            _ => None,
        };
        
        let tokens_u32: Vec<u32> = tokens_vec.iter().map(|&t| t as u32).collect();
        
        // Parse tokens back into messages
        let messages = encoding.parse_messages_from_completion_tokens(tokens_u32, parse_role)
            .expect("Failed to parse tokens");
        
        // Convert to JSON with full channel information
        let json_messages: Vec<JsonHarmonyMessage> = messages.iter().map(|msg| {
            JsonHarmonyMessage {
                role: format!("{:?}", msg.author.role).to_lowercase(),
                content: msg.content.iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(""),
                channel: msg.channel.clone(),
                recipient: msg.recipient.clone(),
                content_type: msg.content_type.clone(),
                name: msg.author.name.clone(),
            }
        }).collect();
        
        serde_json::to_string(&json_messages)
            .expect("Failed to serialize messages")
    });
    
    match result {
        Ok(json) => {
            env.new_string(json)
                .map(|s| s.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
        Err(_) => {
            env.new_string("[]")
                .map(|s| s.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
    }
}

/// Create a streaming parser for real-time token processing.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_createStreamingParser(
    mut env: JNIEnv,
    _class: JClass,
    encoding_ptr: jlong,
    role: JString,
) -> jlong {
    let role_str: String = match env.get_string(&role) {
        Ok(s) => s.into(),
        Err(_) => return 0,
    };
    
    let result = panic::catch_unwind(|| {
        if encoding_ptr == 0 {
            panic!("Null encoding pointer");
        }
        
        let encoding = unsafe { &*(encoding_ptr as *const HarmonyEncoding) };
        
        let parse_role = match role_str.as_str() {
            "assistant" => Role::Assistant,
            "user" => Role::User,
            "system" => Role::System,
            "developer" => Role::Developer,
            "tool" => Role::Tool,
            _ => panic!("Unknown role: {}", role_str),
        };
        
        let parser = StreamableParser::new(encoding, Some(parse_role));
        Box::into_raw(Box::new(parser)) as jlong
    });
    
    match result {
        Ok(ptr) => ptr,
        Err(_) => 0,
    }
}

/// Process a token through the streaming parser.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_processStreamingToken(
    mut env: JNIEnv,
    _class: JClass,
    parser_ptr: jlong,
    token: i32,
) -> jstring {
    let result = panic::catch_unwind(|| {
        if parser_ptr == 0 {
            panic!("Null parser pointer");
        }
        
        let parser = unsafe { &mut *(parser_ptr as *mut StreamableParser) };
        
        // Process the token
        parser.process(token as u32)
            .expect("Failed to process token");
        
        // Get current state as JSON
        let state = serde_json::json!({
            "content": parser.current_content(),
            "role": format!("{:?}", parser.current_role()).to_lowercase(),
            "messages": parser.messages().len(),
            "tokens": parser.tokens().len(),
            "state": parser.state_json()
        });
        
        serde_json::to_string(&state)
            .expect("Failed to serialize parser state")
    });
    
    match result {
        Ok(json) => {
            env.new_string(json)
                .map(|s| s.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
        Err(_) => {
            env.new_string("{}")
                .map(|s| s.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
    }
}

/// Get stop tokens for proper inference termination.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_getStopTokens(
    mut env: JNIEnv,
    _class: JClass,
    encoding_ptr: jlong,
    for_assistant_actions: bool,
) -> jintArray {
    let result = panic::catch_unwind(|| {
        if encoding_ptr == 0 {
            panic!("Null encoding pointer");
        }
        
        let encoding = unsafe { &*(encoding_ptr as *const HarmonyEncoding) };
        
        let stop_tokens = if for_assistant_actions {
            encoding.stop_tokens_for_assistant_actions()
        } else {
            encoding.stop_tokens()
        };
        
        stop_tokens.into_iter().collect::<Vec<_>>()
    });
    
    match result {
        Ok(tokens) => {
            let output = match env.new_int_array(tokens.len() as i32) {
                Ok(a) => a,
                Err(_) => return std::ptr::null_mut(),
            };
            
            let tokens_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            if env.set_int_array_region(&output, 0, &tokens_i32).is_err() {
                return std::ptr::null_mut();
            }
            
            output.into_raw()
        }
        Err(_) => {
            env.new_int_array(0)
                .map(|a| a.into_raw())
                .unwrap_or(std::ptr::null_mut())
        }
    }
}

/// Free a streaming parser.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_freeStreamingParser(
    _env: JNIEnv,
    _class: JClass,
    parser_ptr: jlong,
) {
    if parser_ptr != 0 {
        let _ = panic::catch_unwind(|| {
            unsafe {
                let _ = Box::from_raw(parser_ptr as *mut StreamableParser);
            }
        });
    }
}

/// Free a native encoding object.
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_harmony_HarmonyJNIBridge_freeEncoding(
    _env: JNIEnv,
    _class: JClass,
    encoding_ptr: jlong,
) {
    if encoding_ptr != 0 {
        let _ = panic::catch_unwind(|| {
            unsafe {
                let _ = Box::from_raw(encoding_ptr as *mut HarmonyEncoding);
            }
        });
    }
}

