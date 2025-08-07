// JNI Utility Bindings - Helper functions and utilities
//
// This module contains JNI functions for:
// - Text tokenization and encoding
// - General text processing utilities  
// - Debug and diagnostic functions

use crate::NoesisRuntime;
use jni::JNIEnv;
use jni::objects::{JClass, JString, JObject, JIntArray, JByteArray};
use jni::sys::{jlong, jint, jboolean, jbyteArray, jintArray, jstring, jfloat};

// Full Harmony tokenization - REQUIRED for GPT-OSS models
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use openai_harmony::chat::{
    Message, Role, Conversation, Author, Content, 
    SystemContent, DeveloperContent, ReasoningEffort, ChannelConfig
};
use std::sync::OnceLock;

// Global Harmony encoder instance (initialized once) 
static HARMONY_ENCODER: OnceLock<HarmonyEncoding> = OnceLock::new();

fn get_harmony_encoder() -> &'static HarmonyEncoding {
    HARMONY_ENCODER.get_or_init(|| {
        eprintln!("[JNI] 🚀 Loading Harmony encoder (o200k_harmony) - REQUIRED for GPT-OSS");
        match load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss) {
            Ok(encoder) => {
                eprintln!("[JNI] ✅ Harmony encoder loaded successfully!");
                encoder
            },
            Err(e) => {
                eprintln!("[JNI] ❌ CRITICAL: Failed to load Harmony encoder: {}", e);
                eprintln!("[JNI] GPT-OSS models REQUIRE Harmony format!");
                panic!("Cannot proceed without Harmony encoder - GPT-OSS will not work");
            }
        }
    })
}

/// Tokenize text using Harmony encoder
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeTokenize(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    text_jstring: JString,
) -> jintArray {
    eprintln!("[JNI] Tokenization requested");
    
    // Get runtime from pointer
    let runtime = unsafe {
        let ptr = runtime_ptr as *const NoesisRuntime;
        if ptr.is_null() {
            eprintln!("[JNI] ERROR: Null runtime pointer");
            return JObject::null().into_raw();
        }
        &*ptr
    };
    
    // Convert Java string to Rust string
    let text: String = match env.get_string(&text_jstring) {
        Ok(java_str) => java_str.into(),
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get text string: {}", e);
            return JObject::null().into_raw();
        }
    };
    
    eprintln!("[JNI] Text to tokenize: \"{}\" ({} chars)", 
              text.chars().take(100).collect::<String>(), text.len());
    
    // For now, simulate tokenization
    // TODO: Integrate with actual Harmony tokenizer
    let tokens = simulate_tokenization(&text);
    
    // Convert to Java int array
    let tokens_jni: Vec<i32> = tokens.into_iter().map(|t| t as i32).collect();
    match env.new_int_array(tokens_jni.len() as i32) {
        Ok(token_array) => {
            if env.set_int_array_region(&token_array, 0, &tokens_jni).is_ok() {
                eprintln!("[JNI] ✅ Tokenization completed: {} tokens", tokens_jni.len());
                token_array.into_raw()
            } else {
                eprintln!("[JNI] ERROR: Failed to set token array region");
                JObject::null().into_raw()
            }
        }
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to create token array: {}", e);
            JObject::null().into_raw()
        }
    }
}

/// Generate text from prompt (main interface)
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeGenerateText(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    model_handle: jlong,
    prompt: JString,
    max_tokens: jint,
    temperature: jfloat,
) -> jstring {
    eprintln!("[JNI] Text generation requested: model_handle={}, max_tokens={}, temperature={}", 
              model_handle, max_tokens, temperature);
    
    // Get runtime from pointer
    let runtime = unsafe {
        let ptr = runtime_ptr as *const NoesisRuntime;
        if ptr.is_null() {
            eprintln!("[JNI] ERROR: Null runtime pointer");
            return JObject::null().into_raw();
        }
        &*ptr
    };
    
    // Convert Java string to Rust string
    let prompt_str: String = match env.get_string(&prompt) {
        Ok(java_str) => java_str.into(),
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get prompt string: {}", e);
            return JObject::null().into_raw();
        }
    };
    
    eprintln!("[JNI] Input prompt: \"{}\"", prompt_str);
    
    // Create COMPLETE Harmony conversation following GPT-OSS specification
    eprintln!("[JNI] 🏗️ Building complete Harmony conversation (System+Developer+User)");
    
    // 1. System message with reasoning effort and required channels
    let system_content = SystemContent::default()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_reasoning_effort(ReasoningEffort::Medium)
        .with_conversation_start_date("2025-01-07")
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(vec!["analysis".to_string(), "commentary".to_string(), "final".to_string()]);
    
    let system_message = Message::from_role_and_content(
        Role::System,
        Content::SystemContent(system_content)
    );
    
    // 2. Developer message with instructions (this is the "system prompt")
    let developer_content = DeveloperContent::new()
        .with_instructions("You are a helpful AI assistant. Provide clear and accurate responses.");
    
    let developer_message = Message::from_role_and_content(
        Role::Developer,
        Content::DeveloperContent(developer_content)
    );
    
    // 3. User message
    let user_message = Message::from_role_and_content(
        Role::User,
        Content::from(prompt_str.clone())
    );
    
    let conversation = Conversation::from_messages(vec![system_message, developer_message, user_message]);
    eprintln!("[JNI] ✅ Complete 3-message Harmony conversation created (System→Developer→User)");
    
    // Tokenize using Harmony encoder - this creates the proper format GPT-OSS expects
    let encoder = get_harmony_encoder();
    let input_tokens = match encoder.render_conversation_for_completion(&conversation, openai_harmony::chat::Role::Assistant, None) {
        Ok(tokens) => {
            eprintln!("[JNI] 🎯 Harmony rendered {} tokens (proper GPT-OSS format)", tokens.len());
            tokens
        },
        Err(e) => {
            eprintln!("[JNI] ❌ CRITICAL: Harmony rendering failed: {}", e);
            return JObject::null().into_raw();
        }
    };
    
    // Process tokens through the optimized inference pipeline
    match runtime.process_tokens(&input_tokens) {
        Ok(output_tokens) => {
            eprintln!("[JNI] Generated {} tokens", output_tokens.len());
            
            // Decode using Harmony tokenizer - preserves proper GPT-OSS channel format
            let generated_text = match encoder.tokenizer().decode_utf8(output_tokens.clone()) {
                Ok(text) => {
                    eprintln!("[JNI] 🎯 Harmony decoded {} tokens to GPT-OSS format", output_tokens.len());
                    text
                },
                Err(e) => {
                    eprintln!("[JNI] ❌ Harmony decoding failed: {}", e);
                    eprintln!("[JNI] Falling back to raw token representation");
                    // Fallback: show the token IDs so we can debug
                    format!("DECODING_ERROR: tokens={:?}", &output_tokens[..std::cmp::min(20, output_tokens.len())])
                }
            };
            
            // Create Java string
            match env.new_string(&generated_text) {
                Ok(jstring) => {
                    eprintln!("[JNI] ✅ Text generation completed: \"{}\"", 
                              generated_text.chars().take(100).collect::<String>());
                    jstring.into_raw()
                }
                Err(e) => {
                    eprintln!("[JNI] ERROR: Failed to create Java string: {}", e);
                    JObject::null().into_raw()
                }
            }
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Text generation failed: {}", e);
            JObject::null().into_raw()
        }
    }
}

// Helper functions for tokenization/detokenization simulation

fn simulate_tokenization(text: &str) -> Vec<u32> {
    // Simulate tokenization by converting each word to a token ID
    // In production, this would use the actual Harmony tokenizer
    text.split_whitespace()
        .enumerate()
        .map(|(i, _word)| (i as u32 + 1000)) // Offset to avoid special tokens
        .collect()
}

fn simulate_detokenization(tokens: &[u32]) -> String {
    // Simulate detokenization by converting token IDs back to text
    // In production, this would use the actual Harmony detokenizer
    tokens.iter()
        .map(|&token| format!("token_{} ", token))
        .collect::<String>()
        .trim()
        .to_string()
}