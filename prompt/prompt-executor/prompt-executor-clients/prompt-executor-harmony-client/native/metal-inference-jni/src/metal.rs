#![cfg(target_os = "macos")]

use crate::error::{InferenceError, Result};
use log::{debug, error, info, warn};
use parking_lot::RwLock;
use std::ffi::{c_char, c_float, CString};
use std::path::Path;
use std::ptr;
use std::sync::Arc;

// Link to CoreGraphics framework for MTLCreateSystemDefaultDevice
#[link(name = "CoreGraphics", kind = "framework")]
extern "C" {}

// GPT-OSS types from gpt-oss/types.h
#[repr(C)]
pub enum GptOssStatus {
    Success = 0,
    ErrorGeneric = 1,
    ErrorInvalidArgument = 2,
    ErrorOutOfMemory = 3,
}

#[repr(C)]
pub enum GptOssSpecialToken {
    Start = 0,
    End = 1,
    Channel = 2,
    Recipient = 3,
    Call = 4,
}

type GptOssModel = *mut std::ffi::c_void;
type GptOssTokenizer = *mut std::ffi::c_void;
type GptOssContext = *mut std::ffi::c_void;

// FFI bindings to GPT-OSS C API from gpt-oss/functions.h
extern "C" {
    fn gptoss_model_create_from_file(path: *const c_char, model_out: *mut GptOssModel) -> GptOssStatus;
    fn gptoss_model_get_tokenizer(model: GptOssModel, tokenizer_out: *mut GptOssTokenizer) -> GptOssStatus;
    fn gptoss_model_get_max_context_length(model: GptOssModel, max_context_length_out: *mut usize) -> GptOssStatus;
    fn gptoss_model_retain(model: GptOssModel) -> GptOssStatus;
    fn gptoss_model_release(model: GptOssModel) -> GptOssStatus;
    
    fn gptoss_tokenizer_get_special_token_id(tokenizer: GptOssTokenizer, token_type: GptOssSpecialToken, token_id_out: *mut u32) -> GptOssStatus;
    fn gptoss_tokenizer_get_num_text_tokens(tokenizer: GptOssTokenizer, num_text_tokens_out: *mut u32) -> GptOssStatus;
    fn gptoss_tokenizer_retain(tokenizer: GptOssTokenizer) -> GptOssStatus;
    fn gptoss_tokenizer_release(tokenizer: GptOssTokenizer) -> GptOssStatus;
    
    fn gptoss_context_create(model: GptOssModel, context_length: usize, context_out: *mut GptOssContext) -> GptOssStatus;
    fn gptoss_context_get_num_tokens(context: GptOssContext, num_tokens_out: *mut usize) -> GptOssStatus;
    fn gptoss_context_get_max_tokens(context: GptOssContext, max_tokens_out: *mut usize) -> GptOssStatus;
    fn gptoss_context_append_tokens(context: GptOssContext, num_tokens: usize, tokens: *const u32) -> GptOssStatus;
    fn gptoss_context_reset(context: GptOssContext) -> GptOssStatus;
    fn gptoss_context_process(context: GptOssContext) -> GptOssStatus;
    fn gptoss_context_sample(context: GptOssContext, temperature: c_float, seed: u64, token_out: *mut u32) -> GptOssStatus;
    fn gptoss_context_retain(context: GptOssContext) -> GptOssStatus;
    fn gptoss_context_release(context: GptOssContext) -> GptOssStatus;
}

pub struct MetalInferenceEngine {
    model: GptOssModel,
    tokenizer: GptOssTokenizer,
    context: Arc<RwLock<GptOssContext>>,
    max_context_length: usize,
    end_token: u32,
}

unsafe impl Send for MetalInferenceEngine {}
unsafe impl Sync for MetalInferenceEngine {}

impl MetalInferenceEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        info!("Initializing Metal inference engine with model: {}", model_path);
        
        // Check if model file exists
        if !Path::new(model_path).exists() {
            return Err(InferenceError::ModelNotFound {
                path: model_path.to_string(),
            });
        }
        
        // Verify Metal is available
        if !is_metal_available() {
            return Err(InferenceError::DeviceNotAvailable);
        }
        
        info!("Metal device available");
        
        // Load GPT-OSS model via C API
        let c_path = CString::new(model_path)
            .map_err(|e| InferenceError::InvalidInput(format!("Invalid model path: {}", e)))?;
        
        let mut model: GptOssModel = ptr::null_mut();
        let status = unsafe { gptoss_model_create_from_file(c_path.as_ptr(), &mut model) };
        
        if !matches!(status, GptOssStatus::Success) || model.is_null() {
            return Err(InferenceError::ModelLoadFailed {
                path: model_path.to_string(),
            });
        }
        
        // Get tokenizer
        let mut tokenizer: GptOssTokenizer = ptr::null_mut();
        let status = unsafe { gptoss_model_get_tokenizer(model, &mut tokenizer) };
        if !matches!(status, GptOssStatus::Success) || tokenizer.is_null() {
            unsafe { gptoss_model_release(model) };
            return Err(InferenceError::ModelLoadFailed {
                path: model_path.to_string(),
            });
        }
        
        // Get model parameters
        let mut max_context_length: usize = 0;
        let status = unsafe { gptoss_model_get_max_context_length(model, &mut max_context_length) };
        if !matches!(status, GptOssStatus::Success) {
            unsafe { 
                gptoss_tokenizer_release(tokenizer);
                gptoss_model_release(model);
            }
            return Err(InferenceError::ModelLoadFailed {
                path: model_path.to_string(),
            });
        }
        
        // Get end token ID
        let mut end_token: u32 = 0;
        let status = unsafe { gptoss_tokenizer_get_special_token_id(tokenizer, GptOssSpecialToken::End, &mut end_token) };
        if !matches!(status, GptOssStatus::Success) {
            unsafe { 
                gptoss_tokenizer_release(tokenizer);
                gptoss_model_release(model);
            }
            return Err(InferenceError::ModelLoadFailed {
                path: model_path.to_string(),
            });
        }
        
        info!("Model loaded successfully. Max context: {}, End token: {}", max_context_length, end_token);
        
        // Create inference context
        let mut context: GptOssContext = ptr::null_mut();
        let status = unsafe { gptoss_context_create(model, max_context_length, &mut context) };
        if !matches!(status, GptOssStatus::Success) || context.is_null() {
            unsafe { 
                gptoss_tokenizer_release(tokenizer);
                gptoss_model_release(model);
            }
            return Err(InferenceError::ContextCreation(
                "Failed to create inference context".to_string(),
            ));
        }
        
        Ok(Self {
            model,
            tokenizer,
            context: Arc::new(RwLock::new(context)),
            max_context_length,
            end_token,
        })
    }
    
    pub fn infer_tokens(
        &self,
        input_tokens: &[i32],
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<i32>> {
        debug!(
            "Starting inference: {} input tokens, max_tokens={}, temp={}, top_p={}",
            input_tokens.len(),
            max_tokens,
            temperature,
            top_p
        );
        
        let context = self.context.write();
        
        // Reset context for new inference
        let status = unsafe { gptoss_context_reset(*context) };
        if !matches!(status, GptOssStatus::Success) {
            return Err(InferenceError::InferenceFailed("Failed to reset context".to_string()));
        }
        
        // Convert i32 tokens to u32 for GPT-OSS API
        let input_tokens_u32: Vec<u32> = input_tokens.iter().map(|&t| t as u32).collect();
        
        // Append input tokens to context
        let status = unsafe {
            gptoss_context_append_tokens(
                *context,
                input_tokens_u32.len(),
                input_tokens_u32.as_ptr(),
            )
        };
        if !matches!(status, GptOssStatus::Success) {
            return Err(InferenceError::InferenceFailed("Failed to append input tokens".to_string()));
        }
        
        let mut initial_length: usize = 0;
        unsafe { gptoss_context_get_num_tokens(*context, &mut initial_length) };
        debug!("Context length after input: {}", initial_length);
        
        // Generate response tokens
        let mut response_tokens = Vec::with_capacity(max_tokens);
        
        // Use current timestamp as seed for sampling
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        for i in 0..max_tokens {
            // Check if we're approaching context limit
            let mut current_length: usize = 0;
            unsafe { gptoss_context_get_num_tokens(*context, &mut current_length) };
            if current_length >= self.max_context_length - 10 {
                warn!("Approaching context limit: {}/{}", current_length, self.max_context_length);
                break;
            }
            
            // Process context to prepare for sampling
            let status = unsafe { gptoss_context_process(*context) };
            if !matches!(status, GptOssStatus::Success) {
                return Err(InferenceError::InferenceFailed("Failed to process context".to_string()));
            }
            
            // Sample next token
            let mut token: u32 = 0;
            let status = unsafe { gptoss_context_sample(*context, temperature, seed + i as u64, &mut token) };
            if !matches!(status, GptOssStatus::Success) {
                return Err(InferenceError::InferenceFailed("Failed to sample token".to_string()));
            }
            
            // Check for end token
            if token == self.end_token {
                debug!("End token encountered after {} tokens", i);
                break;
            }
            
            response_tokens.push(token as i32);
            
            // Append generated token back to context for next prediction
            let status = unsafe {
                gptoss_context_append_tokens(*context, 1, &token)
            };
            if !matches!(status, GptOssStatus::Success) {
                return Err(InferenceError::InferenceFailed("Failed to append generated token".to_string()));
            }
            
            // Log progress periodically
            if i > 0 && i % 50 == 0 {
                debug!("Generated {} tokens...", i);
            }
        }
        
        info!("Inference complete: generated {} tokens", response_tokens.len());
        Ok(response_tokens)
    }
    
    pub fn reset_context(&self) {
        let context = self.context.write();
        let status = unsafe { gptoss_context_reset(*context) };
        if matches!(status, GptOssStatus::Success) {
            debug!("Context reset");
        } else {
            error!("Failed to reset context");
        }
    }
    
    pub fn get_context_length(&self) -> usize {
        let context = self.context.read();
        let mut length: usize = 0;
        unsafe { gptoss_context_get_num_tokens(*context, &mut length) };
        length
    }
    
    pub fn get_max_context_length(&self) -> usize {
        self.max_context_length
    }
}

impl Drop for MetalInferenceEngine {
    fn drop(&mut self) {
        let mut context = self.context.write();
        unsafe {
            if !(*context).is_null() {
                gptoss_context_release(*context);
                *context = ptr::null_mut();
            }
            if !self.tokenizer.is_null() {
                gptoss_tokenizer_release(self.tokenizer);
            }
            if !self.model.is_null() {
                gptoss_model_release(self.model);
            }
        }
        info!("Metal inference engine cleaned up");
    }
}

/// Check if Metal is available on this system
pub fn is_metal_available() -> bool {
    // For now, just check if we're on macOS
    // The actual GPT-OSS library will handle Metal device access
    cfg!(target_os = "macos")
}

/// Get information about the Metal device
pub fn get_device_info() -> String {
    if is_metal_available() {
        "Metal device available (via GPT-OSS)".to_string()
    } else {
        "No Metal device available".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metal_availability() {
        let available = is_metal_available();
        println!("Metal available: {}", available);
        if available {
            let info = get_device_info();
            println!("Device info:\n{}", info);
        }
    }
}