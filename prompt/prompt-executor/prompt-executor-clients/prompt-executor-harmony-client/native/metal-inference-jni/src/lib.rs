use jni::objects::{JClass, JObject, JString, JPrimitiveArray, ReleaseMode};
use jni::sys::{jboolean, jint, jintArray, jlong, jstring};
use jni::JNIEnv;
use log::{debug, error, info};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::panic;
use std::sync::Arc;

#[cfg(target_os = "macos")]
mod metal;

mod error;

// use error::{InferenceError, Result}; // Currently unused

#[cfg(target_os = "macos")]
use metal::MetalInferenceEngine;

// Global engine cache for model reuse
static ENGINE_CACHE: OnceCell<RwLock<HashMap<String, Arc<MetalInferenceEngine>>>> = OnceCell::new();

fn get_engine_cache() -> &'static RwLock<HashMap<String, Arc<MetalInferenceEngine>>> {
    ENGINE_CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Initialize the Metal inference library
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_initialize(
    _env: JNIEnv,
    _class: JClass,
) -> jboolean {
    let _ = env_logger::try_init();
    
    // Set panic hook to log panics
    panic::set_hook(Box::new(|panic_info| {
        error!("Metal inference panic: {:?}", panic_info);
    }));
    
    info!("Metal inference JNI library initialized");
    
    #[cfg(target_os = "macos")]
    {
        // Check if Metal is available
        if metal::is_metal_available() {
            info!("Metal GPU acceleration available");
            return 1;
        } else {
            error!("Metal GPU acceleration not available on this system");
            return 0;
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        error!("Metal inference is only supported on macOS");
        return 0;
    }
}

/// Load a GPT-OSS model for inference
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_loadModel(
    mut env: JNIEnv,
    _class: JClass,
    model_path: JString,
) -> jlong {
    #[cfg(target_os = "macos")]
    {
        let model_path_str = match env.get_string(&model_path) {
            Ok(s) => s.to_string_lossy().to_string(),
            Err(e) => {
                error!("Failed to get model path string: {:?}", e);
                return 0;
            }
        };
        
        debug!("Loading model from: {}", model_path_str);
        
        // Check cache first
        {
            let cache = get_engine_cache().read();
            if let Some(engine) = cache.get(&model_path_str) {
                info!("Model already loaded from cache: {}", model_path_str);
                let engine_ptr = Arc::as_ptr(engine) as jlong;
                // Increment reference count since we're handing out a new reference
                std::mem::forget(engine.clone());
                return engine_ptr;
            }
        }
        
        // Load new model
        match MetalInferenceEngine::new(&model_path_str) {
            Ok(engine) => {
                let engine_arc = Arc::new(engine);
                let engine_ptr = Arc::as_ptr(&engine_arc) as jlong;
                
                // Store in cache
                {
                    let mut cache = get_engine_cache().write();
                    cache.insert(model_path_str.clone(), engine_arc.clone());
                }
                
                info!("Successfully loaded model: {}", model_path_str);
                
                // Increment reference count since we're handing out a reference
                std::mem::forget(engine_arc);
                engine_ptr
            }
            Err(e) => {
                error!("Failed to load model: {:?}", e);
                0
            }
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        error!("Metal inference is only supported on macOS");
        0
    }
}

/// Perform inference with the loaded model
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_inferTokens(
    mut env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    input_tokens: jintArray,
    max_tokens: jint,
    temperature: f32,
    top_p: f32,
) -> jintArray {
    #[cfg(target_os = "macos")]
    {
        if engine_ptr == 0 {
            error!("Invalid engine pointer");
            return JObject::null().into_raw();
        }
        
        // Get engine from pointer
        let engine = unsafe {
            Arc::from_raw(engine_ptr as *const MetalInferenceEngine)
        };
        
        // Get input tokens
        let input_array = unsafe { JPrimitiveArray::from_raw(input_tokens) };
        let input_tokens_vec = match unsafe { env.get_array_elements(&input_array, ReleaseMode::NoCopyBack) } {
            Ok(elements) => {
                let slice = unsafe {
                    std::slice::from_raw_parts(elements.as_ptr(), elements.len())
                };
                slice.to_vec()
            }
            Err(e) => {
                error!("Failed to get input tokens: {:?}", e);
                // Don't forget to re-increment reference count
                std::mem::forget(engine);
                return JObject::null().into_raw();
            }
        };
        
        debug!("Inferring with {} input tokens, max_tokens={}, temperature={}, top_p={}", 
               input_tokens_vec.len(), max_tokens, temperature, top_p);
        
        // Perform inference
        let result = {
            // Clone the Arc to keep it alive
            let engine_clone = engine.clone();
            // Re-increment reference count for the original pointer
            std::mem::forget(engine);
            
            engine_clone.infer_tokens(&input_tokens_vec, max_tokens as usize, temperature, top_p)
        };
        
        match result {
            Ok(output_tokens) => {
                debug!("Generated {} output tokens", output_tokens.len());
                
                // Convert to Java array
                match env.new_int_array(output_tokens.len() as i32) {
                    Ok(array) => {
                        if let Err(e) = env.set_int_array_region(&array, 0, &output_tokens) {
                            error!("Failed to set output tokens: {:?}", e);
                            return JObject::null().into_raw();
                        }
                        array.into_raw()
                    }
                    Err(e) => {
                        error!("Failed to create output array: {:?}", e);
                        JObject::null().into_raw()
                    }
                }
            }
            Err(e) => {
                error!("Inference failed: {:?}", e);
                JObject::null().into_raw()
            }
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        error!("Metal inference is only supported on macOS");
        JObject::null().into_raw()
    }
}

/// Release a loaded model
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_releaseModel(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) {
    #[cfg(target_os = "macos")]
    {
        if engine_ptr != 0 {
            // Decrement reference count by dropping the Arc
            unsafe {
                let _ = Arc::from_raw(engine_ptr as *const MetalInferenceEngine);
            }
            debug!("Released model reference");
        }
    }
}

/// Clear the model cache
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_clearCache(
    _env: JNIEnv,
    _class: JClass,
) {
    let mut cache = get_engine_cache().write();
    let count = cache.len();
    cache.clear();
    info!("Cleared {} models from cache", count);
}

/// Get device information
#[no_mangle]
pub extern "system" fn Java_ai_koog_prompt_executor_clients_harmony_MetalInferenceJNI_getDeviceInfo(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    #[cfg(target_os = "macos")]
    {
        let info = metal::get_device_info();
        match env.new_string(info) {
            Ok(s) => s.into_raw(),
            Err(e) => {
                error!("Failed to create device info string: {:?}", e);
                JObject::null().into_raw()
            }
        }
    }
    
    #[cfg(not(target_os = "macos"))]
    {
        match env.new_string("Metal not available on this platform") {
            Ok(s) => s.into_raw(),
            Err(_) => JObject::null().into_raw()
        }
    }
}