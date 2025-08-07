// JNI Runtime Bindings - Core NoesisRuntime operations
//
// This module contains the essential JNI functions for:
// - Runtime initialization and cleanup
// - Model loading and management
// - Basic token processing
// - Statistics and introspection

use crate::NoesisRuntime;
use jni::JNIEnv;
use jni::objects::{JClass, JString, JObject, JIntArray, JByteArray};
use jni::sys::{jlong, jint, jboolean, jbyteArray, jintArray, jstring, jfloat};
use std::ffi::CStr;

/// Initialize NoesisRuntime and return opaque pointer
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeInit(
    mut env: JNIEnv,
    _class: JClass,
) -> jlong {
    eprintln!("[JNI] Initializing NoesisRuntime with unified backend...");
    
    match NoesisRuntime::new() {
        Ok(runtime) => {
            let runtime_ptr = Box::into_raw(Box::new(runtime)) as jlong;
            eprintln!("[JNI] ✅ NoesisRuntime initialized successfully, ptr: {}", runtime_ptr);
            runtime_ptr
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Failed to initialize NoesisRuntime: {}", e);
            0
        }
    }
}

/// Load model and return model handle
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeLoadModel(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    model_path_jstring: JString,
) -> jlong {
    eprintln!("[JNI] Loading model...");
    
    // Get runtime from pointer
    let runtime = unsafe {
        let ptr = runtime_ptr as *const NoesisRuntime;
        if ptr.is_null() {
            eprintln!("[JNI] ERROR: Null runtime pointer");
            return 0;
        }
        &*ptr
    };
    
    // Convert Java string to Rust string
    let model_path: String = match env.get_string(&model_path_jstring) {
        Ok(java_str) => java_str.into(),
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get model path string: {}", e);
            return 0;
        }
    };
    
    eprintln!("[JNI] Model path: {}", model_path);
    
    match runtime.load_model(&model_path) {
        Ok(model_handle) => {
            eprintln!("[JNI] ✅ Model loaded successfully, handle: {}", model_handle);
            model_handle as jlong
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Failed to load model: {}", e);
            0
        }
    }
}

/// Process tokens through the runtime
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeProcessTokens(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    input_tokens: jintArray,
) -> jintArray {
    // Get runtime from pointer
    let runtime = unsafe {
        let ptr = runtime_ptr as *const NoesisRuntime;
        if ptr.is_null() {
            eprintln!("[JNI] ERROR: Null runtime pointer");
            return JObject::null().into_raw();
        }
        &*ptr
    };
    
    // Convert Java int array to Rust Vec<u32>
    let input_array = unsafe { JIntArray::from_raw(input_tokens) };
    let input_len = match env.get_array_length(&input_array) {
        Ok(len) => len as usize,
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get input array length: {}", e);
            return JObject::null().into_raw();
        }
    };
    
    let mut tokens_buffer = vec![0i32; input_len];
    if let Err(e) = env.get_int_array_region(&input_array, 0, &mut tokens_buffer) {
        eprintln!("[JNI] ERROR: Failed to read input tokens: {}", e);
        return JObject::null().into_raw();
    }
    
    let input_tokens: Vec<u32> = tokens_buffer.into_iter().map(|i| i as u32).collect();
    
    // Process tokens
    match runtime.process_tokens(&input_tokens) {
        Ok(output_tokens) => {
            // Convert back to Java int array
            let output_jni: Vec<i32> = output_tokens.into_iter().map(|t| t as i32).collect();
            match env.new_int_array(output_jni.len() as i32) {
                Ok(output_array) => {
                    if env.set_int_array_region(&output_array, 0, &output_jni).is_ok() {
                        output_array.into_raw()
                    } else {
                        eprintln!("[JNI] ERROR: Failed to set output array region");
                        JObject::null().into_raw()
                    }
                }
                Err(e) => {
                    eprintln!("[JNI] ERROR: Failed to create output array: {}", e);
                    JObject::null().into_raw()
                }
            }
        }
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to process tokens: {}", e);
            JObject::null().into_raw()
        }
    }
}

/// Get runtime statistics
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeGetStats(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
) -> jbyteArray {
    // Get runtime from pointer
    let runtime = unsafe {
        let ptr = runtime_ptr as *const NoesisRuntime;
        if ptr.is_null() {
            eprintln!("[JNI] ERROR: Null runtime pointer");
            return JObject::null().into_raw();
        }
        &*ptr
    };
    
    // Get statistics
    let stats = runtime.get_stats();
    
    // Serialize statistics (simplified)
    let stats_json = format!(
        "{{\"gpu_memory_used\":{},\"gpu_memory_total\":{},\"cached_models\":{},\"active_parsers\":{}}}",
        stats.gpu_memory_used,
        stats.gpu_memory_total,
        stats.cached_models,
        stats.active_parsers
    );
    
    // Convert to byte array for Java
    let stats_bytes = stats_json.into_bytes();
    match env.new_byte_array(stats_bytes.len() as i32) {
        Ok(byte_array) => {
            let signed_bytes: Vec<i8> = stats_bytes.into_iter().map(|b| b as i8).collect();
            if env.set_byte_array_region(&byte_array, 0, &signed_bytes).is_ok() {
                byte_array.into_raw()
            } else {
                eprintln!("[JNI] ERROR: Failed to set stats byte array region");
                JObject::null().into_raw()
            }
        }
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to create stats byte array: {}", e);
            JObject::null().into_raw()
        }
    }
}

/// Destroy runtime and free resources
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeDestroy(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
) {
    eprintln!("[JNI] Destroying NoesisRuntime...");
    
    if runtime_ptr == 0 {
        eprintln!("[JNI] WARNING: Attempted to destroy null runtime pointer");
        return;
    }
    
    unsafe {
        let ptr = runtime_ptr as *mut NoesisRuntime;
        drop(Box::from_raw(ptr));
    }
    
    eprintln!("[JNI] ✅ NoesisRuntime destroyed successfully");
}