// JNI Memory Bindings - Cognitive memory operations
//
// This module contains JNI functions for:
// - Memory fork and merge operations  
// - Memory querying and retrieval
// - Cognitive state management

use crate::{NoesisRuntime, MemoryResult};
use jni::JNIEnv;
use jni::objects::{JClass, JString, JObject, JIntArray, JByteArray};
use jni::sys::{jlong, jint, jboolean, jbyteArray, jintArray, jstring, jfloat};

/// Fork cognitive state for parallel reasoning
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeFork(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    checkpoint_id_jstring: JString,
) -> jboolean {
    eprintln!("[JNI] Fork operation requested");
    
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
    let checkpoint_id: String = match env.get_string(&checkpoint_id_jstring) {
        Ok(java_str) => java_str.into(),
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get checkpoint ID string: {}", e);
            return 0;
        }
    };
    
    eprintln!("[JNI] Fork checkpoint ID: {}", checkpoint_id);
    
    match runtime.fork(&checkpoint_id) {
        Ok(_) => {
            eprintln!("[JNI] ✅ Fork operation completed successfully");
            1 // true
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Fork operation failed: {}", e);
            0 // false
        }
    }
}

/// Merge cognitive states after parallel reasoning
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeMerge(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    checkpoint_id_jstring: JString,
) -> jboolean {
    eprintln!("[JNI] Merge operation requested");
    
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
    let checkpoint_id: String = match env.get_string(&checkpoint_id_jstring) {
        Ok(java_str) => java_str.into(),
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get checkpoint ID string: {}", e);
            return 0;
        }
    };
    
    eprintln!("[JNI] Merge checkpoint ID: {}", checkpoint_id);
    
    match runtime.merge(&checkpoint_id) {
        Ok(_) => {
            eprintln!("[JNI] ✅ Merge operation completed successfully");
            1 // true
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Merge operation failed: {}", e);
            0 // false
        }
    }
}

/// Query memory with token-based semantic search
#[no_mangle]
pub extern "C" fn Java_ai_koog_noesis_NoesisRuntime_nativeQueryMemory(
    mut env: JNIEnv,
    _class: JClass,
    runtime_ptr: jlong,
    query_tokens: jintArray,
    k: jint,
) -> jbyteArray {
    eprintln!("[JNI] Memory query requested with k={}", k);
    
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
    let query_array = unsafe { JIntArray::from_raw(query_tokens) };
    let query_len = match env.get_array_length(&query_array) {
        Ok(len) => len as usize,
        Err(e) => {
            eprintln!("[JNI] ERROR: Failed to get query array length: {}", e);
            return JObject::null().into_raw();
        }
    };
    
    let mut tokens_buffer = vec![0i32; query_len];
    if let Err(e) = env.get_int_array_region(&query_array, 0, &mut tokens_buffer) {
        eprintln!("[JNI] ERROR: Failed to read query tokens: {}", e);
        return JObject::null().into_raw();
    }
    
    let query_tokens: Vec<u32> = tokens_buffer.into_iter().map(|i| i as u32).collect();
    eprintln!("[JNI] Query tokens: {} tokens", query_tokens.len());
    
    // Execute memory query
    match runtime.query_memory(&query_tokens, k as usize) {
        Ok(results) => {
            // Serialize memory results
            let serialized_results = serialize_memory_results(results);
            
            // Convert to Java byte array
            match env.new_byte_array(serialized_results.len() as i32) {
                Ok(byte_array) => {
                    let signed_bytes: Vec<i8> = serialized_results.into_iter().map(|b| b as i8).collect();
                    if env.set_byte_array_region(&byte_array, 0, &signed_bytes).is_ok() {
                        eprintln!("[JNI] ✅ Memory query completed successfully");
                        byte_array.into_raw()
                    } else {
                        eprintln!("[JNI] ERROR: Failed to set memory results byte array region");
                        JObject::null().into_raw()
                    }
                }
                Err(e) => {
                    eprintln!("[JNI] ERROR: Failed to create memory results byte array: {}", e);
                    JObject::null().into_raw()
                }
            }
        }
        Err(e) => {
            eprintln!("[JNI] ❌ Memory query failed: {}", e);
            JObject::null().into_raw()
        }
    }
}

/// Serialize memory results for transfer to Java
fn serialize_memory_results(results: Vec<MemoryResult>) -> Vec<u8> {
    // Use bincode serialization for now
    // TODO: Replace with FlatBuffers for zero-copy performance
    bincode::serialize(&results).unwrap_or_default()
}