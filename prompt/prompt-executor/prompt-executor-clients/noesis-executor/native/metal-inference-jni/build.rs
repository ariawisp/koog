use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/metal.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    if target_os == "macos" {
        // Get the path to GPT-OSS Metal library from environment or use default
        let gpt_oss_path = env::var("GPTOSS_PATH")
            .or_else(|_| env::var("GPT_OSS_METAL_PATH"))
            .unwrap_or_else(|_| {
                // Default path to the actual GPT-OSS build
                let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
                path.pop(); // Go up from metal-inference-jni
                path.pop(); // Go up from native
                path.pop(); // Go up from prompt-executor-harmony-client
                path.pop(); // Go up from prompt-executor-clients
                path.pop(); // Go up from prompt-executor
                path.pop(); // Go up from prompt
                path.pop(); // Go up from koog
                path.push("gpt-oss-main");
                path.push("gpt_oss");
                path.push("metal");
                path.push("_build");
                path.to_string_lossy().to_string()
            });
        
        let gpt_oss_lib_path = PathBuf::from(&gpt_oss_path);
        println!("cargo:warning=Looking for GPT-OSS Metal library in: {}", gpt_oss_lib_path.display());
        
        // Check if we should use stub implementation
        if env::var("METAL_INFERENCE_STUB").is_ok() {
            println!("cargo:warning=Building with stub implementation (METAL_INFERENCE_STUB is set)");
            build_stub();
        } else {
            // Check if the library exists
            let lib_path = gpt_oss_lib_path.join("libgptoss.a");
            if !lib_path.exists() {
                println!("cargo:warning=GPT-OSS library not found at {}, using stub", lib_path.display());
                build_stub();
            } else {
                println!("cargo:warning=Found GPT-OSS library at {}", lib_path.display());
                link_gpt_oss(&gpt_oss_lib_path);
            }
        }
        
        // Link to system frameworks (needed for both stub and real implementation)
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=IOKit");
        
        // Set up JNI paths
        if let Ok(java_home) = env::var("JAVA_HOME") {
            println!("cargo:rustc-link-search=native={}/lib", java_home);
            println!("cargo:rustc-link-search=native={}/lib/server", java_home);
        }
    } else {
        println!("cargo:warning=Metal inference is only supported on macOS");
        build_stub();
    }
}

fn build_stub() {
    // Compile stub implementation
    cc::Build::new()
        .file("src/stubs/gpt_oss_stub.c")
        .warnings(false)
        .compile("gptoss_stub");
}

fn link_gpt_oss(gpt_oss_lib_path: &PathBuf) {
    // Link to actual GPT-OSS libraries
    println!("cargo:rustc-link-search=native={}", gpt_oss_lib_path.display());
    println!("cargo:rustc-link-lib=static=gptoss");
    println!("cargo:rustc-link-lib=static=metal-kernels");
    
    // Check for Metal shader library
    let metallib_path = gpt_oss_lib_path.join("default.metallib");
    if metallib_path.exists() {
        println!("cargo:warning=Found Metal shader library at {}", metallib_path.display());
        
        // Copy metallib to output directory so it's available at runtime
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_metallib = out_dir.join("default.metallib");
        
        if let Ok(_) = std::fs::copy(&metallib_path, &target_metallib) {
            println!("cargo:warning=Copied Metal shaders to {}", target_metallib.display());
        }
        
        // Note: Embedding shaders with -sectcreate requires post-build processing
        // This will be handled by the Gradle build
    } else {
        println!("cargo:warning=Metal shader library not found at {}", metallib_path.display());
    }
    
    // Include headers
    let include_path = gpt_oss_lib_path.parent()
        .and_then(|p| Some(p.join("include")))
        .unwrap_or_else(|| gpt_oss_lib_path.clone());
    
    if include_path.exists() {
        println!("cargo:include={}", include_path.display());
    }
}