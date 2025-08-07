use std::env;

fn main() {
    // Platform-specific build configuration
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    
    println!("cargo:warning=Building for target OS: {}", target_os);
    
    match target_os.as_str() {
        "macos" => build_metal(),
        "linux" => build_cuda(),
        "windows" => build_windows(),
        _ => println!("cargo:warning=Unsupported target OS: {}", target_os),
    }
}

fn build_metal() {
    // Only build Metal support if the feature is enabled
    if env::var("CARGO_FEATURE_METAL").is_ok() {
        println!("cargo:warning=Building with Metal support");
        
        // Get the GPT-OSS path from environment or use default
        let gptoss_path = env::var("GPTOSS_PATH")
            .or_else(|_| env::var("GPT_OSS_METAL_PATH"))
            .unwrap_or_else(|_| {
                // Default path based on the Gradle build
                let project_root = env::current_dir().unwrap();
                let gptoss_build = project_root
                    .parent().unwrap() // koog
                    .join("prompt/prompt-executor/prompt-executor-clients/noesis-executor/build/gpt-oss/build");
                gptoss_build.to_str().unwrap().to_string()
            });

        println!("cargo:warning=GPT-OSS path: {}", gptoss_path);

        // Add library search path
        println!("cargo:rustc-link-search=native={}", gptoss_path);
        
        // Link against the GPT-OSS libraries
        println!("cargo:rustc-link-lib=static=gptoss");
        println!("cargo:rustc-link-lib=static=metal-kernels");
        
        // CRITICAL: Embed the Metal shader library into our binary
        // This is required for GPT-OSS to find the shaders
        let metallib_path = format!("{}/default.metallib", gptoss_path);
        if std::path::Path::new(&metallib_path).exists() {
            println!("cargo:warning=Embedding Metal shaders from: {}", metallib_path);
            println!("cargo:rustc-link-arg=-Wl,-sectcreate,__METAL,__shaders,{}", metallib_path);
        } else {
            println!("cargo:warning=WARNING: default.metallib not found at {}", metallib_path);
        }
        
        // Link against system frameworks on macOS
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=IOKit");
        println!("cargo:rustc-link-lib=c++");
        
        // Tell cargo to rerun if the environment variable changes
        println!("cargo:rerun-if-env-changed=GPTOSS_PATH");
        println!("cargo:rerun-if-env-changed=GPT_OSS_METAL_PATH");
    }
}

fn build_cuda() {
    // Only build CUDA support if the feature is enabled
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        println!("cargo:warning=Building with CUDA support");
        
        // Link CUDA libraries
        if let Ok(cuda_path) = env::var("CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
        } else {
            // Default CUDA paths
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
            println!("cargo:rustc-link-search=native=/usr/local/cuda/lib");
        }
        
        // Link CUDA runtime
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cudnn");
        
        // For future GPT-OSS CUDA port
        if let Ok(gptoss_cuda_path) = env::var("GPTOSS_CUDA_PATH") {
            println!("cargo:rustc-link-search=native={}", gptoss_cuda_path);
            println!("cargo:rustc-link-lib=static=gptoss-cuda");
        }
        
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=GPTOSS_CUDA_PATH");
    }
}

fn build_windows() {
    // Windows-specific build (future ONNX+TensorRT)
    if env::var("CARGO_FEATURE_ONNX").is_ok() {
        println!("cargo:warning=Building with ONNX+TensorRT support");
        
        // Future: Link ONNX Runtime and TensorRT
        if let Ok(ort_path) = env::var("ORT_PATH") {
            println!("cargo:rustc-link-search=native={}/lib", ort_path);
            println!("cargo:rustc-link-lib=onnxruntime");
        }
        
        if let Ok(tensorrt_path) = env::var("TENSORRT_PATH") {
            println!("cargo:rustc-link-search=native={}/lib", tensorrt_path);
            println!("cargo:rustc-link-lib=nvinfer");
        }
        
        println!("cargo:rerun-if-env-changed=ORT_PATH");
        println!("cargo:rerun-if-env-changed=TENSORRT_PATH");
    } else if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // WSL2 or native Windows CUDA
        build_cuda();
    }
}