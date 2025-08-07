// Simple test to verify 8-worker parallel pipeline activation
// This will help us understand if the infrastructure actually works

use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    println!("🧪 Testing 8-Worker Parallel Pipeline Activation");
    println!("================================================");
    
    // Test 1: Verify Tokio runtime is available
    println!("\n1. Testing Tokio runtime availability...");
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            println!("✅ Tokio runtime is available");
            println!("   Runtime: {:?}", handle);
        }
        Err(e) => {
            println!("❌ Tokio runtime NOT available: {}", e);
            return Ok(());
        }
    }
    
    // Test 2: Try to create Metal inference engine
    println!("\n2. Testing Metal inference engine creation...");
    use noesis_runtime::inference::MetalInferenceEngine;
    
    let engine = match MetalInferenceEngine::new_without_gptoss() {
        Ok(engine) => {
            println!("✅ Metal inference engine created successfully");
            engine
        }
        Err(e) => {
            println!("❌ Failed to create Metal inference engine: {}", e);
            return Ok(());
        }
    };
    
    // Test 3: Check if parallel pipeline is ready
    println!("\n3. Testing parallel pipeline readiness...");
    if engine.has_async_pipeline() {
        println!("✅ Async pipeline infrastructure is present");
    } else {
        println!("❌ Async pipeline infrastructure missing");
        return Ok(());
    }
    
    // Test 4: Try to trigger pipeline initialization
    println!("\n4. Testing pipeline activation...");
    let start = Instant::now();
    
    // We would need to call an inference method that triggers initialization
    // For now, let's just verify the structure exists
    println!("⚠️ Pipeline initialization requires model loading and inference call");
    println!("   This test verifies basic infrastructure is present");
    
    let duration = start.elapsed();
    println!("   Test completed in {:?}", duration);
    
    // Summary
    println!("\n📊 Test Summary:");
    println!("✅ Tokio runtime: Available");
    println!("✅ Metal engine: Created successfully"); 
    println!("✅ Pipeline infrastructure: Present");
    println!("⚠️ Full activation: Requires model + inference");
    
    println!("\n🎯 Next Steps:");
    println!("1. Load a test model");
    println!("2. Trigger inference to activate workers");
    println!("3. Measure performance difference");
    
    Ok(())
}