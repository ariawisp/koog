package ai.koog.prompt.executor.clients.harmony

import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.assertNotNull
import java.io.File

class MetalInferenceJNITest {
    
    @Test
    fun testMetalInferenceInitialization() {
        println("Testing Metal Inference initialization...")
        
        val initialized = MetalInferenceJNI.initialize()
        println("Metal initialized: $initialized")
        assertTrue(initialized, "Metal should be available on this system")
        
        val deviceInfo = MetalInferenceJNI.getDeviceInfo()
        println("Device info: $deviceInfo")
        assertNotNull(deviceInfo)
        assertTrue(deviceInfo.isNotEmpty())
    }
    
    @Test
    fun testModelLoadingAndInference() {
        println("Testing model loading and inference...")
        
        // Initialize Metal first
        val initialized = MetalInferenceJNI.initialize()
        assertTrue(initialized, "Metal must be initialized")
        
        // Path to the actual GPT-OSS Metal model
        val modelPath = "/Users/aria/gpt-oss-20b/metal/metal/model.bin"
        val modelFile = File(modelPath)
        
        if (modelFile.exists()) {
            println("Loading model from: $modelPath")
            println("Model size: ${modelFile.length() / 1024 / 1024} MB")
            
            val model = MetalInferenceJNI.loadModel(modelPath)
            assertNotNull(model, "Model should load successfully")
            println("Model loaded with handle: $model")
            
            try {
                // Test with simple tokens
                val inputTokens = intArrayOf(1, 2, 3)
                println("Running inference with input tokens: ${inputTokens.contentToString()}")
                
                val outputTokens = MetalInferenceJNI.inferTokens(
                    model,
                    inputTokens,
                    10, // generate 10 tokens
                    0.7f, // temperature
                    0.9f  // top_p
                )
                
                assertNotNull(outputTokens, "Inference should return tokens")
                assertTrue(outputTokens.isNotEmpty(), "Should generate at least one token")
                println("Generated ${outputTokens.size} tokens: ${outputTokens.contentToString()}")
                
            } finally {
                // Always release the model
                MetalInferenceJNI.releaseModel(model)
                println("Model released")
            }
        } else {
            println("WARNING: Model file not found at: $modelPath")
            println("Skipping inference test")
        }
        
        MetalInferenceJNI.clearCache()
    }
}