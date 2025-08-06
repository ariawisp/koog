package ai.koog.prompt.executor.clients.harmony

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.BeforeAll
import java.io.File

class MetalInferenceTest {
    
    companion object {
        private const val TEST_MODEL_PATH = "/Users/aria/gpt-oss-20b/metal/metal/model.bin"
        
        @JvmStatic
        @BeforeAll
        fun setup() {
            // Skip test if not on macOS
            val osName = System.getProperty("os.name").lowercase()
            Assumptions.assumeTrue(osName.contains("mac"), "Metal inference only works on macOS")
            
            // Skip test if model file doesn't exist
            val modelFile = File(TEST_MODEL_PATH)
            Assumptions.assumeTrue(modelFile.exists(), "Test model not found at $TEST_MODEL_PATH")
        }
    }
    
    @Test
    fun testMetalInferenceInitialization() {
        val initialized = MetalInferenceJNI.initialize()
        assertTrue(initialized, "Metal inference should initialize successfully on macOS")
    }
    
    @Test
    fun testDeviceInfo() {
        MetalInferenceJNI.initialize()
        val deviceInfo = MetalInferenceJNI.getDeviceInfo()
        assertNotNull(deviceInfo)
        assertTrue(deviceInfo.isNotEmpty())
        println("Metal device info: $deviceInfo")
    }
    
    @Test
    fun testModelLoading() {
        MetalInferenceJNI.initialize()
        
        val model = MetalInferenceJNI.loadModel(TEST_MODEL_PATH)
        assertNotNull(model, "Model should load successfully")
        
        // Clean up
        model?.let { MetalInferenceJNI.releaseModel(it) }
    }
    
    @Test
    fun testBasicInference() {
        MetalInferenceJNI.initialize()
        
        val model = MetalInferenceJNI.loadModel(TEST_MODEL_PATH)
        assertNotNull(model, "Model should load successfully")
        
        try {
            // Simple test tokens (would need actual tokenization in real use)
            val inputTokens = intArrayOf(1, 2, 3, 4, 5) // Dummy tokens for testing
            val maxTokens = 10
            val temperature = 0.7f
            val topP = 0.9f
            
            val outputTokens = model?.let {
                MetalInferenceJNI.inferTokens(
                    it,
                    inputTokens,
                    maxTokens,
                    temperature,
                    topP
                )
            }
            
            assertNotNull(outputTokens, "Inference should produce output tokens")
            assertTrue(outputTokens?.isNotEmpty() == true, "Output should contain tokens")
            assertTrue((outputTokens?.size ?: 0) <= maxTokens, "Output should not exceed max tokens")
            
            println("Generated ${outputTokens?.size} tokens")
            println("✅ Metal inference with shaders working correctly!")
        } finally {
            model?.let { MetalInferenceJNI.releaseModel(it) }
        }
    }
    
    @Test
    fun testModelCaching() {
        MetalInferenceJNI.initialize()
        
        // Load same model twice
        val model1 = MetalInferenceJNI.loadModel(TEST_MODEL_PATH)
        val model2 = MetalInferenceJNI.loadModel(TEST_MODEL_PATH)
        
        assertNotNull(model1)
        assertNotNull(model2)
        
        // Should get same cached model
        assertEquals(model1?.handle, model2?.handle, "Same model path should return cached instance")
        
        // Clean up
        model1?.let { MetalInferenceJNI.releaseModel(it) }
        
        // Clear cache
        MetalInferenceJNI.clearCache()
    }
}