package ai.koog.noesis

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Paths
import kotlin.test.assertNotNull

/**
 * Test the ZERO-COPY FlatBuffers binary interface
 */
class BinaryInterfaceTest {
    
    @Test
    fun testBinaryGeneration() {
        println("🔥 Testing FlatBuffers ZERO-COPY interface...")
        
        // Create runtime
        val runtime = NoesisRuntime.create()
        assertNotNull(runtime, "Failed to create runtime")
        
        // Load the GPT-OSS-20B model
        val modelPath = Paths.get("/Users/aria/gpt-oss-20b/metal/metal/model.bin")
        if (!modelPath.toFile().exists()) {
            println("⚠️ Model not found at: $modelPath")
            println("⚠️ Skipping binary test (requires GPT-OSS-20B model)")
            return
        }
        
        val modelHandle = runtime.loadModel(modelPath)
        assertTrue(modelHandle > 0, "Failed to load model")
        println("✅ Model loaded: handle=$modelHandle")
        
        // Build a FlatBuffers request
        val systemConfig = SystemConfigData(
            modelIdentity = "You are ChatGPT, a large language model trained by OpenAI.",
            reasoningEffort = ReasoningEffortEnum.MEDIUM,
            knowledgeCutoff = "2025-01",
            conversationStartDate = "2025-08-07",
            requiredChannels = listOf("analysis", "commentary", "final")
        )
        
        val developerConfig = DeveloperConfigData(
            instructions = "You are a helpful AI assistant. Provide clear, accurate responses."
        )
        
        val messages = listOf(
            MessageData(
                role = RoleEnum.USER,
                content = "What is 2 + 2?",
                channel = null,
                recipient = null
            )
        )
        
        println("📦 Building FlatBuffers request...")
        val requestBytes = BinaryInterface.buildGenerationRequest(
            modelHandle = modelHandle,
            messages = messages,
            systemConfig = systemConfig,
            developerConfig = developerConfig,
            maxTokens = 100,
            temperature = 0.7f
        )
        
        println("📤 Request size: ${requestBytes.size} bytes")
        
        // Generate through binary interface
        println("🚀 Calling generateBinary...")
        val responseBytes = runtime.generateBinary(requestBytes)
        
        assertNotNull(responseBytes, "Binary generation returned null")
        assertTrue(responseBytes.isNotEmpty(), "Empty response")
        
        println("📥 Response size: ${responseBytes.size} bytes")
        
        // Parse response
        val response = BinaryInterface.parseGenerationResponse(responseBytes)
        assertNotNull(response, "Failed to parse response")
        
        println("✅ Binary response:")
        println("  Request ID: ${response.requestId}")
        println("  Text: ${response.text}")
        println("  Tokens: ${response.tokens?.size ?: 0} tokens")
        println("  Channels: ${response.channels?.joinToString() ?: "none"}")
        
        if (response.error != null) {
            println("  ⚠️ Error: ${response.error}")
        }
        
        // Verify we got reasonable output
        assertTrue(response.text?.isNotEmpty() == true || response.tokens?.isNotEmpty() == true,
                  "No text or tokens in response")
        
        println("\n🎉 FlatBuffers ZERO-COPY interface test complete!")
        
        runtime.close()
    }
    
    @Test
    fun testBinaryStreaming() {
        println("🌊 Testing FlatBuffers STREAMING interface...")
        
        val runtime = NoesisRuntime.create()
        assertNotNull(runtime, "Failed to create runtime")
        
        val modelPath = Paths.get("/Users/aria/gpt-oss-20b/metal/metal/model.bin")
        if (!modelPath.toFile().exists()) {
            println("⚠️ Model not found, skipping streaming test")
            return
        }
        
        val modelHandle = runtime.loadModel(modelPath)
        
        // Build request for streaming
        val messages = listOf(
            MessageData(
                role = RoleEnum.USER,
                content = "Count from 1 to 5 slowly.",
                channel = null,
                recipient = null
            )
        )
        
        val requestBytes = BinaryInterface.buildGenerationRequest(
            modelHandle = modelHandle,
            messages = messages,
            maxTokens = 200,
            temperature = 0.8f
        )
        
        println("🌊 Starting streaming with ${requestBytes.size} byte request...")
        
        // Test streaming (currently delegates to non-streaming)
        val responseBytes = runtime.streamBinary(
            requestBytes,
            object : NoesisRuntime.TokenCallback {
                override fun onToken(token: Int): Boolean {
                    print("[$token]")
                    return true // Continue streaming
                }
            }
        )
        
        println("\n✅ Streaming complete: ${responseBytes.size} bytes")
        
        runtime.close()
    }
}