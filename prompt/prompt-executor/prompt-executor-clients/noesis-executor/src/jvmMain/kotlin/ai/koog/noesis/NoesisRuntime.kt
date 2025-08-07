package ai.koog.noesis

import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Minimal JNI Wrapper for Noesis Runtime
 * 
 * ALL cognitive processing, memory management, and inference happens in Rust.
 * This is a thin Kotlin wrapper that provides a clean API for JVM integration.
 * 
 * ARCHITECTURE: Kotlin → JNI → Rust Cognitive Engine → GPU Metal
 */
public class NoesisRuntime private constructor(
    private val handle: Long
) : AutoCloseable {
    
    private val closed = AtomicBoolean(false)
    
    init {
        require(handle != 0L) { "Failed to initialize Noesis Runtime" }
    }
    
    /**
     * Load a GPT-OSS model for inference
     */
    public fun loadModel(modelPath: Path): Long {
        checkNotClosed()
        require(Files.exists(modelPath)) { "Model file not found: $modelPath" }
        
        val modelHandle = nativeLoadModel(handle, modelPath.toString())
        if (modelHandle == 0L) {
            throw RuntimeException("Failed to load model: $modelPath")
        }
        
        return modelHandle
    }
    
    /**
     * Generate text using the Rust cognitive engine
     * 
     * This leverages the full token-native cognitive pipeline:
     * - Real-time contradiction detection during generation
     * - Hardware-enforced channel boundaries
     * - GPU-accelerated semantic processing
     * - Zero-copy fork/merge reasoning
     */
    public fun generateText(
        modelHandle: Long,
        prompt: String,
        maxTokens: Int = 100,
        temperature: Float = 0.7f
    ): String {
        checkNotClosed()
        
        val result = nativeGenerateText(handle, modelHandle, prompt, maxTokens, temperature)
        if (result.isEmpty()) {
            throw RuntimeException("Text generation failed")
        }
        
        return result
    }
    
    /**
     * Stream tokens with cognitive processing
     * 
     * Each token is processed through the complete Rust cognitive pipeline
     * including real-time safety filtering and contradiction detection.
     */
    public fun streamTokens(
        modelHandle: Long,
        inputTokens: IntArray, 
        maxTokens: Int = 2048,
        temperature: Float = 0.8f,
        topP: Float = 0.95f,
        onToken: (Int) -> Boolean
    ) {
        checkNotClosed()
        
        val result = nativeStreamTokens(
            handle, 
            modelHandle,
            inputTokens, 
            maxTokens,
            temperature,
            topP,
            object : TokenCallback {
                override fun onToken(token: Int): Boolean = onToken(token)
            }
        )
        
        if (!result) {
            throw RuntimeException("Streaming failed")
        }
    }
    
    /**
     * Tokenize text using the model's tokenizer
     * 
     * This exposes the tokenization functionality so that streamTokens() can be used
     * with proper token arrays instead of requiring manual tokenization.
     */
    public fun tokenize(text: String): IntArray {
        checkNotClosed()
        return nativeTokenize(handle, text)
    }
    
    /**
     * Get runtime statistics from Rust engine
     */
    public fun getStats(): String {
        checkNotClosed()
        return nativeGetStats(handle)
    }
    
    /**
     * Get cognitive processor for advanced cognitive operations
     */
    public fun getCognitiveProcessor(): CognitiveInterface.CognitiveProcessor {
        checkNotClosed()
        return CognitiveInterface.CognitiveProcessor(handle)
    }
    
    override fun close() {
        if (closed.compareAndSet(false, true)) {
            nativeDestroy(handle)
        }
    }
    
    private fun checkNotClosed() {
        check(!closed.get()) { "Runtime is closed" }
    }
    
    // Minimal callback interface for streaming
    public interface TokenCallback {
        fun onToken(token: Int): Boolean
    }
    
    /**
     * Generate using FlatBuffers binary interface - ZERO-COPY
     */
    public fun generateBinary(requestBuffer: ByteArray): ByteArray {
        checkNotClosed()
        return nativeGenerateBinary(handle, requestBuffer)
    }
    
    /**
     * Stream using FlatBuffers binary interface - ZERO-COPY
     */
    public fun streamBinary(requestBuffer: ByteArray, callback: TokenCallback): ByteArray {
        checkNotClosed()
        return nativeStreamBinary(handle, requestBuffer, callback)
    }
    
    // Native method declarations - delegate everything to Rust
    private external fun nativeLoadModel(runtimePtr: Long, modelPath: String): Long
    private external fun nativeGenerateText(runtimePtr: Long, modelHandle: Long, prompt: String, maxTokens: Int, temperature: Float): String
    private external fun nativeTokenize(runtimePtr: Long, text: String): IntArray
    private external fun nativeStreamTokens(
        runtimePtr: Long, 
        modelHandle: Long,
        inputTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        callback: TokenCallback
    ): Boolean
    private external fun nativeGetStats(runtimePtr: Long): String
    private external fun nativeDestroy(runtimePtr: Long)
    
    // ZERO-COPY Binary FlatBuffers Interface - ACTIVATED!
    internal external fun nativeGenerateBinary(runtimePtr: Long, requestBuffer: ByteArray): ByteArray
    internal external fun nativeStreamBinary(runtimePtr: Long, requestBuffer: ByteArray, callback: TokenCallback): ByteArray
    
    public companion object {
        
        init {
            // Load native Rust library
            try {
                System.loadLibrary("noesis_runtime")
            } catch (e: UnsatisfiedLinkError) {
                loadFromResources()
            }
        }
        
        private fun loadFromResources() {
            val libName = System.mapLibraryName("noesis_runtime")
            val osArch = System.getProperty("os.arch").lowercase()
            
            // Try architecture-specific paths first
            val archPaths = listOf(
                "/native/$osArch/$libName",     // e.g., /native/aarch64/libnoesis_runtime.dylib
                "/native/aarch64/$libName",     // Fallback for Apple Silicon
                "/native/arm64/$libName",       // Alternative Apple Silicon naming
                "/native/$libName"              // Legacy fallback
            )
            
            for (resourcePath in archPaths) {
                NoesisRuntime::class.java.getResourceAsStream(resourcePath)?.use { input ->
                    val tempFile = Files.createTempFile("noesis_runtime", libName.substringAfter('.'))
                    tempFile.toFile().deleteOnExit()
                    
                    Files.copy(input, tempFile, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
                    System.load(tempFile.toString())
                    return // Successfully loaded
                }
            }
            
            throw UnsatisfiedLinkError("Cannot find noesis_runtime library in any of: $archPaths")
        }
        
        /**
         * Create new Noesis Runtime instance
         * Initializes the complete Rust cognitive engine with GPU acceleration
         */
        @JvmStatic
        public fun create(): NoesisRuntime {
            val handle = nativeInit()
            if (handle == 0L) {
                throw RuntimeException("Failed to initialize Noesis Runtime")
            }
            
            return NoesisRuntime(handle)
        }
        
        @JvmStatic
        private external fun nativeInit(): Long
    }
}
