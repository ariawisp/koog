package ai.koog.prompt.executor.noesis

import io.github.oshai.kotlinlogging.KotlinLogging
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.concurrent.withLock
import java.util.concurrent.locks.ReentrantLock
import java.util.concurrent.ConcurrentHashMap

/**
 * JNI wrapper for Metal GPU inference backend.
 * Provides native GPU acceleration for GPT-OSS models on macOS.
 */
public object MetalInferenceJNI {
    private val logger = KotlinLogging.logger {}
    private val initLock = ReentrantLock()
    private var initialized = false
    private val modelCache = ConcurrentHashMap<String, Long>()
    
    init {
        try {
            loadNativeLibrary()
            logger.info { "Metal inference native library loaded successfully" }
        } catch (e: Exception) {
            logger.error(e) { "Failed to load Metal inference native library: ${e.message}" }
            // Re-throw to make the error visible in tests
            throw RuntimeException("Failed to load Metal inference native library", e)
        }
    }
    
    private fun loadNativeLibrary() {
        val libName = "metal_inference_jni"
        
        try {
            // Try loading from java.library.path first
            System.loadLibrary(libName)
            logger.info { "Loaded $libName from java.library.path" }
        } catch (e: UnsatisfiedLinkError) {
            logger.debug { "Failed to load from java.library.path: ${e.message}" }
            
            // Try loading from resources
            val osName = System.getProperty("os.name").lowercase()
            val osArch = System.getProperty("os.arch").lowercase()
            
            logger.debug { "OS: $osName, Arch: $osArch" }
            
            val libFileName = when {
                osName.contains("mac") -> "lib$libName.dylib"
                osName.contains("linux") -> "lib$libName.so"
                osName.contains("win") -> "$libName.dll"
                else -> throw UnsupportedOperationException("Unsupported OS: $osName")
            }
            
            val resourcePath = "/native/$osArch/$libFileName"
            logger.debug { "Looking for native library at: $resourcePath" }
            
            val inputStream = this::class.java.getResourceAsStream(resourcePath)
                ?: throw UnsatisfiedLinkError("Native library not found in resources: $resourcePath. Available resources: ${
                    this::class.java.classLoader.getResources("").toList()
                }")
            
            val tempFile = Files.createTempFile("metal_inference_jni", ".${libFileName.substringAfterLast('.')}")
            tempFile.toFile().deleteOnExit()
            
            logger.debug { "Extracting native library to: ${tempFile.toAbsolutePath()}" }
            
            inputStream.use { input ->
                Files.newOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }
            
            System.load(tempFile.toAbsolutePath().toString())
            logger.info { "Loaded $libName from resources at ${tempFile.toAbsolutePath()}" }
        }
    }
    
    /**
     * Initialize the Metal inference backend.
     * @return true if Metal is available and initialized successfully
     */
    public fun initialize(): Boolean {
        initLock.withLock {
            if (initialized) {
                return true
            }
            
            val result = nativeInitialize()
            if (result) {
                initialized = true
                logger.info { "Metal inference backend initialized successfully" }
                logger.info { "Device info: ${getDeviceInfo()}" }
            } else {
                logger.warn { "Metal inference backend not available on this system" }
            }
            return result
        }
    }
    
    /**
     * Load a GPT-OSS model for inference.
     * Models are cached for reuse across multiple inference calls.
     * 
     * @param modelPath Path to the model file
     * @return Model handle for use with inference, or null if loading failed
     */
    public fun loadModel(modelPath: String): ModelHandle? {
        if (!initialized && !initialize()) {
            logger.error { "Metal inference not initialized" }
            return null
        }
        
        // Check cache first
        modelCache[modelPath]?.let { handle ->
            logger.debug { "Using cached model: $modelPath" }
            return ModelHandle(handle, modelPath)
        }
        
        // Validate model file exists
        val path = Paths.get(modelPath)
        if (!Files.exists(path)) {
            logger.error { "Model file not found: $modelPath" }
            return null
        }
        
        val handle = nativeLoadModel(modelPath)
        if (handle == 0L) {
            logger.error { "Failed to load model: $modelPath" }
            return null
        }
        
        modelCache[modelPath] = handle
        logger.info { "Successfully loaded model: $modelPath" }
        return ModelHandle(handle, modelPath)
    }
    
    /**
     * Perform token inference using the loaded model.
     * 
     * @param modelHandle Handle to the loaded model
     * @param inputTokens Input token IDs
     * @param maxTokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0.0 = deterministic, 1.0 = creative)
     * @param topP Nucleus sampling parameter
     * @return Generated token IDs, or null if inference failed
     */
    public fun inferTokens(
        modelHandle: ModelHandle,
        inputTokens: IntArray,
        maxTokens: Int = 2048,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): IntArray? {
        if (!initialized) {
            logger.error { "Metal inference not initialized" }
            return null
        }
        
        if (modelHandle.handle == 0L) {
            logger.error { "Invalid model handle" }
            return null
        }
        
        logger.debug { 
            "Starting inference: ${inputTokens.size} input tokens, " +
            "maxTokens=$maxTokens, temperature=$temperature, topP=$topP" 
        }
        
        val startTime = System.currentTimeMillis()
        val result = nativeInferTokens(
            modelHandle.handle,
            inputTokens,
            maxTokens,
            temperature,
            topP
        )
        
        if (result != null) {
            val elapsed = System.currentTimeMillis() - startTime
            val tokensPerSecond = if (elapsed > 0) {
                (result.size * 1000.0 / elapsed)
            } else 0.0
            
            logger.info { 
                "Inference complete: ${result.size} tokens generated in ${elapsed}ms " +
                "(%.1f tokens/sec)".format(tokensPerSecond)
            }
        } else {
            logger.error { "Inference failed" }
        }
        
        return result
    }
    
    /**
     * Release a loaded model.
     * The model will be removed from cache and its resources freed.
     */
    public fun releaseModel(modelHandle: ModelHandle) {
        if (modelHandle.handle != 0L) {
            nativeReleaseModel(modelHandle.handle)
            modelCache.remove(modelHandle.path)
            logger.debug { "Released model: ${modelHandle.path}" }
        }
    }
    
    /**
     * Clear all cached models.
     */
    public fun clearCache() {
        val count = modelCache.size
        modelCache.values.forEach { handle ->
            nativeReleaseModel(handle)
        }
        modelCache.clear()
        nativeClearCache()
        logger.info { "Cleared $count models from cache" }
    }
    
    /**
     * Get information about the Metal device.
     */
    public fun getDeviceInfo(): String {
        return if (initialized) {
            nativeGetDeviceInfo()
        } else {
            "Metal not initialized"
        }
    }
    
    /**
     * Check if Metal inference is available on this system.
     */
    public fun isAvailable(): Boolean {
        return try {
            initialize()
        } catch (e: Exception) {
            logger.debug(e) { "Metal not available" }
            false
        }
    }
    
    // Native method declarations
    private external fun nativeInitialize(): Boolean
    private external fun nativeLoadModel(modelPath: String): Long
    private external fun nativeInferTokens(
        enginePtr: Long,
        inputTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ): IntArray?
    private external fun nativeReleaseModel(enginePtr: Long)
    private external fun nativeClearCache()
    private external fun nativeGetDeviceInfo(): String
    
    /**
     * Handle to a loaded model.
     */
    public data class ModelHandle(
        internal val handle: Long,
        internal val path: String
    ) {
        override fun toString(): String = "ModelHandle(path=$path)"
    }
}