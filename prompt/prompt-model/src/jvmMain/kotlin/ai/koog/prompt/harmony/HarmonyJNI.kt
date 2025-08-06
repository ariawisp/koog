package ai.koog.prompt.harmony

import java.nio.file.Files
import java.nio.file.StandardCopyOption

/**
 * JNI bridge to the Rust openai-harmony library.
 * 
 * This class provides low-level access to the native Harmony implementation.
 * For high-level usage, use HarmonyProviderAdapter instead.
 */
public class HarmonyJNIBridge {
    
    init {
        HarmonyPlatformSupport.checkSupport()
        loadNativeLibrary()
    }
    
    /**
     * Load a Harmony encoding by name.
     * @param name The encoding name (e.g., "harmony_gpt_oss")
     * @return Pointer to the native encoding object
     */
    public external fun loadEncoding(name: String): Long
    
    /**
     * Get available stop tokens for a given encoding.
     * @param encodingPtr Pointer to the encoding
     * @return Array of stop token IDs
     */
    public external fun getStopTokens(encodingPtr: Long): IntArray
    
    /**
     * Render a conversation to tokens.
     * @param encodingPtr Pointer to the encoding
     * @param messagesJson JSON-serialized messages with channel information
     * @param role The role to complete as (usually "assistant")
     * @param configJson Configuration for rendering options
     * @return Array of token IDs
     */
    public external fun renderConversation(
        encodingPtr: Long,
        messagesJson: String,
        role: String,
        configJson: String
    ): IntArray
    
    /**
     * Parse tokens back into messages.
     * @param encodingPtr Pointer to the encoding
     * @param tokens Array of token IDs
     * @param role The role that generated the tokens
     * @return JSON-serialized messages
     */
    public external fun parseTokens(
        encodingPtr: Long,
        tokens: IntArray,
        role: String
    ): String
    
    /**
     * Create a streaming parser for real-time token processing.
     * @param encodingPtr Pointer to the encoding
     * @param role The role for parsing
     * @return Pointer to the streaming parser
     */
    public external fun createStreamingParser(
        encodingPtr: Long,
        role: String
    ): Long
    
    /**
     * Process a token through the streaming parser.
     * @param parserPtr Pointer to the parser
     * @param token The token to process
     * @return JSON state of the parser
     */
    public external fun processStreamingToken(
        parserPtr: Long,
        token: Int
    ): String
    
    /**
     * Get stop tokens for proper inference termination.
     * @param encodingPtr Pointer to the encoding
     * @param forAssistantActions Whether to get stop tokens for assistant actions
     * @return Array of stop token IDs
     */
    public external fun getStopTokensWithActions(
        encodingPtr: Long,
        forAssistantActions: Boolean
    ): IntArray
    
    /**
     * Free a streaming parser.
     * @param parserPtr Pointer to the parser to free
     */
    public external fun freeStreamingParser(parserPtr: Long)
    
    /**
     * Free a native encoding object.
     * @param encodingPtr Pointer to the encoding to free
     */
    public external fun freeEncoding(encodingPtr: Long)
    
    
    private companion object {
        private var libraryLoaded = false
        
        @Synchronized
        private fun loadNativeLibrary() {
            if (libraryLoaded) return
            
            val os = System.getProperty("os.name").lowercase()
            val arch = System.getProperty("os.arch").lowercase()
            
            val (libName, libExtension) = when {
                os.contains("linux") && arch in listOf("amd64", "x86_64") -> 
                    "harmony_jni_linux_x64" to ".so"
                os.contains("mac") && arch == "x86_64" -> 
                    "harmony_jni_macos_x64" to ".dylib"
                os.contains("mac") && arch == "aarch64" -> 
                    "harmony_jni_macos_arm64" to ".dylib"
                os.contains("windows") && arch == "amd64" -> 
                    "harmony_jni_windows_x64" to ".dll"
                else -> throw UnsupportedOperationException(
                    "No Harmony native library available for $os/$arch"
                )
            }
            
            try {
                // Try to load from system library path first
                System.loadLibrary("harmony_jni")
                libraryLoaded = true
            } catch (e: UnsatisfiedLinkError) {
                // Fall back to extracting from JAR
                val resourcePath = "/native/$libName$libExtension"
                val resource = HarmonyJNIBridge::class.java.getResourceAsStream(resourcePath)
                
                if (resource != null) {
                    val tempFile = Files.createTempFile("harmony_jni", libExtension)
                    tempFile.toFile().deleteOnExit()
                    
                    resource.use { input ->
                        Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING)
                    }
                    
                    System.load(tempFile.toAbsolutePath().toString())
                    libraryLoaded = true
                } else {
                    throw IllegalStateException(
                        "Failed to load Harmony native library. " +
                        "Library not found in system path or JAR resources: $resourcePath", e
                    )
                }
            }
        }
    }
}

/**
 * Platform support detection for Harmony.
 */
public object HarmonyPlatformSupport {
    /**
     * Check if the current platform supports Harmony.
     */
    public val isSupported: Boolean = when {
        // Check if we're on JVM
        System.getProperty("java.vm.name") != null -> {
            val os = System.getProperty("os.name").lowercase()
            val arch = System.getProperty("os.arch").lowercase()
            
            when {
                os.contains("linux") && (arch == "amd64" || arch == "x86_64") -> true
                os.contains("mac") && (arch == "aarch64" || arch == "x86_64") -> true
                os.contains("windows") && arch == "amd64" -> true
                else -> false
            }
        }
        else -> false
    }
    
    /**
     * Check platform support and throw if unsupported.
     * @throws UnsupportedOperationException if the platform is not supported
     */
    public fun checkSupport() {
        if (!isSupported) {
            val os = System.getProperty("os.name") ?: "unknown"
            val arch = System.getProperty("os.arch") ?: "unknown"
            throw UnsupportedOperationException(
                "Harmony format is not yet supported on this platform: $os $arch. " +
                "Currently supported: Linux/Mac/Windows on x64/ARM64 (JVM only)"
            )
        }
    }
}