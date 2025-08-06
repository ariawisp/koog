package ai.koog.prompt.harmony

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Wrapper for the native streaming parser that processes tokens in real-time.
 * Provides channel-aware streaming capabilities for Harmony format.
 */
public class HarmonyStreamableParser private constructor(
    private val encoding: HarmonyEncoding,
    private val jniBridge: HarmonyJNIBridge,
    private val parserPtr: Long
) : AutoCloseable {
    
    private var closed = false
    private val json = Json { ignoreUnknownKeys = true }
    
    /**
     * Process a single token and get the current parser state.
     */
    public fun processToken(token: Int): StreamingState {
        checkNotClosed()
        val stateJson = jniBridge.processStreamingToken(parserPtr, token)
        return parseState(stateJson)
    }
    
    /**
     * Parse the JSON state from the native parser.
     */
    private fun parseState(stateJson: String): StreamingState {
        val jsonElement = json.parseToJsonElement(stateJson).jsonObject
        
        return StreamingState(
            content = jsonElement["content"]?.jsonPrimitive?.content ?: "",
            role = jsonElement["role"]?.jsonPrimitive?.content ?: "",
            channel = extractChannel(jsonElement["state"]?.toString()),
            messageCount = jsonElement["messages"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
            tokenCount = jsonElement["tokens"]?.jsonPrimitive?.content?.toIntOrNull() ?: 0,
            lastContentDelta = extractLastDelta(jsonElement["state"]?.toString())
        )
    }
    
    /**
     * Extract channel from the parser state.
     */
    private fun extractChannel(state: String?): String? {
        if (state == null) return null
        
        // Parse state to extract current channel
        return when {
            state.contains("channel\":\"analysis\"") -> "analysis"
            state.contains("channel\":\"commentary\"") -> "commentary"
            state.contains("channel\":\"final\"") -> "final"
            else -> null
        }
    }
    
    /**
     * Extract the last content delta for streaming.
     */
    private fun extractLastDelta(state: String?): String {
        // This would extract just the new content added since last token
        // For now, returning empty string - would need proper implementation
        return ""
    }
    
    override fun close() {
        if (!closed) {
            jniBridge.freeStreamingParser(parserPtr)
            closed = true
        }
    }
    
    protected fun finalize() {
        if (!closed) {
            close()
        }
    }
    
    private fun checkNotClosed() {
        if (closed) {
            throw IllegalStateException("HarmonyStreamableParser has been closed")
        }
    }
    
    public companion object {
        /**
         * Create a new streaming parser.
         */
        internal fun create(
            encoding: HarmonyEncoding,
            role: Role
        ): HarmonyStreamableParser {
            val parserPtr = encoding.jniBridge.createStreamingParser(
                encoding.encodingPtr,
                role.value
            )
            if (parserPtr == 0L) {
                throw IllegalStateException("Failed to create streaming parser")
            }
            return HarmonyStreamableParser(encoding, encoding.jniBridge, parserPtr)
        }
    }
}

/**
 * Current state of the streaming parser.
 */
public data class StreamingState(
    val content: String,
    val role: String,
    val channel: String?,
    val messageCount: Int,
    val tokenCount: Int,
    val lastContentDelta: String
)