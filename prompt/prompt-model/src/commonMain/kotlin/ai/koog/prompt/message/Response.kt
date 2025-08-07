package ai.koog.prompt.message

import ai.koog.prompt.harmony.Channel

/**
 * Simplified Response type for Harmony-native architecture.
 * Replaces the old Message class hierarchy.
 * 
 * In Noesis Runtime, all responses are HarmonyCore messages with channel separation.
 * This is a wrapper for backward compatibility during the transformation.
 */
public data class Response(
    val content: String,
    val channel: Channel = Channel.FINAL,
    val tokens: IntArray? = null,
    val metadata: Map<String, Any> = emptyMap()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is Response) return false
        return content == other.content && 
               channel == other.channel &&
               tokens?.contentEquals(other.tokens ?: intArrayOf()) == true
    }
    
    override fun hashCode(): Int {
        var result = content.hashCode()
        result = 31 * result + channel.hashCode()
        result = 31 * result + (tokens?.contentHashCode() ?: 0)
        return result
    }
    
    companion object {
        /**
         * Create a Response from tokens with channel separation
         */
        fun fromTokens(tokens: IntArray, channel: Channel = Channel.FINAL): Response {
            return Response(
                content = "", // Will be decoded by tokenizer
                channel = channel,
                tokens = tokens
            )
        }
        
        /**
         * Create a simple text response
         */
        fun text(content: String): Response {
            return Response(content = content)
        }
    }
}

/**
 * Backward compatibility - Message is now just a Response
 */
public typealias Message = Response

/**
 * Request/Response metadata for caching
 */
public data class RequestMetaInfo(
    val modelId: String? = null,
    val temperature: Float? = null,
    val maxTokens: Int? = null
)

public data class ResponseMetaInfo(
    val tokensUsed: Int? = null,
    val latencyMs: Long? = null,
    val cached: Boolean = false
)