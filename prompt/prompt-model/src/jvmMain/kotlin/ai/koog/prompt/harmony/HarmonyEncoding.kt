package ai.koog.prompt.harmony

import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString

/**
 * AutoCloseable wrapper for native Harmony encoding.
 * Ensures proper resource cleanup to prevent memory leaks.
 * 
 * Provides access to full Harmony functionality including channels,
 * streaming parsing, and proper Rust API integration.
 */
public class HarmonyEncoding private constructor(
    internal val jniBridge: HarmonyJNIBridge,
    internal val encodingPtr: Long
) : AutoCloseable {
    
    private var closed = false
    
    /**
     * Render a conversation to tokens.
     * @param messages List of Harmony messages with channel information
     * @param role The role to complete as (usually "assistant")
     * @param config Configuration for rendering options
     * @return Result containing array of token IDs or error
     */
    public fun renderConversation(
        messages: List<HarmonyMessage>,
        role: Role = Role.ASSISTANT,
        config: RenderConfig = RenderConfig.default()
    ): Result<IntArray> = runCatching {
        checkNotClosed()
        val messagesJson = Json.encodeToString(messages.map { it.toJsonMessage() })
        val configJson = Json.encodeToString(config)
        jniBridge.renderConversation(encodingPtr, messagesJson, role.value, configJson)
    }
    
    /**
     * Parse tokens back into messages.
     * @param tokens Array of token IDs to parse
     * @param role The role that generated the tokens
     * @return Result containing parsed Harmony messages
     */
    public fun parseTokens(
        tokens: IntArray,
        role: Role = Role.ASSISTANT
    ): Result<List<HarmonyMessage>> = runCatching {
        checkNotClosed()
        val json = jniBridge.parseTokens(encodingPtr, tokens, role.value)
        val jsonMessages = Json.decodeFromString<List<JsonHarmonyMessage>>(json)
        jsonMessages.map { it.toHarmonyMessage() }
    }
    
    /**
     * Create a streaming parser for real-time token processing.
     * @param role The role for parsing (usually Role.ASSISTANT)
     * @return Result containing streaming parser or error
     */
    public fun createStreamingParser(
        role: Role = Role.ASSISTANT
    ): Result<HarmonyStreamableParser> = runCatching {
        checkNotClosed()
        HarmonyStreamableParser.create(this, role)
    }
    
    /**
     * Get stop tokens for proper inference termination.
     * @param forAssistantActions Whether to get stop tokens for assistant actions
     * @return Result containing stop token IDs or error
     */
    public fun getStopTokens(
        forAssistantActions: Boolean = false
    ): Result<IntArray> = runCatching {
        checkNotClosed()
        if (forAssistantActions) {
            jniBridge.getStopTokensWithActions(encodingPtr, true)
        } else {
            jniBridge.getStopTokens(encodingPtr)
        }
    }
    
    
    /**
     * Close the encoding and free native resources.
     */
    override fun close() {
        if (!closed) {
            jniBridge.freeEncoding(encodingPtr)
            closed = true
        }
    }
    
    /**
     * Ensure the encoding is freed even if close() is not called.
     */
    protected fun finalize() {
        if (!closed) {
            close()
        }
    }
    
    private fun checkNotClosed() {
        if (closed) {
            throw IllegalStateException("HarmonyEncoding has been closed")
        }
    }
    
    public companion object {
        /**
         * Load a Harmony encoding by name.
         * @param name The encoding name (e.g., "harmony_gpt_oss")
         * @param bridge Optional JNI bridge instance (creates new if not provided)
         * @return Result containing the encoding or error
         */
        public fun load(
            name: String = "harmony_gpt_oss",
            bridge: HarmonyJNIBridge = HarmonyJNIBridge()
        ): Result<HarmonyEncoding> = runCatching {
            val ptr = bridge.loadEncoding(name)
            if (ptr == 0L) {
                throw IllegalStateException("Failed to load Harmony encoding: $name")
            }
            HarmonyEncoding(bridge, ptr)
        }
    }
}

/**
 * Configuration for rendering options.
 */
@kotlinx.serialization.Serializable
public data class RenderConfig(
    val maxTokens: Int? = null,
    val includeSystemTokens: Boolean = true,
    val includeDeveloperTokens: Boolean = true
) {
    public companion object {
        public fun default(): RenderConfig = RenderConfig()
        
        public fun compact(): RenderConfig = RenderConfig(
            includeSystemTokens = false,
            includeDeveloperTokens = false
        )
        
        public fun limited(maxTokens: Int): RenderConfig = RenderConfig(
            maxTokens = maxTokens
        )
    }
}


/**
 * Enhanced JSON representation matching Rust JNI bridge.
 */
@kotlinx.serialization.Serializable
public data class JsonHarmonyMessage(
    val role: String,
    val content: String,
    val channel: String? = null,
    val recipient: String? = null,
    val contentType: String? = null,
    val name: String? = null
) {
    /**
     * Convert to HarmonyMessage.
     */
    public fun toHarmonyMessage(): HarmonyMessage {
        val roleEnum = when (role.lowercase()) {
            "user" -> Role.USER
            "assistant" -> Role.ASSISTANT
            "system" -> Role.SYSTEM
            "developer" -> Role.DEVELOPER
            "tool" -> Role.TOOL
            else -> Role.ASSISTANT
        }
        
        val author = if (name != null) {
            HarmonyAuthor.named(roleEnum, name)
        } else {
            HarmonyAuthor.from(roleEnum)
        }
        
        return HarmonyMessage(
            author = author,
            recipient = recipient,
            content = listOf(HarmonyContent.Text(content)),
            channel = channel,
            contentType = contentType
        )
    }
}

/**
 * Extension to convert HarmonyMessage to JSON format.
 */
public fun HarmonyMessage.toJsonMessage(): JsonHarmonyMessage = JsonHarmonyMessage(
    role = author.role.name.lowercase(),
    content = getTextContent(),
    channel = channel,
    recipient = recipient,
    contentType = contentType,
    name = author.name
)