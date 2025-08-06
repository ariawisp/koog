package ai.koog.prompt.executor.clients.anthropic

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

/**
 * Anthropic Messages API Request
 * Uses proper kotlinx.serialization
 */
@Serializable
internal data class AnthropicMessagesRequest(
    val model: String,
    val messages: List<AnthropicMessage>,
    @SerialName("max_tokens") val maxTokens: Int,
    val system: String? = null,
    val metadata: AnthropicMetadata? = null,
    @SerialName("stop_sequences") val stopSequences: List<String>? = null,
    val stream: Boolean? = null,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    val tools: List<AnthropicTool>? = null,
    @SerialName("tool_choice") val toolChoice: AnthropicToolChoice? = null
)

/**
 * Anthropic Message
 */
@Serializable(with = AnthropicMessageSerializer::class)
internal data class AnthropicMessage(
    val role: String,
    val content: @Contextual JsonElement
)

/**
 * Anthropic Metadata
 */
@Serializable
internal data class AnthropicMetadata(
    @SerialName("user_id") val userId: String? = null
)

/**
 * Anthropic Tool Definition
 */
@Serializable
internal data class AnthropicTool(
    val name: String,
    val description: String? = null,
    @SerialName("input_schema") val inputSchema: AnthropicSchema
)

/**
 * Anthropic Schema for tool input
 */
@Serializable
internal data class AnthropicSchema(
    val type: String = "object",
    val properties: JsonElement? = null,
    val required: List<String>? = null
)

/**
 * Anthropic Tool Choice
 */
@Serializable(with = AnthropicToolChoiceSerializer::class)
internal sealed class AnthropicToolChoice {
    @Serializable
    data class Auto(val type: String = "auto") : AnthropicToolChoice()
    
    @Serializable
    data class Any(val type: String = "any") : AnthropicToolChoice()
    
    @Serializable
    data class Tool(
        val type: String = "tool",
        val name: String
    ) : AnthropicToolChoice()
}

/**
 * Anthropic Content Block - for complex content
 */
@Serializable
internal sealed class AnthropicContentBlock {
    @Serializable
    @SerialName("text")
    data class Text(
        val type: String = "text",
        val text: String
    ) : AnthropicContentBlock()
    
    @Serializable
    @SerialName("image")
    data class Image(
        val type: String = "image",
        val source: AnthropicImageSource
    ) : AnthropicContentBlock()
    
    @Serializable
    @SerialName("tool_use")
    data class ToolUse(
        val type: String = "tool_use",
        val id: String,
        val name: String,
        val input: JsonElement
    ) : AnthropicContentBlock()
    
    @Serializable
    @SerialName("tool_result")
    data class ToolResult(
        val type: String = "tool_result",
        @SerialName("tool_use_id") val toolUseId: String,
        val content: String,
        @SerialName("is_error") val isError: Boolean? = null
    ) : AnthropicContentBlock()
}

/**
 * Anthropic Image Source
 */
@Serializable
internal data class AnthropicImageSource(
    val type: String,
    @SerialName("media_type") val mediaType: String? = null,
    val data: String? = null,
    val url: String? = null
)

/**
 * Anthropic Messages Response
 */
@Serializable
internal data class AnthropicMessagesResponse(
    val id: String,
    val type: String,
    val role: String,
    val content: List<AnthropicContentBlock>,
    val model: String,
    @SerialName("stop_reason") val stopReason: String? = null,
    @SerialName("stop_sequence") val stopSequence: String? = null,
    val usage: AnthropicUsage
)

/**
 * Anthropic Usage Statistics
 */
@Serializable
internal data class AnthropicUsage(
    @SerialName("input_tokens") val inputTokens: Int,
    @SerialName("output_tokens") val outputTokens: Int
)

/**
 * Anthropic Streaming Event
 */
@Serializable
internal sealed class AnthropicStreamEvent {
    @Serializable
    @SerialName("message_start")
    data class MessageStart(
        val type: String = "message_start",
        val message: AnthropicMessagesResponse
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("content_block_start")
    data class ContentBlockStart(
        val type: String = "content_block_start",
        val index: Int,
        @SerialName("content_block") val contentBlock: AnthropicContentBlock
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("content_block_delta")
    data class ContentBlockDelta(
        val type: String = "content_block_delta",
        val index: Int,
        val delta: AnthropicDelta
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("content_block_stop")
    data class ContentBlockStop(
        val type: String = "content_block_stop",
        val index: Int
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("message_delta")
    data class MessageDelta(
        val type: String = "message_delta",
        val delta: AnthropicMessageDelta,
        val usage: AnthropicUsage? = null
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("message_stop")
    data class MessageStop(
        val type: String = "message_stop"
    ) : AnthropicStreamEvent()
    
    @Serializable
    @SerialName("ping")
    data class Ping(
        val type: String = "ping"
    ) : AnthropicStreamEvent()
}

/**
 * Anthropic Delta for streaming
 */
@Serializable
internal sealed class AnthropicDelta {
    @Serializable
    @SerialName("text_delta")
    data class TextDelta(
        val type: String = "text_delta",
        val text: String
    ) : AnthropicDelta()
    
    @Serializable
    @SerialName("input_json_delta")
    data class InputJsonDelta(
        val type: String = "input_json_delta",
        @SerialName("partial_json") val partialJson: String
    ) : AnthropicDelta()
}

/**
 * Anthropic Message Delta
 */
@Serializable
internal data class AnthropicMessageDelta(
    @SerialName("stop_reason") val stopReason: String? = null,
    @SerialName("stop_sequence") val stopSequence: String? = null
)