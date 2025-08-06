package ai.koog.prompt.executor.clients.bedrock.modelfamilies.anthropic

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Anthropic data models for use within Bedrock client.
 * These are specific to Bedrock's usage of Anthropic models and separate from the main Anthropic client.
 */

@Serializable
internal data class AnthropicMessageRequest(
    val model: String,
    val messages: List<AnthropicMessage>,
    @SerialName("max_tokens")
    val maxTokens: Int,
    val temperature: Double? = null,
    @SerialName("top_p")
    val topP: Double? = null,
    @SerialName("top_k")
    val topK: Int? = null,
    val system: List<SystemAnthropicMessage>? = null,
    val tools: List<AnthropicTool>? = null,
    @SerialName("tool_choice")
    val toolChoice: AnthropicToolChoice? = null,
    val stop: List<String>? = null
)

@Serializable
internal data class AnthropicMessage(
    val role: String,
    val content: List<AnthropicContent>
)

@Serializable
internal data class SystemAnthropicMessage(
    val text: String
)

@Serializable
internal sealed class AnthropicContent {
    @Serializable
    @SerialName("text")
    data class Text(val text: String) : AnthropicContent()
    
    @Serializable
    @SerialName("image")
    data class Image(val source: ImageSource) : AnthropicContent()
    
    @Serializable
    @SerialName("tool_use")
    data class ToolUse(
        val id: String,
        val name: String,
        val input: JsonObject
    ) : AnthropicContent()
    
    @Serializable
    @SerialName("tool_result")
    data class ToolResult(
        @SerialName("tool_use_id")
        val toolUseId: String,
        val content: String
    ) : AnthropicContent()
}

@Serializable
internal sealed class ImageSource {
    @Serializable
    @SerialName("base64")
    data class Base64(
        val data: String,
        @SerialName("media_type")
        val mediaType: String
    ) : ImageSource()
}

@Serializable
internal data class AnthropicTool(
    val name: String,
    val description: String,
    @SerialName("input_schema")
    val inputSchema: AnthropicToolSchema
)

@Serializable
internal data class AnthropicToolSchema(
    val type: String = "object",
    val properties: JsonObject,
    val required: List<String>
)

@Serializable
internal sealed class AnthropicToolChoice {
    @Serializable
    @SerialName("auto")
    object Auto : AnthropicToolChoice()
    
    @Serializable
    @SerialName("none")
    object None : AnthropicToolChoice()
    
    @Serializable
    @SerialName("any")
    object Any : AnthropicToolChoice()
    
    @Serializable
    @SerialName("tool")
    data class Tool(val name: String) : AnthropicToolChoice()
}

@Serializable
internal data class AnthropicResponse(
    val id: String,
    val model: String,
    val type: String,
    val role: String,
    val content: List<AnthropicResponseContent>,
    @SerialName("stopReason")
    val stopReason: String? = null,
    val usage: AnthropicUsage? = null
)

@Serializable
internal sealed class AnthropicResponseContent {
    @Serializable
    @SerialName("text")
    data class Text(val text: String) : AnthropicResponseContent()
    
    @Serializable
    @SerialName("tool_use")
    data class ToolUse(
        val id: String,
        val name: String,
        val input: JsonObject
    ) : AnthropicResponseContent()
}

@Serializable
internal data class AnthropicUsage(
    @SerialName("inputTokens")
    val inputTokens: Int = 0,
    @SerialName("outputTokens")
    val outputTokens: Int = 0
)

@Serializable
internal data class AnthropicStreamResponse(
    val type: String,
    val delta: AnthropicStreamDelta? = null,
    val message: AnthropicStreamMessage? = null
)

@Serializable
internal data class AnthropicStreamDelta(
    val type: String? = null,
    val text: String? = null
)

@Serializable
internal data class AnthropicStreamMessage(
    val content: List<AnthropicResponseContent>? = null,
    val usage: AnthropicUsage? = null
)