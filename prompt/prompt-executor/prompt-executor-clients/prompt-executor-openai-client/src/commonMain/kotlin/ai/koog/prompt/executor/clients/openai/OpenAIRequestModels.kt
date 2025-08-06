package ai.koog.prompt.executor.clients.openai

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * OpenAI Chat Completion Request
 * Uses proper kotlinx.serialization with snake_case naming
 */
@Serializable
public data class OpenAIChatRequest(
    val model: String,
    val messages: List<OpenAIMessage>,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    @SerialName("min_tokens") val minTokens: Int? = null,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("frequency_penalty") val frequencyPenalty: Double? = null,
    @SerialName("presence_penalty") val presencePenalty: Double? = null,
    val n: Int? = null,
    val stop: List<String>? = null,
    val stream: Boolean? = null,
    @SerialName("stream_options") val streamOptions: OpenAIStreamOptions? = null,
    val tools: List<OpenAITool>? = null,
    @SerialName("tool_choice") val toolChoice: @Contextual JsonElement? = null,
    @SerialName("parallel_tool_calls") val parallelToolCalls: Boolean? = null,
    @SerialName("response_format") val responseFormat: OpenAIResponseFormat? = null,
    val logprobs: Boolean? = null,
    @SerialName("top_logprobs") val topLogprobs: Int? = null,
    val seed: Int? = null,
    val user: String? = null
)

/**
 * OpenAI Message - will use custom serializer for content handling
 */
@Serializable(with = OpenAIMessageSerializer::class)
public data class OpenAIMessage(
    val role: String,
    val content: @Contextual JsonElement? = null,
    @SerialName("tool_calls") val toolCalls: List<OpenAIToolCall>? = null,
    @SerialName("tool_call_id") val toolCallId: String? = null,
    val name: String? = null
)

/**
 * OpenAI Tool Call
 */
@Serializable
public data class OpenAIToolCall(
    val id: String,
    val type: String = "function",
    val function: OpenAIFunctionCall
)

/**
 * OpenAI Function Call
 */
@Serializable
public data class OpenAIFunctionCall(
    val name: String,
    val arguments: String
)

/**
 * OpenAI Tool Definition
 */
@Serializable
public data class OpenAITool(
    val type: String = "function",
    val function: OpenAIFunction
)

/**
 * OpenAI Function Definition
 */
@Serializable
public data class OpenAIFunction(
    val name: String,
    val description: String? = null,
    val parameters: JsonObject
)

/**
 * OpenAI Stream Options
 */
@Serializable
public data class OpenAIStreamOptions(
    @SerialName("include_usage") val includeUsage: Boolean? = null
)

/**
 * OpenAI Response Format
 */
@Serializable(with = OpenAIResponseFormatSerializer::class)
public sealed class OpenAIResponseFormat {
    @Serializable
    public data class Text(val type: String = "text") : OpenAIResponseFormat()
    
    @Serializable
    public data class JsonObject(val type: String = "json_object") : OpenAIResponseFormat()
    
    @Serializable
    public data class JsonSchema(
        val type: String = "json_schema",
        @SerialName("json_schema") val jsonSchema: OpenAIJsonSchema
    ) : OpenAIResponseFormat()
}

/**
 * OpenAI JSON Schema for structured output
 */
@Serializable
public data class OpenAIJsonSchema(
    val name: String,
    val description: String? = null,
    val schema: JsonObject,
    val strict: Boolean? = null
)

/**
 * OpenAI Chat Completion Response
 */
@Serializable
internal data class OpenAIChatResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<OpenAIChoice>,
    val usage: OpenAIUsage? = null,
    @SerialName("system_fingerprint") val systemFingerprint: String? = null
)

/**
 * OpenAI Choice in Response
 */
@Serializable
internal data class OpenAIChoice(
    val index: Int,
    val message: OpenAIMessage,
    @SerialName("finish_reason") val finishReason: String? = null,
    val logprobs: JsonObject? = null
)

/**
 * OpenAI Usage Statistics
 */
@Serializable
internal data class OpenAIUsage(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("completion_tokens") val completionTokens: Int,
    @SerialName("total_tokens") val totalTokens: Int,
    @SerialName("completion_tokens_details") val completionTokensDetails: OpenAITokenDetails? = null
)

/**
 * OpenAI Token Details
 */
@Serializable
internal data class OpenAITokenDetails(
    @SerialName("reasoning_tokens") val reasoningTokens: Int? = null
)

/**
 * OpenAI Streaming Response Chunk
 */
@Serializable
internal data class OpenAIChatStreamChunk(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<OpenAIStreamChoice>,
    val usage: OpenAIUsage? = null
)

/**
 * OpenAI Stream Choice
 */
@Serializable
internal data class OpenAIStreamChoice(
    val index: Int,
    val delta: OpenAIDelta,
    @SerialName("finish_reason") val finishReason: String? = null
)

/**
 * OpenAI Delta for streaming
 */
@Serializable
internal data class OpenAIDelta(
    val role: String? = null,
    val content: String? = null,
    @SerialName("tool_calls") val toolCalls: List<OpenAIToolCallDelta>? = null
)

/**
 * OpenAI Tool Call Delta for streaming
 */
@Serializable
internal data class OpenAIToolCallDelta(
    val index: Int? = null,
    val id: String? = null,
    val type: String? = null,
    val function: OpenAIFunctionCallDelta? = null
)

/**
 * OpenAI Function Call Delta for streaming
 */
@Serializable
internal data class OpenAIFunctionCallDelta(
    val name: String? = null,
    val arguments: String? = null
)