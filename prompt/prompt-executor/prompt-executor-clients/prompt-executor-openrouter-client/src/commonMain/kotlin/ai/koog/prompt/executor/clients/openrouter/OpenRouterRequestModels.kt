package ai.koog.prompt.executor.clients.openrouter

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * OpenRouter Chat Completion Request
 * Uses proper kotlinx.serialization with snake_case naming
 * OpenRouter API is compatible with OpenAI format with some extensions
 */
@Serializable
internal data class OpenRouterChatRequest(
    val model: String,
    val messages: List<OpenRouterMessage>,
    @SerialName("max_tokens") val maxTokens: Int? = null,
    @SerialName("min_tokens") val minTokens: Int? = null,
    val temperature: Double? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("frequency_penalty") val frequencyPenalty: Double? = null,
    @SerialName("presence_penalty") val presencePenalty: Double? = null,
    @SerialName("repetition_penalty") val repetitionPenalty: Double? = null,
    val n: Int? = null,
    val stop: List<String>? = null,
    val stream: Boolean? = null,
    @SerialName("stream_options") val streamOptions: OpenRouterStreamOptions? = null,
    val tools: List<OpenRouterTool>? = null,
    @SerialName("tool_choice") val toolChoice: @Contextual JsonElement? = null,
    @SerialName("parallel_tool_calls") val parallelToolCalls: Boolean? = null,
    @SerialName("response_format") val responseFormat: OpenRouterResponseFormat? = null,
    val logprobs: Boolean? = null,
    @SerialName("top_logprobs") val topLogprobs: Int? = null,
    val seed: Int? = null,
    val user: String? = null,
    // OpenRouter-specific fields
    @SerialName("provider") val provider: OpenRouterProvider? = null,
    @SerialName("route") val route: String? = null,
    @SerialName("transforms") val transforms: List<String>? = null,
    @SerialName("models") val models: List<String>? = null
)

/**
 * OpenRouter Message - will use custom serializer for content handling
 */
@Serializable(with = OpenRouterMessageSerializer::class)
internal data class OpenRouterMessage(
    val role: String,
    val content: @Contextual JsonElement? = null,
    @SerialName("tool_calls") val toolCalls: List<OpenRouterToolCall>? = null,
    @SerialName("tool_call_id") val toolCallId: String? = null,
    val name: String? = null
)

/**
 * OpenRouter Tool Call
 */
@Serializable
internal data class OpenRouterToolCall(
    val id: String,
    val type: String = "function",
    val function: OpenRouterFunctionCall
)

/**
 * OpenRouter Function Call
 */
@Serializable
internal data class OpenRouterFunctionCall(
    val name: String,
    val arguments: String
)

/**
 * OpenRouter Tool Definition
 */
@Serializable
internal data class OpenRouterTool(
    val type: String = "function",
    val function: OpenRouterFunction
)

/**
 * OpenRouter Function Definition
 */
@Serializable
internal data class OpenRouterFunction(
    val name: String,
    val description: String? = null,
    val parameters: JsonObject
)

/**
 * OpenRouter Stream Options
 */
@Serializable
internal data class OpenRouterStreamOptions(
    @SerialName("include_usage") val includeUsage: Boolean? = null
)

/**
 * OpenRouter Provider Selection
 */
@Serializable
internal data class OpenRouterProvider(
    val order: List<String>? = null,
    @SerialName("allow_fallbacks") val allowFallbacks: Boolean? = null,
    @SerialName("data_collection") val dataCollection: String? = null
)

/**
 * OpenRouter Response Format
 */
@Serializable(with = OpenRouterResponseFormatSerializer::class)
internal sealed class OpenRouterResponseFormat {
    @Serializable
    data class Text(val type: String = "text") : OpenRouterResponseFormat()
    
    @Serializable
    data class JsonObject(val type: String = "json_object") : OpenRouterResponseFormat()
    
    @Serializable
    data class JsonSchema(
        val type: String = "json_schema",
        @SerialName("json_schema") val jsonSchema: OpenRouterJsonSchema
    ) : OpenRouterResponseFormat()
}

/**
 * OpenRouter JSON Schema for structured output
 */
@Serializable
internal data class OpenRouterJsonSchema(
    val name: String,
    val description: String? = null,
    val schema: JsonObject,
    val strict: Boolean? = null
)

/**
 * OpenRouter Chat Completion Response
 */
@Serializable
internal data class OpenRouterChatResponse(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<OpenRouterChoice>,
    val usage: OpenRouterUsage? = null,
    @SerialName("system_fingerprint") val systemFingerprint: String? = null,
    // OpenRouter-specific fields
    @SerialName("provider") val provider: String? = null
)

/**
 * OpenRouter Choice in Response
 */
@Serializable
internal data class OpenRouterChoice(
    val index: Int,
    val message: OpenRouterMessage,
    @SerialName("finish_reason") val finishReason: String? = null,
    val logprobs: JsonObject? = null
)

/**
 * OpenRouter Usage Statistics
 */
@Serializable
internal data class OpenRouterUsage(
    @SerialName("prompt_tokens") val promptTokens: Int,
    @SerialName("completion_tokens") val completionTokens: Int,
    @SerialName("total_tokens") val totalTokens: Int,
    @SerialName("completion_tokens_details") val completionTokensDetails: OpenRouterTokenDetails? = null,
    // OpenRouter-specific fields
    @SerialName("native_tokens_prompt") val nativeTokensPrompt: Int? = null,
    @SerialName("native_tokens_completion") val nativeTokensCompletion: Int? = null,
    @SerialName("native_tokens_total") val nativeTokensTotal: Int? = null
)

/**
 * OpenRouter Token Details
 */
@Serializable
internal data class OpenRouterTokenDetails(
    @SerialName("reasoning_tokens") val reasoningTokens: Int? = null
)

/**
 * OpenRouter Streaming Response Chunk
 */
@Serializable
internal data class OpenRouterChatStreamChunk(
    val id: String,
    val `object`: String,
    val created: Long,
    val model: String,
    val choices: List<OpenRouterStreamChoice>,
    val usage: OpenRouterUsage? = null,
    // OpenRouter-specific fields
    @SerialName("provider") val provider: String? = null
)

/**
 * OpenRouter Stream Choice
 */
@Serializable
internal data class OpenRouterStreamChoice(
    val index: Int,
    val delta: OpenRouterDelta,
    @SerialName("finish_reason") val finishReason: String? = null
)

/**
 * OpenRouter Delta for streaming
 */
@Serializable
internal data class OpenRouterDelta(
    val role: String? = null,
    val content: String? = null,
    @SerialName("tool_calls") val toolCalls: List<OpenRouterToolCallDelta>? = null
)

/**
 * OpenRouter Tool Call Delta for streaming
 */
@Serializable
internal data class OpenRouterToolCallDelta(
    val index: Int? = null,
    val id: String? = null,
    val type: String? = null,
    val function: OpenRouterFunctionCallDelta? = null
)

/**
 * OpenRouter Function Call Delta for streaming
 */
@Serializable
internal data class OpenRouterFunctionCallDelta(
    val name: String? = null,
    val arguments: String? = null
)