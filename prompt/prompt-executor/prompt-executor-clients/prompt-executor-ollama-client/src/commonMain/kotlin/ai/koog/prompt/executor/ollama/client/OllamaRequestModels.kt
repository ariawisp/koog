package ai.koog.prompt.executor.ollama.client

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * Ollama Generate Request (for /api/generate endpoint)
 * Used for basic text generation without conversational context
 */
@Serializable
internal data class OllamaGenerateRequest(
    val model: String,
    val prompt: String,
    val system: String? = null,
    val template: String? = null,
    val context: List<Int>? = null,
    val stream: Boolean? = null,
    val raw: Boolean? = null,
    val format: @Contextual JsonElement? = null,
    val options: OllamaOptions? = null,
    @SerialName("keep_alive") val keepAlive: String? = null
)

/**
 * Ollama Chat Request (for /api/chat endpoint)
 * Used for conversational interactions with message history
 */
@Serializable
internal data class OllamaChatRequest(
    val model: String,
    val messages: List<OllamaMessage>,
    val tools: List<OllamaTool>? = null,
    val format: @Contextual JsonElement? = null,
    val options: OllamaOptions? = null,
    val stream: Boolean? = null,
    @SerialName("keep_alive") val keepAlive: String? = null
)

/**
 * Ollama Message
 * Supports text content and images (base64 encoded)
 */
@Serializable(with = OllamaMessageSerializer::class)
internal data class OllamaMessage(
    val role: String,
    val content: String,
    val images: List<String>? = null,
    @SerialName("tool_calls") val toolCalls: List<OllamaToolCall>? = null
)

/**
 * Ollama Tool Call
 */
@Serializable
internal data class OllamaToolCall(
    val function: OllamaFunctionCall
)

/**
 * Ollama Function Call
 */
@Serializable
internal data class OllamaFunctionCall(
    val name: String,
    val arguments: @Contextual JsonObject
)

/**
 * Ollama Tool Definition
 */
@Serializable
internal data class OllamaTool(
    val type: String = "function",
    val function: OllamaFunction
)

/**
 * Ollama Function Definition
 */
@Serializable
internal data class OllamaFunction(
    val name: String,
    val description: String,
    val parameters: @Contextual JsonObject
)

/**
 * Ollama Generation Options
 * Contains all the generation parameters that can be set
 */
@Serializable
internal data class OllamaOptions(
    val temperature: Double? = null,
    @SerialName("top_k") val topK: Int? = null,
    @SerialName("top_p") val topP: Double? = null,
    @SerialName("repeat_last_n") val repeatLastN: Int? = null,
    @SerialName("repeat_penalty") val repeatPenalty: Double? = null,
    val seed: Int? = null,
    @SerialName("num_predict") val numPredict: Int? = null,
    @SerialName("num_ctx") val numCtx: Int? = null,
    @SerialName("num_batch") val numBatch: Int? = null,
    @SerialName("num_gqa") val numGqa: Int? = null,
    @SerialName("num_gpu") val numGpu: Int? = null,
    @SerialName("main_gpu") val mainGpu: Int? = null,
    @SerialName("low_vram") val lowVram: Boolean? = null,
    @SerialName("f16_kv") val f16Kv: Boolean? = null,
    @SerialName("logits_all") val logitsAll: Boolean? = null,
    @SerialName("vocab_only") val vocabOnly: Boolean? = null,
    @SerialName("use_mmap") val useMmap: Boolean? = null,
    @SerialName("use_mlock") val useMlock: Boolean? = null,
    @SerialName("embedding_only") val embeddingOnly: Boolean? = null,
    @SerialName("rope_frequency_base") val ropeFrequencyBase: Double? = null,
    @SerialName("rope_frequency_scale") val ropeFrequencyScale: Double? = null,
    @SerialName("num_thread") val numThread: Int? = null,
    val stop: List<String>? = null
)

/**
 * Ollama Chat Response
 */
@Serializable
internal data class OllamaChatResponse(
    val model: String,
    @SerialName("created_at") val createdAt: String,
    val message: OllamaMessage? = null,
    val done: Boolean,
    @SerialName("done_reason") val doneReason: String? = null,
    @SerialName("total_duration") val totalDuration: Long? = null,
    @SerialName("load_duration") val loadDuration: Long? = null,
    @SerialName("prompt_eval_count") val promptEvalCount: Int? = null,
    @SerialName("prompt_eval_duration") val promptEvalDuration: Long? = null,
    @SerialName("eval_count") val evalCount: Int? = null,
    @SerialName("eval_duration") val evalDuration: Long? = null
)

/**
 * Ollama Generate Response
 */
@Serializable
internal data class OllamaGenerateResponse(
    val model: String,
    @SerialName("created_at") val createdAt: String,
    val response: String,
    val done: Boolean,
    @SerialName("done_reason") val doneReason: String? = null,
    val context: List<Int>? = null,
    @SerialName("total_duration") val totalDuration: Long? = null,
    @SerialName("load_duration") val loadDuration: Long? = null,
    @SerialName("prompt_eval_count") val promptEvalCount: Int? = null,
    @SerialName("prompt_eval_duration") val promptEvalDuration: Long? = null,
    @SerialName("eval_count") val evalCount: Int? = null,
    @SerialName("eval_duration") val evalDuration: Long? = null
)

/**
 * Ollama Error Response
 */
@Serializable
internal data class OllamaErrorResponse(
    val error: String
)

/**
 * Embedding Request for /api/embeddings endpoint
 */
@Serializable
internal data class OllamaEmbeddingRequest(
    val model: String,
    val prompt: String
)

/**
 * Embedding Response from /api/embeddings endpoint
 */
@Serializable
internal data class OllamaEmbeddingResponse(
    val embedding: List<Double>,
    @SerialName("model") val modelId: String? = null
)