package ai.koog.prompt.executor.ollama.client

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.ModerationCategory
import ai.koog.prompt.dsl.ModerationCategoryResult
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.executor.clients.LLMEmbeddingProvider
import ai.koog.prompt.executor.ollama.client.dto.OllamaModelsListResponseDTO
import ai.koog.prompt.executor.ollama.client.dto.OllamaPullModelRequestDTO
import ai.koog.prompt.executor.ollama.client.dto.OllamaPullModelResponseDTO
import ai.koog.prompt.executor.ollama.client.dto.OllamaShowModelRequestDTO
import ai.koog.prompt.executor.ollama.client.dto.OllamaShowModelResponseDTO
import ai.koog.prompt.executor.ollama.client.dto.toOllamaModelCard
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.engine.HttpClientEngineFactory
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.request.get
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsChannel
import io.ktor.http.ContentType
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import io.ktor.utils.io.readUTF8Line
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.datetime.Clock
import kotlinx.serialization.json.Json

/**
 * Client for interacting with the Ollama API with comprehensive model support.
 *
 * @param baseUrl The base URL of the Ollama server. Defaults to "http://localhost:11434".
 * @param baseClient The underlying HTTP client used for making requests.
 * @param timeoutConfig Configuration for connection, request, and socket timeouts.
 * @param clock Clock instance used for tracking response metadata timestamps.
 * Implements:
 * - LLMClient for executing prompts and streaming responses.
 * - LLMEmbeddingProvider for generating embeddings from input text.
 */
public class OllamaClient(
    public val baseUrl: String = "http://localhost:11434",
    baseClient: HttpClient = HttpClient(engineFactoryProvider()),
    timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig(),
    private val clock: Clock = Clock.System
) : LLMClient, LLMEmbeddingProvider {

    private companion object {
        private val logger = KotlinLogging.logger { }

        private const val DEFAULT_MESSAGE_PATH = "api/chat"
        private const val DEFAULT_EMBEDDINGS_PATH = "api/embeddings"
        private const val DEFAULT_LIST_MODELS_PATH = "api/tags"
        private const val DEFAULT_SHOW_MODEL_PATH = "api/show"
        private const val DEFAULT_PULL_MODEL_PATH = "api/pull"

        private val moderationCategoriesMapping: Map<String, List<ModerationCategory>> = mapOf(
            // Violent crimes: unlawful violence towards people and animals
            "S1" to listOf(ModerationCategory.IllicitViolent, ModerationCategory.Violence),

            // Non-violent crimes: fraud, drugs, weapons, hacking, etc.
            "S2" to listOf(ModerationCategory.Illicit),

            // Sex-related crimes: trafficking, harassment, prostitution
            "S3" to listOf(ModerationCategory.IllicitViolent, ModerationCategory.Sexual),

            // Child sexual exploitation
            "S4" to listOf(ModerationCategory.SexualMinors),

            // Defamation (unique)
            "S5" to listOf(ModerationCategory.Defamation),

            // Specialized advice (unique)
            "S6" to listOf(ModerationCategory.SpecializedAdvice),

            // Privacy violations (unique)
            "S7" to listOf(ModerationCategory.Privacy),

            // Intellectual property violations (unique)
            "S8" to listOf(ModerationCategory.IntellectualProperty),

            // Indiscriminate weapons (e.g., nukes, bioweapons)
            "S9" to listOf(ModerationCategory.IllicitViolent),

            // Hate speech (demeaning protected groups)
            "S10" to listOf(ModerationCategory.Hate),

            // Suicide and self-harm
            "S11" to listOf(ModerationCategory.SelfHarm),

            // Sexual content (adult erotica)
            "S12" to listOf(ModerationCategory.Sexual),

            // Election misinformation (unique)
            "S13" to listOf(ModerationCategory.ElectionsMisinformation)
        )

        private val possibleModerationCategories = moderationCategoriesMapping.values.flatten().distinct()
    }


    private val client = baseClient.config {
        defaultRequest {
            url(baseUrl)
            contentType(ContentType.Application.Json)
        }
        install(ContentNegotiation) {
            json(ollamaJson)
        }
        install(HttpTimeout) {
            requestTimeoutMillis = timeoutConfig.requestTimeoutMillis
            connectTimeoutMillis = timeoutConfig.connectTimeoutMillis
            socketTimeoutMillis = timeoutConfig.socketTimeoutMillis
        }
    }
    
    
    private fun convertOllamaResponseToMessages(response: OllamaChatResponse): List<Message.Response> {
        val messages = mutableListOf<Message.Response>()
        
        val metaInfo = ResponseMetaInfo.create(
            clock = clock,
            totalTokensCount = response.evalCount?.let { it + (response.promptEvalCount ?: 0) },
            inputTokensCount = response.promptEvalCount,
            outputTokensCount = response.evalCount
        )
        
        response.message?.let { message ->
            when (message.role) {
                "assistant" -> {
                    // Add text content if present
                    if (!message.content.isNullOrEmpty()) {
                        messages.add(
                            Message.Assistant(
                                content = message.content,
                                metaInfo = metaInfo
                            )
                        )
                    }
                    
                    // Add tool calls if present
                    message.toolCalls?.forEach { toolCall ->
                        messages.add(
                            Message.Tool.Call(
                                id = "ollama_tool_${toolCall.function.name}_${clock.now().toEpochMilliseconds()}",
                                tool = toolCall.function.name,
                                content = toolCall.function.arguments.toString(),
                                metaInfo = metaInfo
                            )
                        )
                    }
                }
                else -> {
                    // Handle other message types if needed
                }
            }
        }
        
        return messages.ifEmpty {
            listOf(
                Message.Assistant(
                    content = "",
                    metaInfo = metaInfo
                )
            )
        }
    }
    

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        require(model.provider == LLMProvider.Ollama) { "Model not supported by Ollama" }

        val chatRequest = HarmonyOllamaDownsampler.downsample(prompt)

        val response = client.post(DEFAULT_MESSAGE_PATH) {
            setBody(chatRequest)
        }

        if (response.status.isSuccess()) {
            val chatResponse = response.body<OllamaChatResponse>()
            return convertOllamaResponseToMessages(chatResponse)
        } else {
            val errorResponse = response.body<OllamaErrorResponse>()
            logger.error { "Ollama error: ${errorResponse.error}" }
            throw RuntimeException("Ollama API error: ${errorResponse.error}")
        }
    }


    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel
    ): Flow<String> = flow {
        require(model.provider == LLMProvider.Ollama) { "Model not supported by Ollama" }

        val chatRequest = HarmonyOllamaDownsampler.downsample(prompt).copy(stream = true)

        val response = client.post(DEFAULT_MESSAGE_PATH) {
            setBody(chatRequest)
        }

        val channel = response.bodyAsChannel()

        while (!channel.isClosedForRead) {
            val line = channel.readUTF8Line() ?: break
            if (line.isBlank()) continue

            try {
                val chunk = ollamaJson.decodeFromString<OllamaChatResponse>(line)
                chunk.message?.content?.let { content ->
                    if (content.isNotEmpty()) {
                        emit(content)
                    }
                }
            } catch (_: Exception) {
                // Skip malformed JSON lines
                continue
            }
        }
    }

    /**
     * Embeds the given text using the Ollama model.
     *
     * @param text The text to embed.
     * @param model The model to use for embedding. Must have the Embed capability.
     * @return A vector representation of the text.
     * @throws IllegalArgumentException if the model does not have the Embed capability.
     */
    override suspend fun embed(text: String, model: LLModel): List<Double> {
        require(model.provider == LLMProvider.Ollama) { "Model not supported by Ollama" }

        if (!model.capabilities.contains(LLMCapability.Embed)) {
            throw IllegalArgumentException("Model ${model.id} does not have the Embed capability")
        }

        val response = client.post(DEFAULT_EMBEDDINGS_PATH) {
            setBody(OllamaEmbeddingRequest(model = model.id, prompt = text))
        }

        val embeddingResponse = response.body<OllamaEmbeddingResponse>()
        return embeddingResponse.embedding
    }

    /**
     * Returns the model cards for all the available models on the server.
     */
    public suspend fun getModels(): List<OllamaModelCard> {
        return try {
            val listModelsResponse = listModels()

            val modelCards = listModelsResponse.models.map { model ->
                showModel(model.name).toOllamaModelCard(model.name, model.size)
            }

            logger.info { "Loaded ${modelCards.size} Ollama model cards" }
            modelCards
        } catch (e: Exception) {
            logger.error(e) { "Failed to fetch model cards from Ollama" }
            throw e
        }
    }

    /**
     * Returns a model card by its model name, on null if no such model exists on the server.
     * @param name the name of the model to get the model card for
     * @param pullIfMissing true if you want to pull the model from the Ollama registry, false otherwise
     */
    public suspend fun getModelOrNull(name: String, pullIfMissing: Boolean = false): OllamaModelCard? {
        var modelCard = loadModelCardOrNull(name)

        if (modelCard == null && pullIfMissing) {
            pullModel(name)
            modelCard = loadModelCardOrNull(name)
        }

        return modelCard
    }

    public override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult {
        if (!model.capabilities.contains(LLMCapability.Moderation)) {
            throw IllegalArgumentException("Model ${model.id} does not support moderation")
        }

        require(prompt.messages.isNotEmpty()) {
            "Can't moderate an empty prompt"
        }

        val responses = execute(prompt, model)

        check(responses.size == 1) { "Moderation model from Ollama must return a single response" }
        val singleResponse = responses.single()
        check(singleResponse is Message.Assistant) {
            "Moderation model from Ollama must return an assistant message" +
                " (actual response: ${singleResponse::class.simpleName})"
        }
        val contentLines = singleResponse.content.lines()
        val moderationResult = contentLines.first()
        val hazardCategories = singleResponse.content.removePrefix(moderationResult)

        return ModerationResult(
            isHarmful = parseModerationResult(moderationResult),
            categories = parseHazardCategories(hazardCategories),
        )
    }

    private fun parseModerationResult(result: String): Boolean {
        return when (result) {
            "safe" -> false
            "unsafe" -> true
            else -> throw IllegalStateException("Unknown moderation result: $result")
        }
    }

    private fun parseHazardCategories(commentWithHazardCodes: String): Map<ModerationCategory, ModerationCategoryResult> {
        return buildMap {
            commentWithHazardCodes.split(",", "\n", ";", ".", "-", "+", " ").forEach { hazardCode ->
                moderationCategoriesMapping[hazardCode]?.let { categories ->
                    categories.forEach { category ->
                        put(category, ModerationCategoryResult(true))
                    }
                }
            }

            possibleModerationCategories.forEach { category ->
                if (category !in this) {
                    put(category, ModerationCategoryResult(false))
                }
            }
        }
    }

    private suspend fun loadModelCardOrNull(name: String): OllamaModelCard? {
        return try {
            val listModelsResponse = listModels()

            val modelInfo = listModelsResponse.models.firstOrNull { it.name.isSameModelAs(name) } ?: return null

            val modelCard = showModel(modelInfo.name).toOllamaModelCard(modelInfo.name, modelInfo.size)

            logger.info { "Loaded Ollama model card for $name" }
            modelCard
        } catch (e: Exception) {
            logger.error(e) { "Failed to fetch model card from Ollama" }
            throw e
        }
    }

    private suspend fun listModels(): OllamaModelsListResponseDTO {
        return client.get(DEFAULT_LIST_MODELS_PATH).body<OllamaModelsListResponseDTO>()
    }

    private suspend fun showModel(name: String): OllamaShowModelResponseDTO {
        return client.post(DEFAULT_SHOW_MODEL_PATH) {
            setBody(OllamaShowModelRequestDTO(name = name))
        }.body<OllamaShowModelResponseDTO>()
    }

    private suspend fun pullModel(name: String) {
        try {
            val response = client.post(DEFAULT_PULL_MODEL_PATH) {
                setBody(OllamaPullModelRequestDTO(name = name, stream = false))
            }.body<OllamaPullModelResponseDTO>()

            if ("success" !in response.status) error("Failed to pull model: '$name'")

            logger.info { "Pulled model '$name'" }
        } catch (e: Exception) {
            logger.error(e) { "Failed to pull model '$name'" }
            throw e
        }
    }
}

internal expect fun engineFactoryProvider(): HttpClientEngineFactory<*>
