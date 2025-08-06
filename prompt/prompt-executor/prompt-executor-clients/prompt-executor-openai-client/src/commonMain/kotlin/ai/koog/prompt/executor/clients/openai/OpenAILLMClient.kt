package ai.koog.prompt.executor.clients.openai

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.utils.SuitableForIO
import ai.koog.prompt.dsl.ModerationCategory
import ai.koog.prompt.dsl.ModerationCategoryResult
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.executor.clients.LLMEmbeddingProvider
import ai.koog.prompt.harmony.*
// LLMChoice removed - use Message.Response directly
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Attachment
import ai.koog.prompt.message.AttachmentContent
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.plugins.sse.SSE
import io.ktor.client.plugins.sse.SSEClientException
import io.ktor.client.plugins.sse.sse
import io.ktor.client.request.accept
import io.ktor.client.request.header
import io.ktor.client.request.headers
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.client.statement.readRawBytes
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.contentType
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock
import kotlinx.serialization.json.*
import kotlin.io.encoding.ExperimentalEncodingApi

/**
 * Represents the settings for configuring an OpenAI client.
 *
 * @property baseUrl The base URL of the OpenAI API. Defaults to "https://api.openai.com".
 * @property timeoutConfig Configuration for connection timeouts, including request, connect, and socket timeouts.
 * @property chatCompletionsPath The path of the OpenAI Chat Completions API. Defaults to "v1/chat/completions".
 * @property embeddingsPath The path of the OpenAI Embeddings API. Defaults to "v1/embeddings".
 * @property moderationsPath The path of the OpenAI Moderations API. Defaults to "v1/moderations".
 */
public class OpenAIClientSettings(
    public val baseUrl: String = "https://api.openai.com",
    public val timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig(),
    public val chatCompletionsPath: String = "v1/chat/completions",
    public val embeddingsPath: String = "v1/embeddings",
    public val moderationsPath: String = "v1/moderations",
)

/**
 * Implementation of [LLMClient] for OpenAI API.
 * Uses Ktor HttpClient to communicate with the OpenAI API.
 *
 * @param apiKey The API key for the OpenAI API
 * @param settings The base URL and timeouts for the OpenAI API, defaults to "https://api.openai.com" and 900 s
 * @param clock Clock instance used for tracking response metadata timestamps.
 */
public open class OpenAILLMClient(
    private val apiKey: String,
    private val settings: OpenAIClientSettings = OpenAIClientSettings(),
    baseClient: HttpClient = HttpClient(),
    private val clock: Clock = Clock.System,
) : LLMEmbeddingProvider, LLMClient {

    private companion object {
        private val logger = KotlinLogging.logger { }
    }

    // Use the properly configured OpenAI JSON
    private val json = openAIJson

    private val httpClient = baseClient.config {
        defaultRequest {
            url(settings.baseUrl)
            contentType(ContentType.Application.Json)
            header("Authorization", "Bearer $apiKey")
        }
        install(SSE)
        install(ContentNegotiation) {
            json(json)
        }
        install(HttpTimeout) {
            requestTimeoutMillis = settings.timeoutConfig.requestTimeoutMillis // Increase timeout to 60 seconds
            connectTimeoutMillis = settings.timeoutConfig.connectTimeoutMillis
            socketTimeoutMillis = settings.timeoutConfig.socketTimeoutMillis
        }
    }

    override suspend fun execute(prompt: Prompt, model: LLModel, tools: List<ToolDescriptor>): List<Message.Response> {
        // Prompt IS HarmonyCore - work with it directly!
        val promptWithTools = if (tools.isNotEmpty()) {
            prompt.copy(
                developerContext = prompt.developerContext.copy(
                    tools = HarmonyConverter.fromToolDescriptors(tools)
                )
            )
        } else prompt
        
        // Downsample to OpenAI format using pure function
        val openAIRequest = HarmonyOpenAIDownsampler.downsample(promptWithTools, model)
        
        return processOpenAIResponse(executeOpenAIRequest(openAIRequest)).flatten()
    }

    override fun executeStreaming(prompt: Prompt, model: LLModel): Flow<String> = flow {
        logger.debug { "Executing streaming prompt: $prompt with model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }

        // Prompt IS HarmonyCore
        val openAIRequest = HarmonyOpenAIDownsampler.downsample(prompt, model).copy(stream = true)

        try {
            httpClient.sse(
                urlString = settings.chatCompletionsPath,
                request = {
                    method = HttpMethod.Post
                    accept(ContentType.Text.EventStream)
                    headers {
                        append(HttpHeaders.CacheControl, "no-cache")
                        append(HttpHeaders.Connection, "keep-alive")
                    }
                    setBody(openAIRequest)
                }
            ) {
                incoming.collect { event ->
                    event
                        .takeIf { it.data != "[DONE]" }
                        ?.data?.trim()?.let { data ->
                            try {
                                val jsonElement = json.parseToJsonElement(data)
                                val openAIResponse = json.decodeFromJsonElement<OpenAIChatStreamChunk>(jsonElement)
                                // Direct streaming without UnifiedResponse conversion
                                openAIResponse.choices.forEach { choice ->
                                    choice.delta.content?.let { content ->
                                        emit(content)
                                    }
                                }
                            } catch (e: Exception) {
                                // Ignore parse errors in streaming
                                logger.debug { "Error parsing stream chunk: $e" }
                            }
                        }
                }
            }
        } catch (e: SSEClientException) {
            e.response?.let { response ->
                val body = response.readRawBytes().decodeToString()
                logger.error(e) { "Error from OpenAI API: ${response.status}: ${e.message}.\nBody:\n$body" }
                error("Error from OpenAI API: ${response.status}: ${e.message}")
            }
        } catch (e: Exception) {
            logger.error { "Exception during streaming: $e" }
            error(e.message ?: "Unknown error during streaming")
        }
    }

    override suspend fun executeMultipleChoices(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        // Prompt IS HarmonyCore now
        val promptWithTools = if (tools.isNotEmpty()) {
            prompt.copy(
                developerContext = prompt.developerContext.copy(
                    tools = HarmonyConverter.fromToolDescriptors(tools)
                )
            )
        } else prompt
        val openAIRequest = HarmonyOpenAIDownsampler.downsample(promptWithTools, model)
        
        return processOpenAIResponse(executeOpenAIRequest(openAIRequest)).flatten()
    }

    /**
     * Embeds the given text using the OpenAI embeddings API.
     *
     * @param text The text to embed.
     * @param model The model to use for embedding. Must have the Embed capability.
     * @return A list of floating-point values representing the embedding.
     * @throws IllegalArgumentException if the model does not have the Embed capability.
     */
    override suspend fun embed(text: String, model: LLModel): List<Double> {
        require(model.capabilities.contains(LLMCapability.Embed)) {
            "Model ${model.id} does not have the Embed capability"
        }
        logger.debug { "Embedding text with model: ${model.id}" }

        val request = OpenAIEmbeddingRequest(
            model = model.id,
            input = text
        )

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(settings.embeddingsPath) {
                setBody(request)
            }

            if (response.status.isSuccess()) {
                val openAIResponse = response.body<OpenAIEmbeddingResponse>()
                if (openAIResponse.data.isNotEmpty()) {
                    openAIResponse.data.first().embedding
                } else {
                    logger.error { "Empty data in OpenAI embedding response" }
                    error("Empty data in OpenAI embedding response")
                }
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from OpenAI API: ${response.status}: $errorBody" }
                error("Error from OpenAI API: ${response.status}: $errorBody")
            }
        }
    }

    /**
     * Moderates text and image content based on the provided model's capabilities.
     *
     * @param prompt The prompt containing text messages and optional attachments to be moderated.
     * @param model The language model to use for moderation. Must have the `Moderation` capability.
     * @return The moderation result, including flagged content, categories, scores, and associated metadata.
     * @throws IllegalArgumentException If the specified model does not support moderation.
     */
    public override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult {
        logger.debug { "Moderating text and image content with model: $model" }

        if (!model.capabilities.contains(LLMCapability.Moderation)) {
            throw IllegalArgumentException("Model ${model.id} does not support moderation")
        }

        require(prompt.conversation.messages.isNotEmpty()) {
            "Can't moderate an empty prompt"
        }

        // Convert messages to moderation input format
        val input = buildJsonArray {
            prompt.conversation.messages.forEach { message ->
                // Extract text content from HarmonyMessage
                val textContent = message.content
                    .filterIsInstance<HarmonyContent.Text>()
                    .joinToString(" ") { it.text }
                
                if (textContent.isNotEmpty()) {
                    add(textContent)
                }
            }
        }.let { array ->
            // If all elements are JsonPrimitives (strings), merge them into a single string
            if (array.all { it is JsonPrimitive && it.isString }) {
                JsonPrimitive(array.joinToString("\n\n") { (it as JsonPrimitive).content })
            } else {
                array
            }
        }

        val request = OpenAIModerationRequest(
            input = input,
            model = model.id
        )

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(settings.moderationsPath) {
                setBody(request)
            }

            if (response.status.isSuccess()) {
                val openAIResponse = response.body<OpenAIModerationResponse>()
                if (openAIResponse.results.isNotEmpty()) {
                    val result = openAIResponse.results.first()

                    // Convert OpenAI categories to a map
                    val categories = mapOf(
                        ModerationCategory.Harassment to result.categories.harassment,
                        ModerationCategory.HarassmentThreatening to result.categories.harassmentThreatening,
                        ModerationCategory.Hate to result.categories.hate,
                        ModerationCategory.HateThreatening to result.categories.hateThreatening,
                        ModerationCategory.Sexual to result.categories.sexual,
                        ModerationCategory.SexualMinors to result.categories.sexualMinors,
                        ModerationCategory.Violence to result.categories.violence,
                        ModerationCategory.ViolenceGraphic to result.categories.violenceGraphic,
                        ModerationCategory.SelfHarm to result.categories.selfHarm,
                        ModerationCategory.SelfHarmIntent to result.categories.selfHarmIntent,
                        ModerationCategory.SelfHarmInstructions to result.categories.selfHarmInstructions,
                        ModerationCategory.Illicit to (result.categories.illicit ?: false),
                        ModerationCategory.IllicitViolent to (result.categories.illicitViolent ?: false)
                    )

                    // Convert OpenAI category scores to a map
                    val categoryScores = mapOf(
                        ModerationCategory.Harassment to result.categoryScores.harassment,
                        ModerationCategory.HarassmentThreatening to result.categoryScores.harassmentThreatening,
                        ModerationCategory.Hate to result.categoryScores.hate,
                        ModerationCategory.HateThreatening to result.categoryScores.hateThreatening,
                        ModerationCategory.Sexual to result.categoryScores.sexual,
                        ModerationCategory.SexualMinors to result.categoryScores.sexualMinors,
                        ModerationCategory.Violence to result.categoryScores.violence,
                        ModerationCategory.ViolenceGraphic to result.categoryScores.violenceGraphic,
                        ModerationCategory.SelfHarm to result.categoryScores.selfHarm,
                        ModerationCategory.SelfHarmIntent to result.categoryScores.selfHarmIntent,
                        ModerationCategory.SelfHarmInstructions to result.categoryScores.selfHarmInstructions,
                        ModerationCategory.Illicit to (result.categoryScores.illicit ?: 0.0),
                        ModerationCategory.IllicitViolent to (result.categoryScores.illicitViolent ?: 0.0)
                    )

                    // Convert category applied input types if available
                    val categoryAppliedInputTypes = result.categoryAppliedInputTypes?.let { appliedTypes ->
                        buildMap {
                            appliedTypes.harassment?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.Harassment, it) }
                            appliedTypes.harassmentThreatening?.map {
                                ModerationResult.InputType.valueOf(it.uppercase())
                            }
                                ?.let { put(ModerationCategory.HarassmentThreatening, it) }
                            appliedTypes.hate?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.Hate, it) }
                            appliedTypes.hateThreatening?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.HateThreatening, it) }
                            appliedTypes.sexual?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.Sexual, it) }
                            appliedTypes.sexualMinors?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.SexualMinors, it) }
                            appliedTypes.violence?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.Violence, it) }
                            appliedTypes.violenceGraphic?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.ViolenceGraphic, it) }
                            appliedTypes.selfHarm?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.SelfHarm, it) }
                            appliedTypes.selfHarmIntent?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.SelfHarmIntent, it) }
                            appliedTypes.selfHarmInstructions?.map {
                                ModerationResult.InputType.valueOf(it.uppercase())
                            }
                                ?.let { put(ModerationCategory.SelfHarmInstructions, it) }
                            appliedTypes.illicit?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.Illicit, it) }
                            appliedTypes.illicitViolent?.map { ModerationResult.InputType.valueOf(it.uppercase()) }
                                ?.let { put(ModerationCategory.IllicitViolent, it) }
                        }
                    } ?: emptyMap()

                    ModerationResult(
                        isHarmful = result.flagged,
                        categories = categories.mapValues { (category, detected) ->
                            ModerationCategoryResult(
                                detected,
                                categoryScores[category],
                                categoryAppliedInputTypes[category] ?: emptyList()
                            )
                        }
                    )
                } else {
                    logger.error { "Empty results in OpenAI moderation response" }
                    error("Empty results in OpenAI moderation response")
                }
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from OpenAI API: ${response.status}: $errorBody" }
                error("Error from OpenAI API: ${response.status}: $errorBody")
            }
        }
    }

    /**
     * Execute OpenAI request directly without UnifiedModel intermediate layer.
     * This is the new Harmony-first approach.
     */
    private suspend fun executeOpenAIRequest(openAIRequest: OpenAIChatRequest): OpenAIChatResponse {
        logger.debug { "Executing OpenAI request: $openAIRequest" }
        
        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(settings.chatCompletionsPath) {
                setBody(openAIRequest)
            }

            if (response.status.isSuccess()) {
                val jsonResponse = json.parseToJsonElement(response.bodyAsText())
                json.decodeFromJsonElement<OpenAIChatResponse>(jsonResponse)
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from OpenAI API: ${response.status}: $errorBody" }
                error("Error from OpenAI API: ${response.status}: $errorBody")
            }
        }
    }


    /**
     * Process OpenAI response directly without UnifiedModel conversion.
     * This is the new Harmony-first approach.
     */
    private fun processOpenAIResponse(response: OpenAIChatResponse): List<List<Message.Response>> {
        if (response.choices.isEmpty()) {
            logger.error { "Empty choices in OpenAI response" }
            error("Empty choices in OpenAI response")
        }

        // Extract token count from the response
        val totalTokensCount = response.usage?.totalTokens
        val inputTokensCount = response.usage?.promptTokens
        val outputTokensCount = response.usage?.completionTokens

        val metaInfo = ResponseMetaInfo.create(
            clock,
            totalTokensCount = totalTokensCount,
            inputTokensCount = inputTokensCount,
            outputTokensCount = outputTokensCount
        )

        return response.choices.map { processOpenAIChoice(it, metaInfo) }
    }

    /**
     * Process individual OpenAI choice directly.
     */
    private fun processOpenAIChoice(choice: OpenAIChoice, metaInfo: ResponseMetaInfo): List<Message.Response> {
        val message = choice.message
        val toolCalls = message.toolCalls
        return when {
            toolCalls != null && toolCalls.isNotEmpty() -> {
                // Handle tool calls
                toolCalls.map { toolCall ->
                    Message.Tool.Call(
                        id = toolCall.id,
                        tool = toolCall.function.name,
                        content = toolCall.function.arguments,
                        metaInfo = metaInfo
                    )
                }
            }
            else -> {
                // Handle regular assistant response
                val content = when (val c = message.content) {
                    is JsonPrimitive -> c.content
                    null -> ""
                    else -> c.toString()
                }
                listOf(
                    Message.Assistant(
                        content = content,
                        finishReason = choice.finishReason,
                        metaInfo = metaInfo
                    )
                )
            }
        }
    }

}
