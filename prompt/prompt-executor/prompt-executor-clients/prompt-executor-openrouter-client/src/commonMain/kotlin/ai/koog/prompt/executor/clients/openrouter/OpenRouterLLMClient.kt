package ai.koog.prompt.executor.clients.openrouter

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.utils.SuitableForIO
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
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
import kotlinx.serialization.json.JsonElement

/**
 * Configuration settings for connecting to the OpenRouter API.
 *
 * @property baseUrl The base URL of the OpenRouter API. Default is "https://openrouter.ai/api/v1".
 * @property timeoutConfig Configuration for connection timeouts including request, connection, and socket timeouts.
 */
public class OpenRouterClientSettings(
    public val baseUrl: String = "https://openrouter.ai",
    public val timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig()
)

/**
 * Implementation of [LLMClient] for OpenRouter API.
 * OpenRouter is an API that routes requests to multiple LLM providers.
 *
 * @param apiKey The API key for the OpenRouter API
 * @param settings The base URL and timeouts for the OpenRouter API, defaults to "https://openrouter.ai" and 900s
 * @param clock Clock instance used for tracking response metadata timestamps.
 */
public class OpenRouterLLMClient(
    private val apiKey: String,
    private val settings: OpenRouterClientSettings = OpenRouterClientSettings(),
    baseClient: HttpClient = HttpClient(),
    private val clock: Clock = Clock.System
) : LLMClient {

    private companion object {
        private val logger = KotlinLogging.logger { }

        private const val DEFAULT_MESSAGE_PATH = "api/v1/chat/completions"
    }

    private val json = openRouterJson

    private val httpClient = baseClient.config {
        defaultRequest {
            url(settings.baseUrl)
            contentType(ContentType.Application.Json)
            header("Authorization", "Bearer $apiKey")
            // OpenRouter requires HTTP_REFERER header to be set
            header("HTTP-Referer", "https://jetbrains.com")
            // Set custom user agent for OpenRouter
            header("User-Agent", "JetBrains/1.0")
        }
        install(SSE)
        install(ContentNegotiation) {
            json(openRouterJson)
        }
        install(HttpTimeout) {
            requestTimeoutMillis = settings.timeoutConfig.requestTimeoutMillis
            connectTimeoutMillis = settings.timeoutConfig.connectTimeoutMillis
            socketTimeoutMillis = settings.timeoutConfig.socketTimeoutMillis
        }
    }

    override suspend fun execute(prompt: Prompt, model: LLModel, tools: List<ToolDescriptor>): List<Message.Response> {
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        require(model.capabilities.contains(LLMCapability.Tools) || tools.isEmpty()) {
            "Model ${model.id} does not support tools"
        }
        logger.debug { "Executing prompt: $prompt with tools: $tools" }

        val response = getOpenRouterResponse(prompt, model, tools)
        return processOpenRouterResponse(response)
    }

    override fun executeStreaming(prompt: Prompt, model: LLModel): Flow<String> = flow {
        logger.debug { "Executing streaming prompt: $prompt" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }

        val openRouterRequest = HarmonyOpenRouterDownsampler.downsample(prompt).copy(stream = true)
        val requestBody = json.encodeToString(OpenRouterChatRequest.serializer(), openRouterRequest)

        try {
            httpClient.sse(
                urlString = DEFAULT_MESSAGE_PATH,
                request = {
                    method = HttpMethod.Post
                    accept(ContentType.Text.EventStream)
                    headers {
                        append(HttpHeaders.CacheControl, "no-cache")
                        append(HttpHeaders.Connection, "keep-alive")
                    }
                    setBody(requestBody)
                }
            ) {
                incoming.collect { event ->
                    event
                        .takeIf { it.data != "[DONE]" }
                        ?.data?.trim()?.let { data ->
                            try {
                                val streamChunk = json.decodeFromString(OpenRouterChatStreamChunk.serializer(), data)
                                streamChunk.choices.forEach { choice ->
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
                logger.error { "Error from OpenRouter API: ${response.status}: ${e.message}" }
                error("Error from OpenRouter API: ${response.status}: ${e.message}")
            }
        } catch (e: Exception) {
            logger.error { "Exception during streaming: $e" }
            error(e.message ?: "Unknown error during streaming")
        }
    }


    private suspend fun getOpenRouterResponse(prompt: Prompt, model: LLModel, tools: List<ToolDescriptor>): OpenRouterChatResponse {
        logger.debug { "Executing prompt: $prompt with tools: $tools and model: $model" }

        val openRouterRequest = HarmonyOpenRouterDownsampler.downsample(prompt)
        val requestBody = json.encodeToString(OpenRouterChatRequest.serializer(), openRouterRequest)

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(DEFAULT_MESSAGE_PATH) {
                setBody(requestBody)
            }

            if (response.status.isSuccess()) {
                response.body<OpenRouterChatResponse>()
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from OpenRouter API: ${response.status}: $errorBody" }
                error("Error from OpenRouter API: ${response.status}: $errorBody")
            }
        }
    }


    private fun processOpenRouterResponse(response: OpenRouterChatResponse): List<Message.Response> {
        if (response.choices.isEmpty()) {
            logger.error { "Empty choices in OpenRouter response" }
            error("Empty choices in OpenRouter response")
        }

        val choice = response.choices.firstOrNull() 
            ?: throw IllegalStateException("No choice found in OpenRouter response")

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

        val message = choice.message
        return when (message.role) {
            "assistant" -> {
                val toolCalls = message.toolCalls
                if (toolCalls != null && toolCalls.isNotEmpty()) {
                    toolCalls.map { toolCall ->
                        Message.Tool.Call(
                            id = toolCall.id,
                            tool = toolCall.function.name,
                            content = toolCall.function.arguments,
                            metaInfo = metaInfo
                        )
                    }
                } else {
                    val textContent = when (val content = message.content) {
                        is JsonPrimitive -> content.content
                        is JsonObject, is JsonArray -> content.toString()
                        null -> ""
                        else -> content.toString()
                    }
                    
                    listOf(
                        Message.Assistant(
                            content = textContent,
                            finishReason = choice.finishReason,
                            metaInfo = metaInfo
                        )
                    )
                }
            }
            else -> {
                logger.error { "Unexpected response from OpenRouter: no assistant message" }
                error("Unexpected response from OpenRouter: no assistant message")
            }
        }
    }

    /**
     * Executes a moderation action on the given prompt using the specified language model.
     * This method is not supported by the OpenRouter API and will always throw an `UnsupportedOperationException`.
     *
     * @param prompt The [Prompt] object to be moderated, containing the messages and respective context.
     * @param model The [LLModel] to be used for the moderation process.
     * @return This method does not return a valid result as it always throws an exception.
     * @throws UnsupportedOperationException Always thrown because moderation is not supported by the OpenRouter API.
     */
    public override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult {
        logger.warn { "Moderation is not supported by OpenRouter API" }
        throw UnsupportedOperationException("Moderation is not supported by OpenRouter API.")
    }
}
