package ai.koog.prompt.executor.clients.anthropic

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.utils.SuitableForIO
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.harmony.*
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
import kotlinx.serialization.json.*

/**
 * Represents the settings for configuring an Anthropic client, including model mapping, base URL, and API version.
 *
 * @property modelVersionsMap Maps specific `LLModel` instances to their corresponding model version strings.
 * This determines which Anthropic model versions are used for operations.
 * @property baseUrl The base URL for accessing the Anthropic API. Defaults to "https://api.anthropic.com".
 * @property apiVersion The version of the Anthropic API to be used. Defaults to "2023-06-01".
 */
public class AnthropicClientSettings(
    public val modelVersionsMap: Map<LLModel, String> = DEFAULT_ANTHROPIC_MODEL_VERSIONS_MAP,
    public val baseUrl: String = "https://api.anthropic.com",
    public val apiVersion: String = "2023-06-01",
    public val timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig()
)

/**
 * A client implementation for interacting with Anthropic's API using Harmony format.
 *
 * This class supports functionalities for executing text prompts and streaming interactions with the Anthropic API.
 * It leverages Kotlin Coroutines to handle asynchronous operations and provides full support for configuring HTTP
 * requests, including timeout handling and JSON serialization.
 * 
 * The client uses HarmonyAnthropicDownsampler to convert from Harmony's unified format to Anthropic's API format.
 *
 * @constructor Creates an instance of the AnthropicLLMClient.
 * @param apiKey The API key required to authenticate with the Anthropic service.
 * @param settings Configurable settings for the Anthropic client, which include the base URL and other options.
 * @param baseClient An optional custom configuration for the underlying HTTP client, defaulting to a Ktor client.
 * @param clock Clock instance used for tracking response metadata timestamps.
 */
public open class AnthropicLLMClient(
    private val apiKey: String,
    private val settings: AnthropicClientSettings = AnthropicClientSettings(),
    baseClient: HttpClient = HttpClient(),
    private val clock: Clock = Clock.System
) : LLMClient {

    private companion object {
        private val logger = KotlinLogging.logger { }

        private const val DEFAULT_MESSAGE_PATH = "v1/messages"
    }

    private val json = anthropicJson

    private val httpClient = baseClient.config {
        defaultRequest {
            url(settings.baseUrl)
            contentType(ContentType.Application.Json)
            header("x-api-key", apiKey)
            header("anthropic-version", settings.apiVersion)
        }
        install(SSE)
        install(ContentNegotiation) {
            json(anthropicJson)
        }
        install(HttpTimeout) {
            requestTimeoutMillis = settings.timeoutConfig.requestTimeoutMillis
            connectTimeoutMillis = settings.timeoutConfig.connectTimeoutMillis
            socketTimeoutMillis = settings.timeoutConfig.socketTimeoutMillis
        }
    }

    override suspend fun execute(prompt: Prompt, model: LLModel, tools: List<ToolDescriptor>): List<Message.Response> {
        logger.debug { "Executing prompt with model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        require(model.capabilities.contains(LLMCapability.Tools) || tools.isEmpty()) {
            "Model ${model.id} does not support tools"
        }

        // Use HarmonyAnthropicDownsampler to convert Harmony Prompt to Anthropic format
        val anthropicRequest = HarmonyAnthropicDownsampler.downsample(prompt, model)
        
        // Add streaming flag if needed
        val requestWithStream = anthropicRequest.copy(stream = false)

        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post(DEFAULT_MESSAGE_PATH) {
                setBody(requestWithStream)
            }

            if (response.status.isSuccess()) {
                val anthropicResponse = response.body<AnthropicMessagesResponse>()
                processAnthropicResponse(anthropicResponse)
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from Anthropic API: ${response.status}: $errorBody" }
                error("Error from Anthropic API: ${response.status}: $errorBody")
            }
        }
    }

    override fun executeStreaming(prompt: Prompt, model: LLModel): Flow<String> = flow {
        logger.debug { "Executing streaming prompt with model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }

        // Use HarmonyAnthropicDownsampler to convert Harmony Prompt to Anthropic format
        val anthropicRequest = HarmonyAnthropicDownsampler.downsample(prompt, model)
        
        // Add streaming flag
        val requestWithStream = anthropicRequest.copy(stream = true)

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
                    setBody(requestWithStream)
                }
            ) {
                incoming.collect { event ->
                    event
                        .takeIf { it.event == "content_block_delta" }
                        ?.data?.trim()?.let { 
                            val delta = anthropicJson.decodeFromString<AnthropicStreamEvent.ContentBlockDelta>(it)
                            (delta.delta as? AnthropicDelta.TextDelta)?.text?.let { emit(it) }
                        }
                }
            }
        } catch (e: SSEClientException) {
            e.response?.let { response ->
                logger.error { "Error from Anthropic API: ${response.status}: ${e.message}" }
                error("Error from Anthropic API: ${response.status}: ${e.message}")
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
        // Anthropic doesn't support multiple choices in a single request
        // We'd need to make multiple requests if this is needed
        logger.warn { "Anthropic doesn't support multiple choices natively, using single choice" }
        return execute(prompt, model, tools)
    }

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult {
        // Anthropic doesn't have a dedicated moderation endpoint
        // Could potentially use their safety features or implement custom logic
        logger.warn { "Anthropic doesn't have a dedicated moderation API" }
        return ModerationResult(
            isHarmful = false,
            categories = emptyMap()
        )
    }

    private fun processAnthropicResponse(response: AnthropicMessagesResponse): List<Message.Response> {
        val content = response.content
        if (content.isEmpty()) {
            logger.error { "Empty content in Anthropic response" }
            error("Empty content in Anthropic response")
        }

        // Extract token count from the response
        val totalTokensCount = response.usage.inputTokens + response.usage.outputTokens
        val inputTokensCount = response.usage.inputTokens
        val outputTokensCount = response.usage.outputTokens

        val metaInfo = ResponseMetaInfo.create(
            clock,
            totalTokensCount = totalTokensCount,
            inputTokensCount = inputTokensCount,
            outputTokensCount = outputTokensCount
        )

        // Process content blocks
        val messages = mutableListOf<Message.Response>()
        val textContent = mutableListOf<String>()
        
        content.forEach { block ->
            when (block) {
                is AnthropicContentBlock.Text -> {
                    textContent.add(block.text)
                }
                is AnthropicContentBlock.ToolUse -> {
                    // If we have accumulated text, add it as an assistant message first
                    if (textContent.isNotEmpty()) {
                        messages.add(
                            Message.Assistant(
                                content = textContent.joinToString("\n"),
                                finishReason = null,
                                metaInfo = metaInfo
                            )
                        )
                        textContent.clear()
                    }
                    
                    // Add tool call
                    messages.add(
                        Message.Tool.Call(
                            id = block.id,
                            tool = block.name,
                            content = block.input.toString(),
                            metaInfo = metaInfo
                        )
                    )
                }
                else -> {
                    // Handle other content block types if needed
                }
            }
        }
        
        // Add any remaining text content
        if (textContent.isNotEmpty()) {
            messages.add(
                Message.Assistant(
                    content = textContent.joinToString("\n"),
                    finishReason = response.stopReason,
                    metaInfo = metaInfo
                )
            )
        }

        return messages.ifEmpty {
            listOf(
                Message.Assistant(
                    content = "",
                    finishReason = response.stopReason,
                    metaInfo = metaInfo
                )
            )
        }
    }
}