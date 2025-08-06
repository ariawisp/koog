package ai.koog.prompt.executor.clients.google

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.agents.utils.SuitableForIO
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.ConnectionTimeoutConfig
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Attachment
import ai.koog.prompt.message.AttachmentContent
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.koog.prompt.params.LLMParams
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
import io.ktor.client.request.parameter
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.bodyAsText
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.HttpMethod
import io.ktor.http.contentType
import io.ktor.http.headers
import io.ktor.http.isSuccess
import io.ktor.serialization.kotlinx.json.json
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock
import kotlinx.serialization.json.Json
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Configuration settings for the Google AI client.
 *
 * @property baseUrl The base URL for the Google AI API.
 * @property timeoutConfig Timeout configuration for API requests.
 */
public class GoogleClientSettings(
    public val baseUrl: String = "https://generativelanguage.googleapis.com",
    public val timeoutConfig: ConnectionTimeoutConfig = ConnectionTimeoutConfig()
)

/**
 * Implementation of [LLMClient] for Google's Gemini API.
 *
 * This client supports both standard and streaming text generation with
 * optional tool calling capabilities.
 *
 * @param apiKey The API key for the Google AI API
 * @param settings Custom client settings, defaults to standard API endpoint and timeouts
 * @param baseClient Optional custom HTTP client
 * @param clock Clock instance used for tracking response metadata timestamps.
 */
public open class GoogleLLMClient(
    private val apiKey: String,
    private val settings: GoogleClientSettings = GoogleClientSettings(),
    baseClient: HttpClient = HttpClient(),
    private val clock: Clock = Clock.System
) : LLMClient {

    private companion object {
        private val logger = KotlinLogging.logger { }

        private const val DEFAULT_PATH = "v1beta/models"
        private const val DEFAULT_METHOD_GENERATE_CONTENT = "generateContent"
        private const val DEFAULT_METHOD_STREAM_GENERATE_CONTENT = "streamGenerateContent"
    }

    private val json = googleJson

    private val httpClient = baseClient.config {
        defaultRequest {
            url(settings.baseUrl)
            url.parameters.append("key", apiKey)
            contentType(ContentType.Application.Json)
        }
        install(SSE)
        install(ContentNegotiation) {
            json(json)
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

        // Use HarmonyGoogleDownsampler to convert Harmony Prompt to Google format
        val googleRequest = HarmonyGoogleDownsampler.downsample(prompt, model)
        
        return withContext(Dispatchers.SuitableForIO) {
            val response = httpClient.post("$DEFAULT_PATH/${model.id}:$DEFAULT_METHOD_GENERATE_CONTENT") {
                setBody(googleRequest)
            }

            if (response.status.isSuccess()) {
                val googleResponse = response.body<GoogleGenerateResponse>()
                processGoogleResponse(googleResponse)
            } else {
                val errorBody = response.bodyAsText()
                logger.error { "Error from Google API: ${response.status}: $errorBody" }
                error("Error from Google API: ${response.status}: $errorBody")
            }
        }
    }

    override fun executeStreaming(prompt: Prompt, model: LLModel): Flow<String> = flow {
        logger.debug { "Executing streaming prompt with model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }

        // Use HarmonyGoogleDownsampler to convert Harmony Prompt to Google format
        val request = HarmonyGoogleDownsampler.downsample(prompt, model)

        try {
            httpClient.sse(
                urlString = "$DEFAULT_PATH/${model.id}:$DEFAULT_METHOD_STREAM_GENERATE_CONTENT",
                request = {
                    method = HttpMethod.Post
                    parameter("alt", "sse")
                    accept(ContentType.Text.EventStream)
                    headers {
                        append(HttpHeaders.CacheControl, "no-cache")
                        append(HttpHeaders.Connection, "keep-alive")
                    }
                    setBody(json.encodeToString(GoogleGenerateRequest.serializer(), request))
                }
            ) {
                incoming.collect { event ->
                    event
                        .takeIf { it.data != "[DONE]" }
                        ?.data?.trim()?.let { json.decodeFromString<GoogleGenerateResponse>(it) }
                        ?.candidates?.firstOrNull()?.content
                        ?.parts?.forEach { part -> if (part is GooglePart.Text) emit(part.text) }
                }
            }
        } catch (e: SSEClientException) {
            e.response?.let { response ->
                logger.error { "Error from GoogleAI API: ${response.status}: ${e.message}" }
                error("Error from GoogleAI API: ${response.status}: ${e.message}")
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
        logger.debug { "Executing prompt with multiple choices: $prompt with tools: $tools and model: $model" }
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        require(model.capabilities.contains(LLMCapability.Tools) || tools.isEmpty()) {
            "Model ${model.id} does not support tools"
        }
        require(model.capabilities.contains(LLMCapability.MultipleChoices)) {
            "Model ${model.id} does not support multiple choices"
        }

        // Google doesn't support multiple choices in a single request
        // We'd need to make multiple requests if this is needed
        logger.warn { "Google doesn't support multiple choices natively, using single choice" }
        return execute(prompt, model, tools)
    }



    /**
     * Processes a single Google AI API candidate into internal message format.
     *
     * @param candidate The candidate from the Google AI API response
     * @param metaInfo The metadata for the response
     * @return A list of response messages
     */
    @OptIn(ExperimentalUuidApi::class)
    private fun processGoogleCandidate(candidate: GoogleCandidate, metaInfo: ResponseMetaInfo): List<Message.Response> {
        val parts = candidate.content?.parts.orEmpty()
        val responses = parts.map { part ->
            when (part) {
                is GooglePart.Text -> Message.Assistant(
                    content = part.text,
                    finishReason = candidate.finishReason,
                    metaInfo = metaInfo
                )

                is GooglePart.FunctionCall -> Message.Tool.Call(
                    id = Uuid.random().toString(),
                    tool = part.functionCall.name,
                    content = part.functionCall.args.toString(),
                    metaInfo = metaInfo
                )

                else -> error("Not supported part type: $part")
            }
        }

        return when {
            // Fix the situation when the model decides to both call tools and talk
            responses.any { it is Message.Tool.Call } -> responses.filterIsInstance<Message.Tool.Call>()
            // If no messages where returned, return an empty message and check finishReason
            responses.isEmpty() -> listOf(
                Message.Assistant(
                    content = "",
                    finishReason = candidate.finishReason,
                    metaInfo = metaInfo
                )
            )
            // Just return responses
            else -> responses
        }
    }

    /**
     * Processes the Google AI API response into messages.
     *
     * @param response The raw response from the Google AI API
     * @return A list of response messages
     */
    private fun processGoogleResponse(response: GoogleGenerateResponse): List<Message.Response> {
        if (response.candidates.isEmpty()) {
            logger.error { "Empty candidates in Gemini response" }
            error("Empty candidates in Gemini response")
        }

        // Extract token count from the response
        val inputTokensCount = response.usageMetadata?.promptTokenCount
        val outputTokensCount = response.usageMetadata?.candidatesTokenCount
        val totalTokensCount = response.usageMetadata?.totalTokenCount

        val metaInfo = ResponseMetaInfo.create(
            clock,
            totalTokensCount = totalTokensCount,
            inputTokensCount = inputTokensCount,
            outputTokensCount = outputTokensCount
        )

        // Process the first candidate (Google typically returns one)
        return if (response.candidates.isNotEmpty()) {
            processGoogleCandidate(response.candidates.first(), metaInfo)
        } else {
            listOf(
                Message.Assistant(
                    content = "",
                    finishReason = null,
                    metaInfo = metaInfo
                )
            )
        }
    }

    /**
     * Moderates the given prompt using the specified language model.
     * This method is not supported by the Google API and will throw an exception when invoked.
     *
     * @param prompt The prompt to be evaluated for moderation.
     * @param model The language model to use for moderation.
     * @return This method does not return a result as moderation is not supported by the Google API.
     * @throws UnsupportedOperationException Always thrown since moderation is not supported.
     */
    public override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult {
        logger.warn { "Moderation is not supported by Google API" }
        throw UnsupportedOperationException("Moderation is not supported by Google API.")
    }
}
