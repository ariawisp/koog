package ai.koog.prompt.executor.noesis

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.agents.core.tools.ToolParameterType
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.harmony.*
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.datetime.Clock
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Native Harmony LLM client that uses the Rust openai-harmony library via JNI.
 * This client directly renders Harmony messages to tokens using the native implementation,
 * avoiding any JSON serialization or downsampling.
 * 
 * This is used for GPT-OSS models that have native Harmony support.
 */
public class HarmonyNativeLLMClient(
    private val inferenceClient: HarmonyInferenceClient = HarmonyInferenceClient(),
    private val clock: Clock = Clock.System
) : LLMClient {
    
    private companion object {
        private val logger = KotlinLogging.logger { }
        private val json = Json { 
            prettyPrint = true
            ignoreUnknownKeys = true
        }
        
        fun mapParameterType(type: ToolParameterType): String {
            return when (type) {
                is ToolParameterType.String -> "string"
                is ToolParameterType.Integer -> "number"
                is ToolParameterType.Float -> "number"
                is ToolParameterType.Boolean -> "boolean"
                is ToolParameterType.List -> "any[]"
                is ToolParameterType.Object -> "object"
                is ToolParameterType.Enum -> "string"
            }
        }
    }
    
    // Lazily initialize the Harmony encoding
    private val harmonyEncoding: HarmonyEncoding by lazy {
        HarmonyEncoding.load("harmony_gpt_oss").getOrThrow()
    }
    
    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        
        logger.debug { "Executing Harmony-native prompt with model: ${model.id}" }
        
        // Convert Prompt (which IS HarmonyCore) to native Harmony messages
        val harmonyMessages = convertPromptToHarmonyMessages(prompt, tools)
        
        // Render to tokens using native Rust implementation
        val tokens = harmonyEncoding.renderConversation(
            messages = harmonyMessages,
            role = Role.ASSISTANT,
            config = RenderConfig(
                includeSystemTokens = true,
                includeDeveloperTokens = true
            )
        ).getOrThrow()
        
        logger.debug { "Rendered ${tokens.size} tokens for Harmony-native request" }
        
        // Get stop tokens for proper inference termination
        val stopTokens = harmonyEncoding.getStopTokens(
            forAssistantActions = tools.isNotEmpty()
        ).getOrThrow()
        
        // Send tokens directly to the model for inference
        val responseTokens = inferenceClient.inferWithTokens(
            model = model,
            tokens = tokens,
            stopTokens = stopTokens,
            maxTokens = prompt.metadata.maxTokens ?: 2048,
            temperature = (prompt.metadata.temperature ?: 0.7).toFloat(),
            topP = (prompt.metadata.topP ?: 0.9).toFloat()
        )
        
        // Parse response tokens back to messages
        val responseMessages = harmonyEncoding.parseTokens(
            tokens = responseTokens,
            role = Role.ASSISTANT
        ).getOrThrow()
        
        logger.debug { "Parsed ${responseMessages.size} messages from response tokens" }
        
        // Convert Harmony messages back to Koog Message.Response
        return convertHarmonyMessagesToResponses(responseMessages)
    }
    
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel
    ): Flow<String> = flow {
        require(model.capabilities.contains(LLMCapability.Completion)) {
            "Model ${model.id} does not support chat completions"
        }
        
        logger.debug { "Executing streaming Harmony-native prompt with model: ${model.id}" }
        
        // Convert Prompt to native Harmony messages
        val harmonyMessages = convertPromptToHarmonyMessages(prompt, emptyList())
        
        // Render to tokens
        val tokens = harmonyEncoding.renderConversation(
            messages = harmonyMessages,
            role = Role.ASSISTANT
        ).getOrThrow()
        
        // Get stop tokens
        val stopTokens = harmonyEncoding.getStopTokens().getOrThrow()
        
        // Create streaming parser for real-time token processing
        val streamingParser = harmonyEncoding.createStreamingParser(
            role = Role.ASSISTANT
        ).getOrThrow()
        
        try {
            // Stream tokens from the model
            inferenceClient.inferStreamingWithTokens(
                model = model,
                tokens = tokens,
                stopTokens = stopTokens,
                temperature = (prompt.metadata.temperature ?: 0.7).toFloat(),
                topP = (prompt.metadata.topP ?: 0.9).toFloat()
            ).collect { token ->
                // Process each token through the parser
                val state = streamingParser.processToken(token)
                
                // Emit content deltas from the FINAL channel only
                if (state.channel == "final" && state.lastContentDelta.isNotEmpty()) {
                    emit(state.lastContentDelta)
                }
            }
        } finally {
            streamingParser.close()
        }
    }
    
    override suspend fun executeMultipleChoices(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        // For multiple choices, we need to call the model multiple times
        // or use a special inference parameter if the model supports it
        logger.debug { "Executing multiple choices with Harmony-native client" }
        
        // For now, just execute once - can be extended to support multiple choices
        return execute(prompt, model, tools)
    }
    
    /**
     * Convert Koog Prompt (which IS HarmonyCore) to native Harmony messages.
     */
    private fun convertPromptToHarmonyMessages(
        prompt: Prompt,
        tools: List<ToolDescriptor>
    ): List<HarmonyMessage> {
        val messages = mutableListOf<HarmonyMessage>()
        
        // Add system message with Harmony structure
        messages.add(
            HarmonyMessage(
                author = HarmonyAuthor.from(Role.SYSTEM),
                content = listOf(
                    HarmonyContent.Text(
                        buildString {
                            appendLine(prompt.systemContext.modelIdentity)
                            appendLine("Knowledge cutoff: ${prompt.systemContext.knowledgeCutoff}")
                            appendLine("Current date: ${prompt.systemContext.currentDate}")
                            appendLine()
                            appendLine("Reasoning: ${prompt.systemContext.reasoningEffort.name.lowercase()}")
                            appendLine()
                            appendLine("# Valid channels: analysis, commentary, final. Channel must be included for every message.")
                            if (tools.isNotEmpty()) {
                                appendLine("Calls to these tools must go to the commentary channel: 'functions'.")
                            }
                        }
                    )
                ),
                channel = null // System messages don't have channels
            )
        )
        
        // Add developer message with instructions and tools
        if (prompt.developerContext.instructions.isNotEmpty() || tools.isNotEmpty()) {
            messages.add(
                HarmonyMessage(
                    author = HarmonyAuthor.from(Role.DEVELOPER),
                    content = listOf(
                        HarmonyContent.Text(
                            buildString {
                                if (prompt.developerContext.instructions.isNotEmpty()) {
                                    appendLine("# Instructions")
                                    appendLine()
                                    appendLine(prompt.developerContext.instructions)
                                }
                                
                                if (tools.isNotEmpty()) {
                                    if (prompt.developerContext.instructions.isNotEmpty()) {
                                        appendLine()
                                    }
                                    appendLine("# Tools")
                                    appendLine()
                                    appendLine("## functions")
                                    appendLine()
                                    appendLine("namespace functions {")
                                    appendLine()
                                    
                                    tools.forEach { tool ->
                                        appendLine("// ${tool.description}")
                                        append("type ${tool.name} = ")
                                        val allParams = tool.requiredParameters + tool.optionalParameters
                                        if (allParams.isEmpty()) {
                                            appendLine("() => any;")
                                        } else {
                                            appendLine("(_: {")
                                            tool.requiredParameters.forEach { param ->
                                                appendLine("// ${param.description}")
                                                append(param.name)
                                                append(": ")
                                                append(mapParameterType(param.type))
                                                appendLine(",")
                                            }
                                            tool.optionalParameters.forEach { param ->
                                                appendLine("// ${param.description}")
                                                append(param.name)
                                                append("?: ")
                                                append(mapParameterType(param.type))
                                                appendLine(",")
                                            }
                                            appendLine("}) => any;")
                                        }
                                        appendLine()
                                    }
                                    
                                    appendLine("} // namespace functions")
                                }
                            }
                        )
                    ),
                    channel = null
                )
            )
        }
        
        // Add conversation messages from the Prompt's conversation graph
        prompt.conversation.messages.forEach { message ->
            when (message) {
                is ChanneledMessage.Analysis -> {
                    messages.add(
                        HarmonyMessage(
                            author = HarmonyAuthor.from(Role.ASSISTANT),
                            content = listOf(HarmonyContent.Text(message.content)),
                            channel = "analysis"
                        )
                    )
                }
                is ChanneledMessage.Commentary -> {
                    messages.add(
                        HarmonyMessage(
                            author = HarmonyAuthor.from(Role.ASSISTANT),
                            content = listOf(HarmonyContent.Text(message.content)),
                            channel = "commentary",
                            recipient = message.recipient
                        )
                    )
                }
                is ChanneledMessage.Final -> {
                    // Determine role from content or context
                    val role = when {
                        message.role != null -> message.role!!
                        message.content.startsWith("User:") -> Role.USER
                        else -> Role.ASSISTANT
                    }
                    messages.add(
                        HarmonyMessage(
                            author = HarmonyAuthor.from(role),
                            content = listOf(HarmonyContent.Text(message.content)),
                            channel = if (role == Role.ASSISTANT) "final" else null
                        )
                    )
                }
            }
        }
        
        return messages
    }
    
    /**
     * Convert Harmony messages back to Koog Message.Response.
     */
    private fun convertHarmonyMessagesToResponses(
        messages: List<HarmonyMessage>
    ): List<Message.Response> {
        val responses = mutableListOf<Message.Response>()
        val metaInfo = ResponseMetaInfo.create(clock)
        
        messages.forEach { message ->
            when (message.channel) {
                "final" -> {
                    // User-facing response
                    responses.add(
                        Message.Assistant(
                            content = message.getTextContent(),
                            finishReason = "stop",
                            metaInfo = metaInfo
                        )
                    )
                }
                "commentary" -> {
                    // Check if this is a tool call
                    val recipient = message.recipient
                    if (recipient != null && recipient.startsWith("functions.")) {
                        val toolName = recipient.removePrefix("functions.")
                        responses.add(
                            Message.Tool.Call(
                                id = "call_${System.currentTimeMillis()}",
                                tool = toolName,
                                content = message.getTextContent(),
                                metaInfo = metaInfo
                            )
                        )
                    }
                }
                "analysis" -> {
                    // Internal reasoning - typically not exposed to user
                    // but can be logged for debugging
                    logger.debug { "Analysis: ${message.getTextContent()}" }
                }
            }
        }
        
        return responses
    }
    
    fun close() {
        harmonyEncoding.close()
    }
    
    override suspend fun moderate(
        prompt: Prompt,
        model: LLModel
    ): ModerationResult {
        logger.warn { "Moderation is not supported by HarmonyNativeLLMClient" }
        throw UnsupportedOperationException("Moderation is not supported by HarmonyNativeLLMClient")
    }
}

/**
 * Client for native GPU inference using Metal backend.
 * Falls back to HTTP inference if Metal is not available.
 */
public class HarmonyInferenceClient(
    private val modelPath: String? = null,
    private val httpFallbackUrl: String? = null
) {
    private companion object {
        private val logger = KotlinLogging.logger { }
    }
    
    private val metalAvailable = MetalInferenceJNI.isAvailable()
    private var metalModel: MetalInferenceJNI.ModelHandle? = null
    
    init {
        if (metalAvailable && modelPath != null) {
            metalModel = MetalInferenceJNI.loadModel(modelPath)
            if (metalModel == null) {
                logger.warn { "Failed to load Metal model from: $modelPath" }
            }
        }
    }
    
    /**
     * Send tokens to the model for inference.
     * Uses Metal if available, otherwise falls back to HTTP.
     * @return Response tokens from the model
     */
    public suspend fun inferWithTokens(
        model: LLModel,
        tokens: IntArray,
        stopTokens: IntArray,
        maxTokens: Int = 2048,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): IntArray {
        // Try Metal inference first
        metalModel?.let { handle ->
            logger.debug { "Using Metal inference for ${tokens.size} input tokens" }
            
            val result = MetalInferenceJNI.inferTokens(
                handle,
                tokens,
                maxTokens,
                temperature,
                topP
            )
            
            if (result != null) {
                return result
            } else {
                logger.warn { "Metal inference failed, falling back to HTTP" }
            }
        }
        
        // Fallback to HTTP inference
        if (httpFallbackUrl != null) {
            logger.debug { "Using HTTP inference fallback" }
            return performHttpInference(model, tokens, stopTokens, maxTokens, temperature, topP)
        }
        
        throw IllegalStateException(
            "No inference backend available. Metal not loaded and no HTTP fallback configured."
        )
    }
    
    /**
     * Stream tokens from the model.
     * Currently only supports HTTP streaming; Metal inference is non-streaming.
     */
    public fun inferStreamingWithTokens(
        model: LLModel,
        tokens: IntArray,
        stopTokens: IntArray,
        temperature: Float = 0.7f,
        topP: Float = 0.9f
    ): Flow<Int> = flow {
        // Metal doesn't support streaming yet, use HTTP
        if (httpFallbackUrl != null) {
            // Stream from HTTP endpoint
            performHttpStreamingInference(model, tokens, stopTokens, temperature, topP)
                .collect { token -> emit(token) }
        } else {
            // Fall back to non-streaming Metal inference
            metalModel?.let { handle ->
                val result = MetalInferenceJNI.inferTokens(
                    handle,
                    tokens,
                    2048,
                    temperature,
                    topP
                )
                result?.forEach { token -> emit(token) }
            } ?: throw IllegalStateException("No inference backend available")
        }
    }
    
    private suspend fun performHttpInference(
        model: LLModel,
        tokens: IntArray,
        stopTokens: IntArray,
        maxTokens: Int,
        temperature: Float,
        topP: Float
    ): IntArray {
        // TODO: Implement actual HTTP call to inference endpoint
        throw NotImplementedError("HTTP inference not yet implemented")
    }
    
    private fun performHttpStreamingInference(
        model: LLModel,
        tokens: IntArray,
        stopTokens: IntArray,
        temperature: Float,
        topP: Float
    ): Flow<Int> = flow {
        // TODO: Implement actual streaming from inference endpoint
        throw NotImplementedError("HTTP streaming inference not yet implemented")
    }
    
    fun close() {
        metalModel?.let {
            MetalInferenceJNI.releaseModel(it)
            metalModel = null
        }
    }
}