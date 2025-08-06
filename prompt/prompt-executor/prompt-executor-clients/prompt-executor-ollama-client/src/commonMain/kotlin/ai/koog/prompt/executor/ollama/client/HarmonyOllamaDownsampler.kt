package ai.koog.prompt.executor.ollama.client

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.*

/**
 * Downsampler that converts Harmony-format Prompts to Ollama API format.
 * 
 * Ollama is primarily used for local LLM execution and has a simpler API:
 * - Supports both chat and generate endpoints
 * - Basic tool support (model-dependent)
 * - Images via base64 encoding
 * 
 * Channel mapping:
 * - ANALYSIS → Drop (not supported by most local models)
 * - COMMENTARY → Tool calls (if model supports)
 * - FINAL → User/Assistant messages
 */
internal object HarmonyOllamaDownsampler : HarmonyDownsamplerBase<OllamaChatRequest>() {
    
    /**
     * Downsample a Harmony Prompt to Ollama chat format.
     * Uses the chat endpoint for full conversation support.
     */
    override fun downsample(prompt: Prompt): OllamaChatRequest {
        // Convert messages with channel filtering
        val messages = downsampleMessages(prompt.conversation)
        
        // Convert tools if the model supports them
        val tools = if (prompt.developerContext.tools.isNotEmpty()) {
            prompt.developerContext.tools.map { tool ->
                OllamaTool(
                    type = "function",
                    function = OllamaFunction(
                        name = tool.name,
                        description = tool.description,
                        parameters = convertToolToJsonSchema(tool)
                    )
                )
            }
        } else null
        
        // Map reasoning effort to Ollama options
        val options = mapReasoningToOllamaOptions(prompt.systemContext.reasoningEffort)
        
        return OllamaChatRequest(
            model = prompt.metadata.model,
            messages = messages,
            tools = tools,
            format = null, // Could be set for JSON mode if needed
            options = options,
            stream = false,
            keepAlive = null // Use default
        )
    }
    
    /**
     * Convert Harmony messages to Ollama format.
     * Ollama has a simpler message structure than other providers.
     */
    private fun downsampleMessages(conversation: ConversationGraph): List<OllamaMessage> {
        val messages = mutableListOf<OllamaMessage>()
        
        // Add system message if present
        val systemPrompt = buildSystemPromptForOllama(conversation)
        if (systemPrompt.isNotEmpty()) {
            messages.add(OllamaMessage(
                role = "system",
                content = systemPrompt,
                images = null,
                toolCalls = null
            ))
        }
        
        // Process conversation messages, filtering out analysis channel
        val visibleMessages = filterUserSafeMessages(conversation.messages)
        
        visibleMessages.forEach { message ->
            when (message.channel) {
                "commentary" -> {
                    // Handle tool interactions
                    if (isToolCall(message)) {
                        val toolName = extractToolName(message.recipient!!)
                        messages.add(OllamaMessage(
                            role = "assistant",
                            content = "",
                            images = null,
                            toolCalls = listOf(
                                OllamaToolCall(
                                    function = OllamaFunctionCall(
                                        name = toolName,
                                        arguments = parseJsonContentSafely(message.extractTextContent()).jsonObject
                                    )
                                )
                            )
                        ))
                    } else if (isToolResponse(message)) {
                        // Tool responses in Ollama are tool role messages
                        messages.add(OllamaMessage(
                            role = "tool",
                            content = message.extractTextContent(),
                            images = null,
                            toolCalls = null
                        ))
                    }
                }
                "final", null -> {
                    // Regular conversation messages
                    val role = when (message.author.role) {
                        Role.USER -> "user"
                        Role.ASSISTANT -> "assistant"
                        Role.SYSTEM -> "system"
                        Role.TOOL -> "tool"
                        else -> "user"
                    }
                    
                    messages.add(OllamaMessage(
                        role = role,
                        content = message.extractTextContent(),
                        images = extractBase64Images(message),
                        toolCalls = null
                    ))
                }
                else -> {
                    // Skip unknown channels
                }
            }
        }
        
        return messages
    }
    
    /**
     * Build system prompt from conversation context.
     * Combines system and developer messages.
     */
    private fun buildSystemPromptForOllama(conversation: ConversationGraph): String {
        val systemMessages = conversation.messages
            .filter { it.author.role == Role.SYSTEM }
            .map { it.extractTextContent() }
        
        val developerMessages = conversation.messages
            .filter { it.author.role == Role.DEVELOPER }
            .map { it.extractTextContent() }
        
        return (systemMessages + developerMessages)
            .filter { it.isNotEmpty() }
            .joinToString("\n\n")
    }
    
    /**
     * Extract base64-encoded images from message content.
     * Ollama expects images as base64 strings.
     */
    private fun extractBase64Images(message: HarmonyMessage): List<String>? {
        // For now, return null as we don't have image support in HarmonyContent yet
        // This would need to be implemented when image support is added to Harmony
        return null
    }
    
    /**
     * Map reasoning effort to Ollama generation options.
     * Ollama uses similar parameters to other providers.
     */
    private fun mapReasoningToOllamaOptions(effort: ReasoningEffort): OllamaOptions {
        val samplingParams = mapReasoningToSamplingParams(effort)
        
        return OllamaOptions(
            temperature = samplingParams.temperature,
            topP = samplingParams.topP,
            topK = samplingParams.topK,
            repeatPenalty = when (effort) {
                ReasoningEffort.LOW -> 1.0  // No penalty
                ReasoningEffort.MEDIUM -> 1.1  // Slight penalty
                ReasoningEffort.HIGH -> 1.2  // Higher penalty for focused reasoning
            },
            seed = null, // Let Ollama choose
            numPredict = null, // Use model default
            numCtx = null, // Use model default
            stop = null // Use model default
        )
    }
    
    /**
     * Alternative conversion for the generate endpoint.
     * Used when simple prompt-based generation is sufficient.
     */
    internal fun downsampleToGenerate(prompt: Prompt): OllamaGenerateRequest {
        // Extract system message
        val systemContent = prompt.systemContext.modelIdentity +
            if (prompt.developerContext.instructions.isNotEmpty()) {
                "\n\n${prompt.developerContext.instructions}"
            } else ""
        
        // Extract the last user message as the prompt
        val userPrompt = prompt.conversation.messages
            .filter { it.author.role == Role.USER }
            .lastOrNull()
            ?.extractTextContent()
            ?: ""
        
        val options = mapReasoningToOllamaOptions(prompt.systemContext.reasoningEffort)
        
        return OllamaGenerateRequest(
            model = prompt.metadata.model,
            prompt = userPrompt,
            system = systemContent.takeIf { it.isNotEmpty() },
            template = null,
            context = null,
            stream = false,
            raw = null,
            format = null,
            options = options,
            keepAlive = null
        )
    }
}

/**
 * Extension function for direct downsampling from Prompt.
 */
internal fun Prompt.toOllama(): OllamaChatRequest = 
    HarmonyOllamaDownsampler.downsample(this)

/**
 * Extension function for generate endpoint.
 */
internal fun Prompt.toOllamaGenerate(): OllamaGenerateRequest = 
    HarmonyOllamaDownsampler.downsampleToGenerate(this)