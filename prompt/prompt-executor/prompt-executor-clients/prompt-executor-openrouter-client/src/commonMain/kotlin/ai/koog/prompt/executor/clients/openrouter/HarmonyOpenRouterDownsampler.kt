package ai.koog.prompt.executor.clients.openrouter

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.*

/**
 * Downsampler that converts Harmony-format Prompts to OpenRouter API format.
 * 
 * OpenRouter is a meta-provider that routes to various LLM providers:
 * - Uses OpenAI-compatible format as its base
 * - Supports a wide variety of models from different providers
 * - Handles provider-specific parameters through transforms
 * 
 * Channel mapping (follows OpenAI convention):
 * - ANALYSIS → System message for HIGH reasoning
 * - COMMENTARY → Tool calls and responses
 * - FINAL → User/Assistant messages
 */
internal object HarmonyOpenRouterDownsampler : HarmonyDownsamplerBase<OpenRouterChatRequest>() {
    
    /**
     * Downsample a Harmony Prompt to OpenRouter format.
     * OpenRouter uses OpenAI-compatible format with provider-specific extensions.
     */
    override fun downsample(prompt: Prompt): OpenRouterChatRequest {
        val messages = downsampleMessages(prompt)
        
        // Convert tools to OpenRouter format (OpenAI-compatible)
        val tools = if (prompt.developerContext.tools.isNotEmpty()) {
            prompt.developerContext.tools.map { tool ->
                OpenRouterTool(
                    type = "function",
                    function = OpenRouterFunction(
                        name = tool.name,
                        description = tool.description,
                        parameters = convertToolToJsonSchema(tool)
                    )
                )
            }
        } else null
        
        // Map reasoning effort to OpenRouter parameters
        val samplingParams = mapReasoningToSamplingParams(prompt.systemContext.reasoningEffort)
        
        return OpenRouterChatRequest(
            model = prompt.metadata.model,
            messages = messages,
            tools = tools,
            temperature = samplingParams.temperature,
            maxTokens = 4096, // Default max tokens, could be configured via params
            topP = samplingParams.topP,
            frequencyPenalty = null,
            presencePenalty = null,
            stop = null,
            stream = false,
            // OpenRouter-specific fields
            provider = null, // Could configure provider preferences
            transforms = null, // Could be used for provider-specific transforms
            route = null // Could be used to specify routing preferences
        )
    }
    
    /**
     * Convert Harmony messages to OpenRouter format.
     * OpenRouter follows the OpenAI message structure.
     */
    private fun downsampleMessages(prompt: Prompt): List<OpenRouterMessage> {
        val messages = mutableListOf<OpenRouterMessage>()
        
        // Build and add system message if needed
        val systemPrompt = buildSystemPrompt(
            prompt.systemContext,
            prompt.developerContext,
            prompt.systemContext.reasoningEffort == ReasoningEffort.HIGH
        )
        if (systemPrompt.isNotEmpty()) {
            messages.add(OpenRouterMessage(
                role = "system",
                content = JsonPrimitive(systemPrompt),
                name = null,
                toolCalls = null,
                toolCallId = null
            ))
        }
        
        // Process conversation messages
        val visibleMessages = filterUserSafeMessages(prompt.conversation.messages)
        
        visibleMessages.forEach { message ->
            when (message.channel) {
                "commentary" -> {
                    // Handle tool interactions
                    if (isToolCall(message)) {
                        val toolName = extractToolName(message.recipient!!)
                        val toolArgs = parseJsonContentSafely(message.extractTextContent())
                        
                        messages.add(OpenRouterMessage(
                            role = "assistant",
                            content = null,
                            name = null,
                            toolCalls = listOf(
                                OpenRouterToolCall(
                                    id = "tool_${toolName}_${message.timestamp}",
                                    type = "function",
                                    function = OpenRouterFunctionCall(
                                        name = toolName,
                                        arguments = toolArgs.toString()
                                    )
                                )
                            ),
                            toolCallId = null
                        ))
                    } else if (isToolResponse(message)) {
                        // Tool responses
                        messages.add(OpenRouterMessage(
                            role = "tool",
                            content = JsonPrimitive(message.extractTextContent()),
                            name = message.author.name,
                            toolCalls = null,
                            toolCallId = "tool_${message.author.name}_${message.timestamp}"
                        ))
                    }
                }
                "final", null -> {
                    // Regular conversation messages
                    val role = when (message.author.role) {
                        Role.USER -> "user"
                        Role.ASSISTANT -> "assistant"
                        else -> return@forEach // Skip other roles in final channel
                    }
                    
                    messages.add(OpenRouterMessage(
                        role = role,
                        content = JsonPrimitive(message.extractTextContent()),
                        name = null,
                        toolCalls = null,
                        toolCallId = null
                    ))
                }
                else -> {
                    // Skip unknown channels
                }
            }
        }
        
        return messages
    }
    
}

/**
 * Extension function for direct downsampling from Prompt.
 */
internal fun Prompt.toOpenRouter(): OpenRouterChatRequest = 
    HarmonyOpenRouterDownsampler.downsample(this)