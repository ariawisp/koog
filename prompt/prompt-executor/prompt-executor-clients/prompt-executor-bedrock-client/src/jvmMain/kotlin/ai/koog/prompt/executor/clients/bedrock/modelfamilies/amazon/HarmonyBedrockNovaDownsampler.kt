package ai.koog.prompt.executor.clients.bedrock.modelfamilies.amazon

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*

/**
 * Harmony-based downsampler for Bedrock Amazon Nova models.
 * Converts from Harmony format to Bedrock's Nova API format.
 */
internal object HarmonyBedrockNovaDownsampler : HarmonyDownsamplerBase<NovaRequest>() {

    override fun downsample(prompt: Prompt): NovaRequest {
        val harmonyCore = prompt.harmony
        val systemMessages = mutableListOf<NovaSystemMessage>()
        val conversationMessages = mutableListOf<NovaMessage>()
        
        // Determine if we should include analysis channel based on reasoning effort
        val includeAnalysis = harmonyCore.metadata.reasoning == ReasoningEffort.HIGH
        
        // Build system prompt from context
        val systemPrompt = buildSystemPrompt(
            harmonyCore.systemContext,
            harmonyCore.developerContext,
            includeAnalysis
        )
        
        if (systemPrompt.isNotEmpty()) {
            systemMessages.add(NovaSystemMessage(text = systemPrompt))
        }
        
        // Add analysis channel messages to system if reasoning is HIGH
        if (includeAnalysis) {
            harmonyCore.messages
                .filter { it.channel == ConversationChannel.ANALYSIS }
                .forEach { msg ->
                    val analysisContent = "[Internal Analysis]\n${msg.extractTextContent()}"
                    systemMessages.add(NovaSystemMessage(text = analysisContent))
                }
        }
        
        // Process user-safe messages (final and commentary channels)
        val userSafeMessages = filterUserSafeMessages(harmonyCore.messages)
        
        userSafeMessages.forEach { msg ->
            when {
                // Handle tool calls - Nova doesn't have native tool support yet
                // So we'll include them as text in assistant messages
                isToolCall(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolCall } as? HarmonyContent.ToolCall
                    toolContent?.let {
                        val toolCallText = "[Tool Call: ${it.name}]\nArguments: ${it.arguments}"
                        conversationMessages.add(
                            NovaMessage(
                                role = "assistant",
                                content = listOf(NovaContent(text = toolCallText))
                            )
                        )
                    }
                }
                
                // Handle tool responses
                isToolResponse(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolResponse } as? HarmonyContent.ToolResponse
                    toolContent?.let {
                        val toolResponseText = "[Tool Response]\n${it.result}"
                        conversationMessages.add(
                            NovaMessage(
                                role = "user",
                                content = listOf(NovaContent(text = toolResponseText))
                            )
                        )
                    }
                }
                
                // Handle regular messages
                else -> {
                    val textContent = msg.extractTextContent()
                    if (textContent.isNotEmpty()) {
                        val role = when (msg.role) {
                            HarmonyRole.USER -> "user"
                            HarmonyRole.ASSISTANT -> "assistant"
                            else -> null
                        }
                        
                        role?.let {
                            conversationMessages.add(
                                NovaMessage(
                                    role = role,
                                    content = listOf(NovaContent(text = textContent))
                                )
                            )
                        }
                    }
                }
            }
        }
        
        // Map reasoning effort to sampling parameters
        val samplingParams = mapReasoningToSamplingParams(harmonyCore.metadata.reasoning)
        
        val inferenceConfig = NovaInferenceConfig(
            maxTokens = harmonyCore.metadata.maxTokens ?: 4096,
            temperature = samplingParams.temperature?.toFloat(),
            topP = samplingParams.topP?.toFloat()
        )
        
        return NovaRequest(
            messages = conversationMessages,
            inferenceConfig = inferenceConfig,
            system = systemMessages.takeIf { it.isNotEmpty() }
        )
    }
}