package ai.koog.prompt.executor.clients.bedrock.modelfamilies.meta

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*

/**
 * Harmony-based downsampler for Bedrock Meta Llama models.
 * Converts from Harmony format to Bedrock's Llama prompt format.
 */
internal object HarmonyBedrockLlamaDownsampler : HarmonyDownsamplerBase<LlamaRequest>() {

    override fun downsample(prompt: Prompt): LlamaRequest {
        val harmonyCore = prompt.harmony
        val promptParts = mutableListOf<String>()
        
        // Start with the Llama format header
        promptParts.add("<|begin_of_text|>")
        
        // Determine if we should include analysis channel based on reasoning effort
        val includeAnalysis = harmonyCore.metadata.reasoning == ReasoningEffort.HIGH
        
        // Build system prompt from context
        val systemPrompt = buildSystemPrompt(
            harmonyCore.systemContext,
            harmonyCore.developerContext,
            includeAnalysis
        )
        
        if (systemPrompt.isNotEmpty()) {
            promptParts.add("<|start_header_id|>system<|end_header_id|>\n\n$systemPrompt<|eot_id|>")
        }
        
        // Add analysis channel messages to system if reasoning is HIGH
        if (includeAnalysis) {
            harmonyCore.messages
                .filter { it.channel == ConversationChannel.ANALYSIS }
                .forEach { msg ->
                    val analysisContent = "[Internal Analysis]\n${msg.extractTextContent()}"
                    promptParts.add("<|start_header_id|>system<|end_header_id|>\n\n$analysisContent<|eot_id|>")
                }
        }
        
        // Process user-safe messages (final and commentary channels)
        val userSafeMessages = filterUserSafeMessages(harmonyCore.messages)
        
        userSafeMessages.forEach { msg ->
            when {
                // Handle tool calls - Llama doesn't have native tool support
                // So we'll include them as formatted text
                isToolCall(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolCall } as? HarmonyContent.ToolCall
                    toolContent?.let {
                        val toolCallText = "[Tool Call: ${it.name}]\nArguments: ${it.arguments}"
                        promptParts.add("<|start_header_id|>assistant<|end_header_id|>\n\n$toolCallText<|eot_id|>")
                    }
                }
                
                // Handle tool responses
                isToolResponse(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolResponse } as? HarmonyContent.ToolResponse
                    toolContent?.let {
                        val toolResponseText = "[Tool Response]\n${it.result}"
                        promptParts.add("<|start_header_id|>user<|end_header_id|>\n\n$toolResponseText<|eot_id|>")
                    }
                }
                
                // Handle regular messages
                else -> {
                    val textContent = msg.extractTextContent()
                    if (textContent.isNotEmpty()) {
                        val headerRole = when (msg.role) {
                            HarmonyRole.USER -> "user"
                            HarmonyRole.ASSISTANT -> "assistant"
                            else -> null
                        }
                        
                        headerRole?.let {
                            promptParts.add("<|start_header_id|>$headerRole<|end_header_id|>\n\n$textContent<|eot_id|>")
                        }
                    }
                }
            }
        }
        
        // Add the assistant header to prompt for response
        promptParts.add("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
        // Map reasoning effort to sampling parameters
        val samplingParams = mapReasoningToSamplingParams(harmonyCore.metadata.reasoning)
        
        return LlamaRequest(
            prompt = promptParts.joinToString(""),
            maxGenLen = harmonyCore.metadata.maxTokens ?: 2048,
            temperature = samplingParams.temperature?.toFloat(),
            topP = samplingParams.topP?.toFloat()
        )
    }
}