package ai.koog.prompt.executor.clients

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.harmony.*
import kotlinx.datetime.Clock
import kotlinx.serialization.json.*

/**
 * Base class for Harmony downsamplers that provides common functionality
 * for converting Harmony Prompts to provider-specific formats.
 * 
 * Common patterns extracted:
 * - Channel filtering (analysis channel safety)
 * - Text content extraction from HarmonyContent
 * - System/Developer context building
 * - Tool parameter conversion
 * - Reasoning effort mapping
 * - Safety validation
 */
public abstract class HarmonyDownsamplerBase<T> {
    
    /**
     * Main downsampling method that each provider must implement.
     */
    public abstract fun downsample(prompt: Prompt): T
    
    /**
     * Extract text content from a HarmonyMessage.
     * Common pattern used by all downsamplers.
     */
    protected fun HarmonyMessage.extractTextContent(): String {
        return content
            .filterIsInstance<HarmonyContent.Text>()
            .joinToString(" ") { it.text }
    }
    
    /**
     * Extract text content from a list of HarmonyContent.
     */
    protected fun List<HarmonyContent>.extractText(): String {
        return filterIsInstance<HarmonyContent.Text>()
            .joinToString(" ") { it.text }
    }
    
    /**
     * Filter messages by channel, ensuring analysis channel is never exposed.
     * This is a critical safety feature common to all downsamplers.
     */
    protected fun filterUserSafeMessages(messages: List<HarmonyMessage>): List<HarmonyMessage> {
        return messages.filter { it.channel != "analysis" }
    }
    
    /**
     * Extract analysis channel messages for providers that support hidden reasoning.
     */
    protected fun extractAnalysisMessages(messages: List<HarmonyMessage>): List<HarmonyMessage> {
        return messages.filter { it.channel == "analysis" }
    }
    
    /**
     * Build a system prompt from SystemContext and DeveloperContext.
     * Common pattern with slight variations per provider.
     */
    protected fun buildSystemPrompt(
        systemContext: SystemContext,
        developerContext: DeveloperContext,
        includeAnalysis: Boolean = false,
        analysisMessages: List<HarmonyMessage> = emptyList()
    ): String {
        return buildString {
            // Identity
            append(systemContext.modelIdentity)
            
            // Knowledge cutoff
            if (systemContext.knowledgeCutoff.isNotEmpty()) {
                append("\nKnowledge cutoff: ${systemContext.knowledgeCutoff}")
            }
            
            // Current date
            if (systemContext.currentDate.isNotEmpty()) {
                append("\nCurrent date: ${systemContext.currentDate}")
            }
            
            // Developer instructions
            if (developerContext.instructions.isNotEmpty()) {
                append("\n\n")
                append(developerContext.instructions)
            }
            
            // Analysis channel (if supported by provider)
            if (includeAnalysis && analysisMessages.isNotEmpty()) {
                append("\n\nInternal reasoning context:")
                analysisMessages.forEach { msg ->
                    append("\n- ${msg.extractTextContent()}")
                }
            }
        }
    }
    
    /**
     * Convert HarmonyTool to JSON representation for tool schemas.
     * Most providers use similar JSON schema format.
     */
    protected fun convertToolToJsonSchema(tool: HarmonyTool): JsonObject {
        return buildJsonObject {
            put("type", "object")
            put("properties", buildJsonObject {
                tool.parameters.properties.forEach { (name, prop) ->
                    put(name, buildJsonObject {
                        put("type", prop.type)
                        put("description", prop.description)
                        prop.enum?.let { enumList ->
                            putJsonArray("enum") {
                                enumList.forEach { add(it) }
                            }
                        }
                    })
                }
            })
            if (tool.parameters.required.isNotEmpty()) {
                putJsonArray("required") {
                    tool.parameters.required.forEach { add(it) }
                }
            }
        }
    }
    
    /**
     * Map reasoning effort to temperature/top_p parameters.
     * Base implementation that can be overridden per provider.
     */
    protected open fun mapReasoningToSamplingParams(effort: ReasoningEffort): SamplingParameters {
        return when (effort) {
            ReasoningEffort.LOW -> SamplingParameters(
                temperature = 1.0,
                topP = 0.95
            )
            ReasoningEffort.MEDIUM -> SamplingParameters(
                temperature = 0.7,
                topP = 0.9
            )
            ReasoningEffort.HIGH -> SamplingParameters(
                temperature = 0.3,
                topP = 0.8
            )
        }
    }
    
    /**
     * Check if a message is a tool call based on recipient pattern.
     */
    protected fun isToolCall(message: HarmonyMessage): Boolean {
        return message.recipient?.startsWith("functions.") == true
    }
    
    /**
     * Extract tool name from recipient.
     */
    protected fun extractToolName(recipient: String): String {
        return recipient.removePrefix("functions.")
    }
    
    /**
     * Check if a message is a tool response.
     */
    protected fun isToolResponse(message: HarmonyMessage): Boolean {
        return message.recipient == "assistant" && message.author.role == Role.TOOL
    }
    
    /**
     * Parse JSON content safely, with fallback to wrapping as string.
     */
    protected fun parseJsonContentSafely(content: String): JsonElement {
        return try {
            Json.parseToJsonElement(content)
        } catch (e: Exception) {
            // If not valid JSON, wrap as object with input field
            buildJsonObject {
                put("input", content)
            }
        }
    }
    
    /**
     * Validate that analysis channel content doesn't leak into the downsampled output.
     * This is a critical safety check.
     */
    protected fun validateChannelSafety(
        originalPrompt: Prompt,
        downsampledContent: String,
        shouldIncludeAnalysis: Boolean = false
    ): ValidationResult {
        if (shouldIncludeAnalysis) {
            return ValidationResult.Success
        }
        
        val analysisContent = originalPrompt.conversation.messages
            .filter { it.channel == "analysis" }
            .map { it.extractTextContent() }
        
        val leaks = analysisContent.filter { analysis ->
            analysis.isNotEmpty() && downsampledContent.contains(analysis)
        }
        
        return if (leaks.isEmpty()) {
            ValidationResult.Success
        } else {
            ValidationResult.Failure(
                leaks.map { "Analysis content leaked: '${it.take(50)}...'" }
            )
        }
    }
    
    /**
     * Common data class for sampling parameters.
     */
    public data class SamplingParameters(
        val temperature: Double,
        val topP: Double,
        val topK: Int? = null
    )
    
    /**
     * Validation result for safety checks.
     */
    public sealed class ValidationResult {
        public object Success : ValidationResult()
        public data class Failure(val issues: List<String>) : ValidationResult()
    }
    
    /**
     * Generate a unique ID for tool calls.
     * Can be overridden for provider-specific formats.
     */
    protected open fun generateToolCallId(): String {
        return "tool_${kotlinx.datetime.Clock.System.now().toEpochMilliseconds()}"
    }
    
    /**
     * Check if reasoning effort should include analysis channel.
     * Base implementation - can be overridden per provider.
     */
    protected open fun shouldIncludeAnalysis(prompt: Prompt): Boolean {
        // By default, only include for HIGH reasoning effort
        // Providers like Anthropic that support hidden system prompts can use this
        return prompt.systemContext.reasoningEffort == ReasoningEffort.HIGH
    }
}