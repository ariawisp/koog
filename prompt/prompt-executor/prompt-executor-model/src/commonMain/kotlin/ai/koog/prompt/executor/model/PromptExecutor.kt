package ai.koog.prompt.executor.model

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.harmony.*
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.agents.core.tools.ToolDescriptor
import kotlinx.coroutines.flow.Flow

/**
 * PromptExecutor - Execution interface for Koog prompts.
 * 
 * Since Prompt IS HarmonyCore internally, this interface works
 * natively with Harmony semantics while maintaining compatibility
 * with existing Koog agents.
 */
public interface PromptExecutor {

    /**
     * Execute a Prompt (which contains HarmonyCore internally).
     * Returns legacy Message.Response for compatibility with agents.
     */
    public suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor> = emptyList()
    ): List<Message.Response>
    
    /**
     * Execute a prompt and return a streaming flow of response chunks.
     */
    public fun executeStreaming(
        prompt: Prompt,
        model: LLModel
    ): Flow<String> = throw UnsupportedOperationException("Streaming not supported by this executor")
    
    /**
     * Execute a prompt and return multiple response choices.
     */
    public suspend fun executeMultipleChoices(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor> = emptyList()
    ): List<Message.Response> = execute(prompt, model, tools)
    
    /**
     * Moderate the content of a prompt for safety.
     */
    public suspend fun moderate(
        prompt: Prompt,
        model: LLModel
    ): ModerationResult = ModerationResult(
        isHarmful = false,
        categories = emptyMap()
    )

    // Future Harmony-native methods (when agents are updated):
    // suspend fun executeHarmony(request: HarmonyCore, model: LLModel): HarmonyResponse
    // suspend fun executeStreamingHarmony(request: HarmonyCore, model: LLModel): Flow<HarmonyDelta>
}

/**
 * Harmony moderation result with channel-aware safety information.
 */
public data class HarmonyModerationResult(
    val flagged: Boolean,
    val categories: Map<String, Boolean>,
    val scores: Map<String, Double>,
    val channelAnalysis: Map<String, ModerationChannelResult>
)

/**
 * Per-channel moderation analysis.
 */
public data class ModerationChannelResult(
    val flagged: Boolean,
    val reason: String?,
    val confidence: Double
)

/**
 * Streaming delta for real-time response processing.
 */
public sealed class HarmonyDelta {
    public data class Content(val text: String, val channel: String?) : HarmonyDelta()
    public data class ToolCall(val call: String) : HarmonyDelta()
    public data object Complete : HarmonyDelta()
    public data class Error(val message: String) : HarmonyDelta()
}