package ai.koog.prompt.dsl

import ai.koog.prompt.harmony.*
import ai.koog.agents.core.tools.ToolDescriptor
import kotlinx.datetime.Clock
import kotlinx.serialization.Serializable

/**
 * Model configuration for Noesis Runtime.
 * Simplified configuration focused on GPT-OSS models.
 */
@Serializable
public data class ModelConfig(
    val model: String = "gpt-oss-20b",
    val temperature: Float = 0.7f,
    val maxTokens: Int = 4096,
    val topP: Float = 1.0f
)

/**
 * Prompt - The canonical Harmony intermediate representation for all LLM operations.
 * 
 * This IS HarmonyCore - the ONLY internal format Koog understands.
 * All requests are processed as Prompt internally, then downsampled
 * to provider-specific formats. This enables semantic understanding, safety by
 * construction, and consistent behavior across all providers.
 */
@Serializable
public data class Prompt(
    val id: String,
    val version: String = "1.0.0",
    val systemContext: SystemContext,
    val developerContext: DeveloperContext,
    val conversation: ConversationGraph,
    val metadata: ModelConfig = ModelConfig()
) {

    /**
     * Access to messages - convenience property
     */
    public val messages: List<HarmonyMessage>
        get() = conversation.messages
    
    /**
     * Native Harmony parameters - alias for metadata
     */
    public val params: ModelConfig
        get() = metadata

    /**
     * Companion object for creating Harmony-native Prompt instances.
     */
    public companion object {
        /**
         * Empty prompt with default values.
         */
        public val Empty: Prompt = Prompt(
            id = "",
            systemContext = SystemContext(),
            developerContext = DeveloperContext.empty(),
            conversation = ConversationGraph.empty(),
            metadata = ModelConfig()
        )

        /**
         * Builds a native Harmony Prompt.
         *
         * @param id The unique identifier for the Prompt.
         * @param model The model ID to use (defaults to gpt-4).
         * @param clock The clock for timestamps.
         * @param init The DSL initialization block.
         */
        public fun build(
            id: String,
            model: String = "gpt-4",
            clock: Clock = Clock.System,
            init: PromptBuilder.() -> Unit
        ): Prompt {
            val builder = PromptBuilder(id, model, clock)
            builder.init()
            return builder.build()
        }
    }

    /**
     * Creates a copy with updated messages.
     */
    public fun withMessages(update: (List<HarmonyMessage>) -> List<HarmonyMessage>): Prompt =
        copy(conversation = conversation.copy(
            messages = update(conversation.messages)
        ))
        
    /**
     * Creates a copy with updated reasoning effort.
     */
    public fun withReasoningEffort(effort: ReasoningEffort): Prompt =
        copy(systemContext = systemContext.copy(reasoningEffort = effort))
        
    /**
     * Creates a copy with updated tools.
     * Converts Koog ToolDescriptors to Harmony format automatically.
     */
    public fun withTools(tools: List<ToolDescriptor>): Prompt =
        copy(developerContext = developerContext.copy(
            tools = HarmonyConverter.fromToolDescriptors(tools)
        ))
    
    /**
     * Extract chain-of-thought reasoning from analysis channel.
     */
    public fun extractChainOfThought(): List<String> = 
        conversation.analysisChannel.map { it.getTextContent() }
    
    /**
     * Enforce channel safety by filtering out analysis content.
     */
    public fun enforceChannelSafety(): Prompt = 
        copy(conversation = conversation.filterUserSafe())
}

/**
 * Native Harmony DSL function for creating prompts.
 */
public fun prompt(
    id: String,
    model: String = "gpt-4", 
    init: PromptBuilder.() -> Unit
): Prompt = Prompt.build(id, model, Clock.System, init)

// Backward compatibility extensions

/**
 * Updates prompt parameters.
 */
public fun Prompt.withUpdatedParams(block: ModelConfig.() -> ModelConfig): Prompt {
    return copy(metadata = metadata.block())
}

/**
 * Updates prompt parameters (alternate name).
 */
public fun Prompt.withParams(block: ModelConfig.() -> ModelConfig): Prompt = 
    withUpdatedParams(block)