package ai.koog.prompt.dsl

import ai.koog.prompt.harmony.*
import ai.koog.prompt.params.LLMParams
import ai.koog.agents.core.tools.ToolDescriptor
import kotlinx.datetime.Clock
import kotlinx.datetime.TimeZone
import kotlinx.datetime.todayIn

/**
 * PromptBuilder - Creates Prompt instances using a fluent DSL.
 *
 * This builder creates Prompt (which IS the Harmony format) directly.
 * All messages are stored as HarmonyMessage internally.
 *
 * Example usage:
 * ```kotlin
 * val prompt = prompt("example-prompt") {
 *     system("You are a helpful assistant.")
 *     user("What is the capital of France?")
 *     reasoning(ReasoningEffort.HIGH)
 * }
 * ```
 *
 * @property id The identifier for the prompt
 * @property model The model ID to use
 * @property clock The clock used for timestamps of messages
 */
@PromptDSL
public class PromptBuilder internal constructor(
    private val id: String,
    private val model: String = "gpt-4",
    private val clock: Clock = Clock.System
) {
    private val harmonyMessages = mutableListOf<HarmonyMessage>()
    private var systemContext = SystemContext()
    private var developerContext = DeveloperContext.empty()
    private var metadata = LLMParams()

    internal companion object {
        internal fun from(prompt: Prompt, clock: Clock = Clock.System): PromptBuilder = PromptBuilder(
            prompt.id,
            "gpt-4", // Default model, actual model is specified at execution time
            clock
        ).apply {
            harmonyMessages.addAll(prompt.conversation.messages)
            systemContext = prompt.systemContext
            developerContext = prompt.developerContext
            metadata = prompt.metadata
        }
    }

    /**
     * Adds a system message with model identity and configuration.
     *
     * This now directly creates HarmonyMessage with structured system content.
     *
     * @param modelIdentity The model identity (defaults to ChatGPT identity)
     * @param knowledgeCutoff Knowledge cutoff date
     * @param currentDate Current date
     * @param reasoningEffort Reasoning effort level
     */
    public fun system(
        modelIdentity: String = "You are ChatGPT, a large language model trained by OpenAI.",
        knowledgeCutoff: String = "2024-06",
        currentDate: String = Clock.System.todayIn(TimeZone.currentSystemDefault()).toString(),
        reasoningEffort: ReasoningEffort = ReasoningEffort.MEDIUM
    ) {
        val systemMessage = HarmonyMessage.system(
            modelIdentity = modelIdentity,
            knowledgeCutoff = knowledgeCutoff,
            currentDate = currentDate,
            reasoningEffort = reasoningEffort
        )
        harmonyMessages.add(systemMessage)
        
        // Update system context
        systemContext = systemContext.copy(
            modelIdentity = modelIdentity,
            knowledgeCutoff = knowledgeCutoff,
            currentDate = currentDate,
            reasoningEffort = reasoningEffort
        )
    }

    /**
     * Set reasoning effort level for the prompt.
     * Note: This is GPT-OSS specific and controls chain-of-thought generation.
     */
    public fun reasoning(effort: ReasoningEffort) {
        systemContext = systemContext.copy(reasoningEffort = effort)
    }
    
    /**
     * Set temperature for generation (standard parameter for all providers).
     */
    public fun temperature(temp: Double) {
        metadata = metadata.copy(temperature = temp)
    }
    
    /**
     * Set max tokens for generation (standard parameter for all providers).
     */
    public fun maxTokens(tokens: Int) {
        metadata = metadata.copy(maxTokens = tokens)
    }
    
    /**
     * Add instructions in developer context.
     */
    public fun instructions(content: String) {
        developerContext = developerContext.copy(instructions = content)
    }
    
    /**
     * Add tools to the developer context.
     */
    public fun tools(tools: List<ToolDescriptor>) {
        developerContext = developerContext.copy(
            tools = HarmonyConverter.fromToolDescriptors(tools)
        )
    }

    /**
     * Adds a user message to the prompt.
     * 
     * Now creates HarmonyMessage directly in the final channel.
     */
    public fun user(content: String) {
        harmonyMessages.add(HarmonyMessage.user(content).withChannel("final"))
    }

    /**
     * Add analysis (chain-of-thought) content.
     * This goes to the analysis channel and is never shown to users.
     */
    public fun analysis(content: String) {
        harmonyMessages.add(HarmonyMessage.assistant(content).withChannel("analysis"))
    }

    /**
     * Add commentary content (tool interactions).
     * This goes to the commentary channel for tool calls and responses.
     */
    public fun commentary(content: String, recipient: String? = null) {
        harmonyMessages.add(
            HarmonyMessage.assistant(content)
                .withChannel("commentary")
                .let { msg -> recipient?.let { msg.withRecipient(it) } ?: msg }
        )
    }

    /**
     * Adds an assistant message to the prompt.
     *
     * Now creates HarmonyMessage directly in the final channel.
     */
    public fun assistant(content: String) {
        harmonyMessages.add(HarmonyMessage.assistant(content).withChannel("final"))
    }

    /**
     * Add a tool call message.
     */
    public fun toolCall(toolName: String, arguments: String) {
        harmonyMessages.add(
            HarmonyMessage.assistant(arguments)
                .withChannel("commentary")
                .withRecipient("functions.$toolName")
        )
    }
    
    /**
     * Add a tool response message.
     */
    public fun toolResponse(toolName: String, output: String) {
        harmonyMessages.add(
            HarmonyMessage.tool(output)
                .withRecipient("assistant")
                .withChannel("commentary")
        )
    }

    /**
     * Add a raw HarmonyMessage.
     */
    public fun harmonyMessage(message: HarmonyMessage) {
        harmonyMessages.add(message)
    }

    /**
     * Add multiple HarmonyMessages.
     */
    public fun harmonyMessages(messages: List<HarmonyMessage>) {
        harmonyMessages.addAll(messages)
    }

    /**
     * Builds and returns a native Harmony Prompt.
     *
     * @return A new Harmony-native Prompt
     */
    internal fun build(): Prompt {
        return Prompt(
            id = id,
            systemContext = systemContext,
            developerContext = developerContext,
            conversation = ConversationGraph(harmonyMessages.toList()),
            metadata = metadata
        )
    }
}