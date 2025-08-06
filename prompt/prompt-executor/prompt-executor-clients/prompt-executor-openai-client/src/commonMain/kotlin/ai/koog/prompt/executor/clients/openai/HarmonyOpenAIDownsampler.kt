package ai.koog.prompt.executor.clients.openai

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.*

/**
 * Pure function downsampler from Harmony IR to OpenAI format.
 * 
 * This is the "best approach" where we have Harmony as the canonical IR
 * and OpenAI as a provider backend that receives downsampled data.
 */
public object HarmonyOpenAIDownsampler : HarmonyDownsamplerBase<OpenAIChatRequest>() {
    
    /**
     * Downsample Prompt (which IS HarmonyCore) to OpenAI chat completion request.
     * Pure function with explicit channel mapping rules.
     */
    override fun downsample(prompt: Prompt, model: ai.koog.prompt.llm.LLModel): OpenAIChatRequest {
        return OpenAIChatRequest(
            model = model.id,
            messages = downsampleMessages(prompt.conversation),
            tools = downsampleTools(prompt.developerContext.tools),
            temperature = prompt.metadata.temperature,
            maxTokens = prompt.metadata.maxTokens,
            topP = null, // Use provider defaults
            frequencyPenalty = 0.0,
            presencePenalty = 0.0,
            stream = false
        )
    }
    
    /**
     * Downsample conversation messages with channel mapping.
     * 
     * Channel mapping rules:
     * - ANALYSIS � Drop (never sent to OpenAI, internal reasoning only)
     * - COMMENTARY � Tool messages (tool calls and responses)
     * - FINAL � User/Assistant messages (user-facing content)
     */
    private fun downsampleMessages(conversation: ConversationGraph): List<OpenAIMessage> {
        // Use base class method for channel filtering
        return filterUserSafeMessages(conversation.messages)
            .mapNotNull { message ->
                when (message.channel) {
                    "commentary" -> downsampleCommentaryMessage(message)
                    "final", null -> downsampleFinalMessage(message)
                    else -> null // Drop unknown channels
                }
            }
    }
    
    /**
     * Downsample commentary channel messages (tool calls/responses).
     */
    private fun downsampleCommentaryMessage(message: HarmonyMessage): OpenAIMessage? {
        return when (message.author.role) {
            Role.ASSISTANT -> {
                // Use base class helper methods
                if (isToolCall(message)) {
                    val toolName = extractToolName(message.recipient!!)
                    OpenAIMessageFactory.Assistant(
                        content = null,
                        toolCalls = listOf(
                            OpenAIToolCall(
                                id = generateToolCallId(),
                                type = "function",
                                function = OpenAIFunctionCall(
                                    name = toolName,
                                    arguments = message.extractTextContent()
                                )
                            )
                        )
                    )
                } else null
            }
            Role.TOOL -> {
                // Tool response back to assistant
                OpenAIMessageFactory.Tool(
                    toolCallId = generateToolCallId(), // Would need proper ID tracking
                    content = message.extractTextContent()
                )
            }
            else -> null
        }
    }
    
    /**
     * Downsample final channel messages (user-facing).
     */
    private fun downsampleFinalMessage(message: HarmonyMessage): OpenAIMessage {
        return when (message.author.role) {
            Role.SYSTEM -> {
                val systemContent = message.content.filterIsInstance<HarmonyContent.System>().firstOrNull()
                OpenAIMessageFactory.System(
                    content = systemContent?.modelIdentity ?: message.extractTextContent()
                )
            }
            Role.USER -> OpenAIMessageFactory.User(
                content = listOf(
                    OpenAIContent.Text(message.extractTextContent())
                )
            )
            Role.ASSISTANT -> OpenAIMessageFactory.Assistant(
                content = message.extractTextContent(),
                toolCalls = null
            )
            Role.DEVELOPER -> {
                // Map developer messages to system for OpenAI
                val devContent = message.content.filterIsInstance<HarmonyContent.Developer>().firstOrNull()
                OpenAIMessageFactory.System(
                    content = devContent?.instructions ?: message.extractTextContent()
                )
            }
            else -> OpenAIMessageFactory.User(
                content = listOf(OpenAIContent.Text(message.extractTextContent()))
            )
        }
    }
    
    /**
     * Downsample HarmonyTool definitions to OpenAI tools.
     */
    private fun downsampleTools(tools: List<HarmonyTool>): List<OpenAITool>? {
        if (tools.isEmpty()) return null
        
        return tools.map { harmonyTool ->
            OpenAITool(
                type = "function",
                function = OpenAIFunction(
                    name = harmonyTool.name,
                    description = harmonyTool.description,
                    // Use base class helper for JSON schema conversion
                    parameters = convertToolToJsonSchema(harmonyTool)
                )
            )
        }
    }
    
    // Note: ReasoningEffort is GPT-OSS specific and controls chain-of-thought generation.
    // For non-GPT-OSS models, reasoning effort is simply ignored during downsampling.
    // The actual reasoning control happens in the Harmony format rendering via Rust JNI.
}

/**
 * Extension function for direct downsampling from Prompt.
 */
public fun Prompt.toOpenAI(model: ai.koog.prompt.llm.LLModel): OpenAIChatRequest = 
    HarmonyOpenAIDownsampler.downsample(this, model)