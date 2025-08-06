package ai.koog.agents.core.agent.config

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.harmony.*

/**
 * Determines how the tool calls which are present in the prompt, but whose definitions are not present in the request,
 * are converted when sending to the Model.
 *
 * Missing tool definitions usually occur when different sets of tools are used between stages/subgraphs,
 * and the same prompt history is used without compression.
 *
 * @property format Formatter used to convert tool calls
 */
public abstract class MissingToolsConversionStrategy(private val format: ToolCallDescriber) {
    /**
     * Converts a given [Prompt] by applying modifications based on the provided list of [ToolDescriptor].
     *
     * @param prompt The original [Prompt] to be converted.
     * @param tools The list of [ToolDescriptor] used to modify or adapt the [Prompt].
     * @return A new [Prompt] instance with applied changes based on the provided tools.
     */
    public abstract fun convertPrompt(prompt: Prompt, tools: List<ToolDescriptor>): Prompt

    /**
     * Converts the given message by formatting specific types of tool-related messages
     * (tool calls and results in commentary channel) into descriptive messages.
     * If the message is not a tool-related message, it remains unchanged.
     *
     * @param message The input message to be converted.
     * @return The converted message, either modified if it's a tool-related message, or unchanged otherwise.
     */
    public fun convertMessage(message: HarmonyMessage): HarmonyMessage {
        return when {
            // Tool call: commentary channel with recipient starting with "functions."
            message.channel == "commentary" && message.recipient?.startsWith("functions.") == true -> {
                format.describeToolCall(message)
            }
            // Tool result: tool role in commentary channel
            message.author.role == Role.TOOL && message.channel == "commentary" -> {
                format.describeToolResult(message)
            }
            else -> message
        }
    }

    /**
     * Replace all real tool call and response messages with their dumps to the specified format,
     * and use them as plaintext messages.
     */
    public class All(format: ToolCallDescriber) : MissingToolsConversionStrategy(format) {
        override fun convertPrompt(prompt: Prompt, tools: List<ToolDescriptor>): Prompt {
            return prompt.withMessages { messages -> messages.map { convertMessage(it) } }
        }
    }

    /**
     * Replace only missing real tool call and response messages with their dumps to the specified format,
     * and use them as plaintext messages. The tool calls whose definitions are not missing, will be left
     * as real tool calls and responses.
     */
    public class Missing(format: ToolCallDescriber) : MissingToolsConversionStrategy(format) {
        override fun convertPrompt(prompt: Prompt, tools: List<ToolDescriptor>): Prompt {
            val toolNames = tools.map { it.name }
            return prompt.withMessages { messages ->
                messages.map { message ->
                    // Check if this is a tool-related message for a missing tool
                    val isToolCall = message.channel == "commentary" && 
                        message.recipient?.startsWith("functions.") == true
                    val isToolResult = message.author.role == Role.TOOL && 
                        message.channel == "commentary"
                    
                    if (isToolCall || isToolResult) {
                        val toolName = when {
                            isToolCall -> message.recipient?.substringAfter("functions.") ?: ""
                            else -> message.author.name ?: ""
                        }
                        
                        if (toolName !in toolNames) {
                            convertMessage(message)
                        } else {
                            message
                        }
                    } else {
                        message
                    }
                }
            }
        }
    }
}