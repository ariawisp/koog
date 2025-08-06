package ai.koog.prompt.executor.clients.anthropic

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.*

/**
 * Downsampler that converts Harmony-format Prompts to Anthropic API format.
 * 
 * Key differences from OpenAI:
 * - System prompt is a separate field, not a message
 * - Analysis channel can be included in system prompt for hidden reasoning
 * - Tool calls use different format (tool_use blocks)
 * - Messages alternate between user and assistant strictly
 * 
 * Channel mapping:
 * - ANALYSIS → System prompt (hidden context)
 * - COMMENTARY → Tool use blocks
 * - FINAL → User/Assistant messages
 */
internal object HarmonyAnthropicDownsampler : HarmonyDownsamplerBase<AnthropicMessagesRequest>() {
    
    /**
     * Downsample a Harmony Prompt to Anthropic API format.
     * 
     * Anthropic allows including analysis in the system prompt,
     * which provides hidden context without exposing to users.
     */
    override fun downsample(prompt: Prompt, model: ai.koog.prompt.llm.LLModel): AnthropicMessagesRequest {
        // Build system prompt including analysis if configured
        val analysisMessages = extractAnalysisMessages(prompt.conversation.messages)
        val systemPrompt = buildSystemPrompt(
            prompt.systemContext,
            prompt.developerContext,
            shouldIncludeAnalysis(prompt),
            analysisMessages
        )
        
        // Convert messages, ensuring strict user/assistant alternation
        val messages = downsampleMessages(prompt.conversation)
        
        // Map reasoning effort to sampling parameters
        val samplingParams = mapReasoningToSamplingParams(prompt.systemContext.reasoningEffort)
        val temperature = samplingParams.temperature
        val topP = samplingParams.topP
        
        return AnthropicMessagesRequest(
            model = model.id,
            system = systemPrompt,
            messages = messages,
            maxTokens = prompt.metadata.maxTokens ?: 4096,
            temperature = temperature,
            topP = topP,
            topK = null,
            stopSequences = null,
            tools = downsampleTools(prompt.developerContext.tools),
            toolChoice = null,
            metadata = null,
            stream = null
        )
    }
    
    
    /**
     * Convert Harmony messages to Anthropic format.
     * Must maintain strict user/assistant alternation.
     */
    private fun downsampleMessages(conversation: ConversationGraph): List<AnthropicMessage> {
        val messages = mutableListOf<AnthropicMessage>()
        
        // Use base class method to filter channels
        val visibleMessages = filterUserSafeMessages(conversation.messages)
        
        // Process messages maintaining alternation
        var lastRole: String? = null
        
        visibleMessages.forEach { message ->
            when (message.channel) {
                "commentary" -> {
                    // Tool calls in Anthropic format
                    if (isToolCall(message)) {
                        val toolName = extractToolName(message.recipient!!)
                        
                        // Anthropic requires tool calls as assistant messages
                        if (lastRole != "assistant") {
                            val textContent = message.extractTextContent()
                            
                            val toolUseBlock = buildJsonArray {
                                add(buildJsonObject {
                                    put("type", "tool_use")
                                    put("id", generateToolCallId())
                                    put("name", toolName)
                                    put("input", parseJsonContentSafely(textContent))
                                })
                            }
                            
                            messages.add(AnthropicMessage(
                                role = "assistant",
                                content = toolUseBlock
                            ))
                            lastRole = "assistant"
                        }
                    } else if (isToolResponse(message)) {
                        // Tool response
                        val textContent = message.extractTextContent()
                        
                        val toolResultBlock = buildJsonArray {
                            add(buildJsonObject {
                                put("type", "tool_result")
                                put("tool_use_id", generateToolCallId())
                                put("content", textContent)
                            })
                        }
                        
                        messages.add(AnthropicMessage(
                            role = "user",
                            content = toolResultBlock
                        ))
                        lastRole = "user"
                    }
                }
                "final", null -> {
                    val role = when (message.author.role) {
                        Role.USER -> "user"
                        Role.ASSISTANT -> "assistant"
                        else -> if (lastRole == "assistant") "user" else "assistant"
                    }
                    
                    // Create text content block
                    val textContent = message.extractTextContent()
                    
                    val textBlock = buildJsonArray {
                        add(buildJsonObject {
                            put("type", "text")
                            put("text", textContent)
                        })
                    }
                    
                    // Ensure alternation
                    if (role == lastRole) {
                        // Skip or merge with previous
                        if (messages.isNotEmpty()) {
                            val last = messages.last()
                            val existingContent = last.content as? JsonArray ?: buildJsonArray { add(last.content) }
                            val newContent = buildJsonArray {
                                existingContent.forEach { add(it) }
                                add(buildJsonObject {
                                    put("type", "text")
                                    put("text", textContent)
                                })
                            }
                            messages[messages.lastIndex] = last.copy(content = newContent)
                        }
                    } else {
                        messages.add(AnthropicMessage(
                            role = role,
                            content = textBlock
                        ))
                        lastRole = role
                    }
                }
                else -> {
                    // Skip other message types
                }
            }
        }
        
        return messages
    }
    
    /**
     * Convert Harmony tools to Anthropic tool format.
     */
    private fun downsampleTools(tools: List<HarmonyTool>): List<AnthropicTool>? {
        if (tools.isEmpty()) return null
        
        return tools.map { tool ->
            val schema = convertToolToJsonSchema(tool)
            AnthropicTool(
                name = tool.name,
                description = tool.description,
                inputSchema = AnthropicSchema(
                    type = "object",
                    properties = schema["properties"] as? JsonObject ?: buildJsonObject {},
                    required = tool.parameters.required.takeIf { it.isNotEmpty() }
                )
            )
        }
    }
    
    
    /**
     * Validate safety - ensure no unintended leakage.
     */
    internal fun validateSafety(prompt: Prompt, downsampled: AnthropicMessagesRequest): HarmonyDownsamplerBase.ValidationResult {
        // Use base class validation with Anthropic-specific extraction
        val downsampledContent = buildString {
            append(downsampled.system)
            append(" ")
            append(extractContentFromMessages(downsampled.messages))
        }
        
        return validateChannelSafety(prompt, downsampledContent, shouldIncludeAnalysis(prompt))
    }
    
    private fun extractContentFromMessages(messages: List<AnthropicMessage>): String {
        return messages.joinToString(" ") { msg ->
            when (val content = msg.content) {
                is JsonArray -> content.joinToString(" ") { element ->
                    (element as? JsonObject)?.get("text")?.jsonPrimitive?.content ?: ""
                }
                is JsonObject -> content["text"]?.jsonPrimitive?.content ?: ""
                is JsonPrimitive -> content.content
                else -> ""
            }
        }
    }
            
}