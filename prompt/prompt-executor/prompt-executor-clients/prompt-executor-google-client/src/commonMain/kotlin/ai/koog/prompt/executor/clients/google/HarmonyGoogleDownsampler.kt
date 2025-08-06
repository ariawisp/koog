package ai.koog.prompt.executor.clients.google

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.*

/**
 * Downsampler that converts Harmony-format Prompts to Google Gemini API format.
 * 
 * Key differences from OpenAI/Anthropic:
 * - Google uses "parts" within messages (text, function calls, etc.)
 * - System instruction is a separate content object
 * - Function calls are inline parts within messages
 * - Strict role alternation between USER and MODEL
 * - No separate tool message type - tool responses are user messages
 * 
 * Channel mapping:
 * - ANALYSIS → Drop (internal reasoning only, no exposure)
 * - COMMENTARY → Function call/response parts
 * - FINAL → Text parts in user/model messages
 */
public object HarmonyGoogleDownsampler : HarmonyDownsamplerBase<GoogleGenerateRequest>() {
    
    /**
     * Downsample a Harmony Prompt to Google Gemini API format.
     * 
     * IMPORTANT: Analysis channel is completely filtered for safety.
     */
    override fun downsample(prompt: Prompt): GoogleGenerateRequest {
        // Build system instruction from system and developer contexts
        val systemInstruction = buildSystemInstruction(prompt)
        
        // Convert messages with channel filtering
        val contents = downsampleMessages(prompt.conversation)
        
        // Convert tools if present
        val tools = if (prompt.developerContext.tools.isNotEmpty()) {
            listOf(GoogleTool(
                functionDeclarations = prompt.developerContext.tools.map { tool ->
                    val schema = convertToolToJsonSchema(tool)
                    GoogleFunctionDeclaration(
                        name = tool.name,
                        description = tool.description,
                        parameters = GoogleFunctionParameters(
                            type = "object",
                            properties = schema["properties"] as? JsonObject ?: buildJsonObject {},
                            required = (schema["required"] as? JsonArray)?.map { 
                                it.jsonPrimitive.content 
                            } ?: emptyList()
                        )
                    )
                }
            ))
        } else null
        
        // Map reasoning effort to generation config using base class method
        val samplingParams = mapReasoningToSamplingParams(prompt.systemContext.reasoningEffort)
        val generationConfig = GoogleGenerationConfig(
            temperature = samplingParams.temperature,
            topP = samplingParams.topP,
            topK = samplingParams.topK ?: 32,
            candidateCount = 1,
            maxOutputTokens = null,
            stopSequences = null
        )
        
        return GoogleGenerateRequest(
            contents = contents,
            tools = tools,
            systemInstruction = systemInstruction?.let { content ->
                GoogleSystemInstruction(parts = content.parts)
            },
            generationConfig = generationConfig
        )
    }
    
    /**
     * Build system instruction from Harmony contexts.
     */
    private fun buildSystemInstruction(prompt: Prompt): GoogleContent? {
        // Use base class method to build system prompt
        val instructionText = buildSystemPrompt(
            prompt.systemContext,
            prompt.developerContext,
            includeAnalysis = false // Google doesn't support analysis in system instruction
        )
        
        return if (instructionText.isNotEmpty()) {
            GoogleContent(
                parts = listOf(GooglePart.Text(instructionText)),
                role = "model" // System instruction requires a role
            )
        } else null
    }
    
    /**
     * Convert Harmony messages to Google format with strict alternation.
     * Google requires USER and MODEL to alternate strictly.
     */
    private fun downsampleMessages(conversation: ConversationGraph): List<GoogleContent> {
        val contents = mutableListOf<GoogleContent>()
        var lastRole: String? = null
        
        // Use base class method to filter channels for safety
        val visibleMessages = filterUserSafeMessages(conversation.messages)
        
        visibleMessages.forEach { message ->
            when (message.channel) {
                "commentary" -> {
                    // Handle tool calls and responses
                    if (isToolCall(message)) {
                        val toolName = extractToolName(message.recipient!!)
                        
                        // Tool calls are MODEL messages with function call parts
                        val functionCall = GooglePart.FunctionCall(
                            GoogleFunctionCall(
                                name = toolName,
                                args = parseJsonContentSafely(message.extractTextContent()).jsonObject
                            )
                        )
                        
                        if (lastRole == "model") {
                            // Add to existing model message
                            val last = contents.last()
                            contents[contents.lastIndex] = last.copy(
                                parts = last.parts + functionCall
                            )
                        } else {
                            contents.add(GoogleContent(
                                parts = listOf(functionCall),
                                role = "model"
                            ))
                            lastRole = "model"
                        }
                    } else if (isToolResponse(message)) {
                        // Tool response - goes in USER message
                        val functionResponse = GooglePart.FunctionResponse(
                            GoogleFunctionResponse(
                                name = "unknown", // Would need to track from call
                                response = GoogleFunctionResponseContent(
                                    content = message.extractTextContent()
                                )
                            )
                        )
                        
                        if (lastRole == "user") {
                            // Add to existing user message
                            val last = contents.last()
                            contents[contents.lastIndex] = last.copy(
                                parts = last.parts + functionResponse
                            )
                        } else {
                            contents.add(GoogleContent(
                                parts = listOf(functionResponse),
                                role = "user"
                            ))
                            lastRole = "user"
                        }
                    }
                }
                "final", null -> {
                    // Map to appropriate role
                    val role = when (message.author.role) {
                        Role.USER -> "user"
                        Role.ASSISTANT -> "model"
                        else -> if (lastRole == "model") "user" else "model"
                    }
                    
                    val textPart = GooglePart.Text(message.extractTextContent())
                    
                    // Ensure alternation
                    if (role == lastRole && contents.isNotEmpty()) {
                        // Merge with previous message
                        val last = contents.last()
                        contents[contents.lastIndex] = last.copy(
                            parts = last.parts + textPart
                        )
                    } else {
                        contents.add(GoogleContent(
                            parts = listOf(textPart),
                            role = role
                        ))
                        lastRole = role
                    }
                }
                else -> {
                    // Skip other message types
                }
            }
        }
        
        // Ensure we start with USER and end ready for MODEL
        if (contents.isEmpty() || contents.first().role != "user") {
            contents.add(0, GoogleContent(
                parts = listOf(GooglePart.Text("")),
                role = "user"
            ))
        }
        
        return contents
    }
    
    
    
    /**
     * Validate safety - ensure Analysis channel never leaks.
     */
    internal fun validateSafety(prompt: Prompt, downsampled: GoogleGenerateRequest): HarmonyDownsamplerBase.ValidationResult {
        val issues = mutableListOf<String>()
        
        // Use base class validation for channel safety
        val allParts = downsampled.contents.flatMap { it.parts }
        val allText = allParts.filterIsInstance<GooglePart.Text>()
            .joinToString(" ") { it.text }
        
        val validationResult = validateChannelSafety(prompt, allText, shouldIncludeAnalysis = false)
        if (validationResult is HarmonyDownsamplerBase.ValidationResult.Failure) {
            issues.addAll(validationResult.issues)
        }
        
        // Verify alternation
        var lastRole: String? = null
        downsampled.contents.forEach { content ->
            if (content.role == lastRole) {
                issues.add("Role alternation violated: consecutive ${content.role} messages")
            }
            lastRole = content.role
        }
        
        return if (issues.isEmpty()) {
            HarmonyDownsamplerBase.ValidationResult.Success
        } else {
            HarmonyDownsamplerBase.ValidationResult.Failure(issues)
        }
    }
}


