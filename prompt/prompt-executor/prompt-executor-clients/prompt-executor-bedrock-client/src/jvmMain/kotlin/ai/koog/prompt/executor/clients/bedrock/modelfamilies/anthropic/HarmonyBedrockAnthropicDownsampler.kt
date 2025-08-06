package ai.koog.prompt.executor.clients.bedrock.modelfamilies.anthropic

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.executor.clients.bedrock.bedrockJson
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Harmony-based downsampler for Bedrock Anthropic models.
 * Converts from Harmony format to Bedrock's Anthropic API format.
 */
internal object HarmonyBedrockAnthropicDownsampler : HarmonyDownsamplerBase<AnthropicMessageRequest>() {

    @OptIn(ExperimentalUuidApi::class)
    override fun downsample(prompt: Prompt): AnthropicMessageRequest {
        val harmonyCore = prompt.harmony
        val messages = mutableListOf<AnthropicMessage>()
        val systemMessages = mutableListOf<SystemAnthropicMessage>()
        
        // Determine if we should include analysis channel based on reasoning effort
        val includeAnalysis = harmonyCore.metadata.reasoning == ReasoningEffort.HIGH
        
        // Build system prompt from context
        val systemPrompt = buildSystemPrompt(
            harmonyCore.systemContext,
            harmonyCore.developerContext,
            includeAnalysis
        )
        
        if (systemPrompt.isNotEmpty()) {
            systemMessages.add(SystemAnthropicMessage(text = systemPrompt))
        }
        
        // Add analysis channel messages if reasoning is HIGH
        if (includeAnalysis) {
            harmonyCore.messages
                .filter { it.channel == ConversationChannel.ANALYSIS }
                .forEach { msg ->
                    val analysisContent = "[Internal Analysis]\n${msg.extractTextContent()}"
                    systemMessages.add(SystemAnthropicMessage(text = analysisContent))
                }
        }
        
        // Process user-safe messages (final and commentary channels)
        val userSafeMessages = filterUserSafeMessages(harmonyCore.messages)
        
        userSafeMessages.forEach { msg ->
            when {
                // Handle tool calls from commentary channel
                isToolCall(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolCall } as? HarmonyContent.ToolCall
                    toolContent?.let {
                        messages.add(
                            AnthropicMessage(
                                role = "assistant",
                                content = listOf(
                                    AnthropicContent.ToolUse(
                                        id = it.id ?: Uuid.random().toString(),
                                        name = it.name,
                                        input = bedrockJson.parseToJsonElement(it.arguments).jsonObject
                                    )
                                )
                            )
                        )
                    }
                }
                
                // Handle tool responses
                isToolResponse(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolResponse } as? HarmonyContent.ToolResponse
                    toolContent?.let {
                        messages.add(
                            AnthropicMessage(
                                role = "user",
                                content = listOf(
                                    AnthropicContent.ToolResult(
                                        toolUseId = it.callId,
                                        content = it.result
                                    )
                                )
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
                            // Handle image content if present
                            val contentParts = mutableListOf<AnthropicContent>()
                            contentParts.add(AnthropicContent.Text(textContent))
                            
                            // Check for image content
                            msg.content.forEach { content ->
                                if (content is HarmonyContent.Image) {
                                    contentParts.add(
                                        AnthropicContent.Image(
                                            source = ImageSource.Base64(
                                                data = content.data,
                                                mediaType = content.mimeType
                                            )
                                        )
                                    )
                                }
                            }
                            
                            messages.add(
                                AnthropicMessage(
                                    role = role,
                                    content = contentParts
                                )
                            )
                        }
                    }
                }
            }
        }
        
        // Convert tools
        val anthropicTools = harmonyCore.tools?.map { tool ->
            AnthropicTool(
                name = tool.name,
                description = tool.description,
                inputSchema = AnthropicToolSchema(
                    properties = convertToolToJsonSchema(tool).get("properties")?.jsonObject ?: buildJsonObject {},
                    required = tool.requiredParams
                )
            )
        }
        
        // Map reasoning effort to sampling parameters
        val samplingParams = mapReasoningToSamplingParams(harmonyCore.metadata.reasoning)
        
        // Determine tool choice based on metadata
        val toolChoice = when {
            anthropicTools.isNullOrEmpty() -> null
            harmonyCore.metadata.forceToolUse -> AnthropicToolChoice.Any
            else -> AnthropicToolChoice.Auto
        }
        
        return AnthropicMessageRequest(
            model = harmonyCore.metadata.modelHint ?: "claude-3-5-sonnet-20241022",
            messages = messages,
            maxTokens = harmonyCore.metadata.maxTokens ?: 4096,
            temperature = samplingParams.temperature?.toFloat(),
            topP = samplingParams.topP?.toFloat(),
            topK = samplingParams.topK,
            system = systemMessages.takeIf { it.isNotEmpty() },
            tools = anthropicTools,
            toolChoice = toolChoice
        )
    }
}