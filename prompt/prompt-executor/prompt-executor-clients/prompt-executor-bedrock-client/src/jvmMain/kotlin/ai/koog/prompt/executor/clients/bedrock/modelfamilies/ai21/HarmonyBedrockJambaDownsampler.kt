package ai.koog.prompt.executor.clients.bedrock.modelfamilies.ai21

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.HarmonyDownsamplerBase
import ai.koog.prompt.executor.clients.bedrock.bedrockJson
import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonArray
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Harmony-based downsampler for Bedrock AI21 Jamba models.
 * Converts from Harmony format to Bedrock's Jamba API format.
 */
internal object HarmonyBedrockJambaDownsampler : HarmonyDownsamplerBase<JambaRequest>() {

    @OptIn(ExperimentalUuidApi::class)
    override fun downsample(prompt: Prompt): JambaRequest {
        val harmonyCore = prompt.harmony
        val messages = mutableListOf<JambaMessage>()
        
        // Determine if we should include analysis channel based on reasoning effort
        val includeAnalysis = harmonyCore.metadata.reasoning == ReasoningEffort.HIGH
        
        // Build system prompt from context
        val systemPrompt = buildSystemPrompt(
            harmonyCore.systemContext,
            harmonyCore.developerContext,
            includeAnalysis
        )
        
        if (systemPrompt.isNotEmpty()) {
            messages.add(JambaMessage(role = "system", content = systemPrompt))
        }
        
        // Add analysis channel messages as system messages if reasoning is HIGH
        if (includeAnalysis) {
            harmonyCore.messages
                .filter { it.channel == ConversationChannel.ANALYSIS }
                .forEach { msg ->
                    val analysisContent = "[Internal Analysis]\n${msg.extractTextContent()}"
                    messages.add(JambaMessage(role = "system", content = analysisContent))
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
                        val toolCalls = listOf(
                            JambaToolCall(
                                id = it.id ?: Uuid.random().toString(),
                                function = JambaFunctionCall(
                                    name = it.name,
                                    arguments = it.arguments
                                )
                            )
                        )
                        
                        // Check if we can add to existing assistant message
                        val lastMessage = messages.lastOrNull()
                        if (lastMessage?.role == "assistant" && lastMessage.toolCalls != null) {
                            messages[messages.lastIndex] = lastMessage.copy(
                                toolCalls = lastMessage.toolCalls + toolCalls
                            )
                        } else {
                            messages.add(
                                JambaMessage(
                                    role = "assistant",
                                    content = null,
                                    toolCalls = toolCalls
                                )
                            )
                        }
                    }
                }
                
                // Handle tool responses
                isToolResponse(msg) -> {
                    val toolContent = msg.content.firstOrNull { it is HarmonyContent.ToolResponse } as? HarmonyContent.ToolResponse
                    toolContent?.let {
                        messages.add(
                            JambaMessage(
                                role = "tool",
                                content = it.result,
                                toolCallId = it.callId
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
                            messages.add(
                                JambaMessage(
                                    role = role,
                                    content = textContent
                                )
                            )
                        }
                    }
                }
            }
        }
        
        // Convert tools
        val jambaTools = harmonyCore.tools?.map { tool ->
            JambaTool(
                function = JambaFunction(
                    name = tool.name,
                    description = tool.description,
                    parameters = buildJsonObject {
                        put("type", "object")
                        put("properties", convertToolToJsonSchema(tool).get("properties")?.jsonObject ?: buildJsonObject {})
                        if (tool.requiredParams.isNotEmpty()) {
                            putJsonArray("required") {
                                tool.requiredParams.forEach { param ->
                                    add(bedrockJson.encodeToJsonElement(kotlinx.serialization.serializer<String>(), param))
                                }
                            }
                        }
                    }
                )
            )
        }
        
        // Map reasoning effort to sampling parameters
        val samplingParams = mapReasoningToSamplingParams(harmonyCore.metadata.reasoning)
        
        return JambaRequest(
            model = harmonyCore.metadata.modelHint ?: "jamba-1.5-large",
            messages = messages,
            maxTokens = harmonyCore.metadata.maxTokens ?: 4096,
            temperature = samplingParams.temperature?.toFloat(),
            topP = samplingParams.topP?.toFloat(),
            tools = jambaTools?.takeIf { it.isNotEmpty() }
        )
    }
}