package ai.koog.prompt.executor.clients.bedrock

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.anthropic.AnthropicMessageRequest
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.ai21.BedrockAI21JambaConverter
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.ai21.JambaRequest
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.amazon.BedrockAmazonNovaConverter
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.amazon.NovaRequest
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.anthropic.BedrockAnthropicClaudeConverter
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.meta.BedrockMetaLlamaConverter
import ai.koog.prompt.executor.clients.bedrock.modelfamilies.meta.LlamaRequest
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import kotlinx.datetime.Clock

/**
 * Converter for Bedrock provider that routes to the correct model family based on model ID
 * Uses proper kotlinx.serialization instead of the old ProviderSerializer pattern
 */
internal object BedrockConverter {
    private val clock: Clock = Clock.System

    /**
     * Convert a prompt to Bedrock request format based on the model family
     */
    fun toBedrockRequest(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): String {
        val modelFamily = getBedrockModelFamily(model)
        
        return when (modelFamily) {
            is BedrockModelFamilies.AI21Jamba -> {
                val request = BedrockAI21JambaConverter.createJambaRequest(prompt, model, tools)
                bedrockJson.encodeToString(JambaRequest.serializer(), request)
            }

            is BedrockModelFamilies.AmazonNova -> {
                val request = BedrockAmazonNovaConverter.createNovaRequest(prompt, model)
                bedrockJson.encodeToString(NovaRequest.serializer(), request)
            }

            is BedrockModelFamilies.AnthropicClaude -> {
                val request = BedrockAnthropicClaudeConverter.createAnthropicRequest(prompt, model, tools)
                bedrockJson.encodeToString(AnthropicMessageRequest.serializer(), request)
            }

            is BedrockModelFamilies.Meta -> {
                val request = BedrockMetaLlamaConverter.createLlamaRequest(prompt, model)
                bedrockJson.encodeToString(LlamaRequest.serializer(), request)
            }
        }
    }

    /**
     * Parse Bedrock response based on the model family
     */
    fun fromBedrockResponse(
        responseBody: String,
        model: LLModel
    ): List<Message.Response> {
        val modelFamily = getBedrockModelFamily(model)
        
        return when (modelFamily) {
            is BedrockModelFamilies.AI21Jamba -> BedrockAI21JambaConverter.parseJambaResponse(
                responseBody,
                clock
            )

            is BedrockModelFamilies.AmazonNova -> BedrockAmazonNovaConverter.parseNovaResponse(
                responseBody,
                clock
            )

            is BedrockModelFamilies.AnthropicClaude -> BedrockAnthropicClaudeConverter.parseAnthropicResponse(
                responseBody,
                clock
            )

            is BedrockModelFamilies.Meta -> BedrockMetaLlamaConverter.parseLlamaResponse(
                responseBody,
                clock
            )
        }
    }

    /**
     * Parse streaming response chunk based on the model family
     */
    fun parseStreamChunk(
        chunkJsonString: String,
        model: LLModel
    ): String {
        val modelFamily = getBedrockModelFamily(model)
        
        return when (modelFamily) {
            is BedrockModelFamilies.AI21Jamba -> BedrockAI21JambaConverter.parseJambaStreamChunk(
                chunkJsonString
            )

            is BedrockModelFamilies.AmazonNova -> BedrockAmazonNovaConverter.parseNovaStreamChunk(
                chunkJsonString
            )

            is BedrockModelFamilies.AnthropicClaude -> BedrockAnthropicClaudeConverter.parseAnthropicStreamChunk(
                chunkJsonString
            )

            is BedrockModelFamilies.Meta -> BedrockMetaLlamaConverter.parseLlamaStreamChunk(chunkJsonString)
        }
    }

    /**
     * Determine the Bedrock model family based on the model ID
     */
    private fun getBedrockModelFamily(model: LLModel): BedrockModelFamilies {
        return when {
            model.id.startsWith("anthropic.claude") -> BedrockModelFamilies.AnthropicClaude
            model.id.startsWith("amazon.nova") -> BedrockModelFamilies.AmazonNova
            model.id.startsWith("ai21.jamba") -> BedrockModelFamilies.AI21Jamba
            model.id.startsWith("meta.llama") -> BedrockModelFamilies.Meta
            else -> throw IllegalArgumentException("Model ${model.id} is not a supported Bedrock model")
        }
    }
}