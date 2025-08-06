package ai.koog.prompt.executor.clients.harmony

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.flow.Flow

/**
 * Registry that routes LLM requests to the appropriate provider based on model capabilities.
 * 
 * - GPT-OSS models with native Harmony support use HarmonyNativeLLMClient
 * - Standard OpenAI models use OpenAILLMClient with downsampling
 * - Other providers can be added as needed
 */
public class HarmonyProviderRegistry(
    private val openAIApiKey: String? = null,
    private val openAISettings: OpenAIClientSettings = OpenAIClientSettings()
) : LLMClient {
    
    private companion object {
        private val logger = KotlinLogging.logger { }
        
        // Models that have native Harmony support
        private val HARMONY_NATIVE_MODELS = setOf(
            "gpt-oss",
            "gpt-oss-preview",
            "gpt-oss-mini",
            "gpt-4-oss",
            "gpt-4-oss-preview"
        )
        
        // Standard OpenAI models that need downsampling
        private val OPENAI_STANDARD_MODELS = setOf(
            "gpt-4",
            "gpt-4-turbo", 
            "gpt-4-turbo-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "o1-preview",
            "o1-mini"
        )
    }
    
    // Lazy initialization of clients
    private val harmonyNativeClient by lazy {
        HarmonyNativeLLMClient()
    }
    
    private val openAIClient by lazy {
        requireNotNull(openAIApiKey) { "OpenAI API key required for standard models" }
        OpenAILLMClient(openAIApiKey, openAISettings)
    }
    
    /**
     * Route request to appropriate provider based on model.
     */
    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        val client = selectClient(model)
        logger.debug { "Routing ${model.id} to ${client::class.simpleName}" }
        return client.execute(prompt, model, tools)
    }
    
    override fun executeStreaming(
        prompt: Prompt,
        model: LLModel
    ): Flow<String> {
        val client = selectClient(model)
        logger.debug { "Routing streaming ${model.id} to ${client::class.simpleName}" }
        return client.executeStreaming(prompt, model)
    }
    
    override suspend fun executeMultipleChoices(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>
    ): List<Message.Response> {
        val client = selectClient(model)
        logger.debug { "Routing multiple choices ${model.id} to ${client::class.simpleName}" }
        return client.executeMultipleChoices(prompt, model, tools)
    }
    
    /**
     * Select the appropriate client based on model characteristics.
     */
    private fun selectClient(model: LLModel): LLMClient {
        return when {
            // Check if model has native Harmony support
            isHarmonyNativeModel(model) -> harmonyNativeClient
            
            // Check if it's a standard OpenAI model
            isOpenAIStandardModel(model) -> openAIClient
            
            // Default to OpenAI client with downsampling for unknown models
            else -> {
                logger.warn { "Unknown model ${model.id}, defaulting to OpenAI client with downsampling" }
                openAIClient
            }
        }
    }
    
    /**
     * Check if a model has native Harmony support.
     */
    private fun isHarmonyNativeModel(model: LLModel): Boolean {
        return HARMONY_NATIVE_MODELS.any { model.id.contains(it, ignoreCase = true) }
    }
    
    /**
     * Check if a model is a standard OpenAI model.
     */
    private fun isOpenAIStandardModel(model: LLModel): Boolean {
        return OPENAI_STANDARD_MODELS.any { model.id.contains(it, ignoreCase = true) }
    }
    
    /**
     * Get information about which client would be used for a model.
     */
    public fun getProviderInfo(model: LLModel): ProviderInfo {
        return when {
            isHarmonyNativeModel(model) -> ProviderInfo(
                provider = "harmony-native",
                usesTokens = true,
                requiresDownsampling = false,
                description = "Native Harmony format with direct token rendering"
            )
            isOpenAIStandardModel(model) -> ProviderInfo(
                provider = "openai-standard",
                usesTokens = false,
                requiresDownsampling = true,
                description = "Standard OpenAI API with Harmony downsampling"
            )
            else -> ProviderInfo(
                provider = "openai-default",
                usesTokens = false,
                requiresDownsampling = true,
                description = "Unknown model, using OpenAI API with downsampling"
            )
        }
    }
    
    override suspend fun moderate(
        prompt: Prompt,
        model: LLModel
    ): ai.koog.prompt.dsl.ModerationResult {
        val client = selectClient(model)
        return client.moderate(prompt, model)
    }
}

/**
 * Information about which provider handles a model.
 */
public data class ProviderInfo(
    val provider: String,
    val usesTokens: Boolean,
    val requiresDownsampling: Boolean,
    val description: String
)