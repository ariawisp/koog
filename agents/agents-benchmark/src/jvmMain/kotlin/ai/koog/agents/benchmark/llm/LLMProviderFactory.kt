package ai.koog.agents.benchmark.llm

import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.clients.anthropic.AnthropicClientSettings
import ai.koog.prompt.executor.clients.anthropic.AnthropicLLMClient
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.koog.prompt.executor.model.PromptExecutor

/**
 * Factory for creating LLM executors based on provider configuration
 */
object LLMProviderFactory {
    
    enum class Provider {
        OPENAI,
        LMSTUDIO,
        ANTHROPIC,
        OPENAI_COMPATIBLE
    }
    
    data class LLMConfig(
        val provider: Provider,
        val apiKey: String? = null,
        val baseUrl: String? = null,
        val model: String? = null,
        val temperature: Double = 0.7,
        val maxTokens: Int = 2000
    )
    
    /**
     * Create an executor based on configuration
     */
    fun createExecutor(config: LLMConfig): PromptExecutor {
        return when (config.provider) {
            Provider.OPENAI -> createOpenAIExecutor(config)
            Provider.LMSTUDIO -> createLMStudioExecutor(config)
            Provider.ANTHROPIC -> createAnthropicExecutor(config)
            Provider.OPENAI_COMPATIBLE -> createOpenAICompatibleExecutor(config)
        }
    }
    
    /**
     * Create from environment variables and CLI options
     */
    fun createFromEnvironment(
        provider: String? = null,
        lmstudioUrl: String? = null,
        model: String? = null
    ): PromptExecutor {
        // Check environment variables
        val openaiKey = System.getenv("OPENAI_API_KEY")
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY")
        val defaultProvider = System.getenv("BENCHMARK_LLM_PROVIDER") ?: "lmstudio"
        
        val selectedProvider = provider ?: defaultProvider
        
        return when (selectedProvider.lowercase()) {
            "openai" -> {
                requireNotNull(openaiKey) { "OPENAI_API_KEY environment variable must be set" }
                createExecutor(LLMConfig(
                    provider = Provider.OPENAI,
                    apiKey = openaiKey,
                    model = model ?: System.getenv("OPENAI_MODEL") ?: "gpt-4o-mini"
                ))
            }
            
            "anthropic" -> {
                requireNotNull(anthropicKey) { "ANTHROPIC_API_KEY environment variable must be set" }
                createExecutor(LLMConfig(
                    provider = Provider.ANTHROPIC,
                    apiKey = anthropicKey,
                    model = model ?: System.getenv("ANTHROPIC_MODEL") ?: "claude-3-sonnet-20240229"
                ))
            }
            
            "lmstudio" -> {
                val url = lmstudioUrl ?: System.getenv("LMSTUDIO_URL") ?: "http://localhost:1234"
                createExecutor(LLMConfig(
                    provider = Provider.LMSTUDIO,
                    baseUrl = url,
                    model = model ?: "local-model"
                ))
            }
            
            else -> throw IllegalArgumentException("Unknown provider: $selectedProvider")
        }
    }
    
    private fun createOpenAIExecutor(config: LLMConfig): PromptExecutor {
        val settings = OpenAIClientSettings(
            baseUrl = "https://api.openai.com"
        )
        val client = OpenAILLMClient(
            apiKey = requireNotNull(config.apiKey) { "API key required for OpenAI" },
            settings = settings
        )
        // TODO: Configure model, temperature, etc. through prompt construction
        return SingleLLMPromptExecutor(client)
    }
    
    private fun createLMStudioExecutor(config: LLMConfig): PromptExecutor {
        val settings = OpenAIClientSettings(
            baseUrl = config.baseUrl ?: "http://localhost:1234"
        )
        val client = OpenAILLMClient(
            apiKey = "lm-studio", // LMStudio doesn't need a real key
            settings = settings
        )
        return SingleLLMPromptExecutor(client)
    }
    
    private fun createAnthropicExecutor(config: LLMConfig): PromptExecutor {
        val settings = AnthropicClientSettings(
            baseUrl = "https://api.anthropic.com"
        )
        val client = AnthropicLLMClient(
            apiKey = requireNotNull(config.apiKey) { "API key required for Anthropic" },
            settings = settings
        )
        return SingleLLMPromptExecutor(client)
    }
    
    private fun createOpenAICompatibleExecutor(config: LLMConfig): PromptExecutor {
        // For services that use OpenAI-compatible APIs (like Together AI, Anyscale, etc.)
        val settings = OpenAIClientSettings(
            baseUrl = requireNotNull(config.baseUrl) { "Base URL required for OpenAI-compatible provider" }
        )
        val client = OpenAILLMClient(
            apiKey = config.apiKey ?: "dummy-key",
            settings = settings
        )
        return SingleLLMPromptExecutor(client)
    }
    
    /**
     * Print available providers and configuration
     */
    fun printAvailableProviders() {
        println("🤖 Available LLM Providers:")
        println("===========================")
        
        // Check OpenAI
        val openaiKey = System.getenv("OPENAI_API_KEY")
        if (openaiKey != null) {
            println("✅ OpenAI (API key found)")
            println("   Models: gpt-4o, gpt-4o-mini, gpt-4-turbo")
        } else {
            println("❌ OpenAI (set OPENAI_API_KEY)")
        }
        
        // Check Anthropic
        val anthropicKey = System.getenv("ANTHROPIC_API_KEY")
        if (anthropicKey != null) {
            println("✅ Anthropic (API key found)")
            println("   Models: claude-3-opus, claude-3-sonnet, claude-3-haiku")
        } else {
            println("❌ Anthropic (set ANTHROPIC_API_KEY)")
        }
        
        // Check LMStudio
        println("✅ LMStudio (always available)")
        println("   Default URL: http://localhost:1234")
        println("   Set LMSTUDIO_URL to override")
        
        println("\n📝 Usage:")
        println("   --provider openai --model gpt-4o")
        println("   --provider anthropic --model claude-3-opus-20240229")
        println("   --provider lmstudio (default)")
    }
}