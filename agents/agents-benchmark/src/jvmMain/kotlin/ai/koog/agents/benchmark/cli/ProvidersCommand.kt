package ai.koog.agents.benchmark.cli

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.core.Context
import ai.koog.agents.benchmark.llm.LLMProviderFactory

/**
 * Command to list available LLM providers
 */
class ProvidersCommand : CliktCommand(name = "providers") {
    
    override fun help(context: Context) = "List available LLM providers and their configuration"
    
    override fun run() {
        LLMProviderFactory.printAvailableProviders()
        
        echo("")
        echo("🚀 Quick Start Examples:")
        echo("======================")
        echo("")
        echo("1. Using OpenAI GPT-4:")
        echo("   export OPENAI_API_KEY=sk-...")
        echo("   koog-benchmark run --provider openai --model gpt-4o --size 50")
        echo("")
        echo("2. Using Anthropic Claude:")
        echo("   export ANTHROPIC_API_KEY=sk-ant-...")
        echo("   koog-benchmark run --provider anthropic --model claude-3-opus-20240229")
        echo("")
        echo("3. Using local LMStudio (default):")
        echo("   koog-benchmark run --size 50")
        echo("")
        echo("4. Compare providers:")
        echo("   # Run with LMStudio")
        echo("   koog-benchmark run --size 20 --output lmstudio-results.json")
        echo("   ")
        echo("   # Run with GPT-4")
        echo("   koog-benchmark run --provider openai --model gpt-4o --size 20 --output gpt4-results.json")
        echo("")
        echo("💡 Tip: Set BENCHMARK_LLM_PROVIDER env var to change default provider")
    }
}