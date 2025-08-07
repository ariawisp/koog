package ai.koog.agents.example.simpleapi

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.singleRunStrategy
import ai.koog.agents.example.ApiKeyService
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import ai.koog.prompt.llm.LLModel
import kotlinx.coroutines.runBlocking

/**
 * Basic single-run agent example.
 * 
 * This demonstrates how Koog's native Harmony-based Prompt system works:
 * - Prompt IS HarmonyCore internally - no conversion needed
 * - Multi-channel semantics built into the prompt DSL
 * - Provider independence through downsamplers
 * - Clean integration with existing agent strategies
 */
fun main() = runBlocking {
    println("🎭 Pure Harmony-First Koog")
    println("==========================")
    
    // Create agent with native Harmony prompt support
    val agent = AIAgent(
        executor = simpleOpenAIExecutor(ApiKeyService.openAIApiKey),
        llmModel = LLModel("gpt-4o-mini", "OpenAI"),
        strategy = singleRunStrategy(),
        systemPrompt = "You are an expert code assistant focused on clean, maintainable solutions",
        temperature = 0.7
    )

    println("Enter your coding question:")
    val userInput = readln()

    // Execute with the agent - Prompt DSL uses Harmony internally
    val response = agent.run(userInput)
    
    println("\n🎯 Response:")
    println("=====================================")
    println(response)
    
    println("\n✅ Execution completed!")
    println("🎭 Native Harmony: Prompt DSL uses HarmonyCore internally")
    println("🔒 Provider independence: Harmony downsampled to OpenAI format")
    println("🧠 Clean architecture: No conversion layers needed")
}
