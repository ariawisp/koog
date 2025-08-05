package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.core.*
import ai.koog.agents.benchmark.datasets.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import kotlinx.coroutines.runBlocking

/**
 * Example showing how simple benchmarking is now
 */
fun main() = runBlocking {
    
    // 1. Load dataset
    val dataset = LettaBenchLoader().load("")
    
    // 2. Create LLM
    val executor = LLMProviderFactory.createFromEnvironment("openai")
    
    // 3. Run benchmark - that's it!
    val results = dataset.benchmark(executor)
    
    println("Accuracy: ${(results.systemResults.first().accuracy * 100).toInt()}%")
    
    // Or optimize automatically
    val optimal = dataset.optimize(executor)
    println("Best config: ${optimal.bestConfig}")
    println("Best accuracy: ${(optimal.bestAccuracy * 100).toInt()}%")
}

/**
 * Before: 200+ lines of code with 5 different systems
 * After: 10 lines with automatic optimization
 * 
 * Benefits:
 * - Auto-selects best strategy based on dataset
 * - Parallel execution built-in
 * - Automatic retry and error handling
 * - One system to maintain instead of 5
 */