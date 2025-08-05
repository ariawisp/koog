package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.core.*
import ai.koog.agents.benchmark.runners.*
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import kotlinx.coroutines.runBlocking

fun main() = runBlocking {
    println("Testing Unified Benchmark System")
    println("================================")
    
    // Check if API key is set
    val apiKey = System.getenv("OPENAI_API_KEY")
    if (apiKey.isNullOrEmpty()) {
        println("❌ OPENAI_API_KEY not set")
        return@runBlocking
    }
    
    // Create a small test dataset
    val testDataset = BenchmarkDataset(
        name = "Test Dataset",
        questions = listOf(
            Question(
                id = "test_1",
                text = "What is 2 + 2?",
                expectedAnswer = "4",
                type = "simple",
                requiredContext = listOf("2 + 2 = 4")
            ),
            Question(
                id = "test_2",
                text = "Who has more items, Alice with 5 or Bob with 3?",
                expectedAnswer = "Alice has more items (5 vs 3)",
                type = "comparison",
                requiredContext = listOf("Alice has 5 items", "Bob has 3 items")
            )
        ),
        contextDescription = "Simple test dataset"
    )
    
    // Create executor
    val executor = simpleOpenAIExecutor(apiKey)
    
    // Test with different strategies
    val strategies = listOf(
        RetrievalStrategy.FULL_CONTEXT,
        RetrievalStrategy.VECTOR,
        RetrievalStrategy.HYBRID_RRF
    )
    
    for (strategy in strategies) {
        println("\n🔧 Testing with strategy: $strategy")
        
        val config = BenchmarkConfig(
            maxConcurrency = 1,
            retrievalStrategies = listOf(strategy),
            numRuns = 1,
            verbose = true
        )
        
        val benchmark = UnifiedBenchmark(config)
        
        try {
            val results = benchmark.run(testDataset, executor)
            
            println("\n📊 Results:")
            println("  Dataset: ${results.dataset}")
            println("  Timestamp: ${results.timestamp}")
            
            results.systemResults.forEach { result ->
                println("  System: ${result.system}")
                println("    Accuracy: ${(result.accuracy * 100).toInt()}%")
                println("    Avg Latency: ${result.avgLatency}ms")
                println("    Avg Tokens: ${result.avgTokens}")
            }
        } catch (e: Exception) {
            println("❌ Error: ${e.message}")
            e.printStackTrace()
        }
    }
    
    println("\n✅ Test completed!")
}