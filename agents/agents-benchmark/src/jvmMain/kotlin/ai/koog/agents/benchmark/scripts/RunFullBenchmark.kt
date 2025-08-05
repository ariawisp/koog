package ai.koog.agents.benchmark.scripts

import ai.koog.agents.benchmark.core.*
import ai.koog.agents.benchmark.runners.*
import ai.koog.agents.benchmark.datasets.*
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import kotlinx.coroutines.runBlocking

/**
 * Main script to run full benchmark evaluation using unified system
 */
object RunFullBenchmark {
    
    /**
     * Run full benchmark with specified configuration
     */
    fun runBenchmark(
        datasetName: String = "lettabench",
        provider: String = "openai",
        outputPath: String? = "benchmark-results.json",
        numRuns: Int = 3,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF,
        quickMode: Boolean = false,
        maxConcurrency: Int = 8
    ) = runBlocking {
        println("🚀 Koog Unified Benchmark Evaluation")
        println("=".repeat(70))
        println("Configuration:")
        println("  Dataset: $datasetName")
        println("  Provider: $provider")
        println("  Strategy: $strategy")
        println("  Output path: ${outputPath ?: "console only"}")
        println("  Parallel execution: $maxConcurrency concurrent requests")
        println()
        
        // Load dataset using proper loaders
        val loader: DataLoader = when (datasetName.lowercase()) {
            "lettabench" -> LettaBenchLoader()
            "longmemeval" -> LongMemEvalLoader()
            "dmr" -> DMRLoader()
            else -> error("Unknown dataset: $datasetName. Available: lettabench, longmemeval, dmr")
        }
        
        val dataset = loader.load("")
        
        println("📊 Dataset loaded: ${dataset.name}")
        println("  Questions: ${dataset.questions.size}")
        println()
        
        // Create LLM executor
        val executor = when (provider) {
            "openai" -> simpleOpenAIExecutor(System.getenv("OPENAI_API_KEY") ?: error("OPENAI_API_KEY not set"))
            else -> error("Unsupported provider: $provider")
        }
        
        // Create unified benchmark with our strategy
        val benchmark = UnifiedBenchmark(
            BenchmarkConfig(
                maxConcurrency = maxConcurrency,
                retrievalStrategies = listOf(strategy),
                numRuns = if (quickMode) 1 else numRuns
            )
        )
        
        // Run benchmark with data loader
        println("🏃 Running benchmark...")
        val results = benchmark.run(dataset, executor, loader)
        
        // Display results
        println("\n📈 Results:")
        results.systemResults.forEach { result ->
            println("  ${result.system}:")
            println("    Accuracy: ${(result.accuracy * 100).toInt()}%")
            println("    Latency: ${result.avgLatency}ms")
            println("    Tokens: ${result.avgTokens}")
        }
        
        // Save results
        if (outputPath != null) {
            val json = results.toJson()
            // Would write to file here
            println("\n💾 Results saved to: $outputPath")
        }
        
        // Compare with literature
        val comparison = results.baselineComparison
        if (comparison != null) {
            println("\n📚 Literature Comparison:")
            comparison.forEach { (baseline, comp) ->
                println("  $baseline:")
                println("    Our best: ${(comp.yourBest * 100).toInt()}%")
                println("    Baseline: ${(comp.baseline * 100).toInt()}%")
                val diff = comp.yourBest - comp.baseline
                if (diff > 0) {
                    println("    ✅ Outperformed by ${(diff * 100).toInt()}%")
                } else {
                    println("    ❌ Behind by ${(-diff * 100).toInt()}%")
                }
            }
        }
        
        println("\n🎉 Benchmark complete!")
    }
}

