package ai.koog.agents.benchmark.scripts

import ai.koog.agents.benchmark.core.*
import ai.koog.agents.benchmark.datasets.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import kotlinx.coroutines.runBlocking

/**
 * Run the FULL benchmark with the actual SOTA system
 * This demonstrates how to properly use Koog's memory infrastructure
 */
suspend fun main() {
    println("🚀 Koog SOTA Memory System - Full Benchmark")
    println("=" * 60)
    println("Using actual KnowledgeGraphRetrievalProvider with:")
    println("  ✅ TokenAwareRetriever for budget optimization")
    println("  ✅ CrossEncoderReranker for precision")
    println("  ✅ TemporalKnowledgeExtractor for time reasoning")
    println("  ✅ SmartRouter for intelligent query routing")
    println("=" * 60)
    
    // Configuration
    val provider = System.getenv("LLM_PROVIDER") ?: "openai"
    val model = System.getenv("LLM_MODEL") ?: "gpt-4o-mini"
    val benchmarkSize = System.getenv("BENCHMARK_SIZE")?.toIntOrNull() ?: 50
    
    println("\n📊 Configuration:")
    println("  Provider: $provider")
    println("  Model: $model")
    println("  Questions: $benchmarkSize")
    
    // Create executor
    val executor = LLMProviderFactory.createFromEnvironment(provider, model)
    
    // Run benchmarks
    val benchmarks = listOf(
        "lettabench" to LettaBenchLoader(),
        "longmemeval" to LongMemEvalLoader(),
        "dmr" to DMRLoader()
    )
    
    for ((name, loader) in benchmarks) {
        println("\n\n📚 Running $name benchmark...")
        println("-" * 50)
        
        try {
            val dataset = loader.load("")
            println("Loaded ${dataset.questions.size} questions")
            
            // Run with unified benchmark
            val config = BenchmarkConfig(
                parallel = 4,
                runs = 1,
                questionLimit = minOf(benchmarkSize, dataset.questions.size),
                verbose = true,
                compareBaseline = true,
                tokenLimit = 1500
            )
            
            val benchmark = UnifiedBenchmark(config)
            val results = benchmark.run(dataset, executor, loader)
            
            // Display results
            displayResults(name, results)
            
        } catch (e: Exception) {
            println("❌ Error running $name: ${e.message}")
            e.printStackTrace()
        }
    }
    
    println("\n\n✅ All benchmarks complete!")
    println("\n💡 Next steps:")
    println("  1. Compare results against published baselines")
    println("  2. Submit to leaderboards if performance exceeds SOTA")
    println("  3. Optimize based on metrics collected")
}

private fun displayResults(benchmarkName: String, results: BenchmarkResults) {
    println("\n📊 Results for $benchmarkName:")
    println("=" * 40)
    
    // System performance
    val bestSystem = results.systemResults.maxByOrNull { it.accuracy }
    if (bestSystem != null) {
        println("Best System: ${bestSystem.system}")
        println("  Accuracy: ${(bestSystem.accuracy * 100).format(1)}%")
        println("  Avg Latency: ${bestSystem.avgLatency}ms")
        println("  Avg Tokens: ${bestSystem.avgTokens}")
        println("  Token Efficiency: ${(bestSystem.avgTokens.toDouble() / 1500 * 100).format(1)}%")
    }
    
    // Baseline comparisons
    if (results.baselineComparison != null) {
        println("\n📈 vs Published Baselines:")
        results.baselineComparison.forEach { (baseline, comparison) ->
            val diff = (comparison.yourBest - comparison.baseline) * 100
            val symbol = if (diff > 0) "✅ +" else "❌ "
            println("  $baseline: ${(comparison.yourBest * 100).format(1)}% (baseline: ${(comparison.baseline * 100).format(1)}%) $symbol${diff.format(1)}%")
        }
    }
    
    // Key insights
    println("\n💡 Insights:")
    val avgAccuracy = results.systemResults.map { it.accuracy }.average()
    when {
        avgAccuracy > 0.9 -> println("  🏆 Exceptional performance! Ready for leaderboard submission.")
        avgAccuracy > 0.7 -> println("  ✅ Strong performance. Minor optimizations could push to SOTA.")
        avgAccuracy > 0.5 -> println("  🔧 Decent baseline. Focus on retrieval quality and prompting.")
        else -> println("  ⚠️  Needs improvement. Check data loading and retrieval.")
    }
}

private fun Double.format(decimals: Int): String = "%.${decimals}f".format(this)
private operator fun String.times(n: Int): String = repeat(n)