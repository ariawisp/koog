package ai.koog.agents.benchmark.cli

import com.github.ajalt.clikt.core.*
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.*
import ai.koog.agents.benchmark.datasets.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import ai.koog.agents.benchmark.runners.*
import ai.koog.agents.benchmark.core.UnifiedBenchmark
import ai.koog.agents.benchmark.core.BenchmarkConfig
import ai.koog.agents.benchmark.core.*
import ai.koog.prompt.executor.model.PromptExecutor
import kotlinx.coroutines.runBlocking
import java.io.File

/**
 * Redesigned CLI with native support for all benchmark types
 */
class BenchmarkCli : NoOpCliktCommand(name = "koog-benchmark") {
    
    override fun help(context: Context) = """
        Koog Benchmark Suite - Test memory systems against industry standards
        
        Supports:
        - LongMemEval: Long-term memory over 115k token conversations
        - LettaBench: Multi-hop reasoning with structured data
        - Deep Memory Retrieval (DMR): Multi-session memory evaluation
        - Custom datasets in various formats
    """.trimIndent()
}

/**
 * Run benchmarks with full dataset support
 */
class RunBenchmarkCommand : CliktCommand(
    name = "run"
) {
    override fun help(context: Context) = "Run benchmarks with native dataset support"
    
    // Dataset selection
    private val benchmark by option("-b", "--benchmark", help = "Benchmark type")
        .choice(
            "longmemeval" to BenchmarkType.LONGMEMEVAL,
            "lettabench" to BenchmarkType.LETTABENCH,
            "dmr" to BenchmarkType.DMR,
            "custom" to BenchmarkType.CUSTOM
        )
        .default(BenchmarkType.LETTABENCH)
    
    private val datasetPath by option("-d", "--dataset", help = "Path to dataset (for custom)")
        .file(mustExist = true)
    
    private val subset by option("--subset", help = "Dataset subset (e.g., 'test', 'dev', 'train')")
        .default("test")
    
    // LLM Configuration
    private val provider by option("-p", "--provider", help = "LLM provider")
        .choice("openai", "anthropic", "lmstudio", "ollama")
        .default("lmstudio")
    
    private val model by option("-m", "--model", help = "Model name")
    
    // Benchmark configuration - simplified
    private val optimize by option("--optimize", help = "Find optimal configuration")
        .flag()
    
    private val size by option("-n", "--size", help = "Number of questions")
        .int()
        .default(50)
    
    private val runs by option("-r", "--runs", help = "Number of runs for statistics")
        .int()
        .default(3)
    
    private val parallel by option("--parallel", help = "Parallel execution")
        .int()
        .default(1)
    
    private val output by option("-o", "--output", help = "Output file for results")
        .file()
    
    private val verbose by option("-v", "--verbose", help = "Verbose output")
        .flag()
    
    private val compareBaseline by option("--compare", help = "Compare against baseline")
        .flag(default = true)
    
    override fun run() {
        echo("🚀 Koog Benchmark Suite", err = false)
        echo("=".repeat(60), err = false)
        
        runBlocking {
            // 1. Load appropriate dataset
            val (dataset, dataLoader) = loadBenchmarkDataset()
            
            echo("\n📊 Dataset: ${dataset.name}", err = false)
            echo("   Type: $benchmark", err = false)
            echo("   Questions: ${dataset.questions.size}", err = false)
            echo("   Context: ${dataset.contextDescription}", err = false)
            
            // 2. Create LLM executor
            val executor = createExecutor()
            
            // 3. Run benchmark with unified system
            val results = UnifiedBenchmark(
                BenchmarkConfig(
                    parallel = parallel,
                    runs = runs,
                    questionLimit = size,
                    verbose = verbose,
                    compareBaseline = compareBaseline
                )
            ).run(
                dataset = dataset,
                executor = executor,
                loader = dataLoader
            )
            
            // 5. Save and display results
            displayResults(results)
            output?.let { saveResults(results, it) }
        }
    }
    
    private suspend fun loadBenchmarkDataset(): Pair<BenchmarkDataset, DataLoader> {
        return when (benchmark) {
            BenchmarkType.LONGMEMEVAL -> {
                echo("📚 Loading LongMemEval dataset...", err = false)
                val loader = LongMemEvalLoader()
                val dataset = loader.load(subset)
                dataset to loader
            }
            
            BenchmarkType.LETTABENCH -> {
                echo("📚 Loading LettaBench dataset...", err = false)
                val loader = LettaBenchLoader()
                val dataset = loader.load("") // Uses resources
                dataset to loader
            }
            
            BenchmarkType.DMR -> {
                echo("📚 Loading Deep Memory Retrieval dataset...", err = false)
                val loader = DMRLoader()
                val dataset = loader.load("")
                dataset to loader
            }
            
            BenchmarkType.CUSTOM -> {
                echo("📚 Loading custom dataset from ${datasetPath?.name}...", err = false)
                val loader = CustomDatasetLoader()
                val dataset = loader.load(datasetPath!!.absolutePath)
                dataset to loader
            }
        }
    }
    
    private fun createExecutor(): PromptExecutor {
        echo("\n🤖 Initializing LLM provider: $provider", err = false)
        return LLMProviderFactory.createFromEnvironment(
            provider = provider,
            model = model
        )
    }
    
    // Removed - now handled by UnifiedBenchmark
    
    private fun displayResults(results: BenchmarkResults) {
        echo("\n" + "=".repeat(70), err = false)
        echo("📊 BENCHMARK RESULTS", err = false)
        echo("=".repeat(70), err = false)
        
        // System performance
        echo("\n🏆 System Performance:", err = false)
        echo("System\t\t\tAccuracy\tLatency\t\tTokens", err = false)
        echo("-".repeat(60), err = false)
        
        results.systemResults
            .sortedByDescending { it.accuracy }
            .forEach { result ->
                echo("${result.system.padEnd(20)}\t${(result.accuracy * 100).toInt()}%\t\t${result.avgLatency}ms\t\t${result.avgTokens}", err = false)
            }
        
        // Baseline comparison
        if (results.baselineComparison != null) {
            echo("\n📈 Baseline Comparison:", err = false)
            results.baselineComparison.forEach { (baseline, comparison) ->
                echo("\nvs $baseline:", err = false)
                echo("  Your best: ${(comparison.yourBest * 100).toInt()}%", err = false)
                echo("  Baseline: ${(comparison.baseline * 100).toInt()}%", err = false)
                echo("  ${if (comparison.yourBest > comparison.baseline) "✅ BEATS" else "❌ Below"} baseline by ${((comparison.yourBest - comparison.baseline) * 100).toInt()}%", err = false)
            }
        }
        
        // Recommendations
        echo("\n💡 Recommendations:", err = false)
        if (results.systemResults.any { it.accuracy > 0.7 }) {
            echo("  ✅ Strong performance! Consider submitting to leaderboards.", err = false)
        } else {
            echo("  🔧 Room for improvement. Try:", err = false)
            echo("     - Better prompts", err = false)
            echo("     - Larger context window", err = false)
            echo("     - Different retrieval strategies", err = false)
        }
    }
    
    private fun saveResults(results: BenchmarkResults, file: File) {
        // Save as JSON
        file.writeText(results.toJson())
        echo("\n💾 Results saved to: ${file.absolutePath}", err = false)
    }
}

/**
 * Dataset preparation command
 */
class PrepareDatasetCommand : CliktCommand(
    name = "prepare"
) {
    override fun help(context: Context) = "Download and prepare benchmark datasets"
    
    private val benchmark by option("-b", "--benchmark", help = "Benchmark to prepare")
        .choice("longmemeval", "lettabench", "dmr", "all")
        .default("all")
    
    override fun run() {
        echo("📥 Preparing benchmark datasets...", err = false)
        
        runBlocking {
            when (benchmark) {
                "longmemeval" -> prepareLongMemEval()
                "lettabench" -> prepareLettaBench()
                "dmr" -> prepareDMR()
                "all" -> {
                    prepareLongMemEval()
                    prepareLettaBench()
                    prepareDMR()
                }
            }
        }
    }
    
    private suspend fun prepareLongMemEval() {
        echo("\n📚 LongMemEval:", err = false)
        echo("   Download from: https://github.com/Buki2/LongMemEval", err = false)
        echo("   Or: https://huggingface.co/datasets/xiaowu0162/longmemeval", err = false)
        // Could add auto-download logic here
    }
    
    private suspend fun prepareLettaBench() {
        echo("\n📚 LettaBench:", err = false)
        echo("   ✅ Already included in: agents/agents-benchmark/datasets/letta_file_bench/", err = false)
    }
    
    private suspend fun prepareDMR() {
        echo("\n📚 Deep Memory Retrieval (DMR):", err = false)
        echo("   Generate from: https://github.com/cpacker/MemGPT", err = false)
    }
}

/**
 * Leaderboard submission command
 */
class SubmitCommand : CliktCommand(
    name = "submit"
) {
    override fun help(context: Context) = "Submit results to benchmark leaderboards"
    
    private val resultFile by option("-f", "--file", help = "Results file to submit")
        .file(mustExist = true)
        .required()
    
    private val benchmark by option("-b", "--benchmark", help = "Target benchmark")
        .choice("longmemeval", "lettabench")
        .required()
    
    private val teamName by option("--team", help = "Team name")
        .default("Koog Framework")
    
    private val contact by option("--contact", help = "Contact email")
        .required()
    
    override fun run() {
        echo("📤 Preparing submission...", err = false)
        
        // Load results
        val results = BenchmarkResults.fromJson(resultFile.readText())
        
        // Validate results meet submission criteria
        val bestResult = results.systemResults.maxByOrNull { it.accuracy }
        if (bestResult == null || bestResult.accuracy < 0.5) {
            echo("❌ Results too low for submission (need >50% accuracy)", err = true)
            return
        }
        
        // Generate submission package
        echo("\n📦 Submission Package:", err = false)
        echo("   Benchmark: $benchmark", err = false)
        echo("   Team: $teamName", err = false)
        echo("   Best System: ${bestResult.system}", err = false)
        echo("   Accuracy: ${(bestResult.accuracy * 100).toInt()}%", err = false)
        echo("   Contact: $contact", err = false)
        
        echo("\n📝 Next Steps:", err = false)
        when (benchmark) {
            "longmemeval" -> {
                echo("   1. Fork https://github.com/Buki2/LongMemEval", err = false)
                echo("   2. Add results to leaderboard/", err = false)
                echo("   3. Submit pull request", err = false)
            }
            "lettabench" -> {
                echo("   1. Visit https://docs.letta.com/leaderboard", err = false)
                echo("   2. Follow submission instructions", err = false)
            }
        }
    }
}

// Data types
enum class BenchmarkType {
    LONGMEMEVAL,
    LETTABENCH,
    DMR,
    CUSTOM
}

/**
 * List available resources
 */
class ListCommand : CliktCommand(name = "list") {
    override fun help(context: Context) = "List available datasets and systems"
    
    override fun run() {
        echo("📚 Available Benchmark Resources", err = false)
        echo("=".repeat(40), err = false)
        
        echo("\n🗂️  Datasets:", err = false)
        echo("  - longmemeval   : Long-term memory evaluation (115k tokens)", err = false)
        echo("  - lettabench    : LettaBench multi-hop QA dataset", err = false)
        echo("  - dmr           : Deep Memory Retrieval dataset", err = false)
        echo("  - custom        : Your own dataset (JSONL, JSON, CSV)", err = false)
        
        echo("\n🔧 Memory Systems:", err = false)
        echo("  - graph         : Knowledge graph with relationship tracking", err = false)
        echo("  - vector        : Pure vector similarity search", err = false)
        echo("  - hybrid-rrf    : Reciprocal Rank Fusion hybrid", err = false)
        echo("  - hybrid-node   : Node distance + vector hybrid", err = false)
        echo("  - baseline      : Full context baseline", err = false)
        echo("  - all           : Test all systems", err = false)
        
        echo("\n📊 Baseline Comparisons:", err = false)
        echo("  - Zep (DMR)     : 94.8% accuracy", err = false)
        echo("  - Zep (LongMem) : 71.2% accuracy", err = false)
        echo("  - MemGPT        : 53.0% accuracy", err = false)
        echo("  - RAG           : 42.5% accuracy", err = false)
        
        echo("\n💡 Examples:", err = false)
        echo("  koog-benchmark run --benchmark lettabench --size 50", err = false)
        echo("  koog-benchmark run --benchmark longmemeval --systems hybrid-node,graph", err = false)
        echo("  koog-benchmark prepare --benchmark all", err = false)
    }
}

/**
 * Main entry point for the CLI
 */
fun main(args: Array<String>) = BenchmarkCli()
    .subcommands(
        RunBenchmarkCommand(),
        PrepareDatasetCommand(),
        SubmitCommand(),
        ProvidersCommand(),
        ListCommand()
    )
    .main(args)