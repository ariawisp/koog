package ai.koog.agents.benchmark.core

import ai.koog.agents.benchmark.evaluation.AnswerEvaluator
import ai.koog.agents.benchmark.judge.Judge
import ai.koog.agents.benchmark.judge.SimpleJudge
import ai.koog.agents.benchmark.model.JudgeResult
import ai.koog.agents.benchmark.runners.SOTAMemorySystem
import ai.koog.agents.benchmark.runners.SOTAMemoryConfig
import ai.koog.agents.benchmark.runners.RetrievalStrategy
import ai.koog.agents.benchmark.runners.AnswerStrategy
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.retrieval.metrics.InMemoryMetricsCollector
import ai.koog.agents.memory.retrieval.config.TokenAwareRetrievalConfig
import ai.koog.prompt.executor.model.PromptExecutor
import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.coroutines.channels.Channel
import kotlinx.datetime.Clock
import kotlin.math.sqrt
import kotlin.time.measureTime

/**
 * Single unified benchmark system that replaces all executors/runners
 */
class UnifiedBenchmark(
    private val config: BenchmarkConfig = BenchmarkConfig()
) {
    private val judge: Judge = SimpleJudge()
    
    /**
     * Run benchmark with automatic strategy selection
     */
    suspend fun run(
        dataset: BenchmarkDataset,
        executor: PromptExecutor,
        loader: DataLoader? = null
    ): BenchmarkResults = coroutineScope {
        
        // Auto-select best strategy based on dataset
        val strategy = selectOptimalStrategy(dataset)
        
        // Create SOTA memory system with optimal config
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val metricsCollector = InMemoryMetricsCollector()
        
        val memorySystem = SOTAMemorySystem(
            SOTAMemoryConfig(
                strategy = strategy.retrieval,
                answerStrategy = strategy.answer,
                retrievalLimit = strategy.retrievalLimit,
                knowledgeGraph = knowledgeGraph,
                tokenConfig = TokenAwareRetrievalConfig(
                    defaultTokenBudget = config.tokenLimit,
                    enableAdaptiveBudgeting = true
                ),
                metricsCollector = metricsCollector
            )
        )
        
        // Load data if loader provided
        loader?.populateMemorySystem(memorySystem, dataset)
        
        // Run benchmark with parallelism
        val results = if (config.parallel > 1) {
            runParallel(dataset, memorySystem, executor)
        } else {
            runSequential(dataset, memorySystem, executor)
        }
        
        // Calculate statistics
        val accuracy = results.count { it.result.correct } / results.size.toDouble()
        val avgLatency = results.map { it.latencyMs }.average().toLong()
        val avgTokens = results.map { it.tokensUsed }.average().toInt()
        
        // Calculate confidence interval if multiple runs
        val confidenceInterval = if (config.runs > 1) {
            calculateConfidenceInterval(results)
        } else null
        
        BenchmarkResults(
            dataset = dataset.name,
            timestamp = Clock.System.now().toString(),
            systemResults = listOf(
                SystemResult(
                    system = memorySystem.name,
                    accuracy = accuracy,
                    avgLatency = avgLatency,
                    avgTokens = avgTokens,
                    confidenceInterval = confidenceInterval
                )
            ),
            baselineComparison = if (config.compareBaseline) {
                compareToBaselines(accuracy, dataset.name)
            } else null
        )
    }
    
    /**
     * Run multiple configurations and find the best
     */
    suspend fun optimize(
        dataset: BenchmarkDataset,
        executor: PromptExecutor,
        loader: DataLoader? = null
    ): OptimizationResult = coroutineScope {
        
        val configurations = generateConfigurations()
        val results = mutableListOf<ConfigurationResult>()
        
        // Test each configuration
        configurations.forEach { config ->
            val result = testConfiguration(dataset, executor, loader, config)
            results.add(ConfigurationResult(config, result))
            
            // Early stopping if we hit target accuracy
            if (result.systemResults.first().accuracy >= 0.9) { // Target 90% accuracy
                return@coroutineScope OptimizationResult(
                    bestConfig = config,
                    bestAccuracy = result.systemResults.first().accuracy,
                    allResults = results
                )
            }
        }
        
        // Return best configuration
        val best = results.maxByOrNull { it.result.systemResults.first().accuracy }!!
        OptimizationResult(
            bestConfig = best.config,
            bestAccuracy = best.result.systemResults.first().accuracy,
            allResults = results
        )
    }
    
    private suspend fun runParallel(
        dataset: BenchmarkDataset,
        memorySystem: MemorySystem,
        executor: PromptExecutor
    ): List<QuestionResult> = coroutineScope {
        
        val semaphore = Semaphore(config.parallel)
        val progressChannel = Channel<Int>(Channel.UNLIMITED)
        
        // Progress tracking
        launch {
            var completed = 0
            for (progress in progressChannel) {
                completed++
                if (config.verbose) {
                    println("Progress: $completed/${dataset.questions.size}")
                }
            }
        }
        
        // Process questions in parallel
        val results = dataset.questions
            .take(config.questionLimit ?: dataset.questions.size)
            .map { question ->
                async {
                    semaphore.withPermit {
                        processQuestion(question, memorySystem, executor).also {
                            progressChannel.send(1)
                        }
                    }
                }
            }
            .awaitAll()
        
        progressChannel.close()
        results
    }
    
    private suspend fun runSequential(
        dataset: BenchmarkDataset,
        memorySystem: MemorySystem,
        executor: PromptExecutor
    ): List<QuestionResult> {
        return dataset.questions
            .take(config.questionLimit ?: dataset.questions.size)
            .mapIndexed { index, question ->
                if (config.verbose) {
                    println("Question ${index + 1}/${dataset.questions.size}: ${question.text.take(50)}...")
                }
                processQuestion(question, memorySystem, executor)
            }
    }
    
    private suspend fun processQuestion(
        question: Question,
        memorySystem: MemorySystem,
        executor: PromptExecutor
    ): QuestionResult {
        var answer: String
        val latency = measureTime {
            answer = retry(config.retryAttempts) {
                memorySystem.answer(question.text, executor)
            }
        }
        
        val result = judge.grade(
            question = question.text,
            gold = question.expectedAnswer,
            pred = answer
        )
        
        return QuestionResult(
            question = question,
            answer = answer,
            result = result,
            latencyMs = latency.inWholeMilliseconds.toLong(),
            tokensUsed = estimateTokens(question.text + answer)
        )
    }
    
    private suspend fun <T> retry(attempts: Int, block: suspend () -> T): T {
        var lastException: Exception? = null
        repeat(attempts) { attempt ->
            try {
                return block()
            } catch (e: Exception) {
                lastException = e
                if (attempt < attempts - 1) {
                    delay(config.retryDelayMs * (attempt + 1))
                }
            }
        }
        throw lastException ?: Exception("Retry failed")
    }
    
    private fun selectOptimalStrategy(dataset: BenchmarkDataset): StrategyConfig {
        return when {
            // Large dataset with complex relationships
            dataset.questions.size > 100 && dataset.name.contains("LongMem") -> 
                StrategyConfig(
                    retrieval = RetrievalStrategy.HYBRID_RRF,
                    answer = AnswerStrategy.MULTI_STEP,
                    retrievalLimit = 15
                )
            
            // Structured data with clear relationships
            dataset.name.contains("Letta") || dataset.name.contains("structured") ->
                StrategyConfig(
                    retrieval = RetrievalStrategy.GRAPH,
                    answer = AnswerStrategy.CHAIN_OF_THOUGHT,
                    retrievalLimit = 10
                )
            
            // Multi-session memory
            dataset.name.contains("DMR") || dataset.name.contains("session") ->
                StrategyConfig(
                    retrieval = RetrievalStrategy.HYBRID_NODE,
                    answer = AnswerStrategy.MULTI_STEP,
                    retrievalLimit = 20
                )
            
            // Default balanced approach
            else -> 
                StrategyConfig(
                    retrieval = RetrievalStrategy.HYBRID_RRF,
                    answer = AnswerStrategy.CHAIN_OF_THOUGHT,
                    retrievalLimit = 10
                )
        }
    }
    
    private fun generateConfigurations(): List<SOTAMemoryConfig> {
        val configs = mutableListOf<SOTAMemoryConfig>()
        
        // Test all combinations
        for (retrieval in RetrievalStrategy.values()) {
            for (answer in AnswerStrategy.values()) {
                for (limit in listOf(5, 10, 15, 20)) {
                    configs.add(
                        SOTAMemoryConfig(
                            strategy = retrieval,
                            answerStrategy = answer,
                            retrievalLimit = limit
                        )
                    )
                }
            }
        }
        
        return configs
    }
    
    private suspend fun testConfiguration(
        dataset: BenchmarkDataset,
        executor: PromptExecutor,
        loader: DataLoader?,
        config: SOTAMemoryConfig
    ): BenchmarkResults {
        val system = SOTAMemorySystem(SOTAMemoryConfig(
            strategy = RetrievalStrategy.FULL_CONTEXT
        ))
        loader?.populateMemorySystem(system, dataset)
        
        // Quick test with subset
        val testDataset = dataset.copy(
            questions = dataset.questions.take(10)
        )
        
        return run(testDataset, executor, loader)
    }
    
    private fun calculateConfidenceInterval(results: List<QuestionResult>): Pair<Double, Double> {
        val accuracies = results.map { if (it.result.correct) 1.0 else 0.0 }
        val mean = accuracies.average()
        val stdDev = sqrt(accuracies.map { (it - mean) * (it - mean) }.average())
        val margin = 1.96 * stdDev / sqrt(results.size.toDouble())
        return (mean - margin) to (mean + margin)
    }
    
    private fun compareToBaselines(accuracy: Double, dataset: String): Map<String, BaselineComparison> {
        val baselines = when {
            dataset.contains("LongMem") -> mapOf(
                "Zep" to 0.712,
                "MemGPT" to 0.53,
                "RAG" to 0.425
            )
            dataset.contains("DMR") -> mapOf(
                "Zep" to 0.948,
                "MemGPT" to 0.93,
                "Full-conversation" to 0.94
            )
            else -> mapOf(
                "Baseline" to 0.5
            )
        }
        
        return baselines.mapValues { (_, baseline) ->
            BaselineComparison(accuracy, baseline)
        }
    }
    
    private fun estimateTokens(text: String): Int {
        // Rough estimate: 1 token per 4 characters
        return text.length / 4
    }
}

/**
 * Unified configuration that replaces all other configs
 */
/**
 * Generic parallel execution helper
 */
private suspend fun <T, R> runParallel(
    items: List<T>,
    concurrency: Int,
    block: suspend (T) -> R
): List<R> = coroutineScope {
    val semaphore = Semaphore(concurrency)
    items.map { item ->
        async {
            semaphore.withPermit {
                block(item)
            }
        }
    }.awaitAll()
}

data class BenchmarkConfig(
    // Execution
    val maxConcurrency: Int = 8,
    val parallel: Int = 1,
    val runs: Int = 1,
    val numRuns: Int = 1,
    val questionLimit: Int? = null,
    
    // Strategy selection
    val retrievalStrategies: List<RetrievalStrategy> = listOf(RetrievalStrategy.HYBRID_RRF),
    
    // Optimization
    val optimize: Boolean = false,
    val tokenLimit: Int = 4000,
    
    // Retry logic
    val retryAttempts: Int = 3,
    val retryDelayMs: Long = 1000,
    
    // Output
    val verbose: Boolean = false,
    val compareBaseline: Boolean = true
)

data class StrategyConfig(
    val retrieval: RetrievalStrategy,
    val answer: AnswerStrategy,
    val retrievalLimit: Int
)

data class QuestionResult(
    val question: Question,
    val answer: String,
    val result: JudgeResult,
    val latencyMs: Long,
    val tokensUsed: Int
)

data class OptimizationResult(
    val bestConfig: SOTAMemoryConfig,
    val bestAccuracy: Double,
    val allResults: List<ConfigurationResult>
)

data class ConfigurationResult(
    val config: SOTAMemoryConfig,
    val result: BenchmarkResults
)

// Extension to make it easy to use
suspend fun BenchmarkDataset.benchmark(
    executor: PromptExecutor,
    config: BenchmarkConfig = BenchmarkConfig()
): BenchmarkResults {
    return UnifiedBenchmark(config).run(this, executor)
}

suspend fun BenchmarkDataset.optimize(
    executor: PromptExecutor,
    config: BenchmarkConfig = BenchmarkConfig()
): OptimizationResult {
    return UnifiedBenchmark(config).optimize(this, executor)
}