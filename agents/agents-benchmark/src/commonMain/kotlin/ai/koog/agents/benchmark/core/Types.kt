package ai.koog.agents.benchmark.core

import ai.koog.prompt.executor.model.PromptExecutor
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.encodeToString
import kotlinx.serialization.decodeFromString

/**
 * Represents a benchmark dataset
 */
data class BenchmarkDataset(
    val name: String,
    val questions: List<Question>,
    val contextDescription: String,
    val metadata: Map<String, Any> = emptyMap()
)

/**
 * Represents a question in a benchmark
 */
data class Question(
    val id: String,
    val text: String,
    val expectedAnswer: String,
    val type: String,
    val difficulty: String? = null,
    val requiredContext: List<String> = emptyList()
)

/**
 * Interface for loading benchmark datasets
 */
interface DataLoader {
    suspend fun load(path: String): BenchmarkDataset
    suspend fun populateMemorySystem(system: MemorySystem, dataset: BenchmarkDataset)
}

/**
 * Interface for memory system implementations
 */
interface MemorySystem {
    val name: String
    suspend fun addContext(context: String, metadata: Map<String, Any> = emptyMap())
    suspend fun retrieve(query: String): List<String>
    suspend fun answer(question: String, executor: PromptExecutor): String
}

/**
 * Benchmark results
 */
@Serializable
data class BenchmarkResults(
    val dataset: String,
    val timestamp: String,
    val systemResults: List<SystemResult>,
    val baselineComparison: Map<String, BaselineComparison>? = null
) {
    fun toJson(): String = Json.encodeToString(this)
    companion object {
        fun fromJson(json: String): BenchmarkResults = Json.decodeFromString(json)
    }
}

/**
 * Results for a single system
 */
@Serializable
data class SystemResult(
    val system: String,
    val accuracy: Double,
    val avgLatency: Long,
    val avgTokens: Int,
    val confidenceInterval: Pair<Double, Double>? = null
)

/**
 * Baseline comparison
 */
@Serializable
data class BaselineComparison(
    val yourBest: Double,
    val baseline: Double
)