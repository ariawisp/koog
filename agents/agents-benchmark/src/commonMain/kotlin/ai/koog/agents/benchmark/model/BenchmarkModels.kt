package ai.koog.agents.benchmark.model

import kotlinx.serialization.Serializable

/**
 * Represents a single question-answer item in a benchmark dataset
 */
@Serializable
public data class QAItem(
    val id: String,
    val question: String,
    val goldAnswer: String,
    // Optional fields for schema-bound datasets (Diffbot/Falkor style)
    val entities: List<String>? = null,
    val relations: List<String>? = null,
    val metadata: Map<String, String>? = null
)

/**
 * Represents a benchmark dataset
 */
@Serializable
public data class Dataset(
    val name: String,
    val items: List<QAItem>
)

/**
 * Configuration for a benchmark run
 */
public data class RunConfig(
    val mode: RetrievalMode,
    val tokenBudget: Int = 1600,
    val maxHops: Int = 3
)

/**
 * Retrieval modes for benchmarking
 */
@Serializable
public enum class RetrievalMode { 
    GRAPH,     // Graph-only retrieval
    HYBRID,    // Combined graph + vector + keyword
    VECTOR     // Vector-only retrieval
}

/**
 * Result from the judge evaluating an answer
 */
public data class JudgeResult(
    val correct: Boolean, 
    val reason: String? = null
)

/**
 * Metrics for a single benchmark sample
 */
@Serializable
public data class SampleMetrics(
    val qaId: String,
    val mode: RetrievalMode,
    val correct: Boolean,
    val latencyMs: Long,
    val tokensUsed: Int
)

/**
 * Summary of a benchmark run
 */
@Serializable
public data class RunSummary(
    val dataset: String,
    val mode: RetrievalMode,
    val accuracy: Double,
    val p95LatencyMs: Long,
    val avgTokens: Int,
    val samples: List<SampleMetrics>
)