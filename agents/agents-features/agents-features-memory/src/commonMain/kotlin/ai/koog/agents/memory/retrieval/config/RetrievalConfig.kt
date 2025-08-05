package ai.koog.agents.memory.retrieval.config

import kotlinx.serialization.Serializable

/**
 * Base configuration for retrieval components
 */
@Serializable
public abstract class RetrievalConfig {
    public abstract val enabled: Boolean
}

/**
 * Configuration for token-aware retrieval
 */
@Serializable
public data class TokenAwareRetrievalConfig(
    override val enabled: Boolean = true,
    val defaultTokenBudget: Int = 2000,
    val minResultTokenSize: Int = 100,
    val enableAdaptiveBudgeting: Boolean = true,
    val reserveTokensForQuery: Int = 100,
    val tokenOverflowTolerance: Int = 50
) : RetrievalConfig()

/**
 * Configuration for cross-encoder reranking
 */
@Serializable
public data class CrossEncoderConfig(
    override val enabled: Boolean = true,
    val maxConcurrency: Int = 4,
    val batchSize: Int = 10,
    val defaultScoreThreshold: Double = 0.5,
    val maxResultsToRerank: Int = 30,
    val enableRetry: Boolean = true,
    val maxRetries: Int = 2,
    val useExplanationsByDefault: Boolean = false
) : RetrievalConfig()

/**
 * Configuration for retrieval metrics collection
 */
@Serializable
public data class RetrievalMetricsConfig(
    override val enabled: Boolean = true,
    val trackLatency: Boolean = true,
    val trackTokenUsage: Boolean = true,
    val trackCacheHits: Boolean = true,
    val metricsWindowSize: Int = 10000,
    val enableDetailedLogging: Boolean = false
) : RetrievalConfig()

/**
 * Configuration for smart routing
 */
@Serializable
public data class SmartRouterConfig(
    override val enabled: Boolean = true,
    val enableGraphRouting: Boolean = true,
    val temporalRequiresGraph: Boolean = true,
    val maxLatencyMs: Int = 300,
    val multiHopPatterns: List<String> = listOf(
        "who.*(now|yesterday|last week|owned|owns)",
        "(closest|nearest)",
        "(allowed|permission)",
        "(attacked|raided|betrayed)"
    ),
    val fallbackOnError: Boolean = true
) : RetrievalConfig()

/**
 * Master configuration for the entire retrieval system
 */
@Serializable
public data class MemoryRetrievalSystemConfig(
    val tokenAwareRetrieval: TokenAwareRetrievalConfig = TokenAwareRetrievalConfig(),
    val crossEncoder: CrossEncoderConfig = CrossEncoderConfig(),
    val metrics: RetrievalMetricsConfig = RetrievalMetricsConfig(),
    val smartRouter: SmartRouterConfig = SmartRouterConfig(),
    val parallelRetrievalEnabled: Boolean = false,
    val maxParallelRequests: Int = 16
) {
    public companion object {
        /**
         * Default configuration optimized for accuracy
         */
        public fun accuracyOptimized(): MemoryRetrievalSystemConfig = MemoryRetrievalSystemConfig(
            tokenAwareRetrieval = TokenAwareRetrievalConfig(
                defaultTokenBudget = 3000,
                enableAdaptiveBudgeting = true
            ),
            crossEncoder = CrossEncoderConfig(
                enabled = true,
                maxResultsToRerank = 50,
                defaultScoreThreshold = 0.7
            )
        )
        
        /**
         * Configuration optimized for speed
         */
        public fun speedOptimized(): MemoryRetrievalSystemConfig = MemoryRetrievalSystemConfig(
            tokenAwareRetrieval = TokenAwareRetrievalConfig(
                defaultTokenBudget = 1000,
                enableAdaptiveBudgeting = false
            ),
            crossEncoder = CrossEncoderConfig(
                enabled = false
            ),
            smartRouter = SmartRouterConfig(
                maxLatencyMs = 100
            ),
            parallelRetrievalEnabled = true
        )
        
        /**
         * Configuration optimized for token efficiency
         */
        public fun tokenEfficiencyOptimized(): MemoryRetrievalSystemConfig = MemoryRetrievalSystemConfig(
            tokenAwareRetrieval = TokenAwareRetrievalConfig(
                defaultTokenBudget = 500,
                minResultTokenSize = 50,
                reserveTokensForQuery = 50
            ),
            crossEncoder = CrossEncoderConfig(
                enabled = true,
                maxResultsToRerank = 10
            )
        )
    }
}