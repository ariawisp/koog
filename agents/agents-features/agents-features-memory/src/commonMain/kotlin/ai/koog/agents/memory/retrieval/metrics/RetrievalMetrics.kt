package ai.koog.agents.memory.retrieval.metrics

import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

/**
 * Interface for collecting retrieval metrics
 */
public interface RetrievalMetricsCollector {
    public fun recordRetrievalLatency(provider: String, latencyMs: Long)
    public fun recordTokenUsage(provider: String, tokensUsed: Int, budget: Int)
    public fun recordResultCount(provider: String, resultCount: Int)
    public fun recordRetrievalScore(provider: String, avgScore: Double)
    public fun recordCacheHit(provider: String)
    public fun recordCacheMiss(provider: String)
}

/**
 * Data class representing retrieval metrics
 */
public data class RetrievalMetrics(
    val provider: String,
    val queryTime: Instant,
    val latencyMs: Long,
    val resultCount: Int,
    val tokensUsed: Int? = null,
    val tokenBudget: Int? = null,
    val avgScore: Double,
    val cacheHit: Boolean = false,
    val metadata: Map<String, Any> = emptyMap()
) {
    public val tokenEfficiency: Double?
        get() = if (tokensUsed != null && tokenBudget != null && tokenBudget > 0) {
            tokensUsed.toDouble() / tokenBudget
        } else null
}

/**
 * Simple in-memory metrics collector
 */
public class InMemoryMetricsCollector : RetrievalMetricsCollector {
    private val metrics = mutableListOf<RetrievalMetrics>()
    private val latencies = mutableMapOf<String, MutableList<Long>>()
    private val tokenUsages = mutableMapOf<String, MutableList<Pair<Int, Int>>>()
    private val scores = mutableMapOf<String, MutableList<Double>>()
    private val cacheStats = mutableMapOf<String, Pair<Int, Int>>() // hits, misses
    
    override fun recordRetrievalLatency(provider: String, latencyMs: Long) {
        latencies.getOrPut(provider) { mutableListOf() }.add(latencyMs)
    }
    
    override fun recordTokenUsage(provider: String, tokensUsed: Int, budget: Int) {
        tokenUsages.getOrPut(provider) { mutableListOf() }.add(tokensUsed to budget)
    }
    
    override fun recordResultCount(provider: String, resultCount: Int) {
        // Tracked as part of full metrics
    }
    
    override fun recordRetrievalScore(provider: String, avgScore: Double) {
        scores.getOrPut(provider) { mutableListOf() }.add(avgScore)
    }
    
    override fun recordCacheHit(provider: String) {
        val currentStats = cacheStats[provider] ?: (0 to 0)
        cacheStats[provider] = (currentStats.first + 1) to currentStats.second
    }
    
    override fun recordCacheMiss(provider: String) {
        val currentStats = cacheStats[provider] ?: (0 to 0)
        cacheStats[provider] = currentStats.first to (currentStats.second + 1)
    }
    
    public fun recordFullMetrics(metrics: RetrievalMetrics) {
        this.metrics.add(metrics)
        recordRetrievalLatency(metrics.provider, metrics.latencyMs)
        if (metrics.tokensUsed != null && metrics.tokenBudget != null) {
            recordTokenUsage(metrics.provider, metrics.tokensUsed, metrics.tokenBudget)
        }
        recordRetrievalScore(metrics.provider, metrics.avgScore)
        if (metrics.cacheHit) {
            recordCacheHit(metrics.provider)
        } else {
            recordCacheMiss(metrics.provider)
        }
    }
    
    public fun getAverageLatency(provider: String): Double? {
        return latencies[provider]?.average()
    }
    
    public fun getAverageTokenEfficiency(provider: String): Double? {
        val usages = tokenUsages[provider] ?: return null
        return usages.map { (used, budget) -> 
            if (budget > 0) used.toDouble() / budget else 0.0 
        }.average()
    }
    
    public fun getCacheHitRate(provider: String): Double? {
        val (hits, misses) = cacheStats[provider] ?: return null
        val total = hits + misses
        return if (total > 0) hits.toDouble() / total else null
    }
    
    public fun getProviderStats(provider: String): ProviderStats {
        return ProviderStats(
            provider = provider,
            avgLatencyMs = getAverageLatency(provider),
            avgTokenEfficiency = getAverageTokenEfficiency(provider),
            avgScore = scores[provider]?.average(),
            cacheHitRate = getCacheHitRate(provider),
            totalQueries = latencies[provider]?.size ?: 0
        )
    }
    
    public fun getAllProviderStats(): List<ProviderStats> {
        val providers = (latencies.keys + tokenUsages.keys + scores.keys + cacheStats.keys).distinct()
        return providers.map { getProviderStats(it) }
    }
}

/**
 * Provider statistics summary
 */
public data class ProviderStats(
    val provider: String,
    val avgLatencyMs: Double?,
    val avgTokenEfficiency: Double?,
    val avgScore: Double?,
    val cacheHitRate: Double?,
    val totalQueries: Int
)

/**
 * Metrics-aware retrieval provider wrapper
 */
public class MetricsAwareRetriever(
    private val baseRetriever: ai.koog.agents.memory.retrieval.RetrievalProvider,
    private val metricsCollector: RetrievalMetricsCollector,
    private val providerName: String = baseRetriever::class.simpleName ?: "Unknown"
) : ai.koog.agents.memory.retrieval.RetrievalProvider {
    
    override fun supports(recipe: ai.koog.agents.memory.retrieval.RetrievalRecipe): Boolean = 
        baseRetriever.supports(recipe)
    
    override suspend fun retrieve(
        query: ai.koog.agents.memory.retrieval.RetrievalQuery,
        securityContext: ai.koog.agents.memory.security.SecurityContext?
    ): List<ai.koog.agents.memory.retrieval.RetrievalResult> {
        val startTime = Clock.System.now()
        
        val results = baseRetriever.retrieve(query, securityContext)
        
        val latencyMs = (Clock.System.now() - startTime).inWholeMilliseconds
        val avgScore = if (results.isNotEmpty()) {
            results.map { it.score }.average()
        } else 0.0
        
        // Record metrics
        metricsCollector.recordRetrievalLatency(providerName, latencyMs)
        metricsCollector.recordResultCount(providerName, results.size)
        metricsCollector.recordRetrievalScore(providerName, avgScore)
        
        // If this is a token-aware query, record token usage
        val tokenAwareQuery = query as? ai.koog.agents.memory.retrieval.TokenAwareRetrievalQuery
        if (tokenAwareQuery != null && metricsCollector is InMemoryMetricsCollector) {
            // Calculate actual token usage from results
            // This would require access to tokenizer, so we'll estimate for now
            val estimatedTokens = results.sumOf { it.content.length / 4 } // Rough estimate
            metricsCollector.recordTokenUsage(
                providerName, 
                estimatedTokens, 
                tokenAwareQuery.tokenBudget
            )
        }
        
        return results
    }
}

/**
 * Extension function to add metrics collection to any retriever
 */
public fun ai.koog.agents.memory.retrieval.RetrievalProvider.withMetrics(
    metricsCollector: RetrievalMetricsCollector,
    providerName: String? = null
): MetricsAwareRetriever {
    return MetricsAwareRetriever(
        this, 
        metricsCollector, 
        providerName ?: this::class.simpleName ?: "Unknown"
    )
}