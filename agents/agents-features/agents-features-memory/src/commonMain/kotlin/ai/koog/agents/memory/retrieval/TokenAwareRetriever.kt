package ai.koog.agents.memory.retrieval

import ai.koog.agents.features.eventHandler.feature.EventHandler
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.prompt.message.RequestMetaInfo
import ai.koog.prompt.message.Attachment
import ai.koog.agents.memory.retrieval.config.TokenAwareRetrievalConfig
import ai.koog.agents.memory.retrieval.metrics.RetrievalMetricsCollector
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlinx.datetime.Clock as KClock

/**
 * Token-aware retrieval provider that manages token budgets for efficient context construction.
 * This is a key component for beating benchmarks by optimizing token usage while maintaining accuracy.
 */
public class TokenAwareRetriever(
    private val baseRetriever: RetrievalProvider,
    private val tokenizer: PromptTokenizer,
    private val config: TokenAwareRetrievalConfig = TokenAwareRetrievalConfig(),
    private val eventHandler: EventHandler? = null,
    private val metricsCollector: RetrievalMetricsCollector? = null
) : RetrievalProvider {
    
    private companion object {
        private val logger = KotlinLogging.logger {}
    }
    
    
    override fun supports(recipe: RetrievalRecipe): Boolean = baseRetriever.supports(recipe)
    
    override suspend fun retrieve(query: RetrievalQuery, securityContext: ai.koog.agents.memory.security.SecurityContext?): List<RetrievalResult> {
        val startTime = KClock.System.now()
        
        // Check if this is a token-aware query wrapper
        val tokenAwareQuery: TokenAwareRetrievalQuery? = null // For now, we'll use extension methods
        val actualQuery = query
        
        // Extract token budget from query or use default
        val tokenBudget = tokenAwareQuery?.tokenBudget ?: config.defaultTokenBudget
        val effectiveBudget = tokenBudget - config.reserveTokensForQuery
        
        logger.debug { 
            "Token-aware retrieval: budget=$tokenBudget, effective=$effectiveBudget, query='${actualQuery.text.take(50)}...'" 
        }
        
        // Get initial results from base retriever
        val candidates = baseRetriever.retrieve(actualQuery, securityContext)
        
        // Apply token-aware filtering and optimization
        val optimizedResults = optimizeResultsForTokenBudget(candidates, effectiveBudget, actualQuery, tokenAwareQuery)
        
        // Record metrics if collector is available
        if (metricsCollector != null) {
            val latencyMs = (KClock.System.now() - startTime).inWholeMilliseconds
            val tokensUsed = optimizedResults.sumOf { result ->
                tokenizer.tokenCountFor(Message.User(result.content, RequestMetaInfo(KClock.System.now())))
            }
            
            metricsCollector.recordRetrievalLatency("TokenAwareRetriever", latencyMs)
            metricsCollector.recordTokenUsage("TokenAwareRetriever", tokensUsed, tokenBudget)
            metricsCollector.recordResultCount("TokenAwareRetriever", optimizedResults.size)
            
            if (optimizedResults.isNotEmpty()) {
                val avgScore = optimizedResults.map { it.score }.average()
                metricsCollector.recordRetrievalScore("TokenAwareRetriever", avgScore)
            }
        }
        
        return optimizedResults
    }
    
    private suspend fun optimizeResultsForTokenBudget(
        candidates: List<RetrievalResult>,
        budget: Int,
        query: RetrievalQuery,
        tokenAwareQuery: TokenAwareRetrievalQuery?
    ): List<RetrievalResult> = coroutineScope {
        if (candidates.isEmpty()) return@coroutineScope emptyList()
        
        // Pre-compute token counts for all candidates
        val tokenizedResults = candidates.map { result ->
            async {
                val tokens = tokenizer.tokenCountFor(Message.User(result.content, RequestMetaInfo(KClock.System.now())))
                TokenizedResult(
                    result = result,
                    tokenCount = tokens,
                    relevancePerToken = if (tokens > 0) result.score / tokens else 0.0
                )
            }
        }.awaitAll()
        
        // Apply optimization strategy based on query type
        val strategy = tokenAwareQuery?.optimizationStrategy 
            ?: OptimizationStrategy.GREEDY
            
        when (strategy) {
            OptimizationStrategy.GREEDY -> greedyOptimization(tokenizedResults, budget)
            OptimizationStrategy.BALANCED -> balancedOptimization(tokenizedResults, budget)
            OptimizationStrategy.PRECISION -> precisionOptimization(tokenizedResults, budget)
        }
    }
    
    private fun greedyOptimization(
        tokenizedResults: List<TokenizedResult>,
        budget: Int
    ): List<RetrievalResult> {
        // Sort by relevance per token (efficiency)
        val sorted = tokenizedResults.sortedByDescending { it.relevancePerToken }
        
        val selected = mutableListOf<RetrievalResult>()
        var usedTokens = 0
        
        for (tokenized in sorted) {
            if (usedTokens + tokenized.tokenCount <= budget) {
                selected.add(tokenized.result)
                usedTokens += tokenized.tokenCount
            } else if (config.enableAdaptiveBudgeting && tokenized.tokenCount < config.minResultTokenSize) {
                // Try to fit small high-value results
                if (usedTokens + tokenized.tokenCount <= budget + config.tokenOverflowTolerance) {
                    selected.add(tokenized.result)
                    logger.trace { "Added small high-value result with overflow: ${tokenized.tokenCount} tokens" }
                    break
                }
            }
        }
        
        // Handle empty selection
        if (selected.isEmpty() && sorted.isNotEmpty()) {
            // If no results fit within budget, return the single most efficient result
            return listOf(sorted.first().result)
        }
        
        return selected
    }
    
    private fun balancedOptimization(
        tokenizedResults: List<TokenizedResult>,
        budget: Int
    ): List<RetrievalResult> {
        // Balance between relevance and diversity
        val selected = mutableListOf<RetrievalResult>()
        var usedTokens = 0
        
        // Group by provenance type for diversity
        val grouped = tokenizedResults.groupBy { 
            it.result.provenance.firstOrNull()?.let { it::class.simpleName } ?: "Unknown"
        }
        
        // Round-robin selection from each group
        var hasMoreResults = true
        var index = 0
        
        while (hasMoreResults && usedTokens < budget) {
            hasMoreResults = false
            
            for ((_, group) in grouped) {
                if (index < group.size) {
                    val tokenized = group[index]
                    if (usedTokens + tokenized.tokenCount <= budget) {
                        selected.add(tokenized.result)
                        usedTokens += tokenized.tokenCount
                        hasMoreResults = true
                    }
                }
            }
            index++
        }
        
        return selected
    }
    
    private fun precisionOptimization(
        tokenizedResults: List<TokenizedResult>,
        budget: Int
    ): List<RetrievalResult> {
        // Focus on highest relevance items only
        val sorted = tokenizedResults.sortedByDescending { it.result.score }
        
        val selected = mutableListOf<RetrievalResult>()
        var usedTokens = 0
        val relevanceThreshold = sorted.firstOrNull()?.result?.score?.times(0.8) ?: 0.0
        
        for (tokenized in sorted) {
            // Only include high-relevance results
            if (tokenized.result.score < relevanceThreshold) break
            
            if (usedTokens + tokenized.tokenCount <= budget) {
                selected.add(tokenized.result)
                usedTokens += tokenized.tokenCount
            }
        }
        
        return selected
    }
    
    /**
     * Helper class to track tokenized results
     */
    private data class TokenizedResult(
        val result: RetrievalResult,
        val tokenCount: Int,
        val relevancePerToken: Double
    )
}

/**
 * Optimization strategies for token budget management
 */
public enum class OptimizationStrategy {
    /**
     * Greedy selection based on relevance per token
     */
    GREEDY,
    
    /**
     * Balance between relevance and diversity
     */
    BALANCED,
    
    /**
     * Focus on highest precision results only
     */
    PRECISION
}

/**
 * Extension to RetrievalQuery to support token budgeting
 */
public data class TokenAwareRetrievalQuery(
    val baseQuery: RetrievalQuery,
    val tokenBudget: Int = 2000,
    val optimizationStrategy: OptimizationStrategy = OptimizationStrategy.GREEDY
) {
    // Delegate to base query for compatibility
    val text: String get() = baseQuery.text
    val centerNode: String? get() = baseQuery.centerNode
    val at: kotlinx.datetime.Instant? get() = baseQuery.at
    val target: RetrievalTarget get() = baseQuery.target
    val k: Int get() = baseQuery.k
    val filters: RetrievalFilters get() = baseQuery.filters
    val recipe: RetrievalRecipe get() = baseQuery.recipe
    val requireCitations: Boolean get() = baseQuery.requireCitations
}

/**
 * Factory function to create token-aware retriever
 */
public fun createTokenAwareRetriever(
    baseRetriever: RetrievalProvider,
    tokenizer: PromptTokenizer,
    config: TokenAwareRetrievalConfig = TokenAwareRetrievalConfig(),
    eventHandler: EventHandler? = null,
    metricsCollector: RetrievalMetricsCollector? = null
): TokenAwareRetriever {
    return TokenAwareRetriever(baseRetriever, tokenizer, config, eventHandler, metricsCollector)
}

/**
 * Extension function to wrap any retriever with token awareness
 */
public fun RetrievalProvider.withTokenAwareness(
    tokenizer: PromptTokenizer,
    config: TokenAwareRetrievalConfig = TokenAwareRetrievalConfig(),
    eventHandler: EventHandler? = null,
    metricsCollector: RetrievalMetricsCollector? = null
): TokenAwareRetriever {
    return TokenAwareRetriever(this, tokenizer, config, eventHandler, metricsCollector)
}

/**
 * Extension function to make a query token-aware
 */
public fun RetrievalQuery.withTokenBudget(
    tokenBudget: Int = 2000,
    optimizationStrategy: OptimizationStrategy = OptimizationStrategy.GREEDY
): TokenAwareRetrievalQuery {
    return TokenAwareRetrievalQuery(this, tokenBudget, optimizationStrategy)
}