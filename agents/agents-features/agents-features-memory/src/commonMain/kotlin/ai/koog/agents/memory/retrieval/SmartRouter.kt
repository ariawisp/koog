package ai.koog.agents.memory.retrieval

import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.agents.memory.retrieval.config.TokenAwareRetrievalConfig
import ai.koog.agents.memory.retrieval.metrics.RetrievalMetricsCollector
import io.github.oshai.kotlinlogging.KotlinLogging

/**
 * Routing policy for the smart retrieval router
 */
public data class RoutingPolicy(
    val enableGraph: Boolean = true,
    val temporalRequiresGraph: Boolean = true,
    val maxLatencyMs: Int = 300,
    val detectMultiHop: (String) -> Boolean = { text ->
        // Simple heuristic for detecting multi-hop queries
        Regex(
            pattern = """(who.*(now|yesterday|last week|owned|owns))|(closest|nearest)|(allowed|permission)|(attacked|raided|betrayed)""", 
            option = RegexOption.IGNORE_CASE
        ).containsMatchIn(text)
    }
)

/**
 * Smart router that automatically selects the best retrieval provider
 * based on query characteristics and availability.
 * 
 * This provides a single API surface while routing to vector search,
 * graph traversal, or future retrieval mechanisms as appropriate.
 */
public class SmartRouter(
    private val providers: List<RetrievalProvider>,
    private val policy: RoutingPolicy = RoutingPolicy(),
    private val tokenizer: PromptTokenizer? = null,
    private val tokenConfig: TokenAwareRetrievalConfig? = null,
    private val metricsCollector: RetrievalMetricsCollector? = null
) : RetrievalProvider {
    
    private val logger = KotlinLogging.logger {}
    
    init {
        require(providers.isNotEmpty()) { "At least one provider must be configured" }
    }
    
    // Wrap providers with token awareness if configured
    private val effectiveProviders: List<RetrievalProvider> = if (tokenizer != null && tokenConfig != null) {
        providers.map { provider ->
            provider.withTokenAwareness(
                tokenizer = tokenizer,
                config = tokenConfig,
                metricsCollector = metricsCollector
            )
        }
    } else {
        providers
    }
    
    override fun supports(recipe: RetrievalRecipe): Boolean = true // Router can handle any recipe
    
    override suspend fun retrieve(query: RetrievalQuery, securityContext: ai.koog.agents.memory.security.SecurityContext?): List<RetrievalResult> {
        // Select the best provider based on query characteristics
        val selectedProvider = selectProvider(query)
        
        logger.debug { 
            "Routing query to ${selectedProvider::class.simpleName}: " +
            "temporal=${query.at != null}, centerNode=${query.centerNode != null}, " +
            "recipe=${query.recipe}"
        }
        
        // Try primary provider, fall back to others if it fails
        return try {
            selectedProvider.retrieve(query, securityContext)
        } catch (e: Exception) {
            logger.warn(e) { "Primary provider failed, trying fallback" }
            fallbackRetrieve(query, selectedProvider, securityContext)
        }
    }
    
    private fun selectProvider(query: RetrievalQuery): RetrievalProvider {
        // Check if a specific recipe is requested and a provider supports it
        val recipeProvider = effectiveProviders.firstOrNull { it.supports(query.recipe) }
        if (recipeProvider != null) {
            return recipeProvider
        }
        
        // Apply heuristics to select the best provider
        val isTemporal = query.at != null
        val hasAnchor = query.centerNode != null
        val isMultiHop = policy.detectMultiHop(query.text)
        
        // Prefer graph provider for temporal, anchored, or multi-hop queries
        val needsGraph = (isTemporal && policy.temporalRequiresGraph) || hasAnchor || isMultiHop
        
        if (needsGraph && policy.enableGraph) {
            // Look for a graph-capable provider
            val graphProvider = effectiveProviders.firstOrNull { 
                it.supports(RetrievalRecipe.HYBRID_NODE_DISTANCE) 
            }
            if (graphProvider != null) {
                return graphProvider
            }
        }
        
        // Default to first available provider
        return effectiveProviders.first()
    }
    
    private suspend fun fallbackRetrieve(
        query: RetrievalQuery, 
        failedProvider: RetrievalProvider,
        securityContext: ai.koog.agents.memory.security.SecurityContext?
    ): List<RetrievalResult> {
        // Try other providers
        for (provider in effectiveProviders) {
            if (provider != failedProvider) {
                try {
                    return provider.retrieve(query, securityContext)
                } catch (e: Exception) {
                    logger.warn(e) { "Fallback provider ${provider::class.simpleName} also failed" }
                }
            }
        }
        
        // All providers failed
        logger.error { "All retrieval providers failed for query: ${query.text}" }
        return emptyList()
    }
}

/**
 * Convenience function to create a smart router with vector provider as default
 */
public fun createSmartRouter(
    memoryProvider: ai.koog.agents.memory.providers.AgentMemoryProvider? = null,
    documentStorage: ai.koog.rag.base.RankedDocumentStorage<String>? = null,
    graphProvider: RetrievalProvider? = null,
    tokenizer: PromptTokenizer? = null,
    tokenConfig: TokenAwareRetrievalConfig? = null,
    metricsCollector: RetrievalMetricsCollector? = null,
    policy: RoutingPolicy = RoutingPolicy()
): SmartRouter {
    val providers = buildList {
        // Add graph provider first if available (highest priority)
        if (graphProvider != null) {
            add(graphProvider)
        }
        
        // Try to create KnowledgeGraphRetrievalProvider from GraphMemoryProvider
        if (graphProvider == null && memoryProvider != null) {
            val graphMemoryProvider = memoryProvider as? ai.koog.agents.memory.providers.GraphMemoryProvider
            if (graphMemoryProvider != null) {
                val knowledgeGraphRetriever = createKnowledgeGraphRetriever(graphMemoryProvider, tokenizer)
                if (knowledgeGraphRetriever != null) {
                    add(knowledgeGraphRetriever)
                }
            }
        }
        
        // Add vector provider as fallback
        if (memoryProvider != null || documentStorage != null) {
            add(
                ai.koog.agents.memory.retrieval.providers.VectorRetrievalProvider(
                    memoryProvider = memoryProvider,
                    documentStorage = documentStorage
                )
            )
        }
    }
    
    return SmartRouter(
        providers = providers, 
        policy = policy,
        tokenizer = tokenizer,
        tokenConfig = tokenConfig,
        metricsCollector = metricsCollector
    )
}

/**
 * Create a KnowledgeGraphRetrievalProvider from a GraphMemoryProvider
 * that uses a KnowledgeGraph.
 */
private fun createKnowledgeGraphRetriever(
    graphMemoryProvider: ai.koog.agents.memory.providers.GraphMemoryProvider,
    tokenizer: PromptTokenizer? = null
): ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider? {
    // Get the graph from GraphMemoryProvider's public property
    val graph = try {
        graphMemoryProvider.knowledgeGraph
    } catch (e: Exception) {
        null
    } ?: return null
    
    return ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider(graph, tokenizer)
}