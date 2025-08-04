package ai.koog.agents.memory.retrieval

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
    private val policy: RoutingPolicy = RoutingPolicy()
) : RetrievalProvider {
    
    private val logger = KotlinLogging.logger {}
    
    init {
        require(providers.isNotEmpty()) { "At least one provider must be configured" }
    }
    
    override fun supports(recipe: RetrievalRecipe): Boolean = true // Router can handle any recipe
    
    override suspend fun retrieve(query: RetrievalQuery): List<RetrievalResult> {
        // Select the best provider based on query characteristics
        val selectedProvider = selectProvider(query)
        
        logger.debug { 
            "Routing query to ${selectedProvider::class.simpleName}: " +
            "temporal=${query.at != null}, centerNode=${query.centerNode != null}, " +
            "recipe=${query.recipe}"
        }
        
        // Try primary provider, fall back to others if it fails
        return try {
            selectedProvider.retrieve(query)
        } catch (e: Exception) {
            logger.warn(e) { "Primary provider failed, trying fallback" }
            fallbackRetrieve(query, selectedProvider)
        }
    }
    
    private fun selectProvider(query: RetrievalQuery): RetrievalProvider {
        // Check if a specific recipe is requested and a provider supports it
        val recipeProvider = providers.firstOrNull { it.supports(query.recipe) }
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
            val graphProvider = providers.firstOrNull { 
                it.supports(RetrievalRecipe.HYBRID_NODE_DISTANCE) 
            }
            if (graphProvider != null) {
                return graphProvider
            }
        }
        
        // Default to first available provider
        return providers.first()
    }
    
    private suspend fun fallbackRetrieve(
        query: RetrievalQuery, 
        failedProvider: RetrievalProvider
    ): List<RetrievalResult> {
        // Try other providers
        for (provider in providers) {
            if (provider != failedProvider) {
                try {
                    return provider.retrieve(query)
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
    policy: RoutingPolicy = RoutingPolicy()
): SmartRouter {
    val providers = buildList {
        // Add graph provider first if available (highest priority)
        if (graphProvider != null) {
            add(graphProvider)
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
    
    return SmartRouter(providers, policy)
}