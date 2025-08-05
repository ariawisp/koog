package ai.koog.agents.memory.retrieval.providers

import ai.koog.agents.memory.graph.Knowledge
import ai.koog.agents.memory.graph.KnowledgeGraph
import ai.koog.agents.memory.graph.KnowledgeRequest
import ai.koog.agents.memory.graph.SearchScope
import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.reranking.CrossEncoderReranker
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import kotlinx.datetime.Instant

/**
 * Retrieval provider that leverages KnowledgeGraph's advanced search capabilities.
 * 
 * This provider works with any KnowledgeGraph implementation, including:
 * - External graph databases for bi-temporal and hybrid search
 * - Neo4jKnowledgeGraph for graph traversal
 * - Future implementations with different capabilities
 * 
 * The provider adapts to the capabilities of the underlying graph,
 * using reflection to detect and use advanced features when available.
 * 
 * Example usage:
 * ```kotlin
 * val knowledgeGraphProvider = KnowledgeGraphRetrievalProvider(myKnowledgeGraph)
 * 
 * // Create a smart router that prefers graph-based search
 * val router = createSmartRouter(
 *     graphProvider = knowledgeGraphProvider,
 *     memoryProvider = existingMemoryProvider
 * )
 * ```
 */
public class KnowledgeGraphRetrievalProvider(
    private val knowledgeGraph: KnowledgeGraph,
    private val tokenizer: PromptTokenizer? = null,
    private val crossEncoderReranker: CrossEncoderReranker? = null,
    private val defaultTokenBudget: Int = 2000
) : RetrievalProvider {
    
    // Detect capabilities at construction time
    private val supportsHybridSearch: Boolean = detectHybridSearchCapability()
    private val supportsTokenCounting: Boolean = tokenizer != null
    private val supportsCrossEncoder: Boolean = crossEncoderReranker != null
    
    override fun supports(recipe: RetrievalRecipe): Boolean = when(recipe) {
        // These are supported through KnowledgeRequest API
        RetrievalRecipe.HYBRID_NODE_DISTANCE,
        RetrievalRecipe.VECTOR_SIMILARITY,
        RetrievalRecipe.TEXT_BM25 -> true
        // These require specific implementations
        RetrievalRecipe.HYBRID_RRF,
        RetrievalRecipe.HYBRID_MMR -> supportsHybridSearch
        // Cross-encoder requires reranker
        RetrievalRecipe.HYBRID_CROSS_ENCODER -> supportsCrossEncoder
    }
    
    override suspend fun retrieve(query: RetrievalQuery, securityContext: ai.koog.agents.memory.security.SecurityContext?): List<RetrievalResult> {
        val results = when {
            // Handle temporal queries
            query.at != null -> retrieveTemporal(query)
            
            // Handle entity-centric queries with center node
            query.centerNode != null -> retrieveEntityCentric(query)
            
            // Default to semantic or hybrid search
            else -> retrieveDefault(query)
        }
        
        // Apply cross-encoder reranking if requested and available
        val rerankedResults = if (query.recipe == RetrievalRecipe.HYBRID_CROSS_ENCODER && supportsCrossEncoder && crossEncoderReranker != null) {
            crossEncoderReranker.rerank(query.text, results)
        } else {
            results
        }
        
        // Apply token budget optimization if tokenizer is available
        return if (supportsTokenCounting && tokenizer != null) {
            val budget = (query as? TokenAwareRetrievalQuery)?.tokenBudget ?: defaultTokenBudget
            optimizeResultsForTokenBudget(rerankedResults, budget)
        } else {
            rerankedResults
        }
    }
    
    private suspend fun retrieveDefault(query: RetrievalQuery): List<RetrievalResult> {
        // Try to use hybrid search if available and requested
        if (supportsHybridSearch && (query.recipe == RetrievalRecipe.HYBRID_RRF || query.recipe == RetrievalRecipe.HYBRID_MMR)) {
            return retrieveHybrid(query)
        }
        
        // Fall back to semantic search
        val request = KnowledgeRequest.Semantic(
            query = query.text,
            scope = SearchScope.ALL,
            limit = query.k
        )
        
        val knowledge = knowledgeGraph.query(request)
        return knowledge.mapIndexed { index, k ->
            knowledgeToResult(k, 1.0 - (index * 0.05))
        }
    }
    
    private suspend fun retrieveHybrid(query: RetrievalQuery): List<RetrievalResult> {
        // For now, fall back to semantic search since we don't have a graph with hybrid search
        // In the future, we can implement hybrid search by combining semantic and BM25 results
        return retrieveDefault(query.copy(recipe = RetrievalRecipe.VECTOR_SIMILARITY))
    }
    
    private suspend fun retrieveEntityCentric(query: RetrievalQuery): List<RetrievalResult> {
        // Use semantic search with entity-centric request
        val request = KnowledgeRequest.EntityCentric(
            centerNode = query.centerNode!!,
            traversal = ai.koog.agents.memory.graph.Traversal.Bidirectional(
                depth = 3, // Search up to 3 hops from center
                relationTypes = emptyList() // All relation types
            ),
            filters = query.filters.entityLabels.toList().map { label ->
                ai.koog.agents.memory.graph.Filter.HasLabel(listOf(label))
            },
            at = query.at,
            limit = query.k * 2 // Get extra results for filtering
        )
        
        val knowledge = knowledgeGraph.query(request)
        
        // Also do a semantic search to get text-relevant results
        val semanticResults = if (query.text.isNotBlank()) {
            knowledgeGraph.query(
                KnowledgeRequest.Semantic(
                    query = query.text,
                    scope = SearchScope.ALL,
                    limit = query.k
                )
            )
        } else {
            emptyList()
        }
        
        // Merge results, preferring entity-centric but including semantic
        val allKnowledge = (knowledge + semanticResults).distinctBy { it.id }
        
        // Apply node distance reranking if requested
        return if (query.recipe == RetrievalRecipe.HYBRID_NODE_DISTANCE) {
            reankByNodeDistance(allKnowledge, query.centerNode!!, query.k)
        } else {
            allKnowledge.take(query.k).mapIndexed { index, k ->
                knowledgeToResult(k, 1.0 - (index * 0.05))
            }
        }
    }
    
    private suspend fun retrieveTemporal(query: RetrievalQuery): List<RetrievalResult> {
        // For temporal queries, we need to use semantic search at a specific time
        val request = KnowledgeRequest.Semantic(
            query = query.text,
            scope = SearchScope.ALL,
            at = query.at,
            limit = query.k
        )
        
        val knowledge = knowledgeGraph.query(request)
        
        return knowledge.mapIndexed { index, k ->
            knowledgeToResult(k, 1.0 - (index * 0.05))
        }
    }
    
    private fun reankByNodeDistance(
        knowledge: List<Knowledge>,
        centerNode: String,
        limit: Int
    ): List<RetrievalResult> {
        // Simple distance-based reranking
        // In a real implementation, this would use graph distance metrics
        val reranked = knowledge.sortedBy { k ->
            when (k) {
                is Knowledge.Entity -> if (k.id == centerNode) 0 else 1
                is Knowledge.Relation -> {
                    if (k.from == centerNode || k.to == centerNode) 1 else 2
                }
                is Knowledge.Composite -> 3
            }
        }
        
        return reranked.take(limit).mapIndexed { index, k ->
            val distanceBoost = when {
                k.id == centerNode -> 1.0
                k is Knowledge.Relation && (k.from == centerNode || k.to == centerNode) -> 0.8
                else -> 0.6
            }
            knowledgeToResult(k, distanceBoost * (1.0 - (index * 0.05)))
        }
    }
    
    private fun knowledgeToResult(
        knowledge: Knowledge,
        score: Double
    ): RetrievalResult {
        val content = when (knowledge) {
            is Knowledge.Entity -> {
                val props = knowledge.properties.entries.joinToString(", ") { 
                    "${it.key}: ${it.value}" 
                }
                "${knowledge.labels.joinToString("/")} - $props"
            }
            is Knowledge.Relation -> {
                "${knowledge.from} -[${knowledge.type}]-> ${knowledge.to}"
            }
            is Knowledge.Composite -> {
                knowledge.summary
            }
        }
        
        val provenance = listOf(
            Provenance.Graph(
                nodeId = if (knowledge is Knowledge.Entity) knowledge.id else null,
                edgeId = if (knowledge is Knowledge.Relation) knowledge.id else null,
                validFrom = knowledge.timestamp,
                validTo = null // Current knowledge
            )
        )
        
        val metadata = mutableMapOf<String, String>()
        metadata["confidence"] = knowledge.confidence.toString()
        metadata["type"] = when (knowledge) {
            is Knowledge.Entity -> "entity"
            is Knowledge.Relation -> "relation"
            is Knowledge.Composite -> "composite"
        }
        
        return RetrievalResult(
            content = content,
            score = score.coerceIn(0.0, 1.0),
            provenance = provenance,
            metadata = metadata
        )
    }
    
    private fun detectHybridSearchCapability(): Boolean {
        // For now, return false since we don't have a graph with hybrid search
        // In the future, we can check for specific graph implementations
        return false
    }
    
    private fun optimizeResultsForTokenBudget(
        results: List<RetrievalResult>,
        budget: Int
    ): List<RetrievalResult> {
        if (tokenizer == null || results.isEmpty()) return results
        
        // Pre-compute token counts
        data class TokenizedResult(
            val result: RetrievalResult,
            val tokenCount: Int,
            val tokensPerScore: Double
        )
        
        val tokenizedResults = results.map { result ->
            val tokens = tokenizer.tokenCountFor(ai.koog.prompt.message.Message.User(result.content, ai.koog.prompt.message.RequestMetaInfo(kotlinx.datetime.Clock.System.now())))
            TokenizedResult(
                result = result,
                tokenCount = tokens,
                tokensPerScore = if (result.score > 0) result.score / tokens else 0.0
            )
        }
        
        // Sort by efficiency (highest score per token)
        val sorted = tokenizedResults.sortedByDescending { it.tokensPerScore }
        
        // Greedy selection within budget
        val selected = mutableListOf<RetrievalResult>()
        var usedTokens = 0
        
        for (tokenized in sorted) {
            if (usedTokens + tokenized.tokenCount <= budget) {
                selected.add(tokenized.result)
                usedTokens += tokenized.tokenCount
            }
        }
        
        // Sort selected results by original score to maintain relevance order
        return selected.sortedByDescending { it.score }
    }
}