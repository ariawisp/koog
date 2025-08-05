package ai.koog.agents.memory.retrieval

import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.model.MemorySubject
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable

/**
 * Represents a query for retrieving knowledge from memory systems.
 * Supports both simple text queries and advanced graph-based retrieval.
 * 
 * @property text The search query text
 * @property centerNode Optional node ID for graph-based proximity search (e.g., player/base UUID)
 * @property at Optional timestamp for temporal queries
 * @property target What type of results to retrieve
 * @property k Maximum number of results to return
 * @property filters Optional filters to apply
 * @property recipe Strategy for how to perform the search
 * @property requireCitations Whether to include source provenance
 */
@Serializable
public data class RetrievalQuery(
    val text: String,
    val centerNode: String? = null,
    val at: Instant? = null,
    val target: RetrievalTarget = RetrievalTarget.ALL,
    val k: Int = 10,
    val filters: RetrievalFilters = RetrievalFilters(),
    val recipe: RetrievalRecipe = RetrievalRecipe.HYBRID_RRF,
    val requireCitations: Boolean = true
)

/**
 * Specifies what type of knowledge to retrieve
 */
@Serializable
public enum class RetrievalTarget {
    /** Retrieve facts/edges (relationships between entities) */
    FACTS,
    /** Retrieve entities/nodes */
    ENTITIES,
    /** Retrieve documents */
    DOCUMENTS,
    /** Retrieve all types */
    ALL
}

/**
 * Filters to apply during retrieval
 */
@Serializable
public data class RetrievalFilters(
    val subjects: Set<MemorySubject> = emptySet(),
    val scopes: Set<MemoryScope> = emptySet(),
    val entityLabels: Set<String> = emptySet(),
    val factTypes: Set<String> = emptySet()
)

/**
 * Strategy/recipe for how to perform retrieval
 */
@Serializable
public enum class RetrievalRecipe {
    /** Hybrid search with Reciprocal Rank Fusion */
    HYBRID_RRF,
    /** Hybrid search with Maximal Marginal Relevance */
    HYBRID_MMR,
    /** Hybrid search with cross-encoder reranking */
    HYBRID_CROSS_ENCODER,
    /** Graph-based search with node distance reranking */
    HYBRID_NODE_DISTANCE,
    /** Simple vector similarity search */
    VECTOR_SIMILARITY,
    /** BM25 text search only */
    TEXT_BM25
}

/**
 * Represents a retrieved item with provenance information
 */
@Serializable
public data class RetrievalResult(
    val content: String,
    val score: Double,
    val provenance: List<Provenance> = emptyList(),
    val metadata: Map<String, String> = emptyMap()
)

/**
 * Source attribution for retrieved information
 */
@Serializable
public sealed interface Provenance {
    /**
     * Provenance from a graph database (entity/edge)
     */
    @Serializable
    public data class Graph(
        val nodeId: String? = null,
        val edgeId: String? = null,
        val episodeId: String? = null,
        val validFrom: Instant? = null,
        val validTo: Instant? = null
    ) : Provenance
    
    /**
     * Provenance from a document
     */
    @Serializable
    public data class Document(
        val path: String,
        val lineStart: Int? = null,
        val lineEnd: Int? = null
    ) : Provenance
    
    /**
     * Provenance from a memory fact
     */
    @Serializable
    public data class Fact(
        val conceptKeyword: String,
        val subject: String,
        val scope: String,
        val timestamp: Long
    ) : Provenance
}

/**
 * Interface for knowledge retrieval providers
 */
public interface RetrievalProvider {
    /**
     * Check if this provider supports the given recipe
     */
    public fun supports(recipe: RetrievalRecipe): Boolean
    
    /**
     * Retrieve knowledge based on the query with optional security context
     * Security context is automatically applied to filter results based on access permissions
     */
    public suspend fun retrieve(
        query: RetrievalQuery,
        securityContext: ai.koog.agents.memory.security.SecurityContext? = null
    ): List<RetrievalResult>
}