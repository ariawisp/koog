package ai.koog.agents.memory.graph

import kotlinx.datetime.Instant

/**
 * Core interface for graph-based knowledge storage and retrieval.
 * This is the primary abstraction for agent memory, treating all knowledge
 * as nodes and edges in a temporal graph.
 */
public interface KnowledgeGraph {
    /**
     * Ingest an episode into the knowledge graph.
     * This may create multiple nodes and edges based on the content.
     */
    public suspend fun ingest(episode: Episode): IngestionResult
    
    /**
     * Query the knowledge graph.
     */
    public suspend fun query(request: KnowledgeRequest): List<Knowledge>
    
    /**
     * Evolve the graph over time - handle decay, consolidation, etc.
     */
    public suspend fun evolve(context: EvolutionContext)
    
    /**
     * Get graph statistics for monitoring
     */
    public suspend fun stats(): GraphStats
    
    /**
     * Resolve an entity mention to an existing entity or create a new one.
     * This is the core of entity resolution - matching text mentions to entities.
     */
    public suspend fun resolveEntity(mention: String, context: EntityResolutionContext): EntityResolutionResult
    
    /**
     * Invalidate edges that contradict new information.
     * Critical for temporal consistency and handling state changes.
     */
    public suspend fun invalidateContradictingEdges(newFact: Knowledge.Relation, at: Instant): List<EdgeInvalidation>
    
    /**
     * Detect communities within the graph for implicit relationship discovery.
     */
    public suspend fun detectCommunities(algorithm: CommunityDetectionAlgorithm = CommunityDetectionAlgorithm.LEIDEN): List<Community>
}

/**
 * Represents a unit of experience to be ingested into the graph
 */
public data class Episode(
    val content: String,
    val timestamp: Instant,
    val source: EpisodeSource,
    val metadata: Map<String, Any> = emptyMap(),
    val references: List<String> = emptyList(), // Entity IDs referenced
    val entityMentions: List<EntityMention> = emptyList(), // Entities mentioned in content
    val validFrom: Instant? = null, // When facts in this episode become valid
    val validTo: Instant? = null, // When facts in this episode stop being valid
    val confidence: Double = 1.0 // Overall confidence in this episode's information
)

/**
 * Represents a mention of an entity within an episode
 */
public data class EntityMention(
    val entityId: String, // UUID or normalized ID
    val text: String, // The actual text that mentions this entity (e.g., "Steve", "the leader")
    val startOffset: Int, // Character position in content where mention starts
    val endOffset: Int, // Character position where mention ends
    val confidence: Double = 1.0, // Confidence that this text refers to this entity
    val type: EntityType = EntityType.UNKNOWN // Type of entity mentioned
)

/**
 * Types of entities that can be mentioned
 */
public enum class EntityType {
    PERSON,
    LOCATION,
    ORGANIZATION,
    ITEM,
    EVENT,
    CONCEPT,
    UNKNOWN
}

/**
 * Source of an episode - kept simple and extensible
 */
public enum class EpisodeSource {
    /** Information from user input/conversation */
    USER_INPUT,
    /** Information from system events */
    SYSTEM_EVENT,
    /** Information from external sources */
    EXTERNAL
}

/**
 * Result of ingesting an episode
 */
public data class IngestionResult(
    val createdNodes: List<NodeId>,
    val createdEdges: List<EdgeId>,
    val updatedNodes: List<NodeId>,
    val conflicts: List<ConflictResolution> = emptyList()
)

/**
 * Request for querying knowledge
 */
public sealed interface KnowledgeRequest {
    /**
     * Query centered around a specific entity
     */
    public data class EntityCentric(
        val centerNode: String,
        val traversal: Traversal = Traversal.Outgoing(depth = 2),
        val filters: List<Filter> = emptyList(),
        val at: Instant? = null,
        val limit: Int = 20
    ) : KnowledgeRequest
    
    /**
     * Pattern-based query (like Cypher)
     */
    public data class Pattern(
        val pattern: String,
        val parameters: Map<String, Any> = emptyMap(),
        val at: Instant? = null,
        val limit: Int = 20
    ) : KnowledgeRequest
    
    /**
     * Semantic search across the graph
     */
    public data class Semantic(
        val query: String,
        val scope: SearchScope = SearchScope.ALL,
        val at: Instant? = null,
        val limit: Int = 20
    ) : KnowledgeRequest
    
    /**
     * Temporal window query
     */
    public data class Temporal(
        val start: Instant,
        val end: Instant,
        val entityFilter: List<String> = emptyList(),
        val eventTypes: List<String> = emptyList(),
        val includeDeleted: Boolean = false
    ) : KnowledgeRequest
}

/**
 * Graph traversal specification
 */
public sealed interface Traversal {
    public data class Outgoing(val depth: Int = 1, val relationTypes: List<String> = emptyList()) : Traversal
    public data class Incoming(val depth: Int = 1, val relationTypes: List<String> = emptyList()) : Traversal
    public data class Bidirectional(val depth: Int = 1, val relationTypes: List<String> = emptyList()) : Traversal
    public data class ShortestPath(val to: String, val maxDepth: Int = 5) : Traversal
}

/**
 * Filters for graph queries
 */
public sealed interface Filter {
    public data class HasLabel(val labels: List<String>) : Filter
    public data class HasProperty(val key: String, val value: Any? = null) : Filter
    public data class PropertyRange(val key: String, val min: Any?, val max: Any?) : Filter
    public data class Custom(val predicate: String) : Filter
}

/**
 * Search scope for semantic queries
 */
public enum class SearchScope {
    /** Search all content */
    ALL,
    /** Only entity properties */
    ENTITIES,
    /** Only relationship properties */
    RELATIONS,
    /** Only episode content */
    EPISODES
}

/**
 * Represents a piece of knowledge from the graph
 */
public sealed interface Knowledge {
    public val id: String
    public val confidence: Double
    public val timestamp: Instant
    public val provenance: List<ProvenanceItem>
    
    /**
     * An entity in the knowledge graph
     */
    public data class Entity(
        override val id: String,
        val labels: Set<String>,
        val properties: Map<String, Any>,
        override val confidence: Double,
        override val timestamp: Instant,
        override val provenance: List<ProvenanceItem>,
        val relations: List<Relation> = emptyList()
    ) : Knowledge
    
    /**
     * A relationship between entities
     */
    public data class Relation(
        override val id: String,
        val type: String,
        val from: String,
        val to: String,
        val properties: Map<String, Any>,
        override val confidence: Double,
        override val timestamp: Instant,
        override val provenance: List<ProvenanceItem>
    ) : Knowledge
    
    /**
     * A composite answer built from multiple graph elements
     */
    public data class Composite(
        override val id: String,
        val summary: String,
        val elements: List<Knowledge>,
        override val confidence: Double,
        override val timestamp: Instant,
        override val provenance: List<ProvenanceItem>
    ) : Knowledge
}

/**
 * Provenance information for knowledge
 */
public sealed interface ProvenanceItem {
    public data class Episode(val episodeId: String, val source: EpisodeSource) : ProvenanceItem
    public data class Inference(val rule: String, val confidence: Double) : ProvenanceItem
    public data class External(val source: String, val reference: String) : ProvenanceItem
}

/**
 * Context for graph evolution
 */
public data class EvolutionContext(
    val currentTime: Instant,
    val decayRules: List<DecayRule> = emptyList(),
    val consolidationRules: List<ConsolidationRule> = emptyList()
)

/**
 * Rule for knowledge decay
 */
public interface DecayRule {
    public fun shouldDecay(knowledge: Knowledge, currentTime: Instant): Boolean
    public fun decay(knowledge: Knowledge): Knowledge?
}

/**
 * Rule for knowledge consolidation
 */
public interface ConsolidationRule {
    public fun shouldConsolidate(knowledge: List<Knowledge>): Boolean
    public fun consolidate(knowledge: List<Knowledge>): Knowledge
}

/**
 * Statistics about the graph
 */
public data class GraphStats(
    val nodeCount: Long,
    val edgeCount: Long,
    val episodeCount: Long,
    val lastIngestion: Instant?,
    val lastEvolution: Instant?,
    val nodesByLabel: Map<String, Long>,
    val edgesByType: Map<String, Long>
)

/**
 * Conflict resolution when ingesting contradictory information
 */
public data class ConflictResolution(
    val existingKnowledge: Knowledge,
    val newKnowledge: Knowledge,
    val resolution: ResolutionStrategy,
    val reason: String
)

/**
 * Strategy for resolving conflicts
 */
public enum class ResolutionStrategy {
    /** Keep the existing knowledge */
    KEEP_EXISTING,
    /** Replace with new knowledge */
    REPLACE,
    /** Merge both pieces of knowledge */
    MERGE,
    /** Create a new version (bi-temporal) */
    VERSION
}

// Type aliases for clarity
public typealias NodeId = String
public typealias EdgeId = String

/**
 * Context for entity resolution operations
 */
public data class EntityResolutionContext(
    val episodeContent: String? = null,
    val nearbyEntities: List<Knowledge.Entity> = emptyList(),
    val confidenceThreshold: Double = 0.8,
    val entityType: EntityType? = null
)

/**
 * Result of entity resolution
 */
public sealed interface EntityResolutionResult {
    public data class Resolved(
        val entityId: String,
        val entity: Knowledge.Entity,
        val confidence: Double,
        val isNew: Boolean
    ) : EntityResolutionResult
    
    public data class Ambiguous(
        val candidates: List<EntityCandidate>,
        val reason: String
    ) : EntityResolutionResult
    
    public data class Failed(
        val reason: String
    ) : EntityResolutionResult
}

/**
 * Candidate entity for resolution
 */
public data class EntityCandidate(
    val entityId: String,
    val entity: Knowledge.Entity,
    val similarityScore: Double,
    val matchingFeatures: Set<String>
)

/**
 * Edge invalidation record
 */
public data class EdgeInvalidation(
    val edgeId: EdgeId,
    val edge: Knowledge.Relation,
    val invalidatedAt: Instant,
    val reason: String,
    val replacedBy: EdgeId? = null
)

/**
 * Community detection algorithms
 */
public enum class CommunityDetectionAlgorithm {
    LEIDEN,
    LOUVAIN,
    LABEL_PROPAGATION,
    CONNECTED_COMPONENTS
}

/**
 * Detected community in the graph
 */
public data class Community(
    val id: String,
    val members: Set<NodeId>,
    val cohesionScore: Double,
    val centralNodes: List<NodeId>,
    val description: String? = null
)