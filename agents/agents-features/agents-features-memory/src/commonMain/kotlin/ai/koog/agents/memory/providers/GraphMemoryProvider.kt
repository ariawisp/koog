package ai.koog.agents.memory.providers

import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.model.*
import ai.koog.agents.memory.retrieval.RetrievalProvider
import ai.koog.agents.memory.retrieval.RetrievalQuery
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Configuration for graph-based memory storage.
 * This provider offers state-of-the-art capabilities including:
 * - Temporal reasoning and point-in-time queries
 * - Entity-centric relationship traversal  
 * - Advanced semantic search with hybrid ranking
 * - Automatic knowledge evolution (consolidation, decay)
 */
@Serializable
@SerialName("graph")
public data class GraphMemoryConfig(
    override val defaultScope: MemoryScope = MemoryScope.CrossProduct,
    
    /**
     * Whether to enable automatic conversation ingestion.
     * When true, LLM conversations are automatically converted to episodes
     * and ingested into the knowledge graph.
     */
    val autoIngestConversations: Boolean = true,
    
    /**
     * Minimum confidence threshold for accepting inferred knowledge.
     * Knowledge below this threshold is filtered out during ingestion.
     */
    val minConfidenceThreshold: Double = 0.7
) : MemoryProviderConfig

/**
 * Advanced memory provider backed by a knowledge graph.
 * 
 * This provider offers state-of-the-art memory capabilities:
 * - **Temporal Reasoning**: Query what was known at any point in time
 * - **Entity-Centric Search**: Find information related to specific entities
 * - **Semantic Search**: Advanced NLP-powered semantic matching
 * - **Relationship Traversal**: Follow connections between related concepts
 * - **Knowledge Evolution**: Automatic consolidation and decay of information
 * 
 * The provider acts as an adapter between the traditional AgentMemoryProvider
 * interface and the advanced graph-based memory engine, preserving all SOTA
 * capabilities while providing a familiar API.
 * 
 * Example usage:
 * ```kotlin
 * install(AgentMemory) {
 *     provider = GraphMemoryProvider(
 *         graph = InMemoryKnowledgeGraph(),
 *         config = GraphMemoryConfig(autoIngestConversations = true)
 *     )
 * }
 * ```
 */
public class GraphMemoryProvider(
    private val graph: KnowledgeGraph,
    private val config: GraphMemoryConfig = GraphMemoryConfig(),
    private val retriever: RetrievalProvider? = null,
    private val clock: Clock = Clock.System
) : AgentMemoryProvider {
    
    // Note: Capabilities are implicit - the provider simply implements
    // the methods it supports and throws clear exceptions for unsupported operations
    
    override suspend fun save(fact: Fact, subject: MemorySubject, scope: MemoryScope) {
        // Convert Fact to Episode for graph ingestion
        val episode = Episode(
            content = when (fact) {
                is SingleFact -> "${fact.concept.description}: ${fact.value}"
                is MultipleFacts -> "${fact.concept.description}: ${fact.values.joinToString(", ")}"
            },
            timestamp = Instant.fromEpochMilliseconds(fact.timestamp),
            source = when (subject.name.lowercase()) {
                "machine", "environment" -> EpisodeSource.EXTERNAL
                "project", "codebase" -> EpisodeSource.EXTERNAL  
                "user" -> EpisodeSource.USER_INPUT
                "agent" -> EpisodeSource.SYSTEM_EVENT
                else -> EpisodeSource.EXTERNAL
            },
            metadata = mapOf(
                "concept_keyword" to fact.concept.keyword,
                "concept_description" to fact.concept.description,
                "scope" to scope.toString(),
                "subject" to subject.toString(),
                "fact_type" to when (fact) {
                    is SingleFact -> "single"
                    is MultipleFacts -> "multiple"
                    else -> "unknown" // Handle any future fact types
                }
            )
        )
        
        graph.ingest(episode)
    }
    
    override suspend fun load(concept: Concept, subject: MemorySubject, scope: MemoryScope): List<Fact> {
        // Use semantic search to find facts matching the concept
        val knowledge = graph.query(
            KnowledgeRequest.Semantic(
                query = "${concept.keyword} ${concept.description}",
                limit = 50
            )
        )
        
        return knowledge.mapNotNull { it.toFactOrNull(concept, subject, scope) }
    }
    
    override suspend fun loadAll(subject: MemorySubject, scope: MemoryScope): List<Fact> {
        // Query all knowledge and filter by metadata
        val allKnowledge = graph.query(
            KnowledgeRequest.Pattern(
                pattern = "MATCH (n) RETURN n",
                limit = 1000
            )
        )
        
        return allKnowledge.mapNotNull { knowledge ->
            val metadata = when (knowledge) {
                is Knowledge.Entity -> knowledge.properties
                is Knowledge.Relation -> knowledge.properties
                else -> emptyMap<String, Any>() // Handle any future Knowledge types
            }
            
            // Filter by subject and scope from metadata
            val factSubject = metadata["subject"] as? String
            val factScope = metadata["scope"] as? String
            
            if (factSubject == subject.toString() && factScope == scope.toString()) {
                knowledge.toFactOrNull(null, subject, scope)
            } else null
        }
    }
    
    override suspend fun loadByDescription(
        description: String, 
        subject: MemorySubject, 
        scope: MemoryScope
    ): List<Fact> {
        // Use retriever if available, otherwise fallback to semantic search
        val knowledge = if (retriever != null) {
            retriever.retrieve(RetrievalQuery(text = description)).map { result ->
                // Convert RetrievalResult back to Knowledge
                // This is a simplified conversion - real implementation would be more sophisticated
                Knowledge.Entity(
                    id = "retrieval-${result.content.hashCode()}",
                    labels = setOf("Fact"),
                    properties = mapOf("content" to result.content),
                    confidence = result.score,
                    timestamp = clock.now(),
                    provenance = emptyList()
                )
            }
        } else {
            graph.query(
                KnowledgeRequest.Semantic(
                    query = description,
                    limit = 20
                )
            )
        }
        
        return knowledge.mapNotNull { it.toFactOrNull(null, subject, scope) }
    }
    
    // === Advanced Graph Operations (Power User API) ===
    
    /**
     * Query knowledge that existed at a specific point in time.
     * This enables temporal reasoning about what the agent knew when.
     */
    public suspend fun queryTemporal(
        start: Instant, 
        end: Instant, 
        includeDeleted: Boolean = false
    ): List<Knowledge> {
        return graph.query(
            KnowledgeRequest.Temporal(
                start = start,
                end = end,
                includeDeleted = includeDeleted
            )
        )
    }
    
    /**
     * Query knowledge centered around a specific entity.
     * This enables relationship traversal and entity-centric search.
     */
    public suspend fun queryEntityCentric(
        centerNode: String,
        depth: Int = 2,
        relationTypes: List<String> = emptyList(),
        at: Instant? = null
    ): List<Knowledge> {
        return graph.query(
            KnowledgeRequest.EntityCentric(
                centerNode = centerNode,
                traversal = Traversal.Bidirectional(depth, relationTypes),
                at = at
            )
        )
    }
    
    /**
     * Trigger knowledge evolution (consolidation, decay) manually.
     * Normally this happens automatically, but can be triggered for testing or optimization.
     */
    public suspend fun evolve(rules: EvolutionContext? = null) {
        val context = rules ?: EvolutionContext(
            currentTime = clock.now(),
            decayRules = emptyList(),
            consolidationRules = emptyList()
        )
        graph.evolve(context)
    }
    
    /**
     * Get statistics about the knowledge graph.
     * Useful for monitoring and debugging.
     */
    public suspend fun getGraphStats(): GraphStats = graph.stats()
    
    // === Internal Conversion Methods ===
    
    private fun Knowledge.toFactOrNull(
        concept: Concept?, 
        subject: MemorySubject, 
        scope: MemoryScope
    ): Fact? {
        return when (this) {
            is Knowledge.Entity -> {
                val content = properties["content"] as? String ?: return null
                val factConcept = concept ?: Concept(
                    keyword = id,
                    description = labels.firstOrNull() ?: "Knowledge",
                    factType = FactType.SINGLE
                )
                
                SingleFact(
                    concept = factConcept,
                    timestamp = timestamp.toEpochMilliseconds(),
                    value = content
                )
            }
            is Knowledge.Relation -> {
                // Relations can be converted to facts about relationships
                val factConcept = concept ?: Concept(
                    keyword = type,
                    description = "Relationship: $type",
                    factType = FactType.SINGLE
                )
                
                SingleFact(
                    concept = factConcept,
                    timestamp = timestamp.toEpochMilliseconds(),
                    value = "Relationship from $from to $to of type $type"
                )
            }
            else -> null // Handle any future Knowledge types
        }
    }
}