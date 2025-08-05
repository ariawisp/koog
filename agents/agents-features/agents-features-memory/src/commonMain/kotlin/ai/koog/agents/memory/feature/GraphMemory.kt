package ai.koog.agents.memory.feature

import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.core.agent.context.AIAgentLLMContext
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.agent.entity.createStorageKey
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.feature.AIAgentFeature
import ai.koog.agents.core.feature.AIAgentPipeline
import ai.koog.agents.core.feature.InterceptContext
import ai.koog.agents.features.common.config.FeatureConfig
import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.structure.json.JsonStructuredData
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.datetime.Clock
import kotlinx.serialization.Serializable
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Graph-based memory implementation for AI agents.
 * 
 * This is a complete redesign of the memory system to be graph-first,
 * treating all agent knowledge as nodes and edges in a temporal graph.
 * 
 * Key concepts:
 * - Episodes: Units of experience that get decomposed into graph elements
 * - Entities: Nodes representing players, bases, factions, items, etc.
 * - Relations: Edges representing relationships, actions, ownership, etc.
 * - Temporal: All knowledge is timestamped and can be queried at points in time
 * 
 * Example usage:
 * ```kotlin
 * val agent = AIAgents(strategy = myStrategy) {
 *     install(GraphMemory) {
 *         graph = Neo4jKnowledgeGraph(config)
 *         
 *         // Configure how episodes are decomposed
 *         episodeProcessor = MyGameEpisodeProcessor()
 *         
 *         // Configure evolution rules
 *         evolutionRules {
 *             decay(UnverifiedInfoDecay(days = 7))
 *             consolidate(DuplicateEntityMerger())
 *         }
 *     }
 * }
 * ```
 */
@OptIn(InternalAgentsApi::class)
public class GraphMemory(
    @property:InternalAgentsApi
    public val graph: KnowledgeGraph,
    @property:InternalAgentsApi
    public val llm: AIAgentLLMContext,
    @property:InternalAgentsApi
    public val episodeProcessor: EpisodeProcessor,
    @property:InternalAgentsApi
    public val evolutionRules: EvolutionConfig,
    @property:InternalAgentsApi
    public val clock: Clock = Clock.System
) {
    /**
     * Configuration for the GraphMemory feature
     */
    public class Config : FeatureConfig() {
        /**
         * The knowledge graph implementation to use
         */
        public var graph: KnowledgeGraph? = null
        
        /**
         * Processor for decomposing episodes into graph elements
         */
        public var episodeProcessor: EpisodeProcessor? = null
        
        /**
         * Clock for timestamps
         */
        public var clock: Clock = Clock.System
        
        /**
         * Evolution configuration
         */
        internal val evolutionConfig = EvolutionConfig()
        
        /**
         * Configure evolution rules
         */
        public fun evolutionRules(block: EvolutionConfig.() -> Unit) {
            evolutionConfig.apply(block)
        }
        
        /**
         * Whether to automatically ingest LLM conversations as episodes
         */
        public var autoIngestConversations: Boolean = true
        
        /**
         * Minimum confidence threshold for accepting inferred knowledge
         */
        public var minConfidenceThreshold: Double = 0.7
    }
    
    /**
     * Configuration for knowledge evolution
     */
    public class EvolutionConfig {
        internal val decayRules = mutableListOf<DecayRule>()
        internal val consolidationRules = mutableListOf<ConsolidationRule>()
        
        public fun decay(rule: DecayRule) {
            decayRules.add(rule)
        }
        
        public fun consolidate(rule: ConsolidationRule) {
            consolidationRules.add(rule)
        }
    }
    
    public companion object Feature : AIAgentFeature<Config, GraphMemory> {
        private val logger = KotlinLogging.logger {}
        
        override val key: AIAgentStorageKey<GraphMemory> = 
            createStorageKey<GraphMemory>("graph-memory-feature")
            
        override fun createInitialConfig(): Config = Config()
        
        override fun install(config: Config, pipeline: AIAgentPipeline) {
            pipeline.interceptContextAgentFeature(this) { agentContext ->
                // Create graph first if not provided
                val graph = config.graph ?: InMemoryKnowledgeGraph(clock = config.clock)
                
                // Use DefaultEpisodeProcessor with entity resolution enabled
                val episodeProcessor = config.episodeProcessor ?: DefaultEpisodeProcessor(
                    llm = agentContext.llm,
                    knowledgeGraph = graph,
                    minConfidence = config.minConfidenceThreshold
                )
                
                val memory = GraphMemory(
                    graph = graph,
                    llm = agentContext.llm,
                    episodeProcessor = episodeProcessor,
                    evolutionRules = config.evolutionConfig,
                    clock = config.clock
                )
                
                // Set up automatic conversation ingestion if enabled
                if (config.autoIngestConversations) {
                    setupConversationIngestion(memory, pipeline)
                }
                
                memory
            }
        }
        
        private fun setupConversationIngestion(memory: GraphMemory, pipeline: AIAgentPipeline) {
            // Register interceptor for LLM calls to automatically ingest conversations as episodes
            pipeline.interceptAfterLLMCall(InterceptContext(Feature, memory)) { eventContext ->
                // Extract conversation content from prompt and responses
                val conversationContent = buildString {
                    // Add the last user message from the prompt
                    eventContext.prompt.messages.lastOrNull { it is Message.User }?.let { userMessage ->
                        appendLine("User: ${userMessage.content}")
                    }
                    
                    // Add the assistant's response
                    eventContext.responses.forEach { response ->
                        when (response) {
                            is Message.Assistant -> appendLine("Assistant: ${response.content}")
                            is Message.Tool.Call -> appendLine("Assistant: [Tool call: ${response.tool}]")
                            else -> {} // Ignore other response types
                        }
                    }
                }
                
                // Only ingest if there's meaningful content
                if (conversationContent.isNotBlank()) {
                    // Launch in the background to avoid blocking the conversation
                    CoroutineScope(Dispatchers.Default).launch {
                        try {
                            memory.ingest(
                                content = conversationContent.trim(),
                                source = EpisodeSource.SYSTEM_EVENT,
                                metadata = mapOf(
                                    "model" to eventContext.model.id,
                                    "runId" to eventContext.runId,
                                    "timestamp" to memory.clock.now().toEpochMilliseconds()
                                )
                            )
                        } catch (e: Exception) {
                            // Log error but don't fail the conversation
                            logger.error(e) { "Failed to ingest conversation episode" }
                        }
                    }
                }
            }
        }
    }
    
    /**
     * Ingest an episode into the knowledge graph
     */
    public suspend fun ingest(
        content: String,
        source: EpisodeSource = EpisodeSource.SYSTEM_EVENT,
        metadata: Map<String, Any> = emptyMap()
    ): IngestionResult {
        val episode = Episode(
            content = content,
            timestamp = clock.now(),
            source = source,
            metadata = metadata
        )
        
        return graph.ingest(episode)
    }
    
    /**
     * Query for knowledge about specific entities
     */
    public suspend fun queryEntity(
        entityId: String,
        depth: Int = 2,
        at: kotlinx.datetime.Instant? = null
    ): List<Knowledge> {
        return graph.query(
            KnowledgeRequest.EntityCentric(
                centerNode = entityId,
                traversal = Traversal.Bidirectional(depth),
                at = at
            )
        )
    }
    
    /**
     * Semantic search across the knowledge graph
     */
    public suspend fun search(
        query: String,
        limit: Int = 20
    ): List<Knowledge> {
        return graph.query(
            KnowledgeRequest.Semantic(
                query = query,
                limit = limit
            )
        )
    }
    
    /**
     * Query using a graph pattern
     */
    public suspend fun queryPattern(
        pattern: String,
        parameters: Map<String, Any> = emptyMap()
    ): List<Knowledge> {
        return graph.query(
            KnowledgeRequest.Pattern(
                pattern = pattern,
                parameters = parameters
            )
        )
    }
    
    /**
     * Get knowledge from a time window
     */
    public suspend fun queryTimeWindow(
        start: kotlinx.datetime.Instant,
        end: kotlinx.datetime.Instant,
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
     * Evolve the graph (decay old knowledge, consolidate, etc.)
     */
    public suspend fun evolve() {
        val context = EvolutionContext(
            currentTime = clock.now(),
            decayRules = evolutionRules.decayRules,
            consolidationRules = evolutionRules.consolidationRules
        )
        
        graph.evolve(context)
    }
    
    /**
     * Get graph statistics
     */
    public suspend fun stats(): GraphStats = graph.stats()
}

/**
 * Interface for processing episodes into graph elements
 */
public interface EpisodeProcessor {
    /**
     * Process an episode and extract graph elements
     */
    public suspend fun process(episode: Episode): ProcessedEpisode
}

/**
 * Result of processing an episode
 */
public data class ProcessedEpisode(
    val entities: List<EntityCandidate>,
    val relations: List<RelationCandidate>,
    val updates: List<PropertyUpdate>
)

/**
 * Candidate entity to create/update
 */
public data class EntityCandidate(
    val id: String,
    val labels: Set<String>,
    val properties: Map<String, Any>,
    val confidence: Double = 1.0
)

/**
 * Candidate relation to create
 */
public data class RelationCandidate(
    val from: String,
    val to: String,
    val type: String,
    val properties: Map<String, Any>,
    val confidence: Double = 1.0
)

/**
 * Property update for existing entity/relation
 */
public data class PropertyUpdate(
    val targetId: String,
    val property: String,
    val value: Any,
    val operation: UpdateOperation = UpdateOperation.SET
)

/**
 * How to update a property
 */
public enum class UpdateOperation {
    SET,
    APPEND,
    INCREMENT,
    DELETE
}

/**
 * Default episode processor using LLM with entity resolution
 */
@OptIn(ExperimentalUuidApi::class)
internal class DefaultEpisodeProcessor(
    private val llm: AIAgentLLMContext,
    private val knowledgeGraph: KnowledgeGraph? = null,
    private val minConfidence: Double = 0.7
) : EpisodeProcessor {
    override suspend fun process(episode: Episode): ProcessedEpisode {
        // Extract raw entities and relations with enhanced prompting
        val rawExtraction = extractRawKnowledge(episode)
        
        // If we have a knowledge graph, perform entity resolution
        return if (knowledgeGraph != null) {
            val resolvedEntities = resolveEntities(rawExtraction.entities, episode)
            val resolvedRelations = updateRelationIds(rawExtraction.relations, rawExtraction.entities, resolvedEntities)
            
            ProcessedEpisode(
                entities = resolvedEntities.filter { it.confidence >= minConfidence },
                relations = resolvedRelations.filter { it.confidence >= minConfidence },
                updates = rawExtraction.updates
            )
        } else {
            // Without knowledge graph, return raw extraction
            ProcessedEpisode(
                entities = rawExtraction.entities
                    .filter { it.confidence >= minConfidence }
                    .map { EntityCandidate(
                        id = it.id,
                        labels = it.labels,
                        properties = it.properties,
                        confidence = it.confidence
                    )},
                relations = rawExtraction.relations.filter { it.confidence >= minConfidence },
                updates = rawExtraction.updates
            )
        }
    }
    
    private suspend fun extractRawKnowledge(episode: Episode): RawKnowledgeExtraction {
        val structuredData = JsonStructuredData.createJsonStructure<EnhancedKnowledgeExtraction>(
            id = "enhanced-knowledge-extraction",
            examples = listOf(createExampleExtraction())
        )
        
        val extraction = llm.writeSession {
            prompt = Prompt.build(prompt) {
                user("""Extract entities, relationships, and temporal information from the following text.
                    
                    Text: ${episode.content}
                    Timestamp: ${episode.timestamp}
                    
                    Instructions:
                    1. For each entity, provide:
                       - Clear text mentions (exact phrases that refer to the entity)
                       - Entity type (PERSON, LOCATION, ORGANIZATION, etc.)
                       - Confidence score (0.0-1.0)
                       - Any aliases or alternative names mentioned
                    
                    2. For relationships:
                       - Use the entity text mentions as references
                       - Provide confidence scores
                       - Note if the relationship is temporal (has start/end times)
                    
                    3. For temporal information:
                       - Extract any time references (dates, "yesterday", "last week", etc.)
                       - Identify which facts are time-bound
                       - Note state changes (was X, now Y)
                    
                    4. Be conservative with confidence scores:
                       - 1.0: Explicitly stated facts
                       - 0.8-0.9: Strong implications
                       - 0.6-0.7: Reasonable inferences
                       - Below 0.6: Speculation
                    
                    Return the structured JSON response.
                    """.trimIndent())
            }
            
            val result = requestLLMStructured(structuredData)
            result.getOrThrow().structure
        }
        
        return RawKnowledgeExtraction(
            entities = extraction.entities.map { entity ->
                InternalEntityCandidate(
                    id = "temp-${Uuid.random()}", // Temporary ID, will be resolved
                    labels = setOf(entity.type),
                    properties = buildMap {
                        put("name", entity.mainMention)
                        entity.aliases?.let { put("aliases", it) }
                        entity.properties.forEach { (k, v) -> put(k, v) }
                    },
                    confidence = entity.confidence,
                    mentions = entity.mentions
                )
            },
            relations = extraction.relations.map { relation ->
                RelationCandidate(
                    from = relation.fromMention, // Will be resolved to entity ID
                    to = relation.toMention,      // Will be resolved to entity ID
                    type = relation.type,
                    properties = buildMap {
                        relation.properties.forEach { (k, v) -> put(k, v) }
                        relation.validFrom?.let { put("valid_from", it) }
                        relation.validTo?.let { put("valid_to", it) }
                    },
                    confidence = relation.confidence
                )
            },
            updates = extraction.updates.map { update ->
                PropertyUpdate(
                    targetId = update.targetMention, // Will be resolved
                    property = update.property,
                    value = update.value,
                    operation = update.operation
                )
            },
            temporalContext = TemporalContext(
                referenceTime = episode.timestamp,
                validFrom = episode.validFrom,
                validTo = episode.validTo
            )
        )
    }
    
    private suspend fun resolveEntities(
        candidates: List<InternalEntityCandidate>,
        episode: Episode
    ): List<EntityCandidate> {
        val resolved = mutableListOf<EntityCandidate>()
        val mentionToId = mutableMapOf<String, String>()
        
        // Group entities by their main mention to handle coreferences
        val entityGroups = candidates.groupBy { it.properties["name"]?.toString() ?: "" }
        
        for ((mainMention, group) in entityGroups) {
            if (mainMention.isBlank()) continue
            
            // Use the highest confidence entity from the group
            val representative = group.maxByOrNull { it.confidence } ?: continue
            
            // Resolve this entity
            val resolution = knowledgeGraph!!.resolveEntity(
                mention = mainMention,
                context = EntityResolutionContext(
                    episodeContent = episode.content,
                    entityType = representative.labels.firstOrNull()?.let { label ->
                        EntityType.entries.find { it.name == label }
                    },
                    confidenceThreshold = 0.7
                )
            )
            
            when (resolution) {
                is EntityResolutionResult.Resolved -> {
                    // Update the candidate with the resolved ID
                    val resolvedCandidate = EntityCandidate(
                        id = resolution.entityId,
                        labels = representative.labels,
                        properties = representative.properties,
                        confidence = representative.confidence * resolution.confidence
                    )
                    resolved.add(resolvedCandidate)
                    
                    // Map all mentions to this ID
                    group.forEach { candidate ->
                        candidate.mentions.forEach { mention ->
                            mentionToId[mention] = resolution.entityId
                        }
                    }
                }
                
                is EntityResolutionResult.Ambiguous -> {
                    // For now, create a new entity if ambiguous
                    val newId = "entity-${Uuid.random()}"
                    val resolvedCandidate = EntityCandidate(
                        id = newId,
                        labels = representative.labels,
                        properties = representative.properties,
                        confidence = representative.confidence * 0.8 // Lower confidence due to ambiguity
                    )
                    resolved.add(resolvedCandidate)
                    
                    group.forEach { candidate ->
                        candidate.mentions.forEach { mention ->
                            mentionToId[mention] = newId
                        }
                    }
                }
                
                is EntityResolutionResult.Failed -> {
                    // Skip this entity
                    continue
                }
            }
        }
        
        // Store the mention mapping in episode metadata for relation resolution
        @Suppress("UNCHECKED_CAST")
        (episode.metadata as MutableMap<String, Any>)["mentionToEntityId"] = mentionToId
        
        return resolved
    }
    
    private fun updateRelationIds(
        relations: List<RelationCandidate>,
        originalEntities: List<InternalEntityCandidate>,
        resolvedEntities: List<EntityCandidate>
    ): List<RelationCandidate> {
        // Create a mapping from mentions to resolved entity IDs
        val mentionToId = mutableMapOf<String, String>()
        
        resolvedEntities.forEach { entity ->
            val name = entity.properties["name"]?.toString()
            if (name != null) {
                mentionToId[name] = entity.id
            }
            // Also map aliases
            (entity.properties["aliases"] as? List<*>)?.forEach { alias ->
                mentionToId[alias.toString()] = entity.id
            }
        }
        
        // Update relations with resolved IDs
        return relations.mapNotNull { relation ->
            val fromId = mentionToId[relation.from]
            val toId = mentionToId[relation.to]
            
            if (fromId != null && toId != null) {
                relation.copy(from = fromId, to = toId)
            } else {
                // Skip relations where we couldn't resolve entities
                null
            }
        }
    }
    
    private fun createExampleExtraction() = EnhancedKnowledgeExtraction(
        entities = listOf(
            EnhancedExtractedEntity(
                mainMention = "Steve",
                mentions = listOf("Steve", "he", "the player"),
                type = "PERSON",
                confidence = 0.95,
                aliases = listOf("SteveTheBuilder"),
                properties = mapOf("role" to "leader")
            )
        ),
        relations = listOf(
            EnhancedExtractedRelation(
                fromMention = "Steve",
                toMention = "Mountain Fortress",
                type = "OWNS",
                confidence = 0.9,
                properties = mapOf("since" to "2024-01-15"),
                validFrom = "2024-01-15"
            )
        ),
        updates = emptyList()
    )
}

/**
 * Data class for LLM knowledge extraction - enhanced version
 */
@Serializable
internal data class EnhancedKnowledgeExtraction(
    val entities: List<EnhancedExtractedEntity>,
    val relations: List<EnhancedExtractedRelation>,
    val updates: List<EnhancedExtractedUpdate>
)

@Serializable
internal data class EnhancedExtractedEntity(
    val mainMention: String,
    val mentions: List<String>,
    val type: String,
    val confidence: Double,
    val aliases: List<String>? = null,
    val properties: Map<String, String> = emptyMap()
)

@Serializable
internal data class EnhancedExtractedRelation(
    val fromMention: String,
    val toMention: String,
    val type: String,
    val confidence: Double,
    val properties: Map<String, String> = emptyMap(),
    val validFrom: String? = null,
    val validTo: String? = null
)

@Serializable
internal data class EnhancedExtractedUpdate(
    val targetMention: String,
    val property: String,
    val value: String,
    val operation: UpdateOperation,
    val confidence: Double = 1.0
)

// Internal data structures for processing

internal data class RawKnowledgeExtraction(
    val entities: List<InternalEntityCandidate>,
    val relations: List<RelationCandidate>,
    val updates: List<PropertyUpdate>,
    val temporalContext: TemporalContext?
)

internal data class InternalEntityCandidate(
    val id: String,
    val labels: Set<String>,
    val properties: Map<String, Any>,
    val confidence: Double,
    val mentions: List<String>
)

internal data class TemporalContext(
    val referenceTime: kotlinx.datetime.Instant,
    val validFrom: kotlinx.datetime.Instant? = null,
    val validTo: kotlinx.datetime.Instant? = null,
    val stateChanges: List<String> = emptyList()
)

/**
 * Extension to access GraphMemory from agent context
 */
public fun AIAgentContextBase.graphMemory(): GraphMemory = featureOrThrow(GraphMemory.Feature)

/**
 * Extension for convenient memory operations
 */
public suspend fun <T> AIAgentContextBase.withGraphMemory(
    action: suspend GraphMemory.() -> T
): T = graphMemory().action()