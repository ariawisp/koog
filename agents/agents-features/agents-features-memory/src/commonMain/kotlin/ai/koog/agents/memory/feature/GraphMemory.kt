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
                val episodeProcessor = config.episodeProcessor ?: DefaultEpisodeProcessor(agentContext.llm)
                val graph = config.graph ?: InMemoryKnowledgeGraph(
                    clock = config.clock,
                    episodeProcessor = episodeProcessor
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
 * Default episode processor using LLM
 */
internal class DefaultEpisodeProcessor(
    private val llm: AIAgentLLMContext
) : EpisodeProcessor {
    override suspend fun process(episode: Episode): ProcessedEpisode {
        // Use structured output to extract entities and relations
        val structuredData = JsonStructuredData.createJsonStructure<KnowledgeExtraction>(
            id = "knowledge-extraction",
            examples = listOf(
                KnowledgeExtraction(
                    entities = listOf(
                        ExtractedEntity(
                            id = "player-steve",
                            labels = setOf("Player"),
                            properties = mapOf("name" to "Steve")
                        )
                    ),
                    relations = listOf(
                        ExtractedRelation(
                            from = "player-steve",
                            relationTo = "base-fortress",
                            type = "OWNS",
                            properties = mapOf("since" to "2024")
                        )
                    ),
                    updates = emptyList()
                )
            )
        )
        
        val extraction = llm.writeSession {
            // Add extraction instructions to the prompt
            prompt = Prompt.build(prompt) {
                user("""Extract entities and their relationships from the following text.
                    
                    Text: ${episode.content}
                    
                    Instructions:
                    - Identify all entities (people, places, organizations, concepts, etc.)
                    - Identify relationships between entities
                    - Extract any property updates (changes to existing entities)
                    - Use clear, consistent naming for entities
                    - Preserve important details as properties
                    - Ensure each entity has a unique ID
                    - Use appropriate relationship types (e.g., OWNS, BELONGS_TO, LOCATED_AT, etc.)
                    
                    Return the result as a JSON object with the following structure:
                    {
                      "entities": [
                        {
                          "id": "unique-entity-id",
                          "labels": ["Entity", "Type"],
                          "properties": {"key": "value"}
                        }
                      ],
                      "relations": [
                        {
                          "from": "entity-id-1",
                          "relationTo": "entity-id-2",
                          "type": "RELATIONSHIP_TYPE",
                          "properties": {"key": "value"}
                        }
                      ],
                      "updates": [
                        {
                          "targetId": "entity-id",
                          "property": "property-name",
                          "value": "new-value",
                          "operation": "SET"
                        }
                      ]
                    }
                    """.trimIndent())
            }
            
            // Request structured output
            val result = requestLLMStructured(structuredData)
            result.getOrThrow().structure
        }
        
        return ProcessedEpisode(
            entities = extraction.entities.map { entity ->
                EntityCandidate(
                    id = entity.id,
                    labels = entity.labels,
                    properties = entity.properties
                )
            },
            relations = extraction.relations.map { relation ->
                RelationCandidate(
                    from = relation.from,
                    to = relation.relationTo,
                    type = relation.type,
                    properties = relation.properties
                )
            },
            updates = extraction.updates.map { update ->
                PropertyUpdate(
                    targetId = update.targetId,
                    property = update.property,
                    value = update.value,
                    operation = update.operation
                )
            }
        )
    }
}

/**
 * Data class for LLM knowledge extraction
 */
@Serializable
internal data class KnowledgeExtraction(
    val entities: List<ExtractedEntity>,
    val relations: List<ExtractedRelation>,
    val updates: List<ExtractedUpdate>
)

@Serializable
internal data class ExtractedEntity(
    val id: String,
    val labels: Set<String>,
    val properties: Map<String, String>
)

@Serializable
internal data class ExtractedRelation(
    val from: String,
    val relationTo: String,
    val type: String,
    val properties: Map<String, String>
)

@Serializable
internal data class ExtractedUpdate(
    val targetId: String,
    val property: String,
    val value: String,
    val operation: UpdateOperation
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