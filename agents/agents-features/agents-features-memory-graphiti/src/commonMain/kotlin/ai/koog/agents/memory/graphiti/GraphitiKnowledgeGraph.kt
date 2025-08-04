package ai.koog.agents.memory.graphiti

import ai.koog.agents.memory.feature.GraphMemory
import ai.koog.agents.memory.graph.*
import io.ktor.client.*
import io.ktor.client.call.*
import io.ktor.client.plugins.contentnegotiation.*
import io.ktor.client.request.*
import io.ktor.http.*
import io.ktor.serialization.kotlinx.json.*
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.datetime.Instant
import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Graphiti-based implementation of KnowledgeGraph.
 * 
 * Graphiti is a temporal knowledge graph system that provides:
 * - Bi-temporal data model (valid_at vs ingested_at)
 * - Hybrid search (BM25 + embeddings + graph)
 * - Center-node reranking for entity queries
 * - Automatic fact extraction and relation inference
 * 
 * This implementation connects to a Graphiti service via HTTP API.
 * 
 * Example usage:
 * ```kotlin
 * val config = GraphitiConfig(
 *     baseUrl = "http://localhost:8000",
 *     apiKey = "your-api-key"
 * )
 * val graph = GraphitiKnowledgeGraph(config)
 * 
 * // Use with GraphMemory feature
 * install(GraphMemory) {
 *     graph = graph
 *     episodeProcessor = MyGameEpisodeProcessor()
 * }
 * ```
 */
public class GraphitiKnowledgeGraph(
    private val config: GraphitiConfig,
    private val httpClient: HttpClient = createDefaultHttpClient()
) : KnowledgeGraph {
    
    private val baseUrl = config.baseUrl.trimEnd('/')
    private val apiKey = config.apiKey
    
    override suspend fun ingest(episode: Episode): IngestionResult {
        val request = GraphitiIngestionRequest(
            episodes = listOf(
                GraphitiEpisode(
                    content = episode.content,
                    timestamp = episode.timestamp.toString(),
                    source = episode.source.name,
                    metadata = episode.metadata,
                    references = episode.references
                )
            )
        )
        
        val response = httpClient.post("$baseUrl/ingest") {
            contentType(ContentType.Application.Json)
            if (apiKey != null) {
                header("Authorization", "Bearer $apiKey")
            }
            setBody(request)
        }.body<GraphitiIngestionResponse>()
        
        return IngestionResult(
            createdNodes = response.createdNodes,
            createdEdges = response.createdEdges,
            updatedNodes = response.updatedNodes,
            conflicts = response.conflicts.map { conflict ->
                ConflictResolution(
                    existingKnowledge = parseKnowledge(conflict.existing),
                    newKnowledge = parseKnowledge(conflict.new),
                    resolution = ResolutionStrategy.valueOf(conflict.resolution),
                    reason = conflict.reason
                )
            }
        )
    }
    
    override suspend fun query(request: KnowledgeRequest): List<Knowledge> {
        val graphitiRequest = when (request) {
            is KnowledgeRequest.EntityCentric -> GraphitiQueryRequest(
                type = "entity_centric",
                centerNode = request.centerNode,
                traversal = serializeTraversal(request.traversal),
                filters = request.filters.map { serializeFilter(it) },
                at = request.at?.toString(),
                limit = request.limit
            )
            
            is KnowledgeRequest.Pattern -> GraphitiQueryRequest(
                type = "pattern",
                pattern = request.pattern,
                parameters = request.parameters,
                at = request.at?.toString(),
                limit = request.limit
            )
            
            is KnowledgeRequest.Semantic -> GraphitiQueryRequest(
                type = "semantic",
                query = request.query,
                scope = request.scope.name.lowercase(),
                at = request.at?.toString(),
                limit = request.limit
            )
            
            is KnowledgeRequest.Temporal -> GraphitiQueryRequest(
                type = "temporal",
                start = request.start.toString(),
                end = request.end.toString(),
                entityFilter = request.entityFilter,
                eventTypes = request.eventTypes,
                includeDeleted = request.includeDeleted
            )
        }
        
        val response = httpClient.post("$baseUrl/query") {
            contentType(ContentType.Application.Json)
            if (apiKey != null) {
                header("Authorization", "Bearer $apiKey")
            }
            setBody(graphitiRequest)
        }.body<GraphitiQueryResponse>()
        
        return response.results.map { parseKnowledge(it) }
    }
    
    override suspend fun evolve(context: EvolutionContext) {
        val request = GraphitiEvolutionRequest(
            currentTime = context.currentTime.toString(),
            decayRules = context.decayRules.map { it::class.simpleName ?: "UnknownRule" },
            consolidationRules = context.consolidationRules.map { it::class.simpleName ?: "UnknownRule" }
        )
        
        httpClient.post("$baseUrl/evolve") {
            contentType(ContentType.Application.Json)
            if (apiKey != null) {
                header("Authorization", "Bearer $apiKey")
            }
            setBody(request)
        }
    }
    
    override suspend fun stats(): GraphStats {
        val response = httpClient.get("$baseUrl/stats") {
            if (apiKey != null) {
                header("Authorization", "Bearer $apiKey")
            }
        }.body<GraphitiStatsResponse>()
        
        return GraphStats(
            nodeCount = response.nodeCount,
            edgeCount = response.edgeCount,
            episodeCount = response.episodeCount,
            lastIngestion = response.lastIngestion?.let { Instant.parse(it) },
            lastEvolution = response.lastEvolution?.let { Instant.parse(it) },
            nodesByLabel = response.nodesByLabel,
            edgesByType = response.edgesByType
        )
    }
    
    /**
     * Stream episodes for continuous ingestion
     */
    public fun streamIngest(episodes: Flow<Episode>): Flow<IngestionResult> = flow {
        episodes.collect { episode ->
            emit(ingest(episode))
        }
    }
    
    /**
     * Hybrid search with Reciprocal Rank Fusion
     */
    public suspend fun hybridSearch(
        query: String,
        limit: Int = 20,
        alpha: Double = 0.5,
        useRRF: Boolean = true,
        useMMR: Boolean = false,
        mmrLambda: Double = 0.5
    ): List<Knowledge> {
        val request = GraphitiHybridSearchRequest(
            query = query,
            limit = limit,
            alpha = alpha,
            useRrf = useRRF,
            useMmr = useMMR,
            mmrLambda = mmrLambda
        )
        
        val response = httpClient.post("$baseUrl/search/hybrid") {
            contentType(ContentType.Application.Json)
            if (apiKey != null) {
                header("Authorization", "Bearer $apiKey")
            }
            setBody(request)
        }.body<GraphitiQueryResponse>()
        
        return response.results.map { parseKnowledge(it) }
    }
    
    private fun parseKnowledge(data: GraphitiKnowledgeData): Knowledge {
        return when (data.type) {
            "entity" -> Knowledge.Entity(
                id = data.id,
                labels = data.labels?.toSet() ?: emptySet(),
                properties = data.properties ?: emptyMap(),
                confidence = data.confidence,
                timestamp = Instant.parse(data.timestamp),
                provenance = parseProvenance(data.provenance)
            )
            
            "relation" -> Knowledge.Relation(
                id = data.id,
                type = data.relationType ?: "UNKNOWN",
                from = data.from ?: "",
                to = data.to ?: "",
                properties = data.properties ?: emptyMap(),
                confidence = data.confidence,
                timestamp = Instant.parse(data.timestamp),
                provenance = parseProvenance(data.provenance)
            )
            
            "composite" -> Knowledge.Composite(
                id = data.id,
                summary = data.summary ?: "",
                elements = data.elements?.map { parseKnowledge(it) } ?: emptyList(),
                confidence = data.confidence,
                timestamp = Instant.parse(data.timestamp),
                provenance = parseProvenance(data.provenance)
            )
            
            else -> throw IllegalArgumentException("Unknown knowledge type: ${data.type}")
        }
    }
    
    private fun parseProvenance(data: List<GraphitiProvenanceData>?): List<ProvenanceItem> {
        return data?.map { item ->
            when (item.type) {
                "episode" -> ProvenanceItem.Episode(
                    item.episodeId ?: "",
                    EpisodeSource.valueOf(item.source ?: "EXTERNAL")
                )
                "inference" -> ProvenanceItem.Inference(
                    item.rule ?: "",
                    item.confidence ?: 0.0
                )
                "external" -> ProvenanceItem.External(
                    item.source ?: "",
                    item.reference ?: ""
                )
                else -> throw IllegalArgumentException("Unknown provenance type: ${item.type}")
            }
        } ?: emptyList()
    }
    
    private fun serializeTraversal(traversal: Traversal): Map<String, @Contextual Any> {
        return when (traversal) {
            is Traversal.Outgoing -> mapOf(
                "direction" to "outgoing",
                "depth" to traversal.depth,
                "relation_types" to traversal.relationTypes
            )
            is Traversal.Incoming -> mapOf(
                "direction" to "incoming",
                "depth" to traversal.depth,
                "relation_types" to traversal.relationTypes
            )
            is Traversal.Bidirectional -> mapOf(
                "direction" to "bidirectional",
                "depth" to traversal.depth,
                "relation_types" to traversal.relationTypes
            )
            is Traversal.ShortestPath -> mapOf(
                "direction" to "shortest_path",
                "to" to traversal.to,
                "max_depth" to traversal.maxDepth
            )
        }
    }
    
    private fun serializeFilter(filter: Filter): Map<String, @Contextual Any?> {
        return when (filter) {
            is Filter.HasLabel -> mapOf(
                "type" to "has_label",
                "labels" to filter.labels
            )
            is Filter.HasProperty -> mapOf(
                "type" to "has_property",
                "key" to filter.key,
                "value" to filter.value
            )
            is Filter.PropertyRange -> mapOf(
                "type" to "property_range",
                "key" to filter.key,
                "min" to filter.min,
                "max" to filter.max
            )
            is Filter.Custom -> mapOf(
                "type" to "custom",
                "predicate" to filter.predicate
            )
        }
    }
    
    public companion object {
        private fun createDefaultHttpClient(): HttpClient {
            return HttpClient {
                install(ContentNegotiation) {
                    json(Json {
                        ignoreUnknownKeys = true
                        isLenient = true
                    })
                }
            }
        }
    }
}

/**
 * Configuration for Graphiti connection
 */
public data class GraphitiConfig(
    val baseUrl: String,
    val apiKey: String? = null,
    val embeddings: EmbeddingsConfig? = null
)

/**
 * Configuration for embeddings provider
 */
public data class EmbeddingsConfig(
    val provider: String = "openai",
    val model: String = "text-embedding-ada-002",
    val dimensions: Int = 1536
)

// Graphiti API data classes

@Serializable
private data class GraphitiIngestionRequest(
    val episodes: List<GraphitiEpisode>
)

@Serializable
private data class GraphitiEpisode(
    val content: String,
    val timestamp: String,
    val source: String,
    val metadata: Map<String, @Contextual Any> = emptyMap(),
    val references: List<String> = emptyList()
)

@Serializable
private data class GraphitiIngestionResponse(
    @SerialName("created_nodes") val createdNodes: List<String>,
    @SerialName("created_edges") val createdEdges: List<String>,
    @SerialName("updated_nodes") val updatedNodes: List<String>,
    val conflicts: List<GraphitiConflict>
)

@Serializable
private data class GraphitiConflict(
    val existing: GraphitiKnowledgeData,
    val new: GraphitiKnowledgeData,
    val resolution: String,
    val reason: String
)

@Serializable
private data class GraphitiQueryRequest(
    val type: String,
    @SerialName("center_node") val centerNode: String? = null,
    val traversal: Map<String, @Contextual Any>? = null,
    val filters: List<Map<String, @Contextual Any?>>? = null,
    val at: String? = null,
    val limit: Int? = null,
    val pattern: String? = null,
    val parameters: Map<String, @Contextual Any>? = null,
    val query: String? = null,
    val scope: String? = null,
    val start: String? = null,
    val end: String? = null,
    @SerialName("entity_filter") val entityFilter: List<String>? = null,
    @SerialName("event_types") val eventTypes: List<String>? = null,
    @SerialName("include_deleted") val includeDeleted: Boolean? = null
)

@Serializable
private data class GraphitiQueryResponse(
    val results: List<GraphitiKnowledgeData>
)

@Serializable
private data class GraphitiKnowledgeData(
    val id: String,
    val type: String,
    val labels: List<String>? = null,
    val properties: Map<String, @Contextual Any>? = null,
    val confidence: Double,
    val timestamp: String,
    val provenance: List<GraphitiProvenanceData>? = null,
    @SerialName("relation_type") val relationType: String? = null,
    val from: String? = null,
    val to: String? = null,
    val summary: String? = null,
    val elements: List<GraphitiKnowledgeData>? = null
)

@Serializable
private data class GraphitiProvenanceData(
    val type: String,
    @SerialName("episode_id") val episodeId: String? = null,
    val source: String? = null,
    val rule: String? = null,
    val confidence: Double? = null,
    val reference: String? = null
)

@Serializable
private data class GraphitiEvolutionRequest(
    @SerialName("current_time") val currentTime: String,
    @SerialName("decay_rules") val decayRules: List<String>,
    @SerialName("consolidation_rules") val consolidationRules: List<String>
)

@Serializable
private data class GraphitiStatsResponse(
    @SerialName("node_count") val nodeCount: Long,
    @SerialName("edge_count") val edgeCount: Long,
    @SerialName("episode_count") val episodeCount: Long,
    @SerialName("last_ingestion") val lastIngestion: String?,
    @SerialName("last_evolution") val lastEvolution: String?,
    @SerialName("nodes_by_label") val nodesByLabel: Map<String, Long>,
    @SerialName("edges_by_type") val edgesByType: Map<String, Long>
)

@Serializable
private data class GraphitiHybridSearchRequest(
    val query: String,
    val limit: Int = 20,
    val alpha: Double = 0.5,
    @SerialName("use_rrf") val useRrf: Boolean = true,
    @SerialName("use_mmr") val useMmr: Boolean = false,
    @SerialName("mmr_lambda") val mmrLambda: Double = 0.5
)

/**
 * Extension for GraphMemory to access Graphiti-specific features
 */
@OptIn(ai.koog.agents.core.annotation.InternalAgentsApi::class)
public suspend fun hybridSearch(
    memory: GraphMemory,
    query: String,
    limit: Int = 20,
    alpha: Double = 0.5
): List<Knowledge> {
    val graphiti = memory.graph as? GraphitiKnowledgeGraph 
        ?: throw IllegalStateException("GraphMemory is not backed by Graphiti")
    
    return graphiti.hybridSearch(query, limit, alpha)
}