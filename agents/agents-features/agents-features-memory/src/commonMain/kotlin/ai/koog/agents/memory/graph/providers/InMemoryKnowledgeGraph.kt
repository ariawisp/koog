package ai.koog.agents.memory.graph.providers

import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.feature.EpisodeProcessor
import ai.koog.agents.memory.feature.UpdateOperation
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlin.jvm.JvmInline
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Configuration for InMemoryKnowledgeGraph
 */
public data class InMemoryConfig(
    val minConfidence: Double = 0.5,
    val maxNodes: Int = 100_000,
    val maxEdges: Int = 500_000,
    val semanticWeights: SemanticWeights = SemanticWeights(),
    val tokenizer: Tokenizer = DefaultTokenizer
)

/**
 * Weights for semantic search scoring
 */
public data class SemanticWeights(
    val labelMatch: Double = 2.0,
    val exactPhraseMatch: Double = 1.5,
    val termMatch: Double = 1.0,
    val propertyKeyMatch: Double = 1.0,
    val individualTermMatch: Double = 0.5
)

/**
 * Interface for tokenizing text
 */
public interface Tokenizer {
    public fun tokenize(text: String): Set<String>
}

/**
 * Default tokenizer implementation
 */
public object DefaultTokenizer : Tokenizer {
    override fun tokenize(text: String): Set<String> {
        return text.lowercase()
            .split(Regex("\\W+"))
            .filter { it.isNotBlank() && it.length > 2 }
            .toSet()
    }
}

// Data classes moved to top level for reusability

internal data class NodeData(
    val id: NodeId,
    val labels: Set<String>,
    val properties: Map<String, Any>,
    val createdAt: Instant,
    val confidence: Double,
    val provenance: List<ProvenanceItem>,
    val validFrom: Instant? = null,
    val validTo: Instant? = null
) {
    // For backward compatibility, use createdAt as validFrom if not specified
    val effectiveValidFrom: Instant get() = validFrom ?: createdAt
    val effectiveValidTo: Instant? get() = validTo
}

internal data class EdgeData(
    val id: EdgeId,
    val type: String,
    val from: NodeId,
    val to: NodeId,
    val properties: Map<String, Any>,
    val createdAt: Instant,
    val confidence: Double,
    val provenance: List<ProvenanceItem>,
    val validFrom: Instant? = null,
    val validTo: Instant? = null
) {
    // For backward compatibility, use createdAt as validFrom if not specified
    val effectiveValidFrom: Instant get() = validFrom ?: createdAt
    val effectiveValidTo: Instant? get() = validTo
}

/**
 * Thread-safe snapshot of the graph state
 */
internal data class GraphSnapshot(
    val nodes: Map<NodeId, NodeData>,
    val edges: Map<EdgeId, EdgeData>,
    val nodesByLabel: Map<String, Set<NodeId>>,
    val edgesByType: Map<String, Set<EdgeId>>,
    val outgoingEdges: Map<NodeId, Set<EdgeId>>,
    val incomingEdges: Map<NodeId, Set<EdgeId>>,
    val nodeTermIndex: Map<String, Set<NodeId>>,
    val edgeTermIndex: Map<String, Set<EdgeId>>
)

/**
 * In-memory implementation of KnowledgeGraph optimized for performance.
 * Uses snapshot-based concurrency, inverted indexing for fast searches,
 * and efficient BFS traversal.
 */
@OptIn(ExperimentalUuidApi::class)
public class InMemoryKnowledgeGraph(
    private val config: InMemoryConfig = InMemoryConfig(),
    private val clock: Clock = Clock.System,
    private val episodeProcessor: EpisodeProcessor? = null
) : KnowledgeGraph {
    
    private val mutex = Mutex()
    private val nodes = mutableMapOf<NodeId, NodeData>()
    private val edges = mutableMapOf<EdgeId, EdgeData>()
    private val episodes = mutableMapOf<String, Episode>()
    private val nodesByLabel = mutableMapOf<String, MutableSet<NodeId>>()
    private val edgesByType = mutableMapOf<String, MutableSet<EdgeId>>()
    private val outgoingEdges = mutableMapOf<NodeId, MutableSet<EdgeId>>()
    private val incomingEdges = mutableMapOf<NodeId, MutableSet<EdgeId>>()
    
    // Inverted indexes for semantic search
    private val nodeTermIndex = mutableMapOf<String, MutableSet<NodeId>>()
    private val edgeTermIndex = mutableMapOf<String, MutableSet<EdgeId>>()
    
    private var lastIngestionTime: Instant? = null
    private var lastEvolutionTime: Instant? = null
    
    /**
     * Creates a snapshot of the current graph state for lock-free queries
     */
    private fun createSnapshot(): GraphSnapshot {
        return GraphSnapshot(
            nodes = nodes.toMap(),
            edges = edges.toMap(),
            nodesByLabel = nodesByLabel.mapValues { it.value.toSet() },
            edgesByType = edgesByType.mapValues { it.value.toSet() },
            outgoingEdges = outgoingEdges.mapValues { it.value.toSet() },
            incomingEdges = incomingEdges.mapValues { it.value.toSet() },
            nodeTermIndex = nodeTermIndex.mapValues { it.value.toSet() },
            edgeTermIndex = edgeTermIndex.mapValues { it.value.toSet() }
        )
    }
    
    override suspend fun ingest(episode: Episode): IngestionResult = mutex.withLock {
        val episodeId = generateId("episode")
        episodes[episodeId] = episode
        lastIngestionTime = clock.now()
        
        val createdNodeIds = mutableListOf<NodeId>()
        val createdEdgeIds = mutableListOf<EdgeId>()
        val updatedNodeIds = mutableListOf<NodeId>()
        
        if (episodeProcessor != null) {
            // Process episode to extract entities and relations
            val processed = episodeProcessor.process(episode)
            
            // Create or update entities
            processed.entities.forEach { candidate ->
                if (candidate.confidence >= config.minConfidence) { // Filter by confidence threshold
                    val nodeId = candidate.id
                    val node = NodeData(
                        id = nodeId,
                        labels = candidate.labels,
                        properties = candidate.properties,
                        createdAt = episode.timestamp,
                        confidence = candidate.confidence,
                        provenance = listOf(ProvenanceItem.Episode(episodeId, episode.source)),
                        validFrom = episode.validFrom ?: episode.timestamp,
                        validTo = episode.validTo
                    )
                    
                    val existing = nodes[nodeId]
                    if (existing == null) {
                        nodes[nodeId] = node
                        candidate.labels.forEach { label ->
                            nodesByLabel.getOrPut(label) { mutableSetOf() }.add(nodeId)
                        }
                        indexNode(nodeId, node)
                        createdNodeIds.add(nodeId)
                    } else {
                        // Unindex old node
                        unindexNode(nodeId, existing)
                        
                        // Merge properties from new node into existing
                        val mergedProps = existing.properties + candidate.properties
                        val mergedLabels = existing.labels + candidate.labels
                        
                        // Update temporal bounds if needed
                        val newValidFrom = minOf(existing.effectiveValidFrom, node.effectiveValidFrom)
                        val newValidTo = when {
                            existing.validTo == null || node.validTo == null -> null
                            else -> maxOf(existing.validTo, node.validTo)
                        }
                        
                        val updatedNode = existing.copy(
                            labels = mergedLabels,
                            properties = mergedProps,
                            provenance = existing.provenance + node.provenance,
                            validFrom = newValidFrom,
                            validTo = newValidTo
                        )
                        nodes[nodeId] = updatedNode
                        
                        // Update label indexes
                        mergedLabels.forEach { label ->
                            nodesByLabel.getOrPut(label) { mutableSetOf() }.add(nodeId)
                        }
                        
                        // Re-index updated node
                        indexNode(nodeId, updatedNode)
                        updatedNodeIds.add(nodeId)
                    }
                }
            }
            
            // Create relations
            processed.relations.forEach { candidate ->
                if (candidate.confidence >= config.minConfidence) { // Filter by confidence threshold
                    val edgeId = generateId("edge")
                    val edge = EdgeData(
                        id = edgeId,
                        type = candidate.type,
                        from = candidate.from,
                        to = candidate.to,
                        properties = candidate.properties,
                        createdAt = episode.timestamp,
                        confidence = candidate.confidence,
                        provenance = listOf(ProvenanceItem.Episode(episodeId, episode.source)),
                        validFrom = episode.validFrom ?: episode.timestamp,
                        validTo = episode.validTo
                    )
                    
                    edges[edgeId] = edge
                    edgesByType.getOrPut(candidate.type) { mutableSetOf() }.add(edgeId)
                    outgoingEdges.getOrPut(candidate.from) { mutableSetOf() }.add(edgeId)
                    incomingEdges.getOrPut(candidate.to) { mutableSetOf() }.add(edgeId)
                    indexEdge(edgeId, edge)
                    createdEdgeIds.add(edgeId)
                }
            }
            
            // Apply property updates
            processed.updates.forEach { update ->
                nodes[update.targetId]?.let { node ->
                    val updatedNode = when (update.operation) {
                        UpdateOperation.SET -> {
                            node.copy(properties = node.properties + (update.property to update.value))
                        }
                        UpdateOperation.DELETE -> {
                            node.copy(properties = node.properties - update.property)
                        }
                        UpdateOperation.APPEND -> {
                            val currentValue = node.properties[update.property]
                            val newValue = when (currentValue) {
                                is List<*> -> currentValue + update.value
                                is String -> currentValue + update.value
                                else -> update.value
                            }
                            node.copy(properties = node.properties + (update.property to newValue))
                        }
                        UpdateOperation.INCREMENT -> {
                            val currentValue = node.properties[update.property] as? Number ?: 0
                            val increment = (update.value as? Number)?.toDouble() ?: 0.0
                            node.copy(properties = node.properties + (update.property to (currentValue.toDouble() + increment)))
                        }
                    }
                    nodes[update.targetId] = updatedNode
                    updatedNodeIds.add(update.targetId)
                }
            }
        } else {
            // Fallback: create a simple node for the episode content
            val nodeId = generateId("node")
            val node = NodeData(
                id = nodeId,
                labels = setOf("Episode"),
                properties = mapOf(
                    "content" to episode.content,
                    "source" to episode.source.name,
                    "timestamp" to episode.timestamp.toEpochMilliseconds()
                ) + episode.metadata,
                createdAt = episode.timestamp,
                confidence = 1.0,
                provenance = listOf(ProvenanceItem.Episode(episodeId, episode.source)),
                validFrom = episode.validFrom ?: episode.timestamp,
                validTo = episode.validTo
            )
            
            nodes[nodeId] = node
            nodesByLabel.getOrPut("Episode") { mutableSetOf() }.add(nodeId)
            indexNode(nodeId, node)
            createdNodeIds.add(nodeId)
        }
        
        IngestionResult(
            createdNodes = createdNodeIds,
            createdEdges = createdEdgeIds,
            updatedNodes = updatedNodeIds,
            conflicts = emptyList()
        )
    }
    
    override suspend fun query(request: KnowledgeRequest): List<Knowledge> {
        // Take a quick snapshot under lock, then query lock-free
        val snapshot = mutex.withLock { createSnapshot() }
        
        return when (request) {
            is KnowledgeRequest.EntityCentric -> queryEntityCentric(request, snapshot)
            is KnowledgeRequest.Pattern -> queryPattern(request, snapshot)
            is KnowledgeRequest.Semantic -> querySemantic(request, snapshot)
            is KnowledgeRequest.Temporal -> queryTemporal(request, snapshot)
        }
    }
    
    private fun queryEntityCentric(request: KnowledgeRequest.EntityCentric, snapshot: GraphSnapshot): List<Knowledge> {
        val centerNode = snapshot.nodes[request.centerNode] ?: return emptyList()
        
        // Check if center node is valid at the requested time
        if (request.at != null) {
            val validFrom = centerNode.effectiveValidFrom
            val validTo = centerNode.effectiveValidTo ?: Instant.DISTANT_FUTURE
            if (request.at < validFrom || request.at >= validTo) {
                return emptyList() // Node not valid at requested time
            }
        }
        
        val visited = mutableSetOf<NodeId>()
        val results = mutableListOf<Knowledge>()
        
        // BFS traversal with ArrayDeque for O(1) operations
        val queue = ArrayDeque<Pair<NodeId, Int>>()
        queue.add(request.centerNode to 0)
        
        while (queue.isNotEmpty()) {
            val (nodeId, depth) = queue.removeFirst()
            if (nodeId in visited || depth > request.traversal.depth()) continue
            
            visited.add(nodeId)
            
            snapshot.nodes[nodeId]?.let { node ->
                // Check temporal validity if point-in-time query
                if (request.at != null) {
                    val validFrom = node.effectiveValidFrom
                    val validTo = node.effectiveValidTo ?: Instant.DISTANT_FUTURE
                    if (request.at < validFrom || request.at >= validTo) {
                        return@let // Skip nodes not valid at requested time
                    }
                }
                
                if (matchesFilters(node, request.filters)) {
                    results.add(nodeToKnowledge(node))
                }
            }
            
            // Add connected nodes to queue
            when (request.traversal) {
                is Traversal.Outgoing, is Traversal.Bidirectional -> {
                    snapshot.outgoingEdges[nodeId]?.forEach { edgeId ->
                        snapshot.edges[edgeId]?.let { edge ->
                            // Check edge temporal validity
                            if (request.at != null) {
                                val validFrom = edge.effectiveValidFrom
                                val validTo = edge.effectiveValidTo ?: Instant.DISTANT_FUTURE
                                if (request.at < validFrom || request.at >= validTo) {
                                    return@let // Skip edges not valid at requested time
                                }
                            }
                            
                            if (shouldIncludeEdge(edge, request.traversal)) {
                                queue.addLast(edge.to to depth + 1)
                                results.add(edgeToKnowledge(edge))
                            }
                        }
                    }
                }
                else -> {}
            }
            
            when (request.traversal) {
                is Traversal.Incoming, is Traversal.Bidirectional -> {
                    snapshot.incomingEdges[nodeId]?.forEach { edgeId ->
                        snapshot.edges[edgeId]?.let { edge ->
                            // Check edge temporal validity
                            if (request.at != null) {
                                val validFrom = edge.effectiveValidFrom
                                val validTo = edge.effectiveValidTo ?: Instant.DISTANT_FUTURE
                                if (request.at < validFrom || request.at >= validTo) {
                                    return@let // Skip edges not valid at requested time
                                }
                            }
                            
                            if (shouldIncludeEdge(edge, request.traversal)) {
                                queue.addLast(edge.from to depth + 1)
                                results.add(edgeToKnowledge(edge))
                            }
                        }
                    }
                }
                else -> {}
            }
        }
        
        // Deduplicate and limit results
        return results.distinctBy { it.id }.take(request.limit)
    }
    
    private fun queryPattern(request: KnowledgeRequest.Pattern, snapshot: GraphSnapshot): List<Knowledge> {
        // Simplified: just return all nodes for now
        // Real implementation would parse the pattern and execute it
        return snapshot.nodes.values
            .filter { node -> request.at == null || node.createdAt <= request.at }
            .map { nodeToKnowledge(it) }
            .take(request.limit)
    }
    
    private fun querySemantic(request: KnowledgeRequest.Semantic, snapshot: GraphSnapshot): List<Knowledge> {
        val queryTerms = config.tokenizer.tokenize(request.query)
        
        // Use inverted index to find candidate nodes
        val candidateNodeIds = queryTerms.flatMap { term ->
            snapshot.nodeTermIndex[term] ?: emptySet()
        }.toSet()
        
        // Score and rank candidate nodes
        val scoredNodes = candidateNodeIds
            .mapNotNull { nodeId -> snapshot.nodes[nodeId]?.let { nodeId to it } }
            .filter { (_, node) -> request.at == null || node.createdAt <= request.at }
            .map { (nodeId, node) ->
                var score = 0.0
                val nodeTerms = extractNodeTerms(node)
                
                // Score based on term overlap
                queryTerms.forEach { queryTerm ->
                    if (queryTerm in nodeTerms) {
                        score += config.semanticWeights.termMatch
                    }
                }
                
                // Bonus for label matches
                node.labels.forEach { label ->
                    if (config.tokenizer.tokenize(label).any { it in queryTerms }) {
                        score += config.semanticWeights.labelMatch
                    }
                }
                
                // Check for exact phrase match in properties
                node.properties.values.forEach { value ->
                    if (value.toString().lowercase().contains(request.query.lowercase())) {
                        score += config.semanticWeights.exactPhraseMatch
                    }
                }
                
                node to score
            }
            .filter { (_, score) -> score > 0 }
            .sortedByDescending { (_, score) -> score }
            .take(request.limit)
            .map { (node, _) -> nodeToKnowledge(node) }
        
        // Use inverted index for edges too
        val candidateEdgeIds = queryTerms.flatMap { term ->
            snapshot.edgeTermIndex[term] ?: emptySet()
        }.toSet()
        
        val scoredEdges = candidateEdgeIds
            .mapNotNull { edgeId -> snapshot.edges[edgeId]?.let { edgeId to it } }
            .filter { (_, edge) -> request.at == null || edge.createdAt <= request.at }
            .map { (edgeId, edge) ->
                var score = 0.0
                val edgeTerms = extractEdgeTerms(edge)
                
                queryTerms.forEach { queryTerm ->
                    if (queryTerm in edgeTerms) {
                        score += config.semanticWeights.termMatch
                    }
                }
                
                if (config.tokenizer.tokenize(edge.type).any { it in queryTerms }) {
                    score += config.semanticWeights.labelMatch
                }
                
                edge to score
            }
            .filter { (_, score) -> score > 0 }
            .sortedByDescending { (_, score) -> score }
            .take(request.limit - scoredNodes.size)
            .map { (edge, _) -> edgeToKnowledge(edge) }
        
        return (scoredNodes + scoredEdges).distinctBy { it.id }
    }
    
    private fun queryTemporal(request: KnowledgeRequest.Temporal, snapshot: GraphSnapshot): List<Knowledge> {
        val results = mutableListOf<Knowledge>()
        
        // Query nodes that were valid during the time window
        snapshot.nodes.values
            .filter { node ->
                // Check if node was valid during the requested time window
                val nodeValidFrom = node.effectiveValidFrom
                val nodeValidTo = node.effectiveValidTo ?: Instant.DISTANT_FUTURE
                
                // Node is included if its validity period overlaps with the query window
                nodeValidFrom <= request.end && nodeValidTo >= request.start
            }
            .filter { node ->
                request.entityFilter.isEmpty() || node.id in request.entityFilter
            }
            .forEach { node ->
                results.add(nodeToKnowledge(node))
            }
        
        // Also query edges that were valid during the time window
        snapshot.edges.values
            .filter { edge ->
                // Check if edge was valid during the requested time window
                val edgeValidFrom = edge.effectiveValidFrom
                val edgeValidTo = edge.effectiveValidTo ?: Instant.DISTANT_FUTURE
                
                // Edge is included if its validity period overlaps with the query window
                edgeValidFrom <= request.end && edgeValidTo >= request.start
            }
            .filter { edge ->
                request.eventTypes.isEmpty() || edge.type in request.eventTypes
            }
            .forEach { edge ->
                results.add(edgeToKnowledge(edge))
            }
        
        return results
    }
    
    override suspend fun evolve(context: EvolutionContext): Unit = mutex.withLock {
        lastEvolutionTime = clock.now()
        
        // Apply decay rules
        val toRemove = mutableListOf<NodeId>()
        nodes.forEach { (id, node) ->
            context.decayRules.forEach { rule ->
                if (rule.shouldDecay(nodeToKnowledge(node), context.currentTime)) {
                    val decayed = rule.decay(nodeToKnowledge(node))
                    if (decayed == null) {
                        toRemove.add(id)
                    }
                }
            }
        }
        
        // Remove decayed nodes
        toRemove.forEach { nodeId ->
            removeNode(nodeId)
        }
        
        // Apply consolidation rules
        context.consolidationRules.forEach { rule ->
            // Group nodes by labels for efficient consolidation
            nodesByLabel.forEach { (label, nodeIds) ->
                val nodesForLabel = nodeIds.mapNotNull { nodes[it] }
                    .map { nodeToKnowledge(it) }
                
                // Check subsets of nodes to see if they should be consolidated
                if (nodesForLabel.size >= 2) {
                    // Simple approach: check pairs and small groups
                    val consolidated = mutableSetOf<NodeId>()
                    
                    for (i in nodesForLabel.indices) {
                        if (nodesForLabel[i].id in consolidated) continue
                        
                        val candidateGroup = mutableListOf(nodesForLabel[i])
                        
                        // Look for similar nodes to consolidate
                        for (j in i + 1 until nodesForLabel.size) {
                            if (nodesForLabel[j].id in consolidated) continue
                            
                            // Check if this group should be consolidated
                            val testGroup = candidateGroup + nodesForLabel[j]
                            if (rule.shouldConsolidate(testGroup)) {
                                candidateGroup.add(nodesForLabel[j])
                            }
                        }
                        
                        // Consolidate if we found a group
                        if (candidateGroup.size > 1) {
                            val consolidatedKnowledge = rule.consolidate(candidateGroup)
                            
                            // Remove old nodes
                            candidateGroup.forEach { knowledge ->
                                consolidated.add(knowledge.id)
                                if (knowledge.id != consolidatedKnowledge.id) {
                                    removeNode(knowledge.id)
                                }
                            }
                            
                            // Add or update consolidated node
                            when (consolidatedKnowledge) {
                                is Knowledge.Entity -> {
                                    val nodeData = NodeData(
                                        id = consolidatedKnowledge.id,
                                        labels = consolidatedKnowledge.labels,
                                        properties = consolidatedKnowledge.properties,
                                        createdAt = consolidatedKnowledge.timestamp,
                                        confidence = consolidatedKnowledge.confidence,
                                        provenance = consolidatedKnowledge.provenance
                                    )
                                    nodes[consolidatedKnowledge.id] = nodeData
                                    consolidatedKnowledge.labels.forEach { lbl ->
                                        nodesByLabel.getOrPut(lbl) { mutableSetOf() }.add(consolidatedKnowledge.id)
                                    }
                                }
                                else -> {
                                    // For now, only handle entity consolidation
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    override suspend fun stats(): GraphStats = mutex.withLock {
        GraphStats(
            nodeCount = nodes.size.toLong(),
            edgeCount = edges.size.toLong(),
            episodeCount = episodes.size.toLong(),
            lastIngestion = lastIngestionTime,
            lastEvolution = lastEvolutionTime,
            nodesByLabel = nodesByLabel.mapValues { it.value.size.toLong() },
            edgesByType = edgesByType.mapValues { it.value.size.toLong() }
        )
    }
    
    override suspend fun resolveEntity(mention: String, context: EntityResolutionContext): EntityResolutionResult {
        val normalizedMention = mention.trim().lowercase()
        val candidates = mutableListOf<EntityCandidate>()
        
        // Take a snapshot for thread-safe querying
        val snapshot = mutex.withLock { createSnapshot() }
        
        // Search for entities that might match this mention
        val searchTerms = config.tokenizer.tokenize(mention)
        
        // Find entities with matching terms
        val candidateNodeIds = searchTerms.flatMap { term ->
            snapshot.nodeTermIndex[term] ?: emptySet()
        }.toSet()
        
        // Score each candidate
        candidateNodeIds.forEach { nodeId ->
            snapshot.nodes[nodeId]?.let { node ->
                // Skip if wrong entity type
                if (context.entityType != null && 
                    !node.labels.contains(context.entityType.name)) {
                    return@let
                }
                
                var score = 0.0
                val matchingFeatures = mutableSetOf<String>()
                
                // Check name property
                val nodeName = node.properties["name"]?.toString()?.lowercase()
                if (nodeName != null) {
                    when {
                        nodeName == normalizedMention -> {
                            score += 1.0
                            matchingFeatures.add("exact_name_match")
                        }
                        nodeName.contains(normalizedMention) || normalizedMention.contains(nodeName) -> {
                            score += 0.7
                            matchingFeatures.add("partial_name_match")
                        }
                        searchTerms.any { it in nodeName } -> {
                            score += 0.4
                            matchingFeatures.add("term_match")
                        }
                    }
                }
                
                // Check aliases
                (node.properties["aliases"] as? List<*>)?.forEach { alias ->
                    val normalizedAlias = alias.toString().lowercase()
                    if (normalizedAlias == normalizedMention) {
                        score += 0.9
                        matchingFeatures.add("alias_match")
                    }
                }
                
                // Context bonus - if mentioned with nearby entities
                if (context.nearbyEntities.any { it.id == nodeId }) {
                    score += 0.2
                    matchingFeatures.add("context_proximity")
                }
                
                // Add to candidates if score is high enough
                if (score > 0.3) {
                    candidates.add(EntityCandidate(
                        entityId = nodeId,
                        entity = nodeToKnowledge(node),
                        similarityScore = score,
                        matchingFeatures = matchingFeatures
                    ))
                }
            }
        }
        
        // Sort by score
        candidates.sortByDescending { it.similarityScore }
        
        return when {
            candidates.isEmpty() -> {
                // No matches found - create new entity
                val newId = generateId("entity")
                val newEntity = Knowledge.Entity(
                    id = newId,
                    labels = setOf(context.entityType?.name ?: "Entity"),
                    properties = mapOf("name" to mention),
                    confidence = 0.8,
                    timestamp = clock.now(),
                    provenance = emptyList()
                )
                
                EntityResolutionResult.Resolved(
                    entityId = newId,
                    entity = newEntity,
                    confidence = 0.8,
                    isNew = true
                )
            }
            
            candidates.size == 1 && candidates[0].similarityScore >= context.confidenceThreshold -> {
                // Clear match
                EntityResolutionResult.Resolved(
                    entityId = candidates[0].entityId,
                    entity = candidates[0].entity,
                    confidence = candidates[0].similarityScore,
                    isNew = false
                )
            }
            
            candidates.size > 1 && 
            candidates[0].similarityScore >= context.confidenceThreshold &&
            candidates[0].similarityScore - candidates[1].similarityScore > 0.2 -> {
                // Clear winner
                EntityResolutionResult.Resolved(
                    entityId = candidates[0].entityId,
                    entity = candidates[0].entity,
                    confidence = candidates[0].similarityScore,
                    isNew = false
                )
            }
            
            else -> {
                // Ambiguous
                EntityResolutionResult.Ambiguous(
                    candidates = candidates.take(5),
                    reason = "Multiple entities match '$mention' with similar confidence"
                )
            }
        }
    }
    
    override suspend fun invalidateContradictingEdges(newFact: Knowledge.Relation, at: Instant): List<EdgeInvalidation> = mutex.withLock {
        val invalidations = mutableListOf<EdgeInvalidation>()
        
        // Find edges that contradict the new fact
        edges.values.forEach { edge ->
            // Same relationship type from same source
            if (edge.from == newFact.from && edge.type == newFact.type && edge.to != newFact.to) {
                // This is a contradiction - e.g., "Steve LEADS faction1" vs "Steve LEADS faction2"
                val invalidationReason = "Contradicted by new fact: ${newFact.from} ${newFact.type} ${newFact.to}"
                invalidations.add(EdgeInvalidation(
                    edgeId = edge.id,
                    edge = edgeToKnowledge(edge),
                    invalidatedAt = at,
                    reason = invalidationReason,
                    replacedBy = newFact.id
                ))
                
                // Mark the edge as invalid by setting validTo
                val updatedEdge = edge.copy(
                    validTo = at,
                    properties = edge.properties + mapOf(
                        "invalidated_by" to newFact.id,
                        "invalidation_reason" to invalidationReason
                    )
                )
                edges[edge.id] = updatedEdge
            }
        }
        
        invalidations
    }
    
    override suspend fun detectCommunities(algorithm: CommunityDetectionAlgorithm): List<Community> {
        // Take snapshot for analysis
        val snapshot = mutex.withLock { createSnapshot() }
        
        return when (algorithm) {
            CommunityDetectionAlgorithm.CONNECTED_COMPONENTS -> {
                detectConnectedComponents(snapshot)
            }
            else -> {
                // Other algorithms would require more complex implementations
                emptyList()
            }
        }
    }
    
    private fun detectConnectedComponents(snapshot: GraphSnapshot): List<Community> {
        val visited = mutableSetOf<NodeId>()
        val communities = mutableListOf<Community>()
        
        snapshot.nodes.keys.forEach { nodeId ->
            if (nodeId !in visited) {
                val component = mutableSetOf<NodeId>()
                val queue = ArrayDeque<NodeId>()
                queue.add(nodeId)
                
                // BFS to find all connected nodes
                while (queue.isNotEmpty()) {
                    val current = queue.removeFirst()
                    if (current in visited) continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    // Add neighbors
                    snapshot.outgoingEdges[current]?.forEach { edgeId ->
                        snapshot.edges[edgeId]?.let { edge ->
                            if (edge.to !in visited) {
                                queue.add(edge.to)
                            }
                        }
                    }
                    
                    snapshot.incomingEdges[current]?.forEach { edgeId ->
                        snapshot.edges[edgeId]?.let { edge ->
                            if (edge.from !in visited) {
                                queue.add(edge.from)
                            }
                        }
                    }
                }
                
                if (component.size > 1) {
                    communities.add(Community(
                        id = "community-${communities.size}",
                        members = component,
                        cohesionScore = 1.0, // Connected components have perfect cohesion
                        centralNodes = findCentralNodes(component, snapshot),
                        description = "Connected component with ${component.size} members"
                    ))
                }
            }
        }
        
        return communities
    }
    
    private fun findCentralNodes(members: Set<NodeId>, snapshot: GraphSnapshot): List<NodeId> {
        // Find nodes with most connections within the community
        return members
            .map { nodeId ->
                val degree = (snapshot.outgoingEdges[nodeId]?.size ?: 0) + 
                           (snapshot.incomingEdges[nodeId]?.size ?: 0)
                nodeId to degree
            }
            .sortedByDescending { it.second }
            .take(3)
            .map { it.first }
    }
    
    // Helper functions
    
    private fun Traversal.depth(): Int = when (this) {
        is Traversal.Outgoing -> depth
        is Traversal.Incoming -> depth
        is Traversal.Bidirectional -> depth
        is Traversal.ShortestPath -> maxDepth
    }
    
    private fun matchesFilters(node: NodeData, filters: List<Filter>): Boolean {
        return filters.all { filter ->
            when (filter) {
                is Filter.HasLabel -> filter.labels.any { it in node.labels }
                is Filter.HasProperty -> {
                    val value = node.properties[filter.key]
                    filter.value == null || value == filter.value
                }
                is Filter.PropertyRange -> {
                    val value = node.properties[filter.key] as? Comparable<Any>
                    value != null &&
                        (filter.min == null || value >= filter.min as Comparable<Any>) &&
                        (filter.max == null || value <= filter.max as Comparable<Any>)
                }
                is Filter.Custom -> true // Not implemented
            }
        }
    }
    
    private fun shouldIncludeEdge(edge: EdgeData, traversal: Traversal): Boolean {
        val types = when (traversal) {
            is Traversal.Outgoing -> traversal.relationTypes
            is Traversal.Incoming -> traversal.relationTypes
            is Traversal.Bidirectional -> traversal.relationTypes
            is Traversal.ShortestPath -> emptyList()
        }
        return types.isEmpty() || edge.type in types
    }
    
    private fun nodeToKnowledge(node: NodeData): Knowledge.Entity {
        return Knowledge.Entity(
            id = node.id,
            labels = node.labels,
            properties = node.properties,
            confidence = node.confidence,
            timestamp = node.createdAt,
            provenance = node.provenance
        )
    }
    
    private fun edgeToKnowledge(edge: EdgeData): Knowledge.Relation {
        return Knowledge.Relation(
            id = edge.id,
            type = edge.type,
            from = edge.from,
            to = edge.to,
            properties = edge.properties,
            confidence = edge.confidence,
            timestamp = edge.createdAt,
            provenance = edge.provenance
        )
    }
    
    private fun removeNode(nodeId: NodeId) {
        nodes.remove(nodeId)?.let { node ->
            // Remove from label index
            node.labels.forEach { label ->
                nodesByLabel[label]?.remove(nodeId)
                if (nodesByLabel[label]?.isEmpty() == true) {
                    nodesByLabel.remove(label)
                }
            }
            
            // Remove from term index
            unindexNode(nodeId, node)
        }
        
        // Remove associated edges
        outgoingEdges[nodeId]?.forEach { edgeId ->
            removeEdge(edgeId)
        }
        incomingEdges[nodeId]?.forEach { edgeId ->
            removeEdge(edgeId)
        }
        
        outgoingEdges.remove(nodeId)
        incomingEdges.remove(nodeId)
    }
    
    private fun removeEdge(edgeId: EdgeId) {
        edges.remove(edgeId)?.let { edge ->
            edgesByType[edge.type]?.remove(edgeId)
            if (edgesByType[edge.type]?.isEmpty() == true) {
                edgesByType.remove(edge.type)
            }
            
            outgoingEdges[edge.from]?.remove(edgeId)
            incomingEdges[edge.to]?.remove(edgeId)
            
            // Remove from term index
            unindexEdge(edgeId, edge)
        }
    }
    
    private fun generateId(prefix: String): String {
        return "$prefix-${Uuid.random()}"
    }
    
    /**
     * Indexes a node's terms for semantic search
     */
    private fun indexNode(nodeId: NodeId, node: NodeData) {
        val terms = extractNodeTerms(node)
        terms.forEach { term ->
            nodeTermIndex.getOrPut(term) { mutableSetOf() }.add(nodeId)
        }
    }
    
    /**
     * Removes a node from the term index
     */
    private fun unindexNode(nodeId: NodeId, node: NodeData) {
        val terms = extractNodeTerms(node)
        terms.forEach { term ->
            nodeTermIndex[term]?.remove(nodeId)
            if (nodeTermIndex[term]?.isEmpty() == true) {
                nodeTermIndex.remove(term)
            }
        }
    }
    
    /**
     * Indexes an edge's terms for semantic search
     */
    private fun indexEdge(edgeId: EdgeId, edge: EdgeData) {
        val terms = extractEdgeTerms(edge)
        terms.forEach { term ->
            edgeTermIndex.getOrPut(term) { mutableSetOf() }.add(edgeId)
        }
    }
    
    /**
     * Removes an edge from the term index
     */
    private fun unindexEdge(edgeId: EdgeId, edge: EdgeData) {
        val terms = extractEdgeTerms(edge)
        terms.forEach { term ->
            edgeTermIndex[term]?.remove(edgeId)
            if (edgeTermIndex[term]?.isEmpty() == true) {
                edgeTermIndex.remove(term)
            }
        }
    }
    
    /**
     * Extracts searchable terms from a node
     */
    private fun extractNodeTerms(node: NodeData): Set<String> {
        val terms = mutableSetOf<String>()
        
        // Add terms from labels
        node.labels.forEach { label ->
            terms.addAll(config.tokenizer.tokenize(label))
        }
        
        // Add terms from properties
        node.properties.forEach { (key, value) ->
            terms.addAll(config.tokenizer.tokenize(key))
            terms.addAll(config.tokenizer.tokenize(value.toString()))
        }
        
        return terms
    }
    
    /**
     * Extracts searchable terms from an edge
     */
    private fun extractEdgeTerms(edge: EdgeData): Set<String> {
        val terms = mutableSetOf<String>()
        
        // Add terms from type
        terms.addAll(config.tokenizer.tokenize(edge.type))
        
        // Add terms from properties
        edge.properties.forEach { (key, value) ->
            terms.addAll(config.tokenizer.tokenize(key))
            terms.addAll(config.tokenizer.tokenize(value.toString()))
        }
        
        return terms
    }
    
    /**
     * Merges a candidate node with an existing node
     */
    private fun NodeData.mergeWith(
        candidate: ai.koog.agents.memory.feature.EntityCandidate,
        newProvenance: ProvenanceItem
    ): NodeData {
        return copy(
            labels = labels + candidate.labels,
            properties = properties + candidate.properties,
            confidence = maxOf(confidence, candidate.confidence),
            provenance = provenance + newProvenance
        )
    }
}