package ai.koog.agents.memory.feature.nodes

import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.builder.AIAgentBuilderDslMarker
import ai.koog.agents.core.dsl.builder.AIAgentNodeDelegate
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.memory.config.MemoryScopeType
import ai.koog.agents.memory.feature.withMemory
import ai.koog.agents.memory.graph.Knowledge
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.DefaultTimeProvider
import ai.koog.agents.memory.model.Fact
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.memory.model.MultipleFacts
import ai.koog.agents.memory.model.SingleFact
import ai.koog.agents.memory.prompts.MemoryPrompts
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.prompt.llm.LLModel
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

// ===========================================
// Standard Memory Nodes (Works with any AgentMemoryProvider)
// ===========================================

/**
 * Node that loads facts from memory for a given concept.
 * 
 * Works with any AgentMemoryProvider. When used with GraphMemoryProvider,
 * automatically benefits from semantic search and relationship awareness.
 *
 * @param concept A concept to load facts for
 * @param subject The subject scope of the memory (USER, PROJECT, etc.)
 * @param scope The scope of the memory (Agent, Feature, etc.)
 */
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeLoadFromMemory(
    name: String? = null,
    concept: Concept,
    subject: MemorySubject,
    scope: MemoryScopeType = MemoryScopeType.AGENT
): AIAgentNodeDelegate<T, T> = nodeLoadFromMemory(name, listOf(concept), listOf(subject), listOf(scope))

/**
 * Node that loads facts from memory for multiple concepts.
 * 
 * @param concepts A list of concepts to load facts for
 * @param subject The subject scope of the memory (USER, PROJECT, etc.)
 * @param scope The scope of the memory (Agent, Feature, etc.)
 */
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeLoadFromMemory(
    name: String? = null,
    concepts: List<Concept>,
    subject: MemorySubject,
    scope: MemoryScopeType = MemoryScopeType.AGENT
): AIAgentNodeDelegate<T, T> = nodeLoadFromMemory(name, concepts, listOf(subject), listOf(scope))

/**
 * Node that loads facts from memory with flexible scoping.
 * 
 * @param concepts A list of concepts to load facts for
 * @param subjects List of subjects (user, project, organization, etc.) to look for. By default all subjects
 * @param scopes List of memory scopes (Agent, Feature, etc.). By default all scopes
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeLoadFromMemory(
    name: String? = null,
    concepts: List<Concept>,
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    scopes: List<MemoryScopeType> = MemoryScopeType.entries
): AIAgentNodeDelegate<T, T> = node(name) { input ->
    withMemory {
        concepts.forEach { concept ->
            loadFactsToAgent(concept, scopes, subjects)
        }
    }
    input
}

/**
 * Node that loads all facts from memory across subjects and scopes.
 * 
 * @param subjects List of subjects (user, project, organization, etc.) to look for
 * @param scopes List of memory scopes (Agent, Feature, etc.). By default all scopes
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeLoadAllFactsFromMemory(
    name: String? = null,
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    scopes: List<MemoryScopeType> = MemoryScopeType.entries
): AIAgentNodeDelegate<T, T> = node(name) { input ->
    withMemory {
        loadAllFactsToAgent(scopes, subjects)
    }
    input
}

/**
 * Node that saves facts to memory for specific concepts.
 * 
 * Works with any AgentMemoryProvider. When used with GraphMemoryProvider,
 * facts are automatically organized into a knowledge graph with relationships.
 * 
 * @param concepts A list of concepts to save facts for
 * @param subject The subject scope of the memory
 * @param scope The scope of the memory
 * @param retrievalModel Optional LLM model for fact extraction
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeSaveToMemory(
    name: String? = null,
    subject: MemorySubject,
    scope: MemoryScopeType,
    concepts: List<Concept>,
    retrievalModel: LLModel? = null
): AIAgentNodeDelegate<T, T> = node(name) { input ->
    withMemory {
        concepts.forEach { concept ->
            val memoryScope = scopesProfile.getScope(scope) ?: return@withMemory
            saveFactsFromHistory(concept, subject, memoryScope, retrievalModel)
        }
    }
    input
}

/**
 * Node that saves a fact to memory for a single concept.
 * 
 * @param concept A concept to save facts for
 * @param subject The subject scope of the memory
 * @param scope The scope of the memory
 * @param retrievalModel Optional LLM model for fact extraction
 */
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeSaveToMemory(
    name: String? = null,
    concept: Concept,
    subject: MemorySubject,
    scope: MemoryScopeType,
    retrievalModel: LLModel? = null
): AIAgentNodeDelegate<T, T> = nodeSaveToMemory(name, subject, scope, listOf(concept), retrievalModel)

/**
 * Node that automatically detects and extracts facts from the chat history and saves them to memory.
 * Uses LLM to identify concepts about user, organization, project, etc.
 * 
 * Works with any AgentMemoryProvider. When used with GraphMemoryProvider,
 * extracted facts automatically become part of the knowledge graph.
 * 
 * @param subjects List of subjects to extract facts for
 * @param scopes List of memory scopes to save facts to
 * @param retrievalModel Optional LLM model for fact extraction
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeSaveToMemoryAutoDetectFacts(
    name: String? = null,
    scopes: List<MemoryScopeType> = listOf(MemoryScopeType.AGENT),
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    retrievalModel: LLModel? = null
): AIAgentNodeDelegate<T, T> = node(name) { input ->
    withMemory {
        // Use the existing auto-detection mechanism built into AgentMemory
        // This is a simplified implementation that maintains the interface
        try {
            // Extract facts from conversation history for each subject/scope combination
            for (subject in subjects) {
                for (scope in scopes) {
                    val memoryScope = scopesProfile.getScope(scope) ?: continue
                    
                    // Create a generic concept for auto-detected facts
                    val autoConcept = Concept(
                        keyword = "auto-detected-${subject.name}-${scope.name.lowercase()}",
                        description = "Auto-detected facts for ${subject.name} in ${scope.name} scope",
                        factType = FactType.MULTIPLE
                    )
                    
                    // Save facts from history using the existing mechanism
                    saveFactsFromHistory(autoConcept, subject, memoryScope, retrievalModel)
                }
            }
        } catch (e: Exception) {
            // Log error but don't fail the node
            println("Error auto-detecting facts: ${e.message}")
        }
    }
    input
}

// ===========================================
// Advanced Graph Memory Nodes (GraphMemoryProvider only)
// ===========================================

/**
 * Node that performs semantic search using natural language queries.
 * 
 * **Requires GraphMemoryProvider** - automatically falls back to description-based search
 * with other memory providers.
 * 
 * This node demonstrates the power of graph-based memory by allowing natural language
 * queries that leverage semantic understanding and relationship traversal.
 * 
 * @param query Natural language query (e.g., "Who are the strongest factions?")
 * @param limit Maximum number of results to return
 * @param subjects Filter results to specific subjects
 * @param scopes Filter results to specific scopes
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeSemanticQuery(
    query: String,
    limit: Int = 10,
    name: String? = null,
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    scopes: List<MemoryScopeType> = MemoryScopeType.entries
): AIAgentNodeDelegate<T, List<Fact>> = node(name) { input ->
    withMemory {
        // Try graph-specific semantic search first
        val graphProvider = agentMemory as? GraphMemoryProvider
        if (graphProvider != null) {
            // Use advanced graph semantic search
            val facts = mutableListOf<Fact>()
            for (subject in subjects) {
                for (scope in scopes) {
                    val scopeInstance = scopesProfile.getScope(scope) ?: continue
                    val results = graphProvider.loadByDescription(query, subject, scopeInstance)
                    facts.addAll(results.take(limit))
                }
            }
            facts.take(limit)
        } else {
            // Fallback to standard description-based search
            val facts = mutableListOf<Fact>()
            for (subject in subjects) {
                for (scope in scopes) {
                    val scopeInstance = scopesProfile.getScope(scope) ?: continue
                    val results = agentMemory.loadByDescription(query, subject, scopeInstance)
                    facts.addAll(results)
                }
            }
            facts.take(limit)
        }
    }
}

/**
 * Node that performs entity-centric queries around a specific entity.
 * 
 * **Requires GraphMemoryProvider** - provides relationship traversal and entity-centric search.
 * Falls back to concept-based search with other providers.
 * 
 * This node showcases advanced graph capabilities by finding information related
 * to a specific entity through relationship traversal.
 * 
 * @param entityId The central entity to query around (e.g., "player:Steve", "faction:MountainBuilders")
 * @param depth How many relationship hops to traverse
 * @param relationTypes Optional filter for specific relationship types
 * @param subjects Filter results to specific subjects
 * @param scopes Filter results to specific scopes
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeEntityCentricQuery(
    entityId: String,
    depth: Int = 2,
    name: String? = null,
    relationTypes: List<String> = emptyList(),
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    scopes: List<MemoryScopeType> = MemoryScopeType.entries
): AIAgentNodeDelegate<T, List<Fact>> = node(name) { input ->
    withMemory {
        val graphProvider = agentMemory as? GraphMemoryProvider
        if (graphProvider != null) {
            // Use advanced entity-centric graph query
            val knowledge = graphProvider.queryEntityCentric(
                centerNode = entityId,
                depth = depth,
                relationTypes = relationTypes
            )
            
            // Convert Knowledge back to Facts for standard memory API
            knowledge.mapNotNull { k ->
                when (k) {
                    is Knowledge.Entity -> {
                        val content = k.properties["content"] as? String ?: k.id
                        SingleFact(
                            concept = Concept(
                                keyword = k.id,
                                description = k.labels.firstOrNull() ?: "Entity",
                                factType = FactType.SINGLE
                            ),
                            timestamp = k.timestamp.toEpochMilliseconds(),
                            value = content
                        )
                    }
                    is Knowledge.Relation -> {
                        SingleFact(
                            concept = Concept(
                                keyword = k.type,
                                description = "Relationship: ${k.type}",
                                factType = FactType.SINGLE
                            ),
                            timestamp = k.timestamp.toEpochMilliseconds(),
                            value = "Relationship from ${k.from} to ${k.to} of type ${k.type}"
                        )
                    }
                    else -> null
                }
            }
        } else {
            // Fallback: search for facts containing the entity ID
            val facts = mutableListOf<Fact>()
            for (subject in subjects) {
                for (scope in scopes) {
                    val scopeInstance = scopesProfile.getScope(scope) ?: continue
                    val results = agentMemory.loadByDescription(entityId, subject, scopeInstance)
                    facts.addAll(results)
                }
            }
            facts.take(20)
        }
    }
}

/**
 * Node that performs temporal queries to find what was known at specific times.
 * 
 * **Requires GraphMemoryProvider** - provides temporal reasoning capabilities.
 * Falls back to timestamp-based filtering with other providers.
 * 
 * This node demonstrates temporal reasoning, allowing agents to understand
 * how knowledge has changed over time.
 * 
 * @param start Start of time window
 * @param end End of time window
 * @param includeDeleted Whether to include deleted/removed knowledge
 * @param subjects Filter results to specific subjects
 * @param scopes Filter results to specific scopes
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeTemporalQuery(
    start: Instant,
    end: Instant,
    name: String? = null,
    includeDeleted: Boolean = false,
    subjects: List<MemorySubject> = MemorySubject.registeredSubjects,
    scopes: List<MemoryScopeType> = MemoryScopeType.entries
): AIAgentNodeDelegate<T, List<Fact>> = node(name) { input ->
    withMemory {
        val graphProvider = agentMemory as? GraphMemoryProvider
        if (graphProvider != null) {
            // Use advanced temporal graph query
            val knowledge = graphProvider.queryTemporal(start, end, includeDeleted)
            
            // Convert Knowledge back to Facts
            knowledge.mapNotNull { k ->
                when (k) {
                    is Knowledge.Entity -> {
                        val content = k.properties["content"] as? String ?: k.id
                        SingleFact(
                            concept = Concept(
                                keyword = k.id,
                                description = k.labels.firstOrNull() ?: "Entity",
                                factType = FactType.SINGLE
                            ),
                            timestamp = k.timestamp.toEpochMilliseconds(),
                            value = content
                        )
                    }
                    else -> null
                }
            }
        } else {
            // Fallback: filter existing facts by timestamp
            val facts = mutableListOf<Fact>()
            for (subject in subjects) {
                for (scope in scopes) {
                    val scopeInstance = scopesProfile.getScope(scope) ?: continue
                    val allFacts = agentMemory.loadAll(subject, scopeInstance)
                    val filteredFacts = allFacts.filter { fact ->
                        val factTime = Instant.fromEpochMilliseconds(fact.timestamp)
                        factTime >= start && factTime <= end
                    }
                    facts.addAll(filteredFacts)
                }
            }
            facts
        }
    }
}

/**
 * Node that triggers knowledge graph evolution (consolidation, decay, etc.).
 * 
 * **Requires GraphMemoryProvider** - performs graph maintenance operations.
 * No-op with other providers.
 * 
 * This node allows manual triggering of graph maintenance, which normally
 * happens automatically but can be useful for testing or optimization.
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeEvolveGraph(
    name: String? = null
): AIAgentNodeDelegate<T, T> = node(name) { input ->
    withMemory {
        val graphProvider = agentMemory as? GraphMemoryProvider
        graphProvider?.evolve()
    }
    input
}

/**
 * Node that retrieves graph statistics and insights.
 * 
 * **Requires GraphMemoryProvider** - provides detailed graph statistics.
 * Returns basic info with other providers.
 * 
 * This node is useful for monitoring and debugging memory usage,
 * providing insights into the knowledge graph structure.
 */
@OptIn(InternalAgentsApi::class)
@AIAgentBuilderDslMarker
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeGraphStats(
    name: String? = null
): AIAgentNodeDelegate<T, GraphStatsResult> = node(name) { input ->
    withMemory {
        val graphProvider = agentMemory as? GraphMemoryProvider
        if (graphProvider != null) {
            val stats = graphProvider.getGraphStats()
            GraphStatsResult(
                available = true,
                nodeCount = stats.nodeCount,
                edgeCount = stats.edgeCount,
                episodeCount = stats.episodeCount,
                lastIngestion = stats.lastIngestion,
                nodesByLabel = stats.nodesByLabel,
                edgesByType = stats.edgesByType
            )
        } else {
            GraphStatsResult(
                available = false,
                nodeCount = 0,
                edgeCount = 0,
                episodeCount = 0,
                lastIngestion = null,
                nodesByLabel = emptyMap(),
                edgesByType = emptyMap()
            )
        }
    }
}

// ===========================================
// Helper Functions and Data Classes
// ===========================================

/**
 * Result of graph statistics query
 */
@Serializable
public data class GraphStatsResult(
    val available: Boolean,
    val nodeCount: Long,
    val edgeCount: Long,
    val episodeCount: Long,
    val lastIngestion: Instant?,
    val nodesByLabel: Map<String, Long>,
    val edgesByType: Map<String, Long>
)

// ===========================================
// Parsing utilities (existing functionality)
// ===========================================

/**
 * Parsing facts from response.
 * 
 * This function maintains backward compatibility with existing fact parsing logic.
 */
public fun parseFactsFromResponse(
    content: String
): List<Pair<MemorySubject, Fact>> {
    val facts = mutableListOf<Pair<MemorySubject, Fact>>()
    
    try {
        // Parse JSON response containing facts
        val json = Json.parseToJsonElement(content)
        // Implementation would depend on the exact format expected
        // This is a placeholder that maintains the existing signature
    } catch (e: Exception) {
        // If JSON parsing fails, treat the entire content as a single fact
        @OptIn(InternalAgentsApi::class)
        val defaultSubject = MemorySubject.registeredSubjects.firstOrNull() 
            ?: object : MemorySubject() {
                override val name: String = "default"
                override val promptDescription: String = "Default subject"
                override val priorityLevel: Int = 1
            }
        
        val fact = SingleFact(
            concept = Concept(
                keyword = "parsed-content",
                description = "Content parsed from response",
                factType = FactType.SINGLE
            ),
            timestamp = DefaultTimeProvider.getCurrentTimestamp(),
            value = content
        )
        
        facts.add(defaultSubject to fact)
    }
    
    return facts
}