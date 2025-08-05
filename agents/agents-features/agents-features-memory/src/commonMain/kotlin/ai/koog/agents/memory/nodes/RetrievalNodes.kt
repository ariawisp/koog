package ai.koog.agents.memory.nodes

import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.builder.AIAgentBuilderDslMarker
import ai.koog.agents.core.dsl.builder.AIAgentNodeDelegate
import ai.koog.agents.core.dsl.builder.AIAgentSubgraphBuilderBase
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.feature.memory
import ai.koog.agents.memory.feature.withMemory
import ai.koog.agents.memory.retrieval.*
import kotlinx.datetime.Instant

/**
 * A node that retrieves knowledge from the memory system using
 * advanced retrieval strategies including vector search, graph traversal,
 * and hybrid approaches.
 * 
 * This node enables:
 * - Semantic search across facts, entities, and documents
 * - Temporal queries for point-in-time knowledge
 * - Entity-centric search with graph distance reranking
 * - Multiple retrieval recipes (RRF, MMR, node distance)
 * 
 * Example usage:
 * ```kotlin
 * val agent = AIAgent(/* ... */) {
 *     install(AgentMemory) {
 *         memoryProvider = myMemoryProvider
 *         retriever = createSmartRouter(
 *             memoryProvider = myMemoryProvider,
 *             graphProvider = myGraphProvider
 *         )
 *     }
 * }
 * 
 * val strategy = strategy<String, String>("search-strategy") {
 *     val search by nodeRetrieveKnowledge<String> {
 *         text = "Who owns the base near spawn?"
 *         k = 5
 *         recipe = RetrievalRecipe.HYBRID_RRF
 *     }
 *     
 *     edge(nodeStart forwardTo search)
 *     edge(search forwardTo nodeFinish transformed { results ->
 *         results.joinToString("\n") { it.content }
 *     })
 * }
 * ```
 */
@AIAgentBuilderDslMarker
@OptIn(InternalAgentsApi::class)
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeRetrieveKnowledge(
    name: String? = null,
    crossinline configure: RetrievalQueryBuilder.() -> Unit
): AIAgentNodeDelegate<T, List<RetrievalResult>> = node(name) { input ->
    val memory = memory()
    
    @OptIn(InternalAgentsApi::class)
    val retriever = memory.retriever
        ?: error("No retriever configured in AgentMemory")
    
    // Build the retrieval query
    val queryBuilder = RetrievalQueryBuilder().apply(configure)
    
    // Extract text from input if needed
    val inputText = when (input) {
        is String -> input
        else -> ""
    }
    
    val query = queryBuilder.build(inputText)
    
    // Execute retrieval
    retriever.retrieve(query)
}

/**
 * Builder for constructing RetrievalQuery instances with a DSL.
 */
public class RetrievalQueryBuilder {
    public var text: String = ""
    public var centerNode: String? = null
    public var at: Instant? = null
    public var target: RetrievalTarget = RetrievalTarget.ALL
    public var k: Int = 10
    public var recipe: RetrievalRecipe = RetrievalRecipe.HYBRID_RRF
    public var requireCitations: Boolean = true
    
    private val filterBuilder = RetrievalFiltersBuilder()
    
    /**
     * Configure filters for the retrieval query.
     */
    public fun filters(configure: RetrievalFiltersBuilder.() -> Unit) {
        filterBuilder.apply(configure)
    }
    
    public fun build(inputText: String): RetrievalQuery {
        return RetrievalQuery(
            text = text.ifEmpty { inputText },
            centerNode = centerNode,
            at = at,
            target = target,
            k = k,
            filters = filterBuilder.build(),
            recipe = recipe,
            requireCitations = requireCitations
        )
    }
}

/**
 * Builder for retrieval filters.
 */
public class RetrievalFiltersBuilder {
    private val subjects = mutableSetOf<ai.koog.agents.memory.model.MemorySubject>()
    private val scopes = mutableSetOf<ai.koog.agents.memory.model.MemoryScope>()
    private val entityLabels = mutableSetOf<String>()
    private val factTypes = mutableSetOf<String>()
    
    public fun subject(subject: ai.koog.agents.memory.model.MemorySubject) {
        subjects.add(subject)
    }
    
    public fun scope(scope: ai.koog.agents.memory.model.MemoryScope) {
        scopes.add(scope)
    }
    
    public fun entityLabel(label: String) {
        entityLabels.add(label)
    }
    
    public fun factType(type: String) {
        factTypes.add(type)
    }
    
    public fun build(): RetrievalFilters {
        return RetrievalFilters(
            subjects = subjects,
            scopes = scopes,
            entityLabels = entityLabels,
            factTypes = factTypes
        )
    }
}

/**
 * Convenience node for retrieving knowledge and formatting as a single string.
 * Useful when you need the results as text rather than structured data.
 * 
 * Example:
 * ```kotlin
 * val searchText by nodeRetrieveKnowledgeAsText<String> {
 *     text = "Recent faction battles"
 *     k = 3
 * }
 * ```
 */
@AIAgentBuilderDslMarker
@OptIn(InternalAgentsApi::class)
public inline fun <reified T> AIAgentSubgraphBuilderBase<*, *>.nodeRetrieveKnowledgeAsText(
    name: String? = null,
    separator: String = "\n",
    crossinline configure: RetrievalQueryBuilder.() -> Unit
): AIAgentNodeDelegate<T, String> = node(name) { input ->
    val memory = memory()
    
    @OptIn(InternalAgentsApi::class)
    val retriever = memory.retriever
        ?: error("No retriever configured in AgentMemory")
    
    val queryBuilder = RetrievalQueryBuilder().apply(configure)
    val inputText = when (input) {
        is String -> input
        else -> ""
    }
    
    val query = queryBuilder.build(inputText)
    val results = retriever.retrieve(query)
    
    // Format results as text
    results.joinToString(separator) { result ->
        if (result.provenance.isNotEmpty() && query.requireCitations) {
            "${result.content} [${formatProvenance(result.provenance.first())}]"
        } else {
            result.content
        }
    }
}

public fun formatProvenance(provenance: Provenance): String {
    return when (provenance) {
        is Provenance.Graph -> "Graph: ${provenance.nodeId ?: provenance.edgeId}"
        is Provenance.Document -> "Doc: ${provenance.path}"
        is Provenance.Fact -> "Fact: ${provenance.conceptKeyword}"
    }
}