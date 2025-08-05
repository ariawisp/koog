package ai.koog.agents.benchmark.runners

import ai.koog.agents.benchmark.core.MemorySystem
import ai.koog.agents.benchmark.llm.UnifiedAnswerGenerator
import ai.koog.agents.memory.graph.KnowledgeGraph
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.graph.Episode
import ai.koog.agents.memory.graph.EpisodeSource
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.agents.memory.providers.GraphMemoryConfig
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.memory.model.*
import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider
import ai.koog.agents.memory.retrieval.providers.VectorRetrievalProvider
import ai.koog.agents.memory.retrieval.reranking.CrossEncoderReranker
import ai.koog.agents.memory.retrieval.config.TokenAwareRetrievalConfig
import ai.koog.agents.memory.retrieval.createSmartRouter
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.rag.base.RankedDocumentStorage
import kotlinx.datetime.Clock

/**
 * SOTA Memory System that properly uses Koog's actual memory infrastructure
 * instead of creating dummy providers
 */
class SOTAMemorySystem(
    private val config: SOTAMemoryConfig = SOTAMemoryConfig()
) : MemorySystem {
    
    override val name = "SOTA Memory System (${config.strategy})"
    
    // Core components
    private val knowledgeGraph: KnowledgeGraph = config.knowledgeGraph ?: InMemoryKnowledgeGraph()
    
    // Memory provider - the actual Koog memory system
    private val memoryProvider: AgentMemoryProvider = GraphMemoryProvider(
        graph = knowledgeGraph,
        config = GraphMemoryConfig(
            autoIngestConversations = true,
            minConfidenceThreshold = 0.7
        )
    )
    
    // Create the smart router with all the SOTA features
    private val smartRouter: SmartRouter = createSmartRouter(
        memoryProvider = memoryProvider,
        tokenizer = config.tokenizer,
        tokenConfig = config.tokenConfig,
        metricsCollector = config.metricsCollector
    )
    
    // For storing raw context during benchmark
    private val contextStore = mutableListOf<Pair<String, Map<String, Any>>>()
    
    override suspend fun addContext(context: String, metadata: Map<String, Any>) {
        contextStore.add(context to metadata)
        
        // Convert to proper memory format
        val concept = Concept(
            keyword = metadata["entity_name"]?.toString() ?: "fact_${System.currentTimeMillis()}",
            description = context,
            factType = FactType.SINGLE
        )
        
        // For benchmarking, use Everything subject
        val subject = MemorySubject.Everything
        
        val fact = SingleFact(
            concept = concept,
            timestamp = Clock.System.now().toEpochMilliseconds(),
            value = context
        )
        
        // Use the actual memory provider to store
        memoryProvider.save(
            fact = fact,
            subject = subject,
            scope = config.defaultScope
        )
        
        // For LettaBench, extract structured data
        extractAndStoreStructuredData(context, metadata)
    }
    
    override suspend fun retrieve(query: String): List<String> {
        // Create a proper retrieval query
        val retrievalQuery = RetrievalQuery(
            text = query,
            k = config.retrievalLimit,
            filters = RetrievalFilters(
                scopes = setOf(config.defaultScope)
            ),
            recipe = when (config.strategy) {
                RetrievalStrategy.GRAPH -> RetrievalRecipe.VECTOR_SIMILARITY
                RetrievalStrategy.VECTOR -> RetrievalRecipe.VECTOR_SIMILARITY
                RetrievalStrategy.HYBRID_RRF -> RetrievalRecipe.HYBRID_RRF
                RetrievalStrategy.HYBRID_NODE -> RetrievalRecipe.HYBRID_NODE_DISTANCE
                RetrievalStrategy.FULL_CONTEXT -> RetrievalRecipe.VECTOR_SIMILARITY
            }
        )
        
        // Use smart router for retrieval
        val results = if (config.strategy == RetrievalStrategy.FULL_CONTEXT) {
            // Return all context for baseline
            listOf(RetrievalResult(
                content = contextStore.joinToString("\n") { it.first },
                score = 1.0,
                provenance = emptyList(),
                metadata = emptyMap()
            ))
        } else {
            smartRouter.retrieve(retrievalQuery)
        }
        
        return results.map { it.content }
    }
    
    override suspend fun answer(question: String, executor: PromptExecutor): String {
        val context = retrieve(question)
        val generator = UnifiedAnswerGenerator(config.answerStrategy)
        return generator.generateAnswer(question, context, executor)
    }
    
    private suspend fun extractAndStoreStructuredData(context: String, metadata: Map<String, Any>) {
        // For benchmarking, we'll ingest the data as episodes
        // This properly uses the KnowledgeGraph API
        val episode = Episode(
            content = context,
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT,
            metadata = metadata + mapOf("benchmark" to true),
            references = extractReferences(context)
        )
        
        // Ingest into the knowledge graph
        knowledgeGraph.ingest(episode)
    }
    
    private fun extractReferences(content: String): List<String> {
        // Extract entity names from content for references
        val references = mutableListOf<String>()
        
        // Extract person names (simple heuristic for benchmark data)
        val words = content.split(" ")
        words.forEachIndexed { index, word ->
            // Look for capitalized words that might be names
            if (word.firstOrNull()?.isUpperCase() == true && 
                word.length > 1 && 
                !word.contains("'") &&
                index > 0) {
                references.add(word.removeSuffix(".").removeSuffix(","))
            }
        }
        
        return references.distinct()
    }
}

/**
 * Configuration for the SOTA memory system
 */
data class SOTAMemoryConfig(
    val strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RRF,
    val answerStrategy: AnswerStrategy = AnswerStrategy.CHAIN_OF_THOUGHT,
    val retrievalLimit: Int = 10,
    val knowledgeGraph: KnowledgeGraph? = null,
    val documentStorage: RankedDocumentStorage<String>? = null,
    val tokenizer: PromptTokenizer? = null,
    val tokenConfig: TokenAwareRetrievalConfig? = null,
    val metricsCollector: ai.koog.agents.memory.retrieval.metrics.RetrievalMetricsCollector? = null,
    val crossEncoderReranker: CrossEncoderReranker? = null,
    val defaultScope: MemoryScope = MemoryScope.CrossProduct
)

enum class RetrievalStrategy {
    GRAPH,           // Pure knowledge graph
    VECTOR,          // Pure vector similarity  
    HYBRID_RRF,      // Reciprocal Rank Fusion
    HYBRID_NODE,     // Node distance weighting
    FULL_CONTEXT     // Baseline - everything
}

enum class AnswerStrategy {
    SIMPLE,          // Direct answer from context
    CHAIN_OF_THOUGHT,// Step-by-step reasoning
    MULTI_STEP       // Multiple retrieval rounds
}