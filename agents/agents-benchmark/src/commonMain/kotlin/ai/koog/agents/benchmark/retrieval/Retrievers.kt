package ai.koog.agents.benchmark.retrieval

import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider
import ai.koog.agents.benchmark.model.RetrievalMode
import ai.koog.agents.benchmark.llm.LLMAnswerGenerator
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.RequestMetaInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.datetime.Clock

/**
 * Interface for benchmark retrievers that can answer questions
 */
public interface Retriever {
    public suspend fun answer(question: String): String
}

/**
 * Graph-based retriever using KnowledgeGraphRetrievalProvider
 */
public class GraphRetriever(
    private val provider: KnowledgeGraphRetrievalProvider,
    private val answerGenerator: LLMAnswerGenerator
) : Retriever {
    override suspend fun answer(question: String): String = withContext(Dispatchers.Default) {
        val query = RetrievalQuery(
            text = question,
            recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE,
            k = 10
        )
        val ctx = provider.retrieve(query)
        answerGenerator.generateAnswer(question, ctx)
    }
}

/**
 * Hybrid retriever using SmartRouter
 */
public class HybridRetriever(
    private val smartRouter: SmartRouter,
    private val answerGenerator: LLMAnswerGenerator
) : Retriever {
    override suspend fun answer(question: String): String = withContext(Dispatchers.Default) {
        val query = RetrievalQuery(
            text = question,
            recipe = RetrievalRecipe.HYBRID_RRF,
            k = 10
        )
        val ctx = smartRouter.retrieve(query)
        answerGenerator.generateAnswer(question, ctx)
    }
}

/**
 * Vector-only retriever using SmartRouter
 */
public class VectorRetriever(
    private val smartRouter: SmartRouter,
    private val answerGenerator: LLMAnswerGenerator
) : Retriever {
    override suspend fun answer(question: String): String = withContext(Dispatchers.Default) {
        val query = RetrievalQuery(
            text = question,
            recipe = RetrievalRecipe.VECTOR_SIMILARITY,
            k = 10
        )
        val ctx = smartRouter.retrieve(query)
        answerGenerator.generateAnswer(question, ctx)
    }
}

// Note: Old synthesizeAnswer function removed - now using LLMAnswerGenerator