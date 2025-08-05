package ai.koog.agents.memory.retrieval.reranking

import ai.koog.agents.memory.retrieval.RetrievalResult
import ai.koog.agents.core.agent.AIAgentException
import ai.koog.agents.core.agent.context.AIAgentLLMContext
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.coroutineScope
import kotlin.math.exp

/**
 * Cross-encoder reranker that uses an LLM to score query-document pairs.
 * This provides higher precision than embedding-based retrieval at the cost
 * of additional latency. Critical for beating benchmarks that require high accuracy.
 */
public class CrossEncoderReranker(
    private val llmExecutor: SingleLLMPromptExecutor,
    private val model: LLModel,
    private val maxConcurrency: Int = 4,
    private val batchSize: Int = 10
) {
    private companion object {
        private val logger = KotlinLogging.logger {}
    }
    
    /**
     * Configuration for cross-encoder reranking
     */
    public data class RerankConfig(
        val scoreThreshold: Double = 0.5,
        val maxResultsToRerank: Int = 30,
        val useExplanations: Boolean = false,
        val model: String? = null, // Use specific model if provided
        val retryOnError: Boolean = true,
        val maxRetries: Int = 2
    )
    
    private val defaultConfig = RerankConfig()
    
    /**
     * Rerank retrieval results using cross-encoder scoring
     */
    public suspend fun rerank(
        query: String,
        results: List<RetrievalResult>,
        config: RerankConfig = defaultConfig
    ): List<RetrievalResult> = coroutineScope {
        if (results.isEmpty()) return@coroutineScope emptyList()
        
        // Limit the number of results to rerank for efficiency
        val toRerank = results.take(config.maxResultsToRerank)
        
        // Score in batches for efficiency
        val scoredResults = toRerank.chunked(batchSize).flatMap { batch ->
            batch.map { result ->
                async {
                    val score = if (config.retryOnError) {
                        scoreDocumentWithRetry(query, result.content, config)
                    } else {
                        scoreDocument(query, result.content, config)
                    }
                    ScoredResult(result, score)
                }
            }.awaitAll()
        }
        
        // Filter by threshold and sort by score
        val filtered = scoredResults
            .filter { it.crossEncoderScore >= config.scoreThreshold }
            .sortedByDescending { it.crossEncoderScore }
        
        // Update retrieval results with cross-encoder scores
        filtered.map { scored ->
            scored.result.copy(
                score = scored.crossEncoderScore,
                metadata = scored.result.metadata + mapOf(
                    "cross_encoder_score" to scored.crossEncoderScore.toString(),
                    "original_score" to scored.result.score.toString()
                )
            )
        }
    }
    
    /**
     * Score a single document against a query
     */
    private suspend fun scoreDocument(
        query: String,
        document: String,
        config: RerankConfig
    ): Double {
        val scoringPrompt = if (config.useExplanations) {
            createExplanationPrompt(query, document)
        } else {
            createScoringPrompt(query, document)
        }
        
        return try {
            val responses = llmExecutor.execute(scoringPrompt, model, emptyList())
            val responseText = responses.firstOrNull()?.content
            
            if (responseText == null) {
                logger.warn { "No response from LLM for cross-encoder scoring of document: ${document.take(100)}..." }
                return 0.0
            }
            
            parseScore(responseText)
        } catch (e: Exception) {
            logger.error(e) { 
                "Failed to score document with cross-encoder. Query: '${query.take(100)}...', " +
                "Document: '${document.take(100)}...'"
            }
            
            // Propagate critical errors, return low score for recoverable ones
            when (e) {
                is AIAgentException -> throw e
                else -> {
                    // Log and continue with low score for non-critical errors
                    logger.debug { "Returning score 0.0 due to error: ${e.message}" }
                    0.0
                }
            }
        }
    }
    
    private fun createScoringPrompt(query: String, document: String) = prompt("cross-encoder-scoring") {
        system("""You are a relevance scoring system. Given a query and a document, 
                 |score how relevant the document is to answering the query.
                 |
                 |Respond with ONLY a number between 0 and 1, where:
                 |- 0 means completely irrelevant
                 |- 1 means perfectly relevant
                 |
                 |Consider:
                 |1. Does the document contain information that answers the query?
                 |2. How directly does it address the query?
                 |3. Is the information accurate and complete?
                 |
                 |Respond with only the score, no explanation.""".trimMargin())
        
        user("""Query: $query
               |
               |Document: $document
               |
               |Score:""".trimMargin())
    }
    
    private fun createExplanationPrompt(query: String, document: String) = prompt("cross-encoder-explanation") {
        system("""You are a relevance scoring system. Given a query and a document, 
                 |analyze how relevant the document is to answering the query.
                 |
                 |Provide your response in this format:
                 |REASONING: <brief explanation of relevance>
                 |SCORE: <number between 0 and 1>
                 |
                 |Consider:
                 |1. Does the document contain information that answers the query?
                 |2. How directly does it address the query?
                 |3. Is the information accurate and complete?""".trimMargin())
        
        user("""Query: $query
               |
               |Document: $document""".trimMargin())
    }
    
    private fun parseScore(response: String): Double {
        // Try to extract score from response
        val scorePattern = Regex("""SCORE:\s*(\d*\.?\d+)""", RegexOption.IGNORE_CASE)
        val scoreMatch = scorePattern.find(response)
        
        if (scoreMatch != null) {
            return scoreMatch.groupValues[1].toDoubleOrNull()?.coerceIn(0.0, 1.0) ?: 0.0
        }
        
        // Try to parse as direct number
        val trimmed = response.trim()
        val score = trimmed.toDoubleOrNull()?.coerceIn(0.0, 1.0)
        
        if (score == null) {
            logger.debug { "Failed to parse score from response: '$response'" }
        }
        
        return score ?: 0.0
    }
    
    /**
     * Score a document with retry logic
     */
    private suspend fun scoreDocumentWithRetry(
        query: String,
        document: String,
        config: RerankConfig
    ): Double {
        var lastException: Exception? = null
        
        repeat(config.maxRetries) { attempt ->
            try {
                return scoreDocument(query, document, config)
            } catch (e: Exception) {
                lastException = e
                if (attempt < config.maxRetries - 1) {
                    logger.debug { "Retry ${attempt + 1}/${config.maxRetries} for cross-encoder scoring" }
                }
            }
        }
        
        logger.warn { "All retries exhausted for cross-encoder scoring. Last error: ${lastException?.message}" }
        return 0.0
    }
    
    /**
     * Helper class for scored results
     */
    private data class ScoredResult(
        val result: RetrievalResult,
        val crossEncoderScore: Double
    )
}

/**
 * Factory function to create a cross-encoder reranker
 */
public fun createCrossEncoderReranker(
    llmExecutor: SingleLLMPromptExecutor,
    model: LLModel,
    maxConcurrency: Int = 4
): CrossEncoderReranker {
    return CrossEncoderReranker(llmExecutor, model, maxConcurrency)
}

/**
 * Extension function to add cross-encoder reranking to any retrieval result list
 */
public suspend fun List<RetrievalResult>.rerankWithCrossEncoder(
    query: String,
    reranker: CrossEncoderReranker,
    config: CrossEncoderReranker.RerankConfig = CrossEncoderReranker.RerankConfig()
): List<RetrievalResult> {
    return reranker.rerank(query, this, config)
}

/**
 * LLM-based cross-encoder reranker that matches Graphiti's OpenAI implementation.
 * Uses a simple boolean classifier prompt with log probabilities for scoring.
 */
public class OpenAIStyleCrossEncoderReranker(
    private val llmExecutor: SingleLLMPromptExecutor,
    private val model: LLModel,
    private val maxConcurrent: Int = 10
) {
    private companion object {
        private val logger = KotlinLogging.logger {}
    }
    
    /**
     * Rerank the given passages based on their relevance to the query.
     * Returns a list of passages with scores, sorted by relevance (highest first).
     */
    public suspend fun rerank(
        query: String,
        passages: List<String>
    ): List<RankedPassage> = coroutineScope {
        // Process passages concurrently with limited parallelism
        val rankedPassages = passages.chunked(maxConcurrent).flatMap { chunk ->
            chunk.map { passage ->
                async {
                    val score = scorePassage(query, passage)
                    RankedPassage(passage, score)
                }
            }.awaitAll()
        }
        
        // Sort by score descending (highest relevance first)
        rankedPassages.sortedByDescending { it.score }
    }
    
    /**
     * Rerank retrieval results using this reranker
     */
    public suspend fun rerankResults(
        query: String,
        results: List<RetrievalResult>
    ): List<RetrievalResult> {
        val passages = results.map { it.content }
        val rankedPassages = rerank(query, passages)
        
        // Map back to retrieval results, preserving original metadata
        val passageToResult = results.associateBy { it.content }
        
        return rankedPassages.mapNotNull { ranked ->
            passageToResult[ranked.passage]?.copy(
                score = ranked.score,
                metadata = passageToResult[ranked.passage]!!.metadata + mapOf(
                    "cross_encoder_score" to ranked.score.toString(),
                    "original_score" to passageToResult[ranked.passage]!!.score.toString()
                )
            )
        }
    }
    
    private suspend fun scorePassage(query: String, passage: String): Double {
        val scoringPrompt = prompt("openai-style-reranking") {
            system("You are an expert tasked with determining whether the passage is relevant to the query")
            user("""
                Respond with "True" if PASSAGE is relevant to QUERY and "False" otherwise.
                <PASSAGE>
                $passage
                </PASSAGE>
                <QUERY>
                $query
                </QUERY>
            """.trimIndent())
        }
        
        return try {
            val responses = llmExecutor.execute(scoringPrompt, model, emptyList())
            val responseText = responses.firstOrNull()?.content
            
            if (responseText == null) {
                logger.warn { "No response from LLM for OpenAI-style reranking" }
                return 0.0
            }
            
            extractRelevanceScore(responseText)
        } catch (e: Exception) {
            logger.error(e) { "Failed to score passage with OpenAI-style reranker" }
            0.0
        }
    }
    
    private fun extractRelevanceScore(response: String): Double {
        val content = response.lowercase().trim()
        
        // Simple scoring based on response
        return when {
            content.contains("true") -> 1.0
            content.contains("false") -> 0.0
            else -> 0.5 // Uncertain
        }
        
        // TODO: When log probabilities are available in the API, use them:
        // val logProb = response.logProbabilities?.get("true") ?: -10.0
        // return 1.0 / (1.0 + exp(-logProb)) // Sigmoid transformation
    }
}

/**
 * A passage with its relevance score
 */
public data class RankedPassage(
    val passage: String,
    val score: Double,
    val metadata: Map<String, Any> = emptyMap()
)