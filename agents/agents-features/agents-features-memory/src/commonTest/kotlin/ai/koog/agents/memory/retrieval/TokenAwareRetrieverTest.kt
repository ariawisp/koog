package ai.koog.agents.memory.retrieval

import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.agents.memory.retrieval.config.TokenAwareRetrievalConfig
import ai.koog.agents.memory.retrieval.metrics.InMemoryMetricsCollector
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.RequestMetaInfo
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TokenAwareRetrieverTest {
    
    /**
     * Mock tokenizer that counts characters as tokens for predictable testing
     */
    private class CharCountTokenizer : PromptTokenizer {
        override fun tokenCountFor(message: Message): Int = message.content.length
        override fun tokenCountFor(prompt: Prompt): Int = prompt.messages.sumOf { tokenCountFor(it) }
    }
    
    /**
     * Mock retrieval provider that returns predictable results
     */
    private class MockRetrievalProvider(
        private val results: List<RetrievalResult>
    ) : RetrievalProvider {
        var retrievalCount = 0
        
        override fun supports(recipe: RetrievalRecipe): Boolean = true
        
        override suspend fun retrieve(query: RetrievalQuery, securityContext: ai.koog.agents.memory.security.SecurityContext?): List<RetrievalResult> {
            retrievalCount++
            return results
        }
    }
    
    @Test
    fun testTokenBudgetFiltering() = runTest {
        // Create results with different token counts
        val results = listOf(
            RetrievalResult(
                content = "Short content", // 13 chars
                score = 0.9,
                provenance = emptyList(),
                metadata = emptyMap()
            ),
            RetrievalResult(
                content = "This is a medium length content that has more tokens", // 52 chars
                score = 0.8,
                provenance = emptyList(),
                metadata = emptyMap()
            ),
            RetrievalResult(
                content = "This is a very long content that has many many tokens and should exceed most budgets", // 84 chars
                score = 0.7,
                provenance = emptyList(),
                metadata = emptyMap()
            )
        )
        
        val mockProvider = MockRetrievalProvider(results)
        val tokenizer = CharCountTokenizer()
        val config = TokenAwareRetrievalConfig(
            defaultTokenBudget = 70,
            reserveTokensForQuery = 10
        )
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer,
            config = config
        )
        
        val query = RetrievalQuery(
            text = "test query",
            k = 10
        )
        
        val filtered = tokenAwareRetriever.retrieve(query)
        
        // With budget of 70 - 10 = 60, should only get first two results (13 + 52 = 65 > 60)
        // But since we're greedy, we take what fits
        assertEquals(2, filtered.size)
        assertEquals("Short content", filtered[0].content)
        assertEquals("This is a medium length content that has more tokens", filtered[1].content)
    }
    
    @Test
    fun testGreedyOptimizationStrategy() = runTest {
        // Create results with different efficiency scores
        val results = listOf(
            RetrievalResult(
                content = "Low score long content with many tokens", // 39 chars, score 0.3 = ~0.0077 score/token
                score = 0.3,
                provenance = emptyList(),
                metadata = emptyMap()
            ),
            RetrievalResult(
                content = "High score", // 10 chars, score 0.9 = 0.09 score/token
                score = 0.9,
                provenance = emptyList(),
                metadata = emptyMap()
            ),
            RetrievalResult(
                content = "Medium", // 6 chars, score 0.6 = 0.1 score/token
                score = 0.6,
                provenance = emptyList(),
                metadata = emptyMap()
            )
        )
        
        val mockProvider = MockRetrievalProvider(results)
        val tokenizer = CharCountTokenizer()
        val config = TokenAwareRetrievalConfig(
            defaultTokenBudget = 20,
            reserveTokensForQuery = 0,
            enableAdaptiveBudgeting = false  // Disable adaptive budgeting for this test
        )
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer,
            config = config
        )
        
        val query = RetrievalQuery(text = "test")
        
        val optimized = tokenAwareRetriever.retrieve(query)
        
        // With budget of 20 and adaptive budgeting disabled, greedy optimization will select:
        // 1. Medium (6 tokens, 0.1 score/token) - running total: 6 tokens
        // 2. High score (10 tokens, 0.09 score/token) - running total: 16 tokens
        // 3. Low score (39 tokens) won't fit as 16 + 39 = 55 > 20
        assertEquals(2, optimized.size)
        assertTrue(optimized.any { it.content == "Medium" }, "Should include Medium (highest score/token)")
        assertTrue(optimized.any { it.content == "High score" }, "Should include High score")
    }
    
    @Test
    fun testAdaptiveBudgeting() = runTest {
        val results = listOf(
            RetrievalResult(
                content = "A".repeat(150), // 150 chars, exceeds budget
                score = 0.9,
                provenance = emptyList(),
                metadata = emptyMap()
            ),
            RetrievalResult(
                content = "Small", // 5 chars, very small
                score = 0.8,
                provenance = emptyList(),
                metadata = emptyMap()
            )
        )
        
        val mockProvider = MockRetrievalProvider(results)
        val tokenizer = CharCountTokenizer()
        val config = TokenAwareRetrievalConfig(
            defaultTokenBudget = 100,
            reserveTokensForQuery = 0,
            enableAdaptiveBudgeting = true,
            minResultTokenSize = 10,
            tokenOverflowTolerance = 10
        )
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer,
            config = config
        )
        
        val query = RetrievalQuery(text = "test")
        val adaptiveResults = tokenAwareRetriever.retrieve(query)
        
        // Should get the small result due to adaptive budgeting
        assertEquals(1, adaptiveResults.size)
        assertEquals("Small", adaptiveResults[0].content)
    }
    
    @Test
    fun testMetricsCollection() = runTest {
        val results = listOf(
            RetrievalResult(
                content = "Test content",
                score = 0.9,
                provenance = emptyList(),
                metadata = emptyMap()
            )
        )
        
        val mockProvider = MockRetrievalProvider(results)
        val tokenizer = CharCountTokenizer()
        val metricsCollector = InMemoryMetricsCollector()
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer,
            metricsCollector = metricsCollector
        )
        
        val query = RetrievalQuery(text = "test")
        tokenAwareRetriever.retrieve(query)
        
        val stats = metricsCollector.getProviderStats("TokenAwareRetriever")
        
        assertEquals(1, stats.totalQueries)
        assertTrue(stats.avgLatencyMs != null && stats.avgLatencyMs!! >= 0)
        assertEquals(0.9, stats.avgScore)
    }
    
    @Test
    fun testEmptyResultHandling() = runTest {
        val mockProvider = MockRetrievalProvider(emptyList())
        val tokenizer = CharCountTokenizer()
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer
        )
        
        val query = RetrievalQuery(text = "test")
        val results = tokenAwareRetriever.retrieve(query)
        
        assertTrue(results.isEmpty())
    }
    
    @Test
    fun testSingleResultExceedingBudget() = runTest {
        val results = listOf(
            RetrievalResult(
                content = "A".repeat(1000), // Way over budget
                score = 0.9,
                provenance = emptyList(),
                metadata = emptyMap()
            )
        )
        
        val mockProvider = MockRetrievalProvider(results)
        val tokenizer = CharCountTokenizer()
        val config = TokenAwareRetrievalConfig(
            defaultTokenBudget = 100,
            reserveTokensForQuery = 0
        )
        
        val tokenAwareRetriever = TokenAwareRetriever(
            baseRetriever = mockProvider,
            tokenizer = tokenizer,
            config = config
        )
        
        val query = RetrievalQuery(text = "test")
        val filtered = tokenAwareRetriever.retrieve(query)
        
        // Should still return the single result as fallback
        assertEquals(1, filtered.size)
        assertEquals(results[0].content, filtered[0].content)
    }
}