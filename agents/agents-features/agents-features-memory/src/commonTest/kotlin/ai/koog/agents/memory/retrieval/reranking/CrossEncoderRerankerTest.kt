package ai.koog.agents.memory.retrieval.reranking

import ai.koog.agents.memory.retrieval.RetrievalResult
import ai.koog.agents.memory.retrieval.Provenance
import ai.koog.agents.core.agent.context.AIAgentLLMContext
import ai.koog.prompt.message.Message
import ai.koog.prompt.dsl.Prompt
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertNotNull

class CrossEncoderRerankerTest {
    
    private fun createTestResults(): List<RetrievalResult> {
        return listOf(
            RetrievalResult(
                content = "Alice is a software engineer at Google.",
                score = 0.7,
                provenance = emptyList(),
                metadata = mapOf("id" to "1")
            ),
            RetrievalResult(
                content = "Bob works as a data scientist at Microsoft.",
                score = 0.6,
                provenance = emptyList(),
                metadata = mapOf("id" to "2")
            ),
            RetrievalResult(
                content = "Charlie is a product manager at Apple.",
                score = 0.5,
                provenance = emptyList(),
                metadata = mapOf("id" to "3")
            )
        )
    }
    
    @Test
    fun testCrossEncoderRerankerConfig() {
        // Test default config
        val defaultConfig = CrossEncoderReranker.RerankConfig()
        assertEquals(0.0, defaultConfig.scoreThreshold)
        assertEquals(30, defaultConfig.maxResultsToRerank)
        assertEquals(false, defaultConfig.useExplanations)
        assertEquals(true, defaultConfig.retryOnError)
        assertEquals(2, defaultConfig.maxRetries)
        
        // Test custom config
        val customConfig = CrossEncoderReranker.RerankConfig(
            scoreThreshold = 0.5,
            maxResultsToRerank = 10,
            useExplanations = true,
            retryOnError = false,
            maxRetries = 1
        )
        assertEquals(0.5, customConfig.scoreThreshold)
        assertEquals(10, customConfig.maxResultsToRerank)
        assertEquals(true, customConfig.useExplanations)
        assertEquals(false, customConfig.retryOnError)
        assertEquals(1, customConfig.maxRetries)
    }
    
    @Test
    fun testRetrievalResultStructure() {
        val results = createTestResults()
        
        assertEquals(3, results.size)
        
        // Check first result
        val first = results[0]
        assertEquals("Alice is a software engineer at Google.", first.content)
        assertEquals(0.7, first.score)
        assertEquals("1", first.metadata["id"])
        
        // Check second result
        val second = results[1]
        assertEquals("Bob works as a data scientist at Microsoft.", second.content)
        assertEquals(0.6, second.score)
        assertEquals("2", second.metadata["id"])
        
        // Check third result
        val third = results[2]
        assertEquals("Charlie is a product manager at Apple.", third.content)
        assertEquals(0.5, third.score)
        assertEquals("3", third.metadata["id"])
    }
    
    @Test
    fun testEmptyResultHandling() {
        val emptyResults = emptyList<RetrievalResult>()
        assertTrue(emptyResults.isEmpty())
    }
    
    // Note: Full integration tests with actual LLM executor would require
    // mock setup which is tested separately in integration tests
    
    @Test
    fun `test OpenAI-style cross-encoder reranking with relevant and irrelevant passages`() = runTest {
        // Create mock LLM context that returns True/False based on content
        val mockLLM = object : AIAgentLLMContext {
            override suspend fun <T> writeSession(block: suspend AIAgentLLMContext.() -> T): T {
                @Suppress("UNCHECKED_CAST")
                return block() as T
            }
            
            override suspend fun requestLLM(
                modelId: String,
                temperature: Double,
                maxTokens: Int
            ): Result<List<Message>> {
                // Extract the user message to determine response
                val messages = (prompt as? Prompt)?.messages ?: emptyList()
                val userMessage = messages.lastOrNull { it.role == "user" }?.content ?: ""
                
                // Simple relevance check based on keywords
                val response = when {
                    userMessage.contains("Steve") && userMessage.contains("leader") -> "True"
                    userMessage.contains("weather") && userMessage.contains("leader") -> "False"
                    userMessage.contains("faction") && userMessage.contains("Red faction") -> "True"
                    else -> "False"
                }
                
                return Result.success(listOf(Message.Assistant(response)))
            }
            
            override var prompt: Prompt? = null
        }
        
        val reranker = OpenAIStyleCrossEncoderReranker(
            llm = mockLLM,
            maxConcurrent = 2
        )
        
        // Test passages
        val passages = listOf(
            "Steve is the leader of the Red faction",
            "The weather is sunny today",
            "Alex joined the Red faction last week",
            "The mountain base has strong defenses"
        )
        
        // Rerank for a query about Red faction leader
        val ranked = reranker.rerank("Who is the leader of the Red faction?", passages)
        
        // Verify results
        assertEquals(4, ranked.size)
        
        // Steve passage should be ranked highest
        assertEquals("Steve is the leader of the Red faction", ranked[0].passage)
        assertEquals(1.0, ranked[0].score)
        
        // Weather passage should have low score
        val weatherResult = ranked.find { it.passage.contains("weather") }
        assertNotNull(weatherResult)
        assertEquals(0.0, weatherResult.score)
    }
    
    @Test
    fun `test cross-encoder integration with retrieval results`() = runTest {
        // Create mock LLM that scores based on relevance
        val mockLLM = object : AIAgentLLMContext {
            override suspend fun <T> writeSession(block: suspend AIAgentLLMContext.() -> T): T {
                @Suppress("UNCHECKED_CAST")
                return block() as T
            }
            
            override suspend fun requestLLM(
                modelId: String,
                temperature: Double,
                maxTokens: Int
            ): Result<List<Message>> {
                val messages = (prompt as? Prompt)?.messages ?: emptyList()
                val userMessage = messages.lastOrNull { it.role == "user" }?.content ?: ""
                
                // Score based on content relevance
                val response = when {
                    userMessage.contains("base location") && userMessage.contains("coordinates") -> "True"
                    userMessage.contains("base defenses") && userMessage.contains("coordinates") -> "True"
                    else -> "False"
                }
                
                return Result.success(listOf(Message.Assistant(response)))
            }
            
            override var prompt: Prompt? = null
        }
        
        val reranker = OpenAIStyleCrossEncoderReranker(llm = mockLLM)
        
        // Create retrieval results
        val results = listOf(
            RetrievalResult(
                content = "The base is located at coordinates 0,64,0",
                score = 0.7,
                provenance = listOf(Provenance.Memory("fact-1")),
                metadata = mapOf("type" to "location")
            ),
            RetrievalResult(
                content = "Players can trade items at the market",
                score = 0.8,
                provenance = listOf(Provenance.Memory("fact-2")),
                metadata = mapOf("type" to "gameplay")
            ),
            RetrievalResult(
                content = "Base defenses include walls and turrets",
                score = 0.6,
                provenance = listOf(Provenance.Memory("fact-3")),
                metadata = mapOf("type" to "defense")
            )
        )
        
        // Rerank results
        val reranked = reranker.rerankResults("What are the base coordinates?", results)
        
        // Verify reranking
        assertEquals(3, reranked.size)
        
        // Base location should be ranked first
        assertTrue(reranked[0].content.contains("coordinates"))
        assertEquals(1.0, reranked[0].score)
        assertEquals("1.0", reranked[0].metadata["cross_encoder_score"])
        assertEquals("0.7", reranked[0].metadata["original_score"])
        
        // Trading info should be ranked last
        val tradingResult = reranked.find { it.content.contains("trade") }
        assertNotNull(tradingResult)
        assertEquals(0.0, tradingResult.score)
    }
}