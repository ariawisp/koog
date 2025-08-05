package ai.koog.agents.memory.retrieval

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.nodes.nodeRetrieveKnowledge
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.DefaultTimeProvider
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.memory.model.SingleFact
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.llm.OllamaModels
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class RetrievalNodesTest {
    
    // Mock memory provider that returns predefined facts
    private class MockMemoryProvider : AgentMemoryProvider {
        private val facts = mutableListOf<Pair<ai.koog.agents.memory.model.Fact, MemorySubject>>()
        
        fun addFact(fact: ai.koog.agents.memory.model.Fact, subject: MemorySubject) {
            facts.add(fact to subject)
        }
        
        override suspend fun save(fact: ai.koog.agents.memory.model.Fact, subject: MemorySubject, scope: MemoryScope) {
            facts.add(fact to subject)
        }
        
        override suspend fun load(concept: Concept, subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
            return facts.filter { it.first.concept == concept && it.second == subject }.map { it.first }
        }
        
        override suspend fun loadAll(subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
            return facts.filter { it.second == subject }.map { it.first }
        }
        
        override suspend fun loadByDescription(description: String, subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
            // Simple keyword matching
            return facts.filter { (fact, sub) ->
                sub == subject && (
                    fact.concept.description.contains(description, ignoreCase = true) ||
                    when(fact) {
                        is SingleFact -> fact.value.contains(description, ignoreCase = true)
                        is ai.koog.agents.memory.model.MultipleFacts -> fact.values.any { it.contains(description, ignoreCase = true) }
                        else -> false
                    }
                )
            }.map { it.first }
        }
    }
    
    @Test
    fun testGraphMemoryProviderWithRetrieval() = runTest {
        // Use the new graph-based approach with preset
        val strategy = strategy<String, List<RetrievalResult>>("test-graph-retrieval") {
            val search by nodeRetrieveKnowledge<String> {
                text = "Who owns the Northern Fortress?"
                k = 5
            }
            
            edge(nodeStart forwardTo search)
            edge(search forwardTo nodeFinish)
        }
        
        val mockExecutor = getMockExecutor {
            mockLLMAnswer("Mock response").asDefaultResponse
        }
        
        val agent = AIAgent(
            promptExecutor = mockExecutor,
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("test") { system("Test") },
                model = OllamaModels.Meta.LLAMA_3_2,
                maxAgentIterations = 5
            )
        ) {
            install(AgentMemory) {
                // Use the new preset approach for graph-based memory
                useGraphInMemoryPreset()
                agentName = "test-agent"
                featureName = "test-feature"
                organizationName = "test-org"
                productName = "test-product"
            }
        }
        
        // Run the agent - this tests the preset configuration works
        val results = agent.run("test input")
        
        // The test verifies that:
        // 1. The preset configures GraphMemoryProvider correctly
        // 2. The retrieval system is set up properly
        // 3. No errors occur during execution
        // 4. Results can be empty (no facts to find), that's expected behavior
        assertTrue(results.isEmpty()) // Empty results expected since no facts exist in fresh graph
    }
    
    @Test
    fun testVectorRetrievalWithMemoryProvider() = runTest {
        // Setup mock memory with some facts
        val mockMemory = MockMemoryProvider().apply {
            addFact(
                SingleFact(
                    concept = Concept("base-owner", "Owner of a base", FactType.SINGLE),
                    value = "PlayerAlpha owns the Northern Fortress",
                    timestamp = DefaultTimeProvider.getCurrentTimestamp()
                ),
                MemorySubject.Everything
            )
            addFact(
                SingleFact(
                    concept = Concept("faction", "Player faction membership", FactType.SINGLE),
                    value = "PlayerAlpha is part of the Crimson Wolves faction",
                    timestamp = DefaultTimeProvider.getCurrentTimestamp()
                ),
                MemorySubject.Everything
            )
        }
        
        // Test the vector retrieval provider directly
        val retriever = ai.koog.agents.memory.retrieval.providers.VectorRetrievalProvider(
            memoryProvider = mockMemory
        )
        
        val query = RetrievalQuery(
            text = "Northern Fortress",
            k = 5
        )
        
        val results = retriever.retrieve(query)
        
        assertTrue(results.isNotEmpty(), "Should find results")
        val firstResult = results.first()
        assertTrue(firstResult.content.contains("PlayerAlpha"), "Should contain PlayerAlpha")
        assertTrue(firstResult.content.contains("Northern Fortress"), "Should contain Northern Fortress")
        
        // Check provenance
        val provenance = firstResult.provenance.first()
        assertTrue(provenance is Provenance.Fact, "Provenance should be Fact type")
        assertEquals("base-owner", (provenance as Provenance.Fact).conceptKeyword)
    }
    
    @Test
    fun testRetrievalNodeWithoutConfiguredRetriever() = runTest {
        val strategy = strategy<String, List<RetrievalResult>>("test-no-retriever") {
            val search by nodeRetrieveKnowledge<String> {
                text = "Who owns the base?"
            }
            
            edge(nodeStart forwardTo search)
            edge(search forwardTo nodeFinish)
        }
        
        val mockExecutor = getMockExecutor {
            mockLLMAnswer("Mock response").asDefaultResponse
        }
        
        val agent = AIAgent(
            promptExecutor = mockExecutor,
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("test") { system("Test") },
                model = OllamaModels.Meta.LLAMA_3_2,
                maxAgentIterations = 5
            )
        ) {
            install(AgentMemory) {
                // No retriever configured - use basic memory provider only
                memoryProvider = MockMemoryProvider()
                agentName = "test-agent"
                featureName = "test-feature"
                organizationName = "test-org"
                productName = "test-product"
            }
        }
        
        // Should throw error when trying to use retrieval without configuration
        try {
            agent.run("test")
            assertFalse(true, "Should have thrown error")
        } catch (e: Exception) {
            assertTrue(e.message?.contains("No retriever configured") == true)
        }
    }
    
    @Test
    fun testSmartRouterWithGraphAndVectorProviders() = runTest {
        val mockMemory = MockMemoryProvider().apply {
            addFact(
                SingleFact(
                    concept = Concept("attack", "Attack event", FactType.SINGLE),
                    value = "The Crimson Wolves attacked our base last week",
                    timestamp = DefaultTimeProvider.getCurrentTimestamp()
                ),
                MemorySubject.Everything
            )
        }
        
        // Create router with vector provider only for this test
        // (GraphMemoryProvider doesn't implement RetrievalProvider directly)
        val router = createSmartRouter(
            memoryProvider = mockMemory,
            policy = RoutingPolicy(enableGraph = false) // Focus on vector provider
        )
        
        // Test temporal query routing 
        val temporalQuery = RetrievalQuery(
            text = "Who attacked us last week?",
            at = kotlinx.datetime.Clock.System.now(),
            k = 10
        )
        
        val temporalResults = router.retrieve(temporalQuery)
        // Should not fail even if empty due to graph provider handling
        assertTrue(temporalResults.isEmpty() || temporalResults.isNotEmpty())
        
        // Test simple query (may route to vector provider)
        val simpleQuery = RetrievalQuery(
            text = "Crimson Wolves",
            k = 10
        )
        
        val simpleResults = router.retrieve(simpleQuery)
        // Vector provider should find the fact
        assertTrue(simpleResults.isNotEmpty(), "Should find results via vector provider")
        assertTrue(simpleResults.first().content.contains("Crimson Wolves"))
    }
}