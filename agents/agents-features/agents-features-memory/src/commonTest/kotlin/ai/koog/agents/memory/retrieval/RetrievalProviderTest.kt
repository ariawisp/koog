package ai.koog.agents.memory.retrieval

import ai.koog.agents.memory.feature.nodes.RetrievalQueryBuilder
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.memory.model.SingleFact
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.memory.retrieval.providers.VectorRetrievalProvider
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Basic tests for the retrieval provider functionality
 */
class RetrievalProviderTest {
    
    @Test
    fun `test vector retrieval provider supports correct recipes`() {
        val provider = VectorRetrievalProvider()
        
        assertTrue(provider.supports(RetrievalRecipe.VECTOR_SIMILARITY))
        assertTrue(provider.supports(RetrievalRecipe.TEXT_BM25))
        assertTrue(provider.supports(RetrievalRecipe.HYBRID_RRF))
        assertTrue(!provider.supports(RetrievalRecipe.HYBRID_NODE_DISTANCE))
        assertTrue(!provider.supports(RetrievalRecipe.HYBRID_CROSS_ENCODER))
    }
    
    @Test
    fun `test retrieval query builder creates correct query`() {
        val builder = RetrievalQueryBuilder().apply {
            text = "Find information about bases"
            k = 10
            centerNode = "player:steve"
            recipe = RetrievalRecipe.HYBRID_RRF
            filters {
                subject(MemorySubject.Everything)
                factType("ownership")
            }
        }
        
        val query = builder.build()
        
        assertEquals("Find information about bases", query.text)
        assertEquals(10, query.k)
        assertEquals("player:steve", query.centerNode)
        assertEquals(RetrievalRecipe.HYBRID_RRF, query.recipe)
        assertTrue(query.filters.subjects.contains(MemorySubject.Everything))
        assertTrue(query.filters.factTypes.contains("ownership"))
    }
    
    @Test
    fun `test retrieval result with fact provenance`() = runTest {
        // Create a mock memory provider
        val mockMemory = object : AgentMemoryProvider {
            override suspend fun save(fact: ai.koog.agents.memory.model.Fact, subject: MemorySubject, scope: MemoryScope) {}
            override suspend fun load(concept: Concept, subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
                return emptyList()
            }
            override suspend fun loadAll(subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
                return emptyList()
            }
            override suspend fun loadByDescription(description: String, subject: MemorySubject, scope: MemoryScope): List<ai.koog.agents.memory.model.Fact> {
                if (description.contains("base")) {
                    return listOf(
                        SingleFact(
                            concept = Concept("base-owner", "Owner of a base", FactType.SINGLE),
                            value = "Player123 owns the main base",
                            timestamp = 1234567890
                        )
                    )
                }
                return emptyList()
            }
        }
        
        val provider = VectorRetrievalProvider(memoryProvider = mockMemory)
        
        val query = RetrievalQuery(
            text = "Who owns the base?",
            k = 5,
            target = RetrievalTarget.FACTS
        )
        
        val results = provider.retrieve(query)
        
        assertTrue(results.isNotEmpty())
        assertEquals("base-owner: Player123 owns the main base", results[0].content)
        assertTrue(results[0].provenance.any { it is Provenance.Fact })
        
        val factProvenance = results[0].provenance.first { it is Provenance.Fact } as Provenance.Fact
        assertEquals("base-owner", factProvenance.conceptKeyword)
    }
}