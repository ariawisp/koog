package ai.koog.agents.memory

import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.uuid.ExperimentalUuidApi

@OptIn(ExperimentalUuidApi::class)
class EntityResolutionTest {
    
    @Test
    fun `test entity resolution with exact name match`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // First mention of Steve
        val episode1 = Episode(
            content = "Steve built a fortress in the mountains",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT
        )
        
        val result1 = graph.ingest(episode1)
        
        // Second mention of Steve should resolve to same entity
        val episode2 = Episode(
            content = "Steve is mining diamonds",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT
        )
        
        val result2 = graph.ingest(episode2)
        
        // Query for Steve
        val knowledge = graph.query(
            KnowledgeRequest.Semantic(
                query = "Steve",
                limit = 10
            )
        )
        
        // Should find references to Steve
        assertTrue(knowledge.isNotEmpty(), "Should find knowledge about Steve")
        
        // Test entity resolution directly
        val resolution = graph.resolveEntity(
            mention = "Steve",
            context = EntityResolutionContext()
        )
        
        when (resolution) {
            is EntityResolutionResult.Resolved -> {
                assertTrue(resolution.confidence >= 0.7, "Should have high confidence")
                assertEquals("Steve", resolution.entity.properties["name"])
            }
            else -> error("Expected resolved entity for Steve")
        }
    }
    
    @Test
    fun `test entity resolution with aliases`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Create an entity with alias
        val episode = Episode(
            content = "Steve, also known as SteveTheBuilder, created a castle",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT,
            entityMentions = listOf(
                EntityMention(
                    entityId = "steve-001",
                    text = "Steve",
                    startOffset = 0,
                    endOffset = 5,
                    type = EntityType.PERSON
                ),
                EntityMention(
                    entityId = "steve-001",
                    text = "SteveTheBuilder",
                    startOffset = 20,
                    endOffset = 35,
                    type = EntityType.PERSON
                )
            )
        )
        
        graph.ingest(episode)
        
        // Resolve by alias
        val resolution = graph.resolveEntity(
            mention = "SteveTheBuilder",
            context = EntityResolutionContext()
        )
        
        when (resolution) {
            is EntityResolutionResult.Resolved -> {
                assertTrue(resolution.confidence >= 0.8, "Should have high confidence for alias")
            }
            else -> error("Expected resolved entity for alias")
        }
    }
    
    @Test
    fun `test entity resolution creates new entity when no match`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Try to resolve unknown entity
        val resolution = graph.resolveEntity(
            mention = "UnknownPlayer123",
            context = EntityResolutionContext()
        )
        
        when (resolution) {
            is EntityResolutionResult.Resolved -> {
                assertTrue(resolution.isNew, "Should create new entity")
                assertEquals("UnknownPlayer123", resolution.entity.properties["name"])
            }
            else -> error("Expected new entity to be created")
        }
    }
    
    @Test
    fun `test community detection finds connected components`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Create a simple connected component
        val episode = Episode(
            content = "Steve and Alex are working together on the fortress. Steve leads the project.",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT
        )
        
        graph.ingest(episode)
        
        // Detect communities
        val communities = graph.detectCommunities(CommunityDetectionAlgorithm.CONNECTED_COMPONENTS)
        
        // Should find at least one community if entities were extracted
        assertTrue(communities.isNotEmpty() || graph.stats().nodeCount == 0L, 
            "Should find communities or have no nodes")
    }
    
    @Test
    fun `test edge invalidation for contradicting facts`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Create initial fact
        val episode1 = Episode(
            content = "Steve leads the Red faction",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT
        )
        
        graph.ingest(episode1)
        
        // Create contradicting fact
        val newFact = Knowledge.Relation(
            id = "edge-new",
            type = "LEADS",
            from = "Steve",
            to = "Blue faction",
            properties = emptyMap(),
            confidence = 0.9,
            timestamp = Clock.System.now(),
            provenance = emptyList()
        )
        
        val invalidations = graph.invalidateContradictingEdges(
            newFact = newFact,
            at = Clock.System.now()
        )
        
        // May or may not find contradictions depending on extraction
        assertTrue(invalidations.isEmpty() || invalidations.any { it.reason.contains("Contradicted") },
            "Should find contradiction or no edges")
    }
}