package ai.koog.agents.memory.graph

import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.feature.*
import ai.koog.agents.memory.feature.*
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class GraphMemoryTest {
    
    @Test
    fun `test basic episode ingestion directly on graph`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        val episode = Episode(
            content = "Player Steve built a fortress at coordinates 100,64,200",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL,
            metadata = mapOf("world" to "survival", "dimension" to "overworld")
        )
        
        val result = graph.ingest(episode)
        
        assertTrue(result.createdNodes.isNotEmpty(), "Should create at least one node")
        
        // Query stats
        val stats = graph.stats()
        assertEquals(1L, stats.episodeCount, "Should have one episode")
        assertTrue(stats.nodeCount > 0, "Should have created nodes")
    }
    
    // @Test // TODO: Fix semantic search implementation in InMemoryKnowledgeGraph
    fun `test semantic search on graph`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Ingest various episodes
        graph.ingest(Episode(
            content = "The Red Dragons faction declared war on Blue Knights",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "Blue Knights built defensive walls around their castle",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "Green Traders remain neutral in the conflict",
            timestamp = Clock.System.now(),
            source = EpisodeSource.USER_INPUT
        ))
        
        // Search for war-related content
        val results = graph.query(
            KnowledgeRequest.Semantic(
                query = "war conflict",
                limit = 10
            )
        )
        
        assertTrue(results.isNotEmpty(), "Should find war-related content")
    }
    
    
    @Test
    fun `test graph temporal queries`() = runTest {
        val clock = Clock.System
        val graph = InMemoryKnowledgeGraph(clock = clock)
        
        val startTime = clock.now()
        
        // Ingest episodes at different times
        graph.ingest(Episode(
            content = "Player Carol joined faction Eagles",
            timestamp = startTime,
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "Eagles faction conquered the Mountain Peak",
            timestamp = startTime,
            source = EpisodeSource.EXTERNAL
        ))
        
        val endTime = clock.now()
        
        // Query the time window
        val knowledge = graph.query(
            KnowledgeRequest.Temporal(
                start = startTime,
                end = endTime,
                includeDeleted = false
            )
        )
        
        assertTrue(knowledge.isNotEmpty(), "Should find knowledge in time window")
    }
}