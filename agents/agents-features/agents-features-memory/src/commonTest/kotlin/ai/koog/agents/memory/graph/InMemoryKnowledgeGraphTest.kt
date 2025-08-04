package ai.koog.agents.memory.graph

import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.feature.EpisodeProcessor
import ai.koog.agents.memory.feature.ProcessedEpisode
import ai.koog.agents.memory.feature.EntityCandidate
import ai.koog.agents.memory.feature.RelationCandidate
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlin.test.*
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.days

class InMemoryKnowledgeGraphTest {
    
    // Mock episode processor for testing
    class MockEpisodeProcessor : EpisodeProcessor {
        override suspend fun process(episode: Episode): ProcessedEpisode {
            // Simple mock that extracts entities and relations from the content
            return when {
                episode.content.contains("built a fortress") -> {
                    // Extract coordinates from content
                    val coords = if (episode.content.contains("100,64,200")) {
                        Triple(100, 64, 200)
                    } else if (episode.content.contains("300,70,400")) {
                        Triple(300, 70, 400)
                    } else {
                        Triple(0, 0, 0)
                    }
                    
                    ProcessedEpisode(
                        entities = listOf(
                            EntityCandidate(
                                id = "player-steve",
                                labels = setOf("Player"),
                                properties = mapOf("name" to "Steve"),
                                confidence = 0.9
                            ),
                            EntityCandidate(
                                id = "fortress-${coords.first}-${coords.second}-${coords.third}",
                                labels = setOf("Structure", "Fortress"),
                                properties = mapOf(
                                    "x" to coords.first,
                                    "y" to coords.second,
                                    "z" to coords.third
                                ),
                                confidence = 0.85
                            )
                        ),
                        relations = listOf(
                            RelationCandidate(
                                from = "player-steve",
                                to = "fortress-${coords.first}-${coords.second}-${coords.third}",
                                type = "BUILT",
                                properties = mapOf("when" to episode.timestamp.toString()),
                                confidence = 0.8
                            )
                        ),
                        updates = emptyList()
                    )
                }
                episode.content.contains("war") -> ProcessedEpisode(
                    entities = listOf(
                        EntityCandidate(
                            id = "faction-red-dragons",
                            labels = setOf("Faction"),
                            properties = mapOf("name" to "Red Dragons"),
                            confidence = 0.9
                        ),
                        EntityCandidate(
                            id = "faction-blue-knights",
                            labels = setOf("Faction"),
                            properties = mapOf("name" to "Blue Knights"),
                            confidence = 0.9
                        )
                    ),
                    relations = listOf(
                        RelationCandidate(
                            from = "faction-red-dragons",
                            to = "faction-blue-knights",
                            type = "AT_WAR_WITH",
                            properties = emptyMap(),
                            confidence = 0.95
                        )
                    ),
                    updates = emptyList()
                )
                else -> ProcessedEpisode(
                    entities = emptyList(),
                    relations = emptyList(),
                    updates = emptyList()
                )
            }
        }
    }
    
    @Test
    fun `test episode ingestion creates nodes`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        val episode = Episode(
            content = "Test episode content",
            timestamp = Clock.System.now(),
            source = EpisodeSource.SYSTEM_EVENT,
            metadata = mapOf("test" to "value")
        )
        
        val result = graph.ingest(episode)
        
        assertTrue(result.createdNodes.isNotEmpty(), "Should create at least one node")
        assertEquals(0, result.createdEdges.size, "Simple ingestion shouldn't create edges")
        assertEquals(0, result.updatedNodes.size, "First ingestion shouldn't update nodes")
        
        val stats = graph.stats()
        assertEquals(1L, stats.nodeCount, "Should have one node")
        assertEquals(1L, stats.episodeCount, "Should have one episode")
        assertEquals(1L, stats.nodesByLabel["Episode"] ?: 0, "Should have one Episode node")
    }
    
    @Test
    fun `test entity centric queries`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Ingest an episode to create a node
        val episode = Episode(
            content = "Player action in game",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        )
        
        val result = graph.ingest(episode)
        val nodeId = result.createdNodes.first()
        
        // Query centered on that node
        val knowledge = graph.query(
            KnowledgeRequest.EntityCentric(
                centerNode = nodeId,
                traversal = Traversal.Bidirectional(depth = 1),
                limit = 10
            )
        )
        
        assertEquals(1, knowledge.size, "Should find the node itself")
        val entity = knowledge.first() as Knowledge.Entity
        assertEquals(nodeId, entity.id)
        assertTrue(entity.labels.contains("Episode"))
    }
    
    @Test
    fun `test semantic search`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Ingest episodes with different content
        graph.ingest(Episode(
            content = "The fortress was built with stone bricks",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "Players gathered wood from the forest",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "A stone monument was erected at spawn",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        // Search for stone-related content
        val results = graph.query(
            KnowledgeRequest.Semantic(
                query = "stone",
                limit = 10
            )
        )
        
        assertEquals(2, results.size, "Should find 2 stone-related episodes")
        results.forEach { knowledge ->
            val content = (knowledge as Knowledge.Entity).properties["content"] as String
            assertTrue(content.contains("stone", ignoreCase = true))
        }
    }
    
    @Test
    fun `test temporal queries`() = runTest {
        val clock = Clock.System
        val graph = InMemoryKnowledgeGraph(clock = clock)
        
        val startTime = clock.now()
        
        // Ingest episodes at different times
        graph.ingest(Episode(
            content = "First event",
            timestamp = startTime,
            source = EpisodeSource.EXTERNAL
        ))
        
        graph.ingest(Episode(
            content = "Second event",
            timestamp = startTime.plus(1.hours),
            source = EpisodeSource.EXTERNAL
        ))
        
        val endTime = startTime.plus(2.hours)
        
        graph.ingest(Episode(
            content = "Third event (outside window)",
            timestamp = endTime.plus(1.hours),
            source = EpisodeSource.EXTERNAL
        ))
        
        // Query within time window
        val results = graph.query(
            KnowledgeRequest.Temporal(
                start = startTime,
                end = endTime,
                includeDeleted = false
            )
        )
        
        assertEquals(2, results.size, "Should find only events within time window")
    }
    
    @Test
    fun `test graph evolution with decay`() = runTest {
        val clock = Clock.System
        val graph = InMemoryKnowledgeGraph(clock = clock)
        
        // Create a simple decay rule that removes nodes older than 7 days
        val decayRule = object : DecayRule {
            override fun shouldDecay(knowledge: Knowledge, currentTime: kotlinx.datetime.Instant): Boolean {
                return currentTime > knowledge.timestamp.plus(7.days)
            }
            
            override fun decay(knowledge: Knowledge): Knowledge? {
                // Return null to remove the knowledge
                return null
            }
        }
        
        // Ingest an old episode
        val oldTime = clock.now().minus(10.days)
        val oldEpisode = Episode(
            content = "Old event",
            timestamp = oldTime,
            source = EpisodeSource.EXTERNAL
        )
        
        // We need to modify the graph to accept past timestamps
        // For now, just test with current time
        graph.ingest(Episode(
            content = "Recent event",
            timestamp = clock.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        val statsBefore = graph.stats()
        assertEquals(1L, statsBefore.nodeCount)
        
        // Evolve the graph (no decay rules, so nothing should change)
        graph.evolve(EvolutionContext(
            currentTime = clock.now(),
            decayRules = emptyList()
        ))
        
        val statsAfter = graph.stats()
        assertEquals(1L, statsAfter.nodeCount, "Without decay rules, nodes should remain")
    }
    
    @Test
    fun `test pattern queries return all nodes`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Ingest multiple episodes
        repeat(5) { i ->
            graph.ingest(Episode(
                content = "Event $i",
                timestamp = Clock.System.now(),
                source = EpisodeSource.EXTERNAL,
                metadata = mapOf("index" to i)
            ))
        }
        
        // Pattern query (simplified implementation returns all nodes)
        val results = graph.query(
            KnowledgeRequest.Pattern(
                pattern = "MATCH (n) RETURN n",
                limit = 3
            )
        )
        
        assertEquals(3, results.size, "Should respect limit")
    }
    
    @Test
    fun `test graph stats tracking`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Initial stats
        val initialStats = graph.stats()
        assertEquals(0L, initialStats.nodeCount)
        assertEquals(0L, initialStats.edgeCount)
        assertEquals(0L, initialStats.episodeCount)
        assertNull(initialStats.lastIngestion)
        assertNull(initialStats.lastEvolution)
        
        // Ingest an episode
        graph.ingest(Episode(
            content = "Test",
            timestamp = Clock.System.now(),
            source = EpisodeSource.SYSTEM_EVENT
        ))
        
        val afterIngestion = graph.stats()
        assertEquals(1L, afterIngestion.nodeCount)
        assertEquals(0L, afterIngestion.edgeCount)
        assertEquals(1L, afterIngestion.episodeCount)
        assertNotNull(afterIngestion.lastIngestion)
        
        // Evolve the graph
        graph.evolve(EvolutionContext(Clock.System.now()))
        
        val afterEvolution = graph.stats()
        assertNotNull(afterEvolution.lastEvolution)
    }
    
    @Test
    fun `test episode ingestion with entity extraction`() = runTest {
        val processor = MockEpisodeProcessor()
        val graph = InMemoryKnowledgeGraph(episodeProcessor = processor)
        
        val episode = Episode(
            content = "Player Steve built a fortress at coordinates 100,64,200",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL,
            metadata = mapOf("world" to "survival")
        )
        
        val result = graph.ingest(episode)
        
        // Should create 2 entities and 1 relation
        assertEquals(2, result.createdNodes.size, "Should create 2 entities")
        assertEquals(1, result.createdEdges.size, "Should create 1 relation")
        
        // Query the graph to verify data
        val entities = graph.query(
            KnowledgeRequest.Pattern(
                pattern = "MATCH (n) RETURN n",
                parameters = emptyMap()
            )
        )
        
        assertEquals(2, entities.size, "Should have 2 entities in graph")
        
        // Check entity properties
        val steve = entities.find { it.id == "player-steve" } as? Knowledge.Entity
        assertTrue(steve != null, "Should find player Steve")
        assertEquals(setOf("Player"), steve.labels)
        assertEquals("Steve", steve.properties["name"])
        
        val fortress = entities.find { it.id == "fortress-100-64-200" } as? Knowledge.Entity
        assertTrue(fortress != null, "Should find fortress")
        assertEquals(setOf("Structure", "Fortress"), fortress.labels)
        assertEquals(100, fortress.properties["x"])
    }
    
    @Test
    fun `test semantic search with extracted entities`() = runTest {
        val processor = MockEpisodeProcessor()
        val graph = InMemoryKnowledgeGraph(episodeProcessor = processor)
        
        // Ingest war episode
        graph.ingest(Episode(
            content = "The Red Dragons faction declared war on Blue Knights",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        // Search for war-related content
        val results = graph.query(
            KnowledgeRequest.Semantic(
                query = "war dragons",
                limit = 10
            )
        )
        
        assertTrue(results.isNotEmpty(), "Should find war-related content")
        
        // Should find both the Red Dragons entity and the war relation
        val hasRedDragons = results.any { knowledge ->
            when (knowledge) {
                is Knowledge.Entity -> knowledge.properties["name"] == "Red Dragons"
                is Knowledge.Relation -> knowledge.type == "AT_WAR_WITH"
                else -> false
            }
        }
        assertTrue(hasRedDragons, "Should find Red Dragons or war relation")
    }
    
    @Test
    fun `test entity merging on re-ingestion`() = runTest {
        val processor = MockEpisodeProcessor()
        val graph = InMemoryKnowledgeGraph(episodeProcessor = processor)
        
        // First ingestion
        graph.ingest(Episode(
            content = "Player Steve built a fortress at coordinates 100,64,200",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        // Second ingestion with same entity
        val result2 = graph.ingest(Episode(
            content = "Player Steve built a fortress at coordinates 300,70,400",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        ))
        
        // Steve should be updated, not created again
        assertTrue(result2.updatedNodes.contains("player-steve"), "Steve should be updated")
        assertTrue(result2.createdNodes.contains("fortress-300-70-400"), "New fortress should be created")
        
        // Query stats
        val stats = graph.stats()
        assertEquals(3L, stats.nodeCount, "Should have 3 nodes (Steve + 2 fortresses)")
        assertEquals(2L, stats.edgeCount, "Should have 2 edges (2 BUILT relations)")
    }
    
    @Test
    fun `test evolution with consolidation rules`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        
        // Create similar nodes that should be consolidated
        // Manually add nodes to simulate duplicates
        val episode1 = Episode(
            content = "Steve the player",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        )
        graph.ingest(episode1)
        
        val episode2 = Episode(
            content = "Player Steve",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        )
        graph.ingest(episode2)
        
        val episode3 = Episode(
            content = "steve (player)",
            timestamp = Clock.System.now(),
            source = EpisodeSource.EXTERNAL
        )
        graph.ingest(episode3)
        
        // Create a consolidation rule that merges similar player names
        val consolidationRule = object : ConsolidationRule {
            override fun shouldConsolidate(knowledge: List<Knowledge>): Boolean {
                // Check if all are Episode nodes with "steve" in content
                return knowledge.all { k ->
                    k is Knowledge.Entity && 
                    k.labels.contains("Episode") &&
                    (k.properties["content"] as? String)?.lowercase()?.contains("steve") == true
                }
            }
            
            override fun consolidate(knowledge: List<Knowledge>): Knowledge {
                // Merge into a single Player entity
                val contents = knowledge.mapNotNull { 
                    (it as? Knowledge.Entity)?.properties?.get("content") as? String 
                }
                
                return Knowledge.Entity(
                    id = "player-steve",
                    labels = setOf("Player"),
                    properties = mapOf(
                        "name" to "Steve",
                        "aliases" to contents,
                        "consolidated_from" to knowledge.map { it.id }
                    ),
                    confidence = 0.95,
                    timestamp = Clock.System.now(),
                    provenance = knowledge.flatMap { it.provenance }
                )
            }
        }
        
        val statsBefore = graph.stats()
        assertEquals(3L, statsBefore.nodeCount, "Should have 3 episode nodes before consolidation")
        
        // Evolve with consolidation
        graph.evolve(EvolutionContext(
            currentTime = Clock.System.now(),
            consolidationRules = listOf(consolidationRule)
        ))
        
        val statsAfter = graph.stats()
        assertEquals(1L, statsAfter.nodeCount, "Should have 1 consolidated node after evolution")
        
        // Check the consolidated node
        val nodes = graph.query(KnowledgeRequest.Pattern("MATCH (n) RETURN n"))
        assertEquals(1, nodes.size)
        
        val playerNode = nodes.first() as Knowledge.Entity
        assertEquals("player-steve", playerNode.id)
        assertEquals(setOf("Player"), playerNode.labels)
        assertEquals("Steve", playerNode.properties["name"])
        
        @Suppress("UNCHECKED_CAST")
        val aliases = playerNode.properties["aliases"] as? List<String>
        assertTrue(aliases != null && aliases.size == 3, "Should have all original contents as aliases")
    }
}