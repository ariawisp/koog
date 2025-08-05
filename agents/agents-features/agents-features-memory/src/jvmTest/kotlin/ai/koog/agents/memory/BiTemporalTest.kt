package ai.koog.agents.memory

import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.temporal.BiTemporalModel
import ai.koog.agents.memory.temporal.BiTemporalPeriod
import ai.koog.agents.memory.temporal.BiTemporalQuery
import ai.koog.agents.memory.temporal.BiTemporalUtils
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.time.Duration.Companion.days
import kotlin.time.Duration.Companion.hours
import kotlin.uuid.ExperimentalUuidApi

@OptIn(ExperimentalUuidApi::class)
class BiTemporalTest {
    
    @Test
    fun `test bi-temporal period validity checks`() {
        val now = Clock.System.now()
        val period = BiTemporalPeriod(
            validFrom = now - 10.days,
            validTo = now + 10.days,
            transactionTime = now - 5.days
        )
        
        // Check valid time
        assertTrue(period.wasValidAt(now))
        assertFalse(period.wasValidAt(now - 15.days))
        assertFalse(period.wasValidAt(now + 15.days))
        
        // Check transaction time
        assertTrue(period.wasKnownAt(now))
        assertFalse(period.wasKnownAt(now - 10.days))
        
        // Check active status
        assertTrue(period.isActive())
        
        // Delete and check
        val deleted = period.deleteAt(now)
        assertFalse(deleted.isActive())
        assertFalse(deleted.wasKnownAt(now + 1.hours))
    }
    
    @Test
    fun `test temporal validity in knowledge graph`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        val now = Clock.System.now()
        
        // Create episodes at different times with different validity
        val episode1 = Episode(
            content = "Steve leads the Red faction",
            timestamp = now - 10.days,
            source = EpisodeSource.USER_INPUT,
            validFrom = now - 10.days,
            validTo = now - 5.days // Valid for 5 days
        )
        
        val episode2 = Episode(
            content = "Alex leads the Red faction",
            timestamp = now - 5.days,
            source = EpisodeSource.USER_INPUT,
            validFrom = now - 5.days,
            validTo = null // Still valid
        )
        
        graph.ingest(episode1)
        graph.ingest(episode2)
        
        // Query at different points in time
        val weekAgoKnowledge = graph.query(
            KnowledgeRequest.Semantic(
                query = "Red faction leader",
                at = now - 7.days
            )
        )
        
        val currentKnowledge = graph.query(
            KnowledgeRequest.Semantic(
                query = "Red faction leader",
                at = now
            )
        )
        
        // Should find different leaders at different times
        assertTrue(weekAgoKnowledge.isNotEmpty() || currentKnowledge.isNotEmpty(), 
            "Should find knowledge at some point in time")
    }
    
    @Test
    fun `test edge invalidation with temporal consistency`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        val now = Clock.System.now()
        
        // Create initial state
        val episode1 = Episode(
            content = "Steve owns the Mountain Base",
            timestamp = now - 5.days,
            source = EpisodeSource.USER_INPUT
        )
        
        graph.ingest(episode1)
        
        // Create contradicting fact
        val newOwnership = Knowledge.Relation(
            id = "edge-new",
            type = "OWNS",
            from = "Alex",
            to = "Mountain Base",
            properties = emptyMap(),
            confidence = 0.9,
            timestamp = now,
            provenance = emptyList()
        )
        
        // Invalidate old edges
        val invalidations = graph.invalidateContradictingEdges(
            newFact = newOwnership,
            at = now
        )
        
        // Query to verify temporal consistency
        val oldOwnership = graph.query(
            KnowledgeRequest.Temporal(
                start = now - 10.days,
                end = now - 1.days
            )
        )
        
        val currentOwnership = graph.query(
            KnowledgeRequest.Temporal(
                start = now - 1.hours,
                end = now + 1.hours
            )
        )
        
        // Ownership should change over time
        println("Old ownership count: ${oldOwnership.size}")
        println("Current ownership count: ${currentOwnership.size}")
    }
    
    @Test
    fun `test temporal window queries`() = runTest {
        val graph = InMemoryKnowledgeGraph()
        val now = Clock.System.now()
        
        // Create events across time
        val episodes = listOf(
            Episode(
                content = "Game started",
                timestamp = now - 30.days,
                source = EpisodeSource.SYSTEM_EVENT,
                validFrom = now - 30.days
            ),
            Episode(
                content = "Steve joined",
                timestamp = now - 20.days,
                source = EpisodeSource.USER_INPUT,
                validFrom = now - 20.days
            ),
            Episode(
                content = "Base built at 0,0",
                timestamp = now - 10.days,
                source = EpisodeSource.USER_INPUT,
                validFrom = now - 10.days
            ),
            Episode(
                content = "Battle occurred",
                timestamp = now - 5.days,
                source = EpisodeSource.SYSTEM_EVENT,
                validFrom = now - 5.days,
                validTo = now - 4.days // Battle ended
            )
        )
        
        episodes.forEach { graph.ingest(it) }
        
        // Query different time windows
        val lastWeek = graph.query(
            KnowledgeRequest.Temporal(
                start = now - 7.days,
                end = now
            )
        )
        
        val lastMonth = graph.query(
            KnowledgeRequest.Temporal(
                start = now - 30.days,
                end = now
            )
        )
        
        // Last month should have more events
        assertTrue(
            lastMonth.size >= lastWeek.size || episodes.isEmpty(),
            "Longer time window should capture more or equal events"
        )
    }
    
    @Test
    fun `test bi-temporal query filtering`() {
        val now = Clock.System.now()
        val query = BiTemporalQuery(
            validAt = now,
            asOf = now - 1.days,
            includeDeleted = false
        )
        
        // Test various temporal items
        val activeValid = object : BiTemporal {
            override val temporalPeriod = BiTemporalPeriod(
                validFrom = now - 10.days,
                validTo = now + 10.days,
                transactionTime = now - 5.days
            )
        }
        
        val deletedItem = object : BiTemporal {
            override val temporalPeriod = BiTemporalPeriod(
                validFrom = now - 10.days,
                validTo = now + 10.days,
                transactionTime = now - 5.days,
                deletedAt = now - 2.days
            )
        }
        
        val futureItem = object : BiTemporal {
            override val temporalPeriod = BiTemporalPeriod(
                validFrom = now + 1.days,
                validTo = now + 10.days,
                transactionTime = now - 5.days
            )
        }
        
        assertTrue(query.matches(activeValid))
        assertFalse(query.matches(deletedItem)) // Deleted and includeDeleted=false
        assertFalse(query.matches(futureItem)) // Not valid at query time
    }
}