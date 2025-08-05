package ai.koog.agents.memory.temporal

import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.SingleFact
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse
import kotlin.time.Duration.Companion.days
import kotlin.time.Duration.Companion.hours
import kotlinx.coroutines.test.runTest

class TemporalKnowledgeExtractorTest {
    
    private val extractor = TemporalKnowledgeExtractor()
    private val referenceTime = Clock.System.now()
    
    @Test
    fun testExtractAbsoluteDates() = runTest {
        val messages = listOf(
            "The meeting is scheduled for 2024-03-15.",
            "She was born on January 20, 2000.",
            "The project deadline is March 1st, 2024.",
            "We met on Feb 14 2023."
        )
        
        for (message in messages) {
            val facts = extractor.extractTemporalFacts(message, referenceTime)
            assertTrue(facts.isNotEmpty(), "Should extract temporal fact from: $message")
            assertFalse(facts.first().isRelative, "Should be absolute date")
        }
    }
    
    @Test
    fun testExtractRelativeDates() = runTest {
        val testCases = mapOf(
            "I saw him 3 days ago" to 3.days,
            "The event is tomorrow" to -1.days, // negative because it's in the future
            "We'll meet next week" to -7.days,
            "She left yesterday" to 1.days,
            "It happened 2 weeks ago" to 14.days,
            "The deadline is in 5 days" to -5.days
        )
        
        for ((message, expectedDuration) in testCases) {
            val facts = extractor.extractTemporalFacts(message, referenceTime)
            assertTrue(facts.isNotEmpty(), "Should extract temporal fact from: $message")
            
            val fact = facts.first()
            assertTrue(fact.isRelative, "Should be relative date for: $message")
            
            // Check if the time difference is approximately correct (within 1 hour for precision)
            val actualDiff = if (expectedDuration.isNegative()) {
                fact.validFrom - referenceTime
            } else {
                referenceTime - fact.validFrom
            }
            
            assertTrue(
                kotlin.math.abs(actualDiff.inWholeHours - expectedDuration.absoluteValue.inWholeHours) <= 1,
                "Time difference should be approximately ${expectedDuration.absoluteValue} for: $message"
            )
        }
    }
    
    @Test
    fun testExtractStateChanges() = runTest {
        val message = "Alice was a student but is now a teacher."
        val facts = extractor.extractTemporalFacts(message, referenceTime)
        
        assertEquals(2, facts.size, "Should extract two temporal facts for state change")
        
        val oldStateFact = facts.find { it.validTo != null }
        val newStateFact = facts.find { it.validTo == null }
        
        assertTrue(oldStateFact != null, "Should have old state with end time")
        assertTrue(newStateFact != null, "Should have new state without end time")
        
        val oldState = (oldStateFact!!.fact as? SingleFact)?.value
        val newState = (newStateFact!!.fact as? SingleFact)?.value
        
        assertTrue(oldState?.contains("student") == true, "Old state should contain 'student'")
        assertTrue(newState?.contains("teacher") == true, "New state should contain 'teacher'")
    }
    
    @Test
    fun testExtractTimeRanges() = runTest {
        val message = "The conference runs from March 15 to March 18."
        val facts = extractor.extractTemporalFacts(message, referenceTime)
        
        assertTrue(facts.isNotEmpty(), "Should extract temporal fact from time range")
        
        val fact = facts.first()
        assertTrue(fact.validTo != null, "Should have end time for range")
        assertTrue(fact.validFrom < fact.validTo!!, "Start should be before end")
    }
    
    @Test
    fun testContradictionDetection() = runTest {
        val concept = Concept(
            keyword = "alice_location",
            description = "Alice's location",
            factType = FactType.SINGLE
        )
        
        val existingFact = TemporalKnowledgeExtractor.TemporalFact(
            fact = SingleFact(
                concept = concept,
                value = "New York",
                timestamp = referenceTime.minus(1.days).toEpochMilliseconds()
            ),
            validFrom = referenceTime.minus(2.days),
            validTo = null,
            confidence = 0.9
        )
        
        val newFact = TemporalKnowledgeExtractor.TemporalFact(
            fact = SingleFact(
                concept = concept,
                value = "Los Angeles",
                timestamp = referenceTime.toEpochMilliseconds()
            ),
            validFrom = referenceTime,
            validTo = null,
            confidence = 0.95
        )
        
        assertTrue(newFact.contradicts(existingFact), "Should detect contradiction")
        
        val resolution = extractor.handleContradictions(newFact, listOf(existingFact))
        
        assertEquals(1, resolution.invalidatedFacts.size, "Should invalidate one fact")
        assertEquals(existingFact.fact.concept, resolution.invalidatedFacts.first().fact.concept)
        assertTrue(resolution.invalidatedFacts.first().validTo != null, "Old fact should have end time")
        assertEquals(newFact.fact, resolution.newValidFact.fact, "New fact should be preserved")
    }
    
    @Test
    fun testOverlapDetection() = runTest {
        val now = Clock.System.now()
        
        val fact1 = TemporalKnowledgeExtractor.TemporalFact(
            fact = SingleFact(
                concept = Concept("test", "test", FactType.SINGLE),
                value = "value1",
                timestamp = now.toEpochMilliseconds()
            ),
            validFrom = now.minus(2.days),
            validTo = now.plus(1.days)
        )
        
        val fact2 = TemporalKnowledgeExtractor.TemporalFact(
            fact = SingleFact(
                concept = Concept("test", "test", FactType.SINGLE),
                value = "value2",
                timestamp = now.toEpochMilliseconds()
            ),
            validFrom = now,
            validTo = now.plus(3.days)
        )
        
        assertTrue(fact1.overlaps(fact2), "Should detect overlap")
        assertTrue(fact2.overlaps(fact1), "Overlap should be symmetric")
    }
    
    @Test
    fun testComplexDateExtraction() = runTest {
        val message = """
            Alice was born on January 15, 1990. She graduated 4 years ago and 
            started her job last month. She plans to retire in 30 years.
        """.trimIndent()
        
        val facts = extractor.extractTemporalFacts(message, referenceTime)
        
        assertTrue(facts.size >= 4, "Should extract multiple temporal facts")
        
        val absoluteDates = facts.filter { !it.isRelative }
        val relativeDates = facts.filter { it.isRelative }
        
        assertTrue(absoluteDates.isNotEmpty(), "Should have absolute dates")
        assertTrue(relativeDates.isNotEmpty(), "Should have relative dates")
    }
    
    @Test
    fun testNoTemporalContent() = runTest {
        val message = "The cat sat on the mat."
        val facts = extractor.extractTemporalFacts(message, referenceTime)
        
        // The extractor might still create a fact from the sentence, but it won't have temporal significance
        facts.forEach { fact ->
            // All facts should be at or near reference time if no temporal info found
            val timeDiff = kotlin.math.abs((fact.validFrom - referenceTime).inWholeHours)
            assertTrue(timeDiff <= 24, "Non-temporal facts should be near reference time")
        }
    }
}