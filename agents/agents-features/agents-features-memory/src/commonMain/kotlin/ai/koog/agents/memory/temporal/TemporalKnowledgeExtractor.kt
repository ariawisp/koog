package ai.koog.agents.memory.temporal

import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.model.Fact
import ai.koog.agents.memory.model.SingleFact
import ai.koog.agents.memory.model.MultipleFacts
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import kotlinx.datetime.*
import kotlin.time.Duration
import kotlin.time.Duration.Companion.days
import kotlin.time.Duration.Companion.hours
import kotlin.time.Duration.Companion.minutes

/**
 * Extracts temporal information from text and handles temporal reasoning
 * for knowledge graphs. This is critical for beating benchmarks that test
 * temporal understanding.
 */
public class TemporalKnowledgeExtractor {
    
    /**
     * Represents a temporal fact with time bounds
     */
    public data class TemporalFact(
        val fact: Fact,
        val validFrom: Instant,
        val validTo: Instant? = null,
        val confidence: Double = 1.0,
        val isRelative: Boolean = false
    ) {
        public val timeRange: ClosedRange<Instant>
            get() = validFrom..(validTo ?: Instant.DISTANT_FUTURE)
        
        public fun overlaps(other: TemporalFact): Boolean {
            val thisEnd = validTo ?: Instant.DISTANT_FUTURE
            val otherEnd = other.validTo ?: Instant.DISTANT_FUTURE
            return validFrom <= otherEnd && thisEnd >= other.validFrom
        }
        
        public fun contradicts(other: TemporalFact): Boolean {
            // Facts contradict if they have the same concept but different values
            return fact.concept == other.fact.concept && 
                   overlaps(other) &&
                   !valuesMatch(other)
        }
        
        private fun valuesMatch(other: TemporalFact): Boolean {
            return when (val thisFact = fact) {
                is SingleFact -> {
                    other.fact is SingleFact && thisFact.value == other.fact.value
                }
                is MultipleFacts -> {
                    other.fact is MultipleFacts && thisFact.values.toSet() == other.fact.values.toSet()
                }
                else -> false
            }
        }
    }
    
    /**
     * Resolution of contradicting temporal facts
     */
    public data class ContradictionResolution(
        val invalidatedFacts: List<TemporalFact>,
        val newValidFact: TemporalFact,
        val resolution: ResolutionStrategy
    )
    
    public enum class ResolutionStrategy {
        INVALIDATE_OLD,      // New fact invalidates old ones
        MERGE_FACTS,         // Merge facts into composite
        SPLIT_TEMPORAL       // Split time ranges to avoid overlap
    }
    
    /**
     * Extract temporal facts from a message with context
     */
    public suspend fun extractTemporalFacts(
        message: String,
        referenceTime: Instant,
        context: List<Message> = emptyList()
    ): List<TemporalFact> {
        val temporalFacts = mutableListOf<TemporalFact>()
        
        // Extract absolute dates
        val absoluteDates = extractAbsoluteDates(message)
        absoluteDates.forEach { (text, instant) ->
            val fact = extractFactAroundDate(message, text)
            if (fact != null) {
                temporalFacts.add(
                    TemporalFact(
                        fact = fact,
                        validFrom = instant,
                        confidence = 0.95,
                        isRelative = false
                    )
                )
            }
        }
        
        // Extract relative dates
        val relativeDates = extractRelativeDates(message, referenceTime)
        relativeDates.forEach { (text, instant, duration) ->
            val fact = extractFactAroundDate(message, text)
            if (fact != null) {
                temporalFacts.add(
                    TemporalFact(
                        fact = fact,
                        validFrom = instant,
                        confidence = 0.85, // Lower confidence for relative dates
                        isRelative = true
                    )
                )
            }
        }
        
        // Extract state changes (was/is/will be patterns)
        val stateChanges = extractStateChanges(message, referenceTime)
        temporalFacts.addAll(stateChanges)
        
        // Extract durations and ranges
        val ranges = extractTimeRanges(message, referenceTime)
        ranges.forEach { (start, end, fact) ->
            temporalFacts.add(
                TemporalFact(
                    fact = fact,
                    validFrom = start,
                    validTo = end,
                    confidence = 0.9
                )
            )
        }
        
        return resolveTemporalConflicts(temporalFacts, context)
    }
    
    /**
     * Handle contradictions between new and existing temporal facts
     */
    public suspend fun handleContradictions(
        newFact: TemporalFact,
        existingFacts: List<TemporalFact>
    ): ContradictionResolution {
        // Find contradicting facts
        val conflicts = existingFacts.filter { existing ->
            existing.contradicts(newFact)
        }
        
        if (conflicts.isEmpty()) {
            return ContradictionResolution(
                invalidatedFacts = emptyList(),
                newValidFact = newFact,
                resolution = ResolutionStrategy.INVALIDATE_OLD
            )
        }
        
        // Apply resolution strategy based on confidence and recency
        val strategy = determineResolutionStrategy(newFact, conflicts)
        
        return when (strategy) {
            ResolutionStrategy.INVALIDATE_OLD -> {
                // Invalidate old facts by setting their validTo to newFact's validFrom
                val invalidated = conflicts.map { conflict ->
                    conflict.copy(validTo = newFact.validFrom.minus(1.minutes))
                }
                ContradictionResolution(
                    invalidatedFacts = invalidated,
                    newValidFact = newFact,
                    resolution = strategy
                )
            }
            
            ResolutionStrategy.MERGE_FACTS -> {
                // Merge facts into a composite fact
                val mergedFact = mergeFacts(newFact, conflicts)
                ContradictionResolution(
                    invalidatedFacts = conflicts,
                    newValidFact = mergedFact,
                    resolution = strategy
                )
            }
            
            ResolutionStrategy.SPLIT_TEMPORAL -> {
                // Split time ranges to avoid overlap
                val splitResult = splitTemporalRanges(newFact, conflicts)
                ContradictionResolution(
                    invalidatedFacts = conflicts,
                    newValidFact = splitResult,
                    resolution = strategy
                )
            }
        }
    }
    
    private fun extractAbsoluteDates(text: String): List<Pair<String, Instant>> {
        val dates = mutableListOf<Pair<String, Instant>>()
        
        // ISO date pattern: 2024-01-15
        val isoDatePattern = Regex("""\b(\d{4})-(\d{2})-(\d{2})\b""")
        isoDatePattern.findAll(text).forEach { match ->
            try {
                val date = LocalDate.parse(match.value)
                val instant = date.atStartOfDayIn(TimeZone.UTC)
                dates.add(match.value to instant)
            } catch (e: Exception) {
                // Ignore invalid dates
            }
        }
        
        // Natural date patterns: January 15, 2024 or Jan 15 2024
        val naturalDatePattern = Regex(
            """\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2}),?\s+(\d{4})\b""",
            RegexOption.IGNORE_CASE
        )
        naturalDatePattern.findAll(text).forEach { match ->
            try {
                val monthStr = match.groupValues[1]
                val day = match.groupValues[2].toInt()
                val year = match.groupValues[3].toInt()
                val month = parseMonth(monthStr)
                
                val date = LocalDate(year, month, day)
                val instant = date.atStartOfDayIn(TimeZone.UTC)
                dates.add(match.value to instant)
            } catch (e: Exception) {
                // Ignore invalid dates
            }
        }
        
        return dates
    }
    
    private fun extractRelativeDates(
        text: String, 
        referenceTime: Instant
    ): List<Triple<String, Instant, Duration>> {
        val relativeDates = mutableListOf<Triple<String, Instant, Duration>>()
        
        // Past relative patterns
        val pastPatterns = listOf(
            Regex("""(\d+)\s+(day|days)\s+ago""", RegexOption.IGNORE_CASE) to { n: Int -> n.days },
            Regex("""(\d+)\s+(week|weeks)\s+ago""", RegexOption.IGNORE_CASE) to { n: Int -> (n * 7).days },
            Regex("""(\d+)\s+(month|months)\s+ago""", RegexOption.IGNORE_CASE) to { n: Int -> (n * 30).days },
            Regex("""(\d+)\s+(year|years)\s+ago""", RegexOption.IGNORE_CASE) to { n: Int -> (n * 365).days },
            Regex("""yesterday""", RegexOption.IGNORE_CASE) to { _: Int -> 1.days },
            Regex("""last\s+week""", RegexOption.IGNORE_CASE) to { _: Int -> 7.days },
            Regex("""last\s+month""", RegexOption.IGNORE_CASE) to { _: Int -> 30.days }
        )
        
        // Future relative patterns
        val futurePatterns = listOf(
            Regex("""in\s+(\d+)\s+(day|days)""", RegexOption.IGNORE_CASE) to { n: Int -> n.days },
            Regex("""in\s+(\d+)\s+(week|weeks)""", RegexOption.IGNORE_CASE) to { n: Int -> (n * 7).days },
            Regex("""in\s+(\d+)\s+(month|months)""", RegexOption.IGNORE_CASE) to { n: Int -> (n * 30).days },
            Regex("""tomorrow""", RegexOption.IGNORE_CASE) to { _: Int -> 1.days },
            Regex("""next\s+week""", RegexOption.IGNORE_CASE) to { _: Int -> 7.days }
        )
        
        // Process past patterns
        pastPatterns.forEach { (pattern, durationFunc) ->
            pattern.findAll(text).forEach { match ->
                val number = if (match.groups.size > 1) {
                    match.groupValues[1].toIntOrNull() ?: 1
                } else {
                    1
                }
                val duration = durationFunc(number)
                val instant = referenceTime - duration
                relativeDates.add(Triple(match.value, instant, duration))
            }
        }
        
        // Process future patterns
        futurePatterns.forEach { (pattern, durationFunc) ->
            pattern.findAll(text).forEach { match ->
                val number = if (match.groups.size > 1) {
                    match.groupValues[1].toIntOrNull() ?: 1
                } else {
                    1
                }
                val duration = durationFunc(number)
                val instant = referenceTime + duration
                relativeDates.add(Triple(match.value, instant, duration))
            }
        }
        
        return relativeDates
    }
    
    private fun extractStateChanges(
        text: String,
        referenceTime: Instant
    ): List<TemporalFact> {
        val stateChanges = mutableListOf<TemporalFact>()
        
        // Pattern: "X was Y but is now Z"
        val changePattern = Regex(
            """(\w+)\s+was\s+(.+?)\s+but\s+(?:is\s+)?now\s+(.+?)(?:\.|,|$)""",
            RegexOption.IGNORE_CASE
        )
        
        changePattern.findAll(text).forEach { match ->
            val entity = match.groupValues[1]
            val oldState = match.groupValues[2]
            val newState = match.groupValues[3]
            
            // Create fact for old state (ended in the past)
            stateChanges.add(
                TemporalFact(
                    fact = SingleFact(
                        concept = Concept(
                            keyword = "state_of_$entity",
                            description = "State of $entity",
                            factType = FactType.SINGLE
                        ),
                        value = oldState,
                        timestamp = (referenceTime - 1.days).toEpochMilliseconds() // Approximate
                    ),
                    validFrom = Instant.DISTANT_PAST,
                    validTo = referenceTime - 1.minutes,
                    confidence = 0.8
                )
            )
            
            // Create fact for new state (started recently)
            stateChanges.add(
                TemporalFact(
                    fact = SingleFact(
                        concept = Concept(
                            keyword = "state_of_$entity",
                            description = "State of $entity",
                            factType = FactType.SINGLE
                        ),
                        value = newState,
                        timestamp = referenceTime.toEpochMilliseconds()
                    ),
                    validFrom = referenceTime,
                    validTo = null,
                    confidence = 0.9
                )
            )
        }
        
        return stateChanges
    }
    
    private fun extractTimeRanges(
        text: String,
        referenceTime: Instant
    ): List<Triple<Instant, Instant, Fact>> {
        val ranges = mutableListOf<Triple<Instant, Instant, Fact>>()
        
        // Pattern: "from X to Y"
        val rangePattern = Regex(
            """from\s+(.+?)\s+(?:to|until)\s+(.+?)(?:\.|,|$)""",
            RegexOption.IGNORE_CASE
        )
        
        rangePattern.findAll(text).forEach { match ->
            val startStr = match.groupValues[1]
            val endStr = match.groupValues[2]
            
            // Try to parse dates from the range
            val startDates = extractAbsoluteDates(startStr) + 
                           extractRelativeDates(startStr, referenceTime).map { it.first to it.second }
            val endDates = extractAbsoluteDates(endStr) + 
                         extractRelativeDates(endStr, referenceTime).map { it.first to it.second }
            
            if (startDates.isNotEmpty() && endDates.isNotEmpty()) {
                val fact = extractFactAroundDate(text, match.value)
                if (fact != null) {
                    ranges.add(
                        Triple(startDates.first().second, endDates.first().second, fact)
                    )
                }
            }
        }
        
        return ranges
    }
    
    private fun extractFactAroundDate(text: String, dateText: String): Fact? {
        // Extract the sentence containing the date
        val sentences = text.split(Regex("""\.\s+"""))
        val relevantSentence = sentences.firstOrNull { it.contains(dateText) } ?: return null
        
        // Simple extraction: create a fact from the sentence
        // In a real implementation, this would use NLP to extract structured information
        return SingleFact(
            concept = Concept(
                keyword = "temporal_event",
                description = "Event with temporal information",
                factType = FactType.SINGLE
            ),
            value = relevantSentence.trim(),
            timestamp = Clock.System.now().toEpochMilliseconds()
        )
    }
    
    private fun resolveTemporalConflicts(
        facts: List<TemporalFact>,
        context: List<Message>
    ): List<TemporalFact> {
        // Group facts by concept
        val grouped = facts.groupBy { it.fact.concept }
        
        val resolved = mutableListOf<TemporalFact>()
        
        grouped.forEach { (concept, conceptFacts) ->
            // Sort by time and confidence
            val sorted = conceptFacts.sortedWith(
                compareByDescending<TemporalFact> { it.validFrom }
                    .thenByDescending { it.confidence }
            )
            
            // Check for overlaps and resolve
            val nonOverlapping = mutableListOf<TemporalFact>()
            
            sorted.forEach { fact ->
                val overlaps = nonOverlapping.filter { it.overlaps(fact) }
                
                if (overlaps.isEmpty()) {
                    nonOverlapping.add(fact)
                } else {
                    // Resolve overlap based on confidence
                    val highestConfidence = (overlaps + fact).maxByOrNull { it.confidence }
                    if (highestConfidence == fact) {
                        // Remove lower confidence overlapping facts
                        nonOverlapping.removeAll(overlaps)
                        nonOverlapping.add(fact)
                    }
                    // Otherwise, keep existing facts
                }
            }
            
            resolved.addAll(nonOverlapping)
        }
        
        return resolved
    }
    
    private fun determineResolutionStrategy(
        newFact: TemporalFact,
        conflicts: List<TemporalFact>
    ): ResolutionStrategy {
        // If new fact has significantly higher confidence, invalidate old
        val avgOldConfidence = conflicts.map { it.confidence }.average()
        if (newFact.confidence > avgOldConfidence * 1.2) {
            return ResolutionStrategy.INVALIDATE_OLD
        }
        
        // If dealing with multi-valued facts, consider merging
        if (newFact.fact is MultipleFacts || conflicts.any { it.fact is MultipleFacts }) {
            return ResolutionStrategy.MERGE_FACTS
        }
        
        // Otherwise, split temporal ranges
        return ResolutionStrategy.SPLIT_TEMPORAL
    }
    
    private fun mergeFacts(
        newFact: TemporalFact,
        existingFacts: List<TemporalFact>
    ): TemporalFact {
        // Collect all values
        val allValues = mutableSetOf<String>()
        
        val addFactValues = { fact: Fact ->
            when (fact) {
                is SingleFact -> allValues.add(fact.value)
                is MultipleFacts -> allValues.addAll(fact.values)
            }
        }
        
        addFactValues(newFact.fact)
        existingFacts.forEach { addFactValues(it.fact) }
        
        // Create merged fact
        val mergedFact = MultipleFacts(
            concept = newFact.fact.concept,
            values = allValues.toList(),
            timestamp = Clock.System.now().toEpochMilliseconds()
        )
        
        // Use the union of all time ranges
        val minStart = (existingFacts + newFact).minOf { it.validFrom }
        val maxEnd = (existingFacts + newFact).mapNotNull { it.validTo }.maxOrNull()
        
        return TemporalFact(
            fact = mergedFact,
            validFrom = minStart,
            validTo = maxEnd,
            confidence = (existingFacts + newFact).map { it.confidence }.average()
        )
    }
    
    private fun splitTemporalRanges(
        newFact: TemporalFact,
        conflicts: List<TemporalFact>
    ): TemporalFact {
        // For now, just return the new fact
        // In a real implementation, this would create non-overlapping ranges
        return newFact
    }
    
    private fun parseMonth(monthStr: String): Month {
        return when (monthStr.lowercase().take(3)) {
            "jan" -> Month.JANUARY
            "feb" -> Month.FEBRUARY
            "mar" -> Month.MARCH
            "apr" -> Month.APRIL
            "may" -> Month.MAY
            "jun" -> Month.JUNE
            "jul" -> Month.JULY
            "aug" -> Month.AUGUST
            "sep" -> Month.SEPTEMBER
            "oct" -> Month.OCTOBER
            "nov" -> Month.NOVEMBER
            "dec" -> Month.DECEMBER
            else -> throw IllegalArgumentException("Invalid month: $monthStr")
        }
    }
    
    /**
     * Message data class for context
     */
    public data class Message(
        val content: String,
        val timestamp: Instant,
        val role: String = "user"
    )
}