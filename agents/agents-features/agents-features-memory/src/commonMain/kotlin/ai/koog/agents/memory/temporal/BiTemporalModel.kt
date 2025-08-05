package ai.koog.agents.memory.temporal

import kotlinx.datetime.Instant

/**
 * Bi-temporal data model implementation for Koog.
 * 
 * This implements the bi-temporal model from Graphiti, tracking both:
 * - Valid time: When a fact was true in the real world
 * - Transaction time: When a fact was recorded in the system
 * 
 * This is critical for temporal queries and historical reconstruction.
 */

/**
 * Represents a bi-temporal validity period
 */
public data class BiTemporalPeriod(
    val validFrom: Instant,
    val validTo: Instant? = null,
    val transactionTime: Instant,
    val deletedAt: Instant? = null
) {
    /**
     * Check if this period was valid at a specific point in time
     */
    public fun wasValidAt(pointInTime: Instant): Boolean {
        val effectiveValidTo = validTo ?: Instant.DISTANT_FUTURE
        return pointInTime >= validFrom && pointInTime < effectiveValidTo
    }
    
    /**
     * Check if this period was known to the system at a specific transaction time
     */
    public fun wasKnownAt(transactionTime: Instant): Boolean {
        val effectiveDeletionTime = deletedAt ?: Instant.DISTANT_FUTURE
        return transactionTime >= this.transactionTime && transactionTime < effectiveDeletionTime
    }
    
    /**
     * Check if this period is currently active (not deleted)
     */
    public fun isActive(): Boolean = deletedAt == null
    
    /**
     * Create a new period marking this one as ended at the given time
     */
    public fun endAt(endTime: Instant): BiTemporalPeriod {
        return copy(validTo = endTime)
    }
    
    /**
     * Create a new period marking this one as deleted
     */
    public fun deleteAt(deletionTime: Instant): BiTemporalPeriod {
        return copy(deletedAt = deletionTime)
    }
}

/**
 * Extension to make any data bi-temporal
 */
public interface BiTemporal {
    public val temporalPeriod: BiTemporalPeriod
}

/**
 * Bi-temporal node data
 */
public data class BiTemporalNode<T>(
    val data: T,
    override val temporalPeriod: BiTemporalPeriod
) : BiTemporal

/**
 * Bi-temporal edge data
 */
public data class BiTemporalEdge<T>(
    val data: T,
    override val temporalPeriod: BiTemporalPeriod
) : BiTemporal

/**
 * Query for bi-temporal data
 */
public data class BiTemporalQuery(
    val validAt: Instant? = null,
    val asOf: Instant? = null,
    val includeDeleted: Boolean = false
) {
    /**
     * Filter function for bi-temporal data
     */
    public fun <T : BiTemporal> matches(item: T): Boolean {
        // Check if item is active or we want deleted items
        if (!item.temporalPeriod.isActive() && !includeDeleted) {
            return false
        }
        
        // Check valid time constraint
        if (validAt != null && !item.temporalPeriod.wasValidAt(validAt)) {
            return false
        }
        
        // Check transaction time constraint
        if (asOf != null && !item.temporalPeriod.wasKnownAt(asOf)) {
            return false
        }
        
        return true
    }
}

/**
 * Utilities for bi-temporal operations
 */
public object BiTemporalUtils {
    /**
     * Create a new bi-temporal period starting now
     */
    public fun newPeriod(
        validFrom: Instant,
        transactionTime: Instant,
        validTo: Instant? = null
    ): BiTemporalPeriod {
        return BiTemporalPeriod(
            validFrom = validFrom,
            validTo = validTo,
            transactionTime = transactionTime
        )
    }
    
    /**
     * Handle temporal state changes by ending the current period and starting a new one
     */
    public fun handleStateChange(
        currentPeriod: BiTemporalPeriod,
        changeTime: Instant,
        transactionTime: Instant
    ): Pair<BiTemporalPeriod, BiTemporalPeriod> {
        // End current period
        val endedPeriod = currentPeriod.endAt(changeTime)
        
        // Start new period
        val newPeriod = BiTemporalPeriod(
            validFrom = changeTime,
            validTo = null,
            transactionTime = transactionTime
        )
        
        return endedPeriod to newPeriod
    }
    
    /**
     * Merge overlapping temporal periods
     */
    public fun mergePeriods(periods: List<BiTemporalPeriod>): List<BiTemporalPeriod> {
        if (periods.isEmpty()) return emptyList()
        
        val sorted = periods.sortedBy { it.validFrom }
        val merged = mutableListOf<BiTemporalPeriod>()
        
        var current = sorted.first()
        for (i in 1 until sorted.size) {
            val next = sorted[i]
            
            // Check if periods can be merged
            val currentEnd = current.validTo ?: Instant.DISTANT_FUTURE
            if (currentEnd >= next.validFrom) {
                // Merge periods
                val mergedEnd = when {
                    current.validTo == null || next.validTo == null -> null
                    else -> maxOf(current.validTo, next.validTo)
                }
                current = current.copy(
                    validTo = mergedEnd,
                    transactionTime = minOf(current.transactionTime, next.transactionTime)
                )
            } else {
                // Can't merge, add current and move to next
                merged.add(current)
                current = next
            }
        }
        merged.add(current)
        
        return merged
    }
}