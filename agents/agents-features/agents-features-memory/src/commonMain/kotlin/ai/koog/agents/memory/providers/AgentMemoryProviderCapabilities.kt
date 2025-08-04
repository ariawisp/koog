package ai.koog.agents.memory.providers

import kotlinx.serialization.Serializable

/**
 * Simple capability flags for documentation purposes.
 * Note: These are descriptive only - providers should implement what they support
 * and handle unsupported operations gracefully with exceptions or fallbacks.
 */
@Serializable
public data class AgentMemoryProviderCapabilities(
    /**
     * Supports advanced semantic search beyond basic text matching
     */
    val advancedSearch: Boolean = false,
    
    /**
     * Supports relationship traversal and entity-centric queries
     */
    val relationshipQueries: Boolean = false,
    
    /**
     * Supports temporal/time-based queries  
     */
    val temporalQueries: Boolean = false
)

// Note: CapableMemoryProvider interface removed - was over-engineered.
// Memory providers should just implement their methods and handle 
// unsupported operations with clear exceptions or fallback behavior.