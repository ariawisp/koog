package ai.koog.prompt.harmony

/**
 * Harmony channel types for multi-stream reasoning.
 * Core to the Noesis Runtime architecture.
 */
public enum class Channel {
    /**
     * Analysis channel - internal chain-of-thought reasoning.
     * NEVER shown to users, contains raw reasoning process.
     */
    ANALYSIS,
    
    /**
     * Commentary channel - tool interactions and action plans.
     * Used for function calls and multi-tool execution.
     */
    COMMENTARY,
    
    /**
     * Final channel - user-facing responses.
     * The only channel that should be shown to users.
     */
    FINAL
}