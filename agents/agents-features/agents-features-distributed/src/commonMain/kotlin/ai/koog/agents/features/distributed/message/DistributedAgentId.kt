package ai.koog.agents.features.distributed.message

import kotlinx.serialization.Serializable
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Defines the roles that distributed agents can play in a multi-agent system.
 * 
 * Roles determine how agents coordinate and what types of tasks they handle:
 * - **PLANNER**: High-level strategy and task decomposition
 * - **EXECUTOR**: Action execution and tool usage
 * - **OBSERVER**: Monitoring, reporting, and analysis
 * - **COORDINATOR**: Multi-agent orchestration and resource management
 */
@Serializable
public enum class AgentRole {
    /**
     * Planning agents focus on high-level strategy, goal decomposition, and task assignment.
     * They typically:
     * - Break down complex goals into executable tasks
     * - Decide which agents should handle specific tasks
     * - Coordinate overall system behavior
     * - Make strategic decisions based on system state
     */
    PLANNER,
    
    /**
     * Executor agents focus on action execution and direct tool usage.
     * They typically:
     * - Execute specific tasks assigned by planners
     * - Use tools to interact with external systems
     * - Report execution results and status
     * - Handle low-level operational details
     */
    EXECUTOR,
    
    /**
     * Observer agents focus on monitoring, analysis, and reporting.
     * They typically:
     * - Monitor system performance and agent behavior
     * - Collect and analyze metrics
     * - Generate reports and alerts
     * - Provide insights for system optimization
     */
    OBSERVER,
    
    /**
     * Coordinator agents manage multi-agent orchestration and resources.
     * They typically:
     * - Manage agent lifecycles and availability
     * - Handle resource allocation and load balancing
     * - Coordinate between multiple planners and executors
     * - Implement system-wide policies and constraints
     */
    COORDINATOR
}

/**
 * Unique identifier for a distributed agent instance.
 * 
 * This identifier combines role-based typing with instance-specific identification,
 * enabling both role-based routing and direct agent-to-agent communication.
 * 
 * @property role The functional role this agent plays in the distributed system
 * @property instanceId Unique identifier for this specific agent instance
 * @property capabilities Set of capabilities this agent can provide (e.g., "minecraft", "web-scraping")
 * @property metadata Additional agent-specific metadata for discovery and routing
 * 
 * Example usage:
 * ```kotlin
 * val plannerId = DistributedAgentId(
 *     role = AgentRole.PLANNER,
 *     capabilities = setOf("task-planning", "resource-optimization"),
 *     metadata = mapOf("version" to "1.2.0", "environment" to "production")
 * )
 * 
 * val executorId = DistributedAgentId(
 *     role = AgentRole.EXECUTOR,
 *     capabilities = setOf("minecraft", "building", "resource-gathering"),
 *     metadata = mapOf("server" to "minecraft-1", "world" to "survival")
 * )
 * ```
 */
@Serializable
@OptIn(ExperimentalUuidApi::class)
public data class DistributedAgentId(
    /**
     * The functional role this agent plays in the distributed system.
     */
    public val role: AgentRole,
    
    /**
     * Unique identifier for this specific agent instance.
     * 
     * Automatically generated if not provided. Each agent instance should have
     * a unique ID even if multiple agents have the same role and capabilities.
     */
    public val instanceId: String = Uuid.random().toString(),
    
    /**
     * Set of capabilities this agent can provide.
     * 
     * Capabilities are used for capability-based routing and agent discovery.
     * Examples: "minecraft", "web-scraping", "image-generation", "data-analysis"
     */
    public val capabilities: Set<String> = emptySet(),
    
    /**
     * Additional agent-specific metadata for discovery and routing.
     * 
     * Can include information like version, environment, location, configuration, etc.
     * This metadata can be used by coordination strategies for advanced routing decisions.
     */
    public val metadata: Map<String, String> = emptyMap()
) {
    
    /**
     * Checks if this agent has the specified capability.
     */
    public fun hasCapability(capability: String): Boolean = capability in capabilities
    
    /**
     * Checks if this agent has all of the specified capabilities.
     */
    public fun hasAllCapabilities(requiredCapabilities: Set<String>): Boolean = 
        capabilities.containsAll(requiredCapabilities)
    
    /**
     * Checks if this agent has any of the specified capabilities.
     */
    public fun hasAnyCapability(requiredCapabilities: Set<String>): Boolean = 
        capabilities.intersect(requiredCapabilities).isNotEmpty()
    
    /**
     * Gets a metadata value by key.
     */
    public fun getMetadata(key: String): String? = metadata[key]
    
    /**
     * Creates a copy of this agent ID with additional capabilities.
     */
    public fun withCapabilities(additionalCapabilities: Set<String>): DistributedAgentId = 
        copy(capabilities = capabilities + additionalCapabilities)
    
    /**
     * Creates a copy of this agent ID with additional metadata.
     */
    public fun withMetadata(additionalMetadata: Map<String, String>): DistributedAgentId = 
        copy(metadata = metadata + additionalMetadata)
    
    /**
     * Returns a human-readable string representation of this agent ID.
     */
    public fun toDisplayString(): String {
        val capabilitiesStr = if (capabilities.isNotEmpty()) {
            " [${capabilities.joinToString(", ")}]"
        } else ""
        
        return "${role.name.lowercase()}-${instanceId.take(8)}$capabilitiesStr"
    }
    
    override fun toString(): String = toDisplayString()
}