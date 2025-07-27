package ai.koog.agents.features.distributed.message

import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Types of messages that can be sent between distributed agents.
 * 
 * Message types enable proper routing and handling of different kinds of coordination:
 * - **TASK_REQUEST**: Request for another agent to perform a specific task
 * - **TASK_RESPONSE**: Response containing the result of a completed task
 * - **EXECUTION_REQUEST**: Request to execute a specific action or command
 * - **EXECUTION_RESULT**: Result of an executed action
 * - **STATUS_UPDATE**: Status information about agent state or progress
 * - **AGENT_DISCOVERY**: Agent presence and capability announcements
 * - **COORDINATION**: Multi-agent coordination and synchronization messages
 */
@Serializable
public enum class MessageType {
    /**
     * Request for another agent to perform a specific task.
     * Used by planners to assign work to executors.
     */
    TASK_REQUEST,
    
    /**
     * Response containing the result of a completed task.
     * Sent by executors back to planners upon task completion.
     */
    TASK_RESPONSE,
    
    /**
     * Request to execute a specific action or command.
     * More direct than TASK_REQUEST, typically for immediate actions.
     */
    EXECUTION_REQUEST,
    
    /**
     * Result of an executed action.
     * Contains the outcome of EXECUTION_REQUEST messages.
     */
    EXECUTION_RESULT,
    
    /**
     * Status information about agent state or progress.
     * Used for monitoring and coordination purposes.
     */
    STATUS_UPDATE,
    
    /**
     * Agent presence and capability announcements.
     * Used for agent discovery and service registration.
     */
    AGENT_DISCOVERY,
    
    /**
     * Multi-agent coordination and synchronization messages.
     * Used for complex coordination patterns like consensus, barriers, etc.
     */
    COORDINATION
}

/**
 * Standard message format for communication between distributed agents.
 * 
 * This class provides a type-safe, serializable message format that supports
 * both role-based routing and direct agent-to-agent communication.
 * 
 * @property id Unique message identifier
 * @property fromAgent The agent that sent this message
 * @property toRole Target agent role (for role-based routing)
 * @property toAgentId Target specific agent (for direct routing)
 * @property correlationId Optional correlation ID for request-response patterns
 * @property messageType The type of message being sent
 * @property payload The message content (typically JSON-serialized data)
 * @property timestamp When the message was created
 * @property priority Message priority for routing and processing
 * @property expiresAt Optional expiration time for time-sensitive messages
 * 
 * Example usage:
 * ```kotlin
 * // Task assignment from planner to executor
 * val taskMessage = DistributedMessage(
 *     fromAgent = plannerId,
 *     toRole = AgentRole.EXECUTOR,
 *     messageType = MessageType.TASK_REQUEST,
 *     payload = TaskRequest(
 *         task = "Build a house",
 *         requirements = listOf("wood", "stone"),
 *         deadline = Clock.System.now().plus(1.hours)
 *     ).toJson()
 * )
 * 
 * // Direct response to specific agent
 * val responseMessage = DistributedMessage(
 *     fromAgent = executorId,
 *     toAgentId = plannerId,
 *     messageType = MessageType.TASK_RESPONSE,
 *     correlationId = taskMessage.id,
 *     payload = TaskResult(
 *         success = true,
 *         result = "House built successfully at coordinates (100, 64, 200)"
 *     ).toJson()
 * )
 * ```
 */
@Serializable
@OptIn(ExperimentalUuidApi::class)
public data class DistributedMessage(
    /**
     * Unique message identifier.
     * Automatically generated if not provided.
     */
    public val id: String = Uuid.random().toString(),
    
    /**
     * The agent that sent this message.
     */
    public val fromAgent: DistributedAgentId,
    
    /**
     * Target agent role for role-based routing.
     * When specified, the message will be delivered to any available agent with this role.
     */
    public val toRole: AgentRole? = null,
    
    /**
     * Target specific agent for direct routing.
     * When specified, the message will be delivered only to this specific agent.
     */
    public val toAgentId: DistributedAgentId? = null,
    
    /**
     * Optional correlation ID for request-response patterns.
     * Should match the ID of the original request when sending responses.
     */
    public val correlationId: String? = null,
    
    /**
     * The type of message being sent.
     */
    public val messageType: MessageType,
    
    /**
     * The message content, typically JSON-serialized data.
     */
    public val payload: String,
    
    /**
     * When the message was created.
     */
    public val timestamp: Instant = Clock.System.now(),
    
    /**
     * Message priority for routing and processing.
     * Higher values indicate higher priority.
     */
    public val priority: Int = 0,
    
    /**
     * Optional expiration time for time-sensitive messages.
     * Messages past their expiration time should be discarded.
     */
    public val expiresAt: Instant? = null,
    
    /**
     * Additional message metadata for custom routing and processing.
     */
    public val metadata: Map<String, String> = emptyMap()
) {
    
    /**
     * Checks if this message has expired.
     */
    public fun isExpired(): Boolean = expiresAt?.let { Clock.System.now() > it } ?: false
    
    /**
     * Checks if this message is addressed to a specific agent role.
     */
    public fun isRoleTargeted(): Boolean = toRole != null
    
    /**
     * Checks if this message is addressed to a specific agent instance.
     */
    public fun isDirectTargeted(): Boolean = toAgentId != null
    
    /**
     * Checks if this message is a response to another message.
     */
    public fun isResponse(): Boolean = correlationId != null
    
    /**
     * Gets a metadata value by key.
     */
    public fun getMetadata(key: String): String? = metadata[key]
    
    /**
     * Creates a response message to this message.
     */
    public fun createResponse(
        fromAgent: DistributedAgentId,
        messageType: MessageType,
        payload: String,
        priority: Int = this.priority
    ): DistributedMessage = DistributedMessage(
        fromAgent = fromAgent,
        toAgentId = this.fromAgent,
        correlationId = this.id,
        messageType = messageType,
        payload = payload,
        priority = priority
    )
    
    /**
     * Creates a copy of this message with additional metadata.
     */
    public fun withMetadata(additionalMetadata: Map<String, String>): DistributedMessage = 
        copy(metadata = metadata + additionalMetadata)
    
    /**
     * Returns a human-readable string representation of this message.
     */
    public fun toDisplayString(): String {
        val target = when {
            toAgentId != null -> "to ${toAgentId.toDisplayString()}"
            toRole != null -> "to role ${toRole.name.lowercase()}"
            else -> "broadcast"
        }
        
        val correlation = correlationId?.let { " (corr: ${it.take(8)})" } ?: ""
        
        return "${messageType.name} from ${fromAgent.toDisplayString()} $target$correlation"
    }
    
    override fun toString(): String = toDisplayString()
}

/**
 * Standard payload for task request messages.
 * 
 * @property task Human-readable description of the task
 * @property parameters Structured parameters for the task
 * @property requirements Capabilities or resources required
 * @property deadline Optional deadline for task completion
 * @property priority Task priority (higher = more urgent)
 */
@Serializable
public data class TaskRequest(
    public val task: String,
    public val parameters: Map<String, String> = emptyMap(),
    public val requirements: Set<String> = emptySet(),
    public val deadline: Instant? = null,
    public val priority: Int = 0
)

/**
 * Standard payload for task response messages.
 * 
 * @property success Whether the task completed successfully
 * @property result Result data or description
 * @property error Error message if the task failed
 * @property metrics Optional performance metrics
 */
@Serializable
public data class TaskResponse(
    public val success: Boolean,
    public val result: String? = null,
    public val error: String? = null,
    public val metrics: Map<String, String> = emptyMap()
)

/**
 * Standard payload for agent status updates.
 * 
 * @property status Current agent status (e.g., "idle", "busy", "error")
 * @property currentTask Description of current task if any
 * @property progress Progress percentage (0-100)
 * @property capabilities Current agent capabilities
 * @property metrics Performance and health metrics
 */
@Serializable
public data class AgentStatus(
    public val status: String,
    public val currentTask: String? = null,
    public val progress: Int? = null,
    public val capabilities: Set<String> = emptySet(),
    public val metrics: Map<String, String> = emptyMap()
)