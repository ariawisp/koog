package ai.koog.agents.features.distributed.message

import kotlinx.datetime.Instant
import kotlinx.serialization.Serializable
import kotlin.time.Duration

/**
 * Actions that can be taken by distributed coordination strategies.
 * 
 * DistributedAction represents the decisions made by coordination strategies
 * about how to handle input or messages. These actions are then executed
 * by the Distributed feature.
 */
@Serializable
public sealed interface DistributedAction {
    
    /**
     * Send a message to agents with a specific role.
     * 
     * This action enables role-based routing where any available agent
     * with the specified role can handle the message.
     * 
     * @property role The target agent role
     * @property payload The message content to send
     * @property messageType The type of message being sent
     * @property responseTopicSuffix Optional suffix for response topic routing
     * @property priority Message priority (higher = more urgent)
     * @property expiresAt Optional expiration time for the message
     */
    @Serializable
    public data class SendToRole(
        public val role: AgentRole,
        public val payload: String,
        public val messageType: MessageType = MessageType.TASK_REQUEST,
        public val responseTopicSuffix: String? = null,
        public val priority: Int = 0,
        public val expiresAt: Instant? = null
    ) : DistributedAction
    
    /**
     * Execute input locally using the current agent's strategy.
     * 
     * This action causes the agent to process the input through its
     * normal strategy execution pipeline instead of forwarding it.
     * 
     * @property input The input to process locally
     * @property metadata Additional metadata for local processing
     */
    @Serializable
    public data class ExecuteLocally(
        public val input: String,
        public val metadata: Map<String, String> = emptyMap()
    ) : DistributedAction
    
    /**
     * Publish a message to a specific PubSub topic.
     * 
     * This action provides direct access to PubSub publishing for
     * custom routing and broadcast scenarios.
     * 
     * @property topic The PubSub topic to publish to
     * @property payload The message content to publish
     * @property attributes Additional message attributes/headers
     */
    @Serializable
    public data class Publish(
        public val topic: String,
        public val payload: String,
        public val attributes: Map<String, String> = emptyMap()
    ) : DistributedAction
    
    /**
     * Delegate a task to a specific agent instance.
     * 
     * This action enables direct agent-to-agent communication for
     * scenarios requiring precise routing or stateful coordination.
     * 
     * @property agentId The target agent instance
     * @property payload The message content to send
     * @property messageType The type of message being sent
     * @property priority Message priority (higher = more urgent)
     * @property expiresAt Optional expiration time for the message
     */
    @Serializable
    public data class DelegateTo(
        public val agentId: DistributedAgentId,
        public val payload: String,
        public val messageType: MessageType = MessageType.TASK_REQUEST,
        public val priority: Int = 0,
        public val expiresAt: Instant? = null
    ) : DistributedAction
    
    /**
     * Wait for a response message with a specific correlation ID.
     * 
     * This action implements request-response patterns by suspending
     * execution until a matching response is received.
     * 
     * @property correlationId The correlation ID to wait for
     * @property timeout Maximum time to wait for the response
     * @property responseFilter Optional filter for response validation
     */
    @Serializable
    public data class WaitForResponse(
        public val correlationId: String,
        public val timeout: Duration,
        public val responseFilter: String? = null
    ) : DistributedAction
    
    /**
     * Broadcast a message to all agents with specific capabilities.
     * 
     * This action enables capability-based routing where messages
     * are sent to all agents that have the required capabilities.
     * 
     * @property requiredCapabilities The capabilities agents must have
     * @property payload The message content to broadcast
     * @property messageType The type of message being sent
     * @property priority Message priority (higher = more urgent)
     * @property maxRecipients Maximum number of recipients (for load balancing)
     */
    @Serializable
    public data class BroadcastToCapabilities(
        public val requiredCapabilities: Set<String>,
        public val payload: String,
        public val messageType: MessageType = MessageType.TASK_REQUEST,
        public val priority: Int = 0,
        public val maxRecipients: Int? = null
    ) : DistributedAction
    
    /**
     * Schedule a delayed action to be executed later.
     * 
     * This action enables time-based coordination patterns like
     * retries, periodic tasks, and delayed execution.
     * 
     * @property delay How long to wait before executing the action
     * @property action The action to execute after the delay
     * @property cancelKey Optional key for canceling the scheduled action
     */
    @Serializable
    public data class ScheduleDelayed(
        public val delay: Duration,
        public val action: DistributedAction,
        public val cancelKey: String? = null
    ) : DistributedAction
    
    /**
     * Forward a message to another coordination strategy.
     * 
     * This action enables strategy composition and delegation where
     * one strategy can hand off decisions to another strategy.
     * 
     * @property strategyName The name of the target strategy
     * @property input The input to forward
     * @property context Additional context for the target strategy
     */
    @Serializable
    public data class ForwardToStrategy(
        public val strategyName: String,
        public val input: String,
        public val context: Map<String, String> = emptyMap()
    ) : DistributedAction
    
    /**
     * Update the agent's status and broadcast it to observers.
     * 
     * This action enables status coordination and monitoring by
     * updating local state and notifying other agents.
     * 
     * @property status The new agent status
     * @property broadcastToObservers Whether to notify observer agents
     * @property metadata Additional status metadata
     */
    @Serializable
    public data class UpdateStatus(
        public val status: AgentStatus,
        public val broadcastToObservers: Boolean = true,
        public val metadata: Map<String, String> = emptyMap()
    ) : DistributedAction
    
    /**
     * Do nothing - ignore the input or message.
     * 
     * This action indicates that the coordination strategy
     * has determined no action should be taken.
     */
    @Serializable
    public data object Ignore : DistributedAction
}

/**
 * Result of executing a distributed action.
 * 
 * @property success Whether the action executed successfully
 * @property result Optional result data from the action
 * @property error Error message if the action failed
 * @property metadata Additional result metadata
 */
@Serializable
public data class DistributedActionResult(
    public val success: Boolean,
    public val result: String? = null,
    public val error: String? = null,
    public val metadata: Map<String, String> = emptyMap()
)

/**
 * Context information available when deciding on distributed actions.
 * 
 * @property currentAgentId The ID of the current agent
 * @property availableAgents Known agents in the distributed system
 * @property systemMetrics Current system performance metrics
 * @property timestamp When this context was created
 */
@Serializable
public data class DistributedContext(
    public val currentAgentId: DistributedAgentId,
    public val availableAgents: List<DistributedAgentId> = emptyList(),
    public val systemMetrics: Map<String, String> = emptyMap(),
    public val timestamp: Instant
) {
    
    /**
     * Finds agents with a specific role.
     */
    public fun findAgentsByRole(role: AgentRole): List<DistributedAgentId> = 
        availableAgents.filter { it.role == role }
    
    /**
     * Finds agents with specific capabilities.
     */
    public fun findAgentsByCapability(capability: String): List<DistributedAgentId> = 
        availableAgents.filter { it.hasCapability(capability) }
    
    /**
     * Finds agents with all required capabilities.
     */
    public fun findAgentsByCapabilities(capabilities: Set<String>): List<DistributedAgentId> = 
        availableAgents.filter { it.hasAllCapabilities(capabilities) }
    
    /**
     * Checks if any agents with a specific role are available.
     */
    public fun hasAvailableRole(role: AgentRole): Boolean = 
        availableAgents.any { it.role == role }
    
    /**
     * Checks if any agents with a specific capability are available.
     */
    public fun hasAvailableCapability(capability: String): Boolean = 
        availableAgents.any { it.hasCapability(capability) }
    
    /**
     * Gets a system metric value.
     */
    public fun getSystemMetric(key: String): String? = systemMetrics[key]
}