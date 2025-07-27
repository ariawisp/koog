package ai.koog.agents.features.distributed.strategies

import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.features.distributed.message.DistributedAction
import ai.koog.agents.features.distributed.message.DistributedAgentId
import ai.koog.agents.features.distributed.message.DistributedContext
import ai.koog.agents.features.distributed.message.DistributedMessage

/**
 * Strategy interface for coordinating distributed agents.
 * 
 * DistributedStrategy defines how an agent should behave in a distributed
 * multi-agent system. It determines when to execute tasks locally versus
 * delegating them to other agents, and how to handle coordination messages.
 * 
 * Unlike Google ADK's pseudo "multi-agent" pattern (which is really just
 * function orchestration), this enables true distributed coordination where
 * agents are autonomous processes that make decisions about collaboration.
 * 
 * Example implementations:
 * - **PlannerExecutorStrategy**: Separates planning from execution
 * - **CoordinatorStrategy**: Central coordination with multiple specialists
 * - **PeerToPeerStrategy**: Decentralized agent collaboration
 * - **HierarchicalStrategy**: Multi-level coordination chains
 */
public interface DistributedStrategy {
    
    /**
     * The name of this coordination strategy.
     * Used for debugging, logging, and strategy selection.
     */
    public val name: String
    
    /**
     * Handles local input to determine if it should be processed locally or distributed.
     * 
     * This method is called whenever the agent receives direct input (e.g., from a user
     * or external system). The strategy decides whether to:
     * - Process the input locally using the agent's normal strategy
     * - Delegate the task to other agents via role-based or direct routing
     * - Broadcast the task to multiple agents
     * - Ignore the input entirely
     * 
     * @param input The input received by the agent
     * @param context The current agent context with access to tools, memory, etc.
     * @param distributedContext Information about available agents and system state
     * @return The action to take, or null if no action should be taken
     * 
     * Example:
     * ```kotlin
     * override suspend fun onLocalInput(
     *     input: String, 
     *     context: AIAgentContext,
     *     distributedContext: DistributedContext
     * ): DistributedAction? {
     *     return when (distributedContext.currentAgentId.role) {
     *         AgentRole.PLANNER -> {
     *             // Generate plan locally, then send to executor
     *             val plan = generatePlan(input, context)
     *             DistributedAction.SendToRole(
     *                 role = AgentRole.EXECUTOR,
     *                 payload = TaskRequest(plan).toJson(),
     *                 messageType = MessageType.TASK_REQUEST
     *             )
     *         }
     *         AgentRole.EXECUTOR -> {
     *             // Execute directly
     *             DistributedAction.ExecuteLocally(input)
     *         }
     *         else -> DistributedAction.Ignore
     *     }
     * }
     * ```
     */
    public suspend fun onLocalInput(
        input: String,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction?
    
    /**
     * Handles messages received from other agents in the distributed system.
     * 
     * This method is called whenever the agent receives a message via PubSub
     * from another agent. The strategy decides how to handle different types
     * of coordination messages and can:
     * - Execute the requested task locally
     * - Forward the message to other agents
     * - Send responses back to the originating agent
     * - Update local state and continue coordination
     * 
     * @param message The message received from another agent
     * @param context The current agent context with access to tools, memory, etc.
     * @param distributedContext Information about available agents and system state
     * @return The action to take, or null if no action should be taken
     * 
     * Example:
     * ```kotlin
     * override suspend fun onRemoteMessage(
     *     message: DistributedMessage,
     *     context: AIAgentContext,
     *     distributedContext: DistributedContext
     * ): DistributedAction? {
     *     return when (message.messageType) {
     *         MessageType.TASK_REQUEST -> {
     *             if (canHandleTask(message.payload, distributedContext.currentAgentId)) {
     *                 DistributedAction.ExecuteLocally(message.payload)
     *             } else {
     *                 // Forward to a more suitable agent
     *                 val suitableAgent = findBestAgent(message.payload, distributedContext)
     *                 DistributedAction.DelegateTo(suitableAgent, message.payload)
     *             }
     *         }
     *         MessageType.TASK_RESPONSE -> {
     *             // Process the response and maybe send next task
     *             handleTaskResponse(message.payload, context)
     *             DistributedAction.Ignore
     *         }
     *         else -> DistributedAction.Ignore
     *     }
     * }
     * ```
     */
    public suspend fun onRemoteMessage(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction?
    
    /**
     * Called when a new agent joins the distributed system.
     * 
     * This allows strategies to react to changes in the agent topology,
     * such as updating routing tables, rebalancing workloads, or establishing
     * direct communication channels.
     * 
     * @param agentId The ID of the agent that joined
     * @param context The current agent context
     * @param distributedContext Current distributed system state
     * @return Optional action to take in response to the agent joining
     */
    public suspend fun onAgentJoined(
        agentId: DistributedAgentId,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? = null
    
    /**
     * Called when an agent leaves the distributed system.
     * 
     * This allows strategies to handle agent failures or shutdowns,
     * such as reassigning tasks, updating routing, or triggering
     * failover procedures.
     * 
     * @param agentId The ID of the agent that left
     * @param context The current agent context
     * @param distributedContext Current distributed system state
     * @return Optional action to take in response to the agent leaving
     */
    public suspend fun onAgentLeft(
        agentId: DistributedAgentId,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? = null
    
    /**
     * Called when a distributed action fails to execute.
     * 
     * This allows strategies to implement error handling, retry logic,
     * or alternative coordination approaches when actions fail.
     * 
     * @param action The action that failed
     * @param error The error that occurred
     * @param context The current agent context
     * @param distributedContext Current distributed system state
     * @return Optional recovery action to take
     */
    public suspend fun onActionFailed(
        action: DistributedAction,
        error: Throwable,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? = null
    
    /**
     * Called periodically to allow strategies to perform maintenance tasks.
     * 
     * This can be used for:
     * - Sending heartbeat or status update messages
     * - Performing health checks on connected agents
     * - Cleaning up expired or stale coordination state
     * - Implementing time-based coordination patterns
     * 
     * @param context The current agent context
     * @param distributedContext Current distributed system state
     * @return Optional maintenance action to take
     */
    public suspend fun onPeriodicMaintenance(
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? = null
    
    /**
     * Validates whether this strategy can be used with the given agent configuration.
     * 
     * This allows strategies to enforce requirements such as:
     * - Required agent roles or capabilities
     * - Minimum number of agents in the system
     * - Specific PubSub provider requirements
     * - Compatibility with other installed features
     * 
     * @param agentId The agent ID this strategy will be used with
     * @param context The agent context for validation
     * @return List of validation error messages, empty if valid
     */
    public fun validateConfiguration(
        agentId: DistributedAgentId,
        context: AIAgentContextBase
    ): List<String> = emptyList()
}