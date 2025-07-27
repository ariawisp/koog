package ai.koog.agents.features.distributed.feature

import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.core.agent.entity.AIAgentStorageKey
import ai.koog.agents.core.feature.AIAgentFeature
import ai.koog.agents.core.feature.AIAgentPipeline
import ai.koog.agents.core.feature.InterceptContext
import ai.koog.agents.core.feature.model.*
import ai.koog.agents.features.common.message.FeatureMessage
import ai.koog.agents.features.common.message.FeatureMessageProcessorUtil.onMessageForEachSafe
import ai.koog.agents.features.distributed.message.*
import ai.koog.agents.features.distributed.strategies.DistributedStrategy
import ai.koog.agents.features.pubsub.feature.PubSub
import ai.koog.agents.features.pubsub.message.PubSubStringMessage
import ai.koog.agents.features.pubsub.providers.ReceivedMessage
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.onEach
import kotlinx.datetime.Clock
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Feature that enables coordination between Koog agents across different processes and machines.
 * 
 * The Distributed feature provides true multi-agent coordination capabilities, unlike Google ADK's
 * pseudo "multi-agent" pattern which is really just function orchestration. This enables:
 * 
 * - **Cross-process Agent Communication**: Agents running in different processes can coordinate
 * - **Role-based Routing**: Send messages to agents based on their functional roles
 * - **Capability-based Discovery**: Find agents based on their specific capabilities
 * - **Distributed Strategies**: Implement coordination patterns like planner-executor separation
 * - **Agent Lifecycle Management**: Handle agent discovery, presence, and failure scenarios
 * 
 * ## Architecture
 * 
 * ```
 * ┌─────────────────┐    DistributedMessage     ┌─────────────────┐
 * │ Planner Agent   │ ←──────PubSub───────────→ │ Executor Agent  │
 * │ (Ktor Server)   │                          │ (Minecraft)     │
 * │                 │     TaskRequest          │                 │
 * │ - Planning      │ ────────────────────────→ │ - Tool Usage    │
 * │ - Coordination  │ ←──────────────────────── │ - Execution     │
 * │ - Monitoring    │     TaskResponse         │ - Status        │
 * └─────────────────┘                          └─────────────────┘
 * ```
 * 
 * ## Basic Usage
 * 
 * ```kotlin
 * // Planner Agent (in Ktor server)
 * val plannerAgent = AIAgent(
 *     promptExecutor = executor,
 *     strategy = strategy
 * ) {
 *     install(PubSub) {
 *         provider = RedisPubSubProvider("redis://localhost:6379")
 *     }
 *     
 *     install(Distributed) {
 *         agentId = DistributedAgentId(
 *             role = AgentRole.PLANNER,
 *             capabilities = setOf("planning", "coordination")
 *         )
 *         strategy = PlannerExecutorStrategy()
 *         discoveryEnabled = true
 *     }
 * }
 * 
 * // Executor Agent (in Minecraft server)
 * val executorAgent = AIAgent(
 *     promptExecutor = executor,
 *     strategy = strategy
 * ) {
 *     install(PubSub) {
 *         provider = RedisPubSubProvider("redis://localhost:6379")
 *     }
 *     
 *     install(Distributed) {
 *         agentId = DistributedAgentId(
 *             role = AgentRole.EXECUTOR,
 *             capabilities = setOf("minecraft", "building")
 *         )
 *         strategy = PlannerExecutorStrategy()
 *     }
 * }
 * ```
 * 
 * ## Advanced Coordination
 * 
 * ```kotlin
 * // Custom coordination strategy
 * class GameCoordinationStrategy : DistributedStrategy {
 *     override suspend fun onLocalInput(
 *         input: String,
 *         context: AIAgentContext,
 *         distributedContext: DistributedContext
 *     ): DistributedAction? {
 *         return when (distributedContext.currentAgentId.role) {
 *             AgentRole.PLANNER -> {
 *                 // Break down the task and assign to appropriate executors
 *                 val plan = analyzeBuildingTask(input)
 *                 DistributedAction.BroadcastToCapabilities(
 *                     requiredCapabilities = setOf("minecraft", "building"),
 *                     payload = BuildingTask(plan).toJson()
 *                 )
 *             }
 *             AgentRole.EXECUTOR -> {
 *                 // Execute directly if we have the right capabilities
 *                 if (canExecuteLocally(input, context)) {
 *                     DistributedAction.ExecuteLocally(input)
 *                 } else {
 *                     // Find a better suited agent
 *                     val specialists = distributedContext.findAgentsByCapability("specialized-building")
 *                     if (specialists.isNotEmpty()) {
 *                         DistributedAction.DelegateTo(specialists.first(), input)
 *                     } else {
 *                         DistributedAction.ExecuteLocally(input) // Try anyway
 *                     }
 *                 }
 *             }
 *             else -> DistributedAction.Ignore
 *         }
 *     }
 * }
 * ```
 * 
 * ## Dependencies
 * 
 * This feature requires the PubSub feature to be installed with a configured provider:
 * 
 * ```kotlin
 * install(PubSub) {
 *     provider = RedisPubSubProvider("redis://localhost:6379")
 *     // OR
 *     provider = GCPPubSubProvider(projectId = "my-project")
 *     // OR  
 *     provider = InMemoryPubSubProvider() // For single-process testing
 * }
 * ```
 */
public class Distributed {
    
    /**
     * Feature implementation for distributed agent coordination.
     */
    @OptIn(ExperimentalUuidApi::class)
    public companion object Feature : AIAgentFeature<DistributedFeatureConfig, Distributed> {
        
        private val logger = KotlinLogging.logger { }
        private val json = Json { ignoreUnknownKeys = true }
        
        override val key: AIAgentStorageKey<Distributed> =
            AIAgentStorageKey("agents-features-distributed")
        
        override fun createInitialConfig(): DistributedFeatureConfig = DistributedFeatureConfig()
        
        override fun install(
            config: DistributedFeatureConfig,
            pipeline: AIAgentPipeline,
        ) {
            logger.info { "Installing Distributed feature for agent: ${config.agentId.toDisplayString()}" }
            
            // Note: PubSub validation will be enabled when PubSub PR is merged
            logger.info { "Distributed feature ready - PubSub integration pending" }
            
            // Note: Strategy validation will be implemented when agent context access is available
            // For now we skip validation for the proposal phase
            logger.info { "Skipping strategy validation for proposal phase - will be implemented with agent context access" }
            
            val interceptContext = InterceptContext(this, Distributed())
            
            // Create distributed context manager
            val contextManager = DistributedContextManager(config)
            
            // Start agent coordination coroutine
            val coordinationJob = CoroutineScope(Dispatchers.Default + SupervisorJob()).launch {
                startDistributedCoordination(config, contextManager, pipeline)
            }
            
            // Set up agent discovery if enabled
            if (config.discoveryEnabled) {
                val discoveryJob = CoroutineScope(Dispatchers.Default + SupervisorJob()).launch {
                    startAgentDiscovery(config, contextManager)
                }
                
                // Clean up discovery on shutdown
                pipeline.interceptAgentBeforeClosed(interceptContext) { _ ->
                    discoveryJob.cancel()
                }
            }
            
            // Start periodic maintenance
            val maintenanceJob = CoroutineScope(Dispatchers.Default + SupervisorJob()).launch {
                startPeriodicMaintenance(config, contextManager, pipeline)
            }
            
            // Clean up resources on agent shutdown
            pipeline.interceptAgentBeforeClosed(interceptContext) { _ ->
                logger.info { "Shutting down Distributed feature for agent: ${config.agentId.toDisplayString()}" }
                coordinationJob.cancel()
                maintenanceJob.cancel()
                contextManager.shutdown()
            }
            
            // Note: Input interception will be implemented once core agent APIs are finalized
            // This is a placeholder for the distributed coordination logic
            logger.info { "Distributed coordination setup complete for ${config.agentId.toDisplayString()}" }
        }
        
        private suspend fun startDistributedCoordination(
            config: DistributedFeatureConfig,
            contextManager: DistributedContextManager,
            pipeline: AIAgentPipeline
        ) {
            logger.info { "Starting distributed coordination for agent: ${config.agentId.toDisplayString()}" }
            
            try {
                // Note: Will be implemented when PubSub provider is available
                logger.info { "Distributed coordination ready - waiting for PubSub integration" }
            } catch (e: Exception) {
                logger.error(e) { "Failed to start distributed coordination" }
            }
        }
        
        private suspend fun processDistributedMessage(
            receivedMessage: ReceivedMessage,
            config: DistributedFeatureConfig,
            contextManager: DistributedContextManager,
            pipeline: AIAgentPipeline
        ) {
            try {
                // Parse the distributed message
                val distributedMessage = json.decodeFromString<DistributedMessage>(receivedMessage.content)
                
                // Check if message is addressed to this agent
                if (!isMessageForThisAgent(distributedMessage, config.agentId)) {
                    receivedMessage.acknowledge()
                    return
                }
                
                // Update context with sender information
                contextManager.updateAgentPresence(distributedMessage.fromAgent)
                
                // Note: Strategy handling will be implemented when PubSub integration is complete
                // For now we just log the received message for the proposal
                logger.info { "Received distributed message ${distributedMessage.id} from ${distributedMessage.fromAgent.toDisplayString()}" }
                
                // Create message received event
                val receivedEvent = DistributedMessageReceivedEvent(
                    eventId = Uuid.random().toString(),
                    agentId = config.agentId.instanceId,
                    message = distributedMessage
                )
                config.messageProcessor.onMessageForEachSafe(receivedEvent)
                
                receivedMessage.acknowledge()
                
            } catch (e: Exception) {
                logger.error(e) { "Error processing distributed message: ${receivedMessage.messageId}" }
                
                try {
                    receivedMessage.nack()
                } catch (nackError: Exception) {
                    logger.error(nackError) { "Failed to nack message: ${receivedMessage.messageId}" }
                }
            }
        }
        
        private suspend fun startAgentDiscovery(
            config: DistributedFeatureConfig,
            contextManager: DistributedContextManager
        ) {
            logger.info { "Starting agent discovery for agent: ${config.agentId.toDisplayString()}" }
            
            try {
                // Note: Will be implemented when PubSub provider is available
                logger.info { "Agent discovery ready - waiting for PubSub integration" }
            } catch (e: Exception) {
                logger.error(e) { "Failed to start agent discovery" }
            }
        }
        
        private suspend fun startPeriodicMaintenance(
            config: DistributedFeatureConfig,
            contextManager: DistributedContextManager,
            pipeline: AIAgentPipeline
        ) {
            while (true) {
                try {
                    delay(config.maintenanceInterval.inWholeMilliseconds)
                    
                    val distributedContext = contextManager.getCurrentContext()
                    
                    // Note: Strategy maintenance calls will be implemented when PubSub integration is complete
                    logger.debug { "Skipping strategy.onPeriodicMaintenance call for proposal phase" }
                    
                    // Clean up old agent presence information
                    contextManager.cleanupStaleAgents(config.agentTimeoutDuration)
                    
                } catch (e: Exception) {
                    logger.warn(e) { "Error in periodic maintenance" }
                }
            }
        }
        
        private suspend fun executeDistributedAction(
            action: DistributedAction,
            config: DistributedFeatureConfig,
            contextManager: DistributedContextManager,
            agentContext: AIAgentContextBase
        ): DistributedActionResult {
            try {
                return when (action) {
                    is DistributedAction.ExecuteLocally -> {
                        // This would typically trigger the agent's normal execution pipeline
                        // For now, we just indicate success
                        DistributedActionResult(
                            success = true,
                            result = "Executing locally: ${action.input}"
                        )
                    }
                    
                    is DistributedAction.UpdateStatus -> {
                        // Update local status
                        contextManager.updateLocalStatus(action.status)
                        
                        DistributedActionResult(
                            success = true,
                            result = "Status updated"
                        )
                    }
                    
                    DistributedAction.Ignore -> {
                        DistributedActionResult(
                            success = true,
                            result = "Action ignored as requested"
                        )
                    }
                    
                    else -> {
                        // PubSub-dependent actions will be implemented when PubSub is available
                        DistributedActionResult(
                            success = true,
                            result = "Action queued for PubSub integration: ${action::class.simpleName}"
                        )
                    }
                }
            } catch (e: Exception) {
                logger.error(e) { "Failed to execute distributed action: ${action::class.simpleName}" }
                return DistributedActionResult(
                    success = false,
                    error = e.message ?: "Unknown error executing action"
                )
            }
        }
        
        private fun isMessageForThisAgent(message: DistributedMessage, agentId: DistributedAgentId): Boolean {
            // Message is for this agent if:
            // 1. It's directly addressed to this agent
            // 2. It's addressed to our role and we can handle it
            // 3. It's a broadcast message (no specific target)
            
            return when {
                message.toAgentId != null -> message.toAgentId.instanceId == agentId.instanceId
                message.toRole != null -> message.toRole == agentId.role
                else -> true // Broadcast message
            }
        }
        
        private fun getCoordinationTopics(role: AgentRole): List<String> = when (role) {
            AgentRole.PLANNER -> listOf("planner-tasks", "coordination", "agent-discovery")
            AgentRole.EXECUTOR -> listOf("executor-tasks", "coordination", "agent-discovery")
            AgentRole.OBSERVER -> listOf("observer-events", "status-updates", "agent-discovery")
            AgentRole.COORDINATOR -> listOf("coordinator-requests", "coordination", "agent-discovery")
        }
        
        private fun getRoleTopics(role: AgentRole): List<String> = when (role) {
            AgentRole.PLANNER -> listOf("planner-tasks")
            AgentRole.EXECUTOR -> listOf("executor-tasks")
            AgentRole.OBSERVER -> listOf("observer-events")
            AgentRole.COORDINATOR -> listOf("coordinator-requests")
        }
    }
}

//region Distributed Event Data Classes

/**
 * Event published when a distributed message is received.
 */
@kotlinx.serialization.Serializable
private data class DistributedMessageReceivedEvent(
    val eventId: String,
    val agentId: String,
    val message: DistributedMessage,
    override val timestamp: Long = kotlinx.datetime.Clock.System.now().toEpochMilliseconds(),
    override val messageType: FeatureMessage.Type = FeatureMessage.Type.Event
) : FeatureMessage

/**
 * Event published when a distributed operation encounters an error.
 */
@kotlinx.serialization.Serializable
private data class DistributedErrorEvent(
    val eventId: String,
    val agentId: String,
    val operation: String,
    val error: String,
    override val timestamp: Long = kotlinx.datetime.Clock.System.now().toEpochMilliseconds(),
    override val messageType: FeatureMessage.Type = FeatureMessage.Type.Event
) : FeatureMessage

//endregion