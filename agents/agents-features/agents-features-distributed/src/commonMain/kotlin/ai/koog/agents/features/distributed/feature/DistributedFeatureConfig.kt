package ai.koog.agents.features.distributed.feature

import ai.koog.agents.features.common.config.FeatureConfig
import ai.koog.agents.features.distributed.message.AgentRole
import ai.koog.agents.features.distributed.message.DistributedAgentId
import ai.koog.agents.features.distributed.strategies.DistributedStrategy
import ai.koog.agents.features.distributed.strategies.PlannerExecutorStrategy
import ai.koog.agents.features.pubsub.providers.PubSubProvider
import ai.koog.agents.features.pubsub.providers.NoPubSubProvider
import kotlin.time.Duration
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

/**
 * Configuration for the Distributed feature.
 * 
 * This configuration class allows you to define how agents coordinate in a distributed
 * multi-agent system. It includes settings for agent identity, coordination strategy,
 * discovery behavior, and operational parameters.
 * 
 * Example usage:
 * ```kotlin
 * val agent = AIAgent(...) {
 *     install(PubSub) {
 *         provider = RedisPubSubProvider("redis://localhost:6379")
 *     }
 *     
 *     install(Distributed) {
 *         // Define this agent's identity and role
 *         agentId = DistributedAgentId(
 *             role = AgentRole.PLANNER,
 *             capabilities = setOf("task-planning", "resource-optimization"),
 *             metadata = mapOf("version" to "1.2.0", "environment" to "production")
 *         )
 *         
 *         // Choose coordination strategy
 *         strategy = PlannerExecutorStrategy()
 *         
 *         // Enable agent discovery
 *         discoveryEnabled = true
 *         agentTimeoutDuration = 2.minutes
 *         
 *         // Configure maintenance
 *         maintenanceInterval = 30.seconds
 *         
 *         // Add message processors
 *         addMessageProcessor(DistributedMessageLogWriter(logger))
 *         addMessageProcessor(DistributedStatusReporter())
 *     }
 * }
 * ```
 */
public class DistributedFeatureConfig : FeatureConfig() {
    
    /**
     * The identity of this agent in the distributed system.
     * 
     * This includes the agent's role, capabilities, and metadata that other
     * agents can use for discovery and routing decisions.
     * 
     * Example:
     * ```kotlin
     * agentId = DistributedAgentId(
     *     role = AgentRole.EXECUTOR,
     *     capabilities = setOf("minecraft", "building", "resource-gathering"),
     *     metadata = mapOf("server" to "minecraft-1", "world" to "survival")
     * )
     * ```
     */
    public var agentId: DistributedAgentId = DistributedAgentId(role = AgentRole.EXECUTOR)
    
    /**
     * The coordination strategy that defines how this agent behaves in the distributed system.
     * 
     * The strategy determines:
     * - How to handle local input (execute locally vs delegate)
     * - How to process messages from other agents
     * - How to react to agent discovery and lifecycle events
     * 
     * Built-in strategies include:
     * - [PlannerExecutorStrategy]: Separates planning from execution
     * - [CoordinatorStrategy]: Central coordination with specialists
     * - [ObserverStrategy]: Monitoring and reporting
     * 
     * You can also implement custom strategies by implementing [DistributedStrategy].
     */
    public var strategy: DistributedStrategy = PlannerExecutorStrategy()
    
    /**
     * Whether to enable automatic agent discovery.
     * 
     * When enabled, this agent will:
     * - Announce its presence and capabilities to other agents
     * - Listen for announcements from other agents
     * - Maintain a registry of available agents for routing decisions
     * 
     * Discovery is useful for dynamic environments where agents come and go,
     * but can be disabled in static environments for better performance.
     */
    public var discoveryEnabled: Boolean = true
    
    /**
     * How long to wait before considering an agent offline.
     * 
     * Agents that haven't sent discovery or status messages within this
     * duration are considered offline and removed from routing decisions.
     * 
     * This should be longer than the discovery announcement interval to
     * avoid premature timeouts.
     */
    public var agentTimeoutDuration: Duration = 2.minutes
    
    /**
     * How often to perform periodic maintenance tasks.
     * 
     * Maintenance includes:
     * - Calling strategy.onPeriodicMaintenance()
     * - Cleaning up stale agent presence information
     * - Sending heartbeat or status messages
     * 
     * Shorter intervals provide more responsive coordination but use more resources.
     */
    public var maintenanceInterval: Duration = 30.seconds
    
    /**
     * Maximum number of messages to process concurrently.
     * 
     * This controls how many coordination messages can be processed
     * simultaneously. Higher values increase throughput but use more resources.
     */
    public var maxConcurrentMessages: Int = 10
    
    /**
     * Whether to enable coordination message filtering.
     * 
     * When enabled, messages are filtered through the strategy before processing,
     * allowing strategies to ignore irrelevant messages for better performance.
     */
    public var enableMessageFiltering: Boolean = true
    
    /**
     * Custom topics for coordination messages.
     * 
     * By default, topics are determined by agent roles (e.g., "planner-tasks",
     * "executor-tasks"). You can override this for custom routing schemes.
     * 
     * Example:
     * ```kotlin
     * customTopics = mapOf(
     *     "high-priority" to setOf(AgentRole.EXECUTOR, AgentRole.COORDINATOR),
     *     "analysis-requests" to setOf(AgentRole.OBSERVER),
     *     "minecraft-commands" to setOf(AgentRole.EXECUTOR)
     * )
     * ```
     */
    public var customTopics: Map<String, Set<AgentRole>> = emptyMap()
    
    /**
     * The PubSub provider to use for distributed messaging.
     * 
     * This is typically set automatically by reading the PubSub feature configuration,
     * but can be overridden for advanced use cases like using different providers
     * for different types of coordination.
     * 
     * Note: This should normally be left as the default and configured through
     * the PubSub feature instead.
     */
    internal var pubSubProvider: PubSubProvider = NoPubSubProvider()
    
    /**
     * Additional strategy-specific configuration.
     * 
     * This map can be used to pass configuration that is specific to
     * the chosen coordination strategy implementation.
     * 
     * Example:
     * ```kotlin
     * strategyConfig = mapOf(
     *     "retry_attempts" to "3",
     *     "timeout_seconds" to "30",
     *     "preferred_executors" to "minecraft-1,minecraft-2"
     * )
     * ```
     */
    public var strategyConfig: Map<String, String> = emptyMap()
    
    /**
     * Gets a strategy-specific configuration value.
     */
    public fun getStrategyConfig(key: String): String? = strategyConfig[key]
    
    /**
     * Gets a strategy-specific configuration value with a default.
     */
    public fun getStrategyConfig(key: String, default: String): String = 
        strategyConfig[key] ?: default
    
    /**
     * Sets strategy-specific configuration.
     */
    public fun setStrategyConfig(key: String, value: String) {
        strategyConfig = strategyConfig + (key to value)
    }
    
    /**
     * Creates a configuration for a planner agent.
     */
    public companion object {
        /**
         * Creates a configuration for a planner agent with common defaults.
         */
        public fun forPlanner(
            capabilities: Set<String> = setOf("planning", "coordination"),
            metadata: Map<String, String> = emptyMap()
        ): DistributedFeatureConfig = DistributedFeatureConfig().apply {
            agentId = DistributedAgentId(
                role = AgentRole.PLANNER,
                capabilities = capabilities,
                metadata = metadata
            )
            strategy = PlannerExecutorStrategy()
            discoveryEnabled = true
        }
        
        /**
         * Creates a configuration for an executor agent with common defaults.
         */
        public fun forExecutor(
            capabilities: Set<String> = setOf("execution", "tools"),
            metadata: Map<String, String> = emptyMap()
        ): DistributedFeatureConfig = DistributedFeatureConfig().apply {
            agentId = DistributedAgentId(
                role = AgentRole.EXECUTOR,
                capabilities = capabilities,
                metadata = metadata
            )
            strategy = PlannerExecutorStrategy()
            discoveryEnabled = true
        }
        
        /**
         * Creates a configuration for an observer agent with common defaults.
         */
        public fun forObserver(
            capabilities: Set<String> = setOf("monitoring", "analysis"),
            metadata: Map<String, String> = emptyMap()
        ): DistributedFeatureConfig = DistributedFeatureConfig().apply {
            agentId = DistributedAgentId(
                role = AgentRole.OBSERVER,
                capabilities = capabilities,
                metadata = metadata
            )
            // Observer strategy would be implemented separately
            discoveryEnabled = true
            maintenanceInterval = 10.seconds // Observers need more frequent updates
        }
        
        /**
         * Creates a configuration for a coordinator agent with common defaults.
         */
        public fun forCoordinator(
            capabilities: Set<String> = setOf("coordination", "orchestration"),
            metadata: Map<String, String> = emptyMap()
        ): DistributedFeatureConfig = DistributedFeatureConfig().apply {
            agentId = DistributedAgentId(
                role = AgentRole.COORDINATOR,
                capabilities = capabilities,
                metadata = metadata
            )
            // Coordinator strategy would be implemented separately
            discoveryEnabled = true
            maxConcurrentMessages = 20 // Coordinators need higher throughput
        }
    }
}