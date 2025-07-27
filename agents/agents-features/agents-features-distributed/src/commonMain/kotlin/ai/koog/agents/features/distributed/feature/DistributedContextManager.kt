package ai.koog.agents.features.distributed.feature

import ai.koog.agents.features.distributed.message.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import io.github.oshai.kotlinlogging.KotlinLogging
import kotlin.time.Duration

/**
 * Manages distributed system context and agent presence information.
 * 
 * The DistributedContextManager is responsible for:
 * - Tracking available agents in the distributed system
 * - Maintaining agent presence and capability information
 * - Providing current system state to coordination strategies
 * - Cleaning up stale agent information
 * - Managing local agent status
 * 
 * This class is thread-safe and handles concurrent updates from multiple
 * coordination coroutines.
 */
internal class DistributedContextManager(
    private val config: DistributedFeatureConfig
) {
    private val logger = KotlinLogging.logger { }
    
    // Thread-safe storage for agent presence information
    private val agentPresence = mutableMapOf<String, AgentPresenceInfo>()
    private val presenceMutex = Mutex()
    
    // Local agent status
    private var localStatus: AgentStatus = AgentStatus(
        status = "online",
        capabilities = config.agentId.capabilities
    )
    private val statusMutex = Mutex()
    
    // System metrics and state
    private var systemMetrics = mutableMapOf<String, String>()
    private val metricsMutex = Mutex()
    
    /**
     * Information about an agent's presence in the distributed system.
     */
    private data class AgentPresenceInfo(
        val agentId: DistributedAgentId,
        val lastSeen: Instant,
        val status: AgentStatus? = null
    )
    
    /**
     * Updates the presence information for an agent.
     * 
     * This is called when:
     * - Discovery messages are received from other agents
     * - Coordination messages indicate an agent is active
     * - Status updates are received from agents
     */
    suspend fun updateAgentPresence(
        agentId: DistributedAgentId, 
        status: AgentStatus? = null
    ) {
        presenceMutex.withLock {
            val currentInfo = agentPresence[agentId.instanceId]
            
            agentPresence[agentId.instanceId] = AgentPresenceInfo(
                agentId = agentId,
                lastSeen = Clock.System.now(),
                status = status ?: currentInfo?.status
            )
            
            logger.debug { "Updated presence for agent: ${agentId.toDisplayString()}" }
        }
    }
    
    /**
     * Updates the local agent's status.
     */
    suspend fun updateLocalStatus(status: AgentStatus) {
        statusMutex.withLock {
            localStatus = status
            logger.debug { "Updated local status: ${status.status}" }
        }
    }
    
    /**
     * Gets the current distributed system context.
     * 
     * This provides a snapshot of the current state for use by coordination strategies.
     */
    suspend fun getCurrentContext(): DistributedContext {
        val currentAgents = presenceMutex.withLock {
            agentPresence.values.map { it.agentId }.toList()
        }
        
        val currentMetrics = metricsMutex.withLock {
            systemMetrics.toMap()
        }
        
        return DistributedContext(
            currentAgentId = config.agentId,
            availableAgents = currentAgents,
            systemMetrics = currentMetrics,
            timestamp = Clock.System.now()
        )
    }
    
    /**
     * Gets information about a specific agent.
     */
    suspend fun getAgentInfo(instanceId: String): DistributedAgentId? {
        return presenceMutex.withLock {
            agentPresence[instanceId]?.agentId
        }
    }
    
    /**
     * Gets the status of a specific agent.
     */
    suspend fun getAgentStatus(instanceId: String): AgentStatus? {
        return presenceMutex.withLock {
            agentPresence[instanceId]?.status
        }
    }
    
    /**
     * Gets all agents with a specific role.
     */
    suspend fun getAgentsByRole(role: AgentRole): List<DistributedAgentId> {
        return presenceMutex.withLock {
            agentPresence.values
                .map { it.agentId }
                .filter { it.role == role }
                .toList()
        }
    }
    
    /**
     * Gets all agents with a specific capability.
     */
    suspend fun getAgentsByCapability(capability: String): List<DistributedAgentId> {
        return presenceMutex.withLock {
            agentPresence.values
                .map { it.agentId }
                .filter { it.hasCapability(capability) }
                .toList()
        }
    }
    
    /**
     * Gets all agents with all required capabilities.
     */
    suspend fun getAgentsByCapabilities(capabilities: Set<String>): List<DistributedAgentId> {
        return presenceMutex.withLock {
            agentPresence.values
                .map { it.agentId }
                .filter { it.hasAllCapabilities(capabilities) }
                .toList()
        }
    }
    
    /**
     * Checks if any agents with a specific role are currently available.
     */
    suspend fun hasAvailableRole(role: AgentRole): Boolean {
        return presenceMutex.withLock {
            agentPresence.values.any { it.agentId.role == role }
        }
    }
    
    /**
     * Checks if any agents with a specific capability are currently available.
     */
    suspend fun hasAvailableCapability(capability: String): Boolean {
        return presenceMutex.withLock {
            agentPresence.values.any { it.agentId.hasCapability(capability) }
        }
    }
    
    /**
     * Updates a system metric.
     */
    suspend fun updateSystemMetric(key: String, value: String) {
        metricsMutex.withLock {
            systemMetrics[key] = value
        }
    }
    
    /**
     * Gets a system metric value.
     */
    suspend fun getSystemMetric(key: String): String? {
        return metricsMutex.withLock {
            systemMetrics[key]
        }
    }
    
    /**
     * Gets all current system metrics.
     */
    suspend fun getAllSystemMetrics(): Map<String, String> {
        return metricsMutex.withLock {
            systemMetrics.toMap()
        }
    }
    
    /**
     * Removes agents that haven't been seen within the timeout duration.
     * 
     * This is called periodically during maintenance to clean up stale agent information.
     */
    suspend fun cleanupStaleAgents(timeoutDuration: Duration) {
        val cutoffTime = Clock.System.now() - timeoutDuration
        
        val removedAgents = presenceMutex.withLock {
            val staleAgents = agentPresence.values
                .filter { it.lastSeen < cutoffTime }
                .map { it.agentId }
                .toList()
            
            staleAgents.forEach { agent ->
                agentPresence.remove(agent.instanceId)
            }
            
            staleAgents
        }
        
        if (removedAgents.isNotEmpty()) {
            logger.info { 
                "Cleaned up ${removedAgents.size} stale agents: ${removedAgents.map { it.toDisplayString() }}" 
            }
        }
    }
    
    /**
     * Gets statistics about the current distributed system state.
     */
    suspend fun getSystemStatistics(): Map<String, Any> {
        val agentStats = presenceMutex.withLock {
            val roleCount = agentPresence.values
                .groupBy { it.agentId.role }
                .mapValues { it.value.size }
            
            val capabilityCount = agentPresence.values
                .flatMap { it.agentId.capabilities }
                .groupBy { it }
                .mapValues { it.value.size }
            
            mapOf(
                "total_agents" to agentPresence.size,
                "agents_by_role" to roleCount,
                "capabilities_available" to capabilityCount.keys.toList(),
                "capability_counts" to capabilityCount
            )
        }
        
        val systemStats = metricsMutex.withLock {
            mapOf("system_metrics" to systemMetrics.toMap())
        }
        
        val localStats = statusMutex.withLock {
            mapOf(
                "local_agent" to config.agentId.toDisplayString(),
                "local_status" to localStatus.status,
                "local_capabilities" to localStatus.capabilities.toList()
            )
        }
        
        return agentStats + systemStats + localStats + mapOf(
            "discovery_enabled" to config.discoveryEnabled,
            "strategy" to config.strategy.name,
            "maintenance_interval_seconds" to config.maintenanceInterval.inWholeSeconds,
            "agent_timeout_seconds" to config.agentTimeoutDuration.inWholeSeconds
        )
    }
    
    /**
     * Finds the best agent to handle a task based on capabilities and current load.
     * 
     * This is a utility method that strategies can use for intelligent routing decisions.
     */
    suspend fun findBestAgentForTask(
        requiredCapabilities: Set<String>,
        preferredRole: AgentRole? = null,
        excludeAgents: Set<String> = emptySet()
    ): DistributedAgentId? {
        return presenceMutex.withLock {
            val candidates = agentPresence.values
                .map { it.agentId }
                .filter { agent ->
                    // Must have required capabilities
                    agent.hasAllCapabilities(requiredCapabilities) &&
                    // Not in exclude list
                    agent.instanceId !in excludeAgents &&
                    // Not the current agent (avoid self-delegation)
                    agent.instanceId != config.agentId.instanceId
                }
            
            // Prefer agents with the preferred role
            val preferred = if (preferredRole != null) {
                candidates.filter { it.role == preferredRole }
            } else {
                candidates
            }
            
            // For now, just return the first suitable agent
            // In the future, this could consider load balancing, performance metrics, etc.
            preferred.firstOrNull() ?: candidates.firstOrNull()
        }
    }
    
    /**
     * Shuts down the context manager and cleans up resources.
     */
    suspend fun shutdown() {
        logger.info { "Shutting down DistributedContextManager for agent: ${config.agentId.toDisplayString()}" }
        
        presenceMutex.withLock {
            agentPresence.clear()
        }
        
        metricsMutex.withLock {
            systemMetrics.clear()
        }
        
        logger.debug { "DistributedContextManager shutdown complete" }
    }
}