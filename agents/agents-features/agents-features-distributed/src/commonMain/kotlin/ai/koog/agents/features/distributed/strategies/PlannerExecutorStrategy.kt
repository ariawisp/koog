package ai.koog.agents.features.distributed.strategies

import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.features.distributed.message.*
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json

/**
 * Coordination strategy that implements planner-executor separation.
 * 
 * This strategy enables a classic distributed agent pattern where:
 * - **Planner agents** handle high-level strategy, goal decomposition, and task assignment
 * - **Executor agents** handle action execution, tool usage, and result reporting
 * 
 * The strategy automatically routes work based on agent roles:
 * - Planners receive user input, generate plans, and delegate tasks to executors
 * - Executors receive task assignments and execute them using their available tools
 * - Results flow back from executors to planners for coordination and next steps
 * 
 * ## Example Architecture
 * 
 * ```
 * User Input → Planner Agent → Task Assignment → Executor Agent
 *                    ↑                              ↓
 *              Next Planning ← Task Results ← Tool Execution
 * ```
 * 
 * ## Usage
 * 
 * ```kotlin
 * // Planner Agent (Ktor server)
 * val plannerAgent = AIAgent(...) {
 *     install(Distributed) {
 *         agentId = DistributedAgentId(role = AgentRole.PLANNER)
 *         strategy = PlannerExecutorStrategy()
 *     }
 * }
 * 
 * // Executor Agent (Minecraft server)  
 * val executorAgent = AIAgent(...) {
 *     install(Distributed) {
 *         agentId = DistributedAgentId(role = AgentRole.EXECUTOR)
 *         strategy = PlannerExecutorStrategy()
 *     }
 * }
 * ```
 * 
 * ## Coordination Flow
 * 
 * 1. **User Input**: Planner receives input and decides if it should plan or delegate
 * 2. **Task Generation**: Planner breaks down complex goals into executable tasks
 * 3. **Task Assignment**: Planner sends tasks to available executor agents
 * 4. **Execution**: Executors receive tasks and execute them using available tools
 * 5. **Result Reporting**: Executors send results back to the originating planner
 * 6. **Continuation**: Planner receives results and decides on next actions
 * 
 * This pattern is particularly effective for:
 * - Separating planning logic from execution environments
 * - Scaling execution capacity independently from planning
 * - Enabling specialized execution environments (e.g., game servers, IoT devices)
 * - Maintaining clean separation of concerns in multi-agent systems
 */
public class PlannerExecutorStrategy : DistributedStrategy {
    
    override val name: String = "planner-executor"
    
    private val json = Json { ignoreUnknownKeys = true }
    
    override suspend fun onLocalInput(
        input: String,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        return when (distributedContext.currentAgentId.role) {
            AgentRole.PLANNER -> handlePlannerInput(input, context, distributedContext)
            AgentRole.EXECUTOR -> handleExecutorInput(input, context, distributedContext)
            else -> {
                // Other roles don't handle local input in this strategy
                DistributedAction.Ignore
            }
        }
    }
    
    override suspend fun onRemoteMessage(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        return when (message.messageType) {
            MessageType.TASK_REQUEST -> handleTaskRequest(message, context, distributedContext)
            MessageType.TASK_RESPONSE -> handleTaskResponse(message, context, distributedContext)
            MessageType.EXECUTION_REQUEST -> handleExecutionRequest(message, context, distributedContext)
            MessageType.EXECUTION_RESULT -> handleExecutionResult(message, context, distributedContext)
            MessageType.STATUS_UPDATE -> handleStatusUpdate(message, context, distributedContext)
            else -> DistributedAction.Ignore
        }
    }
    
    private suspend fun handlePlannerInput(
        input: String,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Check if there are available executors
        val availableExecutors = distributedContext.findAgentsByRole(AgentRole.EXECUTOR)
        
        return if (availableExecutors.isNotEmpty()) {
            // We have executors available, so generate a plan and delegate
            val taskRequest = TaskRequest(
                task = input,
                parameters = mapOf(
                    "originating_planner" to distributedContext.currentAgentId.instanceId,
                    "planning_timestamp" to distributedContext.timestamp.toString()
                ),
                priority = determinePriority(input)
            )
            
            DistributedAction.SendToRole(
                role = AgentRole.EXECUTOR,
                payload = json.encodeToString(taskRequest),
                messageType = MessageType.TASK_REQUEST,
                priority = taskRequest.priority
            )
        } else {
            // No executors available, try to handle locally if possible
            if (canPlannerExecuteLocally(input, distributedContext)) {
                DistributedAction.ExecuteLocally(input)
            } else {
                // Can't execute locally and no executors available
                // This could trigger a "waiting for executors" state
                DistributedAction.UpdateStatus(
                    AgentStatus(
                        status = "waiting_for_executors",
                        currentTask = "Pending task: $input"
                    )
                )
            }
        }
    }
    
    private suspend fun handleExecutorInput(
        input: String,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Executors typically execute input directly unless it requires planning
        return if (requiresPlanning(input)) {
            // This input needs planning, forward to a planner
            val availablePlanners = distributedContext.findAgentsByRole(AgentRole.PLANNER)
            
            if (availablePlanners.isNotEmpty()) {
                val taskRequest = TaskRequest(
                    task = input,
                    parameters = mapOf(
                        "originating_executor" to distributedContext.currentAgentId.instanceId,
                        "requires_planning" to "true"
                    )
                )
                
                DistributedAction.SendToRole(
                    role = AgentRole.PLANNER,
                    payload = json.encodeToString(taskRequest),
                    messageType = MessageType.TASK_REQUEST
                )
            } else {
                // No planners available, try to execute locally
                DistributedAction.ExecuteLocally(input)
            }
        } else {
            // Direct execution
            DistributedAction.ExecuteLocally(input)
        }
    }
    
    private suspend fun handleTaskRequest(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        return when (distributedContext.currentAgentId.role) {
            AgentRole.EXECUTOR -> {
                // Executor receives task request from planner
                try {
                    val taskRequest = json.decodeFromString<TaskRequest>(message.payload)
                    
                    if (canExecuteTask(taskRequest, distributedContext)) {
                        // Update status to indicate we're working on this task
                        updateStatusForTask(taskRequest, "executing")
                        
                        // Execute the task locally
                        DistributedAction.ExecuteLocally(taskRequest.task)
                    } else {
                        // Can't execute this task, send back an appropriate response
                        val response = TaskResponse(
                            success = false,
                            error = "Executor ${distributedContext.currentAgentId.instanceId} cannot handle task: missing capabilities or resources"
                        )
                        
                        message.createResponse(
                            fromAgent = distributedContext.currentAgentId,
                            messageType = MessageType.TASK_RESPONSE,
                            payload = json.encodeToString(response)
                        ).let { responseMessage ->
                            DistributedAction.DelegateTo(
                                agentId = message.fromAgent,
                                payload = json.encodeToString(responseMessage)
                            )
                        }
                    }
                } catch (e: Exception) {
                    // Failed to parse task request
                    val response = TaskResponse(
                        success = false,
                        error = "Failed to parse task request: ${e.message}"
                    )
                    
                    message.createResponse(
                        fromAgent = distributedContext.currentAgentId,
                        messageType = MessageType.TASK_RESPONSE,
                        payload = json.encodeToString(response)
                    ).let { responseMessage ->
                        DistributedAction.DelegateTo(
                            agentId = message.fromAgent,
                            payload = json.encodeToString(responseMessage)
                        )
                    }
                }
            }
            
            AgentRole.PLANNER -> {
                // Planner receives task request (possibly from executor needing planning)
                DistributedAction.ExecuteLocally(
                    json.decodeFromString<TaskRequest>(message.payload).task
                )
            }
            
            else -> DistributedAction.Ignore
        }
    }
    
    private suspend fun handleTaskResponse(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        return when (distributedContext.currentAgentId.role) {
            AgentRole.PLANNER -> {
                // Planner receives task response from executor
                try {
                    val response = json.decodeFromString<TaskResponse>(message.payload)
                    
                    if (response.success) {
                        // Task completed successfully, decide next action
                        val nextAction = determineNextAction(response, message, distributedContext)
                        nextAction ?: DistributedAction.UpdateStatus(
                            AgentStatus(
                                status = "idle",
                                currentTask = null,
                                metrics = mapOf("last_completed_task" to (response.result ?: "unknown"))
                            )
                        )
                    } else {
                        // Task failed, decide on retry or alternative approach
                        handleTaskFailure(response, message, distributedContext)
                    }
                } catch (e: Exception) {
                    // Failed to parse response
                    DistributedAction.UpdateStatus(
                        AgentStatus(
                            status = "error",
                            currentTask = "Failed to parse task response from ${message.fromAgent.toDisplayString()}"
                        )
                    )
                }
            }
            
            else -> DistributedAction.Ignore
        }
    }
    
    private suspend fun handleExecutionRequest(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Handle direct execution requests (more immediate than task requests)
        return when (distributedContext.currentAgentId.role) {
            AgentRole.EXECUTOR -> DistributedAction.ExecuteLocally(message.payload)
            else -> DistributedAction.Ignore
        }
    }
    
    private suspend fun handleExecutionResult(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Handle execution results (similar to task responses but more immediate)
        return handleTaskResponse(message, context, distributedContext)
    }
    
    private suspend fun handleStatusUpdate(
        message: DistributedMessage,
        context: AIAgentContextBase,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Process status updates from other agents
        // Planners might use this information for load balancing and coordination
        return DistributedAction.Ignore // Just acknowledge the status update
    }
    
    private fun determinePriority(input: String): Int {
        // Simple priority determination based on keywords
        // In practice, this would be more sophisticated
        return when {
            input.contains("urgent", ignoreCase = true) -> 10
            input.contains("important", ignoreCase = true) -> 5
            input.contains("critical", ignoreCase = true) -> 15
            else -> 0
        }
    }
    
    private fun canPlannerExecuteLocally(input: String, distributedContext: DistributedContext): Boolean {
        // Determine if a planner can execute the input locally
        // This might depend on available tools, complexity of the task, etc.
        return input.length < 100 && // Simple tasks only
               !input.contains("build", ignoreCase = true) && // No building tasks
               !input.contains("craft", ignoreCase = true) // No crafting tasks
    }
    
    private fun requiresPlanning(input: String): Boolean {
        // Determine if input requires planning before execution
        return input.contains("plan", ignoreCase = true) ||
               input.contains("strategy", ignoreCase = true) ||
               input.contains("coordinate", ignoreCase = true) ||
               input.split(" ").size > 10 // Complex multi-step tasks
    }
    
    private fun canExecuteTask(taskRequest: TaskRequest, distributedContext: DistributedContext): Boolean {
        // Check if this executor can handle the task based on requirements and capabilities
        val requiredCapabilities = taskRequest.requirements
        val agentCapabilities = distributedContext.currentAgentId.capabilities
        
        return if (requiredCapabilities.isNotEmpty()) {
            agentCapabilities.containsAll(requiredCapabilities)
        } else {
            true // No specific requirements
        }
    }
    
    private suspend fun updateStatusForTask(taskRequest: TaskRequest, status: String): DistributedAction {
        return DistributedAction.UpdateStatus(
            AgentStatus(
                status = status,
                currentTask = taskRequest.task,
                progress = 0,
                metrics = mapOf(
                    "task_priority" to taskRequest.priority.toString(),
                    "task_deadline" to (taskRequest.deadline?.toString() ?: "none")
                )
            )
        )
    }
    
    private fun determineNextAction(
        response: TaskResponse,
        originalMessage: DistributedMessage,
        distributedContext: DistributedContext
    ): DistributedAction? {
        // Analyze the response and determine if more work needs to be done
        // This could involve:
        // - Assigning follow-up tasks
        // - Gathering more information
        // - Coordinating with other agents
        // - Reporting completion to the user
        
        return null // For now, just let the agent go idle
    }
    
    private fun handleTaskFailure(
        response: TaskResponse,
        originalMessage: DistributedMessage,
        distributedContext: DistributedContext
    ): DistributedAction {
        // Handle task failures with potential retry logic or alternative approaches
        val availableExecutors = distributedContext.findAgentsByRole(AgentRole.EXECUTOR)
            .filter { it.instanceId != originalMessage.fromAgent.instanceId } // Exclude the one that failed
        
        return if (availableExecutors.isNotEmpty()) {
            // Try with a different executor
            val originalTaskRequest = try {
                json.decodeFromString<TaskRequest>(originalMessage.payload)
            } catch (e: Exception) {
                null
            }
            
            if (originalTaskRequest != null) {
                val retryRequest = originalTaskRequest.copy(
                    parameters = originalTaskRequest.parameters + mapOf(
                        "retry_attempt" to "true",
                        "previous_failure" to (response.error ?: "unknown error")
                    )
                )
                
                DistributedAction.SendToRole(
                    role = AgentRole.EXECUTOR,
                    payload = json.encodeToString(retryRequest),
                    messageType = MessageType.TASK_REQUEST
                )
            } else {
                DistributedAction.UpdateStatus(
                    AgentStatus(
                        status = "error",
                        currentTask = "Failed to retry task due to malformed original request"
                    )
                )
            }
        } else {
            // No other executors available
            DistributedAction.UpdateStatus(
                AgentStatus(
                    status = "error",
                    currentTask = "Task failed and no alternative executors available: ${response.error}"
                )
            )
        }
    }
    
    override fun validateConfiguration(
        agentId: DistributedAgentId,
        context: AIAgentContextBase
    ): List<String> {
        val errors = mutableListOf<String>()
        
        // This strategy requires agents to be either planners or executors
        if (agentId.role !in setOf(AgentRole.PLANNER, AgentRole.EXECUTOR)) {
            errors.add("PlannerExecutorStrategy requires agent role to be PLANNER or EXECUTOR, got ${agentId.role}")
        }
        
        // Additional validation could check for required tools, capabilities, etc.
        
        return errors
    }
}