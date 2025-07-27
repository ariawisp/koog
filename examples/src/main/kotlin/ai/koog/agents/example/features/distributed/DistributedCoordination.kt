package ai.koog.agents.example.features.distributed

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.example.ApiKeyService
import ai.koog.agents.features.distributed.feature.Distributed
import ai.koog.agents.features.distributed.message.AgentRole
import ai.koog.agents.features.distributed.message.DistributedAgentId
import ai.koog.agents.features.distributed.strategies.PlannerExecutorStrategy
import ai.koog.agents.features.pubsub.feature.PubSub
import ai.koog.agents.features.pubsub.providers.local.LocalFilePubSubProvider
import ai.koog.agents.utils.use
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

/**
 * Example demonstrating Distributed Coordination feature for true multi-agent systems.
 * 
 * This example shows how agents can coordinate across different processes using:
 * - Role-based agent architecture (Planner, Executors)
 * - Cross-process communication via PubSub messaging
 * - Intelligent task distribution and coordination
 * - Distributed strategy patterns for agent coordination
 */
fun main() = runBlocking {
    
    println("=== Distributed Coordination Example ===")
    println("Demonstrating multi-agent coordination with role-based architecture")
    
    // Shared PubSub provider for cross-process agent communication
    val pubSubProvider = LocalFilePubSubProvider()
    
    try {
        // Planner Agent - Strategic coordination and task breakdown
        val plannerAgent = AIAgent(
            executor = simpleOpenAIExecutor(ApiKeyService.openAIApiKey),
            llmModel = OpenAIModels.Chat.GPT4o,
            systemPrompt = """
                You are a Strategic Planner Agent in a distributed multi-agent system.
                
                Your role is to:
                - Break down complex business tasks into specialized subtasks
                - Coordinate with specialized executor agents
                - Synthesize results from multiple agents into comprehensive deliverables
                
                When you receive task results from executors, synthesize them into final recommendations.
            """.trimIndent()
        ) {
            install(PubSub) {
                provider = pubSubProvider
                autoSubscribeTopics = listOf("task-results")
                publishAgentEvents = true
            }
            
            install(Distributed) {
                agentId = DistributedAgentId(
                    role = AgentRole.PLANNER,
                    capabilities = setOf("planning", "coordination", "synthesis"),
                    metadata = mapOf("specialization" to "business-strategy")
                )
                strategy = PlannerExecutorStrategy()
            }
        }
        
        // Text Executor Agent - Content creation and writing
        val textExecutorAgent = AIAgent(
            executor = simpleOpenAIExecutor(ApiKeyService.openAIApiKey),
            llmModel = OpenAIModels.Chat.GPT4o,
            systemPrompt = """
                You are a Text Executor Agent specializing in content creation.
                
                Your capabilities include:
                - Executive summaries and business writing
                - Content analysis and documentation
                - Strategic communication development
                
                You receive specific text tasks and deliver professional written outputs.
            """.trimIndent()
        ) {
            install(PubSub) {
                provider = pubSubProvider
                autoSubscribeTopics = listOf("text-tasks")
                publishAgentEvents = true
            }
            
            install(Distributed) {
                agentId = DistributedAgentId(
                    role = AgentRole.EXECUTOR,
                    capabilities = setOf("writing", "content-creation", "analysis"),
                    metadata = mapOf("specialization" to "text-processing")
                )
                strategy = PlannerExecutorStrategy()
            }
        }
        
        // Data Executor Agent - Analysis and insights
        val dataExecutorAgent = AIAgent(
            executor = simpleOpenAIExecutor(ApiKeyService.openAIApiKey),
            llmModel = OpenAIModels.Chat.GPT4o,
            systemPrompt = """
                You are a Data Executor Agent specializing in quantitative analysis.
                
                Your capabilities include:
                - Statistical analysis and modeling
                - Business metrics evaluation
                - Data-driven insight generation
                
                You receive analytical tasks and provide quantitative insights and recommendations.
            """.trimIndent()
        ) {
            install(PubSub) {
                provider = pubSubProvider
                autoSubscribeTopics = listOf("data-tasks")
                publishAgentEvents = true
            }
            
            install(Distributed) {
                agentId = DistributedAgentId(
                    role = AgentRole.EXECUTOR,
                    capabilities = setOf("data-analysis", "statistics", "modeling"),
                    metadata = mapOf("specialization" to "quantitative-analysis")
                )
                strategy = PlannerExecutorStrategy()
            }
        }
        
        // Demonstrate distributed coordination
        plannerAgent.use { planner ->
            textExecutorAgent.use { textExecutor ->
                dataExecutorAgent.use { dataExecutor ->
                    
                    println("\n--- Setting up distributed agent message handlers ---")
                    
                    val completedTasks = mutableListOf<String>()
                    
                    // Text executor message handler
                    launch {
                        pubSubProvider.subscribe("text-tasks").collect { message ->
                            println("📝 Text Executor received: ${message.content}")
                            
                            try {
                                val result = textExecutor(message.content)
                                
                                pubSubProvider.publish(
                                    "task-results",
                                    "TEXT COMPLETE: $result",
                                    mapOf("executor" to "text", "status" to "completed")
                                )
                                
                                println("✅ Text task completed")
                            } catch (e: Exception) {
                                println("❌ Text executor error: ${e.message}")
                            }
                            
                            message.acknowledge()
                        }
                    }
                    
                    // Data executor message handler
                    launch {
                        pubSubProvider.subscribe("data-tasks").collect { message ->
                            println("📊 Data Executor received: ${message.content}")
                            
                            try {
                                val result = dataExecutor(message.content)
                                
                                pubSubProvider.publish(
                                    "task-results",
                                    "DATA COMPLETE: $result",
                                    mapOf("executor" to "data", "status" to "completed")
                                )
                                
                                println("✅ Data task completed")
                            } catch (e: Exception) {
                                println("❌ Data executor error: ${e.message}")
                            }
                            
                            message.acknowledge()
                        }
                    }
                    
                    // Planner result handler
                    launch {
                        pubSubProvider.subscribe("task-results").collect { message ->
                            println("🎯 Planner received result: ${message.content}")
                            completedTasks.add(message.content)
                            
                            // When both tasks complete, synthesize final analysis
                            if (completedTasks.size >= 2) {
                                println("\n--- Planner synthesizing final analysis ---")
                                
                                try {
                                    val synthesis = planner(
                                        "Synthesize these results into business recommendations: ${completedTasks.joinToString("; ")}"
                                    )
                                    
                                    println("🎉 FINAL COORDINATION RESULT:")
                                    println("=" * 50)
                                    println(synthesis)
                                    println("=" * 50)
                                    
                                } catch (e: Exception) {
                                    println("❌ Synthesis error: ${e.message}")
                                }
                            }
                            
                            message.acknowledge()
                        }
                    }
                    
                    // Allow message handlers to initialize
                    delay(200)
                    
                    println("\n--- Starting distributed task execution ---")
                    
                    // Planner initiates coordination
                    val businessTask = "Analyze remote work productivity trends and create actionable recommendations"
                    println("🎯 Coordinating task: $businessTask")
                    
                    val plannerResponse = planner("Break down this business task and coordinate the analysis: $businessTask")
                    println("🧠 Planner strategy: $plannerResponse")
                    
                    // Distribute specialized tasks to executors
                    pubSubProvider.publish(
                        "text-tasks",
                        "Create an executive summary of remote work productivity trends and management best practices",
                        mapOf("priority" to "high")
                    )
                    
                    pubSubProvider.publish(
                        "data-tasks",
                        "Analyze productivity metrics for remote work and provide statistical insights with recommendations",
                        mapOf("priority" to "high")
                    )
                    
                    // Wait for distributed coordination to complete
                    println("\n--- Agents coordinating across distributed system ---")
                    delay(15000) // Allow time for coordination
                    
                    // Show coordination health
                    println("\n--- Distributed System Health ---")
                    val healthInfo = pubSubProvider.getHealthInfo()
                    healthInfo.forEach { (key, value) ->
                        println("$key: $value")
                    }
                }
            }
        }
        
    } finally {
        pubSubProvider.close()
        println("\n--- Distributed coordination example completed ---")
    }
}

/**
 * String repetition operator for formatting
 */
private operator fun String.times(n: Int): String = this.repeat(n)