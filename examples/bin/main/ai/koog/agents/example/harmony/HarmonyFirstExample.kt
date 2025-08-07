package ai.koog.agents.example.harmony

import ai.koog.agents.core.agent.HarmonyAIAgent
import ai.koog.agents.core.agent.HarmonySimpleStrategy
import ai.koog.agents.core.environment.AIAgentEnvironment
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.prompt.dsl.*
import ai.koog.prompt.executor.model.DefaultHarmonyPromptExecutor
import ai.koog.prompt.executor.model.HarmonyProviderRegistry
import ai.koog.prompt.harmony.*
import ai.koog.prompt.llm.LLModel
import kotlinx.coroutines.runBlocking

/**
 * HarmonyFirstExample - Comprehensive demonstration of the Harmony-first architecture.
 * 
 * This example shows how the new Harmony-first Koog provides:
 * 1. Semantic understanding of conversations through channels
 * 2. Safety by construction (analysis never leaks to users)
 * 3. Rich reasoning capabilities with effort control
 * 4. Provider independence through downsamplers
 * 5. Full traceability of agent decision-making
 * 
 * Key features demonstrated:
 * - Harmony DSL for fluent request building
 * - Channel-aware agent execution
 * - Reasoning effort control
 * - Tool interactions tracked in Commentary channel
 * - Safety guarantees enforced automatically
 * - Provider downsampling (OpenAI in this case)
 */

fun main() = runBlocking {
    println("🎭 Harmony-First Koog Architecture Demo")
    println("========================================")
    
    // 1. Set up the Harmony-first infrastructure
    val harmonyAgent = createHarmonyAgent()
    
    try {
        // 2. Demonstrate basic Harmony request with channel semantics
        println("\n📝 Basic Harmony Request with Channel Semantics")
        demonstrateBasicHarmonyRequest(harmonyAgent)
        
        // 3. Show reasoning effort control
        println("\n🧠 Reasoning Effort Control")
        demonstrateReasoningControl(harmonyAgent)
        
        // 4. Demonstrate tool interactions with channel tracking
        println("\n🔧 Tool Interactions with Channel Tracking")
        demonstrateToolInteractions(harmonyAgent)
        
        // 5. Show safety guarantees in action
        println("\n🔒 Safety Guarantees - Analysis Channel Protection")
        demonstrateSafetyGuarantees(harmonyAgent)
        
        // 6. Demonstrate streaming with channel awareness
        println("\n⚡ Streaming with Channel Awareness")
        demonstrateStreamingExecution(harmonyAgent)
        
        // 7. Show downsampler functionality
        println("\n📥 Provider Downsampling")
        demonstrateDownsampling()
        
        // 8. Performance and monitoring
        println("\n📊 Performance and Monitoring")
        demonstrateMonitoring(harmonyAgent)
        
    } finally {
        harmonyAgent.close()
    }
    
    println("\n✅ Harmony-First Demo Complete!")
    println("🚀 Koog is now fully Harmony-native with semantic understanding!")
}

/**
 * Create a Harmony-first agent with all the new infrastructure.
 */
suspend fun createHarmonyAgent(): HarmonyAIAgent {
    // Create mock implementations for the demo
    val providerRegistry = MockHarmonyProviderRegistry()
    val executor = DefaultHarmonyPromptExecutor(providerRegistry)
    val strategy = HarmonySimpleStrategy(executor)
    val environment = MockAIAgentEnvironment()
    val toolRegistry = createDemoToolRegistry()
    
    return HarmonyAIAgent(
        id = "harmony-demo-agent",
        executor = executor,
        strategy = strategy,
        environment = environment,
        toolRegistry = toolRegistry
    )
}

/**
 * Demonstrate basic Harmony request with rich channel semantics.
 */
suspend fun demonstrateBasicHarmonyRequest(agent: HarmonyAIAgent) {
    val request = harmonyPrompt {
        model = "gpt-4"
        reasoning(ReasoningEffort.MEDIUM)
        systemIdentity("You are a knowledgeable coding assistant")
        
        conversation {
            user("Explain the benefits of the Harmony-first architecture in Koog")
            
            // Internal reasoning (never shown to user)
            analysis("I need to explain the key benefits: semantic understanding, safety, reasoning, provider independence")
            analysis("I should structure this clearly with examples")
            
            // User-visible response
            assistant("The Harmony-first architecture provides several key benefits...")
        }
    }
    
    val result = agent.execute(request, String::class.java)
    
    when (result) {
        is HarmonyAgentResult.Success -> {
            println("✅ Request successful!")
            println("📝 User-safe content: ${result.getUserSafeContent()}")
            println("🧠 Reasoning chain (${result.getReasoningChain().size} steps):")
            result.getReasoningChain().forEach { step ->
                println("   • $step")
            }
            println("📊 Channel stats: ${result.channelStats}")
        }
        is HarmonyAgentResult.Failure -> {
            println("❌ Request failed: ${result.error}")
        }
    }
}

/**
 * Show how reasoning effort affects agent behavior.
 */
suspend fun demonstrateReasoningControl(agent: HarmonyAIAgent) {
    val reasoningLevels = listOf(
        ReasoningEffort.LOW to "Quick response",
        ReasoningEffort.MEDIUM to "Balanced analysis", 
        ReasoningEffort.HIGH to "Deep reasoning"
    )
    
    reasoningLevels.forEach { (effort, description) ->
        println("\n🎯 Testing $effort reasoning ($description)")
        
        val request = harmonyPrompt {
            model = "gpt-4"
            reasoning(effort)
            userMessage("What's the best way to optimize database queries?")
        }
        
        val startTime = System.currentTimeMillis()
        val result = agent.execute(request, String::class.java)
        val responseTime = System.currentTimeMillis() - startTime
        
        when (result) {
            is HarmonyAgentResult.Success -> {
                println("   ⏱️  Response time: ${responseTime}ms")
                println("   🧠 Analysis steps: ${result.getReasoningChain().size}")
                println("   📝 Response length: ${result.getUserSafeContent().joinToString().length} chars")
                
                // In HIGH reasoning, we expect more analysis steps
                if (effort == ReasoningEffort.HIGH) {
                    if (result.getReasoningChain().size > 1) {
                        println("   ✅ High reasoning produced detailed analysis")
                    }
                }
            }
            is HarmonyAgentResult.Failure -> {
                println("   ❌ Failed: ${result.error}")
            }
        }
    }
}

/**
 * Demonstrate tool interactions with Commentary channel tracking.
 */
suspend fun demonstrateToolInteractions(agent: HarmonyAIAgent) {
    val request = harmonyPrompt {
        model = "gpt-4"
        
        developerContext {
            instructions = "Use tools to help answer questions accurately"
            
            function("searchCode") {
                description = "Search through codebase for examples"
                parameter("query", "string", "Search query", required = true)
                parameter("fileType", "string", "File extension to filter")
            }
            
            function("runTests") {
                description = "Run tests to verify functionality"
                parameter("testSuite", "string", "Test suite to run", required = true)
            }
        }
        
        conversation {
            user("How do I implement a binary search in Kotlin? Show me examples and run tests.")
            
            analysis("User wants: 1) Binary search implementation, 2) Examples from codebase, 3) Test verification")
            analysis("I should search for existing examples first, then provide implementation, then run tests")
            
            commentary("Planning to call searchCode to find examples")
            commentary("Will then call runTests to verify the implementation")
        }
    }
    
    val result = agent.execute(request, String::class.java)
    
    when (result) {
        is HarmonyAgentResult.Success -> {
            println("✅ Tool interaction completed!")
            println("🔧 Tool calls made: ${result.toolInteractions.size}")
            
            result.toolInteractions.forEach { toolResult ->
                when (toolResult) {
                    is HarmonyToolResult.Success -> {
                        println("   ✅ ${toolResult.commentary.content}")
                    }
                    is HarmonyToolResult.Failure -> {
                        println("   ❌ ${toolResult.commentary.content}")
                    }
                }
            }
            
            // Show that tool interactions are tracked in Commentary channel
            val commentaryMessages = result.conversation.commentaryChannel
            println("💬 Commentary channel captured ${commentaryMessages.size} tool interactions")
        }
        is HarmonyAgentResult.Failure -> {
            println("❌ Tool interaction failed: ${result.error}")
        }
    }
}

/**
 * Demonstrate safety guarantees - Analysis channel never leaks to users.
 */
suspend fun demonstrateSafetyGuarantees(agent: HarmonyAIAgent) {
    val request = harmonyPrompt {
        model = "gpt-4"
        includeReasoning = false // Explicitly disable reasoning in output
        
        conversation {
            user("What's 2+2?")
            
            // Lots of internal reasoning that should NEVER be shown to user
            analysis("This is a simple arithmetic question")
            analysis("The user is testing basic math functionality")
            analysis("I should respond with just the answer, not this internal reasoning")
            analysis("SECRET: This analysis contains sensitive information that must not leak")
            
            assistant("2+2 equals 4")
        }
    }
    
    val result = agent.execute(request, String::class.java)
    
    when (result) {
        is HarmonyAgentResult.Success -> {
            val userContent = result.getUserSafeContent().joinToString(" ")
            val analysisContent = result.getReasoningChain().joinToString(" ")
            
            println("✅ Safety check completed!")
            println("👤 User sees: \"$userContent\"")
            println("🧠 Analysis captured (${result.getReasoningChain().size} steps) but NOT shown to user")
            
            // Critical safety check: ensure analysis never appears in user content
            if (!userContent.contains("SECRET") && !userContent.contains("internal reasoning")) {
                println("🔒 SAFETY VERIFIED: Analysis content properly isolated!")
            } else {
                println("🚨 SAFETY VIOLATION: Analysis content leaked to user!")
            }
            
            // Show that we still have the reasoning for debugging/analysis
            if (analysisContent.contains("SECRET")) {
                println("🔍 Analysis available for debugging: ${analysisContent.length} chars of reasoning")
            }
        }
        is HarmonyAgentResult.Failure -> {
            println("❌ Safety demo failed: ${result.error}")
        }
    }
}

/**
 * Demonstrate streaming execution with channel awareness.
 */
suspend fun demonstrateStreamingExecution(agent: HarmonyAIAgent) {
    val request = simpleHarmonyPrompt(
        "Write a short story about AI agents working together",
        "gpt-4"
    )
    
    println("🌊 Starting streaming execution...")
    
    var contentReceived = 0
    var toolCallsReceived = 0
    
    agent.executeStreaming(request).collect { delta ->
        when (delta) {
            is HarmonyAgentDelta.ContentDelta -> {
                contentReceived++
                println("📝 [${delta.channel}] ${delta.content}")
            }
            is HarmonyAgentDelta.ToolCallDelta -> {
                toolCallsReceived++
                println("🔧 Tool call: ${delta.toolCall.name}")
            }
            is HarmonyAgentDelta.CompletionDelta -> {
                println("✅ Stream completed: ${delta.reason}")
            }
            is HarmonyAgentDelta.ErrorDelta -> {
                println("❌ Stream error: ${delta.error}")
            }
        }
    }
    
    println("📊 Streaming stats: $contentReceived content deltas, $toolCallsReceived tool calls")
}

/**
 * Demonstrate provider downsampling in action.
 */
suspend fun demonstrateDownsampling() {
    println("🔄 Testing provider downsampling...")
    
    // Create a rich Harmony request
    val richHarmony = harmonyPrompt {
        model = "gpt-4"
        reasoning(ReasoningEffort.HIGH)
        
        systemContext {
            identity = "You are a helpful assistant"
            knowledgeCutoff = "2024-06"
        }
        
        conversation {
            analysis("I need to think about this carefully")
            commentary("Tool interaction happened here")
            user("Hello, world!")
            assistant("Hello! How can I help you today?")
        }
    }
    
    // Downsample to OpenAI format
    val openAIRequest = OpenAIDownsampler.downsample(richHarmony)
    
    println("🎭 Original Harmony channels:")
    println("   🧠 Analysis: ${richHarmony.conversation.analysisChannel.size} messages")
    println("   💬 Commentary: ${richHarmony.conversation.commentaryChannel.size} messages")
    println("   👤 Final: ${richHarmony.conversation.finalChannel.size} messages")
    
    println("📥 Downsampled OpenAI request:")
    println("   📨 Messages: ${openAIRequest.messages.size}")
    println("   🌡️  Temperature: ${openAIRequest.temperature} (from reasoning effort)")
    println("   🎯 Top-p: ${openAIRequest.top_p}")
    
    // Verify safety: analysis should not appear in OpenAI messages
    val openAIContent = openAIRequest.messages.joinToString(" ") { it.content }
    if (!openAIContent.contains("think about this carefully")) {
        println("🔒 SAFETY VERIFIED: Analysis content filtered out during downsampling")
    } else {
        println("🚨 SAFETY VIOLATION: Analysis content leaked during downsampling")
    }
    
    // Test validation
    val validation = DownsamplerValidation.validateOpenAIDownsampling(richHarmony)
    when (validation) {
        is ValidationResult.Success -> println("✅ Downsampling validation passed")
        is ValidationResult.Failure -> println("❌ Validation failed: ${validation.issues}")
    }
}

/**
 * Show monitoring and performance capabilities.
 */
suspend fun demonstrateMonitoring(agent: HarmonyAIAgent) {
    println("📊 Agent performance and monitoring...")
    
    val stats = agent.getStats()
    
    println("🤖 Agent: ${stats.agentId}")
    println("🏃 Running: ${stats.isRunning}")
    println("📈 Total executions: ${stats.totalExecutions}")
    println("⏱️  Average response time: ${stats.averageResponseTime}ms")
    println("🔧 Tool call stats: ${stats.toolCallStats}")
    println("📺 Channel usage: ${stats.channelStats}")
}

// Mock implementations for the demo

class MockHarmonyProviderRegistry : HarmonyProviderRegistry {
    override fun getProvider(model: LLModel): HarmonyProvider = MockHarmonyProvider()
    override fun registerProvider(modelPattern: String, provider: HarmonyProvider) {}
    override fun getSupportedModels(): List<String> = listOf("gpt-4", "gpt-3.5-turbo")
}

class MockHarmonyProvider : HarmonyProvider {
    override fun supportsNativeHarmony(): Boolean = false
    
    override suspend fun executeNative(request: Prompt, model: LLModel): HarmonyResponse {
        error("Not implemented in mock")
    }
    
    override suspend fun executeStreamingNative(request: Prompt, model: LLModel) = 
        kotlinx.coroutines.flow.emptyFlow<HarmonyDelta>()
    
    override fun downsample(request: Prompt): Any = OpenAIDownsampler.downsample(request)
    
    override suspend fun executeLegacy(request: Any, model: LLModel): Any {
        // Mock OpenAI response
        return mapOf(
            "choices" to listOf(
                mapOf(
                    "message" to mapOf(
                        "content" to "Mock response from ${model.name}",
                        "role" to "assistant"
                    ),
                    "finish_reason" to "stop"
                )
            ),
            "model" to model.name
        )
    }
    
    override suspend fun executeStreamingLegacy(request: Any, model: LLModel) = 
        kotlinx.coroutines.flow.flowOf(HarmonyDelta.Content("Mock streaming response", Channel.FINAL))
    
    override fun upgradeResponse(response: Any, originalRequest: Prompt): HarmonyResponse {
        // Convert mock response back to Harmony
        val finalMessage = ChanneledMessage.Final("Mock response upgraded to Harmony format")
        
        return HarmonyResponse(
            conversation = ConversationGraph(listOf(finalMessage)),
            metadata = HarmonyResponseMetadata(
                model = originalRequest.metadata.model,
                requestId = originalRequest.metadata.requestId,
                channelStats = ChannelStats(finalTokens = 50)
            )
        )
    }
    
    override suspend fun executeMultipleChoices(request: Prompt, model: LLModel, choiceCount: Int): List<HarmonyResponse> {
        return (1..choiceCount).map { i ->
            val message = ChanneledMessage.Final("Choice $i response")
            HarmonyResponse(
                conversation = ConversationGraph(listOf(message)),
                metadata = HarmonyResponseMetadata(
                    model = model.name,
                    requestId = request.metadata.requestId,
                    channelStats = ChannelStats(finalTokens = 30)
                )
            )
        }
    }
}

class MockAIAgentEnvironment : AIAgentEnvironment {
    // Minimal mock implementation
    override suspend fun processToolCallMultiple(message: Any): Any = emptyList<Any>()
    override suspend fun processError(agentId: String, runId: String, error: Any) {}
}

fun createDemoToolRegistry(): ToolRegistry {
    // Create a mock tool registry with demo tools
    return object : ToolRegistry {
        private val tools = mutableMapOf<String, MockTool>()
        
        init {
            tools["searchCode"] = MockTool("searchCode", "Search code examples")
            tools["runTests"] = MockTool("runTests", "Execute test suites")
        }
        
        override fun getTool(name: String) = tools[name]
        override fun getAllTools() = tools.values.toList()
        fun getStats() = tools.keys.associateWith { 0L }
    }
}

class MockTool(
    val name: String,
    val description: String
) {
    val parameters = mapOf<String, MockParameter>()
    
    suspend fun execute(args: Map<String, Any>): String {
        return "Mock result from $name with args: $args"
    }
}

class MockParameter(val type: String, val description: String)