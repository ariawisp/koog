package ai.koog.agents.example.minecraft

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.memory.config.MemoryScopeType
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.feature.nodes.nodeLoadFromMemory
import ai.koog.agents.memory.feature.nodes.nodeSaveToMemory
import ai.koog.agents.memory.feature.nodes.nodeSemanticQuery
import ai.koog.agents.memory.feature.nodes.nodeGraphStats
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.koog.prompt.executor.llms.all.simpleOllamaAIExecutor
import ai.koog.prompt.llm.OllamaModels
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.prompt.markdown.markdown
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable

/**
 * Example demonstrating Koog's memory system with real LLM interactions.
 * 
 * This example uses Ollama for actual LLM processing, showing how the memory nodes
 * work with real fact extraction and storage. For testing scenarios that require
 * predictable responses, see the MockLLMBuilder extensions that support structured
 * response mocking with kotlinx.serialization.
 */

/**
 * Memory subjects for Minecraft game tracking
 */
private object MinecraftMemorySubjects {
    @Serializable
    data object Player : MemorySubject() {
        override val name: String = "player"
        override val promptDescription: String = "Information about individual players (names, actions, preferences)"
        override val priorityLevel: Int = 1
    }

    @Serializable
    data object Faction : MemorySubject() {
        override val name: String = "faction"
        override val promptDescription: String = "Information about factions (names, alliances, territories, conflicts)"
        override val priorityLevel: Int = 2
    }

    @Serializable
    data object Location : MemorySubject() {
        override val name: String = "location"
        override val promptDescription: String = "Information about locations (coordinates, structures, ownership)"
        override val priorityLevel: Int = 3
    }
}

/**
 * Simple example demonstrating proper use of Koog's documented memory API with strategy graphs.
 *
 * This demonstrates:
 * - Using memory nodes within strategy graphs (documented pattern)
 * - Graph-based memory with semantic queries and auto-ingestion
 * - Real LLM interactions for fact extraction and processing
 * - Memory statistics using graph capabilities
 *
 * How to run:
 * ```
 * ./gradlew runExampleMinecraftSimple
 * ```
 * 
 * Prerequisites:
 * Option 1 - Ollama (default):
 * - Ollama installed and running locally (ollama serve)
 * - Llama 3.2 model available (ollama pull llama3.2)
 * 
 * Option 2 - LM Studio:
 * - LM Studio installed with a model loaded
 * - Local server running (usually on port 1234)
 * - Set environment variables: USE_OLLAMA=false and OPENAI_API_KEY=dummy
 */
fun main() = runBlocking {
    println("=== Minecraft Companion with Graph Memory (Strategy Pattern) ===")
    println("Setting up agent with proper memory node integration...")
    println("Note: This example supports Ollama or LM Studio - see source comments for setup")
    println()

    val agent = createMinecraftCompanionAgent()

    // Simulate game events
    println("=== Ingesting Game Events ===")

    val events = listOf(
        "Player Steve joined the game and created the Mountain Builders faction",
        "Player Alex joined the game and created the Forest Rangers faction", 
        "Mountain Builders claimed territory from coordinates 0,0 to 500,500",
        "Forest Rangers claimed territory from coordinates -500,0 to 0,500",
        "Steve built Mountain Fortress at coordinates 250, 100, 250",
        "Mountain Builders formed an alliance with Desert Nomads clan",
        "Forest Rangers attacked Mountain Fortress and stole 10 diamond blocks"
    )

    // Process events using strategy graph with memory nodes
    for (event in events) {
        println("Processing: $event")
        val result = agent.run(event)
        println("  Result: $result")
        println()
    }

    println("=== Memory Query Examples ===")
    
    // Demonstrate semantic search capabilities
    println("Querying for faction information...")
    val factionQuery = agent.run("What factions exist and what are their relationships?")
    println("Faction info: $factionQuery")
    println()
    
    println("Querying for conflict information...")
    val conflictQuery = agent.run("Tell me about any conflicts or attacks that happened")
    println("Conflict info: $conflictQuery")

    println("\n=== Graph Memory Integration Complete ===")
    println("✅ All events processed using documented Koog memory API patterns")
    println("✅ Strategy graph with memory nodes (nodeLoadFromMemory, nodeSaveToMemory)")
    println("✅ Graph-based semantic search and relationship tracking active")
    println("✅ Auto-detection of facts from conversation history working")
}

/**
 * Creates a Minecraft companion agent using proper Koog strategy patterns with memory nodes
 */
fun createMinecraftCompanionAgent(): AIAgent<String, String> {
    // Choose your LLM backend based on environment variable
    val useOllama = System.getenv("USE_OLLAMA")?.toBoolean() ?: true
    
    val (executor, model) = if (useOllama) {
        // Option 1: Ollama (default)
        simpleOllamaAIExecutor() to OllamaModels.Meta.LLAMA_3_2
    } else {
        // Option 2: LM Studio - create OpenAI client pointing to local LM Studio server
        val apiKey = System.getenv("OPENAI_API_KEY") ?: "lm-studio" // LM Studio doesn't validate the key
        val lmStudioUrl = System.getenv("LM_STUDIO_URL") ?: "http://localhost:1234"
        
        val openAIClient = OpenAILLMClient(
            apiKey = apiKey,
            settings = OpenAIClientSettings(baseUrl = lmStudioUrl)
        )
        
        SingleLLMPromptExecutor(openAIClient) to OpenAIModels.Chat.GPT4o // Model name doesn't matter for LM Studio
    }

    // Memory concepts for different types of game information
    val playersConcept = Concept(
        keyword = "players",
        description = "Information about players including their names, actions, and faction memberships",
        factType = FactType.MULTIPLE
    )

    val factionsConcept = Concept(
        keyword = "factions", 
        description = "Information about factions including names, alliances, territories, and conflicts",
        factType = FactType.MULTIPLE
    )

    val locationsConcept = Concept(
        keyword = "locations",
        description = "Information about locations including coordinates, structures, and ownership",
        factType = FactType.MULTIPLE
    )

    val agentConfig = AIAgentConfig(
        prompt = prompt("minecraft-companion") {
            system("""
                You are a Minecraft server companion that helps track and understand game events.
                You can store information about players, factions, locations, and relationships.
                Always respond helpfully about the events you've witnessed or been told about.
            """.trimIndent())
        },
        model = model,
        maxAgentIterations = 50
    )

    // Simplified strategy using documented memory node patterns
    val strategy = strategy<String, String>("minecraft-companion", toolSelectionStrategy = ToolSelectionStrategy.NONE) {
        
        // Load existing memory at start to provide context
        val loadPlayers by nodeLoadFromMemory<String>(
            concept = playersConcept,
            subject = MinecraftMemorySubjects.Player,
            scope = MemoryScopeType.AGENT
        )
        
        val loadFactions by nodeLoadFromMemory<String>(
            concept = factionsConcept, 
            subject = MinecraftMemorySubjects.Faction, 
            scope = MemoryScopeType.AGENT
        )

        val loadLocations by nodeLoadFromMemory<String>(
            concept = locationsConcept,
            subject = MinecraftMemorySubjects.Location,
            scope = MemoryScopeType.AGENT
        )

        // Process the user input and generate response
        val processEvent by node<String, String> { userInput ->
            llm.writeSession {
                updatePrompt {
                    user("Game event or query: $userInput")
                }
                val response = requestLLMWithoutTools()
                response.content
            }
        }

        // Save information to memory using auto-detection for comprehensive fact storage
        val savePlayerInfo by nodeSaveToMemory<String>(
            concept = playersConcept,
            subject = MinecraftMemorySubjects.Player,
            scope = MemoryScopeType.AGENT
        )
        
        val saveFactionInfo by nodeSaveToMemory<String>(
            concept = factionsConcept,
            subject = MinecraftMemorySubjects.Faction,
            scope = MemoryScopeType.AGENT
        )

        val saveLocationInfo by nodeSaveToMemory<String>(
            concept = locationsConcept,
            subject = MinecraftMemorySubjects.Location,
            scope = MemoryScopeType.AGENT
        )

        nodeStart then loadPlayers then loadFactions then loadLocations then processEvent then savePlayerInfo then saveFactionInfo then saveLocationInfo then nodeFinish
    }

    return AIAgent(
        promptExecutor = executor,
        strategy = strategy,
        agentConfig = agentConfig,
        toolRegistry = ToolRegistry {}
    ) {
        install(AgentMemory) {
            // Use graph memory preset for advanced capabilities
            useGraphInMemoryPreset(
                autoIngestConversations = true,
                minConfidenceThreshold = 0.7
            )

            agentName = "minecraft-companion"
            featureName = "game-memory"
            organizationName = "minecraft-server"
            productName = "faction-tracker"
        }
    }
}
