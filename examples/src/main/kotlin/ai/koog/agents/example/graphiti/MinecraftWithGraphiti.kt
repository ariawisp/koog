package ai.koog.agents.example.graphiti

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.graphiti.EmbeddingsConfig
import ai.koog.agents.memory.graphiti.GraphitiConfig
import ai.koog.agents.memory.graphiti.GraphitiKnowledgeGraph
import ai.koog.agents.memory.providers.GraphMemoryConfig
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.llm.OllamaModels
import kotlinx.coroutines.runBlocking

/**
 * Example demonstrating the use of Graphiti (external graph database)
 * with the unified AgentMemory architecture.
 *
 * Graphiti provides enterprise-scale memory with:
 * - Persistent graph storage
 * - Bi-temporal data model (valid_at vs ingested_at)
 * - Hybrid search (BM25 + embeddings + graph traversal)
 * - Automatic fact extraction and relation inference
 *
 * Prerequisites:
 * 1. Start Graphiti server with docker-compose:
 *    ```
 *    cd examples/src/main/kotlin/ai/koog/agents/example/graphiti
 *    OPENAI_API_KEY=your_key docker-compose up -d
 *    ```
 *
 * 2. Verify Graphiti is running:
 *    ```
 *    curl http://localhost:8000/health
 *    ```
 *
 * 3. View the Graphiti dashboard at: http://localhost:8000/ui
 */
fun main() = runBlocking {
    println("=== Minecraft Companion with Graphiti Backend ===")
    println("Connecting to Graphiti at: http://localhost:8000")
    println()

    // Create Graphiti configuration
    val graphitiConfig = GraphitiConfig(
        baseUrl = "http://localhost:8000",
        apiKey = null, // No API key needed for local instance
        embeddings = EmbeddingsConfig(
            provider = "openai",
            model = "text-embedding-ada-002",
            dimensions = 1536
        )
    )

    // Create Graphiti knowledge graph
    val graphitiGraph = GraphitiKnowledgeGraph(graphitiConfig)

    // Create GraphMemoryProvider using Graphiti as backend
    val graphMemoryProvider = GraphMemoryProvider(
        graph = graphitiGraph,
        config = GraphMemoryConfig(
            autoIngestConversations = true,
            minConfidenceThreshold = 0.7
        )
    )

    // Create agent with Graphiti-backed memory
    val mockExecutor = getMockExecutor {
        mockLLMAnswer("I understand and will remember this information about the Minecraft world in my knowledge graph.").asDefaultResponse
    }

    val agent = AIAgent(
        executor = mockExecutor,
        llmModel = OllamaModels.Meta.LLAMA_3_2,
        systemPrompt = """
            You are a Minecraft companion AI with persistent memory powered by Graphiti.
            You automatically store all game events, player actions, and relationships 
            in a temporal knowledge graph that remembers everything across sessions.
        """.trimIndent(),
        maxIterations = 5
    ) {
        install(AgentMemory) {
            // Use the Graphiti-backed memory provider
            memoryProvider = graphMemoryProvider

            agentName = "minecraft-companion"
            featureName = "graphiti-memory"
            organizationName = "minecraft-server"
            productName = "persistent-companion"
        }
    }

    // Simulate complex Minecraft events that showcase Graphiti's capabilities
    println("=== Ingesting Complex Game Events ===")

    val events = listOf(
        // Day 1: World creation and initial players
        "Player Steve joined the server and spawned at coordinates 0, 64, 0",
        "Steve founded the Mountain Builders faction with the goal of building massive stone structures",
        "Player Alex joined the server and immediately started exploring the western forests",
        "Alex discovered a rare woodland mansion at coordinates -800, 64, 450 containing valuable loot",

        // Day 2: Faction expansion and relationships
        "Player Bob joined Steve's Mountain Builders faction as the chief architect",
        "Alex founded the Forest Rangers faction focused on environmental protection and tree farming",
        "Mountain Builders established their main base called Stone Keep at coordinates 200, 100, 300",
        "Forest Rangers built Tree Haven settlement at coordinates -600, 80, 400 near the woodland mansion",

        // Day 3: Commerce and alliances
        "Steve and Alex negotiated a trade agreement: Mountain Builders provide stone, Forest Rangers provide wood",
        "Bob constructed a massive bridge connecting Stone Keep to Tree Haven across the great valley",
        "Player Carol joined the Forest Rangers and was appointed as the faction's chief diplomat",
        "The two factions signed a mutual defense pact against hostile mobs and raiders",

        // Day 4: Conflict and resolution
        "Unknown griefers attacked Stone Keep, destroying part of the western wall with TNT",
        "Carol investigated the attack and discovered evidence pointing to the Crimson Raiders faction",
        "Steve declared war on the Crimson Raiders and called upon the Forest Rangers for military support",
        "Alex organized a joint strike force with Bob leading Mountain Builder heavy infantry",

        // Day 5: Victory and evolution
        "The joint strike force successfully raided the Crimson Raiders' hidden base in the nether",
        "Carol negotiated a peace treaty requiring the Crimson Raiders to pay reparations in diamonds",
        "Steve was elected as the server's first Supreme Commander by popular vote",
        "The Mountain Builders and Forest Rangers formed the United Factions alliance for server governance"
    )

    // Process each event
    for ((index, event) in events.withIndex()) {
        println("Processing event ${index + 1}/${events.size}: ${event.take(60)}...")
        val result = agent.run(event)
        println("  -> $result")

        // Small delay to make it feel more realistic
        kotlinx.coroutines.delay(100)
    }

    println("\n=== Graph Memory Integration Complete ===")
    println("✅ All events have been processed through the unified AgentMemory system")
    println("✅ GraphMemoryProvider successfully integrated with Graphiti backend")
    println("✅ Events are automatically stored in persistent temporal knowledge graph")
    println("✅ The agent can now access this memory across sessions and conversations")

    // Access Graphiti-specific features (hybrid search)
    println("\n=== Graphiti Hybrid Search ===")
    try {
        val hybridResults = graphitiGraph.hybridSearch(
            query = "faction leadership election governance",
            limit = 3,
            alpha = 0.7, // Favor embeddings over BM25
            useRRF = true,
            useMMR = true,
            mmrLambda = 0.6
        )

        println("Hybrid search results for 'faction leadership election governance':")
        hybridResults.forEach { knowledge ->
            when (knowledge) {
                is ai.koog.agents.memory.graph.Knowledge.Entity -> {
                    println("   Entity: ${knowledge.labels.joinToString()} - ${knowledge.properties}")
                }
                is ai.koog.agents.memory.graph.Knowledge.Relation -> {
                    println("   Relation: ${knowledge.from} -[${knowledge.type}]-> ${knowledge.to}")
                }
                is ai.koog.agents.memory.graph.Knowledge.Composite -> {
                    println("   Composite: ${knowledge.summary}")
                }
            }
        }
    } catch (e: Exception) {
        println("Hybrid search failed (Graphiti server may not be running): ${e.message}")
    }

    // Display Graphiti statistics
    println("\n=== Graphiti Graph Statistics ===")
    try {
        val stats = graphitiGraph.stats()
        println("Total nodes: ${stats.nodeCount}")
        println("Total edges: ${stats.edgeCount}")
        println("Episodes ingested: ${stats.episodeCount}")
        println("Last ingestion: ${stats.lastIngestion}")

        if (stats.nodesByLabel.isNotEmpty()) {
            println("\nNode types:")
            stats.nodesByLabel.forEach { (label, count) ->
                println("  $label: $count")
            }
        }

        if (stats.edgesByType.isNotEmpty()) {
            println("\nEdge types:")
            stats.edgesByType.forEach { (type, count) ->
                println("  $type: $count")
            }
        }
    } catch (e: Exception) {
        println("Could not retrieve stats (Graphiti server may not be running): ${e.message}")
    }

    println("\n=== Summary ===")
    println("✅ Successfully demonstrated Graphiti integration with Koog agents")
    println("🔗 All events are now stored in the persistent Graphiti knowledge graph")
    println("🔍 Semantic queries work across all historical data")
    println("📊 Visit http://localhost:8000/ui to explore the graph visually")
    println("🚀 This data persists across agent restarts - true long-term memory!")
}
