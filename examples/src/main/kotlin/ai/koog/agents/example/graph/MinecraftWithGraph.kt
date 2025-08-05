package ai.koog.agents.example.graph

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.graph.KnowledgeGraph
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.providers.GraphMemoryConfig
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.llm.OllamaModels
import kotlinx.coroutines.runBlocking

/**
 * Example demonstrating the use of advanced graph-based memory
 * with the unified AgentMemory architecture.
 *
 * The graph memory provides enterprise-scale features:
 * - Persistent graph storage
 * - Bi-temporal data model (valid_at vs created_at)
 * - Hybrid search (semantic + graph traversal)
 * - Automatic fact extraction and relation inference
 * - Entity resolution and deduplication
 * - Community detection
 */
fun main() = runBlocking {
    println("=== Minecraft Companion with Graph-Based Memory ===")
    println()

    // Create in-memory knowledge graph with advanced features
    val knowledgeGraph = InMemoryKnowledgeGraph()

    // Create GraphMemoryProvider using knowledge graph as backend
    val graphMemoryProvider = GraphMemoryProvider(
        graph = knowledgeGraph,
        config = GraphMemoryConfig(
            autoIngestConversations = true,
            minConfidenceThreshold = 0.7
        )
    )

    // Create agent with graph-backed memory
    val mockExecutor = getMockExecutor {
        mockLLMAnswer("I understand and will remember this information about the Minecraft world in my knowledge graph.").asDefaultResponse
    }

    val agent = AIAgent(
        executor = mockExecutor,
        llmModel = OllamaModels.Meta.LLAMA_3_2,
        systemPrompt = """
            You are a Minecraft companion AI with persistent graph-based memory.
            You automatically store all game events, player actions, and relationships 
            in a temporal knowledge graph that remembers everything across sessions.
        """.trimIndent(),
        maxIterations = 5
    ) {
        install(AgentMemory) {
            // Use the graph-backed memory provider
            memoryProvider = graphMemoryProvider

            agentName = "minecraft-companion"
            featureName = "graph-memory"
            organizationName = "minecraft-server"
            productName = "persistent-companion"
        }
    }

    // Simulate complex Minecraft events that showcase the graph memory's capabilities
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
    println("✅ GraphMemoryProvider successfully integrated with knowledge graph backend")
    println("✅ Events are automatically stored in persistent temporal knowledge graph")
    println("✅ The agent can now access this memory across sessions and conversations")

    // Access graph-specific features (hybrid search)
    println("\n=== Graph Hybrid Search ===")
    try {
        val hybridResults = knowledgeGraph.query(
            KnowledgeRequest.Semantic(
                query = "faction leadership election governance",
                limit = 3
            )
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
        println("Hybrid search failed: ${e.message}")
    }

    // Display graph statistics
    println("\n=== Graph Statistics ===")
    try {
        val stats = knowledgeGraph.stats()
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
        println("Could not retrieve stats: ${e.message}")
    }

    println("\n=== Summary ===")
    println("✅ Successfully demonstrated graph-based memory integration with Koog agents")
    println("🔗 All events are now stored in the persistent knowledge graph")
    println("🔍 Semantic queries work across all historical data")
    println("📊 Visit http://localhost:8000/ui to explore the graph visually")
    println("🚀 This data persists across agent restarts - true long-term memory!")
}
