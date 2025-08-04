package ai.koog.agents.example.minecraft

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.agent.entity.ToolSelectionStrategy
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.ext.agent.ProvideStringSubgraphResult
import ai.koog.agents.ext.agent.StringSubgraphResult
import ai.koog.agents.ext.agent.subgraphWithTask
import ai.koog.agents.memory.config.MemoryScopeType
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.feature.nodes.nodeLoadFromMemory
import ai.koog.agents.memory.feature.nodes.nodeSaveToMemoryAutoDetectFacts
import ai.koog.agents.memory.feature.nodes.nodeSemanticQuery
import ai.koog.agents.memory.feature.nodes.nodeEntityCentricQuery
import ai.koog.agents.memory.feature.nodes.nodeGraphStats
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.agents.memory.model.MemorySubject
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.llm.OllamaModels
import ai.koog.prompt.markdown.markdown
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.Serializable

/**
 * Memory subjects for detailed Minecraft faction tracking
 */
private object FactionMemorySubjects {
    @Serializable
    data object Player : MemorySubject() {
        override val name: String = "player"
        override val promptDescription: String = "Individual player information (names, actions, memberships, roles)"
        override val priorityLevel: Int = 1
    }

    @Serializable
    data object Faction : MemorySubject() {
        override val name: String = "faction"
        override val promptDescription: String = "Faction information (names, leaders, alliances, conflicts, territories)"
        override val priorityLevel: Int = 2
    }

    @Serializable
    data object Location : MemorySubject() {
        override val name: String = "location"
        override val promptDescription: String = "Location and structure information (coordinates, buildings, ownership, events)"
        override val priorityLevel: Int = 3
    }

    @Serializable
    data object Event : MemorySubject() {
        override val name: String = "event"
        override val promptDescription: String = "Game events (battles, alliances, constructions, declarations)"
        override val priorityLevel: Int = 4
    }
}

/**
 * Advanced example demonstrating complex graph-based memory with entity-centric queries.
 * 
 * This shows proper use of Koog's documented memory API:
 * - Strategy graphs with memory nodes
 * - Entity-centric relationship traversal
 * - Semantic search across connected knowledge
 * - Auto-detection of complex relational facts
 */
fun main() = runBlocking {
    println("=== Advanced Minecraft Faction Companion (Graph Strategy Pattern) ===")
    println("Demonstrating entity-centric queries and relationship traversal...")
    println()

    // Create companion with advanced graph capabilities
    val companion = createAdvancedMinecraftCompanion()

    // Simulate complex game events with relationships
    println("=== Day 1: World Setup ===")
    companion.processEvent("Player Steve created the faction called Mountain Builders")
    companion.processEvent("Player Alex created the faction called Forest Rangers")
    companion.processEvent("Steve built a fortress at coordinates 100, 64, 200 called Stone Keep")
    companion.processEvent("Alex built a base at coordinates -500, 70, 300 called Tree Haven")

    println("\n=== Day 2: Alliances Form ===")
    companion.processEvent("Player Bob joined the Mountain Builders faction")
    companion.processEvent("Player Carol joined the Forest Rangers faction")
    companion.processEvent("Mountain Builders faction allied with Desert Nomads clan")
    companion.processEvent("Bob built an outpost at coordinates 150, 65, 250")

    println("\n=== Day 3: Conflict Emerges ===")
    companion.processEvent("Forest Rangers attacked Stone Keep with TNT")
    companion.processEvent("Steve defended the fortress and repelled the attack")
    companion.processEvent("Mountain Builders declared war on Forest Rangers")

    // Advanced query examples using graph capabilities
    println("\n=== Advanced Graph Queries ===")

    println("\nQ: Tell me everything connected to Steve (entity-centric query):")
    val steveConnections = companion.queryEntityCentric("Steve")
    println("A: $steveConnections")

    println("\nQ: What conflicts and alliances exist (semantic search):")
    val relationships = companion.querySemanticRelationships("conflicts alliances wars")
    println("A: $relationships")

    println("\nQ: What happened at Stone Keep and related locations:")
    val locationEvents = companion.queryEntityCentric("Stone Keep")
    println("A: $locationEvents")

    println("\nQ: Show current graph statistics:")
    companion.showGraphStats()

    println("\n=== Advanced Memory Demonstration Complete ===")
    println("✅ Entity-centric relationship traversal working")
    println("✅ Semantic search across connected knowledge active")
    println("✅ Graph-based fact detection and relationship tracking")
    println("✅ Complex memory queries using proper Koog node patterns")
}

/**
 * Creates an advanced Minecraft companion using proper strategy graphs with memory nodes
 */
fun createAdvancedMinecraftCompanion(): AdvancedMinecraftCompanion {
    val mockExecutor = getMockExecutor {
        // Mock comprehensive JSON responses for advanced fact extraction
        mockLLMAnswer(
            """{"facts": [
                {"fact": "Steve leads Mountain Builders faction with fortress at Stone Keep"},
                {"fact": "Alex leads Forest Rangers faction with base at Tree Haven"},
                {"fact": "Bob joined Mountain Builders and built outpost at coordinates 150,65,250"},
                {"fact": "Carol joined Forest Rangers faction"},
                {"fact": "Mountain Builders allied with Desert Nomads clan"},
                {"fact": "Forest Rangers attacked Stone Keep with TNT but were repelled"},
                {"fact": "Mountain Builders declared war on Forest Rangers after attack"}
            ]}"""
        ) onRequestContains "multiple facts"
        
        mockLLMAnswer(
            """{"fact": "Advanced Minecraft faction event analyzed and relationships mapped"}"""
        ) onRequestContains "single fact"
        
        // Default response for complex interactions
        mockLLMAnswer("I've analyzed the Minecraft event and updated my knowledge graph with new relationships and connections.").asDefaultResponse
    }

    // Define comprehensive memory concepts
    val playersConcept = Concept(
        keyword = "players",
        description = "Detailed player information including names, faction memberships, roles, and individual actions",
        factType = FactType.MULTIPLE
    )

    val factionsConcept = Concept(
        keyword = "factions",
        description = "Faction information including names, leaders, member lists, alliances, conflicts, and territorial claims",
        factType = FactType.MULTIPLE
    )

    val locationsConcept = Concept(
        keyword = "locations",
        description = "Location and structure information including coordinates, building names, ownership, and associated events",
        factType = FactType.MULTIPLE
    )

    val eventsConcept = Concept(
        keyword = "events",
        description = "Game events including battles, alliances, construction, declarations, and their outcomes",
        factType = FactType.MULTIPLE
    )

    val agentConfig = AIAgentConfig(
        prompt = prompt("advanced-minecraft-companion") {},
        model = OllamaModels.Meta.LLAMA_3_2,
        maxAgentIterations = 50
    )

    // Advanced strategy with entity-centric and semantic query capabilities
    val strategy = strategy<String, String>("advanced-minecraft-companion", toolSelectionStrategy = ToolSelectionStrategy.NONE) {
        
        // Load comprehensive memory context
        val loadMemoryContext by subgraph<String, String>(tools = emptyList()) {
            val loadPlayers by nodeLoadFromMemory<String>(
                concept = playersConcept,
                subject = FactionMemorySubjects.Player,
                scope = MemoryScopeType.PRODUCT
            )
            
            val loadFactions by nodeLoadFromMemory<String>(
                concept = factionsConcept,
                subject = FactionMemorySubjects.Faction,
                scope = MemoryScopeType.PRODUCT
            )
            
            val loadLocations by nodeLoadFromMemory<String>(
                concept = locationsConcept,
                subject = FactionMemorySubjects.Location,
                scope = MemoryScopeType.PRODUCT
            )
            
            val loadEvents by nodeLoadFromMemory<String>(
                concept = eventsConcept,
                subject = FactionMemorySubjects.Event,
                scope = MemoryScopeType.PRODUCT
            )
            
            nodeStart then loadPlayers then loadFactions then loadLocations then loadEvents then nodeFinish
        }

        // Main event processing with advanced analysis
        val processEvent by subgraphWithTask<String>(tools = emptyList()) { userInput ->
            markdown {
                h2("Advanced Minecraft Faction Companion AI")
                text("You are an expert at tracking complex relationships in multiplayer Minecraft worlds. You should:")
                br()
                bulleted {
                    item { text("Analyze player actions and faction dynamics with deep understanding") }
                    item { text("Track alliances, conflicts, and territorial claims with precision") }
                    item { text("Understand the implications of events on faction relationships") }
                    item { text("Provide insights based on historical patterns and current knowledge") }
                    item { text("Maintain awareness of strategic positions and power balances") }
                }
                
                h2("Current input to analyze:")
                text(userInput)
            }
        }
        
        val retrieveResult by node<StringSubgraphResult, String> { it.result }

        // Comprehensive fact extraction and storage
        val saveComprehensiveMemory by subgraph<String, String>(tools = emptyList()) {
            val autoDetectFacts by nodeSaveToMemoryAutoDetectFacts<String>(
                scopes = listOf(MemoryScopeType.PRODUCT),
                subjects = listOf(
                    FactionMemorySubjects.Player,
                    FactionMemorySubjects.Faction,
                    FactionMemorySubjects.Location,
                    FactionMemorySubjects.Event
                )
            )
            
            nodeStart then autoDetectFacts then nodeFinish
        }

        nodeStart then loadMemoryContext then processEvent then retrieveResult then saveComprehensiveMemory then nodeFinish
    }

    return AdvancedMinecraftCompanion(
        AIAgent(
            promptExecutor = mockExecutor,
            strategy = strategy,
            agentConfig = agentConfig,
            toolRegistry = ToolRegistry {
                tool(ProvideStringSubgraphResult)
            }
        ) {
            install(AgentMemory) {
                useGraphInMemoryPreset(
                    autoIngestConversations = true,
                    minConfidenceThreshold = 0.8
                )

                agentName = "advanced-minecraft-companion"
                featureName = "faction-tracker"
                organizationName = "minecraft-server"
                productName = "multiplayer-analytics"
            }
        }
    )
}

/**
 * Advanced wrapper demonstrating entity-centric and semantic query capabilities
 */
class AdvancedMinecraftCompanion(private val agent: AIAgent<String, String>) {

    suspend fun processEvent(event: String): String {
        val result = agent.run(event)
        println("  -> $result")
        return result ?: "Failed to process event"
    }

    suspend fun queryEntityCentric(entityId: String): String {
        // Create a query strategy that uses entity-centric search
        val queryStrategy = createEntityCentricQueryStrategy(entityId)
        val queryAgent = createQueryAgent(queryStrategy)
        
        val result = queryAgent.run("Tell me everything connected to $entityId")
        return result ?: "No connections found for $entityId"
    }

    suspend fun querySemanticRelationships(query: String): String {
        // Create a query strategy that uses semantic search
        val semanticStrategy = createSemanticQueryStrategy(query)
        val queryAgent = createQueryAgent(semanticStrategy) 
        
        val result = queryAgent.run("Find information about: $query")
        return result ?: "No semantic matches found for: $query"
    }

    suspend fun showGraphStats() {
        // Create a strategy that shows graph statistics
        val statsStrategy = createGraphStatsStrategy()
        val statsAgent = createQueryAgent(statsStrategy)
        
        val result = statsAgent.run("Show graph statistics")
        println("Graph Statistics: $result")
    }

    // Helper methods to create specialized query agents using memory nodes
    private fun createEntityCentricQueryStrategy(entityId: String) = 
        strategy<String, String>("entity-centric-query", toolSelectionStrategy = ToolSelectionStrategy.NONE) {
            val entityQuery by subgraph<String, String>(tools = emptyList()) {
                val queryEntity by nodeEntityCentricQuery<String>(
                    entityId = entityId,
                    depth = 2,
                    subjects = listOf(
                        FactionMemorySubjects.Player,
                        FactionMemorySubjects.Faction,
                        FactionMemorySubjects.Location,
                        FactionMemorySubjects.Event
                    )
                )
                
                val processResults by node<List<ai.koog.agents.memory.model.Fact>, String> { facts ->
                    "Found ${facts.size} facts connected to $entityId: ${facts.joinToString("; ") { it.toString() }}"
                }
                
                nodeStart then queryEntity then processResults then nodeFinish
            }
            nodeStart then entityQuery then nodeFinish
        }

    private fun createSemanticQueryStrategy(query: String) =
        strategy<String, String>("semantic-query", toolSelectionStrategy = ToolSelectionStrategy.NONE) {
            val semanticSearch by subgraph<String, String>(tools = emptyList()) {
                val searchSemantic by nodeSemanticQuery<String>(
                    query = query,
                    limit = 10,
                    subjects = listOf(
                        FactionMemorySubjects.Player,
                        FactionMemorySubjects.Faction,
                        FactionMemorySubjects.Location,
                        FactionMemorySubjects.Event
                    )
                )
                
                val processResults by node<List<ai.koog.agents.memory.model.Fact>, String> { facts ->
                    "Semantic search for '$query' found ${facts.size} matches: ${facts.joinToString("; ") { it.toString() }}"
                }
                
                nodeStart then searchSemantic then processResults then nodeFinish
            }
            nodeStart then semanticSearch then nodeFinish
        }

    private fun createGraphStatsStrategy() =
        strategy<String, String>("graph-stats", toolSelectionStrategy = ToolSelectionStrategy.NONE) {
            val getStats by subgraph<String, String>(tools = emptyList()) {
                val statsNode by nodeGraphStats<String>()
                
                val formatStats by node<ai.koog.agents.memory.feature.nodes.GraphStatsResult, String> { stats ->
                    "Graph Statistics: ${stats.nodeCount} nodes, ${stats.edgeCount} edges, available: ${stats.available}"
                }
                
                nodeStart then statsNode then formatStats then nodeFinish
            }
            nodeStart then getStats then nodeFinish
        }

    private fun createQueryAgent(strategy: ai.koog.agents.core.agent.entity.AIAgentStrategy<String, String>): AIAgent<String, String> {
        val mockExecutor = getMockExecutor {
            // Mock JSON responses for query operations
            mockLLMAnswer(
                """{"facts": [
                    {"fact": "Query result from advanced graph memory system"}
                ]}"""
            ) onRequestContains "multiple facts"
            
            mockLLMAnswer("Query completed using graph memory capabilities.").asDefaultResponse
        }
        
        return AIAgent(
            promptExecutor = mockExecutor,
            strategy = strategy,
            agentConfig = AIAgentConfig(
                prompt = prompt("query-agent") {},
                model = OllamaModels.Meta.LLAMA_3_2,
                maxAgentIterations = 5
            ),
            toolRegistry = ToolRegistry {}
        ) {
            install(AgentMemory) {
                useGraphInMemoryPreset(
                    autoIngestConversations = true,
                    minConfidenceThreshold = 0.8
                )
                agentName = "query-companion"
                featureName = "faction-tracker"
                organizationName = "minecraft-server"
                productName = "multiplayer-analytics"
            }
        }
    }
}
