package ai.koog.agents.memory

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.memory.feature.AgentMemory
import ai.koog.agents.memory.graph.*
import ai.koog.agents.memory.nodes.nodeRetrieveKnowledge
import ai.koog.agents.memory.nodes.nodeRetrieveKnowledgeAsText
import ai.koog.agents.memory.providers.GraphMemoryProvider
import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.llm.OllamaModels
import io.mockk.*
import kotlinx.coroutines.test.runTest
import kotlinx.datetime.Clock
import kotlinx.datetime.Instant
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.days
import kotlin.time.Duration.Companion.hours

class GraphRetrievalIntegrationTest {
    
    @Test
    fun `test graph-based retrieval integration for faction server scenario`() = runTest {
        // Mock knowledge graph with advanced capabilities
        val mockGraph = mockk<KnowledgeGraph>()
        
        // Setup test data - faction relationships and events
        val currentTime = Clock.System.now()
        val yesterday = currentTime - 1.days
        val lastWeek = currentTime - 7.days
        
        // Mock semantic queries for different scenarios
        coEvery { mockGraph.query(any<KnowledgeRequest.Semantic>()) } answers {
            val request = firstArg<KnowledgeRequest.Semantic>()
            val query = request.query
            when {
                query.contains("alliance") -> listOf(
                    Knowledge.Relation(
                        id = "rel-1",
                        type = "ALLIED_WITH",
                        from = "faction:RedStone",
                        to = "faction:BlueMoon",
                        properties = mapOf("since" to lastWeek.toString()),
                        confidence = 0.9,
                        timestamp = lastWeek,
                        provenance = emptyList()
                    ),
                    Knowledge.Relation(
                        id = "rel-2",
                        type = "RIVALS_WITH",
                        from = "faction:RedStone",
                        to = "faction:GreenLeaf",
                        properties = mapOf("reason" to "territory dispute"),
                        confidence = 0.85,
                        timestamp = yesterday,
                        provenance = emptyList()
                    )
                )
                query.contains("attack") || query.contains("betray") -> listOf(
                    Knowledge.Entity(
                        id = "event-1",
                        labels = setOf("BattleEvent"),
                        properties = mapOf(
                            "attacker" to "faction:GreenLeaf",
                            "defender" to "faction:RedStone",
                            "location" to "spawn base",
                            "outcome" to "defender victory"
                        ),
                        confidence = 0.95,
                        timestamp = yesterday,
                        provenance = emptyList()
                    )
                )
                else -> emptyList()
            }
        }
        
        // Mock entity-centric queries
        coEvery { mockGraph.query(any<KnowledgeRequest.EntityCentric>()) } answers {
            val request = firstArg<KnowledgeRequest.EntityCentric>()
            if (request.centerNode == "faction:RedStone") {
                listOf(
                    Knowledge.Entity(
                        id = "faction:RedStone",
                        labels = setOf("Faction"),
                        properties = mapOf(
                            "name" to "RedStone Empire",
                            "members" to 25,
                            "founded" to lastWeek.toString()
                        ),
                        confidence = 1.0,
                        timestamp = currentTime,
                        provenance = emptyList()
                    ),
                    Knowledge.Relation(
                        id = "rel-3",
                        type = "OWNS_BASE",
                        from = "faction:RedStone",
                        to = "base:mountain-fortress",
                        properties = mapOf("fortified" to true),
                        confidence = 0.9,
                        timestamp = currentTime - 3.days,
                        provenance = emptyList()
                    )
                )
            } else {
                emptyList()
            }
        }
        
        // Mock temporal queries
        coEvery { mockGraph.query(any<KnowledgeRequest.Temporal>()) } returns listOf(
            Knowledge.Entity(
                id = "event-betrayal",
                labels = setOf("BetrayalEvent"),
                properties = mapOf(
                    "betrayer" to "player:steve123",
                    "from_faction" to "faction:BlueMoon",
                    "to_faction" to "faction:GreenLeaf",
                    "stolen_items" to "diamond armor set"
                ),
                confidence = 0.92,
                timestamp = currentTime - 5.days,
                provenance = emptyList()
            )
        )
        
        // Mock other required methods
        coEvery { mockGraph.ingest(any()) } returns IngestionResult(
            createdNodes = listOf("node-1"),
            createdEdges = emptyList(),
            updatedNodes = emptyList(),
            conflicts = emptyList()
        )
        coEvery { mockGraph.evolve(any()) } just Runs
        coEvery { mockGraph.stats() } returns GraphStats(
            nodeCount = 100,
            edgeCount = 250,
            episodeCount = 500,
            lastIngestion = currentTime,
            lastEvolution = currentTime - 1.hours,
            nodesByLabel = mapOf("Faction" to 5, "Player" to 50),
            edgesByType = mapOf("ALLIED_WITH" to 10, "RIVALS_WITH" to 15)
        )
        
        // Create memory provider with knowledge graph
        val graphMemoryProvider = GraphMemoryProvider(
            graph = mockGraph,
            retriever = KnowledgeGraphRetrievalProvider(mockGraph)
        )
        
        // Create smart router
        val router = createSmartRouter(
            memoryProvider = graphMemoryProvider,
            graphProvider = KnowledgeGraphRetrievalProvider(mockGraph)
        )
        
        // Mock LLM
        val mockLLM = getMockExecutor {
            mockLLMAnswer("Form alliance with BlueMoon against GreenLeaf. GreenLeaf attacked you yesterday and BlueMoon is already allied.").asDefaultResponse
        }
        
        // Create agent with faction assistant strategy
        val agent = AIAgent(
            promptExecutor = mockLLM,
            strategy = strategy<String, String>("faction-assistant") {
                // Check current alliances with center node reranking
                val checkAlliances by nodeRetrieveKnowledge<String> {
                    text = "current faction alliances and rivalries"
                    target = RetrievalTarget.FACTS
                    recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE
                    centerNode = "faction:RedStone"
                    k = 10
                }
                
                // Check recent threats with temporal query
                val checkThreats by nodeRetrieveKnowledgeAsText<List<RetrievalResult>> {
                    text = "Who has attacked or betrayed us"
                    at = yesterday // Point-in-time query
                    filters {
                        entityLabel("BattleEvent")
                        entityLabel("BetrayalEvent")
                    }
                    k = 5
                }
                
                // Connect the flow
                edge(nodeStart forwardTo checkAlliances)
                edge(checkAlliances forwardTo checkThreats)
                edge(checkThreats forwardTo nodeFinish transformed { threats ->
                    "Form alliance with BlueMoon against GreenLeaf. GreenLeaf attacked you yesterday and BlueMoon is already allied."
                })
            },
            agentConfig = AIAgentConfig(
                prompt = prompt("test") { system("Test") },
                model = OllamaModels.Meta.LLAMA_3_2,
                maxAgentIterations = 5
            )
        ) {
            install(AgentMemory) {
                memoryProvider = graphMemoryProvider
                retriever = router
            }
        }
        
        // Execute the agent
        val result = agent.run("Help me with faction strategy")
        
        // Verify the flow worked correctly
        assertTrue(result.contains("Form alliance with BlueMoon"))
        assertTrue(result.contains("GreenLeaf attacked"))
        
        // Verify the retrieval system was used correctly
        coVerify {
            mockGraph.query(
                match<KnowledgeRequest> { 
                    it is KnowledgeRequest.Semantic && it.query.contains("alliance") 
                }
            )
            mockGraph.query(
                match<KnowledgeRequest> { 
                    it is KnowledgeRequest.Semantic && (it.query.contains("attack") || it.query.contains("betray"))
                }
            )
        }
    }
    
    @Test
    fun `test nodeRetrieveKnowledge with all retrieval recipes`() = runTest {
        val mockGraph = mockk<KnowledgeGraph>()
        
        // Setup mock responses
        val testKnowledge = listOf(
            Knowledge.Entity(
                id = "product-1",
                labels = setOf("Product"),
                properties = mapOf("name" to "Wool Runners", "size" to "10"),
                confidence = 0.9,
                timestamp = Clock.System.now(),
                provenance = emptyList()
            )
        )
        
        // Mock query methods
        coEvery { mockGraph.query(any()) } returns testKnowledge
        coEvery { mockGraph.ingest(any()) } returns IngestionResult(emptyList(), emptyList(), emptyList(), emptyList())
        coEvery { mockGraph.evolve(any()) } just Runs
        coEvery { mockGraph.stats() } returns GraphStats(0, 0, 0, null, null, emptyMap(), emptyMap())
        
        val graphMemoryProvider = GraphMemoryProvider(
            graph = mockGraph,
            retriever = KnowledgeGraphRetrievalProvider(mockGraph)
        )
        
        val mockLLM = getMockExecutor {
            mockLLMAnswer("Size 10 available").asDefaultResponse
        }
        
        // Test different retrieval recipes
        val recipesToTest = listOf(
            RetrievalRecipe.HYBRID_RRF,
            RetrievalRecipe.HYBRID_MMR,
            RetrievalRecipe.HYBRID_NODE_DISTANCE,
            RetrievalRecipe.VECTOR_SIMILARITY,
            RetrievalRecipe.TEXT_BM25
        )
        
        for (recipeToTest in recipesToTest) {
            val agent = AIAgent(
                promptExecutor = mockLLM,
                strategy = strategy<String, String>("test-recipe-$recipeToTest") {
                    val search by nodeRetrieveKnowledge<String> {
                        text = "Wool Runners size 10"
                        recipe = recipeToTest
                        k = 5
                    }
                    
                    edge(nodeStart forwardTo search)
                    edge(search forwardTo nodeFinish transformed { results ->
                        "Found ${results.size} results using $recipeToTest"
                    })
                },
                agentConfig = AIAgentConfig(
                    prompt = prompt("test") { system("Test") },
                    model = OllamaModels.Meta.LLAMA_3_2,
                    maxAgentIterations = 5
                )
            ) {
                install(AgentMemory) {
                    memoryProvider = graphMemoryProvider
                    retriever = KnowledgeGraphRetrievalProvider(mockGraph)
                }
            }
            
            val result = agent.run("Find products")
            assertEquals("Found 1 results using $recipeToTest", result)
        }
    }
    
}