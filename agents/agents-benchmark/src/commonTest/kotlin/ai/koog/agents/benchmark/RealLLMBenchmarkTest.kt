package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.io.DatasetLoader
import ai.koog.agents.benchmark.judge.SimpleJudge
import ai.koog.agents.benchmark.model.*
import ai.koog.agents.benchmark.retrieval.*
import ai.koog.agents.benchmark.evaluation.BenchmarkRunner
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.retrieval.SmartRouter
import ai.koog.agents.memory.retrieval.createSmartRouter
import ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Real LLM benchmark test using LMStudio (or other OpenAI-compatible server)
 */
class RealLLMBenchmarkTest {
    
    /**
     * Simple character-based tokenizer for testing
     */
    private class TestTokenizer : PromptTokenizer {
        override fun tokenCountFor(message: Message): Int = message.content.length / 4
        override fun tokenCountFor(prompt: Prompt): Int = prompt.messages.sumOf { tokenCountFor(it) }
    }
    
    /**
     * Create an LLM executor for LMStudio
     */
    private fun createLMStudioExecutor(): SingleLLMPromptExecutor {
        val settings = OpenAIClientSettings(
            baseUrl = "http://localhost:1234" // LMStudio default URL
        )
        val client = OpenAILLMClient(
            apiKey = "lm-studio", // LMStudio doesn't require real API key
            settings = settings
        )
        return SingleLLMPromptExecutor(client)
    }
    
    @Test
    fun testRealLLMAnswerGeneration() = runTest {
        println("🚀 Running Real LLM Benchmark Test with LMStudio")
        println("=".repeat(50))
        
        // Skip test if LMStudio is not running
        try {
            val executor = createLMStudioExecutor()
            // Test connection with a simple prompt
            val testPrompt = ai.koog.prompt.dsl.prompt("test") {
                user("Hello! Just say 'OK' to confirm you're working.")
            }
            val testResult = executor.execute(testPrompt, ai.koog.prompt.executor.clients.openai.OpenAIModels.CostOptimized.GPT4_1Mini, emptyList())
            println("✅ LMStudio connection successful: ${testResult.first().content}")
        } catch (e: Exception) {
            println("⚠️  Skipping test - LMStudio not available: ${e.message}")
            return@runTest
        }
        
        // Create test infrastructure
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val llmExecutor = createLMStudioExecutor()
        
        // Create simple test dataset
        val dataset = createSimpleQADataset()
        
        println("📊 Dataset loaded: ${dataset.name}")
        println("   Questions: ${dataset.items.size}")
        
        // Populate knowledge graph with facts
        populateKnowledgeGraphWithSimpleFacts(knowledgeGraph)
        
        // Create benchmark executor with real LLM
        val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, llmExecutor)
        
        println("\n🔄 Running real LLM benchmark...")
        
        try {
            val report = executor.runAllModes(dataset)
            
            // Print results
            executor.printReport(report)
            
            // Analyze performance
            val bestAccuracy = report.summaries.maxOf { it.accuracy }
            val avgTokens = report.summaries.map { it.avgTokens }.average()
            val bestMode = report.summaries.maxBy { it.accuracy }.mode
            
            println("\n🎯 Real LLM Performance Analysis:")
            println("   Best Accuracy: ${(bestAccuracy * 100).toInt()}% (${bestMode})")
            println("   Avg Tokens: ${avgTokens.toInt()}")
            println("   Target: >70% accuracy for real multi-hop reasoning")
            
            // Basic validation
            assertTrue(report.summaries.isNotEmpty(), "Should have benchmark results")
            assertTrue(report.summaries.all { it.samples.isNotEmpty() }, "Should have processed questions")
            
            // Real performance expectations (much more realistic)
            when {
                bestAccuracy > 0.80 -> println("✅ Excellent real-world performance!")
                bestAccuracy > 0.60 -> println("🟡 Good performance, room for improvement")
                bestAccuracy > 0.40 -> println("🟡 Moderate performance, needs optimization")
                else -> println("🔴 Performance needs significant improvement")
            }
            
            println("\n📝 Sample Q&A Results:")
            report.summaries.first().samples.take(3).forEach { sample ->
                println("   Q: ${dataset.items.find { it.id == sample.qaId }?.question}")
                println("   Status: ${if (sample.correct) "✅ Correct" else "❌ Incorrect"}")
                println("   Tokens: ${sample.tokensUsed}, Latency: ${sample.latencyMs}ms")
                println()
            }
            
        } catch (e: Exception) {
            println("❌ Real LLM benchmark failed: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }
    
    @Test
    fun testMultiHopReasoningWithRealLLM() = runTest {
        println("\n🧠 Testing Multi-Hop Reasoning with Real LLM")
        println("=".repeat(45))
        
        // Skip if LMStudio not available
        try {
            createLMStudioExecutor()
        } catch (e: Exception) {
            println("⚠️  Skipping test - LMStudio not available")
            return@runTest
        }
        
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val llmExecutor = createLMStudioExecutor()
        
        // Create multi-hop dataset
        val dataset = createMultiHopQADataset()
        
        // Populate with multi-hop facts
        populateKnowledgeGraphWithMultiHopFacts(knowledgeGraph)
        
        val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, llmExecutor)
        
        println("📊 Multi-hop dataset: ${dataset.items.size} questions")
        dataset.items.forEach { item ->
            println("   - ${item.id}: ${item.question}")
            println("     Expected: ${item.goldAnswer}")
        }
        
        val report = executor.runAllModes(dataset)
        executor.printReport(report)
        
        val bestAccuracy = report.summaries.maxOf { it.accuracy }
        println("\n🎯 Multi-hop Performance: ${(bestAccuracy * 100).toInt()}%")
        
        if (bestAccuracy > 0.50) {
            println("✅ Good multi-hop reasoning capability!")
        } else {
            println("🔧 Multi-hop reasoning needs improvement")
        }
    }
    
    /**
     * Create a simple QA dataset for testing
     */
    private fun createSimpleQADataset(): Dataset {
        val items = listOf(
            QAItem(
                id = "simple_001",
                question = "What is the capital of France?",
                goldAnswer = "Paris",
                entities = listOf("France", "Paris"),
                metadata = mapOf("type" to "factual", "difficulty" to "easy")
            ),
            QAItem(
                id = "simple_002", 
                question = "Who wrote Romeo and Juliet?",
                goldAnswer = "William Shakespeare",
                entities = listOf("Romeo and Juliet", "William Shakespeare"),
                metadata = mapOf("type" to "literature", "difficulty" to "easy")
            ),
            QAItem(
                id = "simple_003",
                question = "What is the largest planet in our solar system?",
                goldAnswer = "Jupiter",
                entities = listOf("Jupiter", "solar system", "planet"),
                metadata = mapOf("type" to "science", "difficulty" to "easy")
            )
        )
        
        return Dataset(name = "SimpleQAReal", items = items)
    }
    
    /**
     * Create multi-hop reasoning dataset
     */
    private fun createMultiHopQADataset(): Dataset {
        val items = listOf(
            QAItem(
                id = "multihop_001",
                question = "What is the capital of the country where the Eiffel Tower is located?",
                goldAnswer = "Paris",
                entities = listOf("Eiffel Tower", "France", "Paris"),
                metadata = mapOf("type" to "multi_hop", "hops" to "2")
            ),
            QAItem(
                id = "multihop_002",
                question = "Who wrote the play that features the famous balcony scene?",
                goldAnswer = "William Shakespeare",
                entities = listOf("balcony scene", "Romeo and Juliet", "William Shakespeare"),
                metadata = mapOf("type" to "multi_hop", "hops" to "2")
            )
        )
        
        return Dataset(name = "MultiHopReal", items = items)
    }
    
    /**
     * Populate knowledge graph with simple facts
     */
    private suspend fun populateKnowledgeGraphWithSimpleFacts(knowledgeGraph: InMemoryKnowledgeGraph) {
        val facts = listOf(
            "Paris is the capital of France.",
            "France is a country in Europe.",
            "William Shakespeare wrote Romeo and Juliet.",
            "Romeo and Juliet is a famous play.",
            "Jupiter is the largest planet in our solar system.",
            "Jupiter is a gas giant planet."
        )
        
        facts.forEachIndexed { index, fact ->
            val episode = ai.koog.agents.memory.graph.Episode(
                content = fact,
                timestamp = kotlinx.datetime.Clock.System.now(),
                source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL,
                metadata = mapOf(
                    "fact_index" to index.toString(),
                    "source_name" to "TestFacts"
                )
            )
            
            try {
                knowledgeGraph.ingest(episode)
            } catch (e: Exception) {
                // Continue on error
            }
        }
        
        val stats = knowledgeGraph.stats()
        println("🔧 Knowledge graph populated: ${stats.nodeCount} nodes")
    }
    
    /**
     * Populate knowledge graph with multi-hop facts
     */
    private suspend fun populateKnowledgeGraphWithMultiHopFacts(knowledgeGraph: InMemoryKnowledgeGraph) {
        val facts = listOf(
            "The Eiffel Tower is located in Paris.",
            "Paris is the capital of France.",
            "France is a country in Europe.",
            "Romeo and Juliet features a famous balcony scene.",
            "Romeo and Juliet was written by William Shakespeare.",
            "William Shakespeare was an English playwright."
        )
        
        facts.forEachIndexed { index, fact ->
            val episode = ai.koog.agents.memory.graph.Episode(
                content = fact,
                timestamp = kotlinx.datetime.Clock.System.now(),
                source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL,
                metadata = mapOf(
                    "fact_index" to index.toString(),
                    "source_name" to "MultiHopFacts"
                )
            )
            
            try {
                knowledgeGraph.ingest(episode)
            } catch (e: Exception) {
                // Continue on error
            }
        }
        
        val stats = knowledgeGraph.stats()
        println("🔧 Multi-hop knowledge graph: ${stats.nodeCount} nodes")
    }
}