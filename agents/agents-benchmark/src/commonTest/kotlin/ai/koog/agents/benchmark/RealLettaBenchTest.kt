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
 * Real LettaBench performance test using LMStudio
 */
class RealLettaBenchTest {
    
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
    fun testRealLettaBenchPerformance() = runTest {
        println("🚀 Running Real LettaBench Performance Test with LMStudio")
        println("=".repeat(55))
        
        // Skip test if LMStudio is not running
        try {
            val testExecutor = createLMStudioExecutor()
            val testPrompt = ai.koog.prompt.dsl.prompt("test") {
                user("Hello! Just say 'OK' to confirm you're working.")
            }
            val testResult = testExecutor.execute(testPrompt, ai.koog.prompt.executor.clients.openai.OpenAIModels.CostOptimized.GPT4_1Mini, emptyList())
            println("✅ LMStudio connection successful: ${testResult.first().content}")
        } catch (e: Exception) {
            println("⚠️  Skipping test - LMStudio not available: ${e.message}")
            return@runTest
        }
        
        // Load the real LettaBench dataset
        val lettaBenchContent = loadLettaBenchFile()
        val dataset = DatasetLoader.loadLettaBench(lettaBenchContent, "RealLettaBench")
        
        println("📊 Dataset loaded: ${dataset.name}")
        println("   Questions: ${dataset.items.size}")
        println("   Sample question: ${dataset.items.first().question.take(80)}...")
        
        // Create test infrastructure with real LLM
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val llmExecutor = createLMStudioExecutor()
        
        // Populate knowledge graph with facts from the dataset
        populateKnowledgeGraph(dataset, knowledgeGraph)
        
        // Create benchmark executor with real LLM
        val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, llmExecutor)
        
        // Run benchmarks on a small subset first (LLM calls are expensive)
        val testSubset = dataset.items.take(5)
        val testDataset = Dataset(name = "RealLettaBench5", items = testSubset)
        
        println("\n🔄 Running real LLM benchmark on ${testSubset.size} questions...")
        println("   (Using subset due to LLM call cost)")
        
        try {
            val report = executor.runAllModes(testDataset)
            
            // Print results
            executor.printReport(report)
            
            // Analyze performance
            val bestAccuracy = report.summaries.maxOf { it.accuracy }
            val avgTokens = report.summaries.map { it.avgTokens }.average()
            val bestMode = report.summaries.maxBy { it.accuracy }.mode
            
            println("\n🎯 Real LettaBench Performance Analysis:")
            println("   Best Accuracy: ${(bestAccuracy * 100).toInt()}% (${bestMode})")
            println("   Avg Tokens: ${avgTokens.toInt()}")
            println("   Target: >70% accuracy for complex multi-hop reasoning")
            
            // Detailed sample analysis
            println("\n📝 Sample Q&A Analysis:")
            val bestSummary = report.summaries.maxBy { it.accuracy }
            bestSummary.samples.take(3).forEach { sample ->
                val qaItem = dataset.items.find { it.id == sample.qaId }
                if (qaItem != null) {
                    println("   Q: ${qaItem.question}")
                    println("   Expected: ${qaItem.goldAnswer}")
                    println("   Status: ${if (sample.correct) "✅ Correct" else "❌ Incorrect"}")
                    println("   Tokens: ${sample.tokensUsed}, Latency: ${sample.latencyMs}ms")
                    println()
                }
            }
            
            // Real performance expectations
            when {
                bestAccuracy > 0.70 -> println("✅ Excellent real-world LettaBench performance!")
                bestAccuracy > 0.50 -> println("🟡 Good performance, room for improvement")
                bestAccuracy > 0.30 -> println("🟡 Moderate performance, retrieval system working")
                bestAccuracy > 0.10 -> println("🟡 Basic functionality, needs optimization")
                else -> println("🔴 System needs significant debugging")
            }
            
            // Basic validation
            assertTrue(report.summaries.isNotEmpty(), "Should have benchmark results")
            assertTrue(report.summaries.all { it.samples.isNotEmpty() }, "Should have processed questions")
            
        } catch (e: Exception) {
            println("❌ Real LettaBench benchmark failed: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }
    
    @Test
    fun testScalabilityWithRealLLM() = runTest {
        println("\n📈 Testing Scalability with Real LLM")
        println("=".repeat(35))
        
        // Skip if LMStudio not available
        try {
            createLMStudioExecutor()
        } catch (e: Exception) {
            println("⚠️  Skipping test - LMStudio not available")
            return@runTest
        }
        
        val lettaBenchContent = loadLettaBenchFile()
        val fullDataset = DatasetLoader.loadLettaBench(lettaBenchContent, "ScalabilityTest")
        
        // Test with small incremental sizes (LLM calls are expensive)
        val testSizes = listOf(2, 3, 5)
        
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val llmExecutor = createLMStudioExecutor()
        
        populateKnowledgeGraph(fullDataset, knowledgeGraph)
        
        println("📊 Real LLM Scalability Analysis:")
        println("Size\tAccuracy\tAvgTokens\tP95ms\tCost")
        println("-".repeat(50))
        
        for (size in testSizes) {
            val subset = fullDataset.items.take(size)
            val testDataset = Dataset(name = "ScalabilityTest$size", items = subset)
            
            val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, llmExecutor)
            val report = executor.runAllModes(testDataset)
            
            val bestSummary = report.summaries.maxBy { it.accuracy }
            val estimatedCost = (bestSummary.avgTokens * size * 3) / 1000.0 * 0.002 // Rough cost estimate
            
            println("${size}\t${(bestSummary.accuracy * 100).toInt()}%\t\t${bestSummary.avgTokens}\t\t${bestSummary.p95LatencyMs}ms\t\$${String.format("%.3f", estimatedCost)}")
        }
        
        println("✅ Real LLM scalability test completed")
        println("💡 Note: Costs are estimated based on typical OpenAI pricing")
    }
    
    /**
     * Load the LettaBench dataset file content
     */
    private fun loadLettaBenchFile(): String {
        // Use properly formatted JSON
        return """{"question": ["Which person collaborated with Eliza Woodley on a charity event that combined music and literature?", "Who created a dessert inspired by one of Eliza Woodley's novels?", "Who participated in a beach cleanup event organized by Joren Sullivar?"], "answer": ["Nora Fishel", "Suzanne Waggert", "Nora Fishel"], "facts": ["Nora Fishel is an accomplished violinist who has performed in orchestras across Europe.", "Joren Sullivar is a marine biologist specializing in coral reef restoration.", "Eliza Woodley is a published author known for her historical fiction novels.", "Suzanne Waggert is a renowned pastry chef who owns a bakery in Paris.", "Nora Fishel and Eliza Woodley collaborated on a charity event that combined music and literature.", "Joren Sullivar and Suzanne Waggert met during a culinary tour in the Mediterranean.", "Eliza Woodley and Suzanne Waggert co-host a podcast about creative careers.", "Nora Fishel and Joren Sullivar both attended the University of Edinburgh, though in different departments.", "Suzanne Waggert created a dessert inspired by one of Eliza Woodley's novels.", "Joren Sullivar once organized a beach cleanup event that Nora Fishel participated in."], "name": ["Nora Fishel", "Joren Sullivar", "Eliza Woodley", "Suzanne Waggert"], "supporting_fact_indices": [[4], [8], [9]], "contradicting_facts": ["Joren Sullivar and Eliza Woodley collaborated on a charity event that combined music and literature.", "Nora Fishel, inspired by Eliza Woodley's novels, created a dessert for a charity event, which was later featured in a Parisian bakery.", "Joren Sullivar once organized a beach cleanup event that Eliza Woodley participated in."], "contradicting_answers": ["Joren Sullivar", "Nora Fishel", "Eliza Woodley"]}
{"question": ["Which university did both Rylan Gallaher and Jonn Coal attend?", "Who published a book featuring photographs of natural reserves studied by Jonn Coal?"], "answer": ["University of Oregon", "Rylan Gallaher"], "facts": ["Rylan Gallaher is an accomplished landscape photographer.", "Jonn Coal is known for his expertise in environmental science.", "Rylan Gallaher and Jonn Coal collaborated on a documentary about climate change.", "Jonn Coal received an award for his research on renewable energy.", "Rylan Gallaher has exhibited his photography in several international galleries.", "Rylan Gallaher and Jonn Coal both attended the University of Oregon.", "Jonn Coal enjoys hiking and often explores new trails with Rylan Gallaher.", "Rylan Gallaher published a book featuring photographs of natural reserves studied by Jonn Coal.", "Jonn Coal and Rylan Gallaher co-founded a non-profit organization focused on conservation.", "Rylan Gallaher is fluent in Spanish, which helped during joint research trips with Jonn Coal to South America."], "name": ["Rylan Gallaher", "Jonn Coal"], "supporting_fact_indices": [[5], [7]], "contradicting_facts": ["Rylan Gallaher attended Stanford University, while Jonn Coal attended the University of Oregon.", "Jonn Coal published a book featuring photographs of natural reserves he studied, with contributions from Rylan Gallaher."], "contradicting_answers": ["They did not attend the same university", "Jonn Coal"]}"""
    }
    
    /**
     * Populate knowledge graph with facts from the dataset
     */
    private suspend fun populateKnowledgeGraph(dataset: Dataset, knowledgeGraph: InMemoryKnowledgeGraph) {
        println("🔧 Populating knowledge graph with dataset facts...")
        
        // For each QA item, create episodes containing the facts and relations
        dataset.items.forEach { item ->
            val facts = item.metadata?.get("facts")?.toString()?.split(" | ") ?: emptyList()
            val entities = item.entities
            
            // Create episodes for each fact to build up the knowledge base
            facts.forEachIndexed { index, fact ->
                val episode = ai.koog.agents.memory.graph.Episode(
                    content = fact,
                    timestamp = kotlinx.datetime.Clock.System.now(),
                    source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL,
                    metadata = mapOf(
                        "entities" to (entities ?: emptyList<String>()),
                        "question_id" to item.id,
                        "fact_index" to index.toString(),
                        "source_name" to "RealLettaBench"
                    )
                )
                
                try {
                    knowledgeGraph.ingest(episode)
                } catch (e: Exception) {
                    // Continue on error - some facts might not be processable
                }
            }
        }
        
        val stats = knowledgeGraph.stats()
        println("   Knowledge graph populated: ${stats.nodeCount} nodes, ${stats.edgeCount} edges")
    }
}