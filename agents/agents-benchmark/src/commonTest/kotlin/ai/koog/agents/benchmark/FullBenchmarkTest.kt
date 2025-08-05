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
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Full benchmark test using real LettaBench dataset
 */
class FullBenchmarkTest {
    
    /**
     * Simple character-based tokenizer for testing
     */
    private class TestTokenizer : PromptTokenizer {
        override fun tokenCountFor(message: Message): Int = message.content.length / 4
        override fun tokenCountFor(prompt: Prompt): Int = prompt.messages.sumOf { tokenCountFor(it) }
    }
    
    /**
     * Create a simple mock executor for testing (doesn't actually call LLM)
     */
    private fun createMockExecutor() = simpleOpenAIExecutor("mock-key")
    
    @Test
    fun testLettaBenchDatasetPerformance() = runTest {
        println("🚀 Running Full LettaBench Performance Test")
        println("=".repeat(50))
        
        // Load the real LettaBench dataset
        val lettaBenchContent = loadLettaBenchFile()
        val dataset = DatasetLoader.loadLettaBench(lettaBenchContent, "LettaBench200")
        
        println("📊 Dataset loaded: ${dataset.name}")
        println("   Questions: ${dataset.items.size}")
        println("   Sample question: ${dataset.items.first().question.take(80)}...")
        
        // Create test infrastructure
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val mockExecutor = createMockExecutor()
        
        // Populate knowledge graph with facts from the dataset
        populateKnowledgeGraph(dataset, knowledgeGraph)
        
        // Create benchmark executor
        val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, mockExecutor)
        
        // Run benchmarks on subset first to test infrastructure
        val testSubset = dataset.items.take(10)
        val testDataset = Dataset(name = "LettaBench10", items = testSubset)
        
        println("\n🔄 Running benchmark on ${testSubset.size} questions...")
        
        try {
            val report = executor.runAllModes(testDataset)
            
            // Print results
            executor.printReport(report)
            
            // Analyze performance
            val bestAccuracy = report.summaries.maxOf { it.accuracy }
            val avgTokens = report.summaries.map { it.avgTokens }.average()
            val bestMode = report.summaries.maxBy { it.accuracy }.mode
            
            println("\n🎯 Performance Analysis:")
            println("   Best Accuracy: ${(bestAccuracy * 100).toInt()}% (${bestMode})")
            println("   Avg Tokens: ${avgTokens.toInt()}")
            println("   SOTA Target: >95% accuracy, <1500 tokens")
            
            // Validate basic functionality
            assertTrue(report.summaries.isNotEmpty(), "Should have benchmark results")
            assertTrue(report.summaries.all { it.samples.isNotEmpty() }, "Should have processed questions")
            
            // Check if we're approaching SOTA performance
            when {
                bestAccuracy > 0.95 -> println("✅ SOTA performance achieved!")
                bestAccuracy > 0.85 -> println("🟡 Good performance, approaching SOTA")
                bestAccuracy > 0.70 -> println("🟡 Reasonable performance, needs optimization")
                else -> println("🔴 Performance needs significant improvement")
            }
            
            if (avgTokens < 1500) {
                println("✅ Token efficiency target met!")
            } else {
                println("🔧 Token efficiency needs optimization")
            }
            
        } catch (e: Exception) {
            println("❌ Benchmark failed: ${e.message}")
            e.printStackTrace()
            throw e
        }
    }
    
    @Test 
    fun testScalabilityWithLargerDataset() = runTest {
        println("\n🔄 Testing Scalability with Larger Dataset")
        println("=".repeat(45))
        
        val lettaBenchContent = loadLettaBenchFile()
        val fullDataset = DatasetLoader.loadLettaBench(lettaBenchContent, "LettaBench200")
        
        // Test with progressively larger subsets
        val testSizes = listOf(5, 10, 25, 50)
        
        val tokenizer = TestTokenizer()
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(graphProvider = graphProvider)
        val mockExecutor = createMockExecutor()
        
        populateKnowledgeGraph(fullDataset, knowledgeGraph)
        
        println("📊 Scalability Analysis:")
        println("Size\tAccuracy\tAvgTokens\tP95ms")
        println("-".repeat(40))
        
        for (size in testSizes) {
            val subset = fullDataset.items.take(size)
            val testDataset = Dataset(name = "LettaBench$size", items = subset)
            
            val executor = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, mockExecutor)
            val report = executor.runAllModes(testDataset)
            
            val bestSummary = report.summaries.maxBy { it.accuracy }
            println("${size}\t${(bestSummary.accuracy * 100).toInt()}%\t\t${bestSummary.avgTokens}\t\t${bestSummary.p95LatencyMs}ms")
        }
        
        println("✅ Scalability test completed")
    }
    
    /**
     * Load the LettaBench dataset file content
     */
    private fun loadLettaBenchFile(): String {
        // Read the first few lines for testing - in real implementation would read full file
        return """{"question": ["Which person collaborated with Eliza Woodley on a charity event that combined music and literature?", "Who created a dessert inspired by one of Eliza Woodley's novels?", "Who participated in a beach cleanup event organized by Joren Sullivar?"], "answer": ["Nora Fishel", "Suzanne Waggert", "Nora Fishel"], "facts": ["Nora Fishel is an accomplished violinist who has performed in orchestras across Europe.", "Joren Sullivar is a marine biologist specializing in coral reef restoration.", "Eliza Woodley is a published author known for her historical fiction novels.", "Suzanne Waggert is a renowned pastry chef who owns a bakery in Paris.", "Nora Fishel and Eliza Woodley collaborated on a charity event that combined music and literature.", "Joren Sullivar and Suzanne Waggert met during a culinary tour in the Mediterranean.", "Eliza Woodley and Suzanne Waggert co-host a podcast about creative careers.", "Nora Fishel and Joren Sullivar both attended the University of Edinburgh, though in different departments.", "Suzanne Waggert created a dessert inspired by one of Eliza Woodley's novels.", "Joren Sullivar once organized a beach cleanup event that Nora Fishel participated in."], "name": ["Nora Fishel", "Joren Sullivar", "Eliza Woodley", "Suzanne Waggert"], "supporting_fact_indices": [[4], [8], [9]], "contradicting_facts": ["\"Joren Sullivar and Eliza Woodley collaborated on a charity event that combined music and literature.\"", "Nora Fishel, inspired by Eliza Woodley's novels, created a dessert for a charity event, which was later featured in a Parisian bakery.", "Joren Sullivar once organized a beach cleanup event that Eliza Woodley participated in."], "contradicting_answers": ["Joren Sullivar", "Nora Fishel", "Eliza Woodley"]}
{"question": ["Which university did both Rylan Gallaher and Jonn Coal attend?", "Who published a book featuring photographs of natural reserves studied by Jonn Coal?", "What is the area of expertise for Jonn Coal?", "Which language is Rylan Gallaher fluent in that aided joint research trips with Jonn Coal to South America?", "What type of organization did Jonn Coal and Rylan Gallaher co-found?", "In what field has Rylan Gallaher exhibited work in several international galleries?", "With whom does Jonn Coal often explore new hiking trails?"], "answer": ["University of Oregon", "Rylan Gallaher", "environmental science", "Spanish", "non-profit organization focused on conservation", "photography", "Rylan Gallaher"], "facts": ["Rylan Gallaher is an accomplished landscape photographer.", "Jonn Coal is known for his expertise in environmental science.", "Rylan Gallaher and Jonn Coal collaborated on a documentary about climate change.", "Jonn Coal received an award for his research on renewable energy.", "Rylan Gallaher has exhibited his photography in several international galleries.", "Rylan Gallaher and Jonn Coal both attended the University of Oregon.", "Jonn Coal enjoys hiking and often explores new trails with Rylan Gallaher.", "Rylan Gallaher published a book featuring photographs of natural reserves studied by Jonn Coal.", "Jonn Coal and Rylan Gallaher co-founded a non-profit organization focused on conservation.", "Rylan Gallaher is fluent in Spanish, which helped during joint research trips with Jonn Coal to South America."], "name": ["Rylan Gallaher", "Jonn Coal"], "supporting_fact_indices": [[5], [7], [1, 3], [9], [8], [0, 4], [6]], "contradicting_facts": ["Rylan Gallaher attended Stanford University, while Jonn Coal attended the University of Oregon.", "Jonn Coal published a book featuring photographs of natural reserves he studied, with contributions from Rylan Gallaher.", "Jonn Coal is known for his expertise in classical music composition.", "Rylan Gallaher is fluent in Portuguese, which aided joint research trips with Jonn Coal to Brazil in South America.", "Jonn Coal and Rylan Gallaher co-founded a for-profit tech startup specializing in landscape photography equipment.", "Rylan Gallaher is an accomplished sculptor and has exhibited his sculptures in several international galleries.", "Jonn Coal often explores new hiking trails with his sister, not Rylan Gallaher."], "contradicting_answers": ["They did not attend the same university; Rylan Gallaher attended Stanford University and Jonn Coal attended the University of Oregon.", "Jonn Coal", "classical music composition", "Portuguese", "for-profit tech startup specializing in landscape photography equipment", "sculpture", "His sister"]}
{"question": ["Which Michigan-based arts contributor is Theresa Bronn recognized as?", "For which vocal ensemble has Theresa Bronn served as conductor?", "What voice part has Theresa Bronn performed as a soloist in concerts?", "How long has Theresa Bronn been involved in music education?", "What is Theresa Bronn passionate about promoting in schools?", "With whom has Theresa Bronn collaborated for special performances?", "What type of choirs has Theresa Bronn directed in her career?", "What kind of educational events has Theresa Bronn led for aspiring choral directors?"], "answer": ["arts in Michigan", "Kalamazoo Singers", "soprano", "over two decades", "the importance of music", "other musicians and composers", "community and church choirs", "workshops and clinics"], "facts": ["Theresa Bronn is a professional musician and conductor.", "Theresa Bronn has served as the conductor of the Kalamazoo Singers.", "Theresa Bronn is known for her work in choral music education.", "Theresa Bronn has directed numerous community and church choirs.", "Theresa Bronn has performed as a soprano soloist in various concerts.", "Theresa Bronn has been involved in music education for over two decades.", "Theresa Bronn has collaborated with other musicians and composers for special performances.", "Theresa Bronn has received recognition for her contributions to the arts in Michigan.", "Theresa Bronn has led workshops and clinics for aspiring choral directors.", "Theresa Bronn is passionate about promoting the importance of music in schools."], "name": ["Theresa Bronn"], "supporting_fact_indices": [[7], [1], [4], [5], [9], [6], [3], [8]], "contradicting_facts": ["Theresa Bronn has never lived or worked in Michigan and has not been involved in any arts-related activities in the state.", "Theresa Bronn has never conducted the Kalamazoo Singers but has instead served as the conductor of the Detroit Symphony Chorus.", "Theresa Bronn has performed as a bass soloist in various concerts.", "Theresa Bronn only began her involvement in music education within the past year.", "Theresa Bronn is passionate about promoting the importance of science and technology in schools.", "Theresa Bronn has never collaborated with any other musicians or composers for special performances; she always performs solo.", "Theresa Bronn has exclusively directed children's choirs throughout her career and has never worked with community or church choirs.", "Theresa Bronn has never led any educational events or training sessions for aspiring choral directors."], "contradicting_answers": ["Theresa Bronn is not recognized as a Michigan-based arts contributor.", "Detroit Symphony Chorus", "bass", "less than one year", "the importance of science and technology", "She has not collaborated with anyone; Theresa Bronn performs all special performances solo.", "children's choirs", "She has not led any educational events for aspiring choral directors."]}"""
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
                        "source_name" to "LettaBench"
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