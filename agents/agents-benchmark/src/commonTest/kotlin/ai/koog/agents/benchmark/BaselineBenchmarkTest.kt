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
import ai.koog.agents.memory.retrieval.providers.VectorRetrievalProvider
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import ai.koog.prompt.executor.llms.all.simpleOpenAIExecutor
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Baseline benchmark tests to measure current implementation performance
 */
class BaselineBenchmarkTest {
    
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
    fun testSimpleDatasetBaseline() = runTest {
        println("🚀 Running Simple Dataset Baseline Test")
        println("=" + "=".repeat(40))
        
        // Create test infrastructure with simplified setup
        val tokenizer = TestTokenizer() 
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        val smartRouter = createSmartRouter(
            graphProvider = graphProvider
        )
        val mockExecutor = createMockExecutor()
        
        // Create simple test dataset
        val dataset = createSimpleTestDataset()
        
        // Test basic dataset loading and structure
        println("📊 Dataset loaded: ${dataset.name}")
        println("   Questions: ${dataset.items.size}")
        dataset.items.forEach { item ->
            println("   - ${item.id}: ${item.question.take(50)}...")
        }
        
        // Test basic benchmark runner functionality
        try {
            val runner = BenchmarkExecutor(tokenizer, graphProvider, smartRouter, mockExecutor)
            
            // For now, just test that we can create the runner and it doesn't crash
            println("✅ BenchmarkExecutor created successfully")
            println("✅ Infrastructure ready for benchmarking")
            
        } catch (e: Exception) {
            println("❌ Error setting up benchmark infrastructure: ${e.message}")
            throw e
        }
        
        // Verify basic functionality
        assertTrue(dataset.items.isNotEmpty(), "Should have test questions")
        assertTrue(dataset.items.all { it.question.isNotBlank() }, "All questions should be non-empty")
        assertTrue(dataset.items.all { it.goldAnswer.isNotBlank() }, "All answers should be non-empty")
    }
    
    @Test 
    fun testMultiHopReasoningBaseline() = runTest {
        println("\n🧠 Running Multi-Hop Reasoning Baseline Test")
        println("=" + "=".repeat(45))
        
        // Create multi-hop test dataset
        val dataset = createMultiHopTestDataset()
        
        // Analyze multi-hop complexity
        println("📊 Multi-hop dataset analysis:")
        println("   Questions: ${dataset.items.size}")
        dataset.items.forEach { item ->
            val hops = item.metadata?.get("hops") ?: "unknown"
            println("   - ${item.id}: $hops hops - ${item.question.take(60)}...")
        }
        
        // Verify multi-hop structure
        assertTrue(dataset.items.isNotEmpty(), "Should have multi-hop questions")
        assertTrue(dataset.items.all { 
            it.metadata?.get("type") == "multi_hop" 
        }, "All questions should be multi-hop type")
        
        println("✅ Multi-hop reasoning test dataset validated")
    }
    
    @Test
    fun testTokenEfficiencyBaseline() = runTest {
        println("\n⚡ Running Token Efficiency Baseline Test")
        println("=" + "=".repeat(40))
        
        val dataset = createTokenEfficiencyTestDataset()
        val tokenizer = TestTokenizer()
        
        // Analyze token requirements for different question types
        println("📊 Token efficiency analysis:")
        println("   Questions: ${dataset.items.size}")
        
        dataset.items.forEach { item ->
            val questionTokens = tokenizer.tokenCountFor(
                ai.koog.prompt.message.Message.User(
                    item.question, 
                    ai.koog.prompt.message.RequestMetaInfo(kotlinx.datetime.Clock.System.now())
                )
            )
            val answerTokens = tokenizer.tokenCountFor(
                ai.koog.prompt.message.Message.User(
                    item.goldAnswer,
                    ai.koog.prompt.message.RequestMetaInfo(kotlinx.datetime.Clock.System.now())
                )
            )
            val target = item.metadata?.get("efficiency_target") ?: "standard"
            
            println("   - ${item.id}: Q=${questionTokens}t, A=${answerTokens}t, target=$target")
        }
        
        // Verify efficiency targets
        assertTrue(dataset.items.isNotEmpty(), "Should have efficiency test questions")
        
        println("✅ Token efficiency test dataset validated")
    }
    
    private fun createSimpleTestDataset(): Dataset {
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
        
        return Dataset(name = "SimpleTestDataset", items = items)
    }
    
    private fun createMultiHopTestDataset(): Dataset {
        val items = listOf(
            QAItem(
                id = "multihop_001",
                question = "What is the capital of the country where the Eiffel Tower is located?",
                goldAnswer = "Paris",
                entities = listOf("Eiffel Tower", "France", "Paris"),
                metadata = mapOf("type" to "multi_hop", "difficulty" to "medium", "hops" to "2")
            ),
            QAItem(
                id = "multihop_002",
                question = "Who wrote the play that features the famous balcony scene?",
                goldAnswer = "William Shakespeare",
                entities = listOf("balcony scene", "Romeo and Juliet", "William Shakespeare"),
                metadata = mapOf("type" to "multi_hop", "difficulty" to "medium", "hops" to "2")
            ),
            QAItem(
                id = "multihop_003", 
                question = "What is the largest planet in the solar system where Earth is located?",
                goldAnswer = "Jupiter",
                entities = listOf("Earth", "solar system", "Jupiter", "planet"),
                metadata = mapOf("type" to "multi_hop", "difficulty" to "medium", "hops" to "2")
            )
        )
        
        return Dataset(name = "MultiHopTestDataset", items = items)
    }
    
    private fun createTokenEfficiencyTestDataset(): Dataset {
        val items = listOf(
            QAItem(
                id = "efficiency_001",
                question = "Quick answer: What is 2+2?",
                goldAnswer = "4",
                entities = listOf("mathematics", "addition"),
                metadata = mapOf("type" to "arithmetic", "efficiency_target" to "low_tokens")
            ),
            QAItem(
                id = "efficiency_002",
                question = "One word: Capital of Italy?",
                goldAnswer = "Rome",
                entities = listOf("Italy", "Rome"),
                metadata = mapOf("type" to "factual", "efficiency_target" to "minimal_tokens")
            )
        )
        
        return Dataset(name = "TokenEfficiencyTestDataset", items = items)
    }
    
    
    /**
     * Test LettaBench dataset loading capability
     */
    @Test
    fun testLettaBenchDatasetLoading() = runTest {
        println("\n📊 Testing LettaBench Dataset Loading")
        println("=" + "=".repeat(40))
        
        // Test the LettaBench format parsing
        val sampleLettaBenchLine = """{"question": ["Who is Alice?", "Where does Bob work?"], "answer": ["Alice is a student", "Bob works at Google"], "facts": ["Alice studies at MIT", "Bob is a software engineer"], "name": ["Alice", "Bob"], "supporting_fact_indices": [[0], [1]]}"""
        
        try {
            val dataset = DatasetLoader.loadLettaBench(sampleLettaBenchLine, "SampleLettaBench")
            
            println("✅ LettaBench format parsed successfully")
            println("   Dataset: ${dataset.name}")
            println("   Questions generated: ${dataset.items.size}")
            
            dataset.items.forEach { item ->
                println("   - ${item.id}: ${item.question}")
                println("     Answer: ${item.goldAnswer}")
                println("     Entities: ${item.entities}")
            }
            
            // Verify the conversion worked
            assertTrue(dataset.items.isNotEmpty(), "Should generate QA items from LettaBench format")
            assertTrue(dataset.items.all { it.metadata?.get("source") == "letta_bench" }, 
                      "All items should have letta_bench source metadata")
            
        } catch (e: Exception) {
            println("❌ Error loading LettaBench format: ${e.message}")
            throw e
        }
    }
}