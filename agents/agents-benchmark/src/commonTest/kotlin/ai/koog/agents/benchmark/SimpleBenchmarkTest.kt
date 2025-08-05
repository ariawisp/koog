package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.llm.LLMAnswerGenerator
import ai.koog.agents.benchmark.literature.*
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.retrieval.RetrievalQuery
import ai.koog.agents.memory.retrieval.RetrievalRecipe
import ai.koog.agents.memory.retrieval.providers.KnowledgeGraphRetrievalProvider
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.message.Message
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

/**
 * Simple, direct benchmark test without over-engineering
 */
class SimpleBenchmarkTest {
    
    private class TestTokenizer : PromptTokenizer {
        override fun tokenCountFor(message: Message): Int = message.content.length / 4
        override fun tokenCountFor(prompt: Prompt): Int = prompt.messages.sumOf { tokenCountFor(it) }
    }
    
    @Test
    fun testSimpleBenchmark() = runTest {
        // Create simple test questions
        val questions = listOf(
            "Who collaborated with Alice on the AI project?",
            "What was the outcome of the research project?",
            "When did Bob receive the award?"
        )
        
        // Setup Koog
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val tokenizer = TestTokenizer()
        val graphProvider = KnowledgeGraphRetrievalProvider(knowledgeGraph, tokenizer)
        
        // Add some test data
        knowledgeGraph.ingest(ai.koog.agents.memory.graph.Episode(
            content = "Alice collaborated with Bob on the AI project",
            timestamp = kotlinx.datetime.Clock.System.now(),
            source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL
        ))
        knowledgeGraph.ingest(ai.koog.agents.memory.graph.Episode(
            content = "The research project resulted in a publication",
            timestamp = kotlinx.datetime.Clock.System.now(),
            source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL
        ))
        knowledgeGraph.ingest(ai.koog.agents.memory.graph.Episode(
            content = "Bob received an award in 2023",
            timestamp = kotlinx.datetime.Clock.System.now(),
            source = ai.koog.agents.memory.graph.EpisodeSource.EXTERNAL
        ))
        
        // Run benchmark with mock LLM
        val mockExecutor = createMockExecutor()
        val answerGenerator = LLMAnswerGenerator(mockExecutor)
        
        var correct = 0
        val total = questions.size
        
        println("\n🚀 Running Simple Koog Benchmark")
        println("=" .repeat(50))
        
        questions.forEachIndexed { index, question ->
            // Retrieve context
            val query = RetrievalQuery(
                text = question,
                recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE,
                k = 5
            )
            val context = graphProvider.retrieve(query)
            
            // Generate answer
            val answer = answerGenerator.generateAnswer(question, context)
            
            // Simple validation
            val isCorrect = when(index) {
                0 -> answer.contains("Bob", ignoreCase = true)
                1 -> answer.contains("publication", ignoreCase = true)
                2 -> answer.contains("2023", ignoreCase = true)
                else -> false
            }
            
            if (isCorrect) correct++
            
            println("Q${index + 1}: $question")
            println("A: $answer")
            println("✓ Correct: $isCorrect")
            println()
        }
        
        val accuracy = correct.toDouble() / total
        println("📊 Results: $correct/$total correct (${(accuracy * 100).toInt()}%)")
        
        // Compare to literature
        val benchmarkResult = BenchmarkResult(
            systemName = "Koog",
            accuracy = accuracy,
            avgLatencyMs = 100, // Mock
            avgTokens = 500, // Mock
            model = "mock"
        )
        
        val literatureComparator = LiteratureComparator()
        val comparison = literatureComparator.compareAgainstLiterature(
            "Deep Memory Retrieval (DMR)",
            benchmarkResult
        )
        
        println("\n📚 Literature Comparison:")
        println("Koog would rank #${comparison.ranking} of ${comparison.totalSystems} systems")
        println("Current SOTA: 94.8% (Zep)")
        println("Koog needs ${((0.948 - accuracy) * 100).toInt()}% improvement to beat SOTA")
    }
    
    private fun createMockExecutor() = getMockExecutor {
        // Mock responses for each question type
        mockLLMAnswer("Bob collaborated with Alice on the AI project") onRequestContains "collaborated with Alice"
        mockLLMAnswer("The research project resulted in a publication") onRequestContains "outcome of the research"
        mockLLMAnswer("Bob received the award in 2023") onRequestContains "receive the award"
        
        // Default fallback
        mockLLMAnswer("Based on the context provided, I cannot determine the answer.").asDefaultResponse
    }
}