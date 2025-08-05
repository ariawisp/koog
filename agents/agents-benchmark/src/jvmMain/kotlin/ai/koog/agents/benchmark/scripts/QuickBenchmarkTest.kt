package ai.koog.agents.benchmark.scripts

import ai.koog.agents.benchmark.datasets.LettaBenchLoader
import ai.koog.agents.benchmark.runners.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlin.time.Duration.Companion.seconds

/**
 * Quick test of the SOTA benchmark system with just a few questions
 */
suspend fun main() {
    println("🚀 Quick Koog SOTA Benchmark Test")
    println("=" * 50)
    
    try {
        // 1. Load dataset
        println("\n📚 Loading LettaBench dataset...")
        val loader = LettaBenchLoader()
        val dataset = loader.load("")
        println("✅ Loaded ${dataset.questions.size} questions")
        
        // 2. Create executor
        println("\n🤖 Creating OpenAI executor...")
        val executor = LLMProviderFactory.createFromEnvironment("openai", "gpt-4o-mini")
        println("✅ Executor ready")
        
        // 3. Create memory system
        println("\n🧠 Setting up SOTA memory system...")
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val memorySystem = SOTAMemorySystem(
            SOTAMemoryConfig(
                strategy = RetrievalStrategy.GRAPH,
                answerStrategy = AnswerStrategy.SIMPLE,
                retrievalLimit = 5,
                knowledgeGraph = knowledgeGraph
            )
        )
        
        // 4. Populate with minimal data
        println("\n📥 Adding context to memory system...")
        val sampleQuestions = dataset.questions.take(5)
        
        // Add just a few key facts for testing
        memorySystem.addContext(
            "Jerry Stevens owns 3 vehicles.",
            mapOf("entity_name" to "Jerry Stevens", "entity_type" to "person")
        )
        memorySystem.addContext(
            "Julie Saunders owns 2 vehicles.",
            mapOf("entity_name" to "Julie Saunders", "entity_type" to "person")
        )
        println("✅ Context added")
        
        // 5. Test with first question
        println("\n🧪 Testing with sample questions...")
        for ((index, question) in sampleQuestions.withIndex()) {
            println("\n❓ Question ${index + 1}: ${question.text}")
            println("   Expected: ${question.expectedAnswer}")
            
            try {
                val answer = withTimeout(30.seconds) {
                    memorySystem.answer(question.text, executor)
                }
                println("   Got: $answer")
                
                val correct = question.expectedAnswer.lowercase() in answer.lowercase()
                println("   Result: ${if (correct) "✅ CORRECT" else "❌ INCORRECT"}")
            } catch (e: Exception) {
                println("   ❌ Error: ${e.message}")
                e.printStackTrace()
            }
        }
        
        println("\n✅ Test complete!")
        
    } catch (e: Exception) {
        println("\n❌ Fatal error: ${e.message}")
        e.printStackTrace()
    }
}

private operator fun String.times(n: Int): String = repeat(n)