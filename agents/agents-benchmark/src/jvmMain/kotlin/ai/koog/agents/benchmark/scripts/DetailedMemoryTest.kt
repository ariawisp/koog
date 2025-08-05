package ai.koog.agents.benchmark.scripts

import ai.koog.agents.benchmark.datasets.LettaBenchLoader
import ai.koog.agents.benchmark.runners.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import ai.koog.agents.memory.model.*
import ai.koog.agents.memory.retrieval.*
import kotlinx.coroutines.runBlocking
import kotlinx.datetime.Clock

/**
 * Detailed test to debug memory retrieval issues
 */
suspend fun main() {
    println("🔍 Detailed SOTA Memory System Test")
    println("=" * 50)
    
    try {
        // 1. Create memory system with debugging
        println("\n🧠 Creating SOTA memory system...")
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val memorySystem = SOTAMemorySystem(
            SOTAMemoryConfig(
                strategy = RetrievalStrategy.GRAPH,
                answerStrategy = AnswerStrategy.SIMPLE,
                retrievalLimit = 10,
                knowledgeGraph = knowledgeGraph
            )
        )
        println("✅ Memory system created")
        
        // 2. Add test data
        println("\n📥 Adding test data...")
        val testData = listOf(
            "Jerry Stevens owns 3 vehicles." to mapOf("entity_name" to "Jerry Stevens", "entity_type" to "person"),
            "Julie Saunders owns 2 vehicles." to mapOf("entity_name" to "Julie Saunders", "entity_type" to "person"),
            "Lisa Norris owns 5 vehicles." to mapOf("entity_name" to "Lisa Norris", "entity_type" to "person"),
            "Grant Martinez owns 3 vehicles." to mapOf("entity_name" to "Grant Martinez", "entity_type" to "person")
        )
        
        for ((context, metadata) in testData) {
            println("  Adding: $context")
            memorySystem.addContext(context, metadata)
        }
        println("✅ Test data added")
        
        // 3. Test retrieval directly
        println("\n🔍 Testing retrieval...")
        val queries = listOf(
            "Jerry Stevens",
            "vehicles",
            "owns 3 vehicles",
            "How many vehicles does Jerry Stevens own?",
            "Who has more vehicles, Jerry Stevens or Julie Saunders?"
        )
        
        for (query in queries) {
            println("\n  Query: '$query'")
            val results = memorySystem.retrieve(query)
            println("  Results (${results.size}):")
            results.forEachIndexed { i, result ->
                println("    [$i] ${result.take(100)}...")
            }
        }
        
        // 4. Test full question answering
        println("\n\n❓ Testing question answering...")
        val executor = LLMProviderFactory.createFromEnvironment("openai", "gpt-4o-mini")
        
        val question = "Who has more vehicles, Jerry Stevens or Julie Saunders?"
        println("Question: $question")
        
        val answer = memorySystem.answer(question, executor)
        println("Answer: $answer")
        
        // 5. Direct graph inspection
        println("\n\n🔍 Inspecting knowledge graph directly...")
        val stats = knowledgeGraph.stats()
        println("Graph stats: nodes=${stats.nodeCount}, edges=${stats.edgeCount}")
        
        println("\n✅ Test complete!")
        
    } catch (e: Exception) {
        println("\n❌ Error: ${e.message}")
        e.printStackTrace()
    }
}

private operator fun String.times(n: Int): String = repeat(n)