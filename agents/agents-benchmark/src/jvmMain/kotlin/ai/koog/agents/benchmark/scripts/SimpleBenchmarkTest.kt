package ai.koog.agents.benchmark.scripts

import ai.koog.agents.benchmark.datasets.LettaBenchLoader
import ai.koog.agents.benchmark.runners.*
import ai.koog.agents.benchmark.llm.LLMProviderFactory
import ai.koog.agents.benchmark.judge.SimpleJudge
import ai.koog.agents.benchmark.core.MemorySystem
import ai.koog.agents.memory.graph.providers.InMemoryKnowledgeGraph
import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*

/**
 * Simple benchmark test that loads data in batches to avoid stalling
 */
suspend fun main() {
    println("🚀 Simple Koog Benchmark Test")
    println("=" * 50)
    
    try {
        // 1. Load questions only (not full dataset)
        println("\n📚 Loading LettaBench questions...")
        val loader = LettaBenchLoader()
        val dataset = loader.load("")
        println("✅ Loaded ${dataset.questions.size} questions")
        
        // 2. Create memory system
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
        
        // 3. Load minimal context data for testing
        println("\n📥 Loading minimal context data...")
        loadMinimalContext(memorySystem)
        println("✅ Context loaded")
        
        // 4. Create executor
        println("\n🤖 Creating OpenAI executor...")
        val executor = LLMProviderFactory.createFromEnvironment("openai", "gpt-4o-mini")
        
        // 5. Run benchmark on subset
        println("\n🧪 Running benchmark on 5 questions...")
        val judge = SimpleJudge()
        var correct = 0
        
        dataset.questions.take(5).forEachIndexed { index, question ->
            println("\n❓ Question ${index + 1}: ${question.text}")
            println("   Expected: ${question.expectedAnswer}")
            
            try {
                val answer = memorySystem.answer(question.text, executor)
                println("   Got: $answer")
                
                val result = judge.grade(question.text, question.expectedAnswer, answer)
                if (result.correct) {
                    correct++
                    println("   ✅ CORRECT")
                } else {
                    println("   ❌ INCORRECT - ${result.reason}")
                }
            } catch (e: Exception) {
                println("   ❌ Error: ${e.message}")
            }
        }
        
        println("\n📊 Results: $correct/5 correct (${correct * 20}%)")
        println("✅ Test complete!")
        
    } catch (e: Exception) {
        println("\n❌ Fatal error: ${e.message}")
        e.printStackTrace()
    }
}

private suspend fun loadMinimalContext(memorySystem: MemorySystem) {
    // Load just vehicles data as it's needed for most questions
    val vehiclesPath = "/datasets/letta_file_bench/data/vehicles.jsonl"
    val peoplePath = "/datasets/letta_file_bench/data/people.jsonl"
    
    // Load people first (smaller dataset)
    var peopleCount = 0
    object {}.javaClass.getResourceAsStream(peoplePath)?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.take(100).forEach { line -> // Only first 100 people
                if (line.isNotBlank()) {
                    val data = Json.parseToJsonElement(line).jsonObject
                    val name = data["full_name"]?.jsonPrimitive?.content ?: ""
                    val id = data["person_id"]?.jsonPrimitive?.content
                    
                    memorySystem.addContext(
                        "Person: $name (ID: $id)",
                        mapOf("entity_type" to "person", "entity_name" to name, "person_id" to id.toString())
                    )
                    peopleCount++
                }
            }
        }
    }
    println("  Loaded $peopleCount people")
    
    // Load vehicles for those people
    var vehicleCount = 0
    val loadedPeopleIds = mutableSetOf<String>()
    
    object {}.javaClass.getResourceAsStream(vehiclesPath)?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.take(300).forEach { line -> // Only first 300 vehicles
                if (line.isNotBlank()) {
                    val data = Json.parseToJsonElement(line).jsonObject
                    val ownerId = data["owner_id"]?.jsonPrimitive?.content
                    
                    if (ownerId != null && ownerId.startsWith("pers-")) {
                        val vehId = data["vehicle_id"]?.jsonPrimitive?.content
                        val make = data["make"]?.jsonPrimitive?.content
                        val model = data["model"]?.jsonPrimitive?.content
                        val year = data["year"]?.jsonPrimitive?.content
                        
                        memorySystem.addContext(
                            "Vehicle: $year $make $model (ID: $vehId, Owner: $ownerId)",
                            mapOf("entity_type" to "vehicle", "vehicle_id" to vehId.toString(), "owner_id" to ownerId)
                        )
                        
                        loadedPeopleIds.add(ownerId)
                        vehicleCount++
                    }
                }
            }
        }
    }
    println("  Loaded $vehicleCount vehicles for ${loadedPeopleIds.size} people")
}

private operator fun String.times(n: Int): String = repeat(n)