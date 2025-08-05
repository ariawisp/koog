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
 * Targeted benchmark test that loads specific entities mentioned in questions
 */
suspend fun main() {
    println("🎯 Targeted Koog Benchmark Test")
    println("=" * 50)
    
    try {
        // 1. Load questions
        println("\n📚 Loading LettaBench questions...")
        val loader = LettaBenchLoader()
        val dataset = loader.load("")
        val testQuestions = dataset.questions.take(10) // Test first 10
        println("✅ Loaded ${dataset.questions.size} questions, testing ${testQuestions.size}")
        
        // 2. Extract entities mentioned in questions
        println("\n🔍 Analyzing questions for required entities...")
        val requiredEntities = extractRequiredEntities(testQuestions.map { it.text })
        println("   People mentioned: ${requiredEntities.people}")
        println("   Vehicles mentioned: ${requiredEntities.vehicles}")
        
        // 3. Create memory system
        println("\n🧠 Setting up SOTA memory system...")
        val knowledgeGraph = InMemoryKnowledgeGraph()
        val memorySystem = SOTAMemorySystem(
            SOTAMemoryConfig(
                strategy = RetrievalStrategy.GRAPH,
                answerStrategy = AnswerStrategy.SIMPLE,
                retrievalLimit = 10,
                knowledgeGraph = knowledgeGraph
            )
        )
        
        // 4. Load targeted context
        println("\n📥 Loading targeted context data...")
        val loadedCount = loadTargetedContext(memorySystem, requiredEntities)
        println("✅ Loaded ${loadedCount.people} people and ${loadedCount.vehicles} vehicles")
        
        // 5. Create executor
        println("\n🤖 Creating OpenAI executor...")
        val executor = LLMProviderFactory.createFromEnvironment("openai", "gpt-4o-mini")
        
        // 6. Run benchmark
        println("\n🧪 Running benchmark...")
        val judge = SimpleJudge()
        var correct = 0
        
        testQuestions.forEachIndexed { index, question ->
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
        
        println("\n📊 Results: $correct/${testQuestions.size} correct (${correct * 100 / testQuestions.size}%)")
        println("✅ Test complete!")
        
    } catch (e: Exception) {
        println("\n❌ Fatal error: ${e.message}")
        e.printStackTrace()
    }
}

data class RequiredEntities(
    val people: Set<String>,
    val vehicles: Set<String>
)

data class LoadedCount(
    val people: Int,
    val vehicles: Int
)

private fun extractRequiredEntities(questions: List<String>): RequiredEntities {
    val people = mutableSetOf<String>()
    val vehicles = mutableSetOf<String>()
    
    questions.forEach { question ->
        // Extract person names (basic pattern matching)
        val namePattern = Regex("\\b([A-Z][a-z]+ [A-Z][a-z]+)\\b")
        namePattern.findAll(question).forEach { match ->
            val name = match.value
            // Filter out common phrases that aren't names
            if (!name.contains("Who has") && !name.contains("What is")) {
                people.add(name)
            }
        }
        
        // Extract vehicle IDs
        val vehiclePattern = Regex("veh-\\d+")
        vehiclePattern.findAll(question).forEach { match ->
            vehicles.add(match.value)
        }
    }
    
    return RequiredEntities(people, vehicles)
}

private suspend fun loadTargetedContext(
    memorySystem: MemorySystem,
    requiredEntities: RequiredEntities
): LoadedCount {
    var peopleCount = 0
    var vehicleCount = 0
    
    // Map to store person ID by name for vehicle loading
    val personIdByName = mutableMapOf<String, String>()
    
    // Load all people data to find the required ones
    println("  Scanning people data...")
    object {}.javaClass.getResourceAsStream("/datasets/letta_file_bench/data/people.jsonl")?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) {
                    val data = Json.parseToJsonElement(line).jsonObject
                    val fullName = data["full_name"]?.jsonPrimitive?.content ?: ""
                    val id = data["person_id"]?.jsonPrimitive?.content ?: ""
                    
                    // Always load if in required list, or load some extras for context
                    if (requiredEntities.people.contains(fullName) || peopleCount < 50) {
                        memorySystem.addContext(
                            "Person: $fullName (ID: $id)",
                            mapOf("entity_type" to "person", "entity_name" to fullName, "person_id" to id)
                        )
                        personIdByName[fullName] = id
                        peopleCount++
                    }
                }
            }
        }
    }
    
    // Collect all person IDs we need vehicles for
    val neededPersonIds = personIdByName.values.toSet()
    
    // Load vehicle data - both specific vehicles and vehicles for loaded people
    println("  Scanning vehicle data...")
    object {}.javaClass.getResourceAsStream("/datasets/letta_file_bench/data/vehicles.jsonl")?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) {
                    val data = Json.parseToJsonElement(line).jsonObject
                    val vehId = data["vehicle_id"]?.jsonPrimitive?.content ?: ""
                    val ownerId = data["owner_id"]?.jsonPrimitive?.content ?: ""
                    
                    // Load if it's a required vehicle or belongs to a loaded person
                    if (requiredEntities.vehicles.contains(vehId) || neededPersonIds.contains(ownerId)) {
                        val make = data["make"]?.jsonPrimitive?.content ?: ""
                        val model = data["model"]?.jsonPrimitive?.content ?: ""
                        val year = data["year"]?.jsonPrimitive?.content ?: ""
                        
                        memorySystem.addContext(
                            "Vehicle: $year $make $model (ID: $vehId, Owner: $ownerId)",
                            mapOf("entity_type" to "vehicle", "vehicle_id" to vehId, "owner_id" to ownerId)
                        )
                        vehicleCount++
                    }
                }
            }
        }
    }
    
    return LoadedCount(peopleCount, vehicleCount)
}

private operator fun String.times(n: Int): String = repeat(n)