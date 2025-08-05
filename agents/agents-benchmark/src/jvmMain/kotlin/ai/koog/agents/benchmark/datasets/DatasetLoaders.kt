package ai.koog.agents.benchmark.datasets

import ai.koog.agents.benchmark.core.*
import ai.koog.agents.benchmark.model.*
import kotlinx.coroutines.*
import kotlinx.serialization.json.*
import java.io.*

/**
 * Loader for LettaBench dataset using resources
 */
class LettaBenchLoader : DataLoader {
    
    override suspend fun load(path: String): BenchmarkDataset {
        // Load questions from resources
        val questions = loadQuestionsFromResources()
        
        return BenchmarkDataset(
            name = "LettaBench",
            questions = questions,
            contextDescription = loadEvidenceDescription(),
            metadata = mapOf(
                "dataTypes" to listOf("people", "vehicles", "pets", "addresses", "employment", "medical")
            )
        )
    }
    
    private fun loadQuestionsFromResources(): List<Question> {
        val questions = mutableListOf<Question>()
        
        this::class.java.getResourceAsStream("/datasets/letta_file_bench/data/llm_generated_questions.jsonl")?.use { stream ->
            stream.bufferedReader().useLines { lines ->
                lines.forEach { line ->
                    if (line.isNotBlank()) {
                        val q = Json.parseToJsonElement(line).jsonObject
                        questions.add(
                            Question(
                                id = "lettabench_${questions.size}",
                                text = q["question"]?.jsonPrimitive?.content ?: "",
                                expectedAnswer = q["answer"]?.jsonPrimitive?.content ?: "",
                                type = q["question_type"]?.jsonPrimitive?.content ?: "unknown",
                                difficulty = q["difficulty"]?.jsonPrimitive?.content,
                                requiredContext = q["required_files"]?.jsonArray?.map {
                                    it.jsonPrimitive.content
                                } ?: emptyList()
                            )
                        )
                    }
                }
            }
        } ?: throw FileNotFoundException("Cannot find LettaBench questions in resources")
        
        return questions
    }
    
    private fun loadEvidenceDescription(): String {
        val evidence = StringBuilder()
        
        // Load people data for context
        this::class.java.getResourceAsStream("/datasets/letta_file_bench/data/people.txt")?.use { stream ->
            evidence.append(stream.bufferedReader().readText())
            evidence.append("\n\n")
        }
        
        // Load pets data
        this::class.java.getResourceAsStream("/datasets/letta_file_bench/data/pets.txt")?.use { stream ->
            evidence.append(stream.bufferedReader().readText())
            evidence.append("\n\n")
        }
        
        // Load vehicles data
        this::class.java.getResourceAsStream("/datasets/letta_file_bench/data/vehicles.txt")?.use { stream ->
            evidence.append(stream.bufferedReader().readText())
        }
        
        return evidence.toString()
    }
    
    override suspend fun populateMemorySystem(system: MemorySystem, dataset: BenchmarkDataset) = coroutineScope {
        // Load and add evidence context from resources
        val dataFiles = mapOf(
            "people" to "/datasets/letta_file_bench/data/people.jsonl",
            "vehicles" to "/datasets/letta_file_bench/data/vehicles.jsonl",
            "pets" to "/datasets/letta_file_bench/data/pets.jsonl",
            "addresses" to "/datasets/letta_file_bench/data/addresses.jsonl",
            "employment" to "/datasets/letta_file_bench/data/employments.jsonl",
            "medical" to "/datasets/letta_file_bench/data/medical_records.jsonl",
            "credit_cards" to "/datasets/letta_file_bench/data/credit_cards.jsonl",
            "bank_accounts" to "/datasets/letta_file_bench/data/bank_accounts.jsonl",
            "insurance" to "/datasets/letta_file_bench/data/insurance_policies.jsonl"
        )
        
        dataFiles.forEach { (type, resourcePath) ->
            this@LettaBenchLoader::class.java.getResourceAsStream(resourcePath)?.use { stream ->
                stream.bufferedReader().useLines { lines ->
                    lines.forEach { line ->
                        if (line.isNotBlank()) {
                            val data = Json.parseToJsonElement(line).jsonObject
                            val content = formatDataAsContent(type, data)
                            
                            system.addContext(
                                context = content,
                                metadata = mapOf(
                                    "type" to type,
                                    "entity_type" to type,
                                    "raw_data" to data.toString()
                                )
                            )
                        }
                    }
                }
            }
        }
    }
    
    private fun formatDataAsContent(type: String, data: JsonObject): String {
        return when (type) {
            "people" -> {
                val name = data["full_name"]?.jsonPrimitive?.content ?: ""
                val id = data["person_id"]?.jsonPrimitive?.content
                val phone = data["phone"]?.jsonPrimitive?.content
                val email = data["email"]?.jsonPrimitive?.content
                "Person: $name (ID: $id, Phone: $phone, Email: $email)"
            }
            "vehicles" -> {
                val owner = data["owner_id"]?.jsonPrimitive?.content
                val vehId = data["vehicle_id"]?.jsonPrimitive?.content
                val make = data["make"]?.jsonPrimitive?.content
                val model = data["model"]?.jsonPrimitive?.content
                val year = data["year"]?.jsonPrimitive?.content
                "Vehicle: $year $make $model (ID: $vehId, Owner: $owner)"
            }
            "pets" -> {
                val name = data["name"]?.jsonPrimitive?.content
                val petId = data["pet_id"]?.jsonPrimitive?.content
                val species = data["species"]?.jsonPrimitive?.content
                val breed = data["breed"]?.jsonPrimitive?.content
                val ownerId = data["owner_id"]?.jsonPrimitive?.content
                "Pet: $name the $species ($breed) (ID: $petId, Owner: $ownerId)"
            }
            else -> {
                // Generic format for other types
                val id = data.entries.firstOrNull { it.key.endsWith("_id") }?.value?.jsonPrimitive?.content
                "$type: ${data.toString()} (ID: $id)"
            }
        }
    }
}

/**
 * Stub loader for LongMemEval dataset
 */
class LongMemEvalLoader : DataLoader {
    override suspend fun load(path: String): BenchmarkDataset {
        // TODO: Implement resource loading for LongMemEval
        return BenchmarkDataset(
            name = "LongMemEval",
            questions = emptyList(),
            contextDescription = "Long-term memory evaluation over 115k token conversations",
            metadata = mapOf("stub" to true)
        )
    }
    
    override suspend fun populateMemorySystem(system: MemorySystem, dataset: BenchmarkDataset) {
        // TODO: Implement
    }
}

/**
 * Stub loader for DMR dataset
 */
class DMRLoader : DataLoader {
    override suspend fun load(path: String): BenchmarkDataset {
        // TODO: Implement resource loading for DMR
        return BenchmarkDataset(
            name = "Deep Memory Retrieval",
            questions = emptyList(),
            contextDescription = "Multi-session memory evaluation",
            metadata = mapOf("stub" to true)
        )
    }
    
    override suspend fun populateMemorySystem(system: MemorySystem, dataset: BenchmarkDataset) {
        // TODO: Implement
    }
}

/**
 * Stub loader for custom datasets
 */
class CustomDatasetLoader : DataLoader {
    override suspend fun load(path: String): BenchmarkDataset {
        // TODO: Implement custom dataset loading
        return BenchmarkDataset(
            name = "Custom Dataset",
            questions = emptyList(),
            contextDescription = "Custom dataset from: $path",
            metadata = mapOf("path" to path)
        )
    }
    
    override suspend fun populateMemorySystem(system: MemorySystem, dataset: BenchmarkDataset) {
        // TODO: Implement
    }
}