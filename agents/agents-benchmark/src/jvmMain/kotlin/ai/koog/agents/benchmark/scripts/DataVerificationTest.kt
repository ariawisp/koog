package ai.koog.agents.benchmark.scripts

import kotlinx.coroutines.runBlocking
import kotlinx.serialization.json.*

/**
 * Verify that the entities mentioned in questions actually exist in the dataset
 */
suspend fun main() {
    println("🔍 LettaBench Data Verification")
    println("=" * 50)
    
    // Names to look for from the first few questions
    val targetPeople = listOf(
        "Jerry Stevens", "Grant Martinez", "Julie Saunders", "Lisa Norris",
        "Brian Bates", "Jenny Donaldson", "Jamie Nelson"
    )
    
    val targetVehicles = listOf("veh-0069", "veh-0063", "veh-0238")
    
    // Check people data
    println("\n📚 Checking people data...")
    val foundPeople = mutableMapOf<String, String>()
    var totalPeople = 0
    
    object {}.javaClass.getResourceAsStream("/datasets/letta_file_bench/data/people.jsonl")?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) {
                    totalPeople++
                    val data = Json.parseToJsonElement(line).jsonObject
                    val firstName = data["first_name"]?.jsonPrimitive?.content ?: ""
                    val lastName = data["last_name"]?.jsonPrimitive?.content ?: ""
                    val fullName = "$firstName $lastName"
                    val id = data["person_id"]?.jsonPrimitive?.content ?: ""
                    val phone = data["phone"]?.jsonPrimitive?.content ?: ""
                    
                    if (targetPeople.contains(fullName)) {
                        foundPeople[fullName] = "$id (phone: $phone)"
                    }
                }
            }
        }
    }
    
    println("Total people in dataset: $totalPeople")
    println("\nTarget people found:")
    targetPeople.forEach { name ->
        if (foundPeople.containsKey(name)) {
            println("✅ $name: ${foundPeople[name]}")
        } else {
            println("❌ $name: NOT FOUND")
        }
    }
    
    // Check vehicle data
    println("\n\n📚 Checking vehicle data...")
    val foundVehicles = mutableMapOf<String, String>()
    val vehiclesByOwner = mutableMapOf<String, MutableList<String>>()
    var totalVehicles = 0
    
    object {}.javaClass.getResourceAsStream("/datasets/letta_file_bench/data/vehicles.jsonl")?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.forEach { line ->
                if (line.isNotBlank()) {
                    totalVehicles++
                    val data = Json.parseToJsonElement(line).jsonObject
                    val vehId = data["vehicle_id"]?.jsonPrimitive?.content ?: ""
                    val ownerId = data["owner_id"]?.jsonPrimitive?.content ?: ""
                    val make = data["make"]?.jsonPrimitive?.content ?: ""
                    val model = data["model"]?.jsonPrimitive?.content ?: ""
                    
                    if (targetVehicles.contains(vehId)) {
                        foundVehicles[vehId] = "Owner: $ownerId, $make $model"
                    }
                    
                    // Track vehicles by owner
                    vehiclesByOwner.getOrPut(ownerId) { mutableListOf() }.add(vehId)
                }
            }
        }
    }
    
    println("Total vehicles in dataset: $totalVehicles")
    println("\nTarget vehicles found:")
    targetVehicles.forEach { vehId ->
        if (foundVehicles.containsKey(vehId)) {
            println("✅ $vehId: ${foundVehicles[vehId]}")
        } else {
            println("❌ $vehId: NOT FOUND")
        }
    }
    
    // Check vehicle ownership for found people
    println("\n\n📊 Vehicle ownership for found people:")
    foundPeople.forEach { (name, info) ->
        val personId = info.substringBefore(" ")
        val vehicles = vehiclesByOwner[personId] ?: emptyList()
        println("$name: ${vehicles.size} vehicles")
    }
    
    // Sample some actual data to see what's there
    println("\n\n📋 Sample of actual people in dataset:")
    var sampleCount = 0
    object {}.javaClass.getResourceAsStream("/datasets/letta_file_bench/data/people.jsonl")?.use { stream ->
        stream.bufferedReader().useLines { lines ->
            lines.take(10).forEach { line ->
                if (line.isNotBlank()) {
                    val data = Json.parseToJsonElement(line).jsonObject
                    val firstName = data["first_name"]?.jsonPrimitive?.content ?: ""
                    val lastName = data["last_name"]?.jsonPrimitive?.content ?: ""
                    val id = data["person_id"]?.jsonPrimitive?.content ?: ""
                    println("  - $firstName $lastName (ID: $id)")
                }
            }
        }
    }
}

private operator fun String.times(n: Int): String = repeat(n)