package ai.koog.agents.benchmark.io

import ai.koog.agents.benchmark.model.Dataset
import ai.koog.agents.benchmark.model.QAItem
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.json.Json

/**
 * Loads benchmark datasets from various formats
 */
public object DatasetLoader {
    private val json = Json { 
        ignoreUnknownKeys = true 
        isLenient = true
    }

    /**
     * Load a dataset from JSON string
     */
    public fun loadFromJson(jsonContent: String): Dataset {
        return json.decodeFromString(Dataset.serializer(), jsonContent)
    }
    
    /**
     * Load a dataset from JSONL (JSON Lines) format
     * Each line should be a valid QAItem JSON object
     */
    public fun loadFromJsonl(jsonlContent: String, datasetName: String): Dataset {
        val items = jsonlContent.lines()
            .filter { it.isNotBlank() }
            .map { line ->
                json.decodeFromString(
                    QAItem.serializer(), 
                    line
                )
            }
        
        return Dataset(name = datasetName, items = items)
    }
    
    /**
     * Load LettaBench format (complex multi-hop dataset)
     * Converts the LettaBench format to our standard QAItem format
     */
    public fun loadLettaBench(jsonlContent: String, datasetName: String): Dataset {
        val items = jsonlContent.lines()
            .filter { it.isNotBlank() }
            .mapIndexed { index, line ->
                val lettaItem = json.decodeFromString<LettaBenchItem>(line)
                convertLettaBenchItem(lettaItem, index)
            }
            .flatten()
        
        return Dataset(name = datasetName, items = items)
    }
    
    /**
     * Load LettaFileBench format (document-based questions)
     */
    public fun loadLettaFileBench(questionsPath: String, datasetName: String): Dataset {
        // This would load the questions from the LettaFileBench structure
        // For now, return empty dataset - can be implemented when needed
        return Dataset(name = datasetName, items = emptyList())
    }
    
    /**
     * Convert a LettaBench item to our QAItem format
     * LettaBench has multiple questions per item, so we split them
     */
    private fun convertLettaBenchItem(lettaItem: LettaBenchItem, baseIndex: Int): List<QAItem> {
        return lettaItem.question.mapIndexed { qIndex, question ->
            val id = "letta_${baseIndex}_${qIndex}"
            val answer = if (qIndex < lettaItem.answer.size) {
                lettaItem.answer[qIndex]
            } else {
                "Unknown" // Fallback
            }
            
            QAItem(
                id = id,
                question = question,
                goldAnswer = answer,
                entities = lettaItem.name,
                relations = null,
                metadata = mapOf(
                    "source" to "letta_bench",
                    "facts" to lettaItem.facts.joinToString(" | "),
                    "supporting_indices" to if (qIndex < lettaItem.supporting_fact_indices.size) {
                        lettaItem.supporting_fact_indices[qIndex].joinToString(",")
                    } else ""
                )
            )
        }
    }
    
    /**
     * Data class for LettaBench format
     */
    @Serializable
    private data class LettaBenchItem(
        val question: List<String>,
        val answer: List<String>,
        val facts: List<String>,
        val name: List<String>,
        val supporting_fact_indices: List<List<Int>>,
        val contradicting_facts: List<String>? = null,
        val contradicting_answers: List<String>? = null
    )
}