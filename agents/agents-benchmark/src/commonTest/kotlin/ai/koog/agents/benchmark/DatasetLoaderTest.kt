package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.io.DatasetLoader
import ai.koog.agents.benchmark.model.Dataset
import ai.koog.agents.benchmark.model.QAItem
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class DatasetLoaderTest {
    
    @Test
    fun testLoadFromJsonl() {
        val jsonlContent = """
            {"id": "test1", "question": "What color is the sky?", "goldAnswer": "Blue", "entities": ["sky"], "relations": [], "metadata": {"type": "simple"}}
            {"id": "test2", "question": "What is 2+2?", "goldAnswer": "4", "entities": [], "relations": [], "metadata": {"type": "math"}}
        """.trimIndent()
        
        val dataset = DatasetLoader.loadFromJsonl(jsonlContent, "test-dataset")
        
        assertEquals("test-dataset", dataset.name)
        assertEquals(2, dataset.items.size)
        
        val first = dataset.items[0]
        assertEquals("test1", first.id)
        assertEquals("What color is the sky?", first.question)
        assertEquals("Blue", first.goldAnswer)
        assertEquals(listOf("sky"), first.entities)
        assertTrue(first.metadata?.get("type") == "simple")
        
        val second = dataset.items[1]
        assertEquals("test2", second.id)
        assertEquals("What is 2+2?", second.question)
        assertEquals("4", second.goldAnswer)
    }
    
    @Test
    fun testLoadFromJson() {
        val jsonContent = """
        {
            "name": "sample-dataset",
            "items": [
                {
                    "id": "q1",
                    "question": "Who is the president?",
                    "goldAnswer": "Joe Biden",
                    "entities": ["president", "Joe Biden"],
                    "relations": ["is"],
                    "metadata": {"category": "politics"}
                }
            ]
        }
        """.trimIndent()
        
        val dataset = DatasetLoader.loadFromJson(jsonContent)
        
        assertEquals("sample-dataset", dataset.name)
        assertEquals(1, dataset.items.size)
        
        val item = dataset.items[0]
        assertEquals("q1", item.id)
        assertEquals("Who is the president?", item.question)
        assertEquals("Joe Biden", item.goldAnswer)
        assertEquals(listOf("president", "Joe Biden"), item.entities)
        assertEquals(listOf("is"), item.relations)
        assertTrue(item.metadata?.get("category") == "politics")
    }
}