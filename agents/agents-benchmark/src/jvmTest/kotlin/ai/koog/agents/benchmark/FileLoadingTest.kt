package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.io.DatasetLoader
import ai.koog.agents.benchmark.io.JvmFileLoader
import ai.koog.agents.benchmark.io.loadDatasetFile
import ai.koog.agents.benchmark.scripts.RunFullBenchmark
import ai.koog.prompt.executor.clients.openai.OpenAIClientSettings
import ai.koog.prompt.executor.clients.openai.OpenAILLMClient
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.test.runTest
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Test file loading functionality
 */
class FileLoadingTest {
    
    @Test
    fun testLoadSampleDataset() = runTest {
        // Check if sample dataset exists
        val sampleFile = File("sample-dataset.jsonl")
        if (!sampleFile.exists()) {
            println("Sample dataset not found, skipping test")
            return@runTest
        }
        
        // Load the dataset
        val content = loadDatasetFile(sampleFile.absolutePath)
        assertTrue(content.isNotEmpty(), "Dataset content should not be empty")
        
        // Parse as JSONL
        val dataset = DatasetLoader.loadFromJsonl(content, "test-dataset")
        assertEquals(10, dataset.items.size, "Sample dataset should have 10 questions")
        
        // Check first question
        val firstQuestion = dataset.items.first()
        assertTrue(firstQuestion.question.contains("quantum computing"), "First question should be about quantum computing")
        assertEquals("Prof. Michael Zhang", firstQuestion.goldAnswer)
        
        println("✅ Successfully loaded dataset with ${dataset.items.size} questions")
    }
    
    @Test
    fun testDatasetDownloader() = runTest {
        // List available datasets
        val availableDatasets = DatasetDownloader.availableDatasets
        assertTrue(availableDatasets.isNotEmpty(), "Should have available datasets")
        
        println("Available datasets:")
        availableDatasets.forEach { dataset ->
            println("  - ${dataset.name}: ${dataset.description}")
        }
        
        // List local datasets
        val localDatasets = DatasetDownloader.listLocalDatasets()
        println("\nLocal datasets: ${localDatasets.size}")
        localDatasets.forEach { file ->
            println("  - ${file.name} (${file.length()} bytes)")
        }
    }
    
    @Test
    fun testFullBenchmarkWithRealDataset() = runTest {
        // Check if sample dataset exists
        val sampleFile = File("sample-dataset.jsonl")
        if (!sampleFile.exists()) {
            println("Sample dataset not found, skipping test")
            return@runTest
        }
        
        // Check if LMStudio is available
        val executor = try {
            val settings = OpenAIClientSettings(baseUrl = "http://localhost:1234")
            SingleLLMPromptExecutor(OpenAILLMClient("lm-studio", settings))
        } catch (e: Exception) {
            println("LMStudio not available, skipping test")
            return@runTest
        }
        
        println("🚀 Running benchmark with real dataset")
        
        // Run benchmark with real dataset
        runBlocking {
            RunFullBenchmark.runBenchmark(
                executor = executor,
                datasetPath = sampleFile.absolutePath,
                datasetSize = 5, // Only run first 5 questions
                outputPath = null
            )
        }
    }
    
    @Test
    fun testFileInfo() {
        val sampleFile = File("sample-dataset.jsonl")
        if (!sampleFile.exists()) {
            println("Sample dataset not found, creating test file")
            sampleFile.writeText("""{"id": "test", "question": "Test?", "goldAnswer": "Test"}""")
        }
        
        val fileInfo = JvmFileLoader.getFileInfo(sampleFile.absolutePath)
        
        println("File info:")
        println("  Path: ${fileInfo.path}")
        println("  Exists: ${fileInfo.exists}")
        println("  Size: ${fileInfo.size} bytes")
        println("  Extension: ${fileInfo.extension}")
        println("  Can read: ${fileInfo.canRead}")
        
        assertTrue(fileInfo.exists, "File should exist")
        assertEquals("jsonl", fileInfo.extension, "Should be JSONL file")
    }
}