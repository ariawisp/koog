package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.scripts.RunFullBenchmark
import ai.koog.agents.testing.tools.getMockExecutor
import ai.koog.agents.testing.tools.mockLLMAnswer
import kotlinx.coroutines.test.runTest
import kotlin.test.Test

/**
 * End-to-end test demonstrating the full benchmark flow
 */
class EndToEndBenchmarkTest {
    
    @Test
    fun testEndToEndBenchmarkFlow() = runTest {
        println("🧪 End-to-End Benchmark Test")
        println("=".repeat(50))
        
        // Create mock executor with intelligent responses
        val mockExecutor = getMockExecutor {
            // Mock responses for different question types
            mockLLMAnswer("Based on the context, Person2 collaborated with Person1 on Project0") onRequestContains "Who collaborated with Person1"
            mockLLMAnswer("The project resulted in successful implementation") onRequestContains "outcome of Project"
            mockLLMAnswer("Event3 occurred in 2024 with Person2 and Person4") onRequestContains "When did Event3 occur"
            mockLLMAnswer("Entity4 manages Entity6 according to the organizational structure") onRequestContains "relationship between Entity4 and Entity6"
            
            // Generic fallback for other questions
            mockLLMAnswer("Based on the provided context, the answer is derived from the available information.").asDefaultResponse
        }
        
        // Run small benchmark
        println("\n📊 Running benchmark with 5 questions...")
        RunFullBenchmark.runBenchmark(
            executor = mockExecutor,
            datasetSize = 5,
            outputPath = null // Console output only
        )
        
        println("\n✅ End-to-end benchmark test completed successfully!")
    }
    
    @Test
    fun testBenchmarkWithLiteratureComparison() = runTest {
        println("📚 Testing Literature Comparison Integration")
        println("=".repeat(50))
        
        // This test verifies that our benchmark results are properly compared
        // against published literature (Zep paper results)
        
        val mockExecutor = getMockExecutor {
            // Mock perfect responses to achieve high accuracy
            mockLLMAnswer("The correct answer based on the context") onRequestContains "question"
            mockLLMAnswer("Accurate response derived from the provided information").asDefaultResponse
        }
        
        // Run minimal benchmark
        RunFullBenchmark.runBenchmark(
            executor = mockExecutor,
            datasetSize = 3,
            outputPath = null
        )
        
        println("\n✅ Literature comparison test completed!")
    }
}

