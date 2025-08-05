package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.comparative.debugParallelExecution
import kotlinx.coroutines.runBlocking
import kotlin.test.Test

class ParallelismDebugTest {
    
    @Test
    fun `verify parallelism is working`() = runBlocking {
        println("\n=== Testing Parallelism Implementation ===\n")
        
        // Test 1: Sequential (1 concurrent)
        println("Test 1: Sequential execution (maxConcurrency = 1)")
        debugParallelExecution(numTasks = 4, maxConcurrency = 1)
        
        println("\n" + "=".repeat(50) + "\n")
        
        // Test 2: Parallel (4 concurrent)
        println("Test 2: Parallel execution (maxConcurrency = 4)")
        debugParallelExecution(numTasks = 4, maxConcurrency = 4)
        
        println("\n" + "=".repeat(50) + "\n")
        
        // Test 3: Partial parallel (2 concurrent for 4 tasks)
        println("Test 3: Partial parallel (maxConcurrency = 2 for 4 tasks)")
        debugParallelExecution(numTasks = 4, maxConcurrency = 2)
    }
}