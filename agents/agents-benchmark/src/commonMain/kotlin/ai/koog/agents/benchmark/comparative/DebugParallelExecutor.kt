package ai.koog.agents.benchmark.comparative

import kotlinx.coroutines.*
import kotlinx.coroutines.sync.Semaphore
import kotlinx.coroutines.sync.withPermit
import kotlinx.datetime.Clock

/**
 * Debug version of parallel executor to verify parallelism
 */
suspend fun debugParallelExecution(numTasks: Int, maxConcurrency: Int) = coroutineScope {
    println("\n🔍 Debug Parallel Execution Test")
    println("Tasks: $numTasks, Max Concurrency: $maxConcurrency")
    println("Expected behavior: Should see up to $maxConcurrency tasks running simultaneously\n")
    
    val semaphore = kotlinx.coroutines.sync.Semaphore(maxConcurrency)
    val startTime = Clock.System.now()
    
    val jobs = (1..numTasks).map { taskId ->
        async<Int> {
            semaphore.withPermit {
                val taskStart = Clock.System.now()
                val startOffset = taskStart.toEpochMilliseconds() - startTime.toEpochMilliseconds()
                println("[${String.format("%4d", startOffset)}ms] Task $taskId: Starting")
                
                // Simulate LLM call with 1 second delay
                delay(1000)
                
                val taskEnd = Clock.System.now()
                val endOffset = taskEnd.toEpochMilliseconds() - startTime.toEpochMilliseconds()
                println("[${String.format("%4d", endOffset)}ms] Task $taskId: Completed")
                
                taskId
            }
        }
    }
    
    val results = jobs.awaitAll()
    val totalTime = Clock.System.now().toEpochMilliseconds() - startTime.toEpochMilliseconds()
    
    println("\n📊 Summary:")
    println("Total time: ${totalTime}ms")
    println("Expected sequential time: ${numTasks * 1000}ms")
    println("Actual speedup: ${(numTasks * 1000.0 / totalTime).format(2)}x")
    println("Theoretical max speedup: ${minOf(numTasks, maxConcurrency)}x")
}

private fun Double.format(decimals: Int): String = "%.${decimals}f".format(this)