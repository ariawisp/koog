package ai.koog.agents.benchmark.comparative

import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import kotlinx.datetime.Clock

/**
 * Baseline system implementations for comparative benchmarking
 */

/**
 * Simple RAG (Retrieval-Augmented Generation) baseline
 * Just concatenates context and asks the LLM directly
 */
class SimpleRAG(
    private val executor: PromptExecutor,
    private val model: LLModel = OpenAIModels.CostOptimized.GPT4_1Mini
) : QASystem {
    
    override val name = "SimpleRAG"
    
    override suspend fun answer(question: String, context: List<String>): QAResponse {
        val startTime = Clock.System.now()
        
        // Simple context concatenation
        val contextText = context.joinToString("\n\n") { "Context: $it" }
        
        val prompt = prompt("simple-rag") {
            system("Answer the question based only on the provided context. Be concise and direct.")
            user("""Context:
$contextText

Question: $question

Answer:""")
        }
        
        return try {
            val result = executor.execute(prompt, model, emptyList())
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            QAResponse(
                answer = result.firstOrNull()?.content ?: "No response",
                latencyMs = latency,
                tokensUsed = estimateTokens(prompt.toString() + (result.firstOrNull()?.content ?: "")),
                confidence = 1.0
            )
        } catch (e: Exception) {
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            QAResponse(
                answer = "Error: ${e.message}",
                latencyMs = latency,
                tokensUsed = 0,
                confidence = 0.0
            )
        }
    }
    
    private fun estimateTokens(text: String): Int = text.length / 4
}

/**
 * Chain-of-Thought reasoning baseline
 * Explicitly asks the LLM to show its reasoning steps
 */
class ChainOfThoughtRAG(
    private val executor: PromptExecutor,
    private val model: LLModel = OpenAIModels.CostOptimized.GPT4_1Mini
) : QASystem {
    
    override val name = "ChainOfThought"
    
    override suspend fun answer(question: String, context: List<String>): QAResponse {
        val startTime = Clock.System.now()
        
        val contextText = context.joinToString("\n\n") { "- $it" }
        
        val prompt = prompt("cot-rag") {
            system("""Answer the question step by step using the provided context.

Instructions:
1. First, identify the relevant information from the context
2. Then, reason through the steps needed to answer the question
3. Finally, provide a clear, concise answer

Format your response as:
Reasoning: [your step-by-step thinking]
Answer: [final answer]""")
            
            user("""Context:
$contextText

Question: $question""")
        }
        
        return try {
            val result = executor.execute(prompt, model, emptyList())
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            val fullResponse = result.firstOrNull()?.content ?: "No response"
            
            // Extract final answer after "Answer:" if present
            val answer = if (fullResponse.contains("Answer:")) {
                fullResponse.substringAfterLast("Answer:").trim()
            } else {
                fullResponse
            }
            
            QAResponse(
                answer = answer,
                latencyMs = latency,
                tokensUsed = estimateTokens(prompt.toString() + fullResponse),
                reasoning = fullResponse,
                confidence = 1.0
            )
        } catch (e: Exception) {
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            QAResponse(
                answer = "Error: ${e.message}",
                latencyMs = latency,
                tokensUsed = 0,
                confidence = 0.0
            )
        }
    }
    
    private fun estimateTokens(text: String): Int = text.length / 4
}

/**
 * ReAct (Reasoning + Acting) pattern baseline
 * Simulates iterative reasoning and action-taking process
 */
class ReActRAG(
    private val executor: PromptExecutor,
    private val model: LLModel = OpenAIModels.CostOptimized.GPT4_1Mini,
    private val maxSteps: Int = 3
) : QASystem {
    
    override val name = "ReAct"
    
    override suspend fun answer(question: String, context: List<String>): QAResponse {
        val startTime = Clock.System.now()
        var totalTokens = 0
        val reasoningSteps = mutableListOf<String>()
        
        // Initial context setup
        val contextText = context.joinToString("\n") { "Fact: $it" }
        
        var currentThought = "I need to analyze the context to answer this question."
        
        for (step in 1..maxSteps) {
            val prompt = prompt("react-step-$step") {
                system("""You are solving a question step by step using ReAct (Reasoning + Acting).

Available context:
$contextText

For each step, think about what you need to find, then search the context.

Format:
Thought: [what you're thinking]
Action: [what information you're looking for]
Observation: [what you found in the context]""")
                
                user("""Question: $question

Previous reasoning:
${reasoningSteps.joinToString("\n")}

Current thought: $currentThought

Continue reasoning:""")
            }
            
            try {
                val result = executor.execute(prompt, model, emptyList())
                val response = result.firstOrNull()?.content ?: ""
                totalTokens += estimateTokens(prompt.toString() + response)
                
                reasoningSteps.add("Step $step: $response")
                
                // Extract next thought or final answer
                if (response.contains("Answer:") || step == maxSteps) {
                    // Final step - extract answer
                    val answer = if (response.contains("Answer:")) {
                        response.substringAfterLast("Answer:").trim()
                    } else if (response.contains("Observation:")) {
                        response.substringAfterLast("Observation:").trim()
                    } else {
                        response.trim()
                    }
                    
                    val endTime = Clock.System.now()
                    val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
                    
                    return QAResponse(
                        answer = answer,
                        latencyMs = latency,
                        tokensUsed = totalTokens,
                        reasoning = reasoningSteps.joinToString("\n"),
                        confidence = 1.0
                    )
                }
                
                // Extract next thought
                currentThought = if (response.contains("Thought:")) {
                    response.substringAfter("Thought:").substringBefore("Action:").trim()
                } else {
                    "Continue analyzing the context."
                }
                
            } catch (e: Exception) {
                break
            }
        }
        
        val endTime = Clock.System.now()
        val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
        
        return QAResponse(
            answer = "Unable to complete reasoning process",
            latencyMs = latency,
            tokensUsed = totalTokens,
            reasoning = reasoningSteps.joinToString("\n"),
            confidence = 0.5
        )
    }
    
    private fun estimateTokens(text: String): Int = text.length / 4
}

/**
 * No-context baseline - just ask the LLM without any context
 * This shows the value of having a memory/retrieval system
 */
class NoContextBaseline(
    private val executor: PromptExecutor,
    private val model: LLModel = OpenAIModels.CostOptimized.GPT4_1Mini
) : QASystem {
    
    override val name = "NoContext"
    
    override suspend fun answer(question: String, context: List<String>): QAResponse {
        val startTime = Clock.System.now()
        
        val prompt = prompt("no-context") {
            system("Answer the question based on your knowledge. Be concise and direct.")
            user("Question: $question\n\nAnswer:")
        }
        
        return try {
            val result = executor.execute(prompt, model, emptyList())
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            QAResponse(
                answer = result.firstOrNull()?.content ?: "No response",
                latencyMs = latency,
                tokensUsed = estimateTokens(prompt.toString() + (result.firstOrNull()?.content ?: "")),
                confidence = 0.5 // Lower confidence since no context
            )
        } catch (e: Exception) {
            val endTime = Clock.System.now()
            val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
            
            QAResponse(
                answer = "Error: ${e.message}",
                latencyMs = latency,
                tokensUsed = 0,
                confidence = 0.0
            )
        }
    }
    
    private fun estimateTokens(text: String): Int = text.length / 4
}

/**
 * Random baseline - provides random answers from a predefined set
 * This establishes the absolute floor for comparison
 */
class RandomBaseline : QASystem {
    
    override val name = "Random"
    
    private val randomAnswers = listOf(
        "Yes", "No", "Unknown", "Maybe", "True", "False",
        "Paris", "London", "New York", "California", "John Smith",
        "2023", "Yesterday", "Never", "Always", "Sometimes"
    )
    
    override suspend fun answer(question: String, context: List<String>): QAResponse {
        val startTime = Clock.System.now()
        
        // Simulate some processing time
        kotlinx.coroutines.delay(50)
        
        val endTime = Clock.System.now()
        val latency = endTime.toEpochMilliseconds() - startTime.toEpochMilliseconds()
        
        return QAResponse(
            answer = randomAnswers.random(),
            latencyMs = latency,
            tokensUsed = 10, // Minimal tokens
            confidence = 0.1 // Very low confidence
        )
    }
}