package ai.koog.agents.benchmark.llm

import ai.koog.agents.memory.retrieval.RetrievalResult
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.executor.clients.openai.OpenAIModels

/**
 * Generates answers using an LLM based on retrieved context
 */
public class LLMAnswerGenerator(
    private val executor: PromptExecutor,
    private val model: LLModel = OpenAIModels.CostOptimized.GPT4_1Mini
) {
    
    /**
     * Generate an answer to a question using retrieved context
     */
    public suspend fun generateAnswer(
        question: String,
        context: List<RetrievalResult>
    ): String {
        if (context.isEmpty()) {
            return "I don't have enough information to answer this question."
        }
        
        // Create context string from retrieval results
        val contextText = context
            .sortedByDescending { it.score }
            .take(5) // Use top 5 most relevant results
            .joinToString("\n\n") { "Context: ${it.content}" }
        
        val prompt = prompt("qa-answer") {
            system("""You are a helpful assistant that answers questions based on provided context.
                
Rules:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Be concise and direct
4. For questions requiring multi-step reasoning, work through the steps using the context
5. Do not make up information not present in the context""")
            
            user("""Context Information:
$contextText

Question: $question

Answer:""")
        }
        
        return try {
            val result = executor.execute(prompt, model, emptyList())
            result.firstOrNull()?.content ?: "No response generated"
        } catch (e: Exception) {
            "Error generating answer: ${e.message}"
        }
    }
}