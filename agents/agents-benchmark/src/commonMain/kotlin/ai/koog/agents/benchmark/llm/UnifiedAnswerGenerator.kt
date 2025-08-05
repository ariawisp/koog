package ai.koog.agents.benchmark.llm

import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.executor.clients.openai.OpenAIModels
import ai.koog.agents.benchmark.runners.AnswerStrategy
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutorExt.firstResponseOrNull

/**
 * Unified answer generator that replaces LLMAnswerGenerator
 * Uses proven prompts from top-performing systems
 */
class UnifiedAnswerGenerator(
    private val strategy: AnswerStrategy = AnswerStrategy.CHAIN_OF_THOUGHT
) {
    
    suspend fun generateAnswer(
        question: String,
        context: List<String>,
        executor: PromptExecutor,
        model: String? = null
    ): String {
        
        // Filter and rank context
        val relevantContext = selectRelevantContext(question, context)
        
        // Generate answer based on strategy
        return when (strategy) {
            AnswerStrategy.SIMPLE -> simpleAnswer(question, relevantContext, executor, model)
            AnswerStrategy.CHAIN_OF_THOUGHT -> chainOfThoughtAnswer(question, relevantContext, executor, model)
            AnswerStrategy.MULTI_STEP -> multiStepAnswer(question, relevantContext, executor, model)
        }
    }
    
    private fun selectRelevantContext(question: String, context: List<String>): String {
        // Smart context selection based on question keywords
        val keywords = extractKeywords(question)
        
        val scored = context.map { ctx ->
            val score = keywords.count { keyword ->
                ctx.contains(keyword, ignoreCase = true)
            }
            ctx to score
        }
        
        // Take top contexts up to token limit
        return scored
            .sortedByDescending { it.second }
            .take(10) // Limit context items
            .joinToString("\n\n") { it.first }
    }
    
    private fun extractKeywords(question: String): List<String> {
        // Extract important words from question
        val stopWords = setOf("who", "what", "when", "where", "why", "how", "is", "are", "the", "a", "an")
        
        return question
            .lowercase()
            .split(Regex("\\W+"))
            .filter { it.length > 2 && it !in stopWords }
            .distinct()
    }
    
    private suspend fun simpleAnswer(
        question: String,
        context: String,
        executor: PromptExecutor,
        model: String?
    ): String {
        val answerPrompt = prompt("simple-answer") {
            system("""You are a helpful assistant answering questions based on provided context.
                     |Instructions:
                     |- Answer ONLY based on the information in the context
                     |- If the answer is not in the context, say "Information not found in context"
                     |- Be concise and direct
                     |- For comparison questions (who has more), state the specific counts if available""".trimMargin())
            user("""
                |Context:
                |$context
                |
                |Question: $question
                |
                |Answer:
            """.trimMargin())
        }
        
        val response = executor.execute(
            prompt = answerPrompt,
            model = OpenAIModels.CostOptimized.GPT4oMini,
            tools = emptyList()
        )
        
        return response.firstResponseOrNull()?.content?.trim() ?: "No answer generated"
    }
    
    private suspend fun chainOfThoughtAnswer(
        question: String,
        context: String,
        executor: PromptExecutor,
        model: String?
    ): String {
        val cotPrompt = prompt("chain-of-thought") {
            system("You are analyzing context to answer a question. Think step by step.")
            user("""
                |Context:
                |$context
                |
                |Question: $question
                |
                |Let's work through this systematically:
                |
                |1. First, identify what specific information the question is asking for
                |2. Search the context for relevant facts
                |3. Connect the facts to form an answer
                |4. Verify the answer makes sense
                |
                |Work through these steps, then provide your final answer.
                |
                |Final Answer:
            """.trimMargin())
        }
        
        val response = executor.execute(
            prompt = cotPrompt,
            model = OpenAIModels.CostOptimized.GPT4oMini,
            tools = emptyList()
        )
        
        // Extract final answer
        val fullResponse = response.firstResponseOrNull()?.content ?: return "No answer generated"
        
        return if (fullResponse.contains("Final Answer:")) {
            fullResponse.substringAfter("Final Answer:").trim()
        } else {
            // Take last paragraph as answer
            fullResponse.split("\n\n").lastOrNull()?.trim() ?: fullResponse.trim()
        }
    }
    
    private suspend fun multiStepAnswer(
        question: String,
        context: String,
        executor: PromptExecutor,
        model: String?
    ): String {
        // Step 1: Understand what's needed
        val analysisPrompt = prompt("multi-step-analysis") {
            system("You are an analytical assistant that breaks down questions.")
            user("""
                |Question: $question
                |
                |What specific pieces of information do I need to find to answer this question?
                |List them:
            """.trimMargin())
        }
        
        val analysis = executor.execute(
            prompt = analysisPrompt,
            model = OpenAIModels.CostOptimized.GPT4oMini,
            tools = emptyList()
        ).firstResponseOrNull()?.content ?: ""
        
        // Step 2: Extract information
        val extractionPrompt = prompt("multi-step-extraction") {
            system("You are an information extraction assistant.")
            user("""
                |I need to find: $analysis
                |
                |Context:
                |$context
                |
                |Extract the relevant information:
            """.trimMargin())
        }
        
        val extraction = executor.execute(
            prompt = extractionPrompt,
            model = OpenAIModels.CostOptimized.GPT4oMini,
            tools = emptyList()
        ).firstResponseOrNull()?.content ?: ""
        
        // Step 3: Synthesize answer
        val synthesisPrompt = prompt("multi-step-synthesis") {
            system("You are an answer synthesis assistant.")
            user("""
                |Question: $question
                |
                |Information found:
                |$extraction
                |
                |Based on this information, the answer is:
            """.trimMargin())
        }
        
        val response = executor.execute(
            prompt = synthesisPrompt,
            model = OpenAIModels.CostOptimized.GPT4oMini,
            tools = emptyList()
        )
        
        return response.firstResponseOrNull()?.content?.trim() ?: "No answer generated"
    }
}

/**
 * Factory for creating answer generators
 */
object AnswerGeneratorFactory {
    
    fun create(strategy: AnswerStrategy = AnswerStrategy.CHAIN_OF_THOUGHT): UnifiedAnswerGenerator {
        return UnifiedAnswerGenerator(strategy)
    }
    
    fun createOptimal(questionType: String): UnifiedAnswerGenerator {
        return when {
            questionType.contains("comparison") -> UnifiedAnswerGenerator(AnswerStrategy.SIMPLE)
            questionType.contains("multi-hop") -> UnifiedAnswerGenerator(AnswerStrategy.MULTI_STEP)
            else -> UnifiedAnswerGenerator(AnswerStrategy.CHAIN_OF_THOUGHT)
        }
    }
}