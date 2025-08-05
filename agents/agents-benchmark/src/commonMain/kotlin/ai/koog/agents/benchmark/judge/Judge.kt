package ai.koog.agents.benchmark.judge

import ai.koog.agents.benchmark.model.JudgeResult
import ai.koog.agents.benchmark.evaluation.AnswerEvaluator
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutorExt.firstResponseOrNull

/**
 * Interface for evaluating benchmark answers
 */
public interface Judge {
    /**
     * Grade a predicted answer against the gold standard
     */
    public suspend fun grade(
        question: String, 
        gold: String, 
        pred: String
    ): JudgeResult
}

/**
 * Improved judge using centralized answer evaluation logic
 */
public class SimpleJudge : Judge {
    override suspend fun grade(
        question: String, 
        gold: String, 
        pred: String
    ): JudgeResult {
        // Use centralized evaluator for consistent evaluation across the system
        val result = AnswerEvaluator.evaluateWithReason(pred, gold)
        return JudgeResult(result.correct, result.reason)
    }
}

/**
 * LLM-based judge for semantic evaluation
 */
public class LlmJudge(
    private val executor: ai.koog.prompt.executor.model.PromptExecutor? = null
) : Judge {
    override suspend fun grade(
        question: String, 
        gold: String, 
        pred: String
    ): JudgeResult {
        // If no executor, fall back to smart evaluation
        if (executor == null) {
            return SimpleJudge().grade(question, gold, pred)
        }
        
        // Use LLM for semantic evaluation
        val evalPrompt = prompt("judge-evaluation") {
            system("""You are an answer evaluation judge. Determine if the given answer is semantically equivalent to the expected answer.
                     |Consider: different phrasings with same meaning, numerical formats (2 vs two), partial correctness.
                     |Respond with only: YES or NO""".trimMargin())
            user("""
                |Question: $question
                |Expected Answer: $gold
                |Given Answer: $pred
                |
                |Are these answers semantically equivalent?
            """.trimMargin())
        }
        
        return try {
            val response = executor.execute(evalPrompt, ai.koog.prompt.executor.clients.openai.OpenAIModels.CostOptimized.GPT4oMini, emptyList())
                .firstResponseOrNull()?.content?.trim()?.uppercase() ?: "NO"
            val isCorrect = response.contains("YES")
            JudgeResult(
                correct = isCorrect,
                reason = if (isCorrect) "LLM: Semantically equivalent" else "LLM: Not equivalent"
            )
        } catch (e: Exception) {
            // Fallback to simple judge
            SimpleJudge().grade(question, gold, pred)
        }
    }
}