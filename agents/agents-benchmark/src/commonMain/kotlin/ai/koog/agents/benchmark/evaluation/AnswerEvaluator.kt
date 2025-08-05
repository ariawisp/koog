package ai.koog.agents.benchmark.evaluation

import ai.koog.agents.benchmark.BenchmarkConstants

/**
 * Centralized answer evaluation logic for benchmarks
 * 
 * This consolidates the answer evaluation logic that was duplicated across:
 * - SimpleJudge
 * - ComparativeBenchmarkExecutor  
 * - ParallelBenchmarkExecutor
 */
object AnswerEvaluator {
    
    /**
     * Evaluate if a predicted answer matches the gold answer
     * 
     * Uses semantic comparison with multiple strategies:
     * 1. Exact match (case-insensitive)
     * 2. Contains match (gold answer contained in prediction)
     * 3. Key word matching (for multi-word answers)
     * 
     * @param predicted The system's predicted answer
     * @param gold The correct gold answer
     * @return true if the answers match semantically
     */
    fun evaluate(predicted: String, gold: String): Boolean {
        val cleanGold = gold.lowercase().trim()
        val cleanPred = predicted.lowercase().trim()
        
        // Strategy 1: Exact match
        if (cleanPred == cleanGold) return true
        
        // Strategy 2: Gold answer contained in prediction
        // This handles cases where the model provides extra context
        if (cleanPred.contains(cleanGold)) return true
        
        // Strategy 3: Key words match for multi-word answers
        val goldWords = cleanGold.split(Regex("\\s+"))
            .filter { it.length > BenchmarkConstants.MIN_WORD_LENGTH_FOR_MATCHING }
            
        if (goldWords.isNotEmpty()) {
            val predWords = cleanPred.split(Regex("\\s+"))
            val matchedWords = goldWords.count { goldWord ->
                predWords.any { predWord -> 
                    predWord.contains(goldWord) || goldWord.contains(predWord)
                }
            }
            val matchRatio = matchedWords.toDouble() / goldWords.size
            if (matchRatio >= BenchmarkConstants.WORD_MATCH_THRESHOLD) return true
        }
        
        // Strategy 4: Handle numeric answers
        if (isNumericMatch(cleanPred, cleanGold)) return true
        
        // Strategy 5: Handle yes/no answers
        if (isYesNoMatch(cleanPred, cleanGold)) return true
        
        return false
    }
    
    /**
     * Check if two answers are numerically equivalent
     */
    private fun isNumericMatch(pred: String, gold: String): Boolean {
        // Extract numbers from both strings
        val predNumbers = extractNumbers(pred)
        val goldNumbers = extractNumbers(gold)
        
        // If both contain exactly one number, compare them
        if (predNumbers.size == 1 && goldNumbers.size == 1) {
            return kotlin.math.abs(predNumbers[0] - goldNumbers[0]) < 0.001
        }
        
        return false
    }
    
    /**
     * Extract all numbers from a string
     */
    private fun extractNumbers(text: String): List<Double> {
        val pattern = Regex("-?\\d+\\.?\\d*")
        return pattern.findAll(text)
            .mapNotNull { it.value.toDoubleOrNull() }
            .toList()
    }
    
    /**
     * Check if answers are yes/no equivalents
     */
    private fun isYesNoMatch(pred: String, gold: String): Boolean {
        val yesPatterns = setOf("yes", "correct", "true", "affirmative", "right")
        val noPatterns = setOf("no", "incorrect", "false", "negative", "wrong")
        
        val predIsYes = yesPatterns.any { pred.contains(it) }
        val predIsNo = noPatterns.any { pred.contains(it) }
        val goldIsYes = yesPatterns.any { gold.contains(it) }
        val goldIsNo = noPatterns.any { gold.contains(it) }
        
        return (predIsYes && goldIsYes) || (predIsNo && goldIsNo)
    }
    
    /**
     * Evaluate with detailed reasoning for debugging
     */
    fun evaluateWithReason(predicted: String, gold: String): EvaluationResult {
        val cleanGold = gold.lowercase().trim()
        val cleanPred = predicted.lowercase().trim()
        
        // Check each strategy and return reason
        if (cleanPred == cleanGold) {
            return EvaluationResult(true, "Exact match")
        }
        
        if (cleanPred.contains(cleanGold)) {
            return EvaluationResult(true, "Gold answer contained in prediction")
        }
        
        val goldWords = cleanGold.split(Regex("\\s+"))
            .filter { it.length > BenchmarkConstants.MIN_WORD_LENGTH_FOR_MATCHING }
            
        if (goldWords.isNotEmpty()) {
            val predWords = cleanPred.split(Regex("\\s+"))
            val matchedWords = goldWords.count { goldWord ->
                predWords.any { predWord -> 
                    predWord.contains(goldWord) || goldWord.contains(predWord)
                }
            }
            val matchRatio = matchedWords.toDouble() / goldWords.size
            if (matchRatio >= BenchmarkConstants.WORD_MATCH_THRESHOLD) {
                return EvaluationResult(
                    true, 
                    "Key word match (${(matchRatio * 100).toInt()}% words matched)"
                )
            }
        }
        
        if (isNumericMatch(cleanPred, cleanGold)) {
            return EvaluationResult(true, "Numeric equivalence")
        }
        
        if (isYesNoMatch(cleanPred, cleanGold)) {
            return EvaluationResult(true, "Yes/No equivalence")
        }
        
        return EvaluationResult(
            false, 
            "No match found. Predicted: '$cleanPred', Gold: '$cleanGold'"
        )
    }
    
    data class EvaluationResult(
        val correct: Boolean,
        val reason: String
    )
}