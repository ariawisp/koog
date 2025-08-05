package ai.koog.agents.benchmark

/**
 * Central location for benchmark constants to avoid magic numbers
 */
object BenchmarkConstants {
    
    // Accuracy thresholds
    const val SOTA_ACCURACY_THRESHOLD = 0.95
    const val GOOD_ACCURACY_THRESHOLD = 0.80
    const val ACCEPTABLE_ACCURACY_THRESHOLD = 0.70
    
    // Token estimation
    const val CHARS_PER_TOKEN_ESTIMATE = 4
    
    // Performance thresholds
    const val MAX_ACCEPTABLE_LATENCY_MS = 5000L
    const val TOKEN_EFFICIENCY_THRESHOLD = 1500
    
    // Statistical parameters
    const val DEFAULT_CONFIDENCE_LEVEL = 0.95
    const val CONFIDENCE_INTERVAL_Z_SCORE = 1.96 // For 95% confidence
    const val MIN_SAMPLE_SIZE_FOR_SIGNIFICANCE = 30
    
    // Execution parameters
    const val DEFAULT_NUM_RUNS = 3
    const val DEFAULT_WARMUP_QUESTIONS = 5
    const val DEFAULT_RETRY_ATTEMPTS = 3
    const val DEFAULT_RETRY_DELAY_MS = 1000L
    const val DEFAULT_TIMEOUT_MS = 30_000L
    
    // Parallel execution
    const val DEFAULT_MAX_CONCURRENCY = 8
    const val SEQUENTIAL_CONCURRENCY = 1
    
    // Formatting
    const val SEPARATOR_LENGTH = 60
    const val SUBSEPARATOR_LENGTH = 40
    
    // Cost estimation (per 1000 tokens)
    const val COST_PER_1K_TOKENS_USD = 0.002
    
    // Answer evaluation thresholds
    const val WORD_MATCH_THRESHOLD = 0.7
    const val MIN_WORD_LENGTH_FOR_MATCHING = 2
}