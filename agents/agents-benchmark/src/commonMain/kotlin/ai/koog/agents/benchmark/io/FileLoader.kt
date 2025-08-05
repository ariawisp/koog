package ai.koog.agents.benchmark.io

/**
 * Platform-specific file loading function
 * Implemented by each platform (JVM, JS, etc.)
 */
expect fun loadDatasetFile(path: String): String