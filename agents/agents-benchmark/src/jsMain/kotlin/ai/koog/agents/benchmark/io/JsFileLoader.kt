package ai.koog.agents.benchmark.io

/**
 * JS implementation of file loading
 * Note: File system access in JS requires Node.js APIs or browser file input
 */
actual fun loadDatasetFile(path: String): String {
    throw UnsupportedOperationException("File loading not supported on JS platform. Use JVM for benchmarking.")
}