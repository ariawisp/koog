package ai.koog.agents.benchmark.io

/**
 * Platform-agnostic file I/O interface
 */
expect class FileSystem {
    fun readText(path: String): String
    fun exists(path: String): Boolean
    fun readLines(path: String): List<String>
    fun writeText(path: String, content: String)
    fun nameWithoutExtension(path: String): String
    fun extension(path: String): String
}

/**
 * Global file system instance
 */
expect val fileSystem: FileSystem