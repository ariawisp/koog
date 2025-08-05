package ai.koog.agents.benchmark.io

import java.io.File

/**
 * JVM implementation of FileSystem
 */
actual class FileSystem {
    actual fun readText(path: String): String = File(path).readText()
    
    actual fun exists(path: String): Boolean = File(path).exists()
    
    actual fun readLines(path: String): List<String> = File(path).readLines()
    
    actual fun writeText(path: String, content: String) {
        File(path).writeText(content)
    }
    
    actual fun nameWithoutExtension(path: String): String = File(path).nameWithoutExtension
    
    actual fun extension(path: String): String = File(path).extension
}

/**
 * Global file system instance
 */
actual val fileSystem: FileSystem = FileSystem()