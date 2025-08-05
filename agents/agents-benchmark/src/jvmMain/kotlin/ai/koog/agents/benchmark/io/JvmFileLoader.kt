package ai.koog.agents.benchmark.io

import java.io.File
import java.nio.file.Files
import java.nio.file.Paths

/**
 * JVM-specific file loading implementation
 */
object JvmFileLoader {
    
    /**
     * Load file content as string
     */
    fun loadFileContent(path: String): String {
        val file = File(path)
        
        if (!file.exists()) {
            throw IllegalArgumentException("File not found: $path")
        }
        
        if (!file.isFile) {
            throw IllegalArgumentException("Path is not a file: $path")
        }
        
        if (!file.canRead()) {
            throw IllegalArgumentException("Cannot read file: $path")
        }
        
        // Check file size to prevent loading huge files
        val maxSize = 100 * 1024 * 1024 // 100MB limit
        if (file.length() > maxSize) {
            throw IllegalArgumentException("File too large: ${file.length()} bytes (max: $maxSize)")
        }
        
        return try {
            file.readText(Charsets.UTF_8)
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to read file: ${e.message}", e)
        }
    }
    
    /**
     * Load file content as lines
     */
    fun loadFileLines(path: String): List<String> {
        val file = File(path)
        
        if (!file.exists()) {
            throw IllegalArgumentException("File not found: $path")
        }
        
        return try {
            file.readLines(Charsets.UTF_8)
        } catch (e: Exception) {
            throw IllegalArgumentException("Failed to read file: ${e.message}", e)
        }
    }
    
    /**
     * Check if file exists
     */
    fun fileExists(path: String): Boolean = File(path).exists()
    
    /**
     * Get file info
     */
    fun getFileInfo(path: String): FileInfo {
        val file = File(path)
        return FileInfo(
            path = file.absolutePath,
            exists = file.exists(),
            size = if (file.exists()) file.length() else 0,
            isDirectory = file.isDirectory,
            canRead = file.canRead(),
            extension = file.extension
        )
    }
    
    /**
     * List files in directory matching pattern
     */
    fun listFiles(directory: String, pattern: String = "*"): List<String> {
        val dir = File(directory)
        
        if (!dir.exists() || !dir.isDirectory) {
            return emptyList()
        }
        
        val regex = pattern.replace("*", ".*").toRegex()
        
        return dir.listFiles()
            ?.filter { it.isFile && it.name.matches(regex) }
            ?.map { it.absolutePath }
            ?: emptyList()
    }
    
    data class FileInfo(
        val path: String,
        val exists: Boolean,
        val size: Long,
        val isDirectory: Boolean,
        val canRead: Boolean,
        val extension: String
    )
}

/**
 * Extension function to make file loading available in common code
 */
actual fun loadDatasetFile(path: String): String = JvmFileLoader.loadFileContent(path)