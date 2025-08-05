package ai.koog.agents.benchmark

import ai.koog.agents.benchmark.io.JvmFileLoader
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

/**
 * Utility to download and prepare benchmark datasets
 */
object DatasetDownloader {
    
    private const val DATASETS_DIR = "benchmark-datasets"
    
    data class DatasetInfo(
        val name: String,
        val url: String,
        val filename: String,
        val description: String
    )
    
    val availableDatasets = listOf(
        DatasetInfo(
            name = "LettaBench-Sample",
            url = "https://raw.githubusercontent.com/letta-ai/letta/main/data/letta_bench/letta_bench_sample.jsonl",
            filename = "lettabench_sample.jsonl",
            description = "Sample of LettaBench multi-hop reasoning questions"
        ),
        // Add more datasets as they become available
    )
    
    /**
     * Download a dataset if not already present
     */
    fun downloadDataset(datasetName: String): String {
        val dataset = availableDatasets.find { it.name == datasetName }
            ?: throw IllegalArgumentException("Unknown dataset: $datasetName")
        
        // Create datasets directory
        val datasetsDir = File(DATASETS_DIR)
        if (!datasetsDir.exists()) {
            datasetsDir.mkdirs()
        }
        
        val targetFile = File(datasetsDir, dataset.filename)
        
        if (targetFile.exists()) {
            println("✅ Dataset already downloaded: ${targetFile.absolutePath}")
            return targetFile.absolutePath
        }
        
        println("📥 Downloading ${dataset.name}...")
        println("   URL: ${dataset.url}")
        
        try {
            val url = URL(dataset.url)
            url.openStream().use { input ->
                Files.copy(input, targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            
            println("✅ Downloaded successfully: ${targetFile.absolutePath}")
            println("   Size: ${targetFile.length()} bytes")
            
            return targetFile.absolutePath
        } catch (e: Exception) {
            println("❌ Download failed: ${e.message}")
            throw e
        }
    }
    
    /**
     * List available local datasets
     */
    fun listLocalDatasets(): List<File> {
        val datasetsDir = File(DATASETS_DIR)
        if (!datasetsDir.exists()) {
            return emptyList()
        }
        
        return datasetsDir.listFiles { file ->
            file.isFile && (file.extension == "jsonl" || file.extension == "json")
        }?.toList() ?: emptyList()
    }
    
    /**
     * Prepare a dataset for benchmarking
     */
    fun prepareDataset(datasetPath: String): DatasetPreparation {
        val file = File(datasetPath)
        
        if (!file.exists()) {
            // Try to download if it's a known dataset name
            val downloaded = availableDatasets.find { it.name == datasetPath }?.let {
                downloadDataset(it.name)
            }
            
            if (downloaded != null) {
                return prepareDataset(downloaded)
            }
            
            throw IllegalArgumentException("Dataset not found: $datasetPath")
        }
        
        val content = JvmFileLoader.loadFileContent(file.absolutePath)
        val lines = content.lines().filter { it.isNotBlank() }
        
        return DatasetPreparation(
            path = file.absolutePath,
            format = when {
                file.name.contains("lettabench", ignoreCase = true) -> "LettaBench"
                file.extension == "jsonl" -> "JSONL"
                else -> "JSON"
            },
            size = file.length(),
            lineCount = lines.size,
            ready = true
        )
    }
    
    data class DatasetPreparation(
        val path: String,
        val format: String,
        val size: Long,
        val lineCount: Int,
        val ready: Boolean
    )
}

/**
 * Main function to download datasets
 */
fun main(args: Array<String>) {
    println("Koog Dataset Downloader")
    println("=".repeat(50))
    
    if (args.isEmpty()) {
        println("\nAvailable datasets:")
        DatasetDownloader.availableDatasets.forEach { dataset ->
            println("\n${dataset.name}:")
            println("  Description: ${dataset.description}")
            println("  Filename: ${dataset.filename}")
        }
        
        println("\nLocal datasets:")
        val localDatasets = DatasetDownloader.listLocalDatasets()
        if (localDatasets.isEmpty()) {
            println("  None")
        } else {
            localDatasets.forEach { file ->
                println("  ${file.name} (${file.length()} bytes)")
            }
        }
        
        println("\nUsage: download-datasets <dataset-name>")
        println("Example: download-datasets LettaBench-Sample")
        return
    }
    
    val datasetName = args[0]
    
    try {
        val path = DatasetDownloader.downloadDataset(datasetName)
        val prep = DatasetDownloader.prepareDataset(path)
        
        println("\n✅ Dataset ready for benchmarking:")
        println("  Path: ${prep.path}")
        println("  Format: ${prep.format}")
        println("  Lines: ${prep.lineCount}")
        
    } catch (e: Exception) {
        println("\n❌ Error: ${e.message}")
    }
}

