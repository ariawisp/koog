package ai.koog.agents.benchmark.literature

import ai.koog.agents.benchmark.model.Dataset
import kotlinx.datetime.Instant

/**
 * Framework for comparing against published literature results
 * Based on the Zep paper (arXiv:2501.13956v1) and other established benchmarks
 */

/**
 * Published benchmark results from academic literature
 */
data class PublishedResult(
    val systemName: String,
    val paperTitle: String,
    val authors: List<String>,
    val arxivId: String?,
    val benchmarkName: String,
    val accuracy: Double,
    val model: String,
    val avgLatencyMs: Long? = null,
    val avgTokens: Int? = null,
    val notes: String? = null,
    val publicationDate: String
)

/**
 * Official benchmark datasets with their properties
 */
data class OfficialBenchmark(
    val name: String,
    val description: String,
    val totalQuestions: Int,
    val avgContextLength: Int,
    val questionTypes: List<String>,
    val evaluationMetric: String,
    val paperReference: String,
    val datasetUrl: String? = null,
    val leaderboardUrl: String? = null
)

/**
 * Literature comparison results
 */
data class LiteratureComparisonReport(
    val benchmarkName: String,
    val koogResult: BenchmarkResult,
    val publishedResults: List<PublishedResult>,
    val ranking: Int, // Where Koog ranks among published results
    val totalSystems: Int,
    val significantImprovement: Boolean,
    val improvement: Double, // Percentage improvement over best published result
    val timestamp: Instant
)

data class BenchmarkResult(
    val systemName: String,
    val accuracy: Double,
    val avgLatencyMs: Long,
    val avgTokens: Int,
    val model: String,
    val confidenceInterval: Pair<Double, Double>? = null
)

/**
 * Repository of published results for legitimate comparison
 */
object LiteratureRepository {
    
    /**
     * Official benchmarks from the literature
     */
    val officialBenchmarks = listOf(
        OfficialBenchmark(
            name = "Deep Memory Retrieval (DMR)",
            description = "500 multi-session conversations with memory evaluation questions",
            totalQuestions = 500,
            avgContextLength = 3600, // ~60 messages * 60 tokens
            questionTypes = listOf("fact-retrieval", "single-turn"),
            evaluationMetric = "Accuracy",
            paperReference = "MemGPT: Towards LLMs as Operating Systems (arXiv:2310.08560)",
            leaderboardUrl = "https://github.com/cpacker/MemGPT"
        ),
        
        OfficialBenchmark(
            name = "LongMemEval",
            description = "Long-term interactive memory benchmark with 115k token conversations",
            totalQuestions = 500, // Approximate from paper
            avgContextLength = 115000,
            questionTypes = listOf(
                "single-session-user", "single-session-assistant", "single-session-preference",
                "multi-session", "knowledge-update", "temporal-reasoning"
            ),
            evaluationMetric = "Accuracy",
            paperReference = "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory (arXiv:2407.11122)",
            datasetUrl = "https://github.com/Buki2/LongMemEval"
        ),
        
        OfficialBenchmark(
            name = "LettaBench",
            description = "Multi-hop reasoning benchmark from Letta team",
            totalQuestions = 200,
            avgContextLength = 1000, // Estimated
            questionTypes = listOf("multi-hop-reasoning", "fact-extraction"),
            evaluationMetric = "Accuracy",
            paperReference = "Letta Memory Leaderboard",
            leaderboardUrl = "https://github.com/letta-ai/letta-leaderboard"
        )
    )
    
    /**
     * Published results from Zep paper (arXiv:2501.13956v1)
     */
    val zepPaperResults = listOf(
        // Deep Memory Retrieval results
        PublishedResult(
            systemName = "Recursive Summarization",
            paperTitle = "MemGPT: Towards LLMs as Operating Systems", 
            authors = listOf("Charles Packer", "Sarah Wooders", "Kevin Lin", "Vivian Fang", "Shishir G. Patil", "Ion Stoica", "Joseph E. Gonzalez"),
            arxivId = "arXiv:2310.08560",
            benchmarkName = "Deep Memory Retrieval (DMR)",
            accuracy = 0.353,
            model = "gpt-4-turbo",
            publicationDate = "2024"
        ),
        
        PublishedResult(
            systemName = "Conversation Summaries",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "Deep Memory Retrieval (DMR)",
            accuracy = 0.786,
            model = "gpt-4-turbo",
            publicationDate = "2025-01"
        ),
        
        PublishedResult(
            systemName = "MemGPT",
            paperTitle = "MemGPT: Towards LLMs as Operating Systems",
            authors = listOf("Charles Packer", "Sarah Wooders", "Kevin Lin", "Vivian Fang", "Shishir G. Patil", "Ion Stoica", "Joseph E. Gonzalez"),
            arxivId = "arXiv:2310.08560",
            benchmarkName = "Deep Memory Retrieval (DMR)",
            accuracy = 0.934,
            model = "gpt-4-turbo",
            publicationDate = "2024"
        ),
        
        PublishedResult(
            systemName = "Full-conversation",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "Deep Memory Retrieval (DMR)",
            accuracy = 0.944,
            model = "gpt-4-turbo",
            publicationDate = "2025-01"
        ),
        
        PublishedResult(
            systemName = "Zep",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "Deep Memory Retrieval (DMR)",
            accuracy = 0.948,
            model = "gpt-4-turbo",
            publicationDate = "2025-01"
        ),
        
        // LongMemEval results
        PublishedResult(
            systemName = "Full-context",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "LongMemEval",
            accuracy = 0.554,
            model = "gpt-4o-mini",
            avgLatencyMs = 31300,
            avgTokens = 115000,
            publicationDate = "2025-01"
        ),
        
        PublishedResult(
            systemName = "Zep",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "LongMemEval",
            accuracy = 0.638,
            model = "gpt-4o-mini",
            avgLatencyMs = 3200,
            avgTokens = 1600,
            publicationDate = "2025-01"
        ),
        
        PublishedResult(
            systemName = "Full-context",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "LongMemEval",
            accuracy = 0.602,
            model = "gpt-4o",
            avgLatencyMs = 28900,
            avgTokens = 115000,
            publicationDate = "2025-01"
        ),
        
        PublishedResult(
            systemName = "Zep",
            paperTitle = "ZEP: A Temporal Knowledge Graph Architecture for Agent Memory",
            authors = listOf("Preston Rasmussen", "Pavlo Paliychuk", "Travis Beauvais", "Jack Ryan", "Daniel Chalef"),
            arxivId = "arXiv:2501.13956v1",
            benchmarkName = "LongMemEval",
            accuracy = 0.712,
            model = "gpt-4o",
            avgLatencyMs = 2580,
            avgTokens = 1600,
            publicationDate = "2025-01"
        )
    )
    
    /**
     * Get published results for a specific benchmark
     */
    fun getPublishedResults(benchmarkName: String): List<PublishedResult> {
        return zepPaperResults.filter { it.benchmarkName == benchmarkName }
    }
    
    /**
     * Get the current state-of-the-art result for a benchmark
     */
    fun getSOTA(benchmarkName: String, model: String? = null): PublishedResult? {
        val results = getPublishedResults(benchmarkName)
        return if (model != null) {
            results.filter { it.model == model }.maxByOrNull { it.accuracy }
        } else {
            results.maxByOrNull { it.accuracy }
        }
    }
    
    /**
     * Get official benchmark specification
     */
    fun getBenchmark(name: String): OfficialBenchmark? {
        return officialBenchmarks.find { it.name == name }
    }
}

/**
 * Compares results against published literature
 */
class LiteratureComparator {
    
    /**
     * Compare against published results for a specific benchmark
     */
    fun compareAgainstLiterature(
        benchmarkName: String,
        koogResult: BenchmarkResult
    ): LiteratureComparisonReport {
        
        val publishedResults = LiteratureRepository.getPublishedResults(benchmarkName)
        val allResults = publishedResults + PublishedResult(
            systemName = koogResult.systemName,
            paperTitle = "Koog Agents Framework",
            authors = listOf("Koog Team"),
            arxivId = null,
            benchmarkName = benchmarkName,
            accuracy = koogResult.accuracy,
            model = koogResult.model,
            avgLatencyMs = koogResult.avgLatencyMs,
            avgTokens = koogResult.avgTokens,
            publicationDate = "2025"
        )
        
        // Sort by accuracy (descending)
        val sortedResults = allResults.sortedByDescending { it.accuracy }
        val koogRanking = sortedResults.indexOf(sortedResults.find { it.systemName == koogResult.systemName }) + 1
        
        // Calculate improvement over current SOTA
        val currentSOTA = publishedResults.maxByOrNull { it.accuracy }
        val improvement = if (currentSOTA != null) {
            ((koogResult.accuracy - currentSOTA.accuracy) / currentSOTA.accuracy) * 100
        } else {
            0.0
        }
        
        val significantImprovement = improvement > 5.0 // 5% improvement threshold
        
        return LiteratureComparisonReport(
            benchmarkName = benchmarkName,
            koogResult = koogResult,
            publishedResults = publishedResults,
            ranking = koogRanking,
            totalSystems = allResults.size,
            significantImprovement = significantImprovement,
            improvement = improvement,
            timestamp = kotlinx.datetime.Clock.System.now()
        )
    }
    
    /**
     * Print detailed comparison report
     */
    fun printComparisonReport(report: LiteratureComparisonReport) {
        println("\n" + "=".repeat(70))
        println("📊 LITERATURE COMPARISON REPORT")
        println("=".repeat(70))
        println("Benchmark: ${report.benchmarkName}")
        println("Timestamp: ${report.timestamp}")
        
        println("\n🏆 RANKING:")
        println("Koog Ranking: #${report.ranking} out of ${report.totalSystems} systems")
        
        if (report.ranking == 1) {
            println("🥇 NEW STATE-OF-THE-ART ACHIEVED!")
        } else if (report.ranking <= 3) {
            println("🥈 Top-3 performance achieved!")
        }
        
        println("\n📈 PERFORMANCE COMPARISON:")
        val sortedResults = (report.publishedResults + PublishedResult(
            systemName = report.koogResult.systemName,
            paperTitle = "Koog Agents Framework",
            authors = listOf("Koog Team"),
            arxivId = null,
            benchmarkName = report.benchmarkName,
            accuracy = report.koogResult.accuracy,
            model = report.koogResult.model,
            avgLatencyMs = report.koogResult.avgLatencyMs,
            avgTokens = report.koogResult.avgTokens,
            publicationDate = "2025"
        )).sortedByDescending { it.accuracy }
        
        println("Rank\tSystem\t\t\tAccuracy\tModel\t\tLatency\tTokens")
        println("-".repeat(80))
        
        sortedResults.forEachIndexed { index, result ->
            val rank = index + 1
            val marker = if (result.systemName == report.koogResult.systemName) "👑" else "  "
            val system = result.systemName.take(15).padEnd(15)
            val accuracy = "${(result.accuracy * 100).toInt()}%".padEnd(8)
            val model = result.model.take(12).padEnd(12)
            val latency = result.avgLatencyMs?.let { "${it}ms" } ?: "N/A"
            val tokens = result.avgTokens?.toString() ?: "N/A"
            
            println("$rank$marker\t$system\t$accuracy\t$model\t$latency\t$tokens")
        }
        
        if (report.significantImprovement) {
            println("\n✅ SIGNIFICANT IMPROVEMENT: +${String.format("%.1f", report.improvement)}% over current SOTA")
        } else if (report.improvement > 0) {
            println("\n🟡 Marginal improvement: +${String.format("%.1f", report.improvement)}% over current SOTA")
        } else {
            println("\n🔴 Below current SOTA: ${String.format("%.1f", report.improvement)}% vs best published result")
        }
        
        println("\n📚 REFERENCES:")
        report.publishedResults.distinctBy { it.paperTitle }.forEach { result ->
            println("- ${result.paperTitle}")
            println("  Authors: ${result.authors.joinToString(", ")}")
            if (result.arxivId != null) {
                println("  ${result.arxivId}")
            }
            println("  Published: ${result.publicationDate}")
        }
        
        println("\n💡 INTERPRETATION:")
        when {
            report.ranking == 1 && report.significantImprovement -> {
                println("🎉 BREAKTHROUGH: Koog has achieved new state-of-the-art performance with significant improvement!")
                println("   This result is suitable for academic publication and competitive claims.")
            }
            report.ranking <= 3 -> {
                println("🏅 COMPETITIVE: Koog achieves top-tier performance comparable to leading systems.")
                println("   This demonstrates the framework's effectiveness for real-world applications.")
            }
            report.ranking <= report.totalSystems / 2 -> {
                println("📊 SOLID: Koog performs in the upper half of evaluated systems.")
                println("   Good foundation with room for optimization.")
            }
            else -> {
                println("🔧 DEVELOPMENT: Performance is below median. Consider:")
                println("   - Optimizing retrieval strategies")
                println("   - Improving prompt engineering")
                println("   - Tuning hyperparameters")
            }
        }
    }
}