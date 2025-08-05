# Koog Memory Retrieval API Guide

This guide provides comprehensive documentation for the enhanced memory retrieval system in Koog, including token-aware retrieval, cross-encoder reranking, and temporal reasoning capabilities.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Token-Aware Retrieval](#token-aware-retrieval)
3. [Cross-Encoder Reranking](#cross-encoder-reranking)
4. [Temporal Knowledge Extraction](#temporal-knowledge-extraction)
5. [Smart Routing](#smart-routing)
6. [Metrics Collection](#metrics-collection)
7. [Configuration](#configuration)
8. [Complete Examples](#complete-examples)

## Quick Start

```kotlin
import ai.koog.agents.features.tokenizer.feature.PromptTokenizer
import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.config.*

// Create a token-aware retriever with default settings
val retriever = baseRetriever.withTokenAwareness(
    tokenizer = promptTokenizer,
    config = TokenAwareRetrievalConfig.tokenEfficiencyOptimized()
)

// Make a token-budget-aware query
val results = retriever.retrieve(
    RetrievalQuery("What is Alice's favorite color?")
        .withTokenBudget(1500, OptimizationStrategy.GREEDY)
)
```

## Token-Aware Retrieval

The `TokenAwareRetriever` optimizes retrieval results to fit within token budgets while maximizing relevance.

### Basic Usage

```kotlin
// Create with custom configuration
val config = TokenAwareRetrievalConfig(
    defaultTokenBudget = 2000,
    minResultTokenSize = 100,
    enableAdaptiveBudgeting = true,
    reserveTokensForQuery = 100,
    tokenOverflowTolerance = 50
)

val tokenAwareRetriever = TokenAwareRetriever(
    baseRetriever = knowledgeGraphRetriever,
    tokenizer = promptTokenizer,
    config = config
)
```

### Optimization Strategies

```kotlin
// Greedy: Maximize relevance per token
val greedyQuery = query.withTokenBudget(1000, OptimizationStrategy.GREEDY)

// Balanced: Ensure diversity across result types
val balancedQuery = query.withTokenBudget(1000, OptimizationStrategy.BALANCED)

// Precision: Focus only on highest relevance items
val precisionQuery = query.withTokenBudget(1000, OptimizationStrategy.PRECISION)
```

### Adaptive Budgeting

The retriever can adaptively include small, high-value results that slightly exceed the budget:

```kotlin
val config = TokenAwareRetrievalConfig(
    enableAdaptiveBudgeting = true,
    minResultTokenSize = 100,      // Consider results under 100 tokens as "small"
    tokenOverflowTolerance = 50    // Allow up to 50 tokens overflow for small results
)
```

## Cross-Encoder Reranking

The `CrossEncoderReranker` uses an LLM to score query-document pairs for higher precision.

### Basic Usage

```kotlin
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.koog.prompt.llm.LLModel

val reranker = CrossEncoderReranker(
    llmExecutor = llmExecutor,
    model = LLModel.GPT_4,
    maxConcurrency = 4,
    batchSize = 10
)

// Rerank results
val rerankedResults = results.rerankWithCrossEncoder(
    query = "What did Alice do yesterday?",
    reranker = reranker,
    config = CrossEncoderReranker.RerankConfig(
        scoreThreshold = 0.7,
        maxResultsToRerank = 30,
        useExplanations = false,
        retryOnError = true,
        maxRetries = 2
    )
)
```

### With Explanations

Get explanations for relevance scores:

```kotlin
val explainedConfig = CrossEncoderReranker.RerankConfig(
    useExplanations = true  // LLM will provide reasoning
)

val rerankedWithExplanations = reranker.rerank(
    query = queryText,
    results = results,
    config = explainedConfig
)

// Access explanations in metadata
rerankedWithExplanations.forEach { result ->
    val explanation = result.metadata["cross_encoder_explanation"]
    println("Score: ${result.score}, Reason: $explanation")
}
```

## Temporal Knowledge Extraction

The `TemporalKnowledgeExtractor` handles time-based information extraction and reasoning.

### Extract Temporal Facts

```kotlin
val extractor = TemporalKnowledgeExtractor()

// Extract temporal facts from a message
val temporalFacts = extractor.extractTemporalFacts(
    message = "Alice moved to Paris last week. She was in London for 3 years.",
    referenceTime = Clock.System.now(),
    context = previousMessages
)

temporalFacts.forEach { fact ->
    println("Fact: ${fact.fact}")
    println("Valid from: ${fact.validFrom}")
    println("Valid to: ${fact.validTo ?: "current"}")
    println("Is relative: ${fact.isRelative}")
    println("Confidence: ${fact.confidence}")
}
```

### Handle Contradictions

```kotlin
// Detect and resolve contradicting temporal facts
val newFact = TemporalFact(
    fact = SingleFact(concept, "Paris", timestamp),
    validFrom = Clock.System.now()
)

val existingFacts = loadExistingFacts()

val resolution = extractor.handleContradictions(newFact, existingFacts)

// Apply resolution
resolution.invalidatedFacts.forEach { oldFact ->
    // Mark old facts as ended
    updateFact(oldFact.copy(validTo = newFact.validFrom))
}

// Save new fact
saveFact(resolution.newValidFact)
```

## Smart Routing

The `SmartRouter` automatically selects the best retrieval provider based on query characteristics.

### Basic Setup

```kotlin
val router = createSmartRouter(
    memoryProvider = graphMemoryProvider,
    documentStorage = documentStorage,
    graphProvider = knowledgeGraphRetriever,
    tokenizer = promptTokenizer,
    policy = RoutingPolicy(
        enableGraph = true,
        temporalRequiresGraph = true,
        maxLatencyMs = 300,
        detectMultiHop = { text ->
            // Custom multi-hop detection logic
            text.contains("who") && text.contains("knows")
        }
    )
)
```

### Query Routing Examples

```kotlin
// Temporal query - routes to graph provider
val temporalResults = router.retrieve(
    RetrievalQuery(
        text = "What was the status last week?",
        at = Clock.System.now().minus(7.days)
    )
)

// Entity-centric query - routes to graph provider
val entityResults = router.retrieve(
    RetrievalQuery(
        text = "Information about Alice",
        centerNode = "entity_alice_uuid"
    )
)

// Simple semantic query - routes to vector provider
val semanticResults = router.retrieve(
    RetrievalQuery(
        text = "blue ocean strategy",
        recipe = RetrievalRecipe.VECTOR_SIMILARITY
    )
)
```

## Metrics Collection

Track performance metrics for optimization and monitoring.

### Setup Metrics Collection

```kotlin
import ai.koog.agents.memory.retrieval.metrics.*

val metricsCollector = InMemoryMetricsCollector()

// Wrap any retriever with metrics
val metricsAwareRetriever = baseRetriever.withMetrics(
    metricsCollector = metricsCollector,
    providerName = "MyCustomRetriever"
)
```

### Access Metrics

```kotlin
// Get provider-specific stats
val stats = metricsCollector.getProviderStats("MyCustomRetriever")
println("Average latency: ${stats.avgLatencyMs}ms")
println("Token efficiency: ${stats.avgTokenEfficiency}")
println("Cache hit rate: ${stats.cacheHitRate}")
println("Total queries: ${stats.totalQueries}")

// Get all provider stats
val allStats = metricsCollector.getAllProviderStats()
allStats.forEach { providerStats ->
    println("${providerStats.provider}: ${providerStats.avgLatencyMs}ms avg latency")
}
```

## Configuration

### System-Wide Configuration

```kotlin
// Load from JSON or create programmatically
val systemConfig = MemoryRetrievalSystemConfig(
    tokenAwareRetrieval = TokenAwareRetrievalConfig(
        defaultTokenBudget = 2000,
        enableAdaptiveBudgeting = true
    ),
    crossEncoder = CrossEncoderConfig(
        enabled = true,
        maxConcurrency = 8,
        defaultScoreThreshold = 0.7
    ),
    metrics = RetrievalMetricsConfig(
        enabled = true,
        trackTokenUsage = true,
        enableDetailedLogging = false
    ),
    smartRouter = SmartRouterConfig(
        enableGraphRouting = true,
        maxLatencyMs = 200
    )
)

// Or use presets
val accuracyConfig = MemoryRetrievalSystemConfig.accuracyOptimized()
val speedConfig = MemoryRetrievalSystemConfig.speedOptimized()
val efficiencyConfig = MemoryRetrievalSystemConfig.tokenEfficiencyOptimized()
```

## Complete Examples

### Example 1: Building a Complete Retrieval Pipeline

```kotlin
import ai.koog.agents.memory.retrieval.*
import ai.koog.agents.memory.retrieval.config.*
import ai.koog.agents.memory.retrieval.metrics.*
import ai.koog.agents.memory.retrieval.providers.*
import ai.koog.agents.memory.retrieval.reranking.*

// 1. Create base components
val knowledgeGraph = createKnowledgeGraph()
val tokenizer = createPromptTokenizer()
val llmExecutor = createLLMExecutor()
val metricsCollector = InMemoryMetricsCollector()

// 2. Create knowledge graph retriever
val graphRetriever = KnowledgeGraphRetrievalProvider(
    knowledgeGraph = knowledgeGraph,
    tokenizer = tokenizer,
    defaultTokenBudget = 3000
)

// 3. Add cross-encoder reranking
val reranker = CrossEncoderReranker(
    llmExecutor = llmExecutor,
    model = LLModel.GPT_4,
    maxConcurrency = 4
)

val rerankedRetriever = object : RetrievalProvider {
    override fun supports(recipe: RetrievalRecipe) = graphRetriever.supports(recipe)
    
    override suspend fun retrieve(query: RetrievalQuery): List<RetrievalResult> {
        val initial = graphRetriever.retrieve(query)
        return if (query.k <= 5) { // Only rerank for small result sets
            reranker.rerank(query.text, initial)
        } else {
            initial
        }
    }
}

// 4. Add token awareness
val tokenAwareRetriever = rerankedRetriever.withTokenAwareness(
    tokenizer = tokenizer,
    config = TokenAwareRetrievalConfig(
        defaultTokenBudget = 2000,
        enableAdaptiveBudgeting = true
    )
)

// 5. Add metrics
val finalRetriever = tokenAwareRetriever.withMetrics(
    metricsCollector = metricsCollector,
    providerName = "EnhancedGraphRetriever"
)

// 6. Use the pipeline
val results = finalRetriever.retrieve(
    RetrievalQuery(
        text = "What are Alice's recent activities?",
        k = 10,
        recipe = RetrievalRecipe.HYBRID_MMR
    ).withTokenBudget(1500, OptimizationStrategy.BALANCED)
)

// 7. Check metrics
val stats = metricsCollector.getProviderStats("EnhancedGraphRetriever")
println("Retrieved ${results.size} results in ${stats.avgLatencyMs}ms")
```

### Example 2: Temporal Query Processing

```kotlin
// Setup temporal extraction
val temporalExtractor = TemporalKnowledgeExtractor()
val now = Clock.System.now()

// Process user message with temporal content
val userMessage = "I met Bob yesterday at the conference. He said he left Google last month."

// Extract temporal facts
val temporalFacts = temporalExtractor.extractTemporalFacts(
    message = userMessage,
    referenceTime = now
)

// Create temporal queries for each fact
val queries = temporalFacts.map { fact ->
    RetrievalQuery(
        text = fact.fact.value,
        at = fact.validFrom,
        target = RetrievalTarget.FACTS,
        recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE
    )
}

// Retrieve with temporal context
val router = createSmartRouter(
    graphProvider = knowledgeGraphRetriever,
    policy = RoutingPolicy(temporalRequiresGraph = true)
)

val temporalResults = queries.flatMap { query ->
    router.retrieve(query)
}.distinctBy { it.content }

println("Found ${temporalResults.size} temporal facts")
```

### Example 3: Benchmark-Optimized Configuration

```kotlin
// Configuration optimized for beating benchmarks
val benchmarkConfig = MemoryRetrievalSystemConfig(
    tokenAwareRetrieval = TokenAwareRetrievalConfig(
        defaultTokenBudget = 1500,  // Tight budget for efficiency
        minResultTokenSize = 50,    // Small results allowed
        enableAdaptiveBudgeting = true,
        reserveTokensForQuery = 100,
        tokenOverflowTolerance = 100  // More tolerance for benchmarks
    ),
    crossEncoder = CrossEncoderConfig(
        enabled = true,
        maxConcurrency = 16,        // High parallelism
        batchSize = 20,             // Large batches
        defaultScoreThreshold = 0.8, // High precision
        maxResultsToRerank = 50,    // Rerank many results
        enableRetry = true
    ),
    metrics = RetrievalMetricsConfig(
        enabled = true,
        trackLatency = true,
        trackTokenUsage = true,
        trackCacheHits = true
    ),
    parallelRetrievalEnabled = true,
    maxParallelRequests = 32
)

// Build the retrieval system
val benchmarkRetriever = buildRetrievalSystem(benchmarkConfig)

// Run benchmark queries
val benchmarkResults = benchmarkQueries.map { query ->
    val start = Clock.System.now()
    val results = benchmarkRetriever.retrieve(query)
    val latency = Clock.System.now() - start
    
    BenchmarkResult(
        query = query,
        results = results,
        latency = latency,
        tokensUsed = results.sumOf { tokenizer.countTokens(it.content) }
    )
}

// Analyze performance
val avgLatency = benchmarkResults.map { it.latency }.average()
val avgTokens = benchmarkResults.map { it.tokensUsed }.average()
println("Average latency: $avgLatency")
println("Average tokens: $avgTokens")
```

## Best Practices

1. **Token Budget Management**
   - Start with conservative budgets and increase as needed
   - Use adaptive budgeting for flexibility
   - Monitor token usage through metrics

2. **Cross-Encoder Usage**
   - Only rerank top results to manage latency
   - Use batch processing for efficiency
   - Enable retry for production reliability

3. **Temporal Queries**
   - Always provide reference time for relative dates
   - Handle contradictions explicitly
   - Use appropriate confidence thresholds

4. **Performance Optimization**
   - Enable metrics to identify bottlenecks
   - Use smart routing to avoid unnecessary work
   - Configure parallelism based on resources

5. **Error Handling**
   - All components log errors appropriately
   - Critical errors are propagated
   - Fallback strategies are built-in

## Troubleshooting

### High Latency
- Reduce cross-encoder batch size
- Lower maxResultsToRerank
- Enable query caching

### Poor Relevance
- Increase token budget
- Enable cross-encoder reranking
- Use PRECISION optimization strategy

### Token Overflow
- Increase tokenOverflowTolerance
- Reduce k (number of results)
- Use more aggressive filtering

### Memory Issues
- Reduce batch sizes
- Limit concurrent operations
- Clear metrics periodically

## Migration Guide

For existing code using the old retrieval system:

```kotlin
// Old approach
val results = memoryProvider.loadByDescription(description, subject, scope)

// New approach
val retriever = createSmartRouter(
    memoryProvider = memoryProvider,
    tokenizer = tokenizer
).withTokenAwareness(tokenizer)

val results = retriever.retrieve(
    RetrievalQuery(
        text = description,
        filters = RetrievalFilters(
            subjects = setOf(subject),
            scopes = setOf(scope)
        )
    ).withTokenBudget(2000)
)
```

This new approach provides token awareness, better routing, and performance metrics while maintaining compatibility with existing memory providers.