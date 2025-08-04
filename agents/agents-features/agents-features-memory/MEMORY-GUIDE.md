# Koog Memory System Guide

## Overview

Koog provides a unified memory system through the `AgentMemory` feature with multiple backend providers. This guide helps you choose and configure the right memory approach for your use case.

## Quick Start

### Basic Memory (File-based)
For simple fact storage and retrieval:

```kotlin
install(AgentMemory) {
    memoryProvider = LocalFileMemoryProvider(
        config = LocalMemoryConfig("agent-memory"),
        storage = SimpleStorage(JVMFileSystemProvider),
        root = Path("memory")
    )
}
```

### Advanced Memory (Graph-based)
For sophisticated reasoning, temporal queries, and relationship traversal:

```kotlin
install(AgentMemory) {
    useGraphInMemoryPreset()
}
```

## When to Use Each Approach

### Use File-based Memory When:
- ✅ Simple fact storage and retrieval
- ✅ Small to medium memory requirements (< 10K facts)
- ✅ Straightforward use cases without complex relationships
- ✅ You want minimal dependencies and setup

### Use Graph-based Memory When:
- ✅ Complex entity relationships and traversal
- ✅ Temporal reasoning ("what did we know at time T?")
- ✅ Large-scale memory with sophisticated search
- ✅ Knowledge evolution (consolidation, decay)
- ✅ Entity-centric queries ("what's related to X?")

## Graph Memory Capabilities

When you use `useGraphInMemoryPreset()`, you get:

### 1. Temporal Reasoning
```kotlin
val graphProvider = agentContext.memory().provider as GraphMemoryProvider
val historicalKnowledge = graphProvider.queryTemporal(
    start = yesterday,
    end = now
)
```

### 2. Entity-Centric Search
```kotlin
val relatedKnowledge = graphProvider.queryEntityCentric(
    centerNode = "player-steve",
    depth = 2
)
```

### 3. Automatic Conversation Ingestion
All LLM conversations are automatically converted to knowledge and stored in the graph.

### 4. Knowledge Evolution
The system automatically:
- Consolidates duplicate information
- Ages out stale facts
- Maintains knowledge quality over time

## Advanced Configuration

### Custom Graph Setup
```kotlin
install(AgentMemory) {
    val customGraph = InMemoryKnowledgeGraph(
        config = InMemoryConfig(
            minConfidence = 0.8,
            maxNodes = 50_000
        )
    )
    
    memoryProvider = GraphMemoryProvider(
        graph = customGraph,
        config = GraphMemoryConfig(
            autoIngestConversations = true,
            minConfidenceThreshold = 0.7
        )
    )
}
```

### Hybrid Approach
```kotlin
install(AgentMemory) {
    useGraphInMemoryPreset()
    
    // Override specific capabilities
    retriever = SmartRouter(
        providers = listOf(
            VectorRetrievalProvider(),
            CustomRetrievalProvider()
        )
    )
}
```

## Migration Path

### From File-based to Graph-based
1. Replace your `LocalFileMemoryProvider` configuration with `useGraphInMemoryPreset()`
2. Your existing `save()` and `load()` calls continue to work unchanged
3. Optionally use advanced graph features through the provider

### Example Migration
```kotlin
// Before
install(AgentMemory) {
    memoryProvider = LocalFileMemoryProvider(...)
}

// After  
install(AgentMemory) {
    useGraphInMemoryPreset()
}
```

## Best Practices

### For Graph Memory
1. **Use descriptive fact content** - The graph uses semantic search, so clear descriptions improve retrieval
2. **Leverage auto-ingestion** - Let conversations flow into the graph automatically
3. **Monitor evolution** - Use `getGraphStats()` to understand knowledge growth
4. **Configure confidence thresholds** - Tune `minConfidenceThreshold` based on your data quality needs

### For Any Memory Type
1. **Choose appropriate scopes** - Use `MemoryScope.Agent`, `MemoryScope.Product`, etc. appropriately
2. **Structure concepts well** - Clear concept names and descriptions improve organization
3. **Test memory operations** - Use the testing utilities to verify memory behavior

## Performance Considerations

### Graph Memory Performance
- **In-memory**: Excellent for < 100K nodes, good for < 1M nodes
- **Semantic search**: O(K) where K is matching terms (vs O(N) full scan)
- **BFS traversal**: O(1) queue operations with ArrayDeque
- **Snapshot-based**: Lock-free reads for high concurrency

### File Memory Performance  
- **Best for**: Simple, sequential access patterns
- **Storage**: Efficient for smaller datasets
- **Search**: Linear scan through facts

## Troubleshooting

### Graph Memory Issues
- **High memory usage**: Reduce `maxNodes` in `InMemoryConfig`
- **Slow semantic search**: Check `minConfidenceThreshold` - higher values filter more
- **Knowledge not evolving**: Verify evolution rules are configured

### General Issues
- **Facts not found**: Check scope and subject matching
- **Slow retrieval**: Consider switching to graph-based memory for large datasets
- **Memory not persisting**: Verify provider configuration and file permissions

## Next Steps

1. Start with the appropriate preset for your use case
2. Use the unified `AgentMemory` API for all operations  
3. Enable advanced features as needed through provider casting
4. Monitor performance with `getGraphStats()` for graph-based setups