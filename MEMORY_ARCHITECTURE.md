# Memory System Architecture

## Overview

The Koog memory system provides a flexible, multi-layered architecture for agent memory management, supporting various storage backends and retrieval strategies.

## Architecture Layers

### 1. Storage Layer
Low-level persistence mechanisms.

- **InMemoryKnowledgeGraph**: Default in-memory graph implementation
- **FileSystemProvider**: File-based storage (via LocalFileMemoryProvider)
- **Future**: Neo4j, PostgreSQL, other backends

### 2. Core Abstractions
Fundamental interfaces and models.

- **KnowledgeGraph**: Core graph interface with entities, relations, episodes
- **Episode**: Information units with temporal context
- **Knowledge**: Query results (Entity, Relation, Composite)
- **AgentMemoryProvider**: Base interface for memory providers

### 3. Memory Providers
High-level memory management implementations.

- **GraphMemoryProvider**: Graph-based memory with KnowledgeGraph backend
  - Exposes `knowledgeGraph` property for direct access
  - Handles episode ingestion, fact extraction
  - Provides temporal queries
  
- **LocalFileMemoryProvider**: File-based memory storage
  - Simple key-value storage
  - JSON serialization
  
- **NoMemory**: Null implementation for stateless agents

### 4. Retrieval Layer
Search and retrieval strategies.

- **RetrievalProvider**: Base interface for retrieval
- **KnowledgeGraphRetrievalProvider**: Graph-based retrieval
  - Semantic search
  - Graph traversal
  - Temporal queries
  - Cross-encoder reranking
  
- **VectorRetrievalProvider**: Embedding-based retrieval
  - Works with AgentMemoryProvider or RankedDocumentStorage
  - Simple similarity search

### 5. Advanced Features

#### Smart Routing
- **SmartRouter**: Intelligently routes queries to best provider
- **RoutingPolicy**: Configurable routing strategies

#### Token Optimization
- **TokenAwareRetriever**: Optimizes results for token budgets
- **TokenAwareRetrievalConfig**: Budget configuration

#### Reranking
- **CrossEncoderReranker**: LLM-based relevance scoring
- **OpenAIStyleCrossEncoderReranker**: Simple boolean classifier

#### Security
- **SecurityContext**: Multi-tenancy support
- **MemoryScope**: Access control (User, Product, CrossProduct)

## Usage Patterns

### Basic Graph Memory
```kotlin
val graph = InMemoryKnowledgeGraph()
val memoryProvider = GraphMemoryProvider(graph)
val agent = AIAgent(...) {
    install(AgentMemory) {
        provider = memoryProvider
    }
}
```

### Advanced Retrieval
```kotlin
val graph = InMemoryKnowledgeGraph()
val graphProvider = GraphMemoryProvider(graph)
val retrievalProvider = KnowledgeGraphRetrievalProvider(
    knowledgeGraph = graphProvider.knowledgeGraph,
    crossEncoderReranker = CrossEncoderReranker(...)
)

val router = SmartRouter(
    providers = listOf(retrievalProvider),
    policy = RoutingPolicy(enableGraph = true)
)
```

### Direct Graph Access
```kotlin
val graphProvider = GraphMemoryProvider(graph)
// Direct access to knowledge graph for advanced operations
val kg = graphProvider.knowledgeGraph
val communities = kg.detectCommunities()
```

## Key Design Decisions

1. **Provider Separation**: Memory providers (storage) are separate from retrieval providers (search)
2. **Direct Graph Access**: GraphMemoryProvider exposes knowledgeGraph as public property
3. **Interface Simplicity**: Removed unnecessary interfaces like GraphMemoryProviderWithGraph
4. **Flexible Routing**: SmartRouter can work with any combination of providers
5. **Token Awareness**: Built-in support for token budget optimization

## Migration Notes

- `GraphMemoryProviderWithGraph` interface has been removed
- Access knowledge graph directly via `GraphMemoryProvider.knowledgeGraph`
- Use `SmartRouter` for automatic provider selection
- Prefer `KnowledgeGraphRetrievalProvider` over direct memory provider queries