# Memory Module Refactoring Plan

## Current Issues

The memory module has accumulated technical debt with confusing provider structures:

### 1. Duplicate Provider Concepts
- **Memory Providers** (`providers/`): Handle memory storage and persistence
  - `AgentMemoryProvider` - base interface
  - `GraphMemoryProvider` - graph-based storage
  - `LocalFileMemoryProvider` - file-based storage
  - `NoMemory` - null implementation

- **Retrieval Providers** (`retrieval/providers/`): Handle search and retrieval
  - `RetrievalProvider` - base interface
  - `KnowledgeGraphRetrievalProvider` - graph search
  - `VectorRetrievalProvider` - vector search

- **Hybrid/Confusing**:
  - `GraphMemoryProviderWithGraph` - unnecessary interface

### 2. Unclear Separation of Concerns
- Memory providers sometimes implement retrieval
- Retrieval providers need access to storage
- No clear boundary between storage and retrieval

## Proposed Clean Architecture

### Layer 1: Storage Layer (`storage/`)
Responsible for persisting and managing data.

```
storage/
├── StorageBackend.kt          # Base interface
├── backends/
│   ├── InMemoryBackend.kt     # In-memory storage
│   ├── FileBackend.kt         # File-based storage
│   └── Neo4jBackend.kt        # Neo4j storage (future)
└── encryption/
    └── StorageEncryptor.kt    # Encryption support
```

### Layer 2: Graph Layer (`graph/`)
Knowledge graph abstractions and implementations.

```
graph/
├── KnowledgeGraph.kt          # Core interface (already exists)
├── Episode.kt                 # Episode model (already exists)
├── Knowledge.kt               # Knowledge types (already exists)
└── providers/
    └── InMemoryKnowledgeGraph.kt  # Default implementation
```

### Layer 3: Memory Features (`features/`)
High-level memory features that combine storage and graph.

```
features/
├── AgentMemory.kt             # Agent memory feature
├── GraphMemory.kt             # Graph memory feature
└── processors/
    └── EpisodeProcessor.kt    # Episode processing
```

### Layer 4: Retrieval Layer (`retrieval/`)
Search and retrieval functionality.

```
retrieval/
├── RetrievalProvider.kt       # Base interface
├── RetrievalTypes.kt          # Query/Result types
├── strategies/
│   ├── GraphRetrieval.kt      # Graph-based retrieval
│   ├── VectorRetrieval.kt     # Vector-based retrieval
│   └── HybridRetrieval.kt     # Combined strategies
├── reranking/
│   └── CrossEncoderReranker.kt
└── SmartRouter.kt             # Routing between strategies
```

## Migration Steps

### Phase 1: Consolidate Storage
1. Move all persistence logic to `storage/` layer
2. Remove storage concerns from memory providers
3. Create clear `StorageBackend` interface

### Phase 2: Simplify Providers
1. Remove `GraphMemoryProviderWithGraph` interface
2. Merge `GraphMemoryProvider` functionality into `GraphMemory` feature
3. Keep retrieval providers focused only on search

### Phase 3: Clear Dependencies
```
Storage <- Graph <- Features <- Retrieval
```
Each layer only depends on layers below it.

## Benefits

1. **Clear Separation**: Storage, graph operations, features, and retrieval are clearly separated
2. **No Redundancy**: Each component has a single, clear responsibility
3. **Easy Extension**: Adding new storage backends or retrieval strategies is straightforward
4. **Better Testing**: Each layer can be tested independently

## Backwards Compatibility

During refactoring, maintain existing public APIs with deprecation warnings:

```kotlin
@Deprecated("Use GraphMemory feature directly")
class GraphMemoryProvider : AgentMemoryProvider {
    // Delegate to new implementation
}
```

## Timeline

1. **Immediate**: Document the plan and get agreement
2. **Phase 1**: Consolidate storage (1-2 days)
3. **Phase 2**: Simplify providers (1-2 days)
4. **Phase 3**: Clear dependencies (1 day)
5. **Cleanup**: Remove deprecated code after migration period