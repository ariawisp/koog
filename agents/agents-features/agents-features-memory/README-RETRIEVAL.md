# Graph-Based Knowledge Retrieval for Koog

This document describes the new unified retrieval system that enables advanced knowledge search including graph-based, temporal, and hybrid retrieval strategies.

## Overview

The retrieval system extends Koog's existing `AgentMemory` feature with:
- **Unified API**: Single `nodeRetrieveKnowledge` that automatically routes to the best backend
- **Multiple Backends**: Support for vector search, graph traversal, and future mechanisms
- **Temporal Queries**: Point-in-time retrieval with `at` parameter
- **Relational Search**: Center-node reranking for entity-specific queries
- **Smart Routing**: Automatic selection of retrieval strategy based on query characteristics

## Quick Start

### Basic Usage

```kotlin
val agent = AIAgent(/* ... */) {
    install(AgentMemory) {
        memoryProvider = myMemoryProvider
        
        // Add retrieval capability
        retriever = createSmartRouter(
            memoryProvider = myMemoryProvider,
            documentStorage = myDocumentStorage
        )
    }
}

// In your strategy
val strategy = strategy<String, String>("my-strategy") {
    val search by nodeRetrieveKnowledge<String> {
        text = "Who owns the base near spawn?"
        k = 5
    }
    
    edge(nodeStart forwardTo search)
    edge(search forwardTo nodeFinish transformed { results ->
        // Format results for output
        results.joinToString("\n") { it.content }
    })
}
```

### Advanced Queries

```kotlin
// Temporal query
val historicalSearch by nodeRetrieveKnowledge {
    text = "Who owned this territory?"
    at = Instant.parse("2024-01-01T00:00:00Z")
    k = 10
}

// Entity-centric search with graph distance reranking
val nearbySearch by nodeRetrieveKnowledge {
    text = "Find nearby allies"
    centerNode = "player:uuid-12345"  // Search around this player
    recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE
}

// Filtered search
val factionSearch by nodeRetrieveKnowledge {
    text = "Recent battles"
    filters {
        entityLabel("Faction")
        factType("battle-event")
    }
}
```

## Architecture

### Core Components

1. **RetrievalProvider**: Interface for retrieval backends
   - `VectorRetrievalProvider`: Works with existing RAG/memory
   - `GraphRetrievalProvider`: (Future) Neo4j/graph database integration
   
2. **SmartRouter**: Automatically selects providers based on:
   - Query characteristics (temporal, multi-hop, anchored)
   - Provider capabilities
   - Routing policy configuration

3. **RetrievalQuery**: Unified query model supporting:
   - Text search
   - Temporal constraints (`at`)
   - Graph anchoring (`centerNode`)
   - Search strategies (`recipe`)
   - Filters and targets

### Search Recipes

- `HYBRID_RRF`: Reciprocal Rank Fusion of multiple search methods
- `HYBRID_MMR`: Maximal Marginal Relevance for diversity
- `HYBRID_CROSS_ENCODER`: Cross-encoder reranking for accuracy
- `HYBRID_NODE_DISTANCE`: Graph distance-based reranking
- `VECTOR_SIMILARITY`: Pure embedding similarity
- `TEXT_BM25`: Traditional text search

## Integration with Graph Databases

When a graph provider is configured, the system enables:

```kotlin
// Configure with graph database backend
val graphProvider = GraphRetrievalProvider(
    neo4jUri = "bolt://localhost:7687",
    neo4jUser = "neo4j", 
    neo4jPassword = "password"
)

val router = createSmartRouter(
    memoryProvider = memoryProvider,
    graphProvider = graphProvider,
    policy = RoutingPolicy(
        enableGraph = true,
        temporalRequiresGraph = true
    )
)

install(AgentMemory) {
    retriever = router
}
```

## Minecraft Faction Server Example

Here's how this enables rich, emotional AI behavior in multiplayer scenarios:

```kotlin
// Track faction relationships over time
val factionStrategy = strategy<String, String>("faction-assistant") {
    // Load historical context
    val checkAlliances by nodeRetrieveKnowledge {
        text = "What are the current faction alliances and rivalries?"
        target = RetrievalTarget.FACTS
        recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE
        centerNode = "faction:${currentFaction}"
    }
    
    // Check for recent betrayals or attacks
    val checkThreats by nodeRetrieveKnowledge {
        text = "Who has attacked or betrayed us recently?"
        at = Clock.System.now() - 7.days  // Last week
        filters {
            factType("betrayal")
            factType("attack") 
        }
    }
    
    // Find potential allies based on shared enemies
    val findAllies by nodeRetrieveKnowledge {
        text = "Which factions share enemies with us?"
        recipe = RetrievalRecipe.HYBRID_NODE_DISTANCE
        centerNode = "faction:${currentFaction}"
    }
    
    // Chain the searches to build comprehensive understanding
    edge(nodeStart forwardTo checkAlliances)
    edge(checkAlliances forwardTo checkThreats)
    edge(checkThreats forwardTo findAllies)
    edge(findAllies forwardTo nodeFinish transformed { results ->
        // AI can now reason with full emotional/relational context
        buildStrategicAdvice(results)
    })
}
```

## Tool Support

When configured, a `knowledge_search` tool is automatically exposed to the LLM:

```kotlin
@Tool
@LLMDescription("Search world knowledge with temporal & relational awareness")
suspend fun knowledge_search(
    query: String,
    centerNode: String? = null,
    atIso: String? = null,
    k: Int = 10
): String
```

This allows the LLM to perform complex searches during conversations.

## Migration Guide

Existing code continues to work unchanged. To add retrieval:

1. Create a retrieval provider (start with `VectorRetrievalProvider`)
2. Add `retriever` to your `AgentMemory` config
3. Use `nodeRetrieveKnowledge` in your strategies

No breaking changes - the existing memory nodes and APIs remain intact.

## Future Enhancements

- Full graph database integration with episodes, bi-temporal queries
- Community detection and clustering
- Custom entity/relationship types
- Cross-encoder reranking models
- Streaming retrieval results