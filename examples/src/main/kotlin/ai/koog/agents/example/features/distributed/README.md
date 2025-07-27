# Distributed Coordination Examples

This directory contains examples demonstrating Koog's **Distributed Coordination** feature, which enables true multi-agent systems across different processes and machines.

## What is Distributed Coordination?

Unlike Google ADK's pseudo "multi-agent" pattern (which is really just function orchestration), Koog's Distributed Coordination enables **true distributed multi-agent systems** where:

- **Agents run in different processes** and coordinate via PubSub messaging
- **Role-based architecture** with Planners, Executors, Observers, and Coordinators
- **Intelligent task distribution** based on agent capabilities and specializations
- **Automatic discovery** and presence tracking across the agent network
- **Strategy-driven coordination** with pluggable coordination patterns

## Examples

### 1. DistributedCoordination.kt
**Comprehensive business analysis with distributed agents**

Demonstrates a complete multi-agent system where:
- **Planner Agent**: Strategic coordination and task breakdown
- **Text Executor**: Writing, summarization, content creation
- **Data Executor**: Mathematical analysis, statistics, modeling
- **Research Executor**: Information gathering, fact verification

**Use Case**: Complex business analysis requiring multiple specialized capabilities.

**Run**: `./gradlew runExampleDistributedCoordination`

### 2. MinecraftDistributedAgents.kt (in ktor-pubsub-minecraft/)
**Cross-process coordination between web servers and game servers**

Demonstrates practical distributed architecture:
- **Ktor Web Server**: Planner agent handling user requests
- **Minecraft Server**: Executor agents handling in-game actions
- **Cross-Process Coordination**: Web requests trigger game actions

**Use Case**: Web applications coordinating with specialized game/IoT environments.

**Run**: `./gradlew runExampleKtorMinecraftDistributed`

## Key Concepts Demonstrated

### Role-Based Architecture
```kotlin
// Planner Agent - Strategic coordination
install(Distributed) {
    agentId = DistributedAgentId(
        role = AgentRole.PLANNER,
        capabilities = setOf("planning", "coordination", "strategy")
    )
    strategy = PlannerExecutorStrategy()
}

// Executor Agent - Specialized execution
install(Distributed) {
    agentId = DistributedAgentId(
        role = AgentRole.EXECUTOR,
        capabilities = setOf("text-processing", "writing", "analysis")
    )
    strategy = PlannerExecutorStrategy()
}
```

### Cross-Process Communication
```kotlin
// Agents communicate via PubSub topics
install(PubSub) {
    provider = pubSubProvider
    autoSubscribeTopics = listOf("task-assignments", "execution-results")
}
```

### Capability-Based Routing
```kotlin
// Tasks are routed to agents based on their declared capabilities
val textAgents = distributedContext.findAgentsByCapability("writing")
val dataAgents = distributedContext.findAgentsByCapability("mathematics")
```

## Real-World Use Cases

### 1. **Microservice Agent Architecture**
- Each microservice runs specialized agents
- Agents coordinate across service boundaries
- Enables intelligent inter-service communication

### 2. **Voice + Game Integration**
- Discord bots coordinate with game servers
- Voice commands trigger in-game actions
- Real-time feedback between platforms

### 3. **IoT + Web Coordination**
- Web interfaces coordinate with IoT device agents
- Edge computing agents handle local processing
- Central coordination for complex workflows

### 4. **Enterprise Multi-Agent Systems**
- Planning agents in central systems
- Execution agents in specialized environments
- Automated task distribution and monitoring

## Architecture Benefits

### vs. Google ADK "Multi-Agent"
- **Google ADK**: Function orchestration within single process
- **Koog Distributed**: True multi-agent systems across processes/machines

### vs. Traditional Microservices
- **Traditional**: Rigid API contracts and manual coordination
- **Koog Distributed**: Intelligent agent coordination with capability discovery

### vs. Message Queues
- **Message Queues**: Simple pub/sub with manual message handling
- **Koog Distributed**: AI-powered coordination strategies with automatic routing

## Getting Started

1. **Install Dependencies**: Distributed feature requires PubSub feature
2. **Choose Provider**: LocalFile (dev), Redis (production), GCP (enterprise)
3. **Define Roles**: Plan your agent roles and capabilities
4. **Implement Strategy**: Use built-in or custom coordination strategies
5. **Deploy Distributed**: Run agents across your infrastructure

## Next Steps

- Explore the examples to understand coordination patterns
- Adapt the patterns to your specific use cases
- Consider custom coordination strategies for specialized workflows
- Scale to production with Redis or GCP PubSub providers