# Distributed Agent Feature

Enables coordination between Koog agents running across different processes and machines through PubSub messaging.

## Overview

The Distributed feature provides a coordination layer that allows agents to communicate and collaborate across process boundaries. Unlike Google ADK's pseudo "multi-agent" pattern (which is really just function orchestration), this enables true distributed multi-agent systems where agents are autonomous processes that can coordinate through messaging.

## Key Features

- **Agent Role Management**: Define agents as PLANNER, EXECUTOR, OBSERVER, or COORDINATOR
- **Distributed Strategy Patterns**: Built-in coordination strategies for common patterns
- **Message-based Communication**: Type-safe messaging through PubSub providers
- **Agent Discovery**: Automatic discovery and lifecycle management of distributed agents
- **Cross-process Coordination**: Enables planner-executor separation across different machines

## Architecture

```
┌─────────────────┐    PubSub Messages     ┌─────────────────┐
│ Planner Agent   │ ←─────────────────────→ │ Executor Agent  │
│ (Ktor Server)   │                        │ (Minecraft)     │
│                 │    Task Assignments    │                 │
│ - Strategy      │ ──────────────────────→ │ - Tool Execution│
│ - Planning      │ ←────────────────────── │ - Results       │
│ - Coordination  │    Execution Results   │ - Status Updates│
└─────────────────┘                        └─────────────────┘
```

## Usage

### Basic Planner-Executor Setup

```kotlin
// Planner Agent (runs in Ktor server)
val plannerAgent = AIAgent(
    promptExecutor = anthropicExecutor,
    strategy = SimplePlannerWithCritic("TaskPlanner", AllToolsStrategy(), anthropicExecutor)
) {
    install(PubSub) {
        provider = RedisPubSubProvider("redis://localhost:6379")
    }
    
    install(Distributed) {
        agentId = DistributedAgentId(
            role = AgentRole.PLANNER,
            capabilities = setOf("planning", "strategy", "coordination")
        )
        strategy = PlannerExecutorStrategy()
        discoveryEnabled = true
    }
}

// Executor Agent (runs in Minecraft server)
val executorAgent = AIAgent(
    promptExecutor = openAIExecutor,
    strategy = MinecraftExecutorStrategy(minecraftTools)
) {
    install(PubSub) {
        provider = RedisPubSubProvider("redis://localhost:6379")
    }
    
    install(Distributed) {
        agentId = DistributedAgentId(
            role = AgentRole.EXECUTOR,
            capabilities = setOf("minecraft", "building", "resource-gathering")
        )
        strategy = PlannerExecutorStrategy()
    }
}
```

### Custom Coordination Strategy

```kotlin
class CustomCoordinationStrategy : DistributedStrategy {
    override suspend fun onLocalInput(input: String, context: AIAgentContext): DistributedAction? {
        return when (context.distributedId.role) {
            AgentRole.PLANNER -> {
                // Generate plan and delegate to executor
                DistributedAction.SendToRole(
                    role = AgentRole.EXECUTOR,
                    payload = PlanningTask(input, priority = "high")
                )
            }
            AgentRole.EXECUTOR -> {
                // Execute locally
                DistributedAction.ExecuteLocally(input)
            }
            else -> DistributedAction.Ignore
        }
    }

    override suspend fun onRemoteMessage(message: DistributedMessage, context: AIAgentContext): DistributedAction? {
        return when (message.messageType) {
            MessageType.TASK_REQUEST -> {
                if (context.distributedId.role == AgentRole.EXECUTOR) {
                    DistributedAction.ExecuteLocally(message.payload)
                } else DistributedAction.Ignore
            }
            else -> DistributedAction.Ignore
        }
    }
}
```

## Built-in Strategies

- **PlannerExecutorStrategy**: Coordinates planner and executor agents
- **CoordinatorStrategy**: Central coordinator managing multiple specialized agents
- **ObserverStrategy**: Monitoring and reporting across agent networks
- **FanOutGatherStrategy**: Parallel task distribution and result aggregation

## Dependencies

This feature requires:
- `agents-features-pubsub`: For cross-process messaging
- A configured PubSub provider (Redis, GCP PubSub, etc.)

## Real-world Use Cases

- **Distributed AI Systems**: Coordinate AI agents across different services
- **Game Development**: Separate planning agents from game execution agents
- **IoT Coordination**: Central planning with distributed device control
- **Microservice Orchestration**: AI-driven coordination between services
- **Multi-tenant SaaS**: Isolated agent coordination per tenant