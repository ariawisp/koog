package ai.koog.agents.core.agent.entity.simple

import ai.koog.agents.core.agent.entity.AIAgentStrategy
import ai.koog.agents.core.agent.entity.graph.ToolSelectionStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi

/**
 * Represents a strategy for managing and executing AI agent workflows built manually using []
 *
 * @property name The unique identifier for the strategy.
 * @property nodeStart The starting node of the strategy, initiating the subgraph execution. By default Start node gets the agent input and returns
 * @property nodeFinish The finishing node of the strategy, marking the subgraph's endpoint.
 * @property toolSelectionStrategy The strategy responsible for determining the toolset available during subgraph execution.
 */
@OptIn(InternalAgentsApi::class)
public abstract class SimpleAIAgentStrategy<Input, Output> internal constructor(
    override val name: String,
    toolSelectionStrategy: ToolSelectionStrategy
) : AIAgentStrategy<Input, Output> {}