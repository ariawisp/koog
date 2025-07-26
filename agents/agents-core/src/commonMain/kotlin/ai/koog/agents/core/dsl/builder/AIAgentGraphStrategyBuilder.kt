package ai.koog.agents.core.dsl.builder

import ai.koog.agents.core.agent.entity.graph.FinishNode
import ai.koog.agents.core.agent.entity.graph.AIAgentGraphStrategy
import ai.koog.agents.core.agent.entity.graph.StartNode
import ai.koog.agents.core.agent.entity.graph.ToolSelectionStrategy

/**
 * A builder class responsible for constructing an instance of `AIAgentStrategy`.
 * The `AIAgentStrategyBuilder` serves as a specific configuration for creating AI agent strategies
 * with a defined start and finish node, along with a designated tool selection strategy.
 *
 * @param name The name of the strategy being built, serving as a unique identifier.
 * @param toolSelectionStrategy The strategy used to determine the subset of tools available during subgraph execution.
 */
public class AIAgentGraphStrategyBuilder<Input, Output>(
    private val name: String,
    private val toolSelectionStrategy: ToolSelectionStrategy,
) : AIAgentSubgraphBuilderBase<Input, Output>(), BaseBuilder<AIAgentGraphStrategy<Input, Output>> {
    public override val nodeStart: StartNode<Input> = StartNode()
    public override val nodeFinish: FinishNode<Output> = FinishNode()

    override fun build(): AIAgentGraphStrategy<Input, Output> {

        val strategy = AIAgentGraphStrategy(
            name = name,
            nodeStart = nodeStart,
            nodeFinish = nodeFinish,
            toolSelectionStrategy = toolSelectionStrategy
        )
        strategy.metadata = buildSubgraphMetadata(nodeStart, name, strategy)
        return strategy
    }
}


/**
 * Builds a local graph-based AI agent strategy that processes user input through a sequence of stages.
 *
 * The agent executes a series of steps defined as a graph workflow, with each step receiving the output
 * of the previous step as its input.
 *
 * @property name The unique identifier for this agent.
 * @param init Lambda that defines stages and nodes of this agent
 */
public fun <Input, Output> graphStrategy(
    name: String,
    toolSelectionStrategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    init: AIAgentGraphStrategyBuilder<Input, Output>.() -> Unit,
): AIAgentGraphStrategy<Input, Output> {
    return AIAgentGraphStrategyBuilder<Input, Output>(name, toolSelectionStrategy)
        .apply(init)
        .build()
}

/**
 * Builds a graph-based strategy for AIAgent.
 *
 * Deprecated, please use [graphStrategy], instead
 * */
@Deprecated(
    level = DeprecationLevel.WARNING,
    message = "Use `graphStrategy` instead",
    replaceWith = ReplaceWith("graphStrategy(name, toolSelectionStrategy, init)", "ai.koog.agents.core")
)
public fun <Input, Output> strategy(
    name: String,
    toolSelectionStrategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    init: AIAgentGraphStrategyBuilder<Input, Output>.() -> Unit,
): AIAgentGraphStrategy<Input, Output> = graphStrategy(name, toolSelectionStrategy, init)