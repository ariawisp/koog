package ai.koog.agents.core.agent.entity

import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.utils.runCatchingCancellable

/**
 * Represents a strategy interface for managing and executing AI agent workflows.
 */
public interface AIAgentStrategy<Input, Output> {
    /**
     * Name of the agent strategy
     * */
    public val name: String

    /**
     * This function determines the execution strategy of the AI agent workflow.
     *
     * @param context The context of the AI agent which includes all necessary resources and metadata for execution.
     * @param input The input object representing the data to be processed by the AI agent.
     * @return The output of the AI agent execution, generated after processing the input.
     */
    public suspend fun execute(context: AIAgentContextBase<*>, input: Input): Output?
}
