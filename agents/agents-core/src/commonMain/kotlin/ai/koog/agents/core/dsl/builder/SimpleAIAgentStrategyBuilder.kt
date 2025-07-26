package ai.koog.agents.core.dsl.builder

import ai.koog.agents.core.agent.AIAgentMaxNumberOfIterationsReachedException
import ai.koog.agents.core.agent.context.AIAgentContextBase
import ai.koog.agents.core.agent.context.AIAgentLLMContext
import ai.koog.agents.core.agent.entity.AIAgentStateManager
import ai.koog.agents.core.agent.entity.simple.SimpleAIAgentStrategy
import ai.koog.agents.core.agent.entity.graph.ToolSelectionStrategy
import ai.koog.agents.core.agent.session.AIAgentLLMReadSession
import ai.koog.agents.core.agent.session.AIAgentLLMWriteSession
import ai.koog.agents.core.environment.AIAgentEnvironment
import ai.koog.agents.core.environment.ReceivedToolResult
import ai.koog.prompt.message.Message
import io.github.oshai.kotlinlogging.KLogger
import io.github.oshai.kotlinlogging.KotlinLogging

/**
 * Builds a simply defined AI agent strategy that processes user input and produces an output.
 *
 * You can write your own custom agent logic (basically, any Kotlin code) inside the [simpleStrategy] builder using
 * the provided SimpleAgen
 *
 * @property name The unique identifier for this agent.
 * @param execute Lambda that defines stages and nodes of this agent
 */
public fun <Input, Output> simpleStrategy(
    name: String,
    toolSelectionStrategy: ToolSelectionStrategy = ToolSelectionStrategy.ALL,
    execute: suspend SimpleAIAgentStrategyContext.(Input) -> Output,
): SimpleAIAgentStrategy<Input, Output> {
    return object : SimpleAIAgentStrategy<Input, Output>(
        name = name,
        toolSelectionStrategy = toolSelectionStrategy
    ) {
        override suspend fun execute(
            context: AIAgentContextBase<*>,
            input: Input
        ): Output {
            val strategyContext = SimpleAIAgentStrategyContext(
                toolSelectionStrategy,
                context as AIAgentContextBase<SimpleAIAgentStrategy<*, *>>
            )
            return strategyContext.execute(input)
        }
    }
}

internal class IterationsChecker(
    val stateManager: AIAgentStateManager,
    val maxAgentIterations: Int,
    val logger: KLogger,
) {
    suspend fun validateMaxIterations() {
        stateManager.withStateLock { state ->
            if (++state.iterations > maxAgentIterations) {
                logger.error { "Max iterations limit ($maxAgentIterations) reached" }

                throw AIAgentMaxNumberOfIterationsReachedException(maxAgentIterations)
            }
        }
    }
}

internal class EnvironmentWrapper(
    val environment: AIAgentEnvironment,
    val iterationsChecker: IterationsChecker
) : AIAgentEnvironment {
    override suspend fun executeTools(toolCalls: List<Message.Tool.Call>): List<ReceivedToolResult> {
        iterationsChecker.validateMaxIterations()

        return environment.executeTools(toolCalls)
    }

    override suspend fun reportProblem(exception: Throwable) {
        return environment.reportProblem(exception)
    }
}

internal class LLMWrapper(
    val llm: AIAgentLLMContext,
    val iterationsChecker: IterationsChecker
) : AIAgentLLMContext(
    llm.tools,
    llm.toolRegistry,
    llm.prompt,
    llm.model,
    llm.promptExecutor,
    llm.environment,
    llm.config,
    llm.clock
) {
    public override suspend fun <T> writeSession(block: suspend AIAgentLLMWriteSession.() -> T): T {
        iterationsChecker.validateMaxIterations()

        return llm.writeSession(block)
    }


    public override suspend fun <T> readSession(block: suspend AIAgentLLMReadSession.() -> T): T {
        iterationsChecker.validateMaxIterations()

        return llm.readSession(block)
    }
}

/**
 * Represents a context for configuring and executing AI agent strategies within a specific operational scope.
 *
 * This class acts as a bridge, integrating a strategy for selecting tools that an AI agent
 * uses during its operation and delegating contextual operations to an existing AI agent context.
 *
 * The `toolSelectionStrategy` provides the mechanism for determining the subset of tools
 * available for use within the defined context, while the `agentContext` serves as the
 * underlying execution context for the AI agent.
 *
 * Delegates functionality to an instance of `AIAgentContextBase`, inheriting its behaviors
 * and configurations for AI agent operations.
 *
 * @property toolSelectionStrategy A strategy defining which tools should be selected for the context.
 * @property agentContext The base context for the AI agent operations, delegated to by this class.
 */
public class SimpleAIAgentStrategyContext(
    private val toolSelectionStrategy: ToolSelectionStrategy,
    public val agentContext: AIAgentContextBase<SimpleAIAgentStrategy<*, *>>
) : AIAgentContextBase<SimpleAIAgentStrategy<*, *>> by agentContext {
    private val logger = KotlinLogging.logger {}

    private val iterationsChecker = IterationsChecker(
        agentContext.stateManager,
        agentContext.config.maxAgentIterations,
        logger
    )

    /**
     * The `environment` property represents the operational environment in which the AI agent interacts.
     * It wraps the base environment from the `agentContext` and adds functionality to monitor
     * and enforce constraints such as iteration limits through an `IterationsChecker`.
     *
     * The `EnvironmentWrapper` ensures that all tool execution and problem reporting operations
     * are delegated to the base environment, while also validating constraints before any actions are executed.
     *
     * This property serves as the primary interface for the AI agent to interact with its external environment
     * while adhering to defined operational rules and limits.
     */
    override val environment: AIAgentEnvironment = EnvironmentWrapper(agentContext.environment, iterationsChecker)

    /**
     * Represents the language model (LLM) context associated with the current AI agent strategy.
     *
     * This property provides an instance of [AIAgentLLMContext], wrapped with additional functionality
     * via [LLMWrapper], which enforces constraints and validations (e.g., iteration checks) during
     * read and write sessions. It serves as the primary interface for managing tools, prompts, models,
     * and other LLM-related operations within the current strategy context.
     *
     * The LLM context is responsible for interaction with the AI agent's execution and environment layers,
     * concurrent operation handling, and context-specific configurations defined in the containing class.
     */
    override val llm: AIAgentLLMContext = LLMWrapper(agentContext.llm, iterationsChecker)
}