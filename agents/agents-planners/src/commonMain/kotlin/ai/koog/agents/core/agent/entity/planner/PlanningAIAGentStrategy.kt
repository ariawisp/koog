package ai.koog.agents.core.agent.entity.planner

import ai.koog.agents.core.agent.context.AIAgentContextBase
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
public abstract class PlanningAIAgentStrategy<PlanType : Plan<PlanType>> internal constructor(
    override val name: String,
    toolSelectionStrategy: ToolSelectionStrategy
) : AIAgentStrategy<String, String> {
    /**
     * Constructs a plan for the AI agent strategy based on the provided state and optional assessment.
     * This method is abstract and must be implemented to define the specific planning logic for the agent.
     *
     * @param state The current state of the planning agent, which provides the context and inputs for plan construction.
     * @param assessment An optional evaluation of a previous plan's performance that can inform the new planning process.
     * @return A plan representing the series of steps or processes to be executed by the AI agent.
     */
    public abstract suspend fun buildPlan(
        state: PlanningAgentState<PlanType>,
        assessment: PlanAssessment? = null
    ): PlanType

    /**
     * Evaluates the provided plan against the specified state of the planning agent and determines the next steps.
     *
     * @param Value The type of the value processed by the plan.
     * @param plan The plan to be evaluated, representing a series of steps or processes.
     * @param state The current state of the planning agent, which includes contextual information and the value being processed.
     * @return A PlanAssessment which indicates whether to continue execution of the plan or replan based on the evaluation.
     */
    public abstract suspend fun evaluatePlan(
        plan: PlanType,
        state: PlanningAgentState<PlanType>
    ): PlanAssessment


    override suspend fun execute(context: AIAgentContextBase<*>, input: String): String? {
        var state = PlanningAgentState<PlanType>(context = context, value = input)
        var plan = buildPlan(state)

        while (!plan.completed(state)) {
            state = plan.executeStep(state)
            val planAssessment = evaluatePlan(plan, state)
            when (planAssessment) {
                is PlanAssessment.Replan -> {
                    plan = buildPlan(state, planAssessment)
                }

                is PlanAssessment.Continue -> {}
            }
        }

        return state.value.toString()
    }
}