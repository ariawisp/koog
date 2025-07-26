package ai.koog.agents.core.agent.entity.planner

import ai.koog.agents.core.agent.context.AIAgentContextBase

public interface PlanBase

/**
 * Represents the state of a planning agent, including the context and current value.
 *
 * @param Value The type of the value held by the state.
 * @property context The context of the AI agent which includes all necessary resources and metadata for execution.
 * @property value The current value held by the state, which can be updated during plan execution.
 */
public data class PlanningAgentState<PlanT : PlanBase>(
    val context: AIAgentContextBase<*>,
    val value: Any?,
    val currentPlan: PlanT? = null,
)

/**
 * Represents a plan with steps to be executed by a planning agent.
 */
public interface Plan<PlanT : PlanBase> : PlanBase {
    /**
     * Checks if the plan has been completed based on the current state.
     *
     * @param state The current state of the planning agent.
     * @return True if the plan has been completed, false otherwise.
     */
    public fun completed(state: PlanningAgentState<*>): Boolean

    /**
     * Executes the next step of the plan based on the current state.
     *
     * @param state The current state of the planning agent.
     * @return The updated state after executing the step.
     */
    public suspend fun executeStep(state: PlanningAgentState<PlanT>): PlanningAgentState<PlanT>
}

/**
 * Represents an assessment of a plan's execution, indicating whether to continue with the current plan or replan.
 */
public sealed class PlanAssessment {
    /**
     * Indicates that the plan should be replanned based on the current state.
     *
     * @property reason The reason for replanning.
     */
    public data class Replan(val reason: String) : PlanAssessment()

    /**
     * Indicates that the plan should continue execution without replanning.
     */
    public data object Continue : PlanAssessment()
}
