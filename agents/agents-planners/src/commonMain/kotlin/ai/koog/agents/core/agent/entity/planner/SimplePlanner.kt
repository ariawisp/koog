package ai.koog.agents.core.agent.entity.planner

import ai.koog.agents.core.agent.entity.graph.ToolSelectionStrategy
import ai.koog.agents.core.agent.entity.planner.SimplePlanner.SimplePlan
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.agents.core.dsl.extension.replaceHistoryWithTLDR
import ai.koog.agents.memory.feature.history.RetrieveFactsFromHistory
import ai.koog.agents.memory.model.Concept
import ai.koog.agents.memory.model.FactType
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.markdown.markdown
import ai.koog.prompt.message.Message
import ai.koog.prompt.structure.json.JsonSchemaGenerator
import ai.koog.prompt.structure.json.JsonStructuredData
import kotlinx.serialization.Serializable

/**
 * A simple planning strategy that uses LLM requests to build a plan.
 *
 * @param Input The type of input data this strategy processes.
 * @param Output The type of output data this strategy produces.
 * @property name The name of the strategy.
 * @property toolSelectionStrategy The strategy for selecting tools.
 */
@OptIn(InternalAgentsApi::class)
public open class SimplePlanner(
    override val name: String, toolSelectionStrategy: ToolSelectionStrategy
) : PlanningAIAgentStrategy<SimplePlan>(name, toolSelectionStrategy) {

    /**
     * Represents a step in the plan.
     *
     * @property description The description of the step.
     * @property isCompleted Whether the step has been completed.
     */
    @Serializable
    public data class PlanStep(
        val description: String, val isCompleted: Boolean = false
    )

    /**
     * Represents a structured plan with steps.
     *
     * @property goal The goal of the plan.
     * @property steps The steps to achieve the goal.
     */
    @Serializable
    private data class StructuredPlan(
        val goal: String, val steps: List<PlanStep>
    )

    /**
     * A simple implementation of the Plan interface that uses a list of steps.
     */
    public class SimplePlan(
        public val goal: String, public val steps: MutableList<PlanStep>
    ) : Plan<SimplePlan> {
        override fun completed(state: PlanningAgentState<*>): Boolean {
            return steps.all { it.isCompleted }
        }

        override suspend fun executeStep(state: PlanningAgentState<SimplePlan>): PlanningAgentState<SimplePlan> {
            val currentStep = steps.firstOrNull { !it.isCompleted } ?: return state

            // Execute the step using LLM
            val result = state.context.llm.writeSession {
                updatePrompt {
                    system("You are executing a step in a plan. The goal is: $goal")
                    user("Execute the following step: ${currentStep.description}")
                    user("Current state: ${state.value}")
                }

                requestLLMWithoutTools()
            }

            // Mark the step as completed
            val stepIndex = steps.indexOf(currentStep)
            steps[stepIndex] = currentStep.copy(isCompleted = true)

            // Return updated state with the result
            return state.copy(value = result.content)
        }
    }

    override suspend fun buildPlan(
        state: PlanningAgentState<SimplePlan>, assessment: PlanAssessment?
    ): SimplePlan {
        val shouldReplan = assessment is PlanAssessment.Replan

        if (shouldReplan) {
            state.context.llm.writeSession {
                replaceHistoryWithTLDR(
                    strategy = RetrieveFactsFromHistory(
                        Concept(
                            keyword = "achievements",
                            description = "What important milestones have been achieved",
                            factType = FactType.MULTIPLE
                        ),
                        Concept(
                            keyword = "struggles",
                            description = "What major problems have been encountered throughout the course of the task, and how they have been resolved",
                            factType = FactType.MULTIPLE
                        )
                    )
                )
            }
        }

        val structuredPlan = state.context.llm.writeSession {
            rewritePrompt { oldPrompt ->
                prompt("planner") {
                    system {
                        markdown {
                            h1("Main Goal -- Create a Plan")
                            textWithNewLine("You are a planning agent. Your task is to create a detailed plan with steps.")

                            if (shouldReplan) {
                                h1("Previous Plan (failed)")

                                textWithNewLine("Previously it was attempted to solve the problem with another plan, but it has failed")
                                textWithNewLine("Below you'll see the previous plan with the reason for replan")

                                if (state.currentPlan != null) {
                                    h2("Previous Plan Overview")

                                    textWithNewLine("Previously, the following plan has been tried:")

                                    h3("Previous Plan Goal")
                                    textWithNewLine("The goal of the previous plan was:")
                                    textWithNewLine(state.currentPlan.goal)

                                    h3("Previous Plan Steps")
                                    textWithNewLine("The previous plan consisted of the following consequtive steps:")

                                    bulleted {
                                        state.currentPlan.steps.forEach {
                                            if (it.isCompleted) {
                                                item("[COMPLETED!] ${it.description}")
                                            } else {
                                                item(it.description)
                                            }
                                        }
                                    }
                                }

                                h2("Reason(s) to Replan")

                                textWithNewLine("The previous plan needs to be revised for the following reason")

                                blockquote(assessment.reason)
                            }

                            h1("What to do next?")

                            textWithNewLine("You need to create a new plan with steps that will solve the user's problem:")

                            blockquote(state.value.toString())

                            if (shouldReplan) {
                                bold("Note: Below you'll see some observations from the ")
                            }
                        }
                    }

                    if (shouldReplan) {
                        oldPrompt.messages.filter { it !is Message.System }.forEach {
                            message(it)
                        }
                    }
                }
            }

            val structuredPlanResult = requestLLMStructured(
                structure = JsonStructuredData.createJsonStructure<StructuredPlan>(
                    schemaFormat = JsonSchemaGenerator.SchemaFormat.JsonSchema, examples = listOf(
                        StructuredPlan(
                            goal = "The main goal to be achieved by the system",
                            steps = listOf(
                                PlanStep("First step description", isCompleted = true),
                                PlanStep("Second step description", isCompleted = true),
                                PlanStep("Some other action", isCompleted = false),
                                PlanStep("Action to be performed on the step 4", isCompleted = false),
                                PlanStep("Next high-level goal (5)", isCompleted = false),
                            )
                        )
                    )
                ), retries = 3
            ).getOrThrow()

            structuredPlanResult.structure
        }

        state.context.llm.writeSession {
            rewritePrompt { oldPrompt ->
                prompt("agent") {
                    system {
                        markdown {
                            h1("Plan")

                            textWithNewLine("You must follow the following plan to solve the problem:")

                            h2("Main Goal:")

                            textWithNewLine(structuredPlan.goal)

                            h2("Plan Steps:")

                            numbered {
                                structuredPlan.steps.forEach {
                                    if (it.isCompleted) {
                                        item("[COMPLETED!] ${it.description}")
                                    } else {
                                        item(it.description)
                                    }
                                }
                            }
                        }
                    }

                    oldPrompt.messages.filter { it !is Message.System }.forEach {
                        message(it)
                    }
                }
            }
        }

        // Create a SimplePlanImpl with the generated steps
        return SimplePlan(goal = structuredPlan.goal, steps = structuredPlan.steps.toMutableList())
    }

    override suspend fun evaluatePlan(
        plan: SimplePlan, state: PlanningAgentState<SimplePlan>
    ): PlanAssessment {
        // Simple implementation always continues with the current plan
        return PlanAssessment.Continue
    }
}
