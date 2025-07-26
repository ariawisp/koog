package ai.koog.agents.core.agent.entity.planner

import ai.koog.agents.core.agent.entity.graph.ToolSelectionStrategy
import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.markdown.markdown
import ai.koog.prompt.message.Message
import ai.koog.prompt.structure.json.JsonSchemaGenerator
import ai.koog.prompt.structure.json.JsonStructuredData
import kotlinx.serialization.Serializable

/**
 * A planning strategy that uses another AIAgent with graphStrategy to build a plan,
 * and uses another AIAgent to critique the plan.
 *
 * @param Input The type of input data this strategy processes.
 * @param Output The type of output data this strategy produces.
 * @property name The name of the strategy.
 * @property toolSelectionStrategy The strategy for selecting tools.
 * @property promptExecutor The prompt executor to use for creating AIAgents.
 */
@OptIn(InternalAgentsApi::class)
public class SimplePlannerWithCritic(
    override val name: String,
    toolSelectionStrategy: ToolSelectionStrategy,
    private val promptExecutor: PromptExecutor
) : SimplePlanner(name, toolSelectionStrategy) {

    /**
     * Data class for structured output from the plan evaluation.
     */
    @Serializable
    private data class PlanEvaluation(
        val shouldContinue: Boolean,
        val reason: String
    )

    override suspend fun evaluatePlan(plan: SimplePlan, state: PlanningAgentState<SimplePlan>): PlanAssessment {
        // Use the LLM session to create a structured request for plan evaluation
        val evaluation = state.context.llm.writeSession {
            val oldPrompt = prompt.copy()

            rewritePrompt {
                // Create a prompt for the LLM critic
                prompt("critic") {
                    system {
                        markdown {
                            h1("Plan Evaluation Task")
                            textWithNewLine("You are a critical evaluator of plans. Your job is to assess whether the current plan is still valid and should be continued, or if it needs to be replanned.")

                            h2("Current Plan")
                            textWithNewLine("Goal: ${plan.goal}")

                            h3("Steps:")
                            numbered {
                                plan.steps.forEach {
                                    if (it.isCompleted) {
                                        item("[COMPLETED] ${it.description}")
                                    } else {
                                        item("${it.description}")
                                    }
                                }
                            }

                            h2("Current State")
                            textWithNewLine("Current state value: ${state.value}")

                            h2("Evaluation Instructions")
                            textWithNewLine("Please evaluate this plan carefully. Consider:")
                            bulleted {
                                item("Is the plan still aligned with the goal?")
                                item("Are the remaining steps sufficient to achieve the goal?")
                                item("Has any new information emerged that makes the plan obsolete?")
                                item("Are there any logical flaws or inefficiencies in the plan?")
                                item("Are there any steps that might be impossible to execute?")
                            }

                            textWithNewLine("Provide a structured response with your decision and reasoning.")
                        }
                    }

                    oldPrompt.messages.filter { it !is Message.System }.forEach {
                        message(it)
                    }
                }
            }

            // Make a structured request to the LLM
            val evaluationResult = requestLLMStructured(
                structure = JsonStructuredData.createJsonStructure<PlanEvaluation>(
                    schemaFormat = JsonSchemaGenerator.SchemaFormat.JsonSchema,
                    examples = listOf(
                        PlanEvaluation(
                            shouldContinue = true,
                            reason = "The plan is well-structured and all steps are logical and achievable. " +
                                    "The completed steps have made good progress toward the goal and nothing prevents " +
                                    "the plan from continuing. " +
                                    "Newly discovered observations do not invalidate the plan."
                        ),
                        PlanEvaluation(
                            shouldContinue = false,
                            reason = "The plan needs to be revised because step 3 is no longer feasible " +
                                    "given the new information in the current state.\n" +
                                    "Specific information is: ........\n"
                        )
                    )
                ),
                retries = 3
            ).getOrThrow()

            // restore original prompt
            rewritePrompt { oldPrompt }

            evaluationResult.structure
        }

        // Return the appropriate PlanAssessment based on the evaluation
        return if (evaluation.shouldContinue) {
            PlanAssessment.Continue
        } else {
            PlanAssessment.Replan(reason = evaluation.reason)
        }
    }
}
