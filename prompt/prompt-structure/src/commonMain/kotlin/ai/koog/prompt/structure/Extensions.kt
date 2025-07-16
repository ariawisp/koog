package ai.koog.prompt.structure

import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.model.PromptExecutor
import ai.koog.prompt.executor.model.PromptExecutorExt.execute
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.markdown.markdown
import ai.koog.prompt.message.Message
import ai.koog.prompt.text.TextContentBuilderBase
import ai.koog.prompt.structure.json.JsonStructuredData
import ai.koog.prompt.structure.json.generator.JsonSchemaGenerator

/**
 * Adds a structured representation of the given value to the [TextContentBuilderBase].
 *
 * @param structure The structure definition
 * @param value The value to be serialized and added to the builder.
 */
public fun <T> TextContentBuilderBase<*>.structure(structure: StructuredData<T, *>, value: T) {
    +structure.pretty(value)
}

/**
 * Represents a container for structured data parsed from response message.
 *
 * This class is designed to encapsulate both the parsed structured output and the original raw
 * text as returned from a processing step, such as a language model execution.
 *
 * @param T The type of the structured data contained within this response.
 * @property structure The parsed structured data corresponding to the specific schema.
 * @property message The original assistant message from which the structure was parsed.
 */
public data class StructuredResponse<T>(val structure: T, val message: Message.Assistant)

/**
 * Configures structured output behavior.
 * Defines which structures in which modes should be used for each provider when requesting a structured output.
 *
 * @property default Fallback [StructuredOutput] to be used when there's no suitable structure found in [byProvider]
 * for a requested [LLMProvider]. Defaults to `null`, meaning structured output would fail with error in such a case.
 *
 * @property byProvider A map matching [LLMProvider] to compatible [StructuredOutput] definitions. Each provider may
 * require different schema formats. E.g. for [JsonStructuredData] this means you have to use the appropriate
 * [JsonSchemaGenerator] implementation for each provider for [StructuredOutput.Native], or fallback to [StructuredOutput.Manual]
 *
 * @property fixingParser Optional parser that handles malformed responses by using an auxiliary LLM to
 * intelligently fix parsing errors. When specified, parsing errors trigger additional
 * LLM calls with error context to attempt correction of the structure format.
 */
public data class StructuredOutputConfig<T>(
    public val default: StructuredOutput<T>? = null,
    public val byProvider: Map<LLMProvider, StructuredOutput<T>> = emptyMap(),
    public val fixingParser: StructureFixingParser? = null
)

/**
 * Defines how structured outputs should be generated.
 *
 * Can be [StructuredOutput.Manual] or [StructuredOutput.Native]
 * 
 * @param T The type of structured data.
 */
public sealed interface StructuredOutput<T> {
    /**
     * The definition of a structure.
     */
    public val structure: StructuredData<T, *>

    /**
     * Instructs the model to produce structured output through explicit prompting.
     * 
     * Uses an additional user message containing [StructuredData.definition] to guide 
     * the model in generating correctly formatted responses.
     * 
     * @property structure The structure definition to be used in output generation.
     */
    public data class Manual<T>(override val structure: StructuredData<T, *>) : StructuredOutput<T>

    /**
     * Leverages a model's built-in structured output capabilities.
     * 
     * Uses [StructuredData.schema] to define the expected response format through the model's
     * native structured output functionality.
     *
     * Note: [StructuredData.examples] are not used with this mode, only the schema is sent via parameters.
     *
     * @property structure The structure definition to be used in output generation.
     */
    public data class Native<T>(override val structure: StructuredData<T, *>) : StructuredOutput<T>
}

/**
 * Executes a prompt with structured output parsing, automatically augmenting the prompt
 * with schema instructions and parsing the response into the defined structure.
 *
 * **Note**: While many language models advertise support for structured output via JSON schema,
 * the actual level of support varies between models and even between versions
 * of the same model. Some models may produce malformed outputs or deviate from
 * the schema in subtle ways, especially with complex structures like polymorphic types.
 * In such cases, consider using [StructuredOutputConfig.fixingParser] to handle potential formatting issues.
 *
 * @param prompt The prompt to be executed.
 * @param model LLM to execute requests.
 * @param config A configuration defining structures and behavior.
 *
 * @return [kotlin.Result] with parsed [StructuredResponse] or error.
 */
public suspend fun <T> PromptExecutor.executeStructured(
    prompt: Prompt,
    model: LLModel,
    config: StructuredOutputConfig<T>,
): Result<StructuredResponse<T>> {
    val mode = config.byProvider[model.provider]
        ?: config.default
        ?: throw IllegalArgumentException("No structure found for provider ${model.provider}")

    val (structure: StructuredData<T, *>, updatedPrompt: Prompt) = when (mode) {
        // Don't set schema parameter in prompt and coerce the model manually with user message to provide a structured response.
        is StructuredOutput.Manual -> {
            mode.structure to prompt(prompt) {
                user {
                    markdown {
                        StructuredOutputPrompts.outputInstruction(this, mode.structure)
                    }
                }
            }
        }

        // Rely on built-in model capabilities to provide structured response.
        is StructuredOutput.Native -> {
            mode.structure to prompt.withUpdatedParams { schema = mode.structure.schema }
        }
    }

    val response = this.execute(prompt = updatedPrompt, model = model)

    return runCatching {
        require(response is Message.Assistant) { "Response for structured output must be an assistant message, got ${response::class.simpleName} instead" }

        // Use fixingParser if provided, otherwise parse directly
        val structureResponse = config.fixingParser
            ?.parse(this, structure, response.content)
            ?: structure.parse(response.content)

        StructuredResponse(
            structure = structureResponse,
            message = response
        )
    }
}
