package ai.koog.prompt.params

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject

/**
 * Represents configuration parameters for controlling the behavior of a language model.
 *
 * @property temperature Controls randomness in output. Range: 0.0 to 2.0. Lower values are more focused,
 * higher values are more diverse. Default: 1.0
 * @property numberOfChoices Number of response choices to generate
 * @property speculation Reserved for speculative output prediction (e.g., OpenAI's PredictedOutput)
 * @property schema Schema for structured data responses
 * @property toolChoice Controls tool calling behavior
 * @property user Optional user identifier for tracking
 *
 * Sampling parameters:
 * @property topP Nucleus sampling: limits choices to top P probability mass (0.0 to 1.0)
 * @property topK Top-K sampling: limits choices to K most likely tokens (0 or above)
 * @property minP Minimum probability threshold relative to top token (0.0 to 1.0)
 * @property topA Top-A sampling: dynamic top-P based on highest probability token (0.0 to 1.0)
 *
 * Penalty parameters:
 * @property frequencyPenalty Reduces repetition based on token frequency (-2.0 to 2.0)
 * @property presencePenalty Reduces repetition regardless of frequency (-2.0 to 2.0)
 * @property repetitionPenalty Alternative repetition control (0.0 to 2.0)
 *
 * Output control:
 * @property maxTokens Maximum tokens to generate
 * @property seed Seed for deterministic generation
 * @property stop List of stop sequences
 *
 * Advanced features:
 * @property logprobs Whether to return log probabilities
 * @property topLogprobs Number of top tokens with log probabilities to return (0-20)
 * @property logitBias Token biases to apply before sampling
 * @property responseFormat Output format specification
 * @property parallelToolCalls Whether to enable parallel function calling
 */
@Serializable
public data class LLMParams(
    val temperature: Double? = null,
    val numberOfChoices: Int? = null,
    val speculation: String? = null,
    val schema: Schema? = null,
    val toolChoice: ToolChoice? = null,
    val user: String? = null,
    
    // Sampling parameters
    val topP: Double? = null,
    val topK: Int? = null,
    val minP: Double? = null,
    val topA: Double? = null,
    
    // Penalty parameters
    val frequencyPenalty: Double? = null,
    val presencePenalty: Double? = null,
    val repetitionPenalty: Double? = null,
    
    // Output control
    val maxTokens: Int? = null,
    val seed: Int? = null,
    val stop: List<String>? = null,
    
    // Advanced features
    val logprobs: Boolean? = null,
    val topLogprobs: Int? = null,
    val logitBias: Map<String, Double>? = null,
    val responseFormat: ResponseFormat? = null,
    val parallelToolCalls: Boolean? = null,
    
    // Streaming control
    val stream: Boolean? = null,
    val streamOptions: StreamOptions? = null,
    
    // Additional output control
    val minTokens: Int? = null,
) {
    /**
     * Combines the parameters of the current `LLMParams` instance with the provided default `LLMParams`
     * to produce a new instance. Fields that are null in the current instance are replaced by the
     * corresponding fields from the default instance.
     *
     * @param default The default `LLMParams` instance used to fill in missing values in the current instance.
     * @return A new `LLMParams` instance with missing fields replaced by corresponding fields from the default instance.
     */
    public fun default(default: LLMParams): LLMParams = copy(
        temperature = temperature ?: default.temperature,
        numberOfChoices = numberOfChoices ?: default.numberOfChoices,
        speculation = speculation ?: default.speculation,
        schema = schema ?: default.schema,
        toolChoice = toolChoice ?: default.toolChoice,
        user = user ?: default.user,
        
        // Sampling parameters
        topP = topP ?: default.topP,
        topK = topK ?: default.topK,
        minP = minP ?: default.minP,
        topA = topA ?: default.topA,
        
        // Penalty parameters
        frequencyPenalty = frequencyPenalty ?: default.frequencyPenalty,
        presencePenalty = presencePenalty ?: default.presencePenalty,
        repetitionPenalty = repetitionPenalty ?: default.repetitionPenalty,
        
        // Output control
        maxTokens = maxTokens ?: default.maxTokens,
        seed = seed ?: default.seed,
        stop = stop ?: default.stop,
        
        // Advanced features
        logprobs = logprobs ?: default.logprobs,
        topLogprobs = topLogprobs ?: default.topLogprobs,
        logitBias = logitBias ?: default.logitBias,
        responseFormat = responseFormat ?: default.responseFormat,
        parallelToolCalls = parallelToolCalls ?: default.parallelToolCalls,
        
        // Streaming control
        stream = stream ?: default.stream,
        streamOptions = streamOptions ?: default.streamOptions,
        
        // Additional output control
        minTokens = minTokens ?: default.minTokens,
    )

    /**
     * Represents a generic schema for structured data, defining a common contract
     * for schemas.
     * This is a sealed interface, enabling a restrictive set of implementations.
     */
    @Serializable
    public sealed interface Schema {
        /**
         * Represents a person's name as a string.
         * This variable is intended to store the full name or a specific format of a name.
         */
        public val name: String

        /**
         * Represents a sealed interface JSON that defines a schema entity.
         * It extends the Schema interface and has a property for schema representation.
         */
        @Serializable
        public sealed interface JSON : Schema {
            /**
             * Represents the JSON schema definition as a JsonObject.
             *
             * This property is used to store and define the structure or format of a JSON-based data schema,
             * enabling serialization, validation, and adherence to a specific format. It is commonly utilized
             * within implementations that require a structured schema for processing or validating JSON data.
             */
            public val schema: JsonObject

            /**
             * Represents a simplified JSON structure with a schema definition.
             *
             * This data class implements the `JSON` interface and provides a basic representation
             * of a JSON structure using a `name` and its corresponding `schema` in the form of a `JsonObject`.
             *
             * Use this class when a lightweight, minimal representation of a JSON schema is sufficient.
             *
             * @property name The identifier or name of the JSON structure.
             * @property schema The JSON schema associated with the structure.
             */
            @Serializable
            public data class Simple(override val name: String, override val schema: JsonObject) : JSON

            /**
             * Represents a complete JSON schema structure.
             *
             * This data class implements the `JSON` interface and provides a representation
             * for a fully described JSON schema object, including its associated name and schema data.
             *
             * @property name The name identifier for the JSON schema structure.
             * @property schema The JSON schema definition as a `JsonObject`.
             */
            @Serializable
            public data class Full(override val name: String, override val schema: JsonObject) : JSON
        }
    }

    /**
     * Used to switch tool calling behavior of LLM
     */
    @Serializable
    public sealed class ToolChoice {
        /**
         *  LLM will call the tool [name] as a response
         */
        @Serializable
        public data class Named(val name: String) : ToolChoice()

        /**
         * LLM will not call tools at all, and only generate text
         */
        @Serializable
        public object None : ToolChoice()

        /**
         * LLM will automatically decide whether to call tools or to generate text
         */
        @Serializable
        public object Auto : ToolChoice()

        /**
         * LLM will only call tools
         */
        @Serializable
        public object Required : ToolChoice()
    }
    
    /**
     * Specifies the format for model responses.
     * Enables structured output modes like JSON.
     */
    @Serializable
    public sealed interface ResponseFormat {
        /**
         * Standard text response format (default)
         */
        @Serializable
        public object TextFormat : ResponseFormat
        
        /**
         * JSON response format without schema validation.
         * Ensures the model output is valid JSON.
         * Note: You should instruct the model to produce JSON in your prompt.
         */
        @Serializable
        public object JsonFormat : ResponseFormat
        
        /**
         * JSON response format with enforced schema validation.
         * Forces the model to produce JSON matching the specified schema.
         *
         * @property name Name identifier for the schema
         * @property description Optional description of the schema
         * @property schemaDefinition JSON schema definition
         * @property strictMode Whether to enforce strict schema validation
         */
        @Serializable
        public data class JsonSchemaFormat(
            val name: String,
            val description: String? = null,
            val schemaDefinition: JsonObject,
            val strictMode: Boolean = false
        ) : ResponseFormat
    }
    
    /**
     * Options for streaming responses.
     * Provides fine-grained control over streaming behavior.
     */
    @Serializable
    public data class StreamOptions(
        /**
         * Include usage information in the stream response.
         */
        val includeUsage: Boolean? = null
    )
}
