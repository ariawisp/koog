package ai.koog.prompt.structure.json.generator.openai

import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.structure.json.generator.core.JsonSchemaConsts
import ai.koog.prompt.structure.json.generator.core.SimpleJsonSchemaGenerator
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonObject

/**
 * Extends [SimpleJsonSchemaGenerator] to generate [LLMParams.Schema.JSON.Simple] in custom OpenAI format.
 */
public open class OpenAISimpleJsonSchemaGenerator : SimpleJsonSchemaGenerator() {
    /**
     * Default instance of [OpenAISimpleJsonSchemaGenerator].
     * Prefer to use it instead of creating new instates manually.
     *
     * @see [OpenAISimpleJsonSchemaGenerator]
     */
    public companion object Default : OpenAISimpleJsonSchemaGenerator()

    override fun processMap(context: GenerationContext): JsonObject {
        throw UnsupportedOperationException("OpenAI JSON schema doesn't support maps")
    }

    override fun processObject(context: GenerationContext): JsonObject {
        val schema = super.processObject(context).toMutableMap()

        // OpenAI requires all existing properties to be present in "required" list
        schema[JsonSchemaConsts.Keys.REQUIRED] = JsonArray(
            schema.getValue(JsonSchemaConsts.Keys.PROPERTIES).jsonObject
                .keys
                .map { JsonPrimitive(it) }
        )

        context.processedTypeDefs[context.descriptor] = JsonObject(schema)

        return JsonObject(schema)
    }
}
