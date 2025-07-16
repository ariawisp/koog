package ai.koog.prompt.executor.clients.google.structure

import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.structure.json.generator.JsonSchemaConsts
import ai.koog.prompt.structure.json.generator.SimpleJsonSchemaGenerator
import kotlinx.serialization.json.JsonObject


/**
 * Extends [SimpleJsonSchemaGenerator] to generate [LLMParams.Schema.JSON.Simple] in custom Google format.
 */
public open class GoogleSimpleJsonSchemaGenerator : SimpleJsonSchemaGenerator() {

    /**
     * Default instance of [GoogleSimpleJsonSchemaGenerator].
     * Prefer to use it instead of creating new instates manually.
     *
     * @see [GoogleSimpleJsonSchemaGenerator]
     */
    public companion object : GoogleSimpleJsonSchemaGenerator()

    override fun processObject(context: GenerationContext): JsonObject {
        val schema = super.processObject(context).toMutableMap()

        // Google does not support "additionalProperties" in simple schema.
        schema.remove(JsonSchemaConsts.Keys.ADDITIONAL_PROPERTIES)

        return JsonObject(schema)
    }
}
