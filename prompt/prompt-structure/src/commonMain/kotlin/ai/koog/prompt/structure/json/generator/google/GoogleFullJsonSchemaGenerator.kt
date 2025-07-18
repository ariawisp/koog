package ai.koog.prompt.structure.json.generator.google

import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.structure.json.generator.core.FullJsonSchemaGenerator
import ai.koog.prompt.structure.json.generator.core.JsonSchemaConsts
import kotlinx.serialization.json.*

/**
 * Extends [FullJsonSchemaGenerator] to generate [LLMParams.Schema.JSON.Full] in custom Google format.
 */
public open class GoogleFullJsonSchemaGenerator : FullJsonSchemaGenerator() {
    /**
     * Default instance of [GoogleFullJsonSchemaGenerator].
     * Prefer to use it instead of creating new instates manually.
     *
     * @see [GoogleFullJsonSchemaGenerator]
     */
    public companion object : GoogleFullJsonSchemaGenerator()

    override fun processObject(context: GenerationContext): JsonObject {
        val refObject = super.processObject(context).toMutableMap()

        /**
         * Google doesn't allow other fields at the same level with "$ref", such as "description"
         * Wrap it to "oneOf" with a single element to work around this limitation.
         */
        if (JsonSchemaConsts.Keys.REF in refObject && refObject.size > 1) {
            refObject[JsonSchemaConsts.Keys.ONE_OF] = buildJsonArray {
                add(buildJsonObject {
                    put(JsonSchemaConsts.Keys.REF, refObject.getValue(JsonSchemaConsts.Keys.REF))
                })
            }

            refObject.remove(JsonSchemaConsts.Keys.REF)
        }

        return JsonObject(refObject)
    }

    override fun processClassDiscriminator(context: GenerationContext): JsonObject = buildJsonObject {
        // Google format does not support "const" property, but we can hack around with "enum" with a single value
        put(JsonSchemaConsts.Keys.TYPE, JsonSchemaConsts.Types.STRING)
        put(JsonSchemaConsts.Keys.ENUM, buildJsonArray { add(context.descriptor.serialName) })
    }
}