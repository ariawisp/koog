package ai.koog.prompt.structure.json.generator.openai

import ai.koog.prompt.params.LLMParams
import ai.koog.prompt.structure.json.generator.core.FullJsonSchemaGenerator
import ai.koog.prompt.structure.json.generator.core.JsonSchemaConsts
import ai.koog.prompt.structure.json.generator.core.countElements
import kotlinx.serialization.KSerializer
import kotlinx.serialization.json.*

/**
 * Extends [FullJsonSchemaGenerator] to generate [LLMParams.Schema.JSON.Full] in custom OpenAI format.
 */
public open class OpenAIFullJsonSchemaGenerator : FullJsonSchemaGenerator() {
    /**
     * Default instance of [OpenAIFullJsonSchemaGenerator].
     * Prefer to use it instead of creating new instates manually.
     *
     * @see [OpenAIFullJsonSchemaGenerator]
     */
    public companion object : OpenAIFullJsonSchemaGenerator()

    /**
     * Generates [LLMParams.Schema.JSON.Full] in custom OpenAI format.
     */
    override fun generate(
        json: Json,
        name: String,
        serializer: KSerializer<*>,
        descriptionOverrides: Map<String, String>
    ): LLMParams.Schema.JSON.Full {
        val param = super.generate(json, name, serializer, descriptionOverrides)
        val schema = param.schema.toMutableMap()

        // OpenAI doesn't accept "$ref" at the root of the schema, so copying this definition explicitly.
        val rootRef = schema.getValue(JsonSchemaConsts.Keys.REF).jsonPrimitive
        val rootDefKey = rootRef.content.removePrefix(JsonSchemaConsts.Keys.REF_PREFIX)
        val rootType = schema.getValue(JsonSchemaConsts.Keys.DEFS).jsonObject
            .getValue(rootDefKey).jsonObject

        schema.remove(JsonSchemaConsts.Keys.REF)

        rootType.forEach { (key, value) ->
            schema[key] = value
        }

        // If root type was not referenced anywhere else, remove it from definitions, such definition is unused
        if (rootType.countElements(rootRef) > 0) {
            val updatedDefs = schema.getValue(JsonSchemaConsts.Keys.DEFS).jsonObject.toMutableMap()
            updatedDefs.remove(rootDefKey)

            // Remove definitions property if it's empty now
            if (updatedDefs.isEmpty()) {
                schema.remove(JsonSchemaConsts.Keys.DEFS)
            } else {
                schema[JsonSchemaConsts.Keys.DEFS] = JsonObject(updatedDefs)
            }
        }

        return param.copy(schema = JsonObject(schema))
    }

    override fun processMap(context: GenerationContext): JsonObject {
        throw UnsupportedOperationException("OpenAI JSON schema doesn't support maps")
    }

    override fun processObject(context: GenerationContext): JsonObject {
        val refObject = super.processObject(context).toMutableMap()

        if (context.descriptor !in context.currentDefPath) {
            val schema = context.processedTypeDefs.getValue(context.descriptor).toMutableMap()

            // OpenAI requires all existing properties to be present in "required" list
            schema[JsonSchemaConsts.Keys.REQUIRED] = JsonArray(
                schema.getValue(JsonSchemaConsts.Keys.PROPERTIES).jsonObject
                    .keys
                    .map { JsonPrimitive(it) }
            )

            context.processedTypeDefs[context.descriptor] = JsonObject(schema)
        }

        /**
         * OpenAI doesn't allow other fields at the same level with "$ref", such as "description"
         * Wrap it to "anyOf" with a single element to work around this limitation.
         */
        if (JsonSchemaConsts.Keys.REF in refObject && refObject.size > 1) {
            refObject[JsonSchemaConsts.Keys.ANY_OF] = buildJsonArray {
                add(buildJsonObject {
                    put(JsonSchemaConsts.Keys.REF, refObject.getValue(JsonSchemaConsts.Keys.REF))
                })
            }

            refObject.remove(JsonSchemaConsts.Keys.REF)
        }

        if (JsonSchemaConsts.Keys.ONE_OF in refObject) {
            refObject.replaceOneOf()
        }

        return JsonObject(refObject)
    }

    override fun processClassDiscriminator(context: GenerationContext): JsonObject = buildJsonObject {
        // OpenAI format does not support "const" property, but we can hack around with "enum" with a single value
        put(JsonSchemaConsts.Keys.TYPE, JsonSchemaConsts.Types.STRING)
        put(JsonSchemaConsts.Keys.ENUM, buildJsonArray { add(context.descriptor.serialName) })
    }

    override fun processPolymorphic(context: GenerationContext): JsonObject {
        val schema = super.processPolymorphic(context).toMutableMap()

        schema.replaceOneOf()

        return JsonObject(schema)
    }

    /*
     OpenAI supports only "anyOf" instead of "oneOf".
     These are not the same, but for our purposes this should work.
    */
    private fun MutableMap<String, JsonElement>.replaceOneOf() {
        this[JsonSchemaConsts.Keys.ANY_OF] = getValue(JsonSchemaConsts.Keys.ONE_OF)
        remove(JsonSchemaConsts.Keys.ONE_OF)
    }
}