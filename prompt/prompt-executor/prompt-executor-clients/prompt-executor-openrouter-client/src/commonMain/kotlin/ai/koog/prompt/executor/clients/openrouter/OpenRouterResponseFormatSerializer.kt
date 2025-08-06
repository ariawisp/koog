package ai.koog.prompt.executor.clients.openrouter

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.json.*

/**
 * Custom serializer for OpenRouterResponseFormat sealed class.
 */
internal object OpenRouterResponseFormatSerializer : KSerializer<OpenRouterResponseFormat> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OpenRouterResponseFormat")
    
    override fun serialize(encoder: Encoder, value: OpenRouterResponseFormat) {
        val jsonEncoder = encoder as? JsonEncoder
            ?: error("OpenRouterResponseFormat can only be serialized as JSON")
        
        val jsonElement = when (value) {
            is OpenRouterResponseFormat.Text -> buildJsonObject {
                put("type", JsonPrimitive("text"))
            }
            is OpenRouterResponseFormat.JsonObject -> buildJsonObject {
                put("type", JsonPrimitive("json_object"))
            }
            is OpenRouterResponseFormat.JsonSchema -> buildJsonObject {
                put("type", JsonPrimitive("json_schema"))
                put("json_schema", buildJsonObject {
                    put("name", JsonPrimitive(value.jsonSchema.name))
                    value.jsonSchema.description?.let { put("description", JsonPrimitive(it)) }
                    put("schema", value.jsonSchema.schema)
                    value.jsonSchema.strict?.let { put("strict", JsonPrimitive(it)) }
                })
            }
        }
        
        jsonEncoder.encodeJsonElement(jsonElement)
    }
    
    override fun deserialize(decoder: Decoder): OpenRouterResponseFormat {
        val jsonDecoder = decoder as? JsonDecoder
            ?: error("OpenRouterResponseFormat can only be deserialized from JSON")
        
        val jsonElement = jsonDecoder.decodeJsonElement()
        val jsonObject = jsonElement.jsonObject
        
        return when (val type = jsonObject["type"]?.jsonPrimitive?.content) {
            "text" -> OpenRouterResponseFormat.Text()
            "json_object" -> OpenRouterResponseFormat.JsonObject()
            "json_schema" -> {
                val schemaObj = jsonObject["json_schema"]?.jsonObject
                    ?: error("json_schema field is required for json_schema type")
                OpenRouterResponseFormat.JsonSchema(
                    jsonSchema = OpenRouterJsonSchema(
                        name = schemaObj["name"]?.jsonPrimitive?.content
                            ?: error("name is required in json_schema"),
                        description = schemaObj["description"]?.jsonPrimitive?.content,
                        schema = schemaObj["schema"]?.jsonObject
                            ?: error("schema is required in json_schema"),
                        strict = schemaObj["strict"]?.jsonPrimitive?.boolean
                    )
                )
            }
            else -> error("Unknown response format type: $type")
        }
    }
}