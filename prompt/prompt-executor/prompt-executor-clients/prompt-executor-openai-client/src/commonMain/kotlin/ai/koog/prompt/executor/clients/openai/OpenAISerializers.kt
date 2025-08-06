package ai.koog.prompt.executor.clients.openai

import kotlinx.serialization.*
import kotlinx.serialization.descriptors.*
import kotlinx.serialization.encoding.*
import kotlinx.serialization.json.*
import kotlinx.serialization.modules.SerializersModule

/**
 * Custom serializer for OpenAI messages
 * Handles the complex content field that can be string or array
 */
@OptIn(ExperimentalSerializationApi::class)
internal object OpenAIMessageSerializer : KSerializer<OpenAIMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OpenAIMessage") {
        element<String>("role")
        element<JsonElement?>("content", isOptional = true)
        element<List<OpenAIToolCall>?>("tool_calls", isOptional = true)
        element<String?>("tool_call_id", isOptional = true)
        element<String?>("name", isOptional = true)
    }
    
    override fun serialize(encoder: Encoder, value: OpenAIMessage) {
        require(encoder is JsonEncoder)
        val json = buildJsonObject {
            put("role", value.role)
            value.content?.let { put("content", it) }
            value.toolCalls?.let { 
                putJsonArray("tool_calls") {
                    it.forEach { call ->
                        add(encoder.json.encodeToJsonElement(call))
                    }
                }
            }
            value.toolCallId?.let { put("tool_call_id", it) }
            value.name?.let { put("name", it) }
        }
        encoder.encodeJsonElement(json)
    }
    
    override fun deserialize(decoder: Decoder): OpenAIMessage {
        require(decoder is JsonDecoder)
        val element = decoder.decodeJsonElement()
        require(element is JsonObject)
        
        return OpenAIMessage(
            role = element["role"]?.jsonPrimitive?.content ?: throw SerializationException("Missing role"),
            content = element["content"],
            toolCalls = element["tool_calls"]?.let {
                if (it is JsonArray) {
                    it.map { toolElement ->
                        decoder.json.decodeFromJsonElement<OpenAIToolCall>(toolElement)
                    }
                } else null
            },
            toolCallId = element["tool_call_id"]?.jsonPrimitive?.content,
            name = element["name"]?.jsonPrimitive?.content
        )
    }
}

/**
 * Custom serializer for OpenAI response format
 */
@OptIn(ExperimentalSerializationApi::class)
internal object OpenAIResponseFormatSerializer : KSerializer<OpenAIResponseFormat> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OpenAIResponseFormat") {
        element<String>("type")
        element<OpenAIJsonSchema?>("json_schema", isOptional = true)
    }
    
    override fun serialize(encoder: Encoder, value: OpenAIResponseFormat) {
        require(encoder is JsonEncoder)
        val json = when (value) {
            is OpenAIResponseFormat.Text -> buildJsonObject {
                put("type", "text")
            }
            is OpenAIResponseFormat.JsonObject -> buildJsonObject {
                put("type", "json_object")
            }
            is OpenAIResponseFormat.JsonSchema -> buildJsonObject {
                put("type", "json_schema")
                put("json_schema", encoder.json.encodeToJsonElement(value.jsonSchema))
            }
        }
        encoder.encodeJsonElement(json)
    }
    
    override fun deserialize(decoder: Decoder): OpenAIResponseFormat {
        require(decoder is JsonDecoder)
        val element = decoder.decodeJsonElement()
        require(element is JsonObject)
        
        return when (element["type"]?.jsonPrimitive?.content) {
            "text" -> OpenAIResponseFormat.Text()
            "json_object" -> OpenAIResponseFormat.JsonObject()
            "json_schema" -> OpenAIResponseFormat.JsonSchema(
                jsonSchema = element["json_schema"]?.let {
                    decoder.json.decodeFromJsonElement<OpenAIJsonSchema>(it)
                } ?: throw SerializationException("Missing json_schema")
            )
            else -> throw SerializationException("Unknown response format type")
        }
    }
}

/**
 * SerializersModule for OpenAI
 */
internal val openAISerializersModule: SerializersModule = SerializersModule {
    // Empty for now - can add contextual serializers as needed
}