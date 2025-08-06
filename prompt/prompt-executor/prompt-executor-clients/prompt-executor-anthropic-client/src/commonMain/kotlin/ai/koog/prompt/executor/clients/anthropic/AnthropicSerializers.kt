package ai.koog.prompt.executor.clients.anthropic

import kotlinx.serialization.*
import kotlinx.serialization.descriptors.*
import kotlinx.serialization.encoding.*
import kotlinx.serialization.json.*

/**
 * Custom serializer for Anthropic messages
 */
@OptIn(ExperimentalSerializationApi::class)
internal object AnthropicMessageSerializer : KSerializer<AnthropicMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("AnthropicMessage") {
        element<String>("role")
        element<JsonElement>("content")
    }
    
    override fun serialize(encoder: Encoder, value: AnthropicMessage) {
        require(encoder is JsonEncoder)
        val json = buildJsonObject {
            put("role", value.role)
            put("content", value.content)
        }
        encoder.encodeJsonElement(json)
    }
    
    override fun deserialize(decoder: Decoder): AnthropicMessage {
        require(decoder is JsonDecoder)
        val element = decoder.decodeJsonElement()
        require(element is JsonObject)
        
        return AnthropicMessage(
            role = element["role"]?.jsonPrimitive?.content ?: throw SerializationException("Missing role"),
            content = element["content"] ?: throw SerializationException("Missing content")
        )
    }
}

/**
 * Custom serializer for Anthropic tool choice
 */
@OptIn(ExperimentalSerializationApi::class)
internal object AnthropicToolChoiceSerializer : KSerializer<AnthropicToolChoice> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("AnthropicToolChoice")
    
    override fun serialize(encoder: Encoder, value: AnthropicToolChoice) {
        require(encoder is JsonEncoder)
        val json = when (value) {
            is AnthropicToolChoice.Auto -> buildJsonObject {
                put("type", "auto")
            }
            is AnthropicToolChoice.Any -> buildJsonObject {
                put("type", "any")
            }
            is AnthropicToolChoice.Tool -> buildJsonObject {
                put("type", "tool")
                put("name", value.name)
            }
        }
        encoder.encodeJsonElement(json)
    }
    
    override fun deserialize(decoder: Decoder): AnthropicToolChoice {
        require(decoder is JsonDecoder)
        val element = decoder.decodeJsonElement()
        require(element is JsonObject)
        
        return when (element["type"]?.jsonPrimitive?.content) {
            "auto" -> AnthropicToolChoice.Auto()
            "any" -> AnthropicToolChoice.Any()
            "tool" -> {
                val name = element["name"]?.jsonPrimitive?.content 
                    ?: throw SerializationException("Missing tool name")
                AnthropicToolChoice.Tool(name = name)
            }
            else -> throw SerializationException("Unknown tool choice type")
        }
    }
}