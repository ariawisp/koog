package ai.koog.prompt.executor.clients.openrouter

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure
import kotlinx.serialization.json.JsonElement

/**
 * Custom serializer for OpenRouterMessage to handle content as JsonElement.
 */
internal object OpenRouterMessageSerializer : KSerializer<OpenRouterMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OpenRouterMessage") {
        element<String>("role")
        element<JsonElement?>("content", isOptional = true)
        element<List<OpenRouterToolCall>?>("tool_calls", isOptional = true)
        element<String?>("tool_call_id", isOptional = true)
        element<String?>("name", isOptional = true)
    }
    
    override fun serialize(encoder: Encoder, value: OpenRouterMessage) {
        encoder.encodeStructure(descriptor) {
            encodeStringElement(descriptor, 0, value.role)
            value.content?.let { encodeSerializableElement(descriptor, 1, kotlinx.serialization.json.JsonElement.serializer(), it) }
            value.toolCalls?.let { encodeSerializableElement(descriptor, 2, kotlinx.serialization.builtins.ListSerializer(OpenRouterToolCall.serializer()), it) }
            value.toolCallId?.let { encodeStringElement(descriptor, 3, it) }
            value.name?.let { encodeStringElement(descriptor, 4, it) }
        }
    }
    
    override fun deserialize(decoder: Decoder): OpenRouterMessage {
        var role = ""
        var content: JsonElement? = null
        var toolCalls: List<OpenRouterToolCall>? = null
        var toolCallId: String? = null
        var name: String? = null
        
        decoder.decodeStructure(descriptor) {
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> role = decodeStringElement(descriptor, 0)
                    1 -> content = decodeSerializableElement(descriptor, 1, kotlinx.serialization.json.JsonElement.serializer())
                    2 -> toolCalls = decodeSerializableElement(descriptor, 2, kotlinx.serialization.builtins.ListSerializer(OpenRouterToolCall.serializer()))
                    3 -> toolCallId = decodeStringElement(descriptor, 3)
                    4 -> name = decodeStringElement(descriptor, 4)
                    kotlinx.serialization.encoding.CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
        }
        
        return OpenRouterMessage(role, content, toolCalls, toolCallId, name)
    }
}