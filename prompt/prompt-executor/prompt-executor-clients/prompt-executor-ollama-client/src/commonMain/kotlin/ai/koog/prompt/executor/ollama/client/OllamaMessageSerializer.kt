package ai.koog.prompt.executor.ollama.client

import kotlinx.serialization.KSerializer
import kotlinx.serialization.descriptors.SerialDescriptor
import kotlinx.serialization.descriptors.buildClassSerialDescriptor
import kotlinx.serialization.descriptors.element
import kotlinx.serialization.encoding.Decoder
import kotlinx.serialization.encoding.Encoder
import kotlinx.serialization.encoding.decodeStructure
import kotlinx.serialization.encoding.encodeStructure
import kotlinx.serialization.json.JsonObject

/**
 * Custom serializer for OllamaMessage to handle optional fields properly.
 */
internal object OllamaMessageSerializer : KSerializer<OllamaMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OllamaMessage") {
        element<String>("role")
        element<String>("content")
        element<List<String>?>("images", isOptional = true)
        element<List<OllamaToolCall>?>("tool_calls", isOptional = true)
    }
    
    override fun serialize(encoder: Encoder, value: OllamaMessage) {
        encoder.encodeStructure(descriptor) {
            encodeStringElement(descriptor, 0, value.role)
            encodeStringElement(descriptor, 1, value.content)
            value.images?.let { encodeSerializableElement(descriptor, 2, kotlinx.serialization.builtins.ListSerializer(kotlinx.serialization.builtins.String.serializer()), it) }
            value.toolCalls?.let { encodeSerializableElement(descriptor, 3, kotlinx.serialization.builtins.ListSerializer(OllamaToolCall.serializer()), it) }
        }
    }
    
    override fun deserialize(decoder: Decoder): OllamaMessage {
        var role = ""
        var content = ""
        var images: List<String>? = null
        var toolCalls: List<OllamaToolCall>? = null
        
        decoder.decodeStructure(descriptor) {
            while (true) {
                when (val index = decodeElementIndex(descriptor)) {
                    0 -> role = decodeStringElement(descriptor, 0)
                    1 -> content = decodeStringElement(descriptor, 1)
                    2 -> images = decodeSerializableElement(descriptor, 2, kotlinx.serialization.builtins.ListSerializer(kotlinx.serialization.builtins.String.serializer()))
                    3 -> toolCalls = decodeSerializableElement(descriptor, 3, kotlinx.serialization.builtins.ListSerializer(OllamaToolCall.serializer()))
                    kotlinx.serialization.encoding.CompositeDecoder.DECODE_DONE -> break
                    else -> error("Unexpected index: $index")
                }
            }
        }
        
        return OllamaMessage(role, content, images, toolCalls)
    }
}