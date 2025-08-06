package ai.koog.prompt.executor.ollama.client

import ai.koog.prompt.executor.clients.unified.*
import kotlinx.serialization.*
import kotlinx.serialization.descriptors.*
import kotlinx.serialization.encoding.*
import kotlinx.serialization.json.*
import kotlinx.serialization.modules.SerializersModule

/**
 * Custom serializer for OllamaMessage
 * Handles the content field and optional images array
 */
@OptIn(ExperimentalSerializationApi::class)
internal object OllamaMessageSerializer : KSerializer<OllamaMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("OllamaMessage") {
        element<String>("role")
        element<String>("content")
        element<List<String>?>("images", isOptional = true)
        element<List<OllamaToolCall>?>("tool_calls", isOptional = true)
    }
    
    override fun serialize(encoder: Encoder, value: OllamaMessage) {
        require(encoder is JsonEncoder)
        val json = buildJsonObject {
            put("role", value.role)
            put("content", value.content)
            value.images?.let { 
                putJsonArray("images") {
                    it.forEach { image -> add(JsonPrimitive(image)) }
                }
            }
            value.toolCalls?.let {
                putJsonArray("tool_calls") {
                    it.forEach { call ->
                        add(encoder.json.encodeToJsonElement(call))
                    }
                }
            }
        }
        encoder.encodeJsonElement(json)
    }
    
    override fun deserialize(decoder: Decoder): OllamaMessage {
        require(decoder is JsonDecoder)
        val element = decoder.decodeJsonElement()
        require(element is JsonObject)
        
        return OllamaMessage(
            role = element["role"]?.jsonPrimitive?.content ?: throw SerializationException("Missing role"),
            content = element["content"]?.jsonPrimitive?.content ?: "",
            images = element["images"]?.jsonArray?.map { it.jsonPrimitive.content },
            toolCalls = element["tool_calls"]?.let {
                decoder.json.decodeFromJsonElement<List<OllamaToolCall>>(it)
            }
        )
    }
}

/**
 * Serializer to convert UnifiedMessage to OllamaMessage
 */
internal class UnifiedToOllamaMessageSerializer : KSerializer<UnifiedMessage> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("UnifiedMessage")
    
    override fun serialize(encoder: Encoder, value: UnifiedMessage) {
        val ollamaMessage = when (value) {
            is UnifiedMessage.System -> OllamaMessage(
                role = "system",
                content = extractTextContent(value.content),
                images = extractImages(value.content)
            )
            is UnifiedMessage.User -> OllamaMessage(
                role = "user",
                content = extractTextContent(value.content),
                images = extractImages(value.content)
            )
            is UnifiedMessage.Assistant -> OllamaMessage(
                role = "assistant",
                content = extractTextContent(value.content),
                toolCalls = value.toolCalls?.map { call ->
                    OllamaToolCall(
                        function = OllamaFunctionCall(
                            name = call.name,
                            arguments = call.arguments
                        )
                    )
                }
            )
            is UnifiedMessage.Tool -> OllamaMessage(
                role = "tool",
                content = extractTextContent(value.content)
                // Note: Ollama doesn't use tool_call_id like OpenAI
            )
        }
        
        OllamaMessageSerializer.serialize(encoder, ollamaMessage)
    }
    
    override fun deserialize(decoder: Decoder): UnifiedMessage {
        val ollamaMessage = OllamaMessageSerializer.deserialize(decoder)
        
        return when (ollamaMessage.role) {
            "system" -> UnifiedMessage.System(
                content = parseContentWithImages(ollamaMessage.content, ollamaMessage.images)
            )
            "user" -> UnifiedMessage.User(
                content = parseContentWithImages(ollamaMessage.content, ollamaMessage.images)
            )
            "assistant" -> UnifiedMessage.Assistant(
                content = if (ollamaMessage.content.isNotEmpty()) 
                    listOf(UnifiedContent.Text(ollamaMessage.content)) else emptyList(),
                toolCalls = ollamaMessage.toolCalls?.mapIndexed { index, call ->
                    UnifiedToolCall(
                        id = "ollama_tool_call_${call.function.name.hashCode()}_$index",
                        name = call.function.name,
                        arguments = call.function.arguments
                    )
                }
            )
            "tool" -> UnifiedMessage.Tool(
                content = listOf(UnifiedContent.Text(ollamaMessage.content)),
                toolId = "", // Ollama doesn't use tool IDs
                toolName = "" // Would need to be tracked separately
            )
            else -> throw SerializationException("Unknown role: ${ollamaMessage.role}")
        }
    }
    
    /**
     * Extract text content from UnifiedContent list
     */
    private fun extractTextContent(content: List<UnifiedContent>): String {
        return content.filterIsInstance<UnifiedContent.Text>()
            .joinToString(" ") { it.text }
    }
    
    /**
     * Extract base64 images from UnifiedContent list
     */
    private fun extractImages(content: List<UnifiedContent>): List<String>? {
        val images = content.filterIsInstance<UnifiedContent.Image>()
            .mapNotNull { imageContent ->
                when (val source = imageContent.source) {
                    is ImageSource.Base64 -> source.data
                    else -> null // Ollama only supports base64 images directly
                }
            }
        return if (images.isNotEmpty()) images else null
    }
    
    /**
     * Parse content with optional images back to UnifiedContent
     */
    private fun parseContentWithImages(content: String, images: List<String>?): List<UnifiedContent> {
        val result = mutableListOf<UnifiedContent>()
        
        if (content.isNotEmpty()) {
            result.add(UnifiedContent.Text(content))
        }
        
        images?.forEach { imageData ->
            result.add(UnifiedContent.Image(ImageSource.Base64(imageData, "image/jpeg")))
        }
        
        return result
    }
}

/**
 * Serializer for UnifiedToolChoice to Ollama format
 * Note: Ollama has limited tool choice support compared to OpenAI
 */
internal object OllamaToolChoiceSerializer : KSerializer<UnifiedToolChoice> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("UnifiedToolChoice")
    
    override fun serialize(encoder: Encoder, value: UnifiedToolChoice) {
        require(encoder is JsonEncoder)
        // Ollama doesn't have explicit tool choice - tools are always available if provided
        // We can ignore this for now or implement basic logic
        val element = when (value) {
            is UnifiedToolChoice.Auto -> JsonPrimitive("auto")
            is UnifiedToolChoice.None -> JsonPrimitive("none") 
            is UnifiedToolChoice.Required -> JsonPrimitive("required")
            is UnifiedToolChoice.Named -> buildJsonObject {
                put("type", "function")
                put("function", buildJsonObject {
                    put("name", value.name)
                })
            }
        }
        encoder.encodeJsonElement(element)
    }
    
    override fun deserialize(decoder: Decoder): UnifiedToolChoice {
        val element = decoder.decodeSerializableValue(JsonElement.serializer())
        
        return when (element) {
            is JsonPrimitive -> when (element.content) {
                "auto" -> UnifiedToolChoice.Auto
                "none" -> UnifiedToolChoice.None
                "required" -> UnifiedToolChoice.Required
                else -> UnifiedToolChoice.Auto // Default fallback
            }
            is JsonObject -> {
                val functionName = element["function"]?.jsonObject?.get("name")?.jsonPrimitive?.content
                    ?: throw SerializationException("Missing function name")
                UnifiedToolChoice.Named(functionName)
            }
            else -> UnifiedToolChoice.Auto // Default fallback
        }
    }
}

/**
 * Custom serializer for response format
 * Ollama supports JSON schema format differently than OpenAI
 */
internal object OllamaResponseFormatSerializer : KSerializer<UnifiedResponseFormat> {
    override val descriptor: SerialDescriptor = buildClassSerialDescriptor("UnifiedResponseFormat")
    
    override fun serialize(encoder: Encoder, value: UnifiedResponseFormat) {
        require(encoder is JsonEncoder)
        val element = when (value) {
            is UnifiedResponseFormat.Text -> {
                // For text format, don't include format parameter
                JsonNull
            }
            is UnifiedResponseFormat.JsonMode -> {
                value.schema ?: buildJsonObject {
                    put("type", "object")
                }
            }
        }
        encoder.encodeJsonElement(element)
    }
    
    override fun deserialize(decoder: Decoder): UnifiedResponseFormat {
        val element = decoder.decodeSerializableValue(JsonElement.serializer())
        
        return when (element) {
            is JsonNull -> UnifiedResponseFormat.Text
            is JsonObject -> UnifiedResponseFormat.JsonMode(schema = element)
            else -> UnifiedResponseFormat.Text // Default fallback
        }
    }
}

/**
 * SerializersModule for Ollama
 */
internal val ollamaSerializersModule: SerializersModule = SerializersModule {
    contextual(UnifiedMessage::class, UnifiedToOllamaMessageSerializer())
    contextual(UnifiedToolChoice::class, OllamaToolChoiceSerializer)
    contextual(UnifiedResponseFormat::class, OllamaResponseFormatSerializer)
}