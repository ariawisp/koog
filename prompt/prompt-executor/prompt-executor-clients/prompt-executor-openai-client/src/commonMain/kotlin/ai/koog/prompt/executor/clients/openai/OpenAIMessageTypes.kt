package ai.koog.prompt.executor.clients.openai

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put

/**
 * OpenAI Content types for multimodal messages
 */
@Serializable
public sealed class OpenAIContent {
    @Serializable
    public data class Text(val text: String) : OpenAIContent()
    
    @Serializable
    public data class ImageUrl(
        val url: String,
        val detail: String? = null
    ) : OpenAIContent()
}

/**
 * OpenAI Tool Function definition
 */
@Serializable
public data class OpenAIToolFunction(
    val name: String,
    val description: String? = null,
    val parameters: OpenAIToolParameters
)

/**
 * OpenAI Tool Parameters
 */
@Serializable
public data class OpenAIToolParameters(
    val type: String = "object",
    val properties: Map<String, Map<String, String>> = emptyMap(),
    val required: List<String> = emptyList()
) {
    public fun toJsonObject(): JsonObject = buildJsonObject {
        put("type", type)
        put("properties", buildJsonObject {
            properties.forEach { (key, value) ->
                put(key, buildJsonObject {
                    value.forEach { (propKey, propValue) ->
                        put(propKey, propValue)
                    }
                })
            }
        })
        put("required", buildJsonArray {
            required.forEach { add(JsonPrimitive(it)) }
        })
    }
}

/**
 * Factory functions to create OpenAI messages conveniently
 */
public object OpenAIMessageFactory {
    public fun System(content: String): ai.koog.prompt.executor.clients.openai.OpenAIMessage =
        ai.koog.prompt.executor.clients.openai.OpenAIMessage(
            role = "system",
            content = JsonPrimitive(content)
        )
    
    public fun User(content: String): ai.koog.prompt.executor.clients.openai.OpenAIMessage =
        ai.koog.prompt.executor.clients.openai.OpenAIMessage(
            role = "user",
            content = JsonPrimitive(content)
        )
    
    public fun User(content: List<OpenAIContent>): ai.koog.prompt.executor.clients.openai.OpenAIMessage =
        ai.koog.prompt.executor.clients.openai.OpenAIMessage(
            role = "user",
            content = buildJsonArray {
                content.forEach { contentPart ->
                    when (contentPart) {
                        is OpenAIContent.Text -> add(buildJsonObject {
                            put("type", "text")
                            put("text", contentPart.text)
                        })
                        is OpenAIContent.ImageUrl -> add(buildJsonObject {
                            put("type", "image_url")
                            put("image_url", buildJsonObject {
                                put("url", contentPart.url)
                                contentPart.detail?.let { put("detail", it) }
                            })
                        })
                    }
                }
            }
        )
    
    public fun Assistant(
        content: String?,
        toolCalls: List<OpenAIToolCall>? = null
    ): ai.koog.prompt.executor.clients.openai.OpenAIMessage =
        ai.koog.prompt.executor.clients.openai.OpenAIMessage(
            role = "assistant",
            content = content?.let { JsonPrimitive(it) },
            toolCalls = toolCalls
        )
    
    public fun Tool(
        toolCallId: String,
        content: String
    ): ai.koog.prompt.executor.clients.openai.OpenAIMessage =
        ai.koog.prompt.executor.clients.openai.OpenAIMessage(
            role = "tool",
            content = JsonPrimitive(content),
            toolCallId = toolCallId
        )
}