package ai.koog.agents.core.agent.config

import ai.koog.prompt.harmony.*
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.datetime.Clock

/**
 * Describes the way to reformat tool call/tool result messages,
 * in case real tool call/tool result messages cannot be used
 */
public interface ToolCallDescriber {
    /**
     * Composes a description of a tool call message.
     *
     * @param message The tool call message to be described. Must be a tool call in commentary channel.
     * @return A HarmonyMessage instance containing the description of the tool call.
     */
    public fun describeToolCall(message: HarmonyMessage): HarmonyMessage

    /**
     * Describes the tool result by transforming it into a user-readable message object.
     *
     * @param message The tool result message to be described. It contains the tool response.
     * @return A transformed message representing the description of the tool result.
     */
    public fun describeToolResult(message: HarmonyMessage): HarmonyMessage

    /**
     * JSON object implementing the `ToolCallDescriber` interface.
     * This object is responsible for describing tool calls and results by converting them into a structured JSON-based format.
     */
    public object JSON : ToolCallDescriber {
        /**
         * A configuration of the kotlinx.serialization.Json instance tailored for serializing and
         * deserializing JSON data.
         *
         * This specific instance has the following options configured:
         * - `encodeDefaults` set to `true`: Ensures that default values are encoded during serialization.
         * - `explicitNulls` set to `false`: Avoids including `null` values explicitly in the resulting JSON output.
         *
         * It is used internally for encoding and decoding JSON representations of tool-related data.
         */
        private val Json = Json {
            encodeDefaults = true
            explicitNulls = false
        }

        /**
         * Formats a tool call message into a standardized assistant response.
         *
         * @param message the tool call message containing details about the tool invocation.
         * @return a HarmonyMessage containing the serialized JSON representation of the tool call information.
         */
        override fun describeToolCall(message: HarmonyMessage): HarmonyMessage {
            // Extract tool name from recipient (e.g., "functions.get_weather" -> "get_weather")
            val toolName = message.recipient?.substringAfter("functions.") ?: "unknown"
            val toolArgs = try {
                Json.parseToJsonElement(message.getTextContent()).jsonObject
            } catch (e: Exception) {
                buildJsonObject { put("content", JsonPrimitive(message.getTextContent())) }
            }
            
            return HarmonyMessage(
                author = HarmonyAuthor.from(Role.ASSISTANT),
                content = listOf(HarmonyContent.Text(
                    Json.encodeToString(
                        buildJsonObject {
                            put("tool_name", JsonPrimitive(toolName))
                            put("tool_args", toolArgs)
                        }
                    )
                )),
                channel = "final",
                timestamp = message.timestamp
            )
        }

        /**
         * Creates a user message containing a structured JSON representation
         * of a tool result.
         *
         * @param message The tool result message containing the tool response.
         * @return A user message with a JSON-encoded representation of the tool result.
         */
        override fun describeToolResult(message: HarmonyMessage): HarmonyMessage {
            // Tool results have author.name set to the tool name
            val toolName = message.author.name ?: "unknown"
            
            return HarmonyMessage(
                author = HarmonyAuthor.from(Role.USER),
                content = listOf(HarmonyContent.Text(
                    Json.encodeToString(
                        buildJsonObject {
                            put("tool_name", JsonPrimitive(toolName))
                            put("tool_result", JsonPrimitive(message.getTextContent()))
                        }
                    )
                )),
                channel = "final",
                timestamp = message.timestamp
            )
        }
    }
}
