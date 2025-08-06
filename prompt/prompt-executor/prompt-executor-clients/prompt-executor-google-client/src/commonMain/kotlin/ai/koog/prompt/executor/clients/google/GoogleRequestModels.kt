package ai.koog.prompt.executor.clients.google

import kotlinx.serialization.Contextual
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject

/**
 * Google Gemini Generate Content Request
 * Uses proper kotlinx.serialization with snake_case naming
 */
@Serializable
public data class GoogleGenerateRequest(
    val contents: List<GoogleContent>,
    @SerialName("systemInstruction") val systemInstruction: GoogleSystemInstruction? = null,
    val tools: List<GoogleTool>? = null,
    @SerialName("toolConfig") val toolConfig: GoogleToolConfig? = null,
    @SerialName("generationConfig") val generationConfig: GoogleGenerationConfig? = null
)

/**
 * Google System Instruction
 */
@Serializable
public data class GoogleSystemInstruction(
    val parts: List<GooglePart>
)

/**
 * Google Content - represents a message in the conversation
 */
@Serializable
public data class GoogleContent(
    val role: String,
    val parts: List<GooglePart>
)

/**
 * Google Part - represents different types of content within a message
 */
@Serializable(with = GooglePartSerializer::class)
public sealed class GooglePart {
    @Serializable
    public data class Text(val text: String) : GooglePart()
    
    @Serializable
    public data class InlineData(@SerialName("inlineData") val inlineData: GoogleInlineData) : GooglePart()
    
    @Serializable
    public data class FunctionCall(@SerialName("functionCall") val functionCall: GoogleFunctionCall) : GooglePart()
    
    @Serializable
    public data class FunctionResponse(@SerialName("functionResponse") val functionResponse: GoogleFunctionResponse) : GooglePart()
}

/**
 * Google Inline Data - for media attachments
 */
@Serializable
public data class GoogleInlineData(
    @SerialName("mimeType") val mimeType: String,
    val data: String
)

/**
 * Google Function Call
 */
@Serializable
public data class GoogleFunctionCall(
    val name: String,
    val args: @Contextual JsonObject
)

/**
 * Google Function Response
 */
@Serializable
public data class GoogleFunctionResponse(
    val name: String,
    val response: GoogleFunctionResponseContent
)

/**
 * Google Function Response Content
 */
@Serializable
public data class GoogleFunctionResponseContent(
    val content: String
)

/**
 * Google Tool Configuration
 */
@Serializable
public data class GoogleToolConfig(
    @SerialName("functionCallingConfig") val functionCallingConfig: GoogleFunctionCallingConfig
)

/**
 * Google Function Calling Configuration
 */
@Serializable
public data class GoogleFunctionCallingConfig(
    val mode: String,
    @SerialName("allowedFunctionNames") val allowedFunctionNames: List<String>? = null
)

/**
 * Google Tool Definition
 */
@Serializable
public data class GoogleTool(
    @SerialName("functionDeclarations") val functionDeclarations: List<GoogleFunctionDeclaration>
)

/**
 * Google Function Declaration
 */
@Serializable
public data class GoogleFunctionDeclaration(
    val name: String,
    val description: String,
    val parameters: GoogleFunctionParameters
)

/**
 * Google Function Parameters
 */
@Serializable
public data class GoogleFunctionParameters(
    val type: String = "object",
    val properties: @Contextual JsonObject,
    val required: List<String>
)

/**
 * Google Generation Configuration
 */
@Serializable
public data class GoogleGenerationConfig(
    val temperature: Double? = null,
    @SerialName("maxOutputTokens") val maxOutputTokens: Int? = null,
    @SerialName("topP") val topP: Double? = null,
    @SerialName("topK") val topK: Int? = null,
    @SerialName("candidateCount") val candidateCount: Int? = null,
    @SerialName("stopSequences") val stopSequences: List<String>? = null,
    @SerialName("responseMimeType") val responseMimeType: String? = null,
    @SerialName("responseSchema") val responseSchema: @Contextual JsonElement? = null
)

/**
 * Google Gemini Generate Content Response
 */
@Serializable
internal data class GoogleGenerateResponse(
    val candidates: List<GoogleCandidate>,
    @SerialName("usageMetadata") val usageMetadata: GoogleUsageMetadata? = null,
    @SerialName("modelVersion") val modelVersion: String? = null
)

/**
 * Google Candidate in Response
 */
@Serializable
internal data class GoogleCandidate(
    val content: GoogleContent? = null,
    @SerialName("finishReason") val finishReason: String? = null,
    val index: Int? = null
)

/**
 * Google Usage Metadata
 */
@Serializable
internal data class GoogleUsageMetadata(
    @SerialName("promptTokenCount") val promptTokenCount: Int? = null,
    @SerialName("candidatesTokenCount") val candidatesTokenCount: Int? = null,
    @SerialName("totalTokenCount") val totalTokenCount: Int? = null
)

/**
 * Legacy compatibility types (for the existing GoogleData references)
 */
internal object GoogleData {
    @Serializable
    data class Blob(
        val mimeType: String,
        val data: String
    )
}