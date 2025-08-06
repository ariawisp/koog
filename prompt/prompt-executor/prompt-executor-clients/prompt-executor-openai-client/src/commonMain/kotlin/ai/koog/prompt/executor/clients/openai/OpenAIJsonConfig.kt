package ai.koog.prompt.executor.clients.openai

import kotlinx.serialization.json.Json

/**
 * JSON configuration for OpenAI API
 * Uses proper kotlinx.serialization configuration instead of manual JSON building
 */
internal val openAIJson = Json {
    // Use the custom serializers module
    serializersModule = openAISerializersModule
    
    // OpenAI uses snake_case for property names
    // Note: JsonNamingStrategy is only available in kotlinx.serialization 1.6.0+
    // For now we use @SerialName annotations
    
    // Don't include null values
    encodeDefaults = false
    explicitNulls = false
    
    // Pretty print for debugging (disable in production)
    prettyPrint = false
    
    // Allow structural classes (for our sealed classes)
    allowStructuredMapKeys = true
    
    // Be lenient when reading responses
    ignoreUnknownKeys = true
    isLenient = true
    
    // Allow special floating-point values
    allowSpecialFloatingPointValues = true
}