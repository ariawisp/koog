package ai.koog.prompt.executor.clients.google

import kotlinx.serialization.json.Json

/**
 * JSON configuration for Google Gemini API
 * Uses proper kotlinx.serialization configuration instead of manual JSON building
 */
internal val googleJson = Json {
    // Use the custom serializers module
    serializersModule = googleSerializersModule
    
    // Google uses camelCase for property names (with some snake_case exceptions via @SerialName)
    // We handle the naming via @SerialName annotations in the data classes
    
    // Don't include null values
    encodeDefaults = false
    explicitNulls = false
    
    // Pretty print for debugging (disable in production)
    prettyPrint = false
    
    // Allow structural classes (for our sealed classes)
    allowStructuredMapKeys = true
    
    // Be lenient when reading responses from Google
    ignoreUnknownKeys = true
    isLenient = true
    
    // Allow special floating-point values
    allowSpecialFloatingPointValues = true
    
    // Use array polymorphism for sealed classes
    useArrayPolymorphism = false
    
    // Don't add class discriminator for sealed classes when we have custom serializers
    classDiscriminator = "type"
}