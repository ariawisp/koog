package ai.koog.prompt.executor.ollama.client

import kotlinx.serialization.json.Json

/**
 * JSON configuration for Ollama API
 * Uses proper kotlinx.serialization configuration instead of manual JSON building
 */
internal val ollamaJson = Json {
    // Use the custom serializers module
    serializersModule = ollamaSerializersModule
    
    // Ollama uses snake_case for property names
    // Note: JsonNamingStrategy is only available in kotlinx.serialization 1.6.0+
    // For now we use @SerialName annotations
    
    // Don't include null values - this helps keep requests clean
    encodeDefaults = false
    explicitNulls = false
    
    // Pretty print for debugging (disable in production)
    prettyPrint = false
    
    // Allow structural classes (for our sealed classes)
    allowStructuredMapKeys = true
    
    // Be lenient when reading responses - Ollama can return varying formats
    ignoreUnknownKeys = true
    isLenient = true
    
    // Allow special floating-point values
    allowSpecialFloatingPointValues = true
    
    // Be strict about duplicates
    allowComments = false
    allowTrailingComma = false
    
    // Class discriminator for sealed classes (if needed)
    classDiscriminator = "type"
}