package ai.koog.prompt.executor.clients.openrouter

import kotlinx.serialization.json.Json

/**
 * JSON configuration for OpenRouter API
 * Uses proper kotlinx.serialization configuration instead of manual JSON building
 * OpenRouter uses the same JSON format as OpenAI with some extensions
 */
internal val openRouterJson = Json {
    // OpenRouter uses snake_case for property names (handled by @SerialName annotations)
    
    // Don't include null values to keep payloads clean
    encodeDefaults = false
    explicitNulls = false
    
    // Pretty print for debugging (disable in production)
    prettyPrint = false
    
    // Allow structural classes (for our sealed classes)
    allowStructuredMapKeys = true
    
    // Be lenient when reading responses - OpenRouter might include extra fields
    ignoreUnknownKeys = true
    isLenient = true
    
    // Allow special floating-point values
    allowSpecialFloatingPointValues = true
    
    // Polymorphic serialization settings
    // OpenRouter API is not polymorphic, it's "dynamic". Don't add polymorphic discriminators
    classDiscriminatorMode = kotlinx.serialization.json.ClassDiscriminatorMode.NONE
}