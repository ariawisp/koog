package ai.koog.prompt.executor.clients.bedrock

import kotlinx.serialization.json.Json

/**
 * JSON configuration for Bedrock API
 * Uses proper kotlinx.serialization configuration instead of manual JSON building
 */
internal val bedrockJson = Json {
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