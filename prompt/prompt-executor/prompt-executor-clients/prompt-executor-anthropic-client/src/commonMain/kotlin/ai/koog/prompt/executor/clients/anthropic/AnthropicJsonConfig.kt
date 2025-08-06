package ai.koog.prompt.executor.clients.anthropic

import kotlinx.serialization.json.Json
import kotlinx.serialization.modules.SerializersModule

/**
 * Serializers module for Anthropic API types
 */
internal val anthropicSerializersModule = SerializersModule {
    // Custom serializers can be registered here if needed
    // For now, we rely on the annotations in the data classes
}

/**
 * JSON configuration for Anthropic API
 * Uses proper kotlinx.serialization configuration
 */
internal val anthropicJson = Json {
    // Use the custom serializers module
    serializersModule = anthropicSerializersModule
    
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