package ai.koog.prompt.executor.clients.google

import kotlinx.serialization.*
import kotlinx.serialization.descriptors.*
import kotlinx.serialization.encoding.*
import kotlinx.serialization.json.*
import kotlinx.serialization.modules.SerializersModule
import kotlinx.serialization.modules.polymorphic

/**
 * Custom serializer for GooglePart sealed class
 */
public object GooglePartSerializer : JsonContentPolymorphicSerializer<GooglePart>(GooglePart::class) {
    override fun selectDeserializer(element: JsonElement): DeserializationStrategy<GooglePart> {
        return when {
            element.jsonObject.containsKey("text") -> GooglePart.Text.serializer()
            element.jsonObject.containsKey("inlineData") -> GooglePart.InlineData.serializer()
            element.jsonObject.containsKey("functionCall") -> GooglePart.FunctionCall.serializer()
            element.jsonObject.containsKey("functionResponse") -> GooglePart.FunctionResponse.serializer()
            else -> error("Unknown GooglePart type: $element")
        }
    }
}

/**
 * Serializers module for Google API models
 */
internal val googleSerializersModule = SerializersModule {
    polymorphic(GooglePart::class) {
        subclass(GooglePart.Text::class, GooglePart.Text.serializer())
        subclass(GooglePart.InlineData::class, GooglePart.InlineData.serializer())
        subclass(GooglePart.FunctionCall::class, GooglePart.FunctionCall.serializer())
        subclass(GooglePart.FunctionResponse::class, GooglePart.FunctionResponse.serializer())
    }
}