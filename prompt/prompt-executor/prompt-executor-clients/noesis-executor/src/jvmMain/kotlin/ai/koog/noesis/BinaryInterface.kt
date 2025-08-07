package ai.koog.noesis

import ai.koog.noesis.protocol.*
import com.google.flatbuffers.FlatBufferBuilder
import java.nio.ByteBuffer

/**
 * Binary FlatBuffers Interface for Zero-Copy Performance
 * 
 * This provides the REAL zero-copy interface to Noesis Runtime.
 * All data exchange uses binary FlatBuffers for maximum performance.
 * 
 * ARCHITECTURE: Kotlin FlatBuffers → JNI ByteArray → Rust Zero-Copy → GPU
 */
object BinaryInterface {
    
    /**
     * Build a binary generation request using FlatBuffers
     */
    fun buildGenerationRequest(
        modelHandle: Long,
        messages: List<MessageData>,
        systemConfig: SystemConfigData? = null,
        developerConfig: DeveloperConfigData? = null,
        maxTokens: Int = 2048,
        temperature: Float = 0.7f,
        topP: Float = 0.9f,
        topK: Int = 40,
        seed: Int = 42
    ): ByteArray {
        val builder = FlatBufferBuilder(1024)
        
        // Build system config
        val systemOffset = systemConfig?.let { config ->
            val identityOffset = config.modelIdentity?.let { builder.createString(it) }
            val cutoffOffset = config.knowledgeCutoff?.let { builder.createString(it) }
            val dateOffset = config.conversationStartDate?.let { builder.createString(it) }
            
            val channelOffsets = config.requiredChannels?.map { builder.createString(it) }
            val channelsVector = channelOffsets?.let {
                SystemConfig.createRequiredChannelsVector(builder, it.toIntArray())
            }
            
            SystemConfig.startSystemConfig(builder)
            identityOffset?.let { SystemConfig.addModelIdentity(builder, it) }
            SystemConfig.addReasoningEffort(builder, config.reasoningEffort.value)
            cutoffOffset?.let { SystemConfig.addKnowledgeCutoff(builder, it) }
            dateOffset?.let { SystemConfig.addConversationStartDate(builder, it) }
            channelsVector?.let { SystemConfig.addRequiredChannels(builder, it) }
            SystemConfig.endSystemConfig(builder)
        }
        
        // Build developer config
        val developerOffset = developerConfig?.let { config ->
            val instructionsOffset = builder.createString(config.instructions)
            
            DeveloperConfig.startDeveloperConfig(builder)
            DeveloperConfig.addInstructions(builder, instructionsOffset)
            DeveloperConfig.endDeveloperConfig(builder)
        }
        
        // Build messages
        val messageOffsets = messages.map { msg ->
            val contentOffset = builder.createString(msg.content)
            val channelOffset = msg.channel?.let { builder.createString(it) }
            val recipientOffset = msg.recipient?.let { builder.createString(it) }
            
            Message.startMessage(builder)
            Message.addRole(builder, msg.role.value)
            Message.addContent(builder, contentOffset)
            channelOffset?.let { Message.addChannel(builder, it) }
            recipientOffset?.let { Message.addRecipient(builder, it) }
            Message.endMessage(builder)
        }
        
        val messagesVector = GenerationRequest.createMessagesVector(
            builder, 
            messageOffsets.toIntArray()
        )
        
        // Build the request
        GenerationRequest.startGenerationRequest(builder)
        GenerationRequest.addModelHandle(builder, modelHandle)
        systemOffset?.let { GenerationRequest.addSystemConfig(builder, it) }
        developerOffset?.let { GenerationRequest.addDeveloperConfig(builder, it) }
        GenerationRequest.addMessages(builder, messagesVector)
        GenerationRequest.addMaxTokens(builder, maxTokens)
        GenerationRequest.addTemperature(builder, temperature)
        GenerationRequest.addTopP(builder, topP)
        GenerationRequest.addTopK(builder, topK)
        GenerationRequest.addSeed(builder, seed.toUInt())
        
        val request = GenerationRequest.endGenerationRequest(builder)
        builder.finish(request)
        
        return builder.sizedByteArray()
    }
    
    /**
     * Parse a binary generation response from FlatBuffers
     */
    fun parseGenerationResponse(buffer: ByteArray): GenerationResponseData {
        val byteBuffer = ByteBuffer.wrap(buffer)
        val response = GenerationResponse.getRootAsGenerationResponse(byteBuffer)
        
        return GenerationResponseData(
            requestId = response.requestId,
            tokens = if (response.tokensLength > 0) {
                (0 until response.tokensLength).map { i -> response.tokens(i).toLong() }
            } else null,
            text = response.text,
            channels = if (response.channelsLength > 0) {
                (0 until response.channelsLength).map { i -> response.channels(i) }
            } else null,
            metadata = response.metadata,
            error = response.error
        )
    }
}

// Kotlin data classes for cleaner API

data class MessageData(
    val role: RoleEnum,
    val content: String,
    val channel: String? = null,
    val recipient: String? = null
)

enum class RoleEnum(val value: Byte) {
    USER(Role.User),
    ASSISTANT(Role.Assistant),
    SYSTEM(Role.System),
    DEVELOPER(Role.Developer),
    TOOL(Role.Tool)
}

data class SystemConfigData(
    val modelIdentity: String? = "You are ChatGPT, a large language model trained by OpenAI.",
    val reasoningEffort: ReasoningEffortEnum = ReasoningEffortEnum.MEDIUM,
    val knowledgeCutoff: String? = "2024-06",
    val conversationStartDate: String? = null,
    val requiredChannels: List<String>? = listOf("analysis", "commentary", "final")
)

enum class ReasoningEffortEnum(val value: Byte) {
    LOW(ReasoningEffort.Low),
    MEDIUM(ReasoningEffort.Medium),
    HIGH(ReasoningEffort.High)
}

data class DeveloperConfigData(
    val instructions: String
)

data class GenerationResponseData(
    val requestId: String?,
    val tokens: List<Long>?,
    val text: String?,
    val channels: List<String?>?,
    val metadata: String?,
    val error: String?
)

/**
 * Extension function for NoesisRuntime to use binary interface
 */
fun NoesisRuntime.generateBinary(
    modelHandle: Long,
    messages: List<MessageData>,
    systemConfig: SystemConfigData? = SystemConfigData(),
    developerConfig: DeveloperConfigData? = DeveloperConfigData(
        instructions = "You are a helpful AI assistant. Provide clear and accurate responses."
    ),
    maxTokens: Int = 2048,
    temperature: Float = 0.7f
): GenerationResponseData {
    val requestBuffer = BinaryInterface.buildGenerationRequest(
        modelHandle = modelHandle,
        messages = messages,
        systemConfig = systemConfig,
        developerConfig = developerConfig,
        maxTokens = maxTokens,
        temperature = temperature
    )
    
    val responseBuffer = this.generateBinary(requestBuffer)
    return BinaryInterface.parseGenerationResponse(responseBuffer)
}

