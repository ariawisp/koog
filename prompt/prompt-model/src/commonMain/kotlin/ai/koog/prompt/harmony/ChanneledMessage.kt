package ai.koog.prompt.harmony

import kotlinx.serialization.Serializable

/**
 * Backward compatibility wrapper for channel-based messages.
 * 
 * This sealed class provides a migration path from the old ChanneledMessage
 * pattern to the new HarmonyMessage with channel property.
 * 
 * @deprecated Use HarmonyMessage with channel property directly
 */
@Deprecated("Use HarmonyMessage with channel property", ReplaceWith("HarmonyMessage"))
@Serializable
public sealed class ChanneledMessage {
    
    public abstract val content: String
    public abstract val role: Role?
    public abstract val recipient: String?
    
    /**
     * Analysis channel message - internal reasoning not shown to users.
     */
    @Serializable
    public data class Analysis(
        override val content: String,
        override val role: Role? = Role.ASSISTANT,
        override val recipient: String? = null
    ) : ChanneledMessage()
    
    /**
     * Commentary channel message - tool interactions and metadata.
     */
    @Serializable
    public data class Commentary(
        override val content: String,
        override val role: Role? = Role.ASSISTANT,
        override val recipient: String? = null
    ) : ChanneledMessage()
    
    /**
     * Final channel message - user-facing content.
     */
    @Serializable
    public data class Final(
        override val content: String,
        override val role: Role? = Role.ASSISTANT,
        override val recipient: String? = null
    ) : ChanneledMessage()
    
    /**
     * Convert to HarmonyMessage.
     */
    public fun toHarmonyMessage(): HarmonyMessage = when (this) {
        is Analysis -> HarmonyMessage(
            author = HarmonyAuthor.from(role ?: Role.ASSISTANT),
            content = listOf(HarmonyContent.Text(content)),
            channel = "analysis",
            recipient = recipient
        )
        is Commentary -> HarmonyMessage(
            author = HarmonyAuthor.from(role ?: Role.ASSISTANT),
            content = listOf(HarmonyContent.Text(content)),
            channel = "commentary",
            recipient = recipient
        )
        is Final -> HarmonyMessage(
            author = HarmonyAuthor.from(role ?: Role.ASSISTANT),
            content = listOf(HarmonyContent.Text(content)),
            channel = "final",
            recipient = recipient
        )
    }
}

/**
 * Extension to convert list of ChanneledMessages to HarmonyMessages.
 */
public fun List<ChanneledMessage>.toHarmonyMessages(): List<HarmonyMessage> = 
    map { it.toHarmonyMessage() }