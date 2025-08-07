package ai.koog.agents.core.dsl.extension

import ai.koog.agents.core.agent.session.AIAgentLLMWriteSession
import ai.koog.prompt.message.Message
// Note: ToolChoice functionality removed in Harmony-native architecture
// Tool usage is now controlled through system message reasoning directives
import kotlinx.datetime.Instant

/**
 * Clears the history of messages in the current AI Agent LLM Write Session.
 *
 * This method resets the message history by setting it to an empty list.
 * It is useful when you want to start a new conversation or reset the session's context.
 */
public fun AIAgentLLMWriteSession.clearHistory() {
    prompt = prompt.withMessages { emptyList() }
}

/**
 * Keeps only the last N messages in the session's prompt by removing all earlier messages.
 *
 * @param n The number of most recent messages to retain in the session's prompt.
 */
public fun AIAgentLLMWriteSession.leaveLastNMessages(n: Int) {
    prompt = prompt.withMessages { it.takeLast(n) }
}

/**
 * Removes the last `n` messages from the current prompt in the write session.
 *
 * @param n The number of messages to remove from the end of the current message list.
 */
public fun AIAgentLLMWriteSession.dropLastNMessages(n: Int) {
    prompt = prompt.withMessages { it.dropLast(n) }
}

/**
 * Removes all messages from the current session's prompt that have a timestamp
 * earlier than the specified timestamp.
 *
 * @param timestamp The threshold timestamp. Messages with a timestamp earlier than this will be removed.
 */
public fun AIAgentLLMWriteSession.leaveMessagesFromTimestamp(timestamp: Instant) {
    prompt = prompt.copy(
        conversation = prompt.conversation.copy(
            messages = prompt.messages.filter { it.timestamp >= timestamp.toEpochMilliseconds() }
        )
    )
}

// Tool choice functionality has been removed in Harmony-native architecture.
// GPT-OSS models handle tool selection through natural reasoning in the analysis channel.
// Use system message directives or developer context to guide tool usage patterns.

/**
 * Rewrites LLM message history, leaving only user message and resulting TLDR.
 *
 * Default is `null`, which means entire history will be used.
 * @param preserveMemory Whether to preserve memory-related messages in the history.
 */
public suspend fun AIAgentLLMWriteSession.replaceHistoryWithTLDR(
    strategy: HistoryCompressionStrategy = HistoryCompressionStrategy.WholeHistory,
    preserveMemory: Boolean = true
) {
    // Store memory-related messages if needed
    val memoryMessages = if (preserveMemory) {
        prompt.messages.filter { message ->
            val textContent = message.getTextContent()
            textContent.contains("Here are the relevant facts from memory") ||
                textContent.contains("Memory feature is not enabled")
        }
    } else {
        emptyList()
    }

    // Pass HarmonyMessages directly
    strategy.compress(this, preserveMemory, memoryMessages)
}

/**
 * Drops all trailing tool call messages from the current prompt
 */
public fun AIAgentLLMWriteSession.dropTrailingToolCalls() {
    rewritePrompt { prompt -> 
        prompt.copy(
            conversation = prompt.conversation.copy(
                messages = prompt.messages.dropLastWhile { msg ->
                    // Check if it's a tool call (assistant message in commentary channel with recipient)
                    msg.author.role == ai.koog.prompt.harmony.Role.ASSISTANT && 
                    msg.channel == "commentary" && 
                    msg.recipient?.startsWith("functions.") == true
                }
            )
        )
    }
}
