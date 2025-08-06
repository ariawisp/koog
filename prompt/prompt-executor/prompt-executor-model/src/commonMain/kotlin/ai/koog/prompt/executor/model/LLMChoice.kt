package ai.koog.prompt.executor.model

import ai.koog.prompt.message.Message

/**
 * Backward compatibility type alias for LLMChoice.
 * 
 * LLMChoice has been replaced with Message.Response in the new architecture.
 * This type alias provides a migration path for existing code.
 * 
 * @deprecated Use Message.Response directly
 */
@Deprecated("Use Message.Response directly", ReplaceWith("Message.Response"))
public typealias LLMChoice = Message.Response