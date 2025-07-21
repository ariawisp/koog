package ai.koog.agents.features.tracing.feature

import ai.koog.agents.features.common.config.FeatureConfig
import ai.koog.agents.features.common.message.FeatureMessage
import ai.koog.agents.features.common.remote.server.config.DefaultServerConnectionConfig

/**
 * Configuration for the tracing feature.
 * 
 * This class allows you to configure how the tracing feature behaves, including:
 * - Which message processors receive trace events
 * - Which events are traced (via message filtering)
 * 
 * Example usage:
 * ```kotlin
 * val agent = AIAgent(...) {
 *     install(Tracing) {
 *         // Add message processors to handle trace events
 *         addMessageProcessor(TraceFeatureMessageLogWriter(logger))
 *         addMessageProcessor(TraceFeatureMessageFileWriter(outputFile, fileSystem::sink))
 *         
 *         // Configure message filtering
 *         messageFilter = { message -> 
 *             // Only trace LLM calls and tool calls
 *             message is LLMCallStartEvent || 
 *             message is LLMCallEndEvent || 
 *             message is ToolCallEvent || 
 *             message is ToolCallResultEvent 
 *         }
 *     }
 * }
 * ```
 */
public class TracingConfig() : FeatureConfig() {

    /**
     * A filter for messages to be sent to the tracing message processors.
     * 
     * This function is called for each trace event before it's sent to the message processors.
     * If the function returns true, the event is processed; if it returns false, the event is ignored.
     * 
     * By default, all messages are processed (the filter returns true for all messages).
     * 
     * Example:
     * ```kotlin
     * // Only trace LLM-related events
     * messageFilter = { message ->
     *     message is LLMCallStartEvent || 
     *     message is LLMCallEndEvent
     * }
     * ```
     */
    public var messageFilter: (FeatureMessage) -> Boolean = { true }

    /**
     * TODO: SD -- fix
     */
    public var port: Int = DefaultServerConnectionConfig.DEFAULT_PORT

    /**
     * TODO: SD -- fix
     */
    public var awaitConnection: Boolean = false
}
