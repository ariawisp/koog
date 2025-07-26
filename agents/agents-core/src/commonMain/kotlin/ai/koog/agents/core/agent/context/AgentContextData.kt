@file:Suppress("MissingKDocForPublicAPI")

package ai.koog.agents.core.agent.context

import ai.koog.agents.core.annotation.InternalAgentsApi
import ai.koog.prompt.message.Message

@InternalAgentsApi
public class AgentContextData(
    public val messageHistory: List<Message>,
    public val nodeId: String,
    public val lastInput: Any?
)