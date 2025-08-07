package ai.koog.prompt.harmony

import kotlinx.serialization.Serializable
import kotlinx.datetime.Clock
import kotlinx.datetime.TimeZone
import kotlinx.datetime.todayIn
import kotlin.uuid.ExperimentalUuidApi
import kotlin.uuid.Uuid

/**
 * Harmony special tokens as defined in the format specification.
 * These tokens structure the conversation format that GPT-OSS models were trained on.
 */
public object HarmonyTokens {
    public const val START: String = "<|start|>"
    public const val END: String = "<|end|>"
    public const val MESSAGE: String = "<|message|>"
    public const val CHANNEL: String = "<|channel|>"
    public const val CONSTRAIN: String = "<|constrain|>"
    public const val RETURN: String = "<|return|>"
    public const val CALL: String = "<|call|>"
    
    // Token IDs for o200k_harmony encoding
    public object TokenIds {
        public const val START: Int = 200006
        public const val END: Int = 200007
        public const val MESSAGE: Int = 200008
        public const val CHANNEL: Int = 200005
        public const val CONSTRAIN: Int = 200003
        public const val RETURN: Int = 200002
        public const val CALL: Int = 200012
    }
}

// HarmonyCore has been merged into Prompt
// Prompt IS the Harmony intermediate representation

/**
 * System context defines the model's identity and behavior parameters.
 */
@Serializable
public data class SystemContext(
    val modelIdentity: String = "You are a helpful assistant",
    val reasoningEffort: ReasoningEffort = ReasoningEffort.MEDIUM,
    val knowledgeCutoff: String = "2024-06",
    val currentDate: String = Clock.System.todayIn(TimeZone.currentSystemDefault()).toString(),
    val securityPolicy: SecurityPolicy = SecurityPolicy.DEFAULT
)

/**
 * Developer context contains instructions and tool definitions.
 */
@Serializable
public data class DeveloperContext(
    val instructions: String = "Be concise and accurate",
    val tools: List<HarmonyTool> = emptyList(),
    val constraints: List<String> = emptyList()
) {
    public companion object {
        public fun empty(): DeveloperContext = DeveloperContext()
    }
}

/**
 * ConversationGraph - Rich multi-channel conversation structure.
 * Uses new HarmonyMessage format for full Rust API compatibility.
 */
@Serializable
public data class ConversationGraph(
    val messages: List<HarmonyMessage>,
    val tools: ToolNamespace = ToolNamespace.empty(),
    val constraints: List<TypeConstraint> = emptyList()
) {
    /**
     * Extract all analysis messages (internal reasoning).
     * This content is never shown to users but drives model behavior.
     */
    public val analysisChannel: List<HarmonyMessage>
        get() = messages.filter { it.channel == "analysis" }
    
    /**
     * Extract all commentary messages (tool calls, metadata).
     * This content provides context about tool interactions.
     */
    public val commentaryChannel: List<HarmonyMessage>
        get() = messages.filter { it.channel == "commentary" }
    
    /**
     * Extract all final messages (user-safe responses).
     * This is the content that gets presented to users.
     */
    public val finalChannel: List<HarmonyMessage>
        get() = messages.filter { it.channel == "final" || it.channel == null }
    
    /**
     * Filter to only user-safe content by removing analysis channel.
     * Critical safety feature that prevents reasoning leakage.
     */
    public fun filterUserSafe(): ConversationGraph = copy(
        messages = messages.filter { it.channel != "analysis" }
    )
    
    /**
     * Add a message to the conversation.
     */
    public fun addMessage(message: HarmonyMessage): ConversationGraph = copy(
        messages = messages + message
    )
    
    /**
     * Add multiple messages to the conversation.
     */
    public fun addMessages(newMessages: List<HarmonyMessage>): ConversationGraph = copy(
        messages = messages + newMessages
    )
    
    public companion object {
        /**
         * Create empty conversation graph.
         */
        public fun empty(): ConversationGraph = ConversationGraph(emptyList())
        
    }
}

/**
 * Author represents the creator of a message with role and optional name.
 * Maps directly to Rust Author struct.
 */
@Serializable
public data class HarmonyAuthor(
    val role: Role,
    val name: String? = null
) {
    public companion object {
        public fun from(role: Role): HarmonyAuthor = HarmonyAuthor(role)
        public fun named(role: Role, name: String): HarmonyAuthor = HarmonyAuthor(role, name)
    }
}

/**
 * Role enum matching the Rust Role enum exactly.
 */
public enum class Role(public val value: String) {
    USER("user"),
    ASSISTANT("assistant"),
    SYSTEM("system"),
    DEVELOPER("developer"),
    TOOL("tool")
}

/**
 * Content types matching Rust content system.
 */
@Serializable
public sealed class HarmonyContent {
    @Serializable
    public data class Text(val text: String) : HarmonyContent()
    
    @Serializable 
    public data class System(
        val modelIdentity: String? = null,
        val instructions: String? = null,
        val knowledgeCutoff: String? = null,
        val currentDate: String? = null,
        val reasoningEffort: ReasoningEffort? = null,
        val validChannels: List<String> = listOf("analysis", "commentary", "final"),
        val builtInTools: List<String> = emptyList() // e.g., ["browser", "python"]
    ) : HarmonyContent()
    
    @Serializable
    public data class Developer(
        val instructions: String? = null,
        val tools: List<HarmonyTool> = emptyList(),
        val constraints: List<String> = emptyList(),
        val responseFormats: Map<String, String> = emptyMap() // format name -> schema
    ) : HarmonyContent()
}

/**
 * Full Harmony message matching Rust Message struct exactly.
 * This replaces the old ChanneledMessage with proper Rust API alignment.
 * 
 * Header format: {role}[ to={recipient}][ {channel}][ <|constrain|>{contentType}]
 * Message format: <|start|>{header}<|message|>{content}<|end|>
 */
@Serializable
public data class HarmonyMessage(
    val author: HarmonyAuthor,
    val recipient: String? = null,
    val content: List<HarmonyContent>,
    val channel: String? = null,
    val contentType: String? = null,
    val timestamp: Long = Clock.System.now().toEpochMilliseconds()
) {
    public companion object {
        /**
         * Create from role and content - matches Rust API.
         */
        public fun fromRoleAndContent(role: Role, content: String): HarmonyMessage = 
            HarmonyMessage(
                author = HarmonyAuthor.from(role),
                content = listOf(HarmonyContent.Text(content))
            )
        
        /**
         * Create system message with structured content.
         */
        public fun system(
            modelIdentity: String? = null, 
            instructions: String? = null,
            knowledgeCutoff: String? = null,
            currentDate: String? = null,
            reasoningEffort: ReasoningEffort? = null,
            builtInTools: List<String> = emptyList()
        ): HarmonyMessage = HarmonyMessage(
            author = HarmonyAuthor.from(Role.SYSTEM),
            content = listOf(HarmonyContent.System(modelIdentity, instructions, knowledgeCutoff, currentDate, reasoningEffort, builtInTools = builtInTools))
        )
        
        /**
         * Create developer message with structured content.
         */
        public fun developer(
            instructions: String? = null,
            tools: List<HarmonyTool> = emptyList(),
            constraints: List<String> = emptyList()
        ): HarmonyMessage = HarmonyMessage(
            author = HarmonyAuthor.from(Role.DEVELOPER),
            content = listOf(HarmonyContent.Developer(instructions, tools, constraints))
        )
        
        /**
         * Create user message.
         */
        public fun user(text: String): HarmonyMessage = fromRoleAndContent(Role.USER, text)
        
        /**
         * Create assistant message.
         */
        public fun assistant(text: String): HarmonyMessage = fromRoleAndContent(Role.ASSISTANT, text)
        
        /**
         * Create tool message.
         */
        public fun tool(text: String, recipient: String? = null): HarmonyMessage = 
            HarmonyMessage(
                author = HarmonyAuthor.from(Role.TOOL),
                content = listOf(HarmonyContent.Text(text)),
                recipient = recipient
            )
    }
    
    /**
     * Add channel information - matches Rust with_channel.
     */
    public fun withChannel(channel: String): HarmonyMessage = copy(channel = channel)
    
    /**
     * Add recipient information - matches Rust with_recipient.
     */
    public fun withRecipient(recipient: String): HarmonyMessage = copy(recipient = recipient)
    
    /**
     * Get text content as string.
     */
    public fun getTextContent(): String = content.filterIsInstance<HarmonyContent.Text>()
        .joinToString("") { it.text }
    
    // Note: Harmony format rendering is handled by Rust library via JNI
    // This data structure is converted to Rust types for processing
    
    /**
     * Create a tool call message with proper format.
     */
    public fun asToolCall(toolName: String, arguments: String): HarmonyMessage {
        return copy(
            channel = "commentary",
            recipient = toolName,
            contentType = "json",
            content = listOf(HarmonyContent.Text(arguments))
        )
    }
    
    /**
     * Create a tool response message.
     */
    public fun asToolResponse(toolName: String, output: String): HarmonyMessage {
        return HarmonyMessage(
            author = HarmonyAuthor.named(Role.TOOL, toolName),
            recipient = "assistant",
            content = listOf(HarmonyContent.Text(output)),
            channel = "commentary"
        )
    }
}


// HarmonyMetadata removed - using ModelConfig for simplified parameters

/**
 * Reasoning effort levels control how much internal analysis the model performs.
 */
public enum class ReasoningEffort {
    LOW,     // Quick responses, minimal chain-of-thought
    MEDIUM,  // Balanced reasoning and response time
    HIGH     // Deep analysis with extensive chain-of-thought
}

/**
 * Security policy for tool execution and content filtering.
 */
@Serializable
public data class SecurityPolicy(
    val allowedTools: Set<String> = emptySet(),
    val blockedPatterns: List<String> = emptyList(),
    val maxToolCalls: Int = 10
) {
    public companion object {
        public val DEFAULT: SecurityPolicy = SecurityPolicy()
    }
}

/**
 * Tool definition in Harmony format.
 * Uses Koog's existing ToolDescriptor as the source of truth.
 */
@Serializable
public data class HarmonyTool(
    val name: String,
    val description: String,
    val parameters: HarmonyToolParameters,
    val namespace: String = "functions"
)

/**
 * Tool parameters with rich type information.
 */
@Serializable
public data class HarmonyToolParameters(
    val type: String = "object",
    val properties: Map<String, HarmonyToolProperty>,
    val required: List<String> = emptyList()
)

/**
 * Individual tool parameter property.
 */
@Serializable
public data class HarmonyToolProperty(
    val type: String,
    val description: String,
    val enum: List<String>? = null
)

/**
 * Tool namespace for organizing and constraining tool access.
 */
@Serializable
public data class ToolNamespace(
    val functions: List<HarmonyTool> = emptyList(),
    val policies: List<String> = emptyList()
) {
    public companion object {
        public fun empty(): ToolNamespace = ToolNamespace()
    }
}

/**
 * Type constraint for validating tool parameters and responses.
 */
@Serializable
public data class TypeConstraint(
    val field: String,
    val type: String,
    val required: Boolean = false
)

// HarmonyCoreBuilder removed - use PromptBuilder directly via prompt DSL

/**
 * Builder for SystemContext.
 */
public class SystemContextBuilder {
    public var modelIdentity: String = "You are a helpful assistant"
    public var reasoningEffort: ReasoningEffort = ReasoningEffort.MEDIUM
    public var knowledgeCutoff: String = "2024-06"
    
    public fun build(): SystemContext = SystemContext(
        modelIdentity = modelIdentity,
        reasoningEffort = reasoningEffort,
        knowledgeCutoff = knowledgeCutoff
    )
}

/**
 * Builder for DeveloperContext.
 */
public class DeveloperContextBuilder {
    public var instructions: String = "Be concise and accurate"
    public var tools: MutableList<HarmonyTool> = mutableListOf()
    
    public fun build(): DeveloperContext = DeveloperContext(
        instructions = instructions,
        tools = tools
    )
}


/**
 * Harmony response structure matching Rust API.
 */
@Serializable
public data class HarmonyResponse(
    val messages: List<HarmonyMessage>,
    val metadata: HarmonyResponseMetadata
) {
    /**
     * Get user-safe content (Final channel only).
     */
    public fun getUserSafeContent(): List<String> = messages
        .filter { it.channel == "final" || it.channel == null }
        .map { it.getTextContent() }
        .filter { it.isNotBlank() }
    
    /**
     * Extract chain-of-thought reasoning from analysis channel.
     */
    public fun getChainOfThought(): List<String> = messages
        .filter { it.channel == "analysis" }
        .map { it.getTextContent() }
        .filter { it.isNotBlank() }
    
    /**
     * Get tool interactions from commentary channel.
     */
    public fun getToolInteractions(): List<String> = messages
        .filter { it.channel == "commentary" }
        .map { it.getTextContent() }
        .filter { it.isNotBlank() }
}

/**
 * Response metadata with channel statistics.
 */
@OptIn(ExperimentalUuidApi::class)
@Serializable
public data class HarmonyResponseMetadata(
    val model: String,
    val requestId: String = Uuid.random().toString(),
    val channelStats: ChannelStats,
    val totalTokens: Int = 0,
    val promptTokens: Int = 0,
    val completionTokens: Int = 0
)

/**
 * Statistics about channel usage.
 */
@Serializable
public data class ChannelStats(
    val analysisTokens: Int = 0,
    val commentaryTokens: Int = 0,
    val finalTokens: Int = 0
) {
    val totalTokens: Int get() = analysisTokens + commentaryTokens + finalTokens
    val reasoningRatio: Double get() = if (totalTokens > 0) analysisTokens.toDouble() / totalTokens else 0.0
}

/**
 * Harmony agent configuration.
 */
@Serializable
public data class HarmonyAgentConfig(
    val defaultModel: LLModel,
    val defaultReasoning: ReasoningEffort = ReasoningEffort.MEDIUM,
    val maxTokens: Int? = null,
    val temperature: Double? = null,
    val includeReasoning: Boolean = false
)

/**
 * Language model specification.
 */
@Serializable
public data class LLModel(
    val name: String,
    val provider: String
)

// HarmonyPromptBuilder removed - use PromptBuilder directly via prompt DSL

/**
 * Builder for conversation within the DSL.
 */
public class ConversationBuilder {
    private val messages = mutableListOf<HarmonyMessage>()
    
    public fun user(content: String): Unit {
        messages.add(HarmonyMessage.user(content).withChannel("final"))
    }
    
    public fun assistant(content: String): Unit {
        messages.add(HarmonyMessage.assistant(content).withChannel("final"))
    }
    
    public fun analysis(content: String): Unit {
        messages.add(HarmonyMessage.assistant(content).withChannel("analysis"))
    }
    
    public fun commentary(content: String, recipient: String? = null): Unit {
        messages.add(HarmonyMessage.tool(content, recipient).withChannel("commentary"))
    }
    
    public fun build(): ConversationGraph = ConversationGraph(messages)
}

/**
 * Converters between Koog's existing types and Harmony format.
 * This preserves all existing tool infrastructure while enabling
 * Harmony's semantic richness.
 */
public object HarmonyConverter {
    
    /**
     * Convert Koog's ToolDescriptor to HarmonyTool.
     * This is the bridge between Koog's existing tool system and Harmony format.
     */
    public fun fromToolDescriptor(descriptor: ai.koog.agents.core.tools.ToolDescriptor): HarmonyTool {
        return HarmonyTool(
            name = descriptor.name,
            description = descriptor.description,
            parameters = HarmonyToolParameters(
                type = "object",
                properties = buildToolProperties(descriptor),
                required = descriptor.requiredParameters.map { it.name }
            )
        )
    }
    
    /**
     * Convert list of ToolDescriptors to Harmony format.
     */
    public fun fromToolDescriptors(descriptors: List<ai.koog.agents.core.tools.ToolDescriptor>): List<HarmonyTool> {
        return descriptors.map { fromToolDescriptor(it) }
    }
    
    /**
     * Generate TypeScript-like tool definition syntax from ToolDescriptor.
     * This produces the format that Harmony models were trained on:
     * 
     * ```typescript
     * namespace functions {
     *   // Gets the current weather in the provided location.
     *   type get_current_weather = (_: {
     *     // The city and state, e.g. San Francisco, CA
     *     location: string,
     *     format?: "celsius" | "fahrenheit", // default: celsius
     *   }) => any;
     * }
     * ```
     */
    public fun toTypeScriptDefinition(descriptor: ai.koog.agents.core.tools.ToolDescriptor): String {
        val name = descriptor.name
        val description = descriptor.description
        val params = descriptor.requiredParameters + descriptor.optionalParameters
        
        return buildString {
            appendLine("namespace functions {")
            appendLine("  // $description")
            append("  type $name = (_: {")
            
            if (params.isNotEmpty()) {
                appendLine()
                params.forEach { param ->
                    val optional = if (param in descriptor.optionalParameters) "?" else ""
                    val tsType = mapParameterTypeToTypeScript(param.type)
                    appendLine("    // ${param.description}")
                    append("    ${param.name}$optional: $tsType,")
                }
                appendLine()
                append("  ")
            }
            
            appendLine("}) => any;")
            appendLine("}")
        }
    }
    
    /**
     * Generate full TypeScript namespace with all tools.
     */
    public fun toTypeScriptNamespace(descriptors: List<ai.koog.agents.core.tools.ToolDescriptor>): String {
        return buildString {
            appendLine("namespace functions {")
            descriptors.forEach { descriptor ->
                appendLine("  // ${descriptor.description}")
                val params = descriptor.requiredParameters + descriptor.optionalParameters
                append("  type ${descriptor.name} = (_: {")
                
                if (params.isNotEmpty()) {
                    appendLine()
                    params.forEach { param ->
                        val optional = if (param in descriptor.optionalParameters) "?" else ""
                        val tsType = mapParameterTypeToTypeScript(param.type)
                        appendLine("    // ${param.description}")
                        append("    ${param.name}$optional: $tsType,")
                    }
                    appendLine()
                    append("  ")
                }
                
                appendLine("}) => any;")
                appendLine()
            }
            appendLine("}")
        }
    }
    
    private fun buildToolProperties(descriptor: ai.koog.agents.core.tools.ToolDescriptor): Map<String, HarmonyToolProperty> {
        val allParams = descriptor.requiredParameters + descriptor.optionalParameters
        return allParams.associate { param ->
            param.name to HarmonyToolProperty(
                type = mapParameterTypeToJsonSchema(param.type),
                description = param.description,
                enum = extractEnumValues(param.type)
            )
        }
    }
    
    private fun mapParameterTypeToJsonSchema(type: ai.koog.agents.core.tools.ToolParameterType): String {
        return when (type) {
            is ai.koog.agents.core.tools.ToolParameterType.String -> "string"
            is ai.koog.agents.core.tools.ToolParameterType.Integer -> "integer"
            is ai.koog.agents.core.tools.ToolParameterType.Float -> "number"
            is ai.koog.agents.core.tools.ToolParameterType.Boolean -> "boolean"
            is ai.koog.agents.core.tools.ToolParameterType.Enum -> "string"
            is ai.koog.agents.core.tools.ToolParameterType.List -> "array"
            is ai.koog.agents.core.tools.ToolParameterType.Object -> "object"
        }
    }
    
    private fun mapParameterTypeToTypeScript(type: ai.koog.agents.core.tools.ToolParameterType): String {
        return when (type) {
            is ai.koog.agents.core.tools.ToolParameterType.String -> "string"
            is ai.koog.agents.core.tools.ToolParameterType.Integer -> "number"
            is ai.koog.agents.core.tools.ToolParameterType.Float -> "number"
            is ai.koog.agents.core.tools.ToolParameterType.Boolean -> "boolean"
            is ai.koog.agents.core.tools.ToolParameterType.Enum -> {
                type.entries.joinToString(" | ") { "\"$it\"" }
            }
            is ai.koog.agents.core.tools.ToolParameterType.List -> {
                val itemType = mapParameterTypeToTypeScript(type.itemsType)
                "${itemType}[]"
            }
            is ai.koog.agents.core.tools.ToolParameterType.Object -> "object"
        }
    }
    
    private fun extractEnumValues(type: ai.koog.agents.core.tools.ToolParameterType): List<String>? {
        return when (type) {
            is ai.koog.agents.core.tools.ToolParameterType.Enum -> type.entries.toList()
            else -> null
        }
    }
}

/**
 * Built-in tools that Harmony models were trained on.
 * These should be defined in the system message, not developer message.
 */
public object HarmonyBuiltInTools {
    
    /**
     * Browser tool definition as specified in Harmony format.
     * Requests go to 'analysis' channel with recipients like 'browser.search'.
     */
    public val BROWSER_TOOL_DEFINITION: String = """
        ## browser
        
        // Tool for browsing.
        // The `cursor` appears in brackets before each browsing display: `[{cursor}]`.
        // Cite information from the tool using the following format:
        // `【{cursor}†L{line_start}(-L{line_end})?】`, for example: `【6†L9-L11】` or `【8†L3】`.
        // Do not quote more than 10 words directly from the tool output.
        // sources=web (default: web)
        namespace browser {
        
        // Searches for information related to `query` and displays `topn` results.
        type search = (_: {
        query: string,
        topn?: number, // default: 10
        source?: string,
        }) => any;
        
        // Opens the link `id` from the page indicated by `cursor` starting at line number `loc`, showing `num_lines` lines.
        // Valid link ids are displayed with the formatting: `【{id}†.*】`.
        // If `cursor` is not provided, the most recent page is implied.
        // If `id` is a string, it is treated as a fully qualified URL associated with `source`.
        // If `loc` is not provided, the viewport will be positioned at the beginning of the document or centered on the most relevant passage, if available.
        // Use this function without `id` to scroll to a new location of an opened page.
        type open = (_: {
        id?: number | string, // default: -1
        cursor?: number, // default: -1
        loc?: number, // default: -1
        num_lines?: number, // default: -1
        view_source?: boolean, // default: false
        source?: string,
        }) => any;
        
        // Finds exact matches of `pattern` in the current page, or the page given by `cursor`.
        type find = (_: {
        pattern: string,
        cursor?: number, // default: -1
        }) => any;
        
        } // namespace browser
    """.trimIndent()
    
    /**
     * Python tool definition as specified in Harmony format.
     * Requests go to 'analysis' channel with recipient 'python'.
     */
    public val PYTHON_TOOL_DEFINITION: String = """
        ## python
        
        Use this tool to execute Python code in your chain of thought. The code will not be shown to the user. This tool should be used for internal reasoning, but not for code that is intended to be visible to the user (e.g. when creating plots, tables, or files).
        
        When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 120.0 seconds. The drive at '/mnt/data' can be used to save and persist user files. Internet access for this session is UNKNOWN. Depends on the cluster.
    """.trimIndent()
}

/**
 * Note: Harmony message parsing is handled by the Rust library via JNI.
 * Kotlin provides the structured data representation, and Rust handles
 * all token-level parsing and rendering operations.
 */