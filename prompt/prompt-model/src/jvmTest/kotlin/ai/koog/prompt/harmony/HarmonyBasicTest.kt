package ai.koog.prompt.harmony

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.assertFalse

/**
 * Basic tests for Harmony data structures and channel safety.
 */
class HarmonyBasicTest {
    
    @Test
    fun `test HarmonyMessage creation and channel separation`() {
        // Create messages with different channels
        val userMessage = HarmonyMessage(
            author = HarmonyAuthor(Role.USER, null),
            content = listOf(HarmonyContent.Text("What is 2 + 2?")),
            channel = null
        )
        
        val analysisMessage = HarmonyMessage(
            author = HarmonyAuthor(Role.ASSISTANT, null),
            content = listOf(HarmonyContent.Text("Simple arithmetic question")),
            channel = "analysis"
        )
        
        val finalMessage = HarmonyMessage(
            author = HarmonyAuthor(Role.ASSISTANT, null),
            content = listOf(HarmonyContent.Text("2 + 2 equals 4")),
            channel = "final"
        )
        
        // Test channel separation
        val messages = listOf(userMessage, analysisMessage, finalMessage)
        val analysisOnly = messages.filter { it.channel == "analysis" }
        val userFacing = messages.filter { it.channel != "analysis" }
        
        assertEquals(1, analysisOnly.size, "Should have one analysis message")
        assertEquals(2, userFacing.size, "Should have two user-facing messages")
        
        // Verify content extraction
        assertEquals("Simple arithmetic question", analysisMessage.getTextContent())
        assertEquals("2 + 2 equals 4", finalMessage.getTextContent())
    }
    
    @Test
    fun `test Role enum values match Rust API`() {
        // Verify role values match what Rust expects
        assertEquals("user", Role.USER.value)
        assertEquals("assistant", Role.ASSISTANT.value)
        assertEquals("system", Role.SYSTEM.value)
        assertEquals("developer", Role.DEVELOPER.value)
        assertEquals("tool", Role.TOOL.value)
    }
    
    @Test
    fun `test HarmonyAuthor creation helpers`() {
        val simpleAuthor = HarmonyAuthor.from(Role.USER)
        assertEquals(Role.USER, simpleAuthor.role)
        assertEquals(null, simpleAuthor.name)
        
        val namedAuthor = HarmonyAuthor.named(Role.ASSISTANT, "GPT-4")
        assertEquals(Role.ASSISTANT, namedAuthor.role)
        assertEquals("GPT-4", namedAuthor.name)
    }
    
    @Test
    fun `test HarmonyContent types`() {
        val textContent = HarmonyContent.Text("Hello world")
        assertTrue(textContent is HarmonyContent.Text)
        assertEquals("Hello world", textContent.text)
        
        val systemContent = HarmonyContent.System(
            modelIdentity = "You are ChatGPT",
            reasoningEffort = ReasoningEffort.HIGH,
            knowledgeCutoff = "2024-06"
        )
        assertTrue(systemContent is HarmonyContent.System)
        assertEquals(ReasoningEffort.HIGH, systemContent.reasoningEffort)
        
        val developerContent = HarmonyContent.Developer(
            instructions = "Be helpful",
            tools = emptyList()
        )
        assertTrue(developerContent is HarmonyContent.Developer)
        assertEquals("Be helpful", developerContent.instructions)
    }
    
    @Test
    fun `test JsonHarmonyMessage conversion`() {
        val jsonMessage = JsonHarmonyMessage(
            role = "assistant",
            content = "Hello from assistant",
            channel = "final",
            recipient = null,
            contentType = "text",
            name = "GPT-4"
        )
        
        val harmonyMessage = jsonMessage.toHarmonyMessage()
        
        assertEquals(Role.ASSISTANT, harmonyMessage.author.role)
        assertEquals("GPT-4", harmonyMessage.author.name)
        assertEquals("final", harmonyMessage.channel)
        assertEquals("text", harmonyMessage.contentType)
        assertEquals("Hello from assistant", harmonyMessage.getTextContent())
    }
    
    @Test
    fun `test channel safety - analysis never in user content`() {
        val messages = listOf(
            HarmonyMessage(
                author = HarmonyAuthor.from(Role.USER),
                content = listOf(HarmonyContent.Text("Question")),
                channel = null
            ),
            HarmonyMessage(
                author = HarmonyAuthor.from(Role.ASSISTANT),
                content = listOf(HarmonyContent.Text("SECRET ANALYSIS")),
                channel = "analysis"
            ),
            HarmonyMessage(
                author = HarmonyAuthor.from(Role.ASSISTANT),
                content = listOf(HarmonyContent.Text("Public answer")),
                channel = "final"
            )
        )
        
        // Filter for user-safe content
        val userSafe = messages.filter { it.channel != "analysis" }
        val userContent = userSafe.joinToString(" ") { it.getTextContent() }
        
        assertFalse(
            userContent.contains("SECRET"),
            "Analysis content should NEVER appear in user-safe messages"
        )
        assertTrue(
            userContent.contains("Public answer"),
            "Final content should appear in user-safe messages"
        )
    }
    
    @Test
    fun `test RenderConfig options`() {
        val defaultConfig = RenderConfig.default()
        assertTrue(defaultConfig.includeSystemTokens)
        assertTrue(defaultConfig.includeDeveloperTokens)
        assertEquals(null, defaultConfig.maxTokens)
        
        val compactConfig = RenderConfig.compact()
        assertFalse(compactConfig.includeSystemTokens)
        assertFalse(compactConfig.includeDeveloperTokens)
        
        val limitedConfig = RenderConfig.limited(1000)
        assertEquals(1000, limitedConfig.maxTokens)
    }
}