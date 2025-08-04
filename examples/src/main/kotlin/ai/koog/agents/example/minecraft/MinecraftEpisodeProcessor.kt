package ai.koog.agents.example.minecraft

/**
 * Minecraft-specific patterns and utilities for extracting game entities
 * from conversation text. These patterns are used by the graph memory system
 * during conversation ingestion to identify relevant entities and relationships.
 *
 * The new architecture uses conversation ingestion rather than custom episode
 * processors, so this file now provides utilities that can be used with
 * the unified AgentMemory system.
 */
public object MinecraftPatterns {

    /**
     * Extract player names from text
     */
    public fun extractPlayers(text: String): List<String> {
        val players = mutableSetOf<String>()

        // Extract from "Player X" mentions
        PLAYER_PATTERN.findAll(text).forEach { match ->
            players.add(match.groupValues[1])
        }

        // Extract from action patterns
        listOf(BUILD_PATTERN, ATTACK_PATTERN, OWNERSHIP_PATTERN).forEach { pattern ->
            pattern.findAll(text).forEach { match ->
                players.add(match.groupValues[1])
            }
        }

        return players.toList()
    }

    /**
     * Extract base/location names from text
     */
    public fun extractLocations(text: String): List<String> {
        val locations = mutableSetOf<String>()

        BASE_PATTERNS.forEach { pattern ->
            pattern.findAll(text).forEach { match ->
                locations.add(match.groupValues[1])
            }
        }

        CASTLE_PATTERN.findAll(text).forEach { match ->
            locations.add(match.groupValues[1])
        }

        return locations.toList()
    }

    /**
     * Extract faction names from text
     */
    public fun extractFactions(text: String): List<String> {
        val factions = mutableListOf<String>()

        FACTION_PATTERN.findAll(text).forEach { match ->
            factions.add(match.groupValues[1])
        }

        return factions
    }

    /**
     * Extract coordinates from text
     */
    public fun extractCoordinates(text: String): List<Triple<Int, Int, Int>> {
        val coordinates = mutableListOf<Triple<Int, Int, Int>>()

        COORDINATE_PATTERN.findAll(text).forEach { match ->
            val x = match.groupValues[1].toIntOrNull() ?: 0
            val y = match.groupValues[2].toIntOrNull() ?: 0
            val z = match.groupValues[3].toIntOrNull() ?: 0
            coordinates.add(Triple(x, y, z))
        }

        return coordinates
    }

    /**
     * Check if text describes an attack or conflict
     */
    public fun isConflictEvent(text: String): Boolean {
        return ATTACK_PATTERN.containsMatchIn(text) ||
            text.contains("war", ignoreCase = true) ||
            text.contains("raid", ignoreCase = true) ||
            text.contains("stole", ignoreCase = true)
    }

    /**
     * Check if text describes an alliance or cooperation
     */
    public fun isAllianceEvent(text: String): Boolean {
        return ALLIANCE_PATTERN.containsMatchIn(text) ||
            text.contains("allied", ignoreCase = true) ||
            text.contains("cooperation", ignoreCase = true)
    }

    /**
     * Check if text describes building or construction
     */
    public fun isBuildingEvent(text: String): Boolean {
        return BUILD_PATTERN.containsMatchIn(text) ||
            text.contains("built", ignoreCase = true) ||
            text.contains("constructed", ignoreCase = true) ||
            text.contains("created", ignoreCase = true)
    }

    // Pattern definitions - these can be used by the conversation ingestion system
    private val PLAYER_PATTERN = Regex("""(?:Player\s+)(\w+)""", RegexOption.IGNORE_CASE)
    private val BASE_PATTERNS = listOf(
        Regex("""(\w+(?:\s+\w+)*)\s+(?:base|fortress|castle|outpost)""", RegexOption.IGNORE_CASE),
        Regex("""(?:base|fortress|castle|outpost)\s+(?:called|named)\s+(\w+(?:\s+\w+)*)""", RegexOption.IGNORE_CASE)
    )
    private val FACTION_PATTERN = Regex("""(\w+(?:\s+\w+)*)\s+(?:faction|team|clan|guild)""", RegexOption.IGNORE_CASE)
    private val CASTLE_PATTERN = Regex("""(?:castle|fortress)\s+called\s+(\w+(?:\s+\w+)*)""", RegexOption.IGNORE_CASE)
    private val COORDINATE_PATTERN = Regex("""(?:at|coordinates?|coords?|pos)\s*:?\s*\(?(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)\)?""", RegexOption.IGNORE_CASE)

    private val OWNERSHIP_PATTERN = Regex("""(\w+)\s+owns?\s+(?:the\s+)?(.+?)(?:\.|,|$)""", RegexOption.IGNORE_CASE)
    private val ALLIANCE_PATTERN = Regex("""(\w+(?:\s+\w+)*)\s+(?:allied?\s+with|allies?\s+of)\s+(\w+(?:\s+\w+)*)""", RegexOption.IGNORE_CASE)
    private val ATTACK_PATTERN = Regex("""(\w+)\s+attacked?\s+(.+?)(?:\.|,|$)""", RegexOption.IGNORE_CASE)
    private val BUILD_PATTERN = Regex("""(\w+)\s+built\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|,|$)""", RegexOption.IGNORE_CASE)
    private val MEMBERSHIP_PATTERN = Regex("""(\w+)\s+(?:is\s+)?(?:member|part)\s+of\s+(?:the\s+)?(.+?)(?:\.|,|$)""", RegexOption.IGNORE_CASE)
}

/**
 * Example usage of Minecraft patterns with the new AgentMemory system
 */
public fun analyzeMinecraftEvent(eventText: String): String {
    val analysis = StringBuilder()
    analysis.append("Minecraft Event Analysis:\n")

    val players = MinecraftPatterns.extractPlayers(eventText)
    if (players.isNotEmpty()) {
        analysis.append("Players involved: ${players.joinToString(", ")}\n")
    }

    val locations = MinecraftPatterns.extractLocations(eventText)
    if (locations.isNotEmpty()) {
        analysis.append("Locations mentioned: ${locations.joinToString(", ")}\n")
    }

    val factions = MinecraftPatterns.extractFactions(eventText)
    if (factions.isNotEmpty()) {
        analysis.append("Factions involved: ${factions.joinToString(", ")}\n")
    }

    val coordinates = MinecraftPatterns.extractCoordinates(eventText)
    if (coordinates.isNotEmpty()) {
        analysis.append("Coordinates: ${coordinates.joinToString(", ") { "(${it.first}, ${it.second}, ${it.third})" }}\n")
    }

    when {
        MinecraftPatterns.isConflictEvent(eventText) -> analysis.append("Event Type: Conflict/Attack\n")
        MinecraftPatterns.isAllianceEvent(eventText) -> analysis.append("Event Type: Alliance/Cooperation\n")
        MinecraftPatterns.isBuildingEvent(eventText) -> analysis.append("Event Type: Construction/Building\n")
        else -> analysis.append("Event Type: General Activity\n")
    }

    return analysis.toString()
}
