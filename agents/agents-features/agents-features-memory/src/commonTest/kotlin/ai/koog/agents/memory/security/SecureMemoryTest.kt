package ai.koog.agents.memory.security

import ai.koog.agents.memory.model.*
import ai.koog.agents.memory.retrieval.RetrievalQuery
import ai.koog.agents.memory.retrieval.RetrievalFilters
import kotlin.test.*

class SecureMemoryTest {
    
    @Test
    fun testSecurityContextFiltersScopes() {
        val context = SecurityContext(
            agentId = "player123",
            groups = setOf("red_faction"),
            roles = setOf("leader")
        )
        
        val query = RetrievalQuery("test query")
        val secureQuery = query.withSecurityFilters(context)
        
        // Should include agent's own scope and group scope
        val allowedScopes = secureQuery.filters.scopes
        assertTrue(allowedScopes.contains(MemoryScope.Agent("player123")))
        assertTrue(allowedScopes.contains(MemoryScope.CrossProduct))
        
        // Should include private scope for the agent
        val privateScope = allowedScopes.find { 
            it is MemoryScope.Secure.Private && it.owners.contains("player123") 
        }
        assertNotNull(privateScope)
        
        // Should include group scope
        val groupScope = allowedScopes.find {
            it is MemoryScope.Secure.Group && it.groupId == "red_faction"
        }
        assertNotNull(groupScope)
    }
    
    @Test
    fun testPrivateMemoryAccess() {
        val player1Context = SecurityContext(agentId = "player1")
        val player2Context = SecurityContext(agentId = "player2")
        
        // Private scope between player1 and companion
        val privateScope = MemoryScope.Secure.Private(
            owners = setOf("player1", "companion1")
        )
        
        // player1 should have access
        assertTrue(player1Context.canAccess(privateScope))
        
        // player2 should not have access
        assertFalse(player2Context.canAccess(privateScope))
    }
    
    @Test
    fun testGroupMemoryAccess() {
        val redLeaderContext = SecurityContext(
            agentId = "player1",
            groups = setOf("red_faction"),
            roles = setOf("leader")
        )
        
        val redMemberContext = SecurityContext(
            agentId = "player2", 
            groups = setOf("red_faction"),
            roles = setOf("member")
        )
        
        val blueContext = SecurityContext(
            agentId = "player3",
            groups = setOf("blue_faction")
        )
        
        // Group scope requiring leader role
        val leaderOnlyScope = MemoryScope.Secure.Group(
            groupId = "red_faction",
            requiredRoles = setOf("leader")
        )
        
        // General group scope
        val generalScope = MemoryScope.Secure.Group(
            groupId = "red_faction"
        )
        
        // Red leader should access both
        assertTrue(redLeaderContext.canAccess(leaderOnlyScope))
        assertTrue(redLeaderContext.canAccess(generalScope))
        
        // Red member should only access general
        assertFalse(redMemberContext.canAccess(leaderOnlyScope))
        assertTrue(redMemberContext.canAccess(generalScope))
        
        // Blue player should access neither
        assertFalse(blueContext.canAccess(leaderOnlyScope))
        assertFalse(blueContext.canAccess(generalScope))
    }
    
    @Test
    fun testHierarchicalMemoryAccess() {
        val adminContext = SecurityContext(
            agentId = "admin1",
            organizationId = "company_a",
            accessLevel = AccessLevel.RESTRICTED
        )
        
        val employeeContext = SecurityContext(
            agentId = "employee1",
            organizationId = "company_a", 
            accessLevel = AccessLevel.INTERNAL
        )
        
        val externalContext = SecurityContext(
            agentId = "external1",
            organizationId = "company_b",
            accessLevel = AccessLevel.PUBLIC
        )
        
        val confidentialScope = MemoryScope.Secure.Hierarchical(
            organizationId = "company_a",
            accessLevel = AccessLevel.CONFIDENTIAL
        )
        
        val internalScope = MemoryScope.Secure.Hierarchical(
            organizationId = "company_a",
            accessLevel = AccessLevel.INTERNAL
        )
        
        // Admin should access both
        assertTrue(adminContext.canAccess(confidentialScope))
        assertTrue(adminContext.canAccess(internalScope))
        
        // Employee should only access internal
        assertFalse(employeeContext.canAccess(confidentialScope))
        assertTrue(employeeContext.canAccess(internalScope))
        
        // External should access neither (wrong org)
        assertFalse(externalContext.canAccess(confidentialScope))
        assertFalse(externalContext.canAccess(internalScope))
    }
    
    @Test
    fun testBasicEncryption() {
        val encryptor = BasicMemoryEncryptor()
        val owners = setOf("player1", "companion1")
        val context = SecurityContext(agentId = "player1")
        
        val originalContent = "Secret base at coordinates -100, 64, 200"
        
        // Test encryption
        val encrypted = kotlinx.coroutines.runBlocking {
            encryptor.encrypt(originalContent, owners)
        }
        
        assertTrue(encrypted.startsWith("ENCRYPTED:"))
        assertNotEquals(originalContent, encrypted)
        
        // Test decryption
        val decrypted = kotlinx.coroutines.runBlocking {
            encryptor.decrypt(encrypted, context)
        }
        
        assertEquals(originalContent, decrypted)
    }
    
    @Test
    fun testCommonSecurityContexts() {
        val playerContext = CommonSecurityContexts.player(
            playerId = "player123",
            factionId = "red_faction",
            role = "leader"
        )
        
        assertEquals("player123", playerContext.agentId)
        assertTrue(playerContext.groups.contains("red_faction"))
        assertTrue(playerContext.roles.contains("leader"))
        
        val companionContext = CommonSecurityContexts.companion(
            companionId = "companion456",
            playerId = "player123",
            factionId = "red_faction"
        )
        
        assertEquals("companion456", companionContext.agentId)
        assertTrue(companionContext.groups.contains("red_faction"))
        assertTrue(companionContext.roles.contains("companion"))
        
        val adminContext = CommonSecurityContexts.admin("admin789")
        assertEquals("admin789", adminContext.agentId)
        assertEquals(AccessLevel.RESTRICTED, adminContext.accessLevel)
        assertTrue(adminContext.roles.contains("admin"))
    }
    
    @Test
    fun testSecurityIntegrationUtilities() {
        val context = SecurityContext(
            agentId = "player1",
            groups = setOf("red_faction"),
            roles = setOf("leader")
        )
        
        // Test allowed scopes utility
        val allowedScopes = SecurityIntegration.getAllowedScopes(context)
        assertTrue(allowedScopes.contains(MemoryScope.Agent("player1")))
        assertTrue(allowedScopes.contains(MemoryScope.CrossProduct))
        
        // Test canAccess utility
        val privateScope = MemoryScope.Secure.Private(owners = setOf("player1"))
        assertTrue(SecurityIntegration.canAccess(context, privateScope))
        
        val otherPrivateScope = MemoryScope.Secure.Private(owners = setOf("player2"))
        assertFalse(SecurityIntegration.canAccess(context, otherPrivateScope))
        
        val groupScope = MemoryScope.Secure.Group(groupId = "red_faction")
        assertTrue(SecurityIntegration.canAccess(context, groupScope))
        
        val otherGroupScope = MemoryScope.Secure.Group(groupId = "blue_faction")
        assertFalse(SecurityIntegration.canAccess(context, otherGroupScope))
    }
    
    @Test
    fun testBuildAllowedScopes() {
        val context = SecurityContext(
            agentId = "player1",
            groups = setOf("red_faction", "guild_warriors"),
            roles = setOf("leader", "member"),
            organizationId = "server_alpha",
            accessLevel = AccessLevel.CONFIDENTIAL
        )
        
        val allowedScopes = buildAllowedScopes(context)
        
        // Should include basic scopes
        assertTrue(allowedScopes.contains(MemoryScope.CrossProduct))
        assertTrue(allowedScopes.contains(MemoryScope.Agent("player1")))
        
        // Should include private scope
        val privateScope = allowedScopes.find {
            it is MemoryScope.Secure.Private && it.owners.contains("player1")
        }
        assertNotNull(privateScope)
        
        // Should include group scopes
        val redFactionScope = allowedScopes.find {
            it is MemoryScope.Secure.Group && it.groupId == "red_faction"
        }
        assertNotNull(redFactionScope)
        
        val guildScope = allowedScopes.find {
            it is MemoryScope.Secure.Group && it.groupId == "guild_warriors"
        }
        assertNotNull(guildScope)
        
        // Should include hierarchical scope
        val hierarchicalScope = allowedScopes.find {
            it is MemoryScope.Secure.Hierarchical && 
            it.organizationId == "server_alpha" &&
            it.accessLevel == AccessLevel.CONFIDENTIAL
        }
        assertNotNull(hierarchicalScope)
    }
}