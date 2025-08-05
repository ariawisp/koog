package ai.koog.agents.memory.security

import ai.koog.agents.memory.model.*
import ai.koog.agents.memory.providers.AgentMemoryProvider
import ai.koog.agents.memory.retrieval.RetrievalProvider
import ai.koog.agents.memory.retrieval.RetrievalQuery
import ai.koog.agents.memory.retrieval.RetrievalResult
import ai.koog.agents.memory.retrieval.RetrievalFilters

/**
 * Enhanced security integration with type safety and proper cryptography
 */
public object SecurityIntegration {
    
    /**
     * Creates a private memory between two entities using type-safe IDs
     */
    public suspend fun storePrivateMemory(
        memoryProvider: AgentMemoryProvider,
        concept: Concept,
        value: String,
        subject: MemorySubject,
        owners: Set<AgentId>,
        cryptography: MemoryCryptography = NoOpMemoryCryptography,
        encrypted: Boolean = true
    ) {
        val privateScope = MemoryScope.Secure.Private(owners = owners.map { it.value }.toSet(), encrypted = encrypted)
        val securityContext = SecurityContext(
            agentId = owners.first().value
        )
        
        // Encrypt the value if requested
        val processedValue = if (encrypted) {
            val encryptedContent = cryptography.encrypt(value, owners)
            encryptedContent.toJson()
        } else {
            value
        }
        
        val fact = SingleFact(
            concept = concept,
            timestamp = kotlinx.datetime.Clock.System.now().toEpochMilliseconds(),
            value = processedValue
        )
        
        memoryProvider.save(
            fact = fact,
            subject = subject,
            scope = privateScope,
            securityContext = securityContext
        )
    }
    
    /**
     * Creates a group memory visible to faction/team members using type-safe IDs
     */
    public suspend fun storeGroupMemory(
        memoryProvider: AgentMemoryProvider,
        concept: Concept,
        value: String,
        subject: MemorySubject,
        groupId: GroupId,
        requiredRoles: Set<Role> = emptySet(),
        creatorContext: SecurityContext
    ) {
        val groupScope = MemoryScope.Secure.Group(
            groupId = groupId.value,
            members = setOf(creatorContext.agentId),
            requiredRoles = requiredRoles.map { it.value }.toSet()
        )
        
        val fact = SingleFact(
            concept = concept,
            timestamp = kotlinx.datetime.Clock.System.now().toEpochMilliseconds(),
            value = value
        )
        
        memoryProvider.save(
            fact = fact,
            subject = subject,
            scope = groupScope,
            securityContext = creatorContext
        )
    }
    
    /**
     * Performs a secure retrieval with type-safe context and policy
     */
    public suspend fun secureRetrieve(
        retrievalProvider: RetrievalProvider,
        query: RetrievalQuery,
        context: SecurityContext,
        action: SecurityAction = SecurityAction.Read,
        policy: SecurityPolicy = DefaultSecurityPolicy,
        cryptography: MemoryCryptography = NoOpMemoryCryptography
    ): List<RetrievalResult> {
        // Build allowed scopes using the policy
        val allowedScopes = SecurityUtils.buildAllowedScopes(context, policy, action)
        
        // Apply security filters to the query
        val secureQuery = query.copy(
            filters = query.filters.copy(
                scopes = query.filters.scopes + allowedScopes
            )
        )
        
        // Retrieve with security context
        val results = retrievalProvider.retrieve(secureQuery, context)
        
        // Decrypt any encrypted results
        return results.map { result ->
            if (result.content.startsWith("{\"ciphertext\":")) {
                try {
                    val encryptedContent = result.content.parseEncryptedContent()
                    val decrypted = cryptography.decrypt(encryptedContent, context)
                    if (decrypted != null) {
                        result.copy(content = decrypted)
                    } else {
                        result // Keep encrypted if can't decrypt
                    }
                } catch (e: Exception) {
                    result // Keep original if parsing fails
                }
            } else {
                result
            }
        }
    }
    
    /**
     * Checks if a security context can access a given memory scope with type safety
     */
    public fun canAccess(
        context: SecurityContext, 
        scope: MemoryScope,
        action: SecurityAction = SecurityAction.Read,
        policy: SecurityPolicy = DefaultSecurityPolicy
    ): Boolean {
        return SecurityUtils.canAccess(context, scope, action, policy)
    }
    
    /**
     * Builds allowed scopes for a security context (useful for debugging)
     */
    public fun getAllowedScopes(
        context: SecurityContext,
        action: SecurityAction = SecurityAction.Read,
        policy: SecurityPolicy = DefaultSecurityPolicy
    ): Set<MemoryScope> {
        return SecurityUtils.buildAllowedScopes(context, policy, action)
    }
}

/**
 * Common security contexts for typical scenarios using type-safe DSL
 */
public object CommonSecurityContexts {
    
    /**
     * Player context with faction membership
     */
    public fun player(
        playerId: String,
        factionId: String,
        role: String = "member"
    ): SecurityContext = securityContext {
        agent(playerId)
        group(factionId) {
            role(role)
        }
    }
    
    /**
     * Companion context linked to a player
     */
    public fun companion(
        companionId: String,
        playerId: String,
        factionId: String? = null
    ): SecurityContext = securityContext {
        agent(companionId)
        factionId?.let { 
            group(it) {
                role("companion")
            }
        }
    }
    
    /**
     * Admin context with elevated access
     */
    public fun admin(
        adminId: String,
        organizationId: String = "default"
    ): SecurityContext = securityContext {
        agent(adminId)
        organization(organizationId)
        accessLevel(AccessLevel.RESTRICTED)
        group("admins") {
            role("admin")
        }
    }
}

/**
 * Example usage patterns for secure memory operations with type safety
 */
public object SecurityExamples {
    
    /**
     * Minecraft faction example: Store a secret base location with encryption
     */
    public suspend fun storeSecretBase(
        memoryProvider: AgentMemoryProvider,
        playerId: String,
        companionId: String,
        baseLocation: String,
        cryptography: MemoryCryptography = AESMemoryCryptography()
    ) {
        SecurityIntegration.storePrivateMemory(
            memoryProvider = memoryProvider,
            concept = Concept(keyword = "secret_base", description = "Secret base location", factType = FactType.SINGLE),
            value = baseLocation,
            subject = MemorySubject.Everything,
            owners = setOf(AgentId(playerId), AgentId(companionId)),
            cryptography = cryptography,
            encrypted = true
        )
    }
    
    /**
     * Minecraft faction example: Store faction battle plans with role restrictions
     */
    public suspend fun storeBattlePlan(
        memoryProvider: AgentMemoryProvider,
        factionId: String,
        battlePlan: String,
        commanderContext: SecurityContext
    ) {
        SecurityIntegration.storeGroupMemory(
            memoryProvider = memoryProvider,
            concept = Concept(keyword = "battle_plan", description = "Battle strategy and plans", factType = FactType.SINGLE),
            value = battlePlan,
            subject = MemorySubject.Everything,
            groupId = GroupId(factionId),
            requiredRoles = setOf(Role("leader"), Role("general")),
            creatorContext = commanderContext
        )
    }
    
    /**
     * Enterprise example: Store department-specific information with hierarchical access
     */
    public suspend fun storeDepartmentInfo(
        memoryProvider: AgentMemoryProvider,
        departmentId: String,
        info: String,
        accessLevel: AccessLevel,
        employeeContext: SecurityContext
    ) {
        val hierarchicalScope = MemoryScope.Secure.Hierarchical(
            organizationId = departmentId,
            accessLevel = accessLevel
        )
        
        val fact = SingleFact(
            concept = Concept(keyword = "department_info", description = "Department information", factType = FactType.SINGLE),
            timestamp = kotlinx.datetime.Clock.System.now().toEpochMilliseconds(),
            value = info
        )
        
        memoryProvider.save(
            fact = fact,
            subject = MemorySubject.Everything,
            scope = hierarchicalScope,
            securityContext = employeeContext
        )
    }
    
    /**
     * Modern gaming example: Fleet-wide strategic planning
     */
    public suspend fun storeFleetStrategy(
        memoryProvider: AgentMemoryProvider,
        fleetId: String,
        strategy: String,
        admiralContext: SecurityContext,
        cryptography: MemoryCryptography = AESMemoryCryptography()
    ) {
        // Store as encrypted group memory visible only to fleet officers
        SecurityIntegration.storeGroupMemory(
            memoryProvider = memoryProvider,
            concept = Concept(keyword = "fleet_strategy", description = "Fleet strategic planning", factType = FactType.SINGLE),
            value = strategy,
            subject = MemorySubject.Everything,
            groupId = GroupId(fleetId),
            requiredRoles = setOf(Role("admiral"), Role("captain"), Role("officer")),
            creatorContext = admiralContext
        )
    }
    
    /**
     * Secure retrieval example with policy enforcement
     */
    public suspend fun retrieveSecrets(
        retrievalProvider: RetrievalProvider,
        query: RetrievalQuery,
        context: SecurityContext,
        cryptography: MemoryCryptography = AESMemoryCryptography()
    ): List<RetrievalResult> {
        // Custom policy that requires specific role for secret access
        val strictPolicy = SecurityPolicy { subject, scope, action ->
            when {
                scope is MemoryScope.Secure.Private -> 
                    subject.agentId in scope.owners
                scope is MemoryScope.Secure.Group && scope.groupId.contains("secret") ->
                    scope.requiredRoles.contains("trusted") && subject.roles.contains("trusted")
                else -> DefaultSecurityPolicy(subject, scope, action)
            }
        }
        
        return SecurityIntegration.secureRetrieve(
            retrievalProvider = retrievalProvider,
            query = query,
            context = context,
            action = SecurityAction.Read,
            policy = strictPolicy,
            cryptography = cryptography
        )
    }
}