package ai.koog.agents.memory.security

import ai.koog.agents.memory.model.AccessLevel
import ai.koog.agents.memory.model.MemoryScope
import ai.koog.agents.memory.retrieval.RetrievalQuery
import ai.koog.agents.memory.retrieval.RetrievalFilters
import kotlinx.serialization.Serializable

/**
 * Security context for memory operations
 * Contains authentication and authorization information for the current operation
 */
@Serializable
public data class SecurityContext(
    /** Unique identifier for the agent/user */
    val agentId: String,
    
    /** Groups/factions the agent belongs to */
    val groups: Set<String> = emptySet(),
    
    /** Roles the agent has (e.g., "leader", "admin", "member") */
    val roles: Set<String> = emptySet(),
    
    /** Organization context for hierarchical access */
    val organizationId: String? = null,
    
    /** Current access level */
    val accessLevel: AccessLevel = AccessLevel.PUBLIC
)

/**
 * Utility functions for security-aware operations
 */

/**
 * Applies security context to a retrieval query, automatically filtering
 * based on the agent's permissions. This is now primarily used internally
 * by retrieval providers since security context is passed directly.
 */
public fun RetrievalQuery.withSecurityFilters(context: SecurityContext): RetrievalQuery {
    val allowedScopes = buildAllowedScopes(context)
    
    return this.copy(
        filters = filters.copy(
            scopes = filters.scopes + allowedScopes
        )
    )
}

/**
 * Convenience method for creating security context from individual parameters
 */
public fun securityContext(
    agentId: String,
    groups: Set<String> = emptySet(),
    roles: Set<String> = emptySet(),
    organizationId: String? = null,
    accessLevel: AccessLevel = AccessLevel.PUBLIC
): SecurityContext {
    return SecurityContext(
        agentId = agentId,
        groups = groups,
        roles = roles,
        organizationId = organizationId,
        accessLevel = accessLevel
    )
}

/**
 * Builds the set of memory scopes that the given security context is allowed to access
 */
public fun buildAllowedScopes(context: SecurityContext): Set<MemoryScope> {
    return buildSet {
        // Always include public scopes
        add(MemoryScope.CrossProduct)
        
        // Agent's own scope
        add(MemoryScope.Agent(context.agentId))
        
        // Private scopes where agent is an owner
        add(MemoryScope.Secure.Private(owners = setOf(context.agentId)))
        
        // Group scopes for groups the agent belongs to
        context.groups.forEach { groupId ->
            add(MemoryScope.Secure.Group(
                groupId = groupId,
                members = setOf(context.agentId),
                requiredRoles = context.roles
            ))
        }
        
        // Hierarchical scopes based on organization and access level
        context.organizationId?.let { orgId ->
            add(MemoryScope.Secure.Hierarchical(
                organizationId = orgId,
                accessLevel = context.accessLevel
            ))
        }
    }
}

/**
 * Checks if a security context can access a given memory scope
 */
public fun SecurityContext.canAccess(scope: MemoryScope): Boolean {
    return when (scope) {
        is MemoryScope.CrossProduct -> true
        is MemoryScope.Agent -> scope.name == agentId
        is MemoryScope.Feature -> true // Features are generally accessible
        is MemoryScope.Product -> true // Products are generally accessible
        
        is MemoryScope.Secure.Private -> {
            agentId in scope.owners
        }
        
        is MemoryScope.Secure.Group -> {
            val inGroup = groups.contains(scope.groupId)
            val hasRequiredRole = scope.requiredRoles.isEmpty() || 
                                 scope.requiredRoles.any { it in roles }
            val isMember = scope.members.isEmpty() || agentId in scope.members
            
            inGroup && hasRequiredRole && isMember
        }
        
        is MemoryScope.Secure.Hierarchical -> {
            val sameOrg = organizationId == scope.organizationId
            val hasAccess = accessLevel.ordinal >= scope.accessLevel.ordinal
            
            sameOrg && hasAccess
        }
    }
}