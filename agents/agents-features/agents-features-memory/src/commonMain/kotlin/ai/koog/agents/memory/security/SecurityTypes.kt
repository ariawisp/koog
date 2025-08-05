package ai.koog.agents.memory.security

import ai.koog.agents.memory.model.AccessLevel
import ai.koog.agents.memory.model.MemoryScope
import kotlinx.serialization.Serializable

/**
 * Type-safe identifiers using value classes to prevent accidental mixing
 */
@JvmInline
@Serializable
public value class AgentId(public val value: String)

@JvmInline
@Serializable
public value class GroupId(public val value: String)

@JvmInline
@Serializable
public value class Role(public val value: String)

@JvmInline
@Serializable
public value class OrganizationId(public val value: String)

/**
 * Type-safe security actions using sealed interface
 */
public sealed interface SecurityAction {
    @Serializable
    public object Read : SecurityAction
    
    @Serializable
    public object Write : SecurityAction
    
    @Serializable
    public object Admin : SecurityAction
    
    @Serializable
    public object Delete : SecurityAction
}

/**
 * Group membership with explicit roles
 */
@Serializable
public data class GroupMembership(
    val group: GroupId,
    val roles: Set<Role> = emptySet()
)

/**
 * Extension functions to add type-safe operations to existing SecurityContext
 */
public val SecurityContext.agent: AgentId get() = AgentId(agentId)

public fun SecurityContext.hasRole(group: GroupId, role: Role): Boolean {
    return roles.contains(role.value) && groups.contains(group.value)
}

public fun SecurityContext.isMemberOf(group: GroupId): Boolean {
    return groups.contains(group.value)
}

/**
 * Enhanced secure memory scopes with type safety
 * These are simple data classes used for security logic but convert to MemoryScope when needed
 */
@Serializable
public data class SecurePrivateScope(
    val owners: Set<AgentId>,
    val encrypted: Boolean = true
)

@Serializable  
public data class SecureGroupScope(
    val group: GroupId,
    val requiredRoles: Set<Role> = emptySet()
)

@Serializable
public data class SecureHierarchicalScope(
    val organization: OrganizationId,
    val accessLevel: AccessLevel,
    val departments: Set<String> = emptySet()
)

/**
 * Policy function for access control decisions
 */
public fun interface SecurityPolicy {
    /**
     * Check if the given security context can perform the action on the scope
     */
    public operator fun invoke(
        subject: SecurityContext,
        scope: MemoryScope,
        action: SecurityAction
    ): Boolean
}

/**
 * Default security policy implementation
 */
public object DefaultSecurityPolicy : SecurityPolicy {
    override fun invoke(
        subject: SecurityContext,
        scope: MemoryScope,
        action: SecurityAction
    ): Boolean {
        return when (scope) {
            is MemoryScope.CrossProduct -> true
            is MemoryScope.Agent -> scope.name == subject.agentId
            is MemoryScope.Feature -> true // Features are generally accessible
            is MemoryScope.Product -> true // Products are generally accessible
            
            // Handle legacy secure scopes
            is MemoryScope.Secure.Private -> {
                subject.agentId in scope.owners
            }
            
            is MemoryScope.Secure.Group -> {
                val inGroup = subject.groups.contains(scope.groupId)
                val hasRequiredRole = scope.requiredRoles.isEmpty() || 
                                     scope.requiredRoles.any { it in subject.roles }
                val isMember = scope.members.isEmpty() || subject.agentId in scope.members
                
                inGroup && hasRequiredRole && isMember
            }
            
            is MemoryScope.Secure.Hierarchical -> {
                val sameOrg = subject.organizationId == scope.organizationId
                val hasAccess = subject.accessLevel.ordinal >= scope.accessLevel.ordinal
                
                sameOrg && hasAccess
            }
            
            // Fallback for any other types
            else -> true
        }
    }
}

/**
 * DSL for building security contexts
 */
public inline fun securityContext(block: SecurityContextBuilder.() -> Unit): SecurityContext =
    SecurityContextBuilder().apply(block).build()

public class SecurityContextBuilder {
    private var agentId: AgentId? = null
    private val memberships = mutableSetOf<GroupMembership>()
    private var organizationId: OrganizationId? = null
    private var accessLevel: AccessLevel = AccessLevel.PUBLIC
    
    public fun agent(id: String) {
        agentId = AgentId(id)
    }
    
    public fun organization(id: String) {
        organizationId = OrganizationId(id)
    }
    
    public fun accessLevel(level: AccessLevel) {
        accessLevel = level
    }
    
    public fun group(id: String, block: GroupBuilder.() -> Unit = {}) {
        val group = GroupBuilder(GroupId(id)).apply(block)
        memberships += GroupMembership(group.group, group.roles.toSet())
    }
    
    public fun build(): SecurityContext {
        val agentIdValue = requireNotNull(agentId) { "agent(...) is required" }.value
        val groupsSet = memberships.map { it.group.value }.toSet()
        val rolesSet = memberships.flatMap { it.roles }.map { it.value }.toSet()
        
        return SecurityContext(
            agentId = agentIdValue,
            groups = groupsSet,
            roles = rolesSet,
            organizationId = organizationId?.value,
            accessLevel = accessLevel
        )
    }
}

public class GroupBuilder(public val group: GroupId) {
    internal val roles = mutableSetOf<Role>()
    
    public fun role(name: String) {
        roles += Role(name)
    }
}

/**
 * Utility functions for common security operations
 */
public object SecurityUtils {
    /**
     * Build allowed scopes for a security context
     */
    public fun buildAllowedScopes(
        context: SecurityContext,
        policy: SecurityPolicy = DefaultSecurityPolicy,
        action: SecurityAction = SecurityAction.Read
    ): Set<MemoryScope> {
        return buildSet {
            // Always include public scopes from legacy MemoryScope
            add(MemoryScope.CrossProduct)
            
            // Agent's own scope
            add(MemoryScope.Agent(context.agentId))
            
            // Private scopes where agent is owner
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
        }.filter { scope -> policy(context, scope, action) }.toSet()
    }
    
    /**
     * Check if a security context can access a given memory scope
     */
    public fun canAccess(
        context: SecurityContext,
        scope: MemoryScope,
        action: SecurityAction = SecurityAction.Read,
        policy: SecurityPolicy = DefaultSecurityPolicy
    ): Boolean {
        return policy(context, scope, action)
    }
}