# SQL Persistence Providers

This module provides SQL-based persistence implementations for agent checkpoints using JetBrains Exposed ORM framework.

## Overview

The SQL persistence providers offer a scalable and reliable way to persist agent checkpoints to relational databases. The implementation leverages Exposed's type-safe SQL DSL and supports multiple database backends.

## Architecture

### Abstract Base Classes

1. **SQLPersistencyStorageProvider** (commonMain)
   - Platform-agnostic abstraction for SQL persistence
   - Defines the contract for SQL-based checkpoint storage
   - Handles persistence ID validation and TTL calculations

2. **ExposedPersistencyStorageProvider** (jvmMain)
   - JVM-specific implementation using JetBrains Exposed ORM
   - Provides transaction management and schema initialization
   - Implements the core persistence operations using Exposed's DSL
   - Database-specific optimizations based on dialect detection

### Concrete Implementations

1. **PostgresPersistencyStorageProvider**
   - PostgreSQL-specific implementation with optimized features
   - Connection pooling with HikariCP
   - Multiple connection options: JDBC URL, HikariCP, or external DataSource

2. **MySQLPersistencyStorageProvider**
   - MySQL/MariaDB implementation (5.7+ recommended for JSON support)
   - Connection pooling with HikariCP
   - Compatible with MySQL replication setups

3. **H2PersistencyStorageProvider**
   - Lightweight embedded database implementation
   - Supports in-memory, file-based, and server modes
   - PostgreSQL/MySQL compatibility modes available
   - Ideal for testing and embedded applications

4. **SQLitePersistencyStorageProvider**
   - Zero-configuration embedded database
   - File-based persistence with single-file deployment
   - Perfect for desktop and mobile applications

## Database Schema

The providers use a single table with the following structure:

```sql
CREATE TABLE agent_checkpoints (
    persistence_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    created_at BIGINT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    ttl_timestamp BIGINT,
    PRIMARY KEY (persistence_id, checkpoint_id)
);

CREATE INDEX idx_agent_checkpoints_created_at ON agent_checkpoints(created_at);
CREATE INDEX idx_agent_checkpoints_ttl ON agent_checkpoints(ttl_timestamp) WHERE ttl_timestamp IS NOT NULL;
```

## Usage

### PostgreSQL Provider

```kotlin
import ai.koog.agents.snapshot.providers.sql.PostgresPersistencyStorageProvider

// Using JDBC URL
val provider = PostgresPersistencyStorageProvider(
    persistenceId = "my-agent",
    jdbcUrl = "jdbc:postgresql://localhost:5432/mydb",
    username = "user",
    password = "password",
    ttlSeconds = 3600 // Optional: 1 hour TTL
)

// Using HikariCP for connection pooling
val hikariConfig = HikariConfig().apply {
    jdbcUrl = "jdbc:postgresql://localhost:5432/mydb"
    username = "user"
    password = "password"
    maximumPoolSize = 10
    minimumIdle = 2
}

val pooledProvider = PostgresPersistencyStorageProvider(
    persistenceId = "my-agent",
    hikariConfig = hikariConfig
)
```

### MySQL Provider

```kotlin
import ai.koog.agents.snapshot.providers.sql.MySQLPersistencyStorageProvider

val provider = MySQLPersistencyStorageProvider(
    persistenceId = "my-agent",
    jdbcUrl = "jdbc:mysql://localhost:3306/mydb?useSSL=false&serverTimezone=UTC",
    username = "user",
    password = "password"
)
```

### H2 Provider

```kotlin
import ai.koog.agents.snapshot.providers.sql.H2PersistencyStorageProvider

// In-memory database (perfect for testing)
val inMemoryProvider = H2PersistencyStorageProvider.inMemory(
    persistenceId = "test-agent",
    databaseName = "test"
)

// File-based database
val fileProvider = H2PersistencyStorageProvider.fileBased(
    persistenceId = "my-agent",
    filePath = "./data/agent-checkpoints"
)

// PostgreSQL compatibility mode
val pgCompatibleProvider = H2PersistencyStorageProvider.postgresCompatible(
    persistenceId = "my-agent"
)
```

### SQLite Provider

```kotlin
import ai.koog.agents.snapshot.providers.sql.SQLitePersistencyStorageProvider

// File-based SQLite
val provider = SQLitePersistencyStorageProvider.fileBased(
    persistenceId = "my-agent",
    filePath = "./data/agent.db"
)

// In-memory SQLite
val inMemoryProvider = SQLitePersistencyStorageProvider.inMemory(
    persistenceId = "test-agent"
)
```

### Common Operations

```kotlin
// Initialize schema (run once)
provider.initializeSchema()

// Save checkpoint
val checkpointData = AgentCheckpointData(
    checkpointId = "checkpoint-1",
    createdAt = Clock.System.now(),
    nodeId = "node-1",
    lastInput = JsonPrimitive("test"),
    messageHistory = listOf()
)
provider.saveCheckpoint(checkpointData)

// Retrieve checkpoints
val allCheckpoints = provider.getCheckpoints()
val latestCheckpoint = provider.getLatestCheckpoint()

// Delete specific checkpoint
provider.deleteCheckpoint("checkpoint-1")

// Clean up
provider.close() // Close connections when done
```

## Configuration

### Connection Pooling

For production use, connection pooling is recommended:

```kotlin
val hikariConfig = HikariConfig().apply {
    jdbcUrl = "jdbc:postgresql://localhost:5432/mydb"
    username = "user"
    password = "password"
    
    // Pool configuration
    maximumPoolSize = 20
    minimumIdle = 5
    idleTimeout = 600000 // 10 minutes
    connectionTimeout = 30000 // 30 seconds
    maxLifetime = 1800000 // 30 minutes
    
    // Connection test
    connectionTestQuery = "SELECT 1"
}
```

### TTL Support

Checkpoints can have an optional time-to-live (TTL):

```kotlin
val provider = PostgresPersistencyStorageProvider(
    persistenceId = "my-agent",
    jdbcUrl = "...",
    username = "...",
    password = "...",
    ttlSeconds = 86400 // 24 hours
)
```

Expired checkpoints are automatically cleaned up during read operations.

## Performance Considerations

1. **Indexing**: The schema includes indexes on `created_at` and `ttl_timestamp` for efficient querying
2. **Connection Pooling**: Use HikariCP for better performance under load
3. **Batch Operations**: The cleanup of expired checkpoints is performed in batch
4. **UPSERT Operations**: Checkpoints use upsert semantics for idempotent saves
5. **Dialect Optimizations**: Database-specific transaction isolation levels and query optimizations

## Production Deployment

### Recommended Setup

1. Use connection pooling with appropriate pool sizes
2. Enable connection validation and health checks
3. Configure proper transaction isolation levels
4. Set up database monitoring and alerting
5. Regular backups of checkpoint data

### High Availability

For HA deployments:
- Use database replication (master-slave or multi-master)
- Configure connection failover in the JDBC URL
- Implement retry logic for transient failures
- Consider using a connection pool with built-in HA support

## Examples

For complete working examples of all SQL providers, see the `examples/src/main/kotlin/ai/koog/agents/example/snapshot/sql` directory.

## Testing

The SQL providers include comprehensive test coverage using H2 in-memory database. Tests can be found in `agents-features-snapshot/src/jvmTest`.

## See Also

- [Redis Persistence Provider](../redis) - For distributed, ephemeral checkpoint storage
- [Persistence Examples](../../../../../../../../../examples/src/main/kotlin/ai/koog/agents/example/snapshot) - Complete working examples