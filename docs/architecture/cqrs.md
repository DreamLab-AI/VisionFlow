# CQRS Architecture Documentation

## Overview

The CQRS (Command Query Responsibility Segregation) layer provides a clean separation between write operations (Commands) and read operations (Queries) in the VisionFlow application. This pattern improves code organization, testability, and maintainability by establishing clear boundaries between different types of operations.

## Architecture

```
┌─────────────────┐
│  API Handlers   │
└────────┬────────┘
         │
    ┌────▼──────┐
    │  Command  │
    │  /Query   │
    │   Bus     │
    └────┬──────┘
         │
    ┌────▼──────────┐
    │  Middleware   │
    │  Pipeline     │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │   Handlers    │
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │ Repositories  │
    │  /Adapters    │
    └───────────────┘
```

## Core Concepts

### Commands

Commands represent **write operations** that modify system state. They are:
- **Immutable**: Once created, their data cannot be changed
- **Validated**: Checked for correctness before execution
- **Audited**: Logged for compliance and debugging
- **Transactional**: Can be wrapped in database transactions

Example:
```rust
#[derive(Debug, Clone)]
pub struct AddNodeCommand {
    pub node: Node,
}

impl Command for AddNodeCommand {
    type Result = u32; // Returns node ID

    fn name(&self) -> &'static str {
        "AddNode"
    }

    fn validate(&self) -> Result<()> {
        if self.node.label.is_empty() {
            return Err(anyhow::anyhow!("Node label cannot be empty"));
        }
        Ok(())
    }
}
```

### Queries

Queries represent **read operations** that do not modify state. They are:
- **Side-effect free**: Never change system state
- **Cacheable**: Results can be cached for performance
- **Optimizable**: Can use read-optimized data models
- **Parallelizable**: Can run concurrently

Example:
```rust
#[derive(Debug, Clone)]
pub struct GetNodeQuery {
    pub node_id: u32,
}

impl Query for GetNodeQuery {
    type Result = Option<Node>;

    fn name(&self) -> &'static str {
        "GetNode"
    }
}
```

### Handlers

Handlers implement the business logic for executing commands and queries. They:
- **Delegate to repositories**: Use repository interfaces
- **Validate input**: Check command/query validity
- **Handle errors**: Convert repository errors to CQRS errors
- **Log operations**: Record execution for auditing

Example:
```rust
pub struct GraphCommandHandler {
    repository: Arc<dyn KnowledgeGraphRepository>,
}

#[async_trait]
impl CommandHandler<AddNodeCommand> for GraphCommandHandler {
    async fn handle(&self, command: AddNodeCommand) -> Result<u32> {
        command.validate()?;
        Ok(self.repository.add_node(&command.node).await?)
    }
}
```

### Command/Query Bus

The bus routes commands/queries to their handlers:
- **Type-safe routing**: Uses Rust's type system
- **Middleware support**: Cross-cutting concerns
- **Error handling**: Consistent error propagation
- **Metrics**: Built-in execution tracking

## Command Catalog

### Knowledge Graph Commands

| Command | Description | Returns | Validation |
|---------|-------------|---------|------------|
| `AddNodeCommand` | Add single node | `u32` (node ID) | Label not empty |
| `AddNodesCommand` | Batch add nodes | `Vec<u32>` | At least one node |
| `UpdateNodeCommand` | Update node | `()` | Label not empty |
| `UpdateNodesCommand` | Batch update | `()` | At least one node |
| `RemoveNodeCommand` | Remove node | `()` | - |
| `RemoveNodesCommand` | Batch remove | `()` | At least one ID |
| `AddEdgeCommand` | Add edge | `String` (edge ID) | Label not empty |
| `AddEdgesCommand` | Batch add edges | `Vec<String>` | At least one edge |
| `UpdateEdgeCommand` | Update edge | `()` | - |
| `RemoveEdgeCommand` | Remove edge | `()` | ID not empty |
| `RemoveEdgesCommand` | Batch remove | `()` | At least one ID |
| `SaveGraphCommand` | Save graph | `()` | - |
| `ClearGraphCommand` | Clear all data | `()` | - |
| `UpdatePositionsCommand` | Update positions | `()` | Valid floats |

### Settings Commands

| Command | Description | Returns | Validation |
|---------|-------------|---------|------------|
| `UpdateSettingCommand` | Update setting | `()` | Key not empty |
| `UpdateBatchSettingsCommand` | Batch update | `()` | At least one setting |
| `DeleteSettingCommand` | Delete setting | `()` | Key not empty |
| `SaveAllSettingsCommand` | Save all settings | `()` | - |
| `SavePhysicsSettingsCommand` | Save physics profile | `()` | Valid parameters |
| `DeletePhysicsProfileCommand` | Delete profile | `()` | Name not empty |
| `ImportSettingsCommand` | Import from JSON | `()` | Valid JSON object |
| `ClearSettingsCacheCommand` | Clear cache | `()` | - |

### Ontology Commands

| Command | Description | Returns | Validation |
|---------|-------------|---------|------------|
| `AddClassCommand` | Add OWL class | `String` (IRI) | IRI not empty |
| `UpdateClassCommand` | Update class | `()` | IRI not empty |
| `RemoveClassCommand` | Remove class | `()` | IRI not empty |
| `AddPropertyCommand` | Add property | `String` (IRI) | IRI not empty |
| `UpdatePropertyCommand` | Update property | `()` | IRI not empty |
| `RemovePropertyCommand` | Remove property | `()` | IRI not empty |
| `AddAxiomCommand` | Add axiom | `u64` (axiom ID) | Subject/object valid |
| `RemoveAxiomCommand` | Remove axiom | `()` | - |
| `SaveOntologyCommand` | Save ontology | `()` | - |
| `SaveOntologyGraphCommand` | Save graph | `()` | - |
| `StoreInferenceResultsCommand` | Store inference | `()` | - |
| `ImportOntologyCommand` | Import OWL XML | `()` | Valid XML |
| `CacheSsspResultCommand` | Cache pathfinding | `()` | - |
| `CacheApspResultCommand` | Cache distances | `()` | Non-empty matrix |
| `InvalidatePathfindingCachesCommand` | Clear caches | `()` | - |

### GPU Physics Commands

| Command | Description | Returns | Validation |
|---------|-------------|---------|------------|
| `InitializePhysicsCommand` | Initialize GPU | `()` | Valid parameters |
| `UpdatePhysicsParametersCommand` | Update params | `()` | Valid parameters |
| `UpdateGraphDataCommand` | Update graph | `()` | - |
| `ApplyExternalForcesCommand` | Apply forces | `()` | Valid floats |
| `PinNodesCommand` | Pin nodes | `()` | Valid positions |
| `UnpinNodesCommand` | Unpin nodes | `()` | At least one ID |
| `ResetPhysicsCommand` | Reset simulation | `()` | - |
| `CleanupPhysicsCommand` | Cleanup GPU | `()` | - |

## Query Catalog

### Knowledge Graph Queries

| Query | Description | Returns |
|-------|-------------|---------|
| `GetNodeQuery` | Get node by ID | `Option<Node>` |
| `GetNodesQuery` | Get multiple nodes | `Vec<Node>` |
| `GetAllNodesQuery` | Get all nodes | `Vec<Node>` |
| `SearchNodesQuery` | Search by label | `Vec<Node>` |
| `GetNodesByMetadataQuery` | Get by metadata | `Vec<Node>` |
| `GetNodeEdgesQuery` | Get node edges | `Vec<Edge>` |
| `GetEdgesBetweenQuery` | Get edges between | `Vec<Edge>` |
| `GetNeighborsQuery` | Get neighbors | `Vec<Node>` |
| `CountNodesQuery` | Count nodes | `usize` |
| `CountEdgesQuery` | Count edges | `usize` |
| `GetGraphStatsQuery` | Get statistics | `GraphStatistics` |
| `LoadGraphQuery` | Load graph | `Arc<GraphData>` |
| `QueryNodesQuery` | Query by properties | `Vec<Node>` |
| `GraphHealthCheckQuery` | Check health | `bool` |

### Settings Queries

| Query | Description | Returns |
|-------|-------------|---------|
| `GetSettingQuery` | Get setting | `Option<SettingValue>` |
| `GetBatchSettingsQuery` | Get multiple | `HashMap<String, SettingValue>` |
| `GetAllSettingsQuery` | Get all settings | `Option<AppFullSettings>` |
| `ListSettingsQuery` | List keys | `Vec<String>` |
| `HasSettingQuery` | Check existence | `bool` |
| `GetPhysicsSettingsQuery` | Get physics | `PhysicsSettings` |
| `ListPhysicsProfilesQuery` | List profiles | `Vec<String>` |
| `ExportSettingsQuery` | Export JSON | `serde_json::Value` |
| `SettingsHealthCheckQuery` | Check health | `bool` |

### Ontology Queries

| Query | Description | Returns |
|-------|-------------|---------|
| `GetClassQuery` | Get class | `Option<OwlClass>` |
| `ListClassesQuery` | List classes | `Vec<OwlClass>` |
| `GetClassHierarchyQuery` | Get hierarchy | `Vec<OwlClass>` |
| `GetPropertyQuery` | Get property | `Option<OwlProperty>` |
| `ListPropertiesQuery` | List properties | `Vec<OwlProperty>` |
| `GetAxiomsForClassQuery` | Get axioms | `Vec<OwlAxiom>` |
| `GetInferenceResultsQuery` | Get inference | `Option<InferenceResults>` |
| `ValidateOntologyQuery` | Validate | `ValidationReport` |
| `QueryOntologyQuery` | SPARQL query | `Vec<HashMap<String, String>>` |
| `GetOntologyMetricsQuery` | Get metrics | `OntologyMetrics` |
| `LoadOntologyGraphQuery` | Load graph | `Arc<GraphData>` |
| `ExportOntologyQuery` | Export OWL XML | `String` |
| `GetCachedSsspQuery` | Get cached path | `Option<PathfindingCacheEntry>` |
| `GetCachedApspQuery` | Get cached matrix | `Option<Vec<Vec<f32>>>` |

### GPU Physics Queries

| Query | Description | Returns |
|-------|-------------|---------|
| `GetGpuStatusQuery` | Get GPU status | `GpuDeviceInfo` |
| `GetPhysicsStatisticsQuery` | Get statistics | `PhysicsStatistics` |
| `ListGpuDevicesQuery` | List devices | `Vec<GpuDeviceInfo>` |
| `GetPerformanceMetricsQuery` | Get metrics | `PerformanceMetrics` |
| `IsGpuAvailableQuery` | Check availability | `bool` |

## Usage Examples

### Basic Command Execution

```rust
use visionflow::cqrs::{CommandBus, commands::AddNodeCommand};
use visionflow::models::node::Node;

// Create command bus and register handlers
let command_bus = CommandBus::new();
command_bus.register(Box::new(GraphCommandHandler::new(repo))).await;

// Create and execute command
let mut node = Node::default();
node.label = "My Node".to_string();
let command = AddNodeCommand { node };
let node_id = command_bus.execute(command).await?;
```

### Basic Query Execution

```rust
use visionflow::cqrs::{QueryBus, queries::GetNodeQuery};

// Create query bus and register handlers
let query_bus = QueryBus::new();
query_bus.register(Box::new(GraphQueryHandler::new(repo))).await;

// Create and execute query
let query = GetNodeQuery { node_id: 1 };
let node = query_bus.execute(query).await?;
```

### Batch Operations

```rust
// Batch add nodes
let nodes = vec![node1, node2, node3];
let command = AddNodesCommand { nodes };
let node_ids = command_bus.execute(command).await?;

// Batch update positions (for physics)
let positions = vec![
    (1, 10.0, 20.0, 30.0),
    (2, 15.0, 25.0, 35.0),
];
let command = UpdatePositionsCommand { positions };
command_bus.execute(command).await?;
```

### Using Middleware

```rust
use visionflow::cqrs::types::LoggingMiddleware;
use visionflow::cqrs::bus::MetricsMiddleware;

// Create bus with middleware
let logging = Box::new(LoggingMiddleware);
let metrics = Box::new(MetricsMiddleware::new());
let command_bus = CommandBus::with_middleware(vec![logging, metrics]);

// Commands will now be logged and metrics tracked
```

### Settings Management

```rust
use visionflow::cqrs::commands::UpdateSettingCommand;
use visionflow::ports::settings_repository::SettingValue;

// Update a setting
let command = UpdateSettingCommand {
    key: "app.theme".to_string(),
    value: SettingValue::String("dark".to_string()),
    description: Some("Application theme".to_string()),
};
command_bus.execute(command).await?;

// Get the setting
let query = GetSettingQuery {
    key: "app.theme".to_string(),
};
let value = query_bus.execute(query).await?;
```

### Physics Operations

```rust
use visionflow::cqrs::commands::InitializePhysicsCommand;
use visionflow::ports::gpu_physics_adapter::PhysicsParameters;

// Initialize physics simulation
let command = InitializePhysicsCommand {
    graph: graph_arc,
    params: PhysicsParameters::default(),
};
command_bus.execute(command).await?;

// Get GPU status
let query = GetGpuStatusQuery;
let gpu_info = query_bus.execute(query).await?;
```

## Error Handling

All commands and queries return `Result<T>` using `anyhow::Error`:

```rust
match command_bus.execute(command).await {
    Ok(result) => {
        // Handle success
    }
    Err(e) => {
        // Handle error
        tracing::error!("Command failed: {}", e);
    }
}
```

### Common Error Scenarios

1. **Validation Errors**: Command/query data is invalid
2. **Handler Not Found**: No handler registered for type
3. **Repository Errors**: Database/adapter failures
4. **Middleware Errors**: Cross-cutting concern failures

## Testing Strategies

### Unit Testing Commands

```rust
#[test]
fn test_command_validation() {
    let mut node = Node::default();
    node.label = "".to_string(); // Invalid

    let command = AddNodeCommand { node };
    assert!(command.validate().is_err());
}
```

### Integration Testing

```rust
#[tokio::test]
async fn test_add_and_get_node() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    // Register handlers
    command_bus.register(Box::new(GraphCommandHandler::new(repo.clone()))).await;
    query_bus.register(Box::new(GraphQueryHandler::new(repo))).await;

    // Execute command
    let node_id = command_bus.execute(AddNodeCommand { node }).await?;

    // Verify with query
    let result = query_bus.execute(GetNodeQuery { node_id }).await?;
    assert!(result.is_some());

    Ok(())
}
```

### Mocking Repositories

```rust
struct MockRepository;

#[async_trait]
impl KnowledgeGraphRepository for MockRepository {
    async fn add_node(&self, node: &Node) -> Result<u32> {
        Ok(1) // Mock implementation
    }
    // ... other methods
}

#[tokio::test]
async fn test_with_mock() {
    let repo = Arc::new(MockRepository);
    let handler = GraphCommandHandler::new(repo);
    // Test handler behavior
}
```

## Performance Considerations

### Batching

Always prefer batch operations for multiple items:

```rust
// ❌ Bad: Multiple individual commands
for node in nodes {
    command_bus.execute(AddNodeCommand { node }).await?;
}

// ✅ Good: Single batch command
command_bus.execute(AddNodesCommand { nodes }).await?;
```

### Query Optimization

Use specific queries instead of loading all data:

```rust
// ❌ Bad: Load all then filter in memory
let all_nodes = query_bus.execute(GetAllNodesQuery).await?;
let filtered = all_nodes.iter().filter(|n| n.label.contains("test"));

// ✅ Good: Use search query
let nodes = query_bus.execute(SearchNodesQuery {
    label_pattern: "test".to_string(),
}).await?;
```

### Caching

Use query result caching for read-heavy operations:

```rust
// Implement caching middleware
pub struct CachingMiddleware {
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
}
```

## Migration Guide

### From Direct Repository Access

**Before:**
```rust
let node_id = repository.add_node(&node).await?;
```

**After:**
```rust
let command = AddNodeCommand { node };
let node_id = command_bus.execute(command).await?;
```

### Benefits

1. **Testability**: Mock handlers instead of repositories
2. **Validation**: Centralized validation logic
3. **Auditing**: Automatic command logging
4. **Middleware**: Cross-cutting concerns (metrics, logging)
5. **Type Safety**: Compile-time command/query type checking

## Best Practices

1. **Command Naming**: Use verb-noun format (AddNode, UpdateSetting)
2. **Query Naming**: Use Get/List/Search prefix (GetNode, ListClasses)
3. **Validation**: Always validate in command, not handler
4. **Immutability**: Commands/queries should be immutable structs
5. **Single Responsibility**: One handler per command/query type
6. **Error Context**: Provide meaningful error messages
7. **Documentation**: Document expected behavior and validation rules

## Future Enhancements

1. **Event Sourcing**: Store commands as events for audit trail
2. **Saga Pattern**: Distributed transactions across services
3. **Caching Layer**: Automatic query result caching
4. **Retry Logic**: Automatic retry for transient failures
5. **Circuit Breaker**: Fault tolerance for repository failures
6. **Metrics Dashboard**: Real-time command/query monitoring
7. **Performance Profiling**: Identify slow handlers

## References

- [CQRS Pattern - Martin Fowler](https://martinfowler.com/bliki/CQRS.html)
- [Hexagonal Architecture](https://alistair.cockburn.us/hexagonal-architecture/)
- [Domain-Driven Design](https://www.domainlanguage.com/ddd/)
