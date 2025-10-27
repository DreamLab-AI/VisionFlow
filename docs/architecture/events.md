# Event-Driven Architecture

## Overview

VisionFlow v1.0.0 implements a comprehensive event-driven architecture to enable loose coupling between components, support event sourcing, and provide a foundation for distributed systems.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Application Layer                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Commands │  │ Queries  │  │ Services │  │   UI     │   │
│  └─────┬────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│        │            │              │             │          │
│        └────────────┴──────────────┴─────────────┘          │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         Event Bus                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Middleware Pipeline                     │    │
│  │  ┌───────────┐  ┌──────────┐  ┌──────────────┐     │    │
│  │  │Validation │→ │ Logging  │→ │   Metrics    │     │    │
│  │  └───────────┘  └──────────┘  └──────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           Event Handlers (Subscribers)              │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌────────────┐     │    │
│  │  │Graph │  │Onto- │  │Audit │  │Notification│     │    │
│  │  │Handler│ │logy  │  │Handler│ │  Handler   │     │    │
│  │  └──────┘  └──────┘  └──────┘  └────────────┘     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Event Store                            │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐        │
│  │   Events   │  │ Snapshots  │  │  Event Replay  │        │
│  └────────────┘  └────────────┘  └────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Domain Events

Domain events represent significant state changes in the system. All events implement the `DomainEvent` trait:

```rust
pub trait DomainEvent: Send + Sync + Clone + Debug {
    fn event_type(&self) -> &'static str;
    fn aggregate_id(&self) -> &str;
    fn timestamp(&self) -> DateTime<Utc>;
    fn aggregate_type(&self) -> &'static str;
    fn version(&self) -> u32 { 1 }
}
```

#### Event Categories

**Graph Events**
- `NodeAddedEvent` - Node created in graph
- `NodeUpdatedEvent` - Node properties modified
- `NodeRemovedEvent` - Node deleted
- `EdgeAddedEvent` - Edge created between nodes
- `EdgeRemovedEvent` - Edge deleted
- `GraphSavedEvent` - Graph persisted to storage
- `GraphClearedEvent` - All nodes/edges removed

**Ontology Events**
- `ClassAddedEvent` - OWL class added
- `PropertyAddedEvent` - Property (object/data/annotation) added
- `AxiomAddedEvent` - Axiom asserted
- `OntologyImportedEvent` - Ontology loaded from file
- `InferenceCompletedEvent` - Reasoning finished

**Physics Events**
- `SimulationStartedEvent` - Physics simulation began
- `SimulationStoppedEvent` - Simulation halted
- `LayoutOptimizedEvent` - Graph layout optimized
- `PositionsUpdatedEvent` - Node positions changed

**Settings Events**
- `SettingUpdatedEvent` - Configuration changed
- `PhysicsProfileSavedEvent` - Physics parameters saved
- `SettingsImportedEvent` - Settings loaded from file

### 2. Event Bus

The central pub/sub mechanism for routing events to handlers.

```rust
let bus = EventBus::new();

// Subscribe handlers
bus.subscribe(Arc::new(GraphEventHandler::new("graph"))).await;
bus.subscribe(Arc::new(AuditEventHandler::new("audit"))).await;

// Publish events
let event = NodeAddedEvent {
    node_id: "node-123".to_string(),
    label: "Example Node".to_string(),
    node_type: "Person".to_string(),
    properties: HashMap::new(),
    timestamp: Utc::now(),
};

bus.publish(event).await?;
```

#### Features

- **Async Processing**: All handlers execute asynchronously
- **Error Isolation**: Handler failures don't affect other handlers
- **Retry Logic**: Configurable retry with exponential backoff
- **Middleware Pipeline**: Cross-cutting concerns (logging, metrics, validation)
- **Enable/Disable**: Can be toggled at runtime

### 3. Event Handlers

Handlers process events and trigger side effects.

```rust
#[async_trait]
pub trait EventHandler: Send + Sync {
    fn event_type(&self) -> &'static str;
    fn handler_id(&self) -> &str;
    async fn handle(&self, event: &StoredEvent) -> Result<(), EventError>;
    fn max_retries(&self) -> u32 { 3 }
}
```

#### Built-in Handlers

**GraphEventHandler**
- Maintains in-memory graph cache
- Updates node/edge counts
- Invalidates stale data

```rust
let handler = GraphEventHandler::new("graph-handler");
bus.subscribe(Arc::new(handler)).await;
```

**OntologyEventHandler**
- Tracks ontology statistics
- Triggers inference on changes
- Monitors reasoning progress

```rust
let handler = OntologyEventHandler::new("ontology-handler");
bus.subscribe(Arc::new(handler)).await;
```

**AuditEventHandler**
- Logs all events for compliance
- Provides audit trail
- Supports querying by aggregate or type

```rust
let handler = AuditEventHandler::new("audit-handler");
let entries = handler.get_entries_for_aggregate("node-123").await;
```

**NotificationEventHandler**
- Creates WebSocket notifications
- Formats user-friendly messages
- Tracks delivery status

```rust
let handler = NotificationEventHandler::new("notification-handler");
let unsent = handler.get_unsent_notifications().await;
for notification in unsent {
    // Send via WebSocket
    handler.mark_sent(&notification.notification_id).await;
}
```

### 4. Middleware

Cross-cutting concerns applied to all events.

**LoggingMiddleware**
```rust
let middleware = LoggingMiddleware::new(true); // verbose
bus.add_middleware(Arc::new(middleware)).await;
```

**MetricsMiddleware**
```rust
let metrics = Arc::new(MetricsMiddleware::new());
bus.add_middleware(metrics.clone()).await;

// Query metrics
let count = metrics.get_published_count("NodeAdded").await;
let errors = metrics.get_error_count("handler-id").await;
```

**ValidationMiddleware**
```rust
let validation = ValidationMiddleware::new();
bus.add_middleware(Arc::new(validation)).await;
// Validates:
// - Non-empty aggregate IDs
// - Valid JSON data
// - Required metadata fields
```

**EnrichmentMiddleware**
```rust
let enrichment = EnrichmentMiddleware::new()
    .with_user_id("user-123".to_string())
    .with_correlation_id("corr-456".to_string());

bus.add_middleware(Arc::new(enrichment)).await;
```

### 5. Event Store

Persistent storage for event sourcing.

```rust
let repo = Arc::new(InMemoryEventRepository::new());
let store = EventStore::new(repo)
    .with_snapshot_threshold(100);

// Append events
store.append(&event).await?;

// Retrieve events
let events = store.get_events("node-123").await?;
let recent = store.get_events_after(sequence).await?;

// Replay for aggregate reconstruction
let replayed = store.replay_events("node-123").await?;
```

#### Repository Trait

```rust
#[async_trait]
pub trait EventRepository: Send + Sync {
    async fn append(&self, event: StoredEvent) -> EventResult<()>;
    async fn get_events(&self, aggregate_id: &str) -> EventResult<Vec<StoredEvent>>;
    async fn get_events_after(&self, sequence: i64) -> EventResult<Vec<StoredEvent>>;
    async fn get_events_by_type(&self, event_type: &str) -> EventResult<Vec<StoredEvent>>;
    async fn save_snapshot(&self, snapshot: EventSnapshot) -> EventResult<()>;
    async fn get_snapshot(&self, aggregate_id: &str) -> EventResult<Option<EventSnapshot>>;
}
```

Implementations:
- `InMemoryEventRepository` - For testing and development
- `SqliteEventRepository` - Persistent storage (TODO)
- `PostgresEventRepository` - Production-grade storage (TODO)

## Integration with CQRS

Commands publish events when they succeed:

```rust
#[async_trait]
impl CommandHandler<AddNodeCommand> for GraphCommandHandler {
    async fn handle(&self, cmd: AddNodeCommand) -> Result<()> {
        // Execute command
        self.graph_repo.add_node(cmd.node.clone()).await?;

        // Publish event
        self.event_bus.publish(NodeAddedEvent {
            node_id: cmd.node.id.clone(),
            label: cmd.node.label.clone(),
            node_type: cmd.node.node_type.clone(),
            properties: cmd.node.properties.clone(),
            timestamp: Utc::now(),
        }).await?;

        Ok(())
    }
}
```

Event handlers can trigger additional commands:

```rust
#[async_trait]
impl EventHandler for CacheInvalidationHandler {
    async fn handle(&self, event: &StoredEvent) -> EventResult<()> {
        match event.metadata.event_type.as_str() {
            "NodeAdded" | "NodeUpdated" => {
                // Trigger cache refresh command
                self.command_bus.send(RefreshCacheCommand {
                    cache_key: format!("node:{}", event.metadata.aggregate_id),
                }).await?;
            }
            _ => {}
        }
        Ok(())
    }
}
```

## Event Sourcing

Event sourcing reconstructs aggregate state from events:

```rust
pub struct NodeAggregate {
    pub id: String,
    pub label: String,
    pub node_type: String,
    pub properties: HashMap<String, String>,
    pub version: i64,
}

impl NodeAggregate {
    pub async fn load(id: &str, store: &EventStore) -> Result<Self> {
        let events = store.replay_events(id).await?;

        let mut aggregate = Self::default();
        for event in events {
            aggregate.apply(&event)?;
        }

        Ok(aggregate)
    }

    fn apply(&mut self, event: &StoredEvent) -> Result<()> {
        match event.metadata.event_type.as_str() {
            "NodeAdded" => {
                let data: NodeAddedEvent = serde_json::from_str(&event.data)?;
                self.id = data.node_id;
                self.label = data.label;
                self.node_type = data.node_type;
                self.properties = data.properties;
            }
            "NodeUpdated" => {
                let data: NodeUpdatedEvent = serde_json::from_str(&event.data)?;
                if let Some(label) = data.label {
                    self.label = label;
                }
                if let Some(props) = data.properties {
                    self.properties = props;
                }
            }
            _ => {}
        }
        self.version = event.sequence;
        Ok(())
    }
}
```

## Performance Optimizations

### Snapshots

Create snapshots every N events to speed up replay:

```rust
let store = EventStore::new(repo)
    .with_snapshot_threshold(100);

// After 100 events, create snapshot
let snapshot = EventSnapshot {
    aggregate_id: "node-123".to_string(),
    aggregate_type: "Node".to_string(),
    sequence: 100,
    timestamp: Utc::now(),
    state: serde_json::to_string(&aggregate)?,
};

store.save_snapshot(snapshot).await?;
```

### Async Handlers

Mark handlers as async for non-blocking execution:

```rust
impl EventHandler for NotificationHandler {
    fn is_async(&self) -> bool { true }

    async fn handle(&self, event: &StoredEvent) -> EventResult<()> {
        // Long-running operation won't block other handlers
        self.send_notification(event).await?;
        Ok(())
    }
}
```

### Event Batching

Publish multiple events atomically:

```rust
pub async fn batch_publish(&self, events: Vec<Box<dyn DomainEvent>>) -> EventResult<()> {
    for event in events {
        self.publish(event).await?;
    }
    Ok(())
}
```

## Testing

### Unit Tests

Test individual event handlers:

```rust
#[tokio::test]
async fn test_graph_handler() {
    let handler = GraphEventHandler::new("test");

    let event = NodeAddedEvent {
        node_id: "node-1".to_string(),
        label: "Test".to_string(),
        node_type: "Person".to_string(),
        properties: HashMap::new(),
        timestamp: Utc::now(),
    };

    let stored = StoredEvent {
        metadata: EventMetadata::new(
            event.node_id.clone(),
            "Node".to_string(),
            "NodeAdded".to_string(),
        ),
        data: serde_json::to_string(&event).unwrap(),
        sequence: 1,
    };

    handler.handle(&stored).await.unwrap();
    assert_eq!(handler.get_node_count().await, 1);
}
```

### Integration Tests

Test complete event flows:

```rust
#[tokio::test]
async fn test_complete_workflow() {
    let bus = EventBus::new();
    let store = EventStore::new(Arc::new(InMemoryEventRepository::new()));

    // Setup handlers
    bus.subscribe(Arc::new(GraphEventHandler::new("graph"))).await;
    bus.subscribe(Arc::new(AuditEventHandler::new("audit"))).await;

    // Execute workflow
    let event = NodeAddedEvent { /* ... */ };
    bus.publish(event.clone()).await.unwrap();
    store.append(&event).await.unwrap();

    // Verify
    let events = store.get_events("node-1").await.unwrap();
    assert_eq!(events.len(), 1);
}
```

## Best Practices

1. **Event Immutability**: Never modify events after creation
2. **Fine-Grained Events**: Prefer specific events over generic ones
3. **Idempotent Handlers**: Handlers should handle duplicate events gracefully
4. **Schema Versioning**: Use `version` field for event schema evolution
5. **Correlation IDs**: Track related events with `correlation_id`
6. **Causation IDs**: Link events to triggering commands/events
7. **Error Handling**: Handlers should catch and log errors, not crash
8. **Testing**: Write tests for all event flows

## Monitoring

Track event metrics:

```rust
let metrics = Arc::new(MetricsMiddleware::new());
bus.add_middleware(metrics.clone()).await;

// Monitor
println!("NodeAdded events: {}", metrics.get_published_count("NodeAdded").await);
println!("Handler errors: {}", metrics.get_error_count("graph-handler").await);
println!("Handler executions: {}", metrics.get_handler_count("audit-handler").await);
```

## Future Enhancements

- [ ] Distributed event bus (Kafka, NATS)
- [ ] Event replay UI for debugging
- [ ] Dead letter queue for failed events
- [ ] Event streaming to external systems
- [ ] CQRS projection materialization
- [ ] Saga orchestration
- [ ] Event versioning and upcasting
- [ ] Performance profiling dashboard

## References

- CQRS: `/docs/architecture/cqrs.md`
- Hexagonal Architecture: `/docs/architecture/hexagonal.md`
- Domain Model: `/docs/architecture/domain-model.md`
