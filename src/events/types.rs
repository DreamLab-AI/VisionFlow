use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Core trait for all domain events
/// Note: This trait is object-safe (can be used as `dyn DomainEvent`)
pub trait DomainEvent: Send + Sync + Debug {
    /// Returns the type identifier of this event
    fn event_type(&self) -> &'static str;

    /// Returns the ID of the aggregate this event relates to
    fn aggregate_id(&self) -> &str;

    /// Returns when this event occurred
    fn timestamp(&self) -> DateTime<Utc>;

    /// Returns the aggregate type (e.g., "Graph", "Node", "Ontology")
    fn aggregate_type(&self) -> &'static str;

    /// Returns version for event schema evolution
    fn version(&self) -> u32 {
        1
    }

    /// Serializes the event to JSON (must be implemented by each event)
    fn to_json_string(&self) -> Result<String, serde_json::Error>;
}

/// Metadata associated with every event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventMetadata {
    /// Unique event identifier
    pub event_id: String,

    /// ID of the aggregate
    pub aggregate_id: String,

    /// Type of the aggregate
    pub aggregate_type: String,

    /// Type of the event
    pub event_type: String,

    /// When the event occurred
    pub timestamp: DateTime<Utc>,

    /// ID of the command/event that caused this event
    pub causation_id: Option<String>,

    /// Correlation ID for tracking related events
    pub correlation_id: Option<String>,

    /// User who triggered this event
    pub user_id: Option<String>,

    /// Event schema version
    pub version: u32,
}

impl EventMetadata {
    pub fn new(aggregate_id: String, aggregate_type: String, event_type: String) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            aggregate_id,
            aggregate_type,
            event_type,
            timestamp: Utc::now(),
            causation_id: None,
            correlation_id: None,
            user_id: None,
            version: 1,
        }
    }

    pub fn with_causation(mut self, causation_id: String) -> Self {
        self.causation_id = Some(causation_id);
        self
    }

    pub fn with_correlation(mut self, correlation_id: String) -> Self {
        self.correlation_id = Some(correlation_id);
        self
    }

    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }
}

/// Stored representation of an event for event sourcing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEvent {
    /// Event metadata
    pub metadata: EventMetadata,

    /// Serialized event data
    pub data: String,

    /// Sequence number for ordering
    pub sequence: i64,
}

/// Handler for domain events
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Returns the event type this handler processes
    fn event_type(&self) -> &'static str;

    /// Returns unique handler identifier
    fn handler_id(&self) -> &str;

    /// Handles the event
    async fn handle(&self, event: &StoredEvent) -> Result<(), EventError>;

    /// Optional: specify if handler should run asynchronously
    fn is_async(&self) -> bool {
        true
    }

    /// Optional: specify retry policy
    fn max_retries(&self) -> u32 {
        3
    }
}

/// Middleware for event processing pipeline
#[async_trait]
pub trait EventMiddleware: Send + Sync {
    /// Called before event is published
    async fn before_publish(&self, event: &mut StoredEvent) -> Result<(), EventError>;

    /// Called after event is published
    async fn after_publish(&self, event: &StoredEvent) -> Result<(), EventError>;

    /// Called before handler execution
    async fn before_handle(&self, event: &StoredEvent, handler_id: &str) -> Result<(), EventError>;

    /// Called after handler execution
    async fn after_handle(
        &self,
        event: &StoredEvent,
        handler_id: &str,
        result: &Result<(), EventError>,
    ) -> Result<(), EventError>;
}

/// Event-related errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum EventError {
    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Handler error: {0}")]
    Handler(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Bus error: {0}")]
    Bus(String),

    #[error("Middleware error: {0}")]
    Middleware(String),

    #[error("Event not found: {0}")]
    NotFound(String),

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Concurrency error: {0}")]
    Concurrency(String),
}

pub type EventResult<T> = Result<T, EventError>;

/// Envelope for type-erased events
pub struct EventEnvelope {
    pub metadata: EventMetadata,
    pub event: Box<dyn std::any::Any + Send + Sync>,
}

/// Snapshot for event sourcing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSnapshot {
    pub aggregate_id: String,
    pub aggregate_type: String,
    pub sequence: i64,
    pub timestamp: DateTime<Utc>,
    pub state: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_metadata_creation() {
        let metadata = EventMetadata::new(
            "node-123".to_string(),
            "Node".to_string(),
            "NodeAdded".to_string(),
        );

        assert_eq!(metadata.aggregate_id, "node-123");
        assert_eq!(metadata.aggregate_type, "Node");
        assert_eq!(metadata.event_type, "NodeAdded");
        assert!(metadata.causation_id.is_none());
    }

    #[test]
    fn test_event_metadata_builder() {
        let metadata = EventMetadata::new(
            "node-123".to_string(),
            "Node".to_string(),
            "NodeAdded".to_string(),
        )
        .with_causation("cmd-456".to_string())
        .with_correlation("corr-789".to_string())
        .with_user("user-1".to_string());

        assert_eq!(metadata.causation_id, Some("cmd-456".to_string()));
        assert_eq!(metadata.correlation_id, Some("corr-789".to_string()));
        assert_eq!(metadata.user_id, Some("user-1".to_string()));
    }
}
