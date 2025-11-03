// src/services/pipeline_events.rs
//! Pipeline Event Definitions
//!
//! Event types for the end-to-end ontology processing pipeline.
//! Events enable reactive, decoupled communication between pipeline stages.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::models::constraints::ConstraintSet;
use crate::reasoning::custom_reasoner::InferredAxiom;

/// Correlation ID for tracing pipeline execution across stages
pub type CorrelationId = String;

/// Base event trait for all pipeline events
pub trait PipelineEvent {
    fn correlation_id(&self) -> &str;
    fn timestamp(&self) -> DateTime<Utc>;
    fn event_type(&self) -> &str;
}

/// Event fired when ontology data is modified (GitHub sync, manual upload, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyModifiedEvent {
    pub ontology_id: i64,
    pub class_count: usize,
    pub axiom_count: usize,
    pub source: String, // "github_sync", "manual_upload", "import"
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for OntologyModifiedEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "OntologyModified"
    }
}

/// Event fired when reasoning completes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningCompleteEvent {
    pub ontology_id: i64,
    pub inferred_axioms: Vec<InferredAxiom>,
    pub inference_time_ms: u64,
    pub cache_hit: bool,
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for ReasoningCompleteEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "ReasoningComplete"
    }
}

/// Event fired when constraints are generated from axioms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintsGeneratedEvent {
    pub constraint_set: ConstraintSet,
    pub axiom_count: usize,
    pub constraint_count: usize,
    pub generation_time_ms: u64,
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for ConstraintsGeneratedEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "ConstraintsGenerated"
    }
}

/// Event fired when constraints are uploaded to GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUUploadCompleteEvent {
    pub constraint_count: usize,
    pub upload_time_ms: f32,
    pub gpu_memory_used: usize,
    pub success: bool,
    pub fallback_to_cpu: bool,
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for GPUUploadCompleteEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "GPUUploadComplete"
    }
}

/// Event fired when physics simulation updates positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionsUpdatedEvent {
    pub updated_count: usize,
    pub iteration: u32,
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for PositionsUpdatedEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "PositionsUpdated"
    }
}

/// Event fired when pipeline stage fails
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineErrorEvent {
    pub stage: String, // "reasoning", "constraint_generation", "gpu_upload"
    pub error_message: String,
    pub retry_count: u32,
    pub will_retry: bool,
    pub correlation_id: CorrelationId,
    pub timestamp: DateTime<Utc>,
}

impl PipelineEvent for PipelineErrorEvent {
    fn correlation_id(&self) -> &str {
        &self.correlation_id
    }

    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }

    fn event_type(&self) -> &str {
        "PipelineError"
    }
}

/// Event bus for pipeline events
pub struct PipelineEventBus {
    handlers: HashMap<String, Vec<Box<dyn EventHandler + Send + Sync>>>,
    event_log: Vec<EventLogEntry>,
    max_log_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLogEntry {
    pub event_type: String,
    pub correlation_id: String,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Trait for event handlers
pub trait EventHandler {
    fn handle(&self, event: &dyn PipelineEvent) -> Result<(), String>;
}

impl PipelineEventBus {
    pub fn new(max_log_size: usize) -> Self {
        Self {
            handlers: HashMap::new(),
            event_log: Vec::new(),
            max_log_size,
        }
    }

    /// Subscribe a handler to specific event type
    pub fn subscribe<H: EventHandler + Send + Sync + 'static>(
        &mut self,
        event_type: &str,
        handler: H,
    ) {
        self.handlers
            .entry(event_type.to_string())
            .or_insert_with(Vec::new)
            .push(Box::new(handler));
    }

    /// Publish event to all subscribed handlers
    pub async fn publish(&mut self, event: &dyn PipelineEvent) -> Result<(), String> {
        let event_type = event.event_type();
        let correlation_id = event.correlation_id();

        // Log event
        self.log_event(event);

        // Call handlers
        if let Some(handlers) = self.handlers.get(event_type) {
            for handler in handlers {
                if let Err(e) = handler.handle(event) {
                    log::warn!(
                        "[{}] Event handler failed for {}: {}",
                        correlation_id,
                        event_type,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    fn log_event(&mut self, event: &dyn PipelineEvent) {
        let entry = EventLogEntry {
            event_type: event.event_type().to_string(),
            correlation_id: event.correlation_id().to_string(),
            timestamp: event.timestamp(),
            metadata: HashMap::new(),
        };

        self.event_log.push(entry);

        // Trim log if too large
        if self.event_log.len() > self.max_log_size {
            self.event_log.drain(0..self.event_log.len() - self.max_log_size);
        }
    }

    /// Get recent events for a correlation ID
    pub fn get_events_by_correlation(
        &self,
        correlation_id: &str,
    ) -> Vec<EventLogEntry> {
        self.event_log
            .iter()
            .filter(|e| e.correlation_id == correlation_id)
            .cloned()
            .collect()
    }

    /// Get event log statistics
    pub fn get_stats(&self) -> EventBusStats {
        let mut event_counts = HashMap::new();
        for entry in &self.event_log {
            *event_counts.entry(entry.event_type.clone()).or_insert(0) += 1;
        }

        EventBusStats {
            total_events: self.event_log.len(),
            event_counts,
            handler_counts: self
                .handlers
                .iter()
                .map(|(k, v)| (k.clone(), v.len()))
                .collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventBusStats {
    pub total_events: usize,
    pub event_counts: HashMap<String, usize>,
    pub handler_counts: HashMap<String, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
use crate::utils::time;

    struct TestHandler {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }

    impl EventHandler for TestHandler {
        fn handle(&self, _event: &dyn PipelineEvent) -> Result<(), String> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_event_bus_subscribe_publish() {
        let mut bus = PipelineEventBus::new(1000);
        let calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));

        bus.subscribe("OntologyModified", TestHandler { calls: calls.clone() });

        let event = OntologyModifiedEvent {
            ontology_id: 1,
            class_count: 5,
            axiom_count: 10,
            source: "test".to_string(),
            correlation_id: "test-123".to_string(),
            timestamp: time::now(),
        };

        bus.publish(&event).await.unwrap();

        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_event_log() {
        let mut bus = PipelineEventBus::new(10);

        let event = OntologyModifiedEvent {
            ontology_id: 1,
            class_count: 5,
            axiom_count: 10,
            source: "test".to_string(),
            correlation_id: "test-123".to_string(),
            timestamp: time::now(),
        };

        bus.publish(&event).await.unwrap();

        let events = bus.get_events_by_correlation("test-123");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "OntologyModified");
    }
}
