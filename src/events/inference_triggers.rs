// src/events/inference_triggers.rs
//! Automatic Inference Triggers
//!
//! Event-driven inference that automatically runs reasoning when ontology changes occur.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, instrument};

use crate::application::inference_service::InferenceService;
use crate::events::{EventBus, EventHandler};

/// Ontology-related events that trigger inference
#[derive(Debug, Clone)]
pub enum OntologyEvent {
    /// Ontology imported
    OntologyImported {
        ontology_id: String,
        class_count: usize,
        axiom_count: usize,
    },

    /// Class added
    ClassAdded {
        ontology_id: String,
        class_iri: String,
    },

    /// Axiom added
    AxiomAdded {
        ontology_id: String,
        axiom_id: String,
    },

    /// Ontology modified
    OntologyModified {
        ontology_id: String,
        change_type: String,
    },
}

/// Configuration for automatic inference
#[derive(Debug, Clone)]
pub struct AutoInferenceConfig {
    /// Enable automatic inference on ontology import
    pub auto_on_import: bool,

    /// Enable incremental inference on class addition
    pub auto_on_class_add: bool,

    /// Enable incremental inference on axiom addition
    pub auto_on_axiom_add: bool,

    /// Minimum delay between automatic inferences (milliseconds)
    pub min_delay_ms: u64,

    /// Batch multiple changes before inferring
    pub batch_changes: bool,
}

impl Default for AutoInferenceConfig {
    fn default() -> Self {
        Self {
            auto_on_import: true,
            auto_on_class_add: false, // Usually too frequent
            auto_on_axiom_add: false, // Usually too frequent
            min_delay_ms: 1000,       // 1 second
            batch_changes: true,
        }
    }
}

/// Automatic inference trigger handler
pub struct InferenceTriggerHandler {
    /// Inference service
    inference_service: Arc<RwLock<InferenceService>>,

    /// Configuration
    config: AutoInferenceConfig,

    /// Last inference timestamp per ontology
    last_inference: Arc<RwLock<std::collections::HashMap<String, std::time::Instant>>>,
}

impl InferenceTriggerHandler {
    /// Create new inference trigger handler
    pub fn new(
        inference_service: Arc<RwLock<InferenceService>>,
        config: AutoInferenceConfig,
    ) -> Self {
        Self {
            inference_service,
            config,
            last_inference: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Handle ontology event
    #[instrument(skip(self), level = "debug")]
    pub async fn handle_event(&self, event: OntologyEvent) {
        match event {
            OntologyEvent::OntologyImported { ontology_id, .. } => {
                if self.config.auto_on_import {
                    info!("Auto-triggering inference for imported ontology: {}", ontology_id);
                    self.trigger_inference(&ontology_id).await;
                }
            }

            OntologyEvent::ClassAdded { ontology_id, .. } => {
                if self.config.auto_on_class_add {
                    debug!("Auto-triggering inference for class addition: {}", ontology_id);
                    self.trigger_incremental_inference(&ontology_id).await;
                }
            }

            OntologyEvent::AxiomAdded { ontology_id, .. } => {
                if self.config.auto_on_axiom_add {
                    debug!("Auto-triggering inference for axiom addition: {}", ontology_id);
                    self.trigger_incremental_inference(&ontology_id).await;
                }
            }

            OntologyEvent::OntologyModified { ontology_id, .. } => {
                debug!("Ontology modified, considering inference: {}", ontology_id);
                self.trigger_incremental_inference(&ontology_id).await;
            }
        }
    }

    /// Trigger full inference
    async fn trigger_inference(&self, ontology_id: &str) {
        // Check rate limiting
        if !self.should_run_inference(ontology_id).await {
            debug!("Skipping inference due to rate limiting: {}", ontology_id);
            return;
        }

        let service = self.inference_service.read().await;

        match service.run_inference(ontology_id).await {
            Ok(results) => {
                info!(
                    "Auto-inference completed for {}: {} inferred axioms in {}ms",
                    ontology_id,
                    results.inferred_axioms.len(),
                    results.inference_time_ms
                );

                // Update last inference time
                self.update_last_inference(ontology_id).await;
            }
            Err(e) => {
                warn!("Auto-inference failed for {}: {:?}", ontology_id, e);
            }
        }
    }

    /// Trigger incremental inference
    async fn trigger_incremental_inference(&self, ontology_id: &str) {
        // For now, batch changes and run full inference
        // TODO: Implement true incremental reasoning
        if self.config.batch_changes {
            debug!("Batching changes for: {}", ontology_id);
            // In a real implementation, we'd accumulate changes and run inference periodically
        } else {
            self.trigger_inference(ontology_id).await;
        }
    }

    /// Check if inference should run based on rate limiting
    async fn should_run_inference(&self, ontology_id: &str) -> bool {
        let last_inference = self.last_inference.read().await;

        if let Some(last_time) = last_inference.get(ontology_id) {
            let elapsed = last_time.elapsed().as_millis() as u64;
            elapsed >= self.config.min_delay_ms
        } else {
            true // Never run before
        }
    }

    /// Update last inference timestamp
    async fn update_last_inference(&self, ontology_id: &str) {
        let mut last_inference = self.last_inference.write().await;
        last_inference.insert(ontology_id.to_string(), std::time::Instant::now());
    }
}

/// Register inference triggers with event bus
pub async fn register_inference_triggers(
    event_bus: Arc<RwLock<EventBus>>,
    inference_service: Arc<RwLock<InferenceService>>,
    config: AutoInferenceConfig,
) {
    let handler = Arc::new(InferenceTriggerHandler::new(inference_service, config));

    // Register event listeners
    // This would integrate with the actual EventBus implementation

    info!("Inference triggers registered with event bus");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_inference_config_default() {
        let config = AutoInferenceConfig::default();
        assert!(config.auto_on_import);
        assert!(!config.auto_on_class_add);
        assert!(config.batch_changes);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = AutoInferenceConfig {
            min_delay_ms: 100,
            ..Default::default()
        };

        // Handler creation would need mock service
        // This is a placeholder for the test structure
    }
}
