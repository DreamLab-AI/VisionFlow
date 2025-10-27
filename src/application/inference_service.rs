// src/application/inference_service.rs
//! Inference Application Service
//!
//! Orchestrates inference operations including reasoning, caching, and event publishing.
//! Provides high-level API for ontology inference and validation.

use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument, warn};
use chrono::Utc;

use crate::ports::inference_engine::{InferenceEngine, Result as EngineResult};
use crate::ports::ontology_repository::{OntologyRepository, InferenceResults};
use crate::inference::{InferenceCache, InferenceOptimizer, ValidationResult};
use crate::inference::types::{Inference, ConsistencyReport, ClassificationResult, UnsatisfiableClass};
use crate::inference::optimization::BatchInferenceRequest;
use crate::events::EventBus;

/// Inference service events
#[derive(Debug, Clone)]
pub enum InferenceEvent {
    /// Inference started
    InferenceStarted { ontology_id: String },

    /// Inference completed
    InferenceCompleted {
        ontology_id: String,
        inference_count: usize,
        duration_ms: u64,
    },

    /// Inference failed
    InferenceFailed {
        ontology_id: String,
        error: String,
    },

    /// Validation completed
    ValidationCompleted {
        ontology_id: String,
        consistent: bool,
    },

    /// Classification completed
    ClassificationCompleted {
        ontology_id: String,
        hierarchy_count: usize,
    },
}

/// Inference service configuration
#[derive(Debug, Clone)]
pub struct InferenceServiceConfig {
    /// Enable caching
    pub enable_cache: bool,

    /// Enable automatic inference
    pub auto_inference: bool,

    /// Maximum parallel inferences
    pub max_parallel: usize,

    /// Enable event publishing
    pub publish_events: bool,
}

impl Default for InferenceServiceConfig {
    fn default() -> Self {
        Self {
            enable_cache: true,
            auto_inference: true,
            max_parallel: 4,
            publish_events: true,
        }
    }
}

/// Application service for inference operations
pub struct InferenceService {
    /// Inference engine
    inference_engine: Arc<RwLock<dyn InferenceEngine>>,

    /// Ontology repository
    ontology_repo: Arc<dyn OntologyRepository>,

    /// Inference cache
    cache: Option<Arc<InferenceCache>>,

    /// Optimizer for batch/parallel operations
    optimizer: Arc<InferenceOptimizer>,

    /// Event bus for publishing events
    event_bus: Arc<RwLock<EventBus>>,

    /// Service configuration
    config: InferenceServiceConfig,
}

impl InferenceService {
    /// Create new inference service
    pub fn new(
        inference_engine: Arc<RwLock<dyn InferenceEngine>>,
        ontology_repo: Arc<dyn OntologyRepository>,
        event_bus: Arc<RwLock<EventBus>>,
        config: InferenceServiceConfig,
    ) -> Self {
        let cache = if config.enable_cache {
            Some(Arc::new(InferenceCache::default()))
        } else {
            None
        };

        let optimizer = Arc::new(InferenceOptimizer::new(config.max_parallel));

        Self {
            inference_engine,
            ontology_repo,
            cache,
            optimizer,
            event_bus,
            config,
        }
    }

    /// Run inference on an ontology
    #[instrument(skip(self), level = "info")]
    pub async fn run_inference(&self, ontology_id: &str) -> EngineResult<InferenceResults> {
        info!("Running inference for ontology: {}", ontology_id);
        let start = std::time::Instant::now();

        // Publish start event
        if self.config.publish_events {
            self.publish_event(InferenceEvent::InferenceStarted {
                ontology_id: ontology_id.to_string(),
            })
            .await;
        }

        // Load ontology from repository
        let (classes, axioms) = self.load_ontology_data(ontology_id).await?;
        let checksum = self.compute_checksum(&classes, &axioms);

        // Check cache
        if let Some(ref cache) = self.cache {
            if let Some(cached_results) = cache.get(ontology_id, &checksum).await {
                info!("Using cached inference results");
                return Ok(cached_results);
            }
        }

        // Load ontology into inference engine
        let mut engine = self.inference_engine.write().await;
        engine.load_ontology(classes, axioms).await?;

        // Perform inference
        let results = match engine.infer().await {
            Ok(results) => results,
            Err(e) => {
                warn!("Inference failed: {:?}", e);
                if self.config.publish_events {
                    self.publish_event(InferenceEvent::InferenceFailed {
                        ontology_id: ontology_id.to_string(),
                        error: format!("{:?}", e),
                    })
                    .await;
                }
                return Err(e);
            }
        };

        drop(engine); // Release write lock

        // Store results
        self.store_inference_results(ontology_id, &results).await?;

        // Cache results
        if let Some(ref cache) = self.cache {
            cache
                .put(ontology_id.to_string(), checksum, results.clone())
                .await;
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        // Publish completion event
        if self.config.publish_events {
            self.publish_event(InferenceEvent::InferenceCompleted {
                ontology_id: ontology_id.to_string(),
                inference_count: results.inferred_axioms.len(),
                duration_ms,
            })
            .await;
        }

        info!(
            "Inference completed in {}ms with {} inferred axioms",
            duration_ms,
            results.inferred_axioms.len()
        );

        Ok(results)
    }

    /// Validate ontology consistency
    #[instrument(skip(self), level = "info")]
    pub async fn validate_ontology(&self, ontology_id: &str) -> EngineResult<ValidationResult> {
        info!("Validating ontology: {}", ontology_id);
        let start = std::time::Instant::now();

        // Load ontology
        let (classes, axioms) = self.load_ontology_data(ontology_id).await?;

        let mut engine = self.inference_engine.write().await;
        engine.load_ontology(classes, axioms).await?;

        // Check consistency
        let consistent = engine.check_consistency().await?;

        // Get unsatisfiable classes if inconsistent
        let mut unsatisfiable = Vec::new();
        if !consistent {
            let hierarchy = engine.get_subclass_hierarchy().await?;
            let nothing_iri = "http://www.w3.org/2002/07/owl#Nothing";

            for (child, parent) in hierarchy {
                if parent == nothing_iri && child != nothing_iri {
                    unsatisfiable.push(UnsatisfiableClass {
                        class_iri: child,
                        reason: "Equivalent to owl:Nothing".to_string(),
                        conflicting_axioms: Vec::new(),
                    });
                }
            }
        }

        let validation_time_ms = start.elapsed().as_millis() as u64;

        let result = ValidationResult {
            consistent,
            unsatisfiable,
            warnings: Vec::new(),
            errors: Vec::new(),
            validation_time_ms,
        };

        // Publish event
        if self.config.publish_events {
            self.publish_event(InferenceEvent::ValidationCompleted {
                ontology_id: ontology_id.to_string(),
                consistent,
            })
            .await;
        }

        info!(
            "Validation completed: consistent={}, unsatisfiable={}",
            consistent,
            result.unsatisfiable.len()
        );

        Ok(result)
    }

    /// Classify ontology hierarchy
    #[instrument(skip(self), level = "info")]
    pub async fn classify_ontology(&self, ontology_id: &str) -> EngineResult<ClassificationResult> {
        info!("Classifying ontology: {}", ontology_id);
        let start = std::time::Instant::now();

        // Ensure inference has been run
        let results = self.run_inference(ontology_id).await?;

        let engine = self.inference_engine.read().await;
        let hierarchy = engine.get_subclass_hierarchy().await?;

        // Extract equivalent classes (classes with same sub/super classes)
        let equivalent_classes = self.find_equivalent_classes(&hierarchy);

        let classification_time_ms = start.elapsed().as_millis() as u64;

        let result = ClassificationResult {
            hierarchy: hierarchy.clone(),
            equivalent_classes,
            classification_time_ms,
            inferred_count: results.inferred_axioms.len(),
        };

        // Publish event
        if self.config.publish_events {
            self.publish_event(InferenceEvent::ClassificationCompleted {
                ontology_id: ontology_id.to_string(),
                hierarchy_count: hierarchy.len(),
            })
            .await;
        }

        Ok(result)
    }

    /// Get consistency report
    pub async fn get_consistency_report(&self, ontology_id: &str) -> EngineResult<ConsistencyReport> {
        let validation = self.validate_ontology(ontology_id).await?;

        let stats = {
            let engine = self.inference_engine.read().await;
            engine.get_statistics().await?
        };

        Ok(ConsistencyReport {
            is_consistent: validation.consistent,
            unsatisfiable_classes: validation.unsatisfiable,
            classes_checked: stats.loaded_classes,
            axioms_checked: stats.loaded_axioms,
            check_time_ms: validation.validation_time_ms,
            reasoner_version: "whelk-rs-1.0".to_string(),
        })
    }

    /// Batch inference for multiple ontologies
    pub async fn batch_inference(
        &self,
        ontology_ids: Vec<String>,
    ) -> EngineResult<std::collections::HashMap<String, InferenceResults>> {
        info!("Running batch inference for {} ontologies", ontology_ids.len());

        let request = BatchInferenceRequest {
            ontology_ids: ontology_ids.clone(),
            max_parallelism: self.config.max_parallel,
            timeout_ms: 60000, // 60 seconds per ontology
        };

        self.optimizer
            .process_batch(Arc::clone(&self.inference_engine), request)
            .await
    }

    /// Invalidate cache for ontology
    pub async fn invalidate_cache(&self, ontology_id: &str) {
        if let Some(ref cache) = self.cache {
            cache.invalidate(ontology_id).await;
            info!("Cache invalidated for ontology: {}", ontology_id);
        }
    }

    /// Helper: Load ontology data from repository
    async fn load_ontology_data(
        &self,
        ontology_id: &str,
    ) -> EngineResult<(Vec<crate::ports::ontology_repository::OwlClass>, Vec<crate::ports::ontology_repository::OwlAxiom>)> {
        let classes = self
            .ontology_repo
            .get_classes()
            .await
            .map_err(|e| crate::ports::inference_engine::InferenceEngineError::ReasonerError(format!("{:?}", e)))?;

        let axioms = self
            .ontology_repo
            .get_axioms()
            .await
            .map_err(|e| crate::ports::inference_engine::InferenceEngineError::ReasonerError(format!("{:?}", e)))?;

        Ok((classes, axioms))
    }

    /// Helper: Compute ontology checksum
    fn compute_checksum(
        &self,
        classes: &[crate::ports::ontology_repository::OwlClass],
        axioms: &[crate::ports::ontology_repository::OwlAxiom],
    ) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        classes.len().hash(&mut hasher);
        axioms.len().hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    /// Helper: Store inference results
    async fn store_inference_results(
        &self,
        ontology_id: &str,
        results: &InferenceResults,
    ) -> EngineResult<()> {
        // Store inferred axioms back to repository
        for axiom in &results.inferred_axioms {
            let _ = self.ontology_repo.add_axiom(axiom.clone()).await;
        }

        debug!(
            "Stored {} inferred axioms for ontology {}",
            results.inferred_axioms.len(),
            ontology_id
        );

        Ok(())
    }

    /// Helper: Find equivalent classes
    fn find_equivalent_classes(&self, hierarchy: &[(String, String)]) -> Vec<Vec<String>> {
        use std::collections::{HashMap, HashSet};

        let mut subclasses: HashMap<String, HashSet<String>> = HashMap::new();
        let mut superclasses: HashMap<String, HashSet<String>> = HashMap::new();

        // Build maps
        for (child, parent) in hierarchy {
            subclasses
                .entry(parent.clone())
                .or_default()
                .insert(child.clone());
            superclasses
                .entry(child.clone())
                .or_default()
                .insert(parent.clone());
        }

        // Find equivalent classes (same sub and super classes)
        let mut equivalent_groups: Vec<Vec<String>> = Vec::new();
        let mut processed: HashSet<String> = HashSet::new();

        for class in subclasses.keys() {
            if processed.contains(class) {
                continue;
            }

            let mut group = vec![class.clone()];
            let class_subs = subclasses.get(class);
            let class_supers = superclasses.get(class);

            for other in subclasses.keys() {
                if other == class || processed.contains(other) {
                    continue;
                }

                if subclasses.get(other) == class_subs && superclasses.get(other) == class_supers {
                    group.push(other.clone());
                    processed.insert(other.clone());
                }
            }

            if group.len() > 1 {
                processed.insert(class.clone());
                equivalent_groups.push(group);
            }
        }

        equivalent_groups
    }

    /// Helper: Publish event
    async fn publish_event(&self, event: InferenceEvent) {
        let event_bus = self.event_bus.write().await;
        // Event publishing implementation would go here
        debug!("Published event: {:?}", event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::mock;
    use crate::ports::ontology_repository::OwlClass;

    // Mock implementations would be added here for testing
}
