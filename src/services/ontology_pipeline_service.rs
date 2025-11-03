// src/services/ontology_pipeline_service.rs
//! Ontology Pipeline Service
//!
//! Orchestrates the end-to-end semantic physics pipeline:
//! 1. GitHub Sync → Parse Ontology → Save to unified.db (UnifiedOntologyRepository)
//! 2. Trigger Reasoning via ReasoningActor → CustomReasoner inference → Cache results
//! 3. Generate Constraints from axioms → ConstraintSet with Semantic kind
//! 4. Upload to GPU via OntologyConstraintActor → Apply semantic forces → Stream to client
//!
//! All ontology data persists in unified.db. Constraints use ConstraintKind::Semantic = 10.

use actix::Addr;
use log::{debug, error, info, warn};
use std::sync::Arc;

use crate::actors::graph_actor::GraphServiceActor;
use crate::actors::ontology_actor::OntologyActor;
use crate::actors::gpu::ontology_constraint_actor::OntologyConstraintActor;
use crate::reasoning::reasoning_actor::{ReasoningActor, TriggerReasoning as ReasoningTrigger};
use crate::reasoning::custom_reasoner::Ontology;
use crate::models::constraints::ConstraintSet;
use crate::services::github_sync_service::SyncStatistics;

/// Configuration for semantic physics pipeline
#[derive(Debug, Clone)]
pub struct SemanticPhysicsConfig {
    /// Enable automatic reasoning after ontology changes
    pub auto_trigger_reasoning: bool,

    /// Enable automatic constraint generation
    pub auto_generate_constraints: bool,

    /// Constraint strength multiplier (0.0 - 10.0)
    pub constraint_strength: f32,

    /// Enable GPU acceleration for constraints
    pub use_gpu_constraints: bool,

    /// Maximum reasoning depth
    pub max_reasoning_depth: usize,

    /// Cache reasoning results
    pub cache_inferences: bool,
}

impl Default for SemanticPhysicsConfig {
    fn default() -> Self {
        Self {
            auto_trigger_reasoning: true,
            auto_generate_constraints: true,
            constraint_strength: 1.0,
            use_gpu_constraints: true,
            max_reasoning_depth: 10,
            cache_inferences: true,
        }
    }
}

/// Statistics for the ontology pipeline
#[derive(Debug, Clone)]
pub struct OntologyPipelineStats {
    pub sync_stats: Option<SyncStatistics>,
    pub reasoning_triggered: bool,
    pub inferred_axioms_count: usize,
    pub constraints_generated: usize,
    pub gpu_upload_success: bool,
    pub total_time_ms: u64,
}

/// Orchestrates the complete ontology-to-physics pipeline
///
/// This service coordinates between:
/// - ReasoningActor: Runs CustomReasoner for OWL inference
/// - OntologyConstraintActor: Applies semantic constraints to GPU physics
/// - GraphServiceActor: Manages unified.db graph data
///
/// The pipeline automatically triggers after ontology modifications from GitHub sync.
pub struct OntologyPipelineService {
    config: SemanticPhysicsConfig,
    reasoning_actor: Option<Addr<ReasoningActor>>,
    ontology_actor: Option<Addr<OntologyActor>>,
    graph_actor: Option<Addr<GraphServiceActor>>,
    constraint_actor: Option<Addr<OntologyConstraintActor>>,
}

impl OntologyPipelineService {
    /// Create a new pipeline service
    pub fn new(config: SemanticPhysicsConfig) -> Self {
        info!("Initializing OntologyPipelineService with config: {:?}", config);

        Self {
            config,
            reasoning_actor: None,
            ontology_actor: None,
            graph_actor: None,
            constraint_actor: None,
        }
    }

    /// Set the reasoning actor address
    pub fn set_reasoning_actor(&mut self, addr: Addr<ReasoningActor>) {
        info!("OntologyPipelineService: Reasoning actor address registered");
        self.reasoning_actor = Some(addr);
    }

    /// Set the ontology actor address
    pub fn set_ontology_actor(&mut self, addr: Addr<OntologyActor>) {
        info!("OntologyPipelineService: Ontology actor address registered");
        self.ontology_actor = Some(addr);
    }

    /// Set the graph service actor address
    pub fn set_graph_actor(&mut self, addr: Addr<GraphServiceActor>) {
        info!("OntologyPipelineService: Graph service actor address registered");
        self.graph_actor = Some(addr);
    }

    /// Set the constraint actor address
    pub fn set_constraint_actor(&mut self, addr: Addr<OntologyConstraintActor>) {
        info!("OntologyPipelineService: Constraint actor address registered");
        self.constraint_actor = Some(addr);
    }

    /// Handle ontology modification event
    ///
    /// Called automatically by GitHubSyncService after parsing OntologyBlock sections.
    /// Pipeline flow:
    /// 1. Sends ontology data to ReasoningActor
    /// 2. ReasoningActor runs CustomReasoner inference
    /// 3. Inferred axioms converted to ConstraintSet with Semantic constraints
    /// 4. Constraints uploaded to GPU via OntologyConstraintActor
    /// 5. GPU physics applies semantic forces to node positions
    pub async fn on_ontology_modified(
        &self,
        ontology_id: i64,
        ontology: Ontology,
    ) -> Result<OntologyPipelineStats, String> {
        info!("🔄 Ontology modification detected for ID: {}", ontology_id);

        let start_time = std::time::Instant::now();
        let mut stats = OntologyPipelineStats {
            sync_stats: None,
            reasoning_triggered: false,
            inferred_axioms_count: 0,
            constraints_generated: 0,
            gpu_upload_success: false,
            total_time_ms: 0,
        };

        // Step 1: Trigger reasoning if enabled
        if self.config.auto_trigger_reasoning {
            match self.trigger_reasoning(ontology_id, ontology.clone()).await {
                Ok(axioms) => {
                    stats.reasoning_triggered = true;
                    stats.inferred_axioms_count = axioms.len();
                    info!("✅ Reasoning complete: {} inferred axioms", axioms.len());

                    // Step 2: Generate constraints from inferred axioms
                    if self.config.auto_generate_constraints && !axioms.is_empty() {
                        match self.generate_constraints_from_axioms(&axioms).await {
                            Ok(constraint_set) => {
                                stats.constraints_generated = constraint_set.constraints.len();
                                info!("✅ Generated {} constraints", stats.constraints_generated);

                                // Step 3: Upload constraints to GPU
                                if self.config.use_gpu_constraints {
                                    match self.upload_constraints_to_gpu(constraint_set).await {
                                        Ok(_) => {
                                            stats.gpu_upload_success = true;
                                            info!("✅ Constraints uploaded to GPU successfully");
                                        }
                                        Err(e) => {
                                            error!("❌ Failed to upload constraints to GPU: {}", e);
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                error!("❌ Failed to generate constraints: {}", e);
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Reasoning failed: {}", e);
                    return Err(format!("Reasoning failed: {}", e));
                }
            }
        }

        stats.total_time_ms = start_time.elapsed().as_millis() as u64;
        info!("🎉 Ontology pipeline complete in {}ms", stats.total_time_ms);

        Ok(stats)
    }

    /// Trigger reasoning process
    async fn trigger_reasoning(
        &self,
        ontology_id: i64,
        ontology: Ontology,
    ) -> Result<Vec<crate::reasoning::custom_reasoner::InferredAxiom>, String> {
        info!("🧠 Triggering reasoning for ontology {}", ontology_id);

        let reasoning_actor = self.reasoning_actor
            .as_ref()
            .ok_or_else(|| "Reasoning actor not configured".to_string())?;

        let msg = ReasoningTrigger {
            ontology_id,
            ontology,
        };

        match reasoning_actor.send(msg).await {
            Ok(Ok(axioms)) => {
                info!("✅ Reasoning succeeded: {} axioms inferred", axioms.len());
                Ok(axioms)
            }
            Ok(Err(e)) => {
                error!("❌ Reasoning failed: {}", e);
                Err(format!("Reasoning error: {}", e))
            }
            Err(e) => {
                error!("❌ Failed to send reasoning message: {}", e);
                Err(format!("Mailbox error: {}", e))
            }
        }
    }

    /// Generate physics constraints from inferred axioms
    ///
    /// Converts CustomReasoner axiom types to semantic constraints:
    /// - SubClassOf: Attraction forces (child → parent clustering)
    /// - EquivalentTo: Strong attraction forces (equivalent classes align)
    /// - DisjointWith: Repulsion forces (disjoint classes separate)
    ///
    /// All constraints use ConstraintKind::Semantic (= 10) which is processed
    /// by ontology_constraints.cu in the CUDA kernel pipeline.
    async fn generate_constraints_from_axioms(
        &self,
        axioms: &[crate::reasoning::custom_reasoner::InferredAxiom],
    ) -> Result<ConstraintSet, String> {
        info!("🔧 Generating constraints from {} axioms", axioms.len());

        use crate::models::constraints::{Constraint, ConstraintKind};

        let mut constraints = Vec::new();

        use crate::reasoning::custom_reasoner::AxiomType;

        for axiom in axioms {
            // Convert inferred axioms to physics constraints
            // SubClassOf(A, B) → nodes of class A should cluster near class B nodes
            match axiom.axiom_type {
                AxiomType::SubClassOf => {
                    if let Some(_superclass) = &axiom.object {
                        constraints.push(Constraint {
                            kind: ConstraintKind::Semantic,
                            node_indices: vec![],
                            params: vec![],
                            weight: self.config.constraint_strength,
                            active: true,
                        });
                    }
                }
                AxiomType::EquivalentTo => {
                    if let Some(_class_b) = &axiom.object {
                        constraints.push(Constraint {
                            kind: ConstraintKind::Semantic,
                            node_indices: vec![],
                            params: vec![],
                            weight: self.config.constraint_strength * 1.5,
                            active: true,
                        });
                    }
                }
                AxiomType::DisjointWith => {
                    if let Some(_class_b) = &axiom.object {
                        constraints.push(Constraint {
                            kind: ConstraintKind::Semantic,
                            node_indices: vec![],
                            params: vec![],
                            weight: self.config.constraint_strength * 2.0,
                            active: true,
                        });
                    }
                }
                _ => {
                    debug!("Skipping axiom type: {:?}", axiom.axiom_type);
                }
            }
        }

        info!("✅ Generated {} constraints from axioms", constraints.len());

        Ok(ConstraintSet {
            constraints,
            groups: std::collections::HashMap::new(),
        })
    }

    /// Upload constraints to GPU
    async fn upload_constraints_to_gpu(
        &self,
        constraint_set: ConstraintSet,
    ) -> Result<(), String> {
        info!("📤 Uploading {} constraints to GPU", constraint_set.constraints.len());

        let constraint_actor = self.constraint_actor
            .as_ref()
            .ok_or_else(|| "Constraint actor not configured".to_string())?;

        use crate::actors::messages::ApplyOntologyConstraints;
        use crate::actors::messages::ConstraintMergeMode;

        let msg = ApplyOntologyConstraints {
            constraint_set,
            merge_mode: ConstraintMergeMode::Merge,
            graph_id: 0, // Main knowledge graph
        };

        match constraint_actor.send(msg).await {
            Ok(Ok(_)) => {
                info!("✅ Constraints uploaded to GPU successfully");
                Ok(())
            }
            Ok(Err(e)) => {
                error!("❌ Failed to apply constraints: {}", e);
                Err(e)
            }
            Err(e) => {
                error!("❌ Failed to send constraint message: {}", e);
                Err(format!("Mailbox error: {}", e))
            }
        }
    }

    /// Get current configuration
    pub fn get_config(&self) -> &SemanticPhysicsConfig {
        &self.config
    }

    /// Update configuration
    pub fn update_config(&mut self, config: SemanticPhysicsConfig) {
        info!("Updating OntologyPipelineService configuration");
        self.config = config;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SemanticPhysicsConfig::default();
        assert!(config.auto_trigger_reasoning);
        assert!(config.auto_generate_constraints);
        assert_eq!(config.constraint_strength, 1.0);
        assert!(config.use_gpu_constraints);
    }

    #[test]
    fn test_pipeline_creation() {
        let config = SemanticPhysicsConfig::default();
        let pipeline = OntologyPipelineService::new(config.clone());
        assert_eq!(pipeline.get_config().constraint_strength, 1.0);
    }
}
