//! Ontology Constraint Actor - GPU-accelerated ontology constraint evaluation
//!
//! This actor handles ontology-derived physics constraints on the GPU, translating
//! OWL axioms and ontology rules into physics forces that guide graph layout.
//!
//! ## Architecture
//!
//! Follows the established GPU actor pattern:
//! - SharedGPUContext for unified GPU access
//! - Graceful CPU fallback on GPU errors
//! - Memory pooling for constraint buffers
//! - Integration with ontology validation system

use actix::prelude::*;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

use super::shared::{GPUState, SharedGPUContext};
use crate::actors::messages::*;
use crate::models::constraints::{Constraint, ConstraintData, ConstraintSet};
use crate::physics::ontology_constraints::{
    OWLAxiom, OntologyConstraintTranslator, OntologyReasoningReport,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyConstraintStats {
    pub total_axioms_processed: u32,
    pub active_ontology_constraints: u32,
    pub constraint_evaluation_count: u32,
    pub last_update_time_ms: f32,
    pub gpu_failure_count: u32,
    pub cpu_fallback_count: u32,
    pub constraint_cache_hits: u32,
    pub constraint_cache_misses: u32,
}

impl Default for OntologyConstraintStats {
    fn default() -> Self {
        Self {
            total_axioms_processed: 0,
            active_ontology_constraints: 0,
            constraint_evaluation_count: 0,
            last_update_time_ms: 0.0,
            gpu_failure_count: 0,
            cpu_fallback_count: 0,
            constraint_cache_hits: 0,
            constraint_cache_misses: 0,
        }
    }
}

pub struct OntologyConstraintActor {

    shared_context: Option<Arc<SharedGPUContext>>,


    translator: OntologyConstraintTranslator,


    ontology_constraints: Vec<Constraint>,


    constraint_buffer: Vec<ConstraintData>,


    gpu_state: GPUState,


    stats: OntologyConstraintStats,


    #[allow(dead_code)]
    last_update: Instant,


    gpu_initialized: bool,

    /// Address of ForceComputeActor for sending constraint buffer updates
    force_compute_addr: Option<actix::Addr<super::force_compute_actor::ForceComputeActor>>,
}

impl OntologyConstraintActor {
    
    pub fn new() -> Self {
        info!("Creating new OntologyConstraintActor");

        Self {
            shared_context: None,
            translator: OntologyConstraintTranslator::new(),
            ontology_constraints: Vec::new(),
            constraint_buffer: Vec::new(),
            gpu_state: GPUState::default(),
            stats: OntologyConstraintStats::default(),
            last_update: Instant::now(),
            gpu_initialized: false,
            force_compute_addr: None,
        }
    }

    
    fn initialize_gpu(&mut self) -> Result<(), String> {
        if self.shared_context.is_none() {
            return Err("GPU context not available".to_string());
        }

        info!("OntologyConstraintActor: GPU initialization - context available");
        self.gpu_initialized = true;
        Ok(())
    }

    
    #[allow(dead_code)]
    fn apply_ontology_constraints(
        &mut self,
        reasoning_report: &OntologyReasoningReport,
        graph_data: &crate::models::graph::GraphData,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        info!(
            "OntologyConstraintActor: Applying ontology constraints - {} axioms, {} inferences",
            reasoning_report.axioms.len(),
            reasoning_report.inferences.len()
        );

        
        let constraint_set = self
            .translator
            .apply_ontology_constraints(graph_data, reasoning_report)
            .map_err(|e| format!("Failed to translate ontology constraints: {}", e))?;

        
        self.ontology_constraints = constraint_set.constraints.clone();
        self.stats.total_axioms_processed += reasoning_report.axioms.len() as u32;
        self.stats.active_ontology_constraints = self
            .ontology_constraints
            .iter()
            .filter(|c| c.active)
            .count() as u32;

        
        self.constraint_buffer = constraint_set.to_gpu_data();

        
        if self.gpu_initialized && self.shared_context.is_some() {
            match self.upload_constraints_to_gpu() {
                Ok(_) => {
                    info!(
                        "OntologyConstraintActor: Successfully uploaded {} constraints to GPU",
                        self.constraint_buffer.len()
                    );
                }
                Err(e) => {
                    warn!(
                        "OntologyConstraintActor: GPU upload failed, using CPU fallback: {}",
                        e
                    );
                    self.stats.gpu_failure_count += 1;
                    self.stats.cpu_fallback_count += 1;
                    
                }
            }
        } else {
            debug!(
                "OntologyConstraintActor: GPU not available, constraints stored for CPU processing"
            );
            self.stats.cpu_fallback_count += 1;
        }

        self.last_update = Instant::now();
        self.stats.last_update_time_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.stats.constraint_evaluation_count += 1;

        info!(
            "OntologyConstraintActor: Constraint application completed in {:.2}ms",
            self.stats.last_update_time_ms
        );

        // Notify ForceComputeActor about the new constraint buffer
        self.notify_force_compute_actor();

        Ok(())
    }

    /// Send the updated constraint buffer to ForceComputeActor
    fn notify_force_compute_actor(&self) {
        if let Some(ref addr) = self.force_compute_addr {
            info!(
                "OntologyConstraintActor: Sending {} constraints to ForceComputeActor",
                self.constraint_buffer.len()
            );
            addr.do_send(UpdateOntologyConstraintBuffer {
                constraint_buffer: self.constraint_buffer.clone(),
            });
        } else {
            debug!("OntologyConstraintActor: ForceComputeActor address not set, skipping notification");
        }
    }


    fn update_constraints(&mut self, axioms: &[OWLAxiom]) -> Result<(), String> {
        info!(
            "OntologyConstraintActor: Updating constraints with {} new axioms",
            axioms.len()
        );

        
        
        warn!("OntologyConstraintActor: Dynamic constraint updates require graph context");
        warn!("Consider using ApplyOntologyConstraints message with full context");

        self.stats.total_axioms_processed += axioms.len() as u32;

        Ok(())
    }

    
    fn upload_constraints_to_gpu(&self) -> Result<(), String> {
        let shared_context = self
            .shared_context
            .as_ref()
            .ok_or("GPU context not available")?;

        
        let mut unified_compute = shared_context
            .unified_compute
            .lock()
            .map_err(|e| format!("Failed to acquire GPU compute lock: {}", e))?;

        
        if self.constraint_buffer.is_empty() {
            debug!("OntologyConstraintActor: No constraints to upload, clearing GPU constraints");
            unified_compute
                .clear_constraints()
                .map_err(|e| format!("Failed to clear GPU constraints: {}", e))?;
        } else {
            unified_compute
                .upload_constraints(&self.constraint_buffer)
                .map_err(|e| format!("Failed to upload constraints to GPU: {}", e))?;
        }

        Ok(())
    }

    
    fn get_ontology_stats(&self) -> OntologyConstraintStats {
        self.stats.clone()
    }

    
    fn cleanup(&mut self) -> Result<(), String> {
        info!("OntologyConstraintActor: Cleaning up resources");

        
        self.ontology_constraints.clear();
        self.constraint_buffer.clear();

        
        if let Some(ref shared_context) = self.shared_context {
            if let Ok(mut unified_compute) = shared_context.unified_compute.lock() {
                if let Err(e) = unified_compute.clear_constraints() {
                    warn!("OntologyConstraintActor: Failed to clear GPU constraints during cleanup: {}", e);
                }
            }
        }

        
        self.translator.clear_cache();

        info!("OntologyConstraintActor: Cleanup completed");
        Ok(())
    }
}

impl Actor for OntologyConstraintActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("Ontology Constraint Actor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("Ontology Constraint Actor stopped");
        let _ = self.cleanup();
    }
}

// === Message Handlers ===

impl Handler<ApplyOntologyConstraints> for OntologyConstraintActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ApplyOntologyConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "OntologyConstraintActor: Received ApplyOntologyConstraints message for graph_id {}",
            msg.graph_id
        );

        
        let constraint_count = msg.constraint_set.constraints.len();
        match msg.merge_mode {
            ConstraintMergeMode::Replace => {
                
                self.ontology_constraints = msg.constraint_set.constraints.clone();
                info!(
                    "OntologyConstraintActor: Replaced all constraints with {} new constraints",
                    self.ontology_constraints.len()
                );
            }
            ConstraintMergeMode::Merge => {
                
                let existing_count = self.ontology_constraints.len();
                self.ontology_constraints
                    .extend(msg.constraint_set.constraints.clone());
                info!("OntologyConstraintActor: Merged {} new constraints with {} existing (total: {})",
                      constraint_count, existing_count, self.ontology_constraints.len());
            }
            ConstraintMergeMode::AddIfNoConflict => {
                
                let initial_count = self.ontology_constraints.len();
                for constraint in msg.constraint_set.constraints.clone() {
                    
                    let has_conflict = self.ontology_constraints.iter().any(|existing| {
                        existing.node_indices == constraint.node_indices
                            && existing.kind == constraint.kind
                    });

                    if !has_conflict {
                        self.ontology_constraints.push(constraint);
                    }
                }
                let added_count = self.ontology_constraints.len() - initial_count;
                info!(
                    "OntologyConstraintActor: Added {} non-conflicting constraints (skipped {})",
                    added_count,
                    constraint_count - added_count
                );
            }
        }

        self.constraint_buffer = msg.constraint_set.to_gpu_data();

        self.stats.active_ontology_constraints = self
            .ontology_constraints
            .iter()
            .filter(|c| c.active)
            .count() as u32;

        
        if self.gpu_initialized && self.shared_context.is_some() {
            match self.upload_constraints_to_gpu() {
                Ok(_) => {
                    info!("OntologyConstraintActor: Uploaded {} constraints via ApplyOntologyConstraints",
                          self.constraint_buffer.len());
                }
                Err(e) => {
                    warn!("OntologyConstraintActor: GPU upload failed: {}", e);
                    self.stats.gpu_failure_count += 1;
                }
            }
        }

        Ok(())
    }
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateOntologyConstraints {
    pub axioms: Vec<OWLAxiom>,
}

impl Handler<UpdateOntologyConstraints> for OntologyConstraintActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateOntologyConstraints, _ctx: &mut Self::Context) -> Self::Result {
        self.update_constraints(&msg.axioms)
    }
}

#[derive(Message)]
#[rtype(result = "Result<OntologyConstraintStats, String>")]
pub struct GetOntologyStats;

impl Handler<GetOntologyStats> for OntologyConstraintActor {
    type Result = Result<OntologyConstraintStats, String>;

    fn handle(&mut self, _msg: GetOntologyStats, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.get_ontology_stats())
    }
}

impl Handler<GetOntologyConstraintStats> for OntologyConstraintActor {
    type Result = Result<crate::actors::messages::OntologyConstraintStats, String>;

    fn handle(
        &mut self,
        _msg: GetOntologyConstraintStats,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("OntologyConstraintActor: Received GetOntologyConstraintStats message");

        
        let stats = crate::actors::messages::OntologyConstraintStats {
            total_axioms_processed: self.stats.total_axioms_processed,
            active_ontology_constraints: self.stats.active_ontology_constraints,
            constraint_evaluation_count: self.stats.constraint_evaluation_count,
            last_update_time_ms: self.stats.last_update_time_ms,
            gpu_failure_count: self.stats.gpu_failure_count,
            cpu_fallback_count: self.stats.cpu_fallback_count,
        };

        Ok(stats)
    }
}

/// Handler for setting the ForceComputeActor address for bidirectional communication
impl Handler<SetForceComputeAddr> for OntologyConstraintActor {
    type Result = ();

    fn handle(&mut self, msg: SetForceComputeAddr, _ctx: &mut Self::Context) -> Self::Result {
        info!("OntologyConstraintActor: Received ForceComputeActor address for constraint synchronization");
        self.force_compute_addr = Some(msg.addr);

        // If we already have constraints buffered, send them immediately
        if !self.constraint_buffer.is_empty() {
            self.notify_force_compute_actor();
        }
    }
}

impl Handler<SetSharedGPUContext> for OntologyConstraintActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        info!("OntologyConstraintActor: Received SharedGPUContext from ResourceActor");

        self.shared_context = Some(msg.context);

        match self.initialize_gpu() {
            Ok(_) => {
                info!("OntologyConstraintActor: GPU initialization successful");
                Ok(())
            }
            Err(e) => {
                warn!("OntologyConstraintActor: GPU initialization failed: {}", e);
                
                Ok(())
            }
        }
    }
}

impl Handler<GetConstraintStats> for OntologyConstraintActor {
    type Result = Result<ConstraintStats, String>;

    fn handle(&mut self, _msg: GetConstraintStats, _ctx: &mut Self::Context) -> Self::Result {

        let mut stats = ConstraintStats {
            total_constraints: self.ontology_constraints.len(),
            active_constraints: self.stats.active_ontology_constraints as usize,
            constraint_groups: std::collections::HashMap::new(),
            ontology_constraints: self.ontology_constraints.len(),
            user_constraints: 0,
        };


        stats.constraint_groups.insert(
            "ontology_derived".to_string(),
            self.ontology_constraints.len(),
        );

        Ok(stats)
    }
}

/// Handler for GetConstraintBuffer - provides GPU-ready constraint data
/// This is the key integration point for P0-2: it returns the constraint_buffer
/// that ForceComputeActor needs to upload to GPU via UnifiedGPUCompute::upload_constraints()
impl Handler<crate::actors::messages::GetConstraintBuffer> for OntologyConstraintActor {
    type Result = Result<Vec<ConstraintData>, String>;

    fn handle(&mut self, _msg: crate::actors::messages::GetConstraintBuffer, _ctx: &mut Self::Context) -> Self::Result {
        debug!("OntologyConstraintActor: Providing constraint buffer for GPU upload ({} constraints)",
               self.constraint_buffer.len());

        // Return a clone of the constraint buffer for GPU upload
        // This buffer contains ConstraintData structs ready for GPU processing
        Ok(self.constraint_buffer.clone())
    }
}

impl Handler<UpdateConstraints> for OntologyConstraintActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateConstraints, _ctx: &mut Self::Context) -> Self::Result {
        info!("OntologyConstraintActor: Received UpdateConstraints message");

        
        let constraints =
            match serde_json::from_value::<Vec<Constraint>>(msg.constraint_data.clone()) {
                Ok(constraints) => constraints,
                Err(e) => {
                    
                    match serde_json::from_value::<ConstraintSet>(msg.constraint_data) {
                        Ok(constraint_set) => constraint_set.constraints,
                        Err(_) => {
                            error!(
                                "OntologyConstraintActor: Failed to parse constraint_data: {}",
                                e
                            );
                            return Err(format!("Failed to parse constraints: {}", e));
                        }
                    }
                }
            };

        self.ontology_constraints = constraints;
        self.constraint_buffer = self
            .ontology_constraints
            .iter()
            .filter(|c| c.active)
            .map(|c| ConstraintData::from_constraint(c))
            .collect();

        if self.gpu_initialized && self.shared_context.is_some() {
            self.upload_constraints_to_gpu()?;
        }

        Ok(())
    }
}

impl Handler<InitializeGPU> for OntologyConstraintActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitializeGPU, _ctx: &mut Self::Context) -> Self::Result {
        info!("OntologyConstraintActor: InitializeGPU received");

        
        self.gpu_state.num_nodes = msg.graph.nodes.len() as u32;
        self.gpu_state.num_edges = msg.graph.edges.len() as u32;

        info!(
            "OntologyConstraintActor: Graph dimensions stored - {} nodes, {} edges",
            self.gpu_state.num_nodes, self.gpu_state.num_edges
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::ontology_constraints::OWLAxiomType;

    #[test]
    fn test_actor_creation() {
        let actor = OntologyConstraintActor::new();
        assert_eq!(actor.ontology_constraints.len(), 0);
        assert_eq!(actor.constraint_buffer.len(), 0);
        assert!(!actor.gpu_initialized);
    }

    #[test]
    fn test_stats_default() {
        let stats = OntologyConstraintStats::default();
        assert_eq!(stats.total_axioms_processed, 0);
        assert_eq!(stats.active_ontology_constraints, 0);
        assert_eq!(stats.gpu_failure_count, 0);
    }

    #[test]
    fn test_constraint_buffer_conversion() {
        let mut actor = OntologyConstraintActor::new();

        let constraints = vec![
            Constraint::fixed_position(0, 10.0, 20.0, 30.0),
            Constraint::separation(1, 2, 50.0),
        ];

        actor.ontology_constraints = constraints;
        actor.constraint_buffer = actor
            .ontology_constraints
            .iter()
            .map(|c| ConstraintData::from_constraint(c))
            .collect();

        assert_eq!(actor.constraint_buffer.len(), 2);
    }
}
