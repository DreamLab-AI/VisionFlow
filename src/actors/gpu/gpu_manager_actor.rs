//! GPU Manager Actor - Supervisor for specialized GPU computation actors

use actix::prelude::*;
use log::{error, info}; // debug, warn unused
// use std::sync::Arc; // Unused
// use std::collections::HashMap; // Unused

use crate::actors::messages::*;
// use crate::models::graph::GraphData;
// use crate::utils::unified_gpu_compute::UnifiedGPUCompute;
// use crate::actors::gpu_compute_actor::ComputeMode;
use super::shared::{SharedGPUContext, GPUState, ChildActorAddresses};
use super::{GPUResourceActor, ForceComputeActor, ClusteringActor, 
           AnomalyDetectionActor, StressMajorizationActor, ConstraintActor};

/// GPU Manager Actor - coordinates all specialized GPU actors
pub struct GPUManagerActor {
    /// Child actor addresses
    child_actors: Option<ChildActorAddresses>,
    
    /// Shared GPU state
    gpu_state: GPUState,
    
    /// Shared GPU context (None until initialized)
    shared_context: Option<SharedGPUContext>,
    
    /// Flag to track if child actors have been spawned
    children_spawned: bool,
}

impl GPUManagerActor {
    pub fn new() -> Self {
        Self {
            child_actors: None,
            gpu_state: GPUState::default(),
            shared_context: None,
            children_spawned: false,
        }
    }
    
    /// Spawn all child actors and establish supervision
    fn spawn_child_actors(&mut self, _ctx: &mut Context<Self>) -> Result<(), String> {
        if self.children_spawned {
            return Ok(()); // Already spawned
        }
        
        info!("GPU Manager: Spawning specialized child actors");
        
        // Spawn child actors with supervision
        let resource_actor = GPUResourceActor::new().start();
        let force_compute_actor = ForceComputeActor::new().start();
        let clustering_actor = ClusteringActor::new().start();
        let anomaly_detection_actor = AnomalyDetectionActor::new().start();
        let stress_majorization_actor = StressMajorizationActor::new().start();
        let constraint_actor = ConstraintActor::new().start();
        
        self.child_actors = Some(ChildActorAddresses {
            resource_actor,
            force_compute_actor,
            clustering_actor,
            anomaly_detection_actor,
            stress_majorization_actor,
            constraint_actor,
        });
        
        self.children_spawned = true;
        info!("GPU Manager: All child actors spawned successfully");
        Ok(())
    }
    
    /// Get child actor addresses, spawning them if needed
    fn get_child_actors(&mut self, ctx: &mut Context<Self>) -> Result<&ChildActorAddresses, String> {
        if !self.children_spawned {
            self.spawn_child_actors(ctx)?;
        }
        
        self.child_actors.as_ref()
            .ok_or_else(|| "Failed to get child actor addresses".to_string())
    }
}

impl Actor for GPUManagerActor {
    type Context = Context<Self>;
    
    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Manager Actor started");
        
        // Don't spawn children immediately - wait for first message
        // This prevents resource allocation until actually needed
    }
    
    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Manager Actor stopped");
    }
}

// === Message Routing Handlers ===

/// GPU Initialization - delegate to ResourceActor
impl Handler<InitializeGPU> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;
    
    fn handle(&mut self, msg: InitializeGPU, ctx: &mut Self::Context) -> Self::Result {
        info!("GPU Manager: InitializeGPU received");
        
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => {
                error!("Failed to get child actors: {}", e);
                return Box::pin(async move { Err(e) }.into_actor(self));
            }
        };
        
        // Delegate to ResourceActor
        let fut = child_actors.resource_actor.send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| {
                match res {
                    Ok(result) => result,
                    Err(e) => {
                        error!("GPU Manager: ResourceActor communication failed: {}", e);
                        Err(format!("ResourceActor communication failed: {}", e))
                    }
                }
            });
        
        Box::pin(fut)
    }
}

/// Graph data updates - delegate to ResourceActor
impl Handler<UpdateGPUGraphData> for GPUManagerActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateGPUGraphData, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;
        
        // Send to ResourceActor and handle synchronously for now
        // TODO: Could be made async if needed
        match child_actors.resource_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to send UpdateGPUGraphData to ResourceActor: {}", e);
                Err("Failed to delegate graph update".to_string())
            }
        }
    }
}

/// Force computation - delegate to ForceComputeActor
impl Handler<ComputeForces> for GPUManagerActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: ComputeForces, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;
        
        match child_actors.force_compute_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => {
                error!("Failed to send ComputeForces to ForceComputeActor: {}", e);
                Err("Failed to delegate force computation".to_string())
            }
        }
    }
}

/// Clustering operations - delegate to ClusteringActor
impl Handler<RunKMeans> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<KMeansResult, String>>;
    
    fn handle(&mut self, msg: RunKMeans, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self))
        };
        
        let fut = child_actors.clustering_actor.send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| {
                match res {
                    Ok(result) => result,
                    Err(e) => Err(format!("ClusteringActor communication failed: {}", e))
                }
            });
        
        Box::pin(fut)
    }
}

/// Community detection - delegate to ClusteringActor
impl Handler<RunCommunityDetection> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<CommunityDetectionResult, String>>;
    
    fn handle(&mut self, msg: RunCommunityDetection, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self))
        };
        
        let fut = child_actors.clustering_actor.send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| {
                match res {
                    Ok(result) => result,
                    Err(e) => Err(format!("ClusteringActor communication failed: {}", e))
                }
            });
        
        Box::pin(fut)
    }
}

/// Anomaly detection - delegate to AnomalyDetectionActor
impl Handler<RunAnomalyDetection> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<AnomalyResult, String>>;
    
    fn handle(&mut self, msg: RunAnomalyDetection, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self))
        };
        
        let fut = child_actors.anomaly_detection_actor.send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| {
                match res {
                    Ok(result) => result,
                    Err(e) => Err(format!("AnomalyDetectionActor communication failed: {}", e))
                }
            });
        
        Box::pin(fut)
    }
}

/// Stress majorization - delegate to StressMajorizationActor
impl Handler<TriggerStressMajorization> for GPUManagerActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: TriggerStressMajorization, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;
        
        match child_actors.stress_majorization_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate stress majorization: {}", e))
        }
    }
}

/// Constraint operations - delegate to ConstraintActor
impl Handler<UpdateConstraints> for GPUManagerActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, msg: UpdateConstraints, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;
        
        match child_actors.constraint_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate constraint update: {}", e))
        }
    }
}

/// Status queries - delegate to ResourceActor
impl Handler<GetGPUStatus> for GPUManagerActor {
    type Result = MessageResult<GetGPUStatus>;
    
    fn handle(&mut self, _msg: GetGPUStatus, _ctx: &mut Self::Context) -> Self::Result {
        // For status queries, we'll maintain some state locally for quick response
        // This avoids the async complexity for simple status checks
        
        MessageResult(GPUStatus {
            is_initialized: self.shared_context.is_some(),
            failure_count: self.gpu_state.gpu_failure_count,
            num_nodes: self.gpu_state.num_nodes,
            iteration_count: self.gpu_state.iteration_count,
        })
    }
}

// TODO: Add more handlers as needed for complete delegation
// This covers the main message types that need routing