//! GPU Manager Actor - Supervisor for specialized GPU computation actors

use actix::prelude::*;
use log::{debug, error, info};
use std::sync::Arc;

use super::shared::{ChildActorAddresses, GPUState, SharedGPUContext};
use super::{
    AnomalyDetectionActor, ClusteringActor, ConstraintActor, ForceComputeActor, GPUResourceActor,
    StressMajorizationActor,
};
use crate::actors::messages::*;
use crate::telemetry::agent_telemetry::{
    get_telemetry_logger, CorrelationId, LogLevel, TelemetryEvent,
};
use crate::utils::socket_flow_messages::BinaryNodeData;

///
pub struct GPUManagerActor {
    
    child_actors: Option<ChildActorAddresses>,

    
    gpu_state: GPUState,

    
    shared_context: Option<Arc<SharedGPUContext>>,

    
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

    
    fn spawn_child_actors(&mut self, _ctx: &mut Context<Self>) -> Result<(), String> {
        if self.children_spawned {
            debug!("Child actors already spawned, skipping");
            return Ok(()); 
        }

        info!("GPU Manager: Spawning specialized child actors");

        
        debug!("Creating GPUResourceActor...");
        let resource_actor = GPUResourceActor::new().start();
        debug!("GPUResourceActor created: {:?}", resource_actor);

        debug!("Creating ForceComputeActor...");
        
        
        let force_compute_actor = actix::Actor::create(|ctx| {
            ctx.set_mailbox_capacity(2048); 
            ForceComputeActor::new()
        });
        debug!("Creating ClusteringActor...");
        let clustering_actor = ClusteringActor::new().start();
        debug!("Creating AnomalyDetectionActor...");
        let anomaly_detection_actor = AnomalyDetectionActor::new().start();
        debug!("Creating StressMajorizationActor...");
        let stress_majorization_actor = StressMajorizationActor::new().start();
        debug!("Creating ConstraintActor...");
        let constraint_actor = ConstraintActor::new().start();
        debug!("Creating OntologyConstraintActor...");
        let ontology_constraint_actor = super::OntologyConstraintActor::new().start();

        self.child_actors = Some(ChildActorAddresses {
            resource_actor,
            force_compute_actor,
            clustering_actor,
            anomaly_detection_actor,
            stress_majorization_actor,
            constraint_actor,
            ontology_constraint_actor,
        });

        self.children_spawned = true;
        info!("GPU Manager: All child actors spawned successfully");
        Ok(())
    }

    
    fn get_child_actors(
        &mut self,
        ctx: &mut Context<Self>,
    ) -> Result<&ChildActorAddresses, String> {
        if !self.children_spawned {
            self.spawn_child_actors(ctx)?;
        }

        self.child_actors
            .as_ref()
            .ok_or_else(|| "Failed to get child actor addresses".to_string())
    }
}

impl Actor for GPUManagerActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Manager Actor started");

        
        if let Some(logger) = get_telemetry_logger() {
            let correlation_id = CorrelationId::new();
            let event = TelemetryEvent::new(
                correlation_id,
                LogLevel::INFO,
                "gpu_system",
                "manager_startup",
                "GPU Manager Actor started - child actors will be spawned on first message",
                "gpu_manager_actor",
            );
            logger.log_event(event);
        }

        
        
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GPU Manager Actor stopped");
    }
}

// === Message Routing Handlers ===

///
impl Handler<InitializeGPU> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, msg: InitializeGPU, ctx: &mut Self::Context) -> Self::Result {
        debug!("GPUManagerActor::handle(InitializeGPU) - Message received");
        info!(
            "GPU Manager: InitializeGPU received with {} nodes",
            msg.graph.nodes.len()
        );
        debug!(
            "Graph service address present: {}",
            msg.graph_service_addr.is_some()
        );

        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => {
                debug!("Child actors retrieved successfully");
                actors.clone()
            }
            Err(e) => {
                error!("Failed to get child actors: {}", e);
                return Box::pin(async move { Err(e) }.into_actor(self));
            }
        };

        
        let mut msg_with_manager = msg;
        msg_with_manager.gpu_manager_addr = Some(ctx.address());

        
        debug!("Delegating InitializeGPU to ResourceActor with manager address");
        let fut = child_actors
            .resource_actor
            .send(msg_with_manager)
            .into_actor(self)
            .map(|res, _actor, _ctx| match res {
                Ok(result) => result,
                Err(e) => {
                    error!("GPU Manager: ResourceActor communication failed: {}", e);
                    Err(format!("ResourceActor communication failed: {}", e))
                }
            });

        Box::pin(fut)
    }
}

///
impl Handler<UpdateGPUGraphData> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGPUGraphData, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        
        
        
        let graph = msg.graph.clone();


        if let Err(e) = child_actors.resource_actor.try_send(UpdateGPUGraphData {
            graph: graph.clone(),
            correlation_id: None,
        }) {
            error!("Failed to send UpdateGPUGraphData to ResourceActor: {}", e);
            return Err("Failed to delegate graph update to ResourceActor".to_string());
        }


        if let Err(e) = child_actors
            .force_compute_actor
            .try_send(UpdateGPUGraphData { graph: graph, correlation_id: None })
        {
            error!(
                "Failed to send UpdateGPUGraphData to ForceComputeActor: {}",
                e
            );
            return Err("Failed to delegate graph update to ForceComputeActor".to_string());
        }

        debug!("UpdateGPUGraphData sent to both ResourceActor and ForceComputeActor");
        Ok(())
    }
}

///
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

///
impl Handler<RunKMeans> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<KMeansResult, String>>;

    fn handle(&mut self, msg: RunKMeans, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self)),
        };

        let fut = child_actors
            .clustering_actor
            .send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| match res {
                Ok(result) => result,
                Err(e) => Err(format!("ClusteringActor communication failed: {}", e)),
            });

        Box::pin(fut)
    }
}

///
impl Handler<RunCommunityDetection> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<CommunityDetectionResult, String>>;

    fn handle(&mut self, msg: RunCommunityDetection, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self)),
        };

        let fut = child_actors
            .clustering_actor
            .send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| match res {
                Ok(result) => result,
                Err(e) => Err(format!("ClusteringActor communication failed: {}", e)),
            });

        Box::pin(fut)
    }
}

///
impl Handler<RunAnomalyDetection> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<AnomalyResult, String>>;

    fn handle(&mut self, msg: RunAnomalyDetection, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self)),
        };

        let fut = child_actors
            .anomaly_detection_actor
            .send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| match res {
                Ok(result) => result,
                Err(e) => Err(format!("AnomalyDetectionActor communication failed: {}", e)),
            });

        Box::pin(fut)
    }
}

///
impl Handler<PerformGPUClustering> for GPUManagerActor {
    type Result = ResponseActFuture<
        Self,
        Result<Vec<crate::handlers::api_handler::analytics::Cluster>, String>,
    >;

    fn handle(&mut self, msg: PerformGPUClustering, ctx: &mut Self::Context) -> Self::Result {
        info!(
            "GPU Manager: PerformGPUClustering received with method: {}",
            msg.method
        );

        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => return Box::pin(async move { Err(e) }.into_actor(self)),
        };

        
        match msg.method.as_str() {
            "kmeans" => {
                let kmeans_msg = RunKMeans {
                    params: KMeansParams {
                        num_clusters: msg.params.num_clusters.unwrap_or(8) as usize,
                        max_iterations: Some(msg.params.max_iterations.unwrap_or(100)),
                        tolerance: Some(msg.params.tolerance.unwrap_or(0.001) as f32),
                        seed: msg.params.seed.map(|s| s as u32),
                    },
                };

                Box::pin(
                    child_actors
                        .clustering_actor
                        .send(kmeans_msg)
                        .into_actor(self)
                        .map(|res, _actor, _ctx| match res {
                            Ok(Ok(kmeans_result)) => Ok(kmeans_result.clusters),
                            Ok(Err(e)) => Err(format!("K-means clustering failed: {}", e)),
                            Err(e) => Err(format!("ClusteringActor communication failed: {}", e)),
                        }),
                )
            }
            "spectral" | "louvain" | _ => {
                
                let community_msg = RunCommunityDetection {
                    params: CommunityDetectionParams {
                        algorithm: if msg.method == "louvain" {
                            CommunityDetectionAlgorithm::Louvain
                        } else {
                            CommunityDetectionAlgorithm::LabelPropagation
                        },
                        max_iterations: Some(msg.params.max_iterations.unwrap_or(100)),
                        convergence_tolerance: Some(0.001), 
                        synchronous: Some(true),            
                        seed: None,                         
                    },
                };

                Box::pin(
                    child_actors
                        .clustering_actor
                        .send(community_msg)
                        .into_actor(self)
                        .map(|res, _actor, _ctx| {
                            match res {
                                Ok(Ok(community_result)) => {
                                    
                                    let clusters = community_result
                                        .communities
                                        .into_iter()
                                        .map(|c| {
                                            let node_count = c.nodes.len() as u32;
                                            let label = format!("Community {}", c.id);
                                            crate::handlers::api_handler::analytics::Cluster {
                                                id: c.id,
                                                nodes: c.nodes,
                                                label,
                                                node_count,
                                                coherence: 0.8, 
                                                color: "#4ECDC4".to_string(),
                                                keywords: vec![],
                                                centroid: None,
                                            }
                                        })
                                        .collect();

                                    Ok(clusters)
                                }
                                Ok(Err(e)) => Err(format!("Community detection failed: {}", e)),
                                Err(e) => {
                                    Err(format!("ClusteringActor communication failed: {}", e))
                                }
                            }
                        }),
                )
            }
        }
    }
}

///
impl Handler<TriggerStressMajorization> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: TriggerStressMajorization, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        match child_actors.stress_majorization_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate stress majorization: {}", e)),
        }
    }
}

///
impl Handler<UpdateConstraints> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateConstraints, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        match child_actors.constraint_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate constraint update: {}", e)),
        }
    }
}

///
impl Handler<GetGPUStatus> for GPUManagerActor {
    type Result = MessageResult<GetGPUStatus>;

    fn handle(&mut self, _msg: GetGPUStatus, _ctx: &mut Self::Context) -> Self::Result {
        
        

        MessageResult(GPUStatus {
            is_initialized: self.shared_context.is_some(),
            failure_count: self.gpu_state.gpu_failure_count,
            num_nodes: self.gpu_state.num_nodes,
            iteration_count: self.gpu_state.iteration_count,
        })
    }
}

///
impl Handler<GetForceComputeActor> for GPUManagerActor {
    type Result = Result<Addr<ForceComputeActor>, String>;

    fn handle(&mut self, _msg: GetForceComputeActor, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;
        Ok(child_actors.force_compute_actor.clone())
    }
}

// Additional handlers for messages that need delegation

///
impl Handler<UploadConstraintsToGPU> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UploadConstraintsToGPU, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        
        match child_actors.constraint_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate UploadConstraintsToGPU: {}", e)),
        }
    }
}

///
impl Handler<GetNodeData> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<Vec<BinaryNodeData>, String>>;

    fn handle(&mut self, msg: GetNodeData, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => {
                return Box::pin(async move { Err(e) }.into_actor(self));
            }
        };

        
        let fut = child_actors
            .force_compute_actor
            .send(msg)
            .into_actor(self)
            .map(|res, _actor, _ctx| match res {
                Ok(result) => result,
                Err(e) => {
                    error!("GPU Manager: ForceComputeActor communication failed: {}", e);
                    Err(format!("ForceComputeActor communication failed: {}", e))
                }
            });

        Box::pin(fut)
    }
}

///
impl Handler<UpdateSimulationParams> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSimulationParams, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        
        match child_actors.force_compute_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate UpdateSimulationParams: {}", e)),
        }
    }
}

///
impl Handler<UpdateAdvancedParams> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateAdvancedParams, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        
        match child_actors.force_compute_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("Failed to delegate UpdateAdvancedParams: {}", e)),
        }
    }
}

///
impl Handler<SetSharedGPUContext> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: SetSharedGPUContext, ctx: &mut Self::Context) -> Self::Result {
        info!("GPUManagerActor: Received SharedGPUContext, distributing to all child actors");

        let child_actors = self.get_child_actors(ctx)?;
        let context = msg.context;
        let graph_service_addr = msg.graph_service_addr;
        let mut errors = Vec::new();


        if let Err(e) = child_actors
            .force_compute_actor
            .try_send(SetSharedGPUContext {
                context: context.clone(),
                graph_service_addr: graph_service_addr.clone(),
                correlation_id: None,
            })
        {
            errors.push(format!("ForceComputeActor: {}", e));
        } else {
            info!("SharedGPUContext sent to ForceComputeActor with GraphServiceActor address");
        }


        if let Err(e) = child_actors.clustering_actor.try_send(SetSharedGPUContext {
            context: context.clone(),
            graph_service_addr: graph_service_addr.clone(),
            correlation_id: None,
        }) {
            errors.push(format!("ClusteringActor: {}", e));
        } else {
            info!("SharedGPUContext sent to ClusteringActor");
        }


        if let Err(e) = child_actors.constraint_actor.try_send(SetSharedGPUContext {
            context: context.clone(),
            graph_service_addr: graph_service_addr.clone(),
            correlation_id: None,
        }) {
            errors.push(format!("ConstraintActor: {}", e));
        } else {
            info!("SharedGPUContext sent to ConstraintActor");
        }


        if let Err(e) = child_actors
            .stress_majorization_actor
            .try_send(SetSharedGPUContext {
                context: context.clone(),
                graph_service_addr: graph_service_addr.clone(),
                correlation_id: None,
            })
        {
            errors.push(format!("StressMajorizationActor: {}", e));
        } else {
            info!("SharedGPUContext sent to StressMajorizationActor");
        }


        if let Err(e) = child_actors
            .anomaly_detection_actor
            .try_send(SetSharedGPUContext {
                context: context.clone(),
                graph_service_addr: graph_service_addr.clone(),
                correlation_id: None,
            })
        {
            errors.push(format!("AnomalyDetectionActor: {}", e));
        } else {
            info!("SharedGPUContext sent to AnomalyDetectionActor");
        }


        if let Err(e) = child_actors
            .ontology_constraint_actor
            .try_send(SetSharedGPUContext {
                context: context.clone(),
                graph_service_addr: graph_service_addr.clone(),
                correlation_id: None,
            })
        {
            errors.push(format!("OntologyConstraintActor: {}", e));
        } else {
            info!("SharedGPUContext sent to OntologyConstraintActor");
        }

        if errors.is_empty() {
            info!("SharedGPUContext successfully distributed to all child actors");
            Ok(())
        } else {
            error!(
                "Failed to distribute SharedGPUContext to some actors: {:?}",
                errors
            );
            
            if !errors.iter().any(|e| e.starts_with("ForceComputeActor")) {
                Ok(())
            } else {
                Err(format!(
                    "Critical: Failed to send context to ForceComputeActor"
                ))
            }
        }
    }
}

///
impl Handler<ApplyOntologyConstraints> for GPUManagerActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ApplyOntologyConstraints, ctx: &mut Self::Context) -> Self::Result {
        let child_actors = self.get_child_actors(ctx)?;

        match child_actors.ontology_constraint_actor.try_send(msg) {
            Ok(_) => Ok(()),
            Err(e) => Err(format!(
                "Failed to delegate ApplyOntologyConstraints: {}",
                e
            )),
        }
    }
}

/// Handler for GetOntologyConstraintStats message
impl Handler<GetOntologyConstraintStats> for GPUManagerActor {
    type Result = ResponseActFuture<Self, Result<OntologyConstraintStats, String>>;

    fn handle(&mut self, _msg: GetOntologyConstraintStats, ctx: &mut Self::Context) -> Self::Result {
        info!("GPUManagerActor: GetOntologyConstraintStats received - delegating to OntologyConstraintActor");

        let child_actors = match self.get_child_actors(ctx) {
            Ok(actors) => actors.clone(),
            Err(e) => {
                error!("Failed to get child actors: {}", e);
                return Box::pin(async move {
                    Err(format!("Failed to get child actors: {}", e))
                }.into_actor(self));
            }
        };

        let fut = async move {
            // Send message to OntologyConstraintActor to get stats
            match child_actors.ontology_constraint_actor
                .send(GetOntologyConstraintStats)
                .await
            {
                Ok(result) => {
                    info!("Successfully retrieved ontology constraint stats from child actor");
                    result
                }
                Err(e) => {
                    error!("Failed to retrieve ontology constraint stats: {}", e);
                    Err(format!("Failed to retrieve ontology constraint stats: {}", e))
                }
            }
        }
        .into_actor(self);

        Box::pin(fut)
    }
}
