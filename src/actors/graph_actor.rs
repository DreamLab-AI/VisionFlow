//! Graph Service Actor with hybrid solver orchestration
//!
//! This module implements an advanced graph service actor that integrates:
//! - Advanced GPU compute with constraint-aware physics
//! - Semantic analysis for knowledge graph enhancement
//! - Multi-modal edge generation with similarity computation
//! - Stress-majorization periodic optimization
//! - Dynamic constraint generation and update handling
//!
//! ## Integration Overview
//!
//! The hybrid solver orchestration combines multiple advanced techniques:
//!
//! ### 1. Advanced GPU Physics
//! - **Enhanced GPU Context**: Uses `AdvancedGPUContext` for constraint-aware physics
//! - **Enhanced Node Data**: Extends `BinaryNodeData` with semantic weights and metadata
//! - **Fallback Compatibility**: Maintains compatibility with legacy GPU compute actor
//!
//! ### 2. Semantic Analysis & Edge Generation
//! - **Semantic Analyzer**: Extracts features from file metadata and content
//! - **Multi-modal Edges**: Generates edges based on semantic, structural, and temporal similarities
//! - **Feature Caching**: Maintains semantic feature cache for performance
//!
//! ### 3. Constraint Management
//! - **Dynamic Constraints**: Auto-generates constraints based on semantic analysis
//! - **Constraint Groups**: Organizes constraints into logical groups (boundary, clustering, etc.)
//! - **Real-time Updates**: Handles constraint updates via WebSocket control frames
//!
//! ### 4. Stress-Majorization Integration
//! - **Periodic Optimization**: Executes stress-majorization every N frames
//! - **Global Layout**: Optimizes overall graph layout to minimize stress function
//! - **Constraint Satisfaction**: Balances force-directed physics with constraint satisfaction
//!
//! ### 5. Control Flow & Performance
//! - **Hybrid Execution**: Seamlessly switches between advanced and legacy GPU modes
//! - **Performance Monitoring**: Tracks semantic analysis status and timing
//! - **Error Handling**: Robust error handling with graceful fallbacks
//!
//! ## Usage Patterns
//!
//! The enhanced actor maintains backward compatibility while providing advanced features:
//!
//! ```rust
//! 
//! let actor = GraphServiceActor::new(client_manager, gpu_compute_addr);
//!
//! 
//! actor.update_advanced_physics_params(advanced_params)?;
//! actor.trigger_stress_optimization()?;
//! let status = actor.get_semantic_analysis_status();
//! ```

use crate::types::Vec3Data;
use actix::prelude::*;
use glam::Vec3;
use log::{debug, error, info, trace, warn};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::Duration;

use crate::actors::graph_messages::{
    BatchAddEdges, BatchAddNodes, BatchGraphUpdate, ConfigureUpdateQueue, FlushUpdateQueue,
};
use crate::actors::messages::*;
use crate::errors::VisionFlowError;
// use crate::utils::binary_protocol; 
use crate::actors::client_coordinator_actor::ClientCoordinatorActor;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::{FileMetadata, MetadataStore};
use crate::models::node::Node;
use crate::utils::socket_flow_messages::{glam_to_vec3data, BinaryNodeData, BinaryNodeDataClient}; 
                                                                                                  
use crate::actors::gpu::GPUManagerActor;
use crate::config::AutoBalanceConfig;
use crate::models::simulation_params::SimulationParams;

// Advanced physics and AI modules
use crate::models::constraints::{AdvancedParams, Constraint, ConstraintSet};
use crate::services::edge_generation::{AdvancedEdgeGenerator, EdgeGenerationConfig};
use crate::services::semantic_analyzer::{SemanticAnalyzer, SemanticFeatures};
use crate::utils::unified_gpu_compute::UnifiedGPUCompute;
// use crate::models::simulation_params::SimParams; 
use crate::physics::stress_majorization::StressMajorizationSolver;
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;
use std::sync::Mutex;

pub struct GraphServiceActor {
    graph_data: Arc<GraphData>,        
    node_map: Arc<HashMap<u32, Node>>, 
    gpu_compute_addr: Option<Addr<GPUManagerActor>>, 
    kg_repo: Arc<dyn KnowledgeGraphRepository>, 
    client_manager: Addr<ClientCoordinatorActor>,
    simulation_running: AtomicBool,
    shutdown_complete: Arc<AtomicBool>,
    next_node_id: AtomicU32,
    bots_graph_data: Arc<GraphData>, 
    simulation_params: SimulationParams, 

    
    gpu_init_in_progress: bool, 
    gpu_initialized: bool,      
    constraint_set: ConstraintSet,
    semantic_analyzer: SemanticAnalyzer,
    edge_generator: AdvancedEdgeGenerator,
    stress_solver: StressMajorizationSolver,
    semantic_features_cache: HashMap<String, SemanticFeatures>,

    
    advanced_params: AdvancedParams,

    
    stress_step_counter: u32,
    constraint_update_counter: u32,
    last_semantic_analysis: Option<std::time::Instant>,

    
    settings_addr: Option<Addr<crate::actors::optimized_settings_actor::OptimizedSettingsActor>>,
    auto_balance_history: Vec<f32>, 
    stable_count: u32,              
    auto_balance_notifications: Arc<Mutex<Vec<AutoBalanceNotification>>>, 
    kinetic_energy_history: Vec<f32>, 
    last_adjustment_time: std::time::Instant, 
    current_state: AutoBalanceState,  
    frames_since_last_broadcast: Option<u32>, 
    last_broadcast_time: Option<std::time::Instant>, 
    initial_positions_sent: bool, 

    
    pending_broadcasts: u32,     
    max_pending_broadcasts: u32, 
    broadcast_skip_count: u32,   
    last_backpressure_warning: Option<std::time::Instant>, 

    
    target_params: SimulationParams, 
    param_transition_rate: f32,      

    
    previous_positions: HashMap<u32, crate::types::vec3::Vec3Data>, 

    
    update_queue: UpdateQueue,
    queue_config: UpdateQueueConfig,

    
    batch_metrics: BatchMetrics,
}

///
#[derive(Debug, Clone)]
struct UpdateQueueConfig {
    max_operations: usize,
    flush_interval_ms: u64,
    enable_auto_flush: bool,
}

impl Default for UpdateQueueConfig {
    fn default() -> Self {
        Self {
            max_operations: 1000,
            flush_interval_ms: 100,
            enable_auto_flush: true,
        }
    }
}

///
#[derive(Debug, Default)]
struct UpdateQueue {
    pending_nodes: Vec<Node>,
    pending_edges: Vec<Edge>,
    pending_node_removals: Vec<u32>,
    pending_edge_removals: Vec<String>,
    last_flush_time: Option<std::time::Instant>,
    operation_count: usize,
}

impl UpdateQueue {
    fn new() -> Self {
        Self {
            pending_nodes: Vec::new(),
            pending_edges: Vec::new(),
            pending_node_removals: Vec::new(),
            pending_edge_removals: Vec::new(),
            last_flush_time: Some(std::time::Instant::now()),
            operation_count: 0,
        }
    }

    fn add_node(&mut self, node: Node) {
        self.pending_nodes.push(node);
        self.operation_count += 1;
    }

    fn add_edge(&mut self, edge: Edge) {
        self.pending_edges.push(edge);
        self.operation_count += 1;
    }

    fn remove_node(&mut self, node_id: u32) {
        self.pending_node_removals.push(node_id);
        self.operation_count += 1;
    }

    fn remove_edge(&mut self, edge_id: String) {
        self.pending_edge_removals.push(edge_id);
        self.operation_count += 1;
    }

    fn is_empty(&self) -> bool {
        self.pending_nodes.is_empty()
            && self.pending_edges.is_empty()
            && self.pending_node_removals.is_empty()
            && self.pending_edge_removals.is_empty()
    }

    fn clear(&mut self) {
        self.pending_nodes.clear();
        self.pending_edges.clear();
        self.pending_node_removals.clear();
        self.pending_edge_removals.clear();
        self.operation_count = 0;
        self.last_flush_time = Some(std::time::Instant::now());
    }

    fn should_flush(&self, config: &UpdateQueueConfig) -> bool {
        if !config.enable_auto_flush {
            return false;
        }

        
        if self.operation_count >= config.max_operations {
            return true;
        }

        
        if let Some(last_flush) = self.last_flush_time {
            let elapsed = last_flush.elapsed();
            if elapsed.as_millis() >= config.flush_interval_ms as u128 {
                return true;
            }
        }

        false
    }
}

///
#[derive(Debug, Default)]
struct BatchMetrics {
    total_batches_processed: u64,
    total_nodes_batched: u64,
    total_edges_batched: u64,
    average_batch_size: f64,
    max_batch_size: usize,
    total_flush_count: u64,
    auto_flush_count: u64,
    manual_flush_count: u64,
    last_flush_duration_ms: u64,
}

impl BatchMetrics {
    fn record_batch(
        &mut self,
        node_count: usize,
        edge_count: usize,
        flush_duration: std::time::Duration,
        is_auto_flush: bool,
    ) {
        self.total_batches_processed += 1;
        self.total_nodes_batched += node_count as u64;
        self.total_edges_batched += edge_count as u64;

        let batch_size = node_count + edge_count;
        if batch_size > self.max_batch_size {
            self.max_batch_size = batch_size;
        }

        
        let total_operations = self.total_nodes_batched + self.total_edges_batched;
        self.average_batch_size = total_operations as f64 / self.total_batches_processed as f64;

        self.total_flush_count += 1;
        if is_auto_flush {
            self.auto_flush_count += 1;
        } else {
            self.manual_flush_count += 1;
        }

        self.last_flush_duration_ms = flush_duration.as_millis() as u64;
    }
}

///
#[derive(Debug, Clone, PartialEq)]
enum AutoBalanceState {
    Stable,      
    Spreading,   
    Clustering,  
    Bouncing,    
    Oscillating, 
    Adjusting,   
}

///
///
#[derive(Debug, Clone, Serialize)]
pub struct PhysicsState {
    pub is_settled: bool,
    pub stable_frame_count: u32,
    pub kinetic_energy: f32,
    pub current_state: String, 
}

// Auto-balance notification structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoBalanceNotification {
    pub message: String,
    pub timestamp: i64,
    pub severity: String, 
}

impl GraphServiceActor {
    fn smooth_transition_params(&mut self) {
        
        let rate = self.param_transition_rate;

        
        self.simulation_params.repel_k =
            self.simulation_params.repel_k * (1.0 - rate) + self.target_params.repel_k * rate;
        self.simulation_params.damping =
            self.simulation_params.damping * (1.0 - rate) + self.target_params.damping * rate;
        self.simulation_params.max_velocity = self.simulation_params.max_velocity * (1.0 - rate)
            + self.target_params.max_velocity * rate;
        self.simulation_params.spring_k =
            self.simulation_params.spring_k * (1.0 - rate) + self.target_params.spring_k * rate;
        self.simulation_params.viewport_bounds = self.simulation_params.viewport_bounds
            * (1.0 - rate)
            + self.target_params.viewport_bounds * rate;

        
        self.simulation_params.max_repulsion_dist = self.simulation_params.max_repulsion_dist
            * (1.0 - rate)
            + self.target_params.max_repulsion_dist * rate;
        self.simulation_params.boundary_force_strength =
            self.simulation_params.boundary_force_strength * (1.0 - rate)
                + self.target_params.boundary_force_strength * rate;
        self.simulation_params.cooling_rate = self.simulation_params.cooling_rate * (1.0 - rate)
            + self.target_params.cooling_rate * rate;
        self.simulation_params.spring_k =
            self.simulation_params.spring_k * (1.0 - rate) + self.target_params.spring_k * rate;

        
        if (self.target_params.enable_bounds as i32 - self.simulation_params.enable_bounds as i32)
            .abs()
            > 0
        {
            self.simulation_params.enable_bounds = self.target_params.enable_bounds;
        }
    }

    fn set_target_params(&mut self, new_params: SimulationParams) {
        self.target_params = new_params;
        
        let damping_change = (self.target_params.damping - self.simulation_params.damping).abs();
        let repulsion_cutoff_change = (self.target_params.max_repulsion_dist
            - self.simulation_params.max_repulsion_dist)
            .abs();
        let grid_cell_change = (self.target_params.boundary_force_strength
            - self.simulation_params.boundary_force_strength)
            .abs();

        if damping_change > 0.3 || repulsion_cutoff_change > 20.0 || grid_cell_change > 0.5 {
            self.param_transition_rate = 0.2; 
        } else {
            self.param_transition_rate = 0.1; 
        }
    }

    
    fn determine_auto_balance_state(
        &self,
        max_distance: f32,
        boundary_nodes: u32,
        total_nodes: usize,
        has_numerical_instability: bool,
        _has_spatial_issues: bool,
        config: &crate::config::AutoBalanceConfig,
    ) -> AutoBalanceState {
        
        if has_numerical_instability {
            return AutoBalanceState::Adjusting;
        }

        
        if boundary_nodes as f32 > (total_nodes as f32 * config.bouncing_node_percentage) {
            return AutoBalanceState::Bouncing;
        }

        
        if self.auto_balance_history.len() >= config.oscillation_detection_frames {
            let recent = &self.auto_balance_history
                [self.auto_balance_history.len() - config.oscillation_detection_frames..];
            let changes = recent
                .windows(2)
                .filter(|w| (w[0] - w[1]).abs() > config.oscillation_change_threshold)
                .count();
            if changes > config.min_oscillation_changes {
                return AutoBalanceState::Oscillating;
            }
        }

        
        match self.current_state {
            AutoBalanceState::Spreading => {
                
                if max_distance
                    < (config.spreading_distance_threshold - config.spreading_hysteresis_buffer)
                {
                    if max_distance
                        < (config.clustering_distance_threshold
                            + config.clustering_hysteresis_buffer)
                    {
                        AutoBalanceState::Clustering
                    } else {
                        AutoBalanceState::Stable
                    }
                } else {
                    AutoBalanceState::Spreading 
                }
            }
            AutoBalanceState::Clustering => {
                
                if max_distance
                    > (config.clustering_distance_threshold + config.clustering_hysteresis_buffer)
                {
                    if max_distance
                        > (config.spreading_distance_threshold - config.spreading_hysteresis_buffer)
                    {
                        AutoBalanceState::Spreading
                    } else {
                        AutoBalanceState::Stable
                    }
                } else {
                    AutoBalanceState::Clustering 
                }
            }
            _ => {
                
                if max_distance > config.spreading_distance_threshold {
                    AutoBalanceState::Spreading
                } else if max_distance < config.clustering_distance_threshold {
                    AutoBalanceState::Clustering
                } else {
                    AutoBalanceState::Stable
                }
            }
        }
    }

    
    fn apply_gradual_adjustment(
        &mut self,
        state: AutoBalanceState,
        config: &crate::config::AutoBalanceConfig,
    ) -> bool {
        let mut adjustment_made = false;
        let adjustment_rate = config.parameter_adjustment_rate;

        match state {
            AutoBalanceState::Spreading => {
                
                let mut new_target = self.target_params.clone();

                let attraction_factor = 1.0 + adjustment_rate;
                new_target.spring_k = (self.simulation_params.spring_k * attraction_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));

                let spring_factor = 1.0 + adjustment_rate * 0.5; 
                new_target.spring_k = (self.simulation_params.spring_k * spring_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));

                self.set_target_params(new_target);
                self.send_auto_balance_notification(
                    "Gradual adjustment: Increasing attraction to counter spreading",
                );
                adjustment_made = true;
            }
            AutoBalanceState::Clustering => {
                
                let mut new_target = self.target_params.clone();

                let repulsion_factor = 1.0 + adjustment_rate;
                new_target.repel_k = (self.simulation_params.repel_k * repulsion_factor)
                    .max(self.simulation_params.repel_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.repel_k * (1.0 + config.max_adjustment_factor));

                
                let attraction_factor = 1.0 - adjustment_rate * 0.5;
                new_target.spring_k = (self.simulation_params.spring_k * attraction_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));

                self.set_target_params(new_target);
                self.send_auto_balance_notification(
                    "Gradual adjustment: Increasing repulsion to counter clustering",
                );
                adjustment_made = true;
            }
            AutoBalanceState::Bouncing => {
                
                let mut new_target = self.target_params.clone();

                let damping_factor = 1.0 + adjustment_rate * 0.5;
                new_target.damping = (self.simulation_params.damping * damping_factor).min(0.99);

                let velocity_factor = 1.0 - adjustment_rate * 0.5;
                new_target.max_velocity =
                    (self.simulation_params.max_velocity * velocity_factor).max(1.0);

                self.set_target_params(new_target);
                self.send_auto_balance_notification(
                    "Gradual adjustment: Increasing damping to reduce bouncing",
                );
                adjustment_made = true;
            }
            AutoBalanceState::Oscillating => {
                
                let mut new_target = self.target_params.clone();
                new_target.damping = (self.simulation_params.damping * 1.2).min(0.98);
                new_target.max_velocity = self.simulation_params.max_velocity * 0.7;

                
                self.param_transition_rate = config.parameter_dampening_factor;

                self.set_target_params(new_target);
                self.send_auto_balance_notification(
                    "Emergency adjustment: Stopping oscillation with increased damping",
                );
                adjustment_made = true;
            }
            AutoBalanceState::Adjusting => {
                
                let mut new_target = self.target_params.clone();
                new_target.dt = (self.simulation_params.dt * 0.8).max(0.001);
                new_target.damping = (self.simulation_params.damping * 1.1).min(0.99);

                self.set_target_params(new_target);
                self.send_auto_balance_notification(
                    "Emergency adjustment: Fixing numerical instability",
                );
                adjustment_made = true;
            }
            AutoBalanceState::Stable => {
                
                self.param_transition_rate = config.parameter_dampening_factor;
                
            }
        }

        if adjustment_made {
            self.last_adjustment_time = std::time::Instant::now();
            self.current_state = state;
        }

        adjustment_made
    }

    fn send_auto_balance_notification(&self, message: &str) {
        info!("[AUTO-BALANCE NOTIFICATION] {}", message);

        
        let severity = if message.contains("disabled") || message.contains("failed") {
            "warning"
        } else if message.contains("stable") || message.contains("found") {
            "success"
        } else {
            "info"
        }
        .to_string();

        
        let notification = AutoBalanceNotification {
            message: message.to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            severity,
        };

        if let Ok(mut notifications) = self.auto_balance_notifications.lock() {
            notifications.push(notification);
            
            if notifications.len() > 50 {
                let drain_count = notifications.len() - 50;
                notifications.drain(0..drain_count);
            }
        }
    }

    fn notify_settings_update(&self) {
        
        
        info!("[AUTO-BALANCE] Notifying settings system of parameter changes");

        
        if let Some(settings_addr) = self.settings_addr.as_ref() {
            let physics_update = serde_json::json!({
                "visualisation": {
                    "graphs": {
                        "logseq": {
                            "physics": {
                                "repelK": self.simulation_params.repel_k,
                                "damping": self.simulation_params.damping,
                                "maxVelocity": self.simulation_params.max_velocity,
                                "springK": self.simulation_params.spring_k,
                                "enableBounds": self.simulation_params.enable_bounds,
                                "boundsSize": self.simulation_params.viewport_bounds,
                            }
                        }
                    }
                }
            });

            let update_msg =
                crate::actors::messages::UpdatePhysicsFromAutoBalance { physics_update };
            settings_addr.do_send(update_msg);
        }
    }

    pub fn new(
        client_manager: Addr<ClientCoordinatorActor>,
        gpu_compute_addr: Option<Addr<GPUManagerActor>>,
        kg_repo: Arc<dyn KnowledgeGraphRepository>,
        settings_addr: Option<
            Addr<crate::actors::optimized_settings_actor::OptimizedSettingsActor>,
        >,
    ) -> Self {
        let advanced_params = AdvancedParams::default();
        let semantic_analyzer = SemanticAnalyzer::new(
            crate::services::semantic_analyzer::SemanticAnalyzerConfig::default(),
        );
        let edge_generator = AdvancedEdgeGenerator::new(EdgeGenerationConfig::default());
        let stress_solver = StressMajorizationSolver::from_advanced_params(&advanced_params);

        
        
        let simulation_params = {
            info!("Using default SimulationParams (database-first architecture)");
            info!("Settings will be loaded from database via DatabaseService");
            SimulationParams::default()
        };

        
        let target_params = simulation_params.clone();

        Self {
            graph_data: Arc::new(GraphData::new()), 
            node_map: Arc::new(HashMap::new()),     
            gpu_compute_addr,
            kg_repo,
            client_manager,
            simulation_running: AtomicBool::new(false),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            next_node_id: AtomicU32::new(1),
            bots_graph_data: Arc::new(GraphData::new()), 
            simulation_params,                           

            
            gpu_init_in_progress: false,
            gpu_initialized: false, 
            constraint_set: ConstraintSet::default(),
            semantic_analyzer,
            edge_generator,
            stress_solver,
            semantic_features_cache: HashMap::new(),
            advanced_params,

            
            stress_step_counter: 0,
            constraint_update_counter: 0,
            last_semantic_analysis: None,

            
            settings_addr,
            auto_balance_history: Vec::with_capacity(60), 
            stable_count: 0,
            auto_balance_notifications: Arc::new(Mutex::new(Vec::new())),
            kinetic_energy_history: Vec::with_capacity(60), 
            last_adjustment_time: std::time::Instant::now(), 
            current_state: AutoBalanceState::Stable,        
            frames_since_last_broadcast: None,
            last_broadcast_time: None, 
            initial_positions_sent: false,

            
            target_params,
            param_transition_rate: 0.1, 

            
            previous_positions: HashMap::new(),

            
            pending_broadcasts: 0,
            max_pending_broadcasts: 5, 
            broadcast_skip_count: 0,
            last_backpressure_warning: None,

            
            update_queue: UpdateQueue::new(),
            queue_config: UpdateQueueConfig::default(),
            batch_metrics: BatchMetrics::default(),
        }
    }

    pub fn get_graph_data(&self) -> &GraphData {
        
        &self.graph_data 
    }

    pub fn get_node_map(&self) -> &HashMap<u32, Node> {
        &self.node_map
    }

    pub fn add_node(&mut self, node: Node) {
        let node_id = node.id; 

        
        if self.queue_config.enable_auto_flush {
            
            self.update_queue.add_node(node);

            
            if self.update_queue.should_flush(&self.queue_config) {
                if let Err(e) = self.flush_update_queue_internal(true) {
                    error!("Failed to flush update queue: {}", e);
                    
                }
            }
        } else {
            
            self.add_node_direct(node);
        }

        debug!("Added/updated node: {}", node_id);
    }

    
    fn add_node_direct(&mut self, node: Node) {
        
        Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());

        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        if !graph_data_mut.nodes.iter().any(|n| n.id == node.id) {
            graph_data_mut.nodes.push(node);
        } else {
            
            if let Some(existing) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node.id) {
                *existing = node; 
            }
        }
    }

    pub fn remove_node(&mut self, node_id: u32) {
        
        Arc::make_mut(&mut self.node_map).remove(&node_id);

        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        graph_data_mut.nodes.retain(|n| n.id != node_id);

        
        graph_data_mut
            .edges
            .retain(|e| e.source != node_id && e.target != node_id);

        
        self.previous_positions.remove(&node_id);

        debug!("Removed node: {}", node_id);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        let edge_id = edge.id.clone(); 

        
        if self.queue_config.enable_auto_flush {
            
            self.update_queue.add_edge(edge);

            
            if self.update_queue.should_flush(&self.queue_config) {
                if let Err(e) = self.flush_update_queue_internal(true) {
                    error!("Failed to flush update queue: {}", e);
                }
            }
        } else {
            
            self.add_edge_direct(edge);
        }

        debug!("Added/updated edge: {}", edge_id);
    }

    
    fn add_edge_direct(&mut self, edge: Edge) {
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        if !graph_data_mut.edges.iter().any(|e| e.id == edge.id) {
            graph_data_mut.edges.push(edge);
        } else {
            
            if let Some(existing) = graph_data_mut.edges.iter_mut().find(|e| e.id == edge.id) {
                *existing = edge; 
            }
        }
    }

    pub fn remove_edge(&mut self, edge_id: &str) {
        Arc::make_mut(&mut self.graph_data)
            .edges
            .retain(|e| e.id != edge_id);
        debug!("Removed edge: {}", edge_id);
    }

    
    
    pub fn batch_add_nodes(&mut self, nodes: Vec<Node>) -> Result<(), String> {
        if nodes.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();
        let node_count = nodes.len();

        
        let mut new_nodes = Vec::with_capacity(node_count);
        let mut updated_nodes = Vec::new();

        
        let node_map_mut = Arc::make_mut(&mut self.node_map);

        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        
        let existing_node_ids: std::collections::HashSet<u32> =
            graph_data_mut.nodes.iter().map(|n| n.id).collect();

        
        for node in nodes {
            let node_id = node.id;

            
            node_map_mut.insert(node_id, node.clone());

            
            if existing_node_ids.contains(&node_id) {
                updated_nodes.push(node);
            } else {
                new_nodes.push(node);
            }
        }

        
        graph_data_mut.nodes.extend(new_nodes);

        
        for update_node in updated_nodes {
            if let Some(existing) = graph_data_mut
                .nodes
                .iter_mut()
                .find(|n| n.id == update_node.id)
            {
                *existing = update_node;
            }
        }

        let duration = start_time.elapsed();
        self.batch_metrics
            .record_batch(node_count, 0, duration, false);

        debug!(
            "Batch added {} nodes in {}ms",
            node_count,
            duration.as_millis()
        );
        Ok(())
    }

    
    
    pub fn batch_add_edges(&mut self, edges: Vec<Edge>) -> Result<(), String> {
        if edges.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();
        let edge_count = edges.len();

        
        let mut new_edges = Vec::with_capacity(edge_count);
        let mut updated_edges = Vec::new();

        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        
        let existing_edge_ids: std::collections::HashSet<String> =
            graph_data_mut.edges.iter().map(|e| e.id.clone()).collect();

        
        for edge in edges {
            
            if existing_edge_ids.contains(&edge.id) {
                updated_edges.push(edge);
            } else {
                new_edges.push(edge);
            }
        }

        
        graph_data_mut.edges.extend(new_edges);

        
        for update_edge in updated_edges {
            if let Some(existing) = graph_data_mut
                .edges
                .iter_mut()
                .find(|e| e.id == update_edge.id)
            {
                *existing = update_edge;
            }
        }

        let duration = start_time.elapsed();
        self.batch_metrics
            .record_batch(0, edge_count, duration, false);

        debug!(
            "Batch added {} edges in {}ms",
            edge_count,
            duration.as_millis()
        );
        Ok(())
    }

    
    pub fn batch_graph_update(
        &mut self,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
        remove_node_ids: Vec<u32>,
        remove_edge_ids: Vec<String>,
    ) -> Result<(), String> {
        let start_time = std::time::Instant::now();
        let total_operations =
            nodes.len() + edges.len() + remove_node_ids.len() + remove_edge_ids.len();

        if total_operations == 0 {
            return Ok(());
        }

        
        let node_map_mut = Arc::make_mut(&mut self.node_map);
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        
        for node_id in &remove_node_ids {
            node_map_mut.remove(node_id);
            graph_data_mut.nodes.retain(|n| n.id != *node_id);
            graph_data_mut
                .edges
                .retain(|e| e.source != *node_id && e.target != *node_id);
            self.previous_positions.remove(node_id);
        }

        
        for edge_id in &remove_edge_ids {
            graph_data_mut.edges.retain(|e| e.id != *edge_id);
        }

        
        let nodes_len = nodes.len();
        let edges_len = edges.len();

        
        for node in nodes {
            node_map_mut.insert(node.id, node.clone());

            if !graph_data_mut.nodes.iter().any(|n| n.id == node.id) {
                graph_data_mut.nodes.push(node);
            } else {
                if let Some(existing) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node.id) {
                    *existing = node;
                }
            }
        }

        
        for edge in edges {
            if !graph_data_mut.edges.iter().any(|e| e.id == edge.id) {
                graph_data_mut.edges.push(edge);
            } else {
                if let Some(existing) = graph_data_mut.edges.iter_mut().find(|e| e.id == edge.id) {
                    *existing = edge;
                }
            }
        }

        let duration = start_time.elapsed();
        self.batch_metrics.record_batch(
            nodes_len + remove_node_ids.len(),
            edges_len + remove_edge_ids.len(),
            duration,
            false,
        );

        debug!(
            "Batch operation completed: {} total operations in {}ms",
            total_operations,
            duration.as_millis()
        );
        Ok(())
    }

    
    pub fn queue_add_node(&mut self, node: Node) -> Result<(), String> {
        self.update_queue.add_node(node);

        if self.update_queue.should_flush(&self.queue_config) {
            self.flush_update_queue_internal(true)?;
        }

        Ok(())
    }

    
    pub fn queue_add_edge(&mut self, edge: Edge) -> Result<(), String> {
        self.update_queue.add_edge(edge);

        if self.update_queue.should_flush(&self.queue_config) {
            self.flush_update_queue_internal(true)?;
        }

        Ok(())
    }

    
    pub fn flush_update_queue(&mut self) -> Result<(), String> {
        self.flush_update_queue_internal(false)
    }

    
    
    fn flush_update_queue_internal(&mut self, is_auto_flush: bool) -> Result<(), String> {
        if self.update_queue.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();

        
        let node_count = self.update_queue.pending_nodes.len();
        let edge_count = self.update_queue.pending_edges.len();
        let _removal_count = self.update_queue.pending_node_removals.len()
            + self.update_queue.pending_edge_removals.len();

        
        let nodes = std::mem::take(&mut self.update_queue.pending_nodes);
        let edges = std::mem::take(&mut self.update_queue.pending_edges);
        let remove_node_ids = std::mem::take(&mut self.update_queue.pending_node_removals);
        let remove_edge_ids = std::mem::take(&mut self.update_queue.pending_edge_removals);

        
        if !remove_node_ids.is_empty()
            || !remove_edge_ids.is_empty()
            || (!nodes.is_empty() && !edges.is_empty())
        {
            
            self.batch_graph_update(nodes, edges, remove_node_ids, remove_edge_ids)?;
        } else {
            
            if !nodes.is_empty() {
                self.batch_add_nodes(nodes)?;
            }
            if !edges.is_empty() {
                self.batch_add_edges(edges)?;
            }
        }

        
        self.update_queue.clear();

        let duration = start_time.elapsed();
        self.batch_metrics
            .record_batch(node_count, edge_count, duration, is_auto_flush);

        if is_auto_flush {
            trace!("Auto-flushed update queue in {}ms", duration.as_millis());
        } else {
            debug!(
                "Manually flushed update queue in {}ms",
                duration.as_millis()
            );
        }

        Ok(())
    }

    
    pub fn configure_update_queue(
        &mut self,
        max_operations: usize,
        flush_interval_ms: u64,
        enable_auto_flush: bool,
    ) {
        self.queue_config = UpdateQueueConfig {
            max_operations,
            flush_interval_ms,
            enable_auto_flush,
        };
        debug!(
            "Updated queue config: max_ops={}, interval={}ms, auto_flush={}",
            max_operations, flush_interval_ms, enable_auto_flush
        );
    }

    
    pub fn get_batch_metrics(&self) -> &BatchMetrics {
        &self.batch_metrics
    }

    
    
    pub fn batch_update_optimized(
        &mut self,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    ) -> Result<(), String> {
        if nodes.is_empty() && edges.is_empty() {
            return Ok(());
        }

        let start_time = std::time::Instant::now();
        let node_count = nodes.len();
        let edge_count = edges.len();
        let total_operations = node_count + edge_count;

        
        let node_map_mut = Arc::make_mut(&mut self.node_map);
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        
        if !nodes.is_empty() {
            
            let existing_node_ids: std::collections::HashSet<u32> =
                graph_data_mut.nodes.iter().map(|n| n.id).collect();

            let mut new_nodes = Vec::with_capacity(node_count);
            let mut updated_nodes = Vec::new();

            for node in nodes {
                let node_id = node.id;

                
                node_map_mut.insert(node_id, node.clone());

                
                if existing_node_ids.contains(&node_id) {
                    updated_nodes.push(node);
                } else {
                    new_nodes.push(node);
                }
            }

            
            graph_data_mut.nodes.extend(new_nodes);
            for update_node in updated_nodes {
                if let Some(existing) = graph_data_mut
                    .nodes
                    .iter_mut()
                    .find(|n| n.id == update_node.id)
                {
                    *existing = update_node;
                }
            }
        }

        
        if !edges.is_empty() {
            
            let existing_edge_ids: std::collections::HashSet<String> =
                graph_data_mut.edges.iter().map(|e| e.id.clone()).collect();

            let mut new_edges = Vec::with_capacity(edge_count);
            let mut updated_edges = Vec::new();

            for edge in edges {
                
                if existing_edge_ids.contains(&edge.id) {
                    updated_edges.push(edge);
                } else {
                    new_edges.push(edge);
                }
            }

            
            graph_data_mut.edges.extend(new_edges);
            for update_edge in updated_edges {
                if let Some(existing) = graph_data_mut
                    .edges
                    .iter_mut()
                    .find(|e| e.id == update_edge.id)
                {
                    *existing = update_edge;
                }
            }
        }

        let duration = start_time.elapsed();
        self.batch_metrics
            .record_batch(node_count, edge_count, duration, false);

        debug!(
            "Optimized batch update: {} operations in {}ms",
            total_operations,
            duration.as_millis()
        );
        Ok(())
    }

    
    
    pub fn queue_batch_operations(
        &mut self,
        nodes: Vec<Node>,
        edges: Vec<Edge>,
    ) -> Result<(), String> {
        
        for node in nodes {
            self.update_queue.add_node(node);
        }

        
        for edge in edges {
            self.update_queue.add_edge(edge);
        }

        
        if self.update_queue.should_flush(&self.queue_config) {
            self.flush_update_queue_internal(true)?;
        }

        Ok(())
    }

    
    
    pub fn force_flush_with_metrics(
        &mut self,
    ) -> Result<(usize, usize, std::time::Duration), String> {
        let start_time = std::time::Instant::now();
        let node_count = self.update_queue.pending_nodes.len();
        let edge_count = self.update_queue.pending_edges.len();

        self.flush_update_queue_internal(false)?;

        let duration = start_time.elapsed();
        Ok((node_count, edge_count, duration))
    }

    
    
    pub fn configure_for_high_throughput(&mut self) {
        self.queue_config = UpdateQueueConfig {
            max_operations: 5000,  
            flush_interval_ms: 50, 
            enable_auto_flush: true,
        };
        debug!("Configured queue for high-throughput mode: max_ops=5000, interval=50ms");
    }

    
    
    pub fn configure_for_memory_conservation(&mut self) {
        self.queue_config = UpdateQueueConfig {
            max_operations: 500,    
            flush_interval_ms: 200, 
            enable_auto_flush: true,
        };
        debug!("Configured queue for memory conservation: max_ops=500, interval=200ms");
    }

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    pub fn build_from_metadata(
        &mut self,
        metadata: MetadataStore,
        ctx: &mut Context<Self>,
    ) -> Result<(), String> {
        let mut new_graph_data = GraphData::new();

        
        
        let mut existing_positions: HashMap<
            String,
            (crate::types::vec3::Vec3Data, crate::types::vec3::Vec3Data),
        > = HashMap::new();

        
        for node in self.node_map.values() {
            existing_positions.insert(
                node.metadata_id.clone(),
                (node.data.position(), node.data.velocity()),
            );
            debug!(
                "Saved position for existing node '{}': ({}, {}, {})",
                node.metadata_id, node.data.x, node.data.y, node.data.z
            );
        }
        debug!(
            "Total existing positions saved: {}",
            existing_positions.len()
        );

        Arc::make_mut(&mut self.node_map).clear();
        self.semantic_features_cache.clear();

        
        info!("Phase 1: Building nodes with semantic analysis");
        info!("Metadata contains {} entries", metadata.len());

        if metadata.is_empty() {
            warn!("Metadata is empty! No nodes will be created.");
        }

        let mut node_count = 0;
        for (filename_with_ext, file_meta_data) in &metadata {
            node_count += 1;
            let node_id_val = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            let metadata_id_val = filename_with_ext.trim_end_matches(".md").to_string();

            let mut node = Node::new_with_id(metadata_id_val.clone(), Some(node_id_val));
            node.label = file_meta_data.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(file_meta_data.file_size as u64);
            
            

            
            
            debug!(
                "Looking for existing position for metadata_id: '{}'",
                metadata_id_val
            );
            debug!(
                "Available keys in existing_positions: {:?}",
                existing_positions.keys().collect::<Vec<_>>()
            );

            if let Some((saved_position, saved_velocity)) = existing_positions.get(&metadata_id_val)
            {
                node.data.x = saved_position.x;
                node.data.y = saved_position.y;
                node.data.z = saved_position.z;
                node.data.vx = saved_velocity.x;
                node.data.vy = saved_velocity.y;
                node.data.vz = saved_velocity.z;
                debug!(
                    "Restored position for node '{}': ({}, {}, {})",
                    metadata_id_val, saved_position.x, saved_position.y, saved_position.z
                );
            } else {
                debug!(
                    "New node '{}' will use generated position: ({}, {}, {})",
                    metadata_id_val, node.data.x, node.data.y, node.data.z
                );
            }

            
            node.metadata
                .insert("fileName".to_string(), file_meta_data.file_name.clone());
            node.metadata
                .insert("fileSize".to_string(), file_meta_data.file_size.to_string());
            node.metadata
                .insert("nodeSize".to_string(), file_meta_data.node_size.to_string());
            node.metadata.insert(
                "hyperlinkCount".to_string(),
                file_meta_data.hyperlink_count.to_string(),
            );
            node.metadata
                .insert("sha1".to_string(), file_meta_data.sha1.clone());
            node.metadata.insert(
                "lastModified".to_string(),
                file_meta_data.last_modified.to_rfc3339(),
            );
            if !file_meta_data.perplexity_link.is_empty() {
                node.metadata.insert(
                    "perplexityLink".to_string(),
                    file_meta_data.perplexity_link.clone(),
                );
            }
            if let Some(last_process) = file_meta_data.last_perplexity_process {
                node.metadata.insert(
                    "lastPerplexityProcess".to_string(),
                    last_process.to_rfc3339(),
                );
            }
            node.metadata
                .insert("metadataId".to_string(), metadata_id_val.clone());

            
            let features = self.semantic_analyzer.analyze_metadata(file_meta_data);
            self.semantic_features_cache
                .insert(metadata_id_val, features);

            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
            new_graph_data.nodes.push(node);
        }

        info!(
            "Phase 1 complete: Processed {} nodes from metadata",
            node_count
        );
        info!(
            "new_graph_data now contains {} nodes",
            new_graph_data.nodes.len()
        );
        info!("node_map now contains {} entries", self.node_map.len());

        
        info!("Phase 2: Generating enhanced edges with multi-modal similarities");
        info!(
            "Semantic features cache contains {} entries",
            self.semantic_features_cache.len()
        );
        for (id, features) in &self.semantic_features_cache {
            debug!(
                "Semantic feature ID: '{}' (topics: {})",
                id,
                features.topics.len()
            );
        }

        let enhanced_edges = self.edge_generator.generate(&self.semantic_features_cache);
        info!("Generated {} enhanced edges", enhanced_edges.len());

        
        let mut edge_map: HashMap<(u32, u32), f32> = HashMap::new();

        
        for enhanced_edge in &enhanced_edges {
            debug!(
                "Processing enhanced edge: {} -> {} (weight: {})",
                enhanced_edge.source, enhanced_edge.target, enhanced_edge.weight
            );

            
            if let (Some(source_node), Some(target_node)) = (
                self.node_map
                    .values()
                    .find(|n| n.metadata_id == enhanced_edge.source),
                self.node_map
                    .values()
                    .find(|n| n.metadata_id == enhanced_edge.target),
            ) {
                let edge_key = if source_node.id < target_node.id {
                    (source_node.id, target_node.id)
                } else {
                    (target_node.id, source_node.id)
                };
                
                *edge_map.entry(edge_key).or_insert(0.0) += enhanced_edge.weight;
            }
        }

        
        for (source_filename_ext, source_meta) in &metadata {
            let source_metadata_id = source_filename_ext.trim_end_matches(".md");
            if let Some(source_node) = self
                .node_map
                .values()
                .find(|n| n.metadata_id == source_metadata_id)
            {
                for (target_filename_ext, count) in &source_meta.topic_counts {
                    let target_metadata_id = target_filename_ext.trim_end_matches(".md");
                    if let Some(target_node) = self
                        .node_map
                        .values()
                        .find(|n| n.metadata_id == target_metadata_id)
                    {
                        if source_node.id != target_node.id {
                            let edge_key = if source_node.id < target_node.id {
                                (source_node.id, target_node.id)
                            } else {
                                (target_node.id, source_node.id)
                            };
                            
                            *edge_map.entry(edge_key).or_insert(0.0) += (*count as f32) * 0.3;
                        }
                    }
                }
            }
        }

        
        for ((source_id, target_id), weight) in edge_map {
            new_graph_data
                .edges
                .push(Edge::new(source_id, target_id, weight));
        }

        
        info!("Phase 3: Generating initial semantic constraints");

        
        self.graph_data = Arc::new(new_graph_data.clone());

        
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone, ctx);

        
        
        if self.gpu_compute_addr.is_none() {
            trace!("Advanced GPU context will be initialized on first physics step");
        }

        
        Arc::make_mut(&mut self.graph_data).metadata = metadata.clone();
        self.last_semantic_analysis = Some(std::time::Instant::now());

        info!(
            "Built enhanced graph: {} nodes, {} edges, {} constraints",
            self.graph_data.nodes.len(),
            self.graph_data.edges.len(),
            self.constraint_set.constraints.len()
        );

        

        Ok(())
    }

    

    fn prepare_node_positions(&self) -> Vec<(f32, f32, f32)> {
        self.graph_data
            .nodes
            .iter()
            .map(|node| (node.data.x, node.data.y, node.data.z))
            .collect()
    }

    fn execute_stress_majorization_step(&mut self) {
        if self.graph_data.nodes.len() < 3 {
            return; 
        }

        
        if self.gpu_compute_addr.is_none() {
            trace!("Skipping stress majorization - no GPU context available");
            return;
        }

        let mut graph_data_clone = self.graph_data.as_ref().clone(); 

        match self
            .stress_solver
            .optimize(&mut graph_data_clone, &self.constraint_set)
        {
            Ok(result) => {
                if result.converged || result.final_stress < f32::INFINITY {
                    
                    let graph_data_mut = Arc::make_mut(&mut self.graph_data);
                    for (i, node) in graph_data_mut.nodes.iter_mut().enumerate() {
                        if let Some(optimized_node) = graph_data_clone.nodes.get(i) {
                            
                            let new_x = optimized_node.data.x;
                            let new_y = optimized_node.data.y;
                            let new_z = optimized_node.data.z;

                            if new_x.is_finite() && new_y.is_finite() && new_z.is_finite() {
                                
                                let old_pos = Vec3Data::new(node.data.x, node.data.y, node.data.z);
                                let displacement_x = new_x - old_pos.x;
                                let displacement_y = new_y - old_pos.y;
                                let displacement_z = new_z - old_pos.z;
                                let displacement_magnitude = (displacement_x * displacement_x
                                    + displacement_y * displacement_y
                                    + displacement_z * displacement_z)
                                    .sqrt();

                                
                                let layout_extent =
                                    self.simulation_params.viewport_bounds.max(1000.0);
                                let max_displacement = layout_extent * 0.05;

                                let (final_x, final_y, final_z) =
                                    if displacement_magnitude > max_displacement {
                                        
                                        let scale = max_displacement / displacement_magnitude;
                                        (
                                            old_pos.x + displacement_x * scale,
                                            old_pos.y + displacement_y * scale,
                                            old_pos.z + displacement_z * scale,
                                        )
                                    } else {
                                        (new_x, new_y, new_z)
                                    };

                                
                                if self.simulation_params.enable_bounds
                                    && self.simulation_params.viewport_bounds > 0.0
                                {
                                    let boundary_limit = self.simulation_params.viewport_bounds;
                                    node.data.x = final_x.clamp(-boundary_limit, boundary_limit);
                                    node.data.y = final_y.clamp(-boundary_limit, boundary_limit);
                                    node.data.z = final_z.clamp(-boundary_limit, boundary_limit);
                                } else {
                                    
                                    let default_bound = 10000.0;
                                    node.data.x = final_x.clamp(-default_bound, default_bound);
                                    node.data.y = final_y.clamp(-default_bound, default_bound);
                                    node.data.z = final_z.clamp(-default_bound, default_bound);
                                }
                            } else {
                                warn!("Skipping invalid position from stress majorization for node {}: ({}, {}, {})",
                                      node.id, new_x, new_y, new_z);
                            }
                        }
                    }

                    
                    for node in &graph_data_mut.nodes {
                        if let Some(node_in_map) =
                            Arc::make_mut(&mut self.node_map).get_mut(&node.id)
                        {
                            
                            node_in_map.data.x = node.data.x;
                            node_in_map.data.y = node.data.y;
                            node_in_map.data.z = node.data.z;
                        }
                    }

                    trace!(
                        "Stress majorization step completed: {} iterations, stress = {:.6}",
                        result.iterations,
                        result.final_stress
                    );
                }
            }
            Err(e) => {
                error!("Stress majorization step failed: {}", e);
            }
        }
    }

    fn update_dynamic_constraints(&mut self, ctx: &mut Context<Self>) {
        
        trace!("Updating dynamic constraints based on semantic analysis");

        
        if self.last_semantic_analysis.is_none() {
            return;
        }

        
        self.constraint_set
            .set_group_active("semantic_dynamic", false);

        
        if let Ok(constraints) = self.generate_dynamic_semantic_constraints() {
            let constraint_count = constraints.len();
            for constraint in constraints {
                self.constraint_set
                    .add_to_group("semantic_dynamic", constraint);
            }
            trace!("Updated {} dynamic semantic constraints", constraint_count);
        } else {
            trace!("Failed to generate dynamic semantic constraints");
        }

        
        if let Ok(clustering_constraints) = self.generate_clustering_constraints() {
            self.constraint_set
                .set_group_active("clustering_dynamic", false);
            let constraint_count = clustering_constraints.len();
            for constraint in clustering_constraints {
                self.constraint_set
                    .add_to_group("clustering_dynamic", constraint);
            }
            trace!(
                "Updated {} dynamic clustering constraints",
                constraint_count
            );
        } else {
            trace!("Failed to generate dynamic clustering constraints");
        }

        
        self.upload_constraints_to_gpu(ctx);
    }

    fn generate_initial_semantic_constraints(
        &mut self,
        graph_data: &std::sync::Arc<GraphData>,
        ctx: &mut Context<Self>,
    ) {
        
        let mut domain_clusters: HashMap<String, Vec<u32>> = HashMap::new();

        for node in &graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                if !features.domains.is_empty() {
                    let domain_key = format!("{:?}", features.domains[0]);
                    domain_clusters
                        .entry(domain_key)
                        .or_insert_with(Vec::new)
                        .push(node.id);
                }
            }
        }

        
        self.constraint_set
            .set_group_active("domain_clustering", false);

        
        for (domain, node_ids) in domain_clusters {
            if node_ids.len() >= 2 {
                let cluster_constraint = Constraint::cluster(
                    node_ids,
                    domain.len() as f32, 
                    0.6,                 
                );
                self.constraint_set
                    .add_to_group("domain_clustering", cluster_constraint);
            }
        }

        
        self.upload_constraints_to_gpu(ctx);

        info!(
            "Generated {} initial semantic constraints",
            self.constraint_set.active_constraints().len()
        );
    }

    fn generate_dynamic_semantic_constraints(
        &self,
    ) -> Result<Vec<Constraint>, Box<dyn std::error::Error>> {
        let mut constraints = Vec::new();

        
        let high_importance_nodes: Vec<_> = self
            .semantic_features_cache
            .iter()
            .filter(|(_, features)| features.importance_score > 0.7)
            .filter_map(|(id, _)| {
                self.node_map
                    .values()
                    .find(|n| n.metadata_id == *id)
                    .map(|n| n.id)
            })
            .collect();

        for i in 0..high_importance_nodes.len() {
            for j in i + 1..high_importance_nodes.len() {
                let constraint = Constraint::separation(
                    high_importance_nodes[i],
                    high_importance_nodes[j],
                    100.0, 
                );
                constraints.push(constraint);
            }
        }

        Ok(constraints)
    }

    fn generate_clustering_constraints(
        &self,
    ) -> Result<Vec<Constraint>, Box<dyn std::error::Error>> {
        let mut constraints = Vec::new();

        
        let mut type_clusters: HashMap<String, Vec<u32>> = HashMap::new();

        for node in &self.graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                type_clusters
                    .entry(features.content.language.clone())
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }

        
        for (file_type, node_ids) in type_clusters {
            if node_ids.len() >= 2 {
                let constraint = Constraint::cluster(
                    node_ids,
                    file_type.len() as f32, 
                    0.4,                    
                );
                constraints.push(constraint);
            }
        }

        Ok(constraints)
    }

    
    fn upload_constraints_to_gpu(&mut self, ctx: &mut Context<Self>) {
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            
            let active_constraints = self.constraint_set.active_constraints();
            let constraint_data: Vec<crate::models::constraints::ConstraintData> =
                active_constraints
                    .iter()
                    .map(|c| c.to_gpu_format())
                    .collect();

            
            let upload_msg = crate::actors::messages::UploadConstraintsToGPU {
                constraint_data: constraint_data.clone(),
            };

            let gpu_addr_clone = gpu_addr.clone();
            ctx.spawn(
                async move {
                    if let Err(e) = gpu_addr_clone.send(upload_msg).await {
                        error!("Failed to send constraints to ForceComputeActor: {}", e);
                    } else {
                        trace!(
                            "Successfully sent {} constraints to ForceComputeActor",
                            constraint_data.len()
                        );
                    }
                }
                .into_actor(self),
            );
        } else {
            trace!("No GPU compute actor available for constraint upload");
        }
    }

    
    pub fn handle_constraint_update(
        &mut self,
        constraint_data: serde_json::Value,
        ctx: &mut Context<Self>,
    ) -> Result<(), String> {
        match constraint_data.get("action").and_then(|v| v.as_str()) {
            Some("add_fixed_position") => {
                if let (Some(node_id), Some(x), Some(y), Some(z)) = (
                    constraint_data
                        .get("node_id")
                        .and_then(|v| v.as_u64())
                        .map(|v| v as u32),
                    constraint_data
                        .get("x")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32),
                    constraint_data
                        .get("y")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32),
                    constraint_data
                        .get("z")
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32),
                ) {
                    let constraint = Constraint::fixed_position(node_id, x, y, z);
                    self.constraint_set.add_to_group("user_fixed", constraint);
                    info!("Added fixed position constraint for node {}", node_id);
                }
            }
            Some("toggle_clustering") => {
                if let Some(enabled) = constraint_data.get("enabled").and_then(|v| v.as_bool()) {
                    self.constraint_set
                        .set_group_active("domain_clustering", enabled);
                    info!("Toggled domain clustering: {}", enabled);
                }
            }
            Some("update_separation_factor") => {
                if let Some(factor) = constraint_data
                    .get("factor")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32)
                {
                    self.advanced_params.separation_factor = factor;
                    info!("Updated separation factor to {}", factor);
                }
            }
            Some("enable_hierarchical") => {
                if let Some(enabled) = constraint_data.get("enabled").and_then(|v| v.as_bool()) {
                    self.advanced_params.hierarchical_mode = enabled;
                    if enabled {
                        
                        self.generate_hierarchical_constraints();
                    }
                    info!("Set hierarchical mode: {}", enabled);
                }
            }
            _ => {
                warn!(
                    "Unknown constraint update action: {:?}",
                    constraint_data.get("action")
                );
                return Err("Unknown constraint action".to_string());
            }
        }

        
        self.upload_constraints_to_gpu(ctx);

        Ok(())
    }

    fn generate_hierarchical_constraints(&mut self) {
        
        let mut depth_layers: HashMap<u32, Vec<u32>> = HashMap::new();

        for node in &self.graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                depth_layers
                    .entry(features.structural.directory_depth)
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }

        
        for (depth, node_ids) in &depth_layers {
            if node_ids.len() >= 2 {
                let z_position = -(*depth as f32) * self.advanced_params.layer_separation;
                let constraint = Constraint {
                    kind: crate::models::constraints::ConstraintKind::AlignmentDepth,
                    node_indices: node_ids.clone(),
                    params: vec![z_position],
                    weight: 0.8,
                    active: true,
                };
                self.constraint_set
                    .add_to_group("hierarchical_layers", constraint);
            }
        }

        info!(
            "Generated hierarchical layer constraints for {} depths",
            depth_layers.len()
        );
    }

    
    fn detect_spatial_hashing_issues(
        &self,
        positions: &[(f32, f32, f32)],
        config: &AutoBalanceConfig,
    ) -> (bool, f32) {
        if positions.len() < 2 {
            return (false, 1.0);
        }

        let current_grid_cell_size = self.simulation_params.max_repulsion_dist; 
        let mut clustering_detected = false;
        let mut efficiency_score = 1.0;

        
        let mut total_distance = 0.0f32;
        let mut distance_count = 0;

        for i in 0..positions.len() {
            for j in i + 1..std::cmp::min(i + 10, positions.len()) {
                
                let pos1 = positions[i];
                let pos2 = positions[j];
                let dist = ((pos1.0 - pos2.0).powi(2)
                    + (pos1.1 - pos2.1).powi(2)
                    + (pos1.2 - pos2.2).powi(2))
                .sqrt();
                total_distance += dist;
                distance_count += 1;
            }
        }

        let avg_distance = if distance_count > 0 {
            total_distance / distance_count as f32
        } else {
            1.0
        };

        
        if avg_distance < current_grid_cell_size * 0.5 {
            clustering_detected = true;
            efficiency_score = avg_distance / current_grid_cell_size;
        }

        
        let cluster_density = positions.len() as f32 / (avg_distance * avg_distance);
        if cluster_density > config.cluster_density_threshold {
            clustering_detected = true;
            efficiency_score = efficiency_score.min(0.3);
        }

        (clustering_detected, efficiency_score)
    }

    
    fn detect_numerical_instability(
        &self,
        positions: &[(f32, f32, f32)],
        config: &AutoBalanceConfig,
    ) -> bool {
        
        for &(x, y, z) in positions {
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return true;
            }
        }

        
        if let Some(&recent_ke) = self.kinetic_energy_history.last() {
            if recent_ke > config.numerical_instability_threshold {
                
                if self.kinetic_energy_history.len() >= 5 {
                    let last_5: Vec<f32> = self
                        .kinetic_energy_history
                        .iter()
                        .rev()
                        .take(5)
                        .cloned()
                        .collect();
                    let is_growing = last_5.windows(2).all(|w| w[0] > w[1] * 1.5);
                    if is_growing {
                        return true;
                    }
                }
            }
        }

        false
    }

    
    fn calculate_adaptive_max_pending_broadcasts(&self) -> u32 {
        let node_count = self.node_map.len();
        match node_count {
            0..=999 => 10,    
            1000..=9999 => 5, 
            _ => 3,           
        }
    }

    
    pub fn acknowledge_broadcast(&mut self) {
        if self.pending_broadcasts > 0 {
            self.pending_broadcasts -= 1;
            debug!(
                "Broadcast acknowledged, pending: {}/{}",
                self.pending_broadcasts, self.max_pending_broadcasts
            );
        }
    }

    
    pub fn get_backpressure_metrics(&self) -> (u32, u32, u32) {
        (
            self.pending_broadcasts,
            self.max_pending_broadcasts,
            self.broadcast_skip_count,
        )
    }

    pub fn update_node_positions(&mut self, positions: Vec<(u32, BinaryNodeData)>) {
        
        
        if self.simulation_params.is_physics_paused {
            debug!(
                "Physics is paused, skipping position update for {} nodes",
                positions.len()
            );
            return;
        }

        
        self.max_pending_broadcasts = self.calculate_adaptive_max_pending_broadcasts();

        let mut updated_count = 0;
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        for (node_id, mut position_data) in positions {
            
            
            const MAX_COORD: f32 = 500.0;
            const MIN_Z: f32 = -50.0;
            const MAX_Z: f32 = 50.0;

            
            position_data.x = position_data.x.clamp(-MAX_COORD, MAX_COORD);
            position_data.y = position_data.y.clamp(-MAX_COORD, MAX_COORD);
            position_data.z = position_data.z.clamp(MIN_Z, MAX_Z);

            
            if position_data.z.abs() > 45.0 {
                debug!(
                    "Node {} has extreme z position: {}, clamped to range [{}, {}]",
                    node_id, position_data.z, MIN_Z, MAX_Z
                );
            }

            
            if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&node_id) {
                node.data.x = position_data.x;
                node.data.y = position_data.y;
                node.data.z = position_data.z;
                node.data.vx = position_data.vx;
                node.data.vy = position_data.vy;
                node.data.vz = position_data.vz;
                updated_count += 1;
            }

            
            if let Some(node) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node_id) {
                node.data.x = position_data.x;
                node.data.y = position_data.y;
                node.data.z = position_data.z;
                node.data.vx = position_data.vx;
                node.data.vy = position_data.vy;
                node.data.vz = position_data.vz;
            }
        }

        

        
        let config = &self.simulation_params.auto_balance_config;
        let mut extreme_count = 0;
        let mut boundary_nodes = 0;
        let mut max_distance = 0.0f32;
        let mut total_distance = 0.0f32;
        let mut positions = Vec::new();
        let mut total_kinetic_energy = 0.0f32;

        for (_, node) in self.node_map.iter() {
            let dist = node
                .data
                .x
                .abs()
                .max(node.data.y.abs())
                .max(node.data.z.abs());
            max_distance = max_distance.max(dist);
            total_distance += dist;
            positions.push(dist);

            
            
            let velocity_squared = node.data.vx * node.data.vx
                + node.data.vy * node.data.vy
                + node.data.vz * node.data.vz;
            total_kinetic_energy += 0.5 * velocity_squared;

            
            let viewport_bounds = self.simulation_params.viewport_bounds;
            let boundary_min_threshold = viewport_bounds * (config.boundary_min_distance / 100.0); 
            let boundary_max_threshold = viewport_bounds * (config.boundary_max_distance / 100.0); 

            if dist > config.extreme_distance_threshold {
                extreme_count += 1;
            } else if dist >= boundary_min_threshold && dist <= boundary_max_threshold {
                
                boundary_nodes += 1;
            }
        }

        let avg_distance = if !self.node_map.is_empty() {
            total_distance / self.node_map.len() as f32
        } else {
            0.0
        };

        
        self.check_and_handle_equilibrium(total_kinetic_energy, self.node_map.len());

        
        
        if self.simulation_params.auto_balance {
            
            
            if self.stable_count > 30 {
                debug!(
                    "Graph is stable (stable_count: {}), skipping auto-balance",
                    self.stable_count
                );
                return;
            }

            
            let avg_kinetic_energy = if !self.node_map.is_empty() {
                total_kinetic_energy / self.node_map.len() as f32
            } else {
                0.0
            };

            if crate::utils::logging::is_debug_enabled() {
                info!("[AUTO-BALANCE] Stats - max: {:.1}, avg: {:.1}, KE: {:.3}, boundary: {}/{}, extreme: {}/{}",
                      max_distance, avg_distance, avg_kinetic_energy, boundary_nodes, self.node_map.len(),
                      extreme_count, self.node_map.len());
            }

            
            self.auto_balance_history.push(max_distance);
            if self.auto_balance_history.len() > 60 {
                self.auto_balance_history.remove(0);
            }

            
            self.kinetic_energy_history.push(avg_kinetic_energy);
            if self.kinetic_energy_history.len() > 60 {
                self.kinetic_energy_history.remove(0);
            }

            
            let now = std::time::Instant::now();
            let config = &self.simulation_params.auto_balance_config;

            
            let adjustment_cooldown_duration =
                std::time::Duration::from_millis(config.adjustment_cooldown_ms);
            let time_since_last_adjustment = now.duration_since(self.last_adjustment_time);

            if time_since_last_adjustment >= adjustment_cooldown_duration {
                
                let position_data: Vec<(f32, f32, f32)> = self
                    .node_map
                    .values()
                    .map(|node| (node.data.x, node.data.y, node.data.z))
                    .collect();

                
                let (_has_spatial_issues, _efficiency_score) =
                    self.detect_spatial_hashing_issues(&position_data, config);
                let has_numerical_instability =
                    self.detect_numerical_instability(&position_data, config);

                
                let new_state = self.determine_auto_balance_state(
                    max_distance,
                    boundary_nodes,
                    self.node_map.len(),
                    has_numerical_instability,
                    _has_spatial_issues,
                    config,
                );

                
                let is_stable = if self.auto_balance_history.len() >= 30
                    && self.kinetic_energy_history.len() >= 30
                {
                    
                    let recent_avg = self.auto_balance_history
                        [self.auto_balance_history.len() - 30..]
                        .iter()
                        .sum::<f32>()
                        / 30.0;
                    let position_variance = self.auto_balance_history
                        [self.auto_balance_history.len() - 30..]
                        .iter()
                        .map(|x| (x - recent_avg).powi(2))
                        .sum::<f32>()
                        / 30.0;

                    
                    let recent_ke = self.kinetic_energy_history
                        [self.kinetic_energy_history.len() - 30..]
                        .iter()
                        .sum::<f32>()
                        / 30.0;
                    let ke_variance = self.kinetic_energy_history
                        [self.kinetic_energy_history.len() - 30..]
                        .iter()
                        .map(|x| (x - recent_ke).powi(2))
                        .sum::<f32>()
                        / 30.0;

                    
                    let ke_threshold = 0.01; 
                    let ke_variance_threshold = 0.001; 
                    position_variance < config.stability_variance_threshold
                        && recent_ke < ke_threshold
                        && ke_variance < ke_variance_threshold
                } else {
                    false
                };

                
                let new_state_clone = new_state.clone();
                if new_state != self.current_state || new_state != AutoBalanceState::Stable {
                    
                    

                    info!("[AUTO-BALANCE] State transition: {:?} -> {:?} (max_distance: {:.1}, boundary: {}/{})",
                          self.current_state, new_state_clone, max_distance, boundary_nodes, self.node_map.len());
                }

                
                if is_stable && new_state_clone == AutoBalanceState::Stable {
                    
                    self.stable_count += 1;

                    
                    if self.stable_count == config.stability_frame_count {
                        info!("[AUTO-BALANCE] Stable equilibrium found at {:.1} units - updating UI sliders", max_distance);

                        
                        self.send_auto_balance_notification(
                            "Auto-Balance: Stable equilibrium achieved!",
                        );

                        
                        self.notify_settings_update();

                        
                        self.stable_count = 181; 
                    } else if self.stable_count < 180 {
                        debug!("[AUTO-BALANCE] Stability detected for {} frames (need 180 for UI update)", self.stable_count);
                    }
                } else {
                    
                    if self.stable_count > 0 && self.stable_count < 180 {
                        debug!(
                            "[AUTO-BALANCE] Lost stability after {} frames",
                            self.stable_count
                        );
                    }
                    self.stable_count = 0;
                }
            }
        }

        
        

        
        let now = std::time::Instant::now();
        let should_broadcast = if let Some(last_time) = self.last_broadcast_time {
            
            
            let stable_broadcast_interval = std::time::Duration::from_millis(1000); 
            let active_broadcast_interval = std::time::Duration::from_millis(50); 

            let is_stable =
                self.current_state == AutoBalanceState::Stable && self.stable_count > 30;
            let required_interval = if is_stable {
                stable_broadcast_interval
            } else {
                active_broadcast_interval
            };

            now.duration_since(last_time) >= required_interval
        } else {
            
            true
        };

        
        
        let force_broadcast = !self.initial_positions_sent;

        if should_broadcast || force_broadcast {
            
            let backpressure_active =
                self.pending_broadcasts > self.max_pending_broadcasts && !force_broadcast;

            if backpressure_active {
                
                self.broadcast_skip_count += 1;

                
                let should_warn = if let Some(last_warning) = self.last_backpressure_warning {
                    now.duration_since(last_warning) >= std::time::Duration::from_secs(5)
                } else {
                    true
                };

                if should_warn {
                    warn!("Backpressure active: pending_broadcasts ({}) > max_pending_broadcasts ({}), skipped {} broadcasts total. Node count: {}",
                          self.pending_broadcasts, self.max_pending_broadcasts, self.broadcast_skip_count, self.node_map.len());
                    self.last_backpressure_warning = Some(now);
                }

                return; 
            }

            
            
            let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
            let mut knowledge_ids: Vec<u32> = Vec::new();
            let mut agent_ids: Vec<u32> = Vec::new();

            
            for (node_id, node) in self.node_map.iter() {
                position_data.push((
                    *node_id,
                    BinaryNodeDataClient::new(*node_id, node.data.position(), node.data.velocity()),
                ));
                knowledge_ids.push(*node_id);
            }

            
            
            for node in &self.bots_graph_data.nodes {
                position_data.push((
                    node.id,
                    BinaryNodeDataClient::new(
                        node.id,
                        glam_to_vec3data(Vec3::new(node.data.x, node.data.y, node.data.z)),
                        glam_to_vec3data(Vec3::new(node.data.vx, node.data.vy, node.data.vz)),
                    ),
                ));
                agent_ids.push(node.id);
            }

            
            if !position_data.is_empty() {
                
                
                let binary_data = crate::utils::binary_protocol::encode_node_data_with_types(
                    &position_data,
                    &agent_ids,
                    &knowledge_ids,
                );

                
                self.client_manager
                    .do_send(crate::actors::messages::BroadcastNodePositions {
                        positions: binary_data,
                    });

                
                self.pending_broadcasts += 1;

                
                self.last_broadcast_time = Some(now);
                if !self.initial_positions_sent {
                    self.initial_positions_sent = true;
                    info!("Sent initial unified graph positions to clients ({} nodes: {} knowledge + {} agents)",
                          position_data.len(), knowledge_ids.len(), agent_ids.len());
                } else if force_broadcast {
                    info!("Force broadcast unified graph positions to new clients ({} nodes: {} knowledge + {} agents)",
                          position_data.len(), knowledge_ids.len(), agent_ids.len());
                }

                if crate::utils::logging::is_debug_enabled() && !force_broadcast {
                    debug!("Broadcast unified positions: {} total ({} knowledge + {} agents), stable: {}, pending: {}/{}",
                           position_data.len(), knowledge_ids.len(), agent_ids.len(),
                           self.current_state == AutoBalanceState::Stable,
                           self.pending_broadcasts,
                           self.max_pending_broadcasts);
                }
            }
        }

        debug!("Updated positions for {} nodes", updated_count);

        
        
        
        
        
    }

    fn start_simulation_loop(&mut self, ctx: &mut Context<Self>) {
        if self.simulation_running.load(Ordering::SeqCst) {
            warn!("Simulation already running");
            return;
        }

        self.simulation_running.store(true, Ordering::SeqCst);
        if crate::utils::logging::is_debug_enabled() {
            info!("Starting physics simulation loop");
        }

        
        ctx.run_later(Duration::from_millis(100), |actor, ctx| {
            
            ctx.run_interval(Duration::from_millis(16), move |actor_ref, ctx_ref| {
                if !actor_ref.simulation_running.load(Ordering::SeqCst) {
                    return;
                }

                actor_ref.run_simulation_step(ctx_ref);
            });
        });
    }

    fn run_simulation_step(&mut self, ctx: &mut Context<Self>) {
        
        if !self.gpu_initialized && self.gpu_compute_addr.is_some() {
            
            if self.gpu_init_in_progress {
                
                return;
            }
            warn!("Skipping physics simulation - waiting for GPU initialization");
            return;
        }

        
        
        if self.simulation_params.auto_balance {
            self.smooth_transition_params();
        }

        
        self.stress_step_counter += 1;
        self.constraint_update_counter += 1;

        
        
        
        
        
        

        
        
        
        
        
        

        
        if self.gpu_compute_addr.is_none()
            && !self.gpu_init_in_progress
            && !self.graph_data.nodes.is_empty()
        {
            
            self.gpu_init_in_progress = true;

            
            let graph_data_clone = Arc::clone(&self.graph_data);

            ctx.run_later(Duration::from_secs(2), |actor, ctx| {
                let self_addr = ctx.address();
                let graph_data_clone = Arc::clone(&actor.graph_data);

                ctx.spawn(
                    async move {
                        info!("Starting GPU initialization...");

                        
                        let ptx_content = match crate::utils::ptx::load_ptx().await {
                            Ok(content) => {
                                info!("PTX content loaded successfully");
                                content
                            }
                            Err(e) => {
                                error!("Failed to load PTX content: {}", e);
                                error!("PTX load error details: {:?}", e);
                                
                                self_addr.do_send(ResetGPUInitFlag {});
                                return;
                            }
                        };

                        
                        let num_directed_edges = graph_data_clone.edges.len() * 2;
                        info!("Creating UnifiedGPUCompute with {} nodes and {} directed edges (from {} undirected edges)",
                              graph_data_clone.nodes.len(), num_directed_edges, graph_data_clone.edges.len());
                        info!("PTX content size: {} bytes", ptx_content.len());

                        match UnifiedGPUCompute::new(
                            graph_data_clone.nodes.len(),
                            num_directed_edges,
                            &ptx_content,
                        ) {
                            Ok(_context) => {
                                info!("✅ Successfully initialized advanced GPU context with {} nodes and {} edges",
                                      graph_data_clone.nodes.len(), num_directed_edges);
                                info!("GPU physics simulation is now active for knowledge graph");
                                
                            }
                            Err(e) => {
                                error!("❌ Failed to initialize advanced GPU context: {}", e);
                                error!(
                                    "GPU Details: {} nodes, {} directed edges, PTX size: {} bytes",
                                    graph_data_clone.nodes.len(),
                                    num_directed_edges,
                                    ptx_content.len()
                                );
                                error!("Full error: {:?}", e);

                                
                                let error_str = e.to_string();
                                if error_str.contains("PTX") {
                                    error!("PTX compilation or loading issue detected");
                                } else if error_str.contains("memory") {
                                    error!("GPU memory allocation issue - may need to reduce graph size");
                                } else if error_str.contains("device") {
                                    error!("CUDA device issue - check GPU availability");
                                }

                                
                                self_addr.do_send(ResetGPUInitFlag {});
                            }
                        }
                    }
                    .into_actor(actor)
                );
            });
        }

        
        if self.gpu_compute_addr.is_some() && self.gpu_initialized {
            self.run_advanced_gpu_step(ctx);
        } else if self.gpu_compute_addr.is_none() {
            warn!("No GPU compute context available for physics simulation");
        }
        
    }

    fn run_advanced_gpu_step(&mut self, ctx: &mut Context<Self>) {
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        if !self.simulation_params.enabled {
            if crate::utils::logging::is_debug_enabled() {
                info!("[GPU STEP] Physics disabled - skipping simulation");
            }
            return;
        }

        
        if self.simulation_params.is_physics_paused {
            if crate::utils::logging::is_debug_enabled() {
                trace!("[GPU STEP] Physics paused (equilibrium reached) - skipping simulation");
            }
            return;
        }

        

        
        if let Some(ref _gpu_addr) = self.gpu_compute_addr {
            
            let gpu_addr_clone = _gpu_addr.clone();
            let ctx_addr = Context::address(ctx).recipient();

            ctx.spawn(
                async move {
                    match gpu_addr_clone
                        .send(crate::actors::messages::ComputeForces)
                        .await
                    {
                        Ok(Ok(())) => {
                            
                            match gpu_addr_clone
                                .send(crate::actors::messages::GetNodeData)
                                .await
                            {
                                Ok(Ok(node_data)) => {
                                    
                                    let update_msg = crate::actors::messages::UpdateNodePositions {
                                        positions: node_data
                                            .iter()
                                            .enumerate()
                                            .map(|(i, data)| (i as u32, data.clone()))
                                            .collect(),
                                    };
                                    let _ = ctx_addr.do_send(update_msg);
                                }
                                Ok(Err(e)) => error!("Failed to get node data from GPU: {}", e),
                                Err(e) => error!("Failed to send GetNodeData message: {}", e),
                            }
                        }
                        Ok(Err(e)) => error!("GPU force computation failed: {}", e),
                        Err(e) => error!("Failed to send ComputeForces message: {}", e),
                    }
                }
                .into_actor(self),
            );

            
            return;
        }

        
        trace!("No GPU compute actor available for physics simulation");
    }
    
    

    
    pub fn update_advanced_physics_params(&mut self, params: AdvancedParams) -> Result<(), String> {
        self.advanced_params = params.clone();
        self.stress_solver = StressMajorizationSolver::from_advanced_params(&params);

        
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            let update_msg = crate::actors::messages::UpdateSimulationParams {
                params: self.simulation_params.clone(),
            };
            gpu_addr.do_send(update_msg);
        }

        info!("Updated advanced physics parameters via public API");
        Ok(())
    }

    
    pub fn get_constraint_set(&self) -> &ConstraintSet {
        &self.constraint_set
    }

    
    pub fn trigger_stress_optimization(&mut self) -> Result<(), String> {
        self.execute_stress_majorization_step();
        info!("Manually triggered stress majorization via public API");
        Ok(())
    }

    
    pub fn has_advanced_gpu(&self) -> bool {
        self.gpu_compute_addr.is_some()
    }

    
    pub fn get_semantic_analysis_status(&self) -> (usize, Option<std::time::Duration>) {
        let feature_count = self.semantic_features_cache.len();
        let age = self.last_semantic_analysis.map(|t| t.elapsed());
        (feature_count, age)
    }

    
    
    
    
    fn calculate_communication_intensity(
        &self,
        source_type: &crate::types::claude_flow::AgentType,
        target_type: &crate::types::claude_flow::AgentType,
        source_active_tasks: u32,
        target_active_tasks: u32,
        source_success_rate: f32,
        target_success_rate: f32,
    ) -> f32 {
        
        let base_intensity = match (source_type, target_type) {
            
            (crate::types::claude_flow::AgentType::Coordinator, _)
            | (_, crate::types::claude_flow::AgentType::Coordinator) => 0.9,

            
            (
                crate::types::claude_flow::AgentType::Coder,
                crate::types::claude_flow::AgentType::Tester,
            )
            | (
                crate::types::claude_flow::AgentType::Tester,
                crate::types::claude_flow::AgentType::Coder,
            ) => 0.8,

            (
                crate::types::claude_flow::AgentType::Researcher,
                crate::types::claude_flow::AgentType::Analyst,
            )
            | (
                crate::types::claude_flow::AgentType::Analyst,
                crate::types::claude_flow::AgentType::Researcher,
            ) => 0.7,

            (
                crate::types::claude_flow::AgentType::Architect,
                crate::types::claude_flow::AgentType::Coder,
            )
            | (
                crate::types::claude_flow::AgentType::Coder,
                crate::types::claude_flow::AgentType::Architect,
            ) => 0.7,

            
            (
                crate::types::claude_flow::AgentType::Architect,
                crate::types::claude_flow::AgentType::Analyst,
            )
            | (
                crate::types::claude_flow::AgentType::Analyst,
                crate::types::claude_flow::AgentType::Architect,
            ) => 0.6,

            (
                crate::types::claude_flow::AgentType::Reviewer,
                crate::types::claude_flow::AgentType::Coder,
            )
            | (
                crate::types::claude_flow::AgentType::Coder,
                crate::types::claude_flow::AgentType::Reviewer,
            ) => 0.6,

            (
                crate::types::claude_flow::AgentType::Optimizer,
                crate::types::claude_flow::AgentType::Analyst,
            )
            | (
                crate::types::claude_flow::AgentType::Analyst,
                crate::types::claude_flow::AgentType::Optimizer,
            ) => 0.6,

            
            _ => 0.4,
        };

        
        let max_tasks = std::cmp::max(source_active_tasks, target_active_tasks);
        let activity_factor = if max_tasks > 0 {
            1.0 + (max_tasks as f32 * 0.1).min(0.5) 
        } else {
            0.7 
        };

        
        let avg_success_rate = (source_success_rate + target_success_rate) / 200.0; 
        let performance_factor = 0.5 + avg_success_rate * 0.5; 

        
        let final_intensity = base_intensity * activity_factor * performance_factor;

        
        final_intensity.min(1.0).max(0.0)
    }

    
    fn check_and_handle_equilibrium(&mut self, total_kinetic_energy: f32, node_count: usize) {
        if !self.simulation_params.auto_pause_config.enabled || node_count == 0 {
            return;
        }

        
        
        let avg_kinetic_energy = total_kinetic_energy / node_count as f32;
        let avg_velocity = (2.0 * avg_kinetic_energy).sqrt();

        let config = &self.simulation_params.auto_pause_config;

        
        let is_in_equilibrium = avg_velocity < config.equilibrium_velocity_threshold
            && avg_kinetic_energy < config.equilibrium_energy_threshold;

        if is_in_equilibrium {
            
            self.simulation_params.equilibrium_stability_counter += 1;

            
            if self.simulation_params.equilibrium_stability_counter
                >= config.equilibrium_check_frames
            {
                if !self.simulation_params.is_physics_paused && config.pause_on_equilibrium {
                    
                    self.simulation_params.is_physics_paused = true;

                    if crate::utils::logging::is_debug_enabled() {
                        info!("[AUTO-PAUSE] Physics paused - equilibrium reached (avg_velocity: {:.4}, avg_energy: {:.4})",
                              avg_velocity, avg_kinetic_energy);
                    }

                    
                    let pause_msg = PhysicsPauseMessage {
                        pause: true,
                        reason: format!(
                            "Equilibrium reached (vel: {:.4}, energy: {:.4})",
                            avg_velocity, avg_kinetic_energy
                        ),
                    };

                    
                    self.client_manager.do_send(BroadcastMessage {
                        message: format!(
                            "{{\"type\": \"physics_paused\", \"reason\": \"{}\"}}",
                            pause_msg.reason
                        ),
                    });
                }
            }
        } else {
            
            
            if !self.simulation_params.is_physics_paused {
                
                self.simulation_params.equilibrium_stability_counter = 0;
            }
            
            
        }

        if crate::utils::logging::is_debug_enabled() {
            trace!("[AUTO-PAUSE] Equilibrium check: velocity={:.4}, energy={:.4}, stable_frames={}/{}, paused={}",
                   avg_velocity, avg_kinetic_energy,
                   self.simulation_params.equilibrium_stability_counter,
                   config.equilibrium_check_frames,
                   self.simulation_params.is_physics_paused);
        }
    }

    
    fn resume_physics_if_paused(&mut self, reason: String) {
        if self.simulation_params.is_physics_paused {
            self.simulation_params.is_physics_paused = false;
            self.simulation_params.equilibrium_stability_counter = 0;

            if crate::utils::logging::is_debug_enabled() {
                info!("[AUTO-PAUSE] Physics resumed: {}", reason);
            }

            
            let _resume_msg = PhysicsPauseMessage {
                pause: false,
                reason: reason.clone(),
            };

            self.client_manager.do_send(BroadcastMessage {
                message: format!(
                    "{{\"type\": \"physics_resumed\", \"reason\": \"{}\"}}",
                    reason
                ),
            });
        }
    }

    
    pub fn add_nodes_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        debug!("Adding {} new nodes incrementally", metadata.len());

        let graph_data_mut = Arc::make_mut(&mut self.graph_data);

        for (filename_with_ext, file_meta_data) in &metadata {
            let metadata_id_val = filename_with_ext.trim_end_matches(".md").to_string();

            
            if self
                .node_map
                .values()
                .any(|n| n.metadata_id == metadata_id_val)
            {
                debug!("Node {} already exists, skipping", metadata_id_val);
                continue;
            }

            let node_id_val = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            let mut node = Node::new_with_id(metadata_id_val.clone(), Some(node_id_val));
            node.label = file_meta_data.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(file_meta_data.file_size as u64);
            
            

            
            node.metadata
                .insert("fileName".to_string(), file_meta_data.file_name.clone());
            node.metadata
                .insert("fileSize".to_string(), file_meta_data.file_size.to_string());
            node.metadata
                .insert("nodeSize".to_string(), file_meta_data.node_size.to_string());
            node.metadata.insert(
                "hyperlinkCount".to_string(),
                file_meta_data.hyperlink_count.to_string(),
            );
            node.metadata
                .insert("sha1".to_string(), file_meta_data.sha1.clone());
            node.metadata.insert(
                "lastModified".to_string(),
                file_meta_data.last_modified.to_rfc3339(),
            );
            node.metadata
                .insert("metadataId".to_string(), metadata_id_val.clone());

            
            let features = self.semantic_analyzer.analyze_metadata(file_meta_data);
            self.semantic_features_cache
                .insert(metadata_id_val, features);

            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
            graph_data_mut.nodes.push(node);
        }

        info!("Added {} new nodes incrementally", metadata.len());
        Ok(())
    }

    
    pub fn update_node_from_metadata(
        &mut self,
        metadata_id: String,
        metadata: FileMetadata,
    ) -> Result<(), String> {
        debug!("Updating node {} incrementally", metadata_id);

        
        let mut node_found = false;
        if let Some(node) = Arc::make_mut(&mut self.node_map)
            .values_mut()
            .find(|n| n.metadata_id == metadata_id)
        {
            node.label = metadata.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(metadata.file_size as u64);

            
            node.metadata
                .insert("fileName".to_string(), metadata.file_name.clone());
            node.metadata
                .insert("fileSize".to_string(), metadata.file_size.to_string());
            node.metadata
                .insert("nodeSize".to_string(), metadata.node_size.to_string());
            node.metadata.insert(
                "hyperlinkCount".to_string(),
                metadata.hyperlink_count.to_string(),
            );
            node.metadata
                .insert("sha1".to_string(), metadata.sha1.clone());
            node.metadata.insert(
                "lastModified".to_string(),
                metadata.last_modified.to_rfc3339(),
            );

            node_found = true;
        }

        if !node_found {
            return Err(format!("Node {} not found for update", metadata_id));
        }

        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        if let Some(node) = graph_data_mut
            .nodes
            .iter_mut()
            .find(|n| n.metadata_id == metadata_id)
        {
            node.label = metadata.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(metadata.file_size as u64);
        }

        
        let features = self.semantic_analyzer.analyze_metadata(&metadata);
        self.semantic_features_cache
            .insert(metadata_id.clone(), features);

        info!("Updated node {} incrementally", metadata_id);
        Ok(())
    }

    
    pub fn remove_node_by_metadata(&mut self, metadata_id: String) -> Result<(), String> {
        debug!("Removing node {} incrementally", metadata_id);

        
        let node_id = self
            .node_map
            .values()
            .find(|n| n.metadata_id == metadata_id)
            .map(|n| n.id);

        if let Some(node_id) = node_id {
            
            Arc::make_mut(&mut self.node_map).remove(&node_id);

            
            let graph_data_mut = Arc::make_mut(&mut self.graph_data);
            graph_data_mut.nodes.retain(|n| n.id != node_id);
            graph_data_mut
                .edges
                .retain(|e| e.source != node_id && e.target != node_id);

            
            self.semantic_features_cache.remove(&metadata_id);

            info!("Removed node {} incrementally", metadata_id);
            Ok(())
        } else {
            Err(format!("Node {} not found for removal", metadata_id))
        }
    }
}

impl Actor for GraphServiceActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("GraphServiceActor started");
        
        if std::env::var("GPU_SMOKE_ON_START").ok().as_deref() == Some("1") {
            let report = crate::utils::gpu_diagnostics::ptx_module_smoke_test();
            info!("{}", report);
        }
        
        ctx.address()
            .do_send(crate::actors::messages::InitializeActor);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        self.simulation_running.store(false, Ordering::SeqCst);
        self.shutdown_complete.store(true, Ordering::SeqCst);
        info!("GraphServiceActor stopped");
    }
}

// Message handlers
impl Handler<crate::actors::messages::InitializeActor> for GraphServiceActor {
    type Result = ();

    fn handle(
        &mut self,
        _msg: crate::actors::messages::InitializeActor,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("GraphServiceActor: Initializing simulation loop (deferred from started)");
        self.start_simulation_loop(ctx);
    }
}

impl Handler<GetGraphData> for GraphServiceActor {
    type Result = Result<std::sync::Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("DEBUG_VERIFICATION: GraphServiceActor handling GetGraphData with Arc reference (NO CLONE!).");
        Ok(std::sync::Arc::clone(&self.graph_data)) 
    }
}

impl Handler<UpdateNodePositions> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_positions(msg.positions);
        Ok(())
    }
}

// WEBSOCKET SETTLING FIX: Handler for forced position broadcasts
impl Handler<crate::actors::messages::ForcePositionBroadcast> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        msg: crate::actors::messages::ForcePositionBroadcast,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        info!("Force broadcasting positions: {}", msg.reason);

        
        let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
        let mut knowledge_ids: Vec<u32> = Vec::new();
        let mut agent_ids: Vec<u32> = Vec::new();

        
        for (node_id, node) in self.node_map.iter() {
            position_data.push((
                *node_id,
                BinaryNodeDataClient::new(*node_id, node.data.position(), node.data.velocity()),
            ));
            knowledge_ids.push(*node_id);
        }

        
        for node in &self.bots_graph_data.nodes {
            position_data.push((
                node.id,
                BinaryNodeDataClient::new(
                    node.id,
                    glam_to_vec3data(Vec3::new(node.data.x, node.data.y, node.data.z)),
                    glam_to_vec3data(Vec3::new(node.data.vx, node.data.vy, node.data.vz)),
                ),
            ));
            agent_ids.push(node.id);
        }

        
        if !position_data.is_empty() {
            
            let binary_data = crate::utils::binary_protocol::encode_node_data_with_types(
                &position_data,
                &agent_ids,
                &knowledge_ids,
            );

            
            self.client_manager
                .do_send(crate::actors::messages::BroadcastNodePositions {
                    positions: binary_data,
                });

            
            self.last_broadcast_time = Some(std::time::Instant::now());
            self.initial_positions_sent = true;

            info!(
                "Force broadcast complete: {} nodes ({} knowledge + {} agents) sent (reason: {})",
                position_data.len(),
                knowledge_ids.len(),
                agent_ids.len(),
                msg.reason
            );
        } else {
            warn!(
                "Force broadcast requested but no position data available (reason: {})",
                msg.reason
            );
        }

        Ok(())
    }
}

impl Handler<InitialClientSync> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitialClientSync, _ctx: &mut Self::Context) -> Self::Result {
        info!(
            "Initial client sync requested by {} from {}",
            msg.client_identifier, msg.trigger_source
        );

        
        let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
        let mut knowledge_ids: Vec<u32> = Vec::new();
        let mut agent_ids: Vec<u32> = Vec::new();

        
        for (node_id, node) in self.node_map.iter() {
            position_data.push((
                *node_id,
                BinaryNodeDataClient::new(*node_id, node.data.position(), node.data.velocity()),
            ));
            knowledge_ids.push(*node_id);
        }

        
        for node in &self.bots_graph_data.nodes {
            position_data.push((
                node.id,
                BinaryNodeDataClient::new(
                    node.id,
                    glam_to_vec3data(Vec3::new(node.data.x, node.data.y, node.data.z)),
                    glam_to_vec3data(Vec3::new(node.data.vx, node.data.vy, node.data.vz)),
                ),
            ));
            agent_ids.push(node.id);
        }

        if !position_data.is_empty() {
            
            let binary_data = crate::utils::binary_protocol::encode_node_data_with_types(
                &position_data,
                &agent_ids,
                &knowledge_ids,
            );

            
            self.client_manager
                .do_send(crate::actors::messages::BroadcastNodePositions {
                    positions: binary_data,
                });

            
            self.last_broadcast_time = Some(std::time::Instant::now());
            self.initial_positions_sent = true;

            info!("Initial sync broadcast complete: {} nodes ({} knowledge + {} agents) sent for client {}",
                  position_data.len(), knowledge_ids.len(), agent_ids.len(), msg.client_identifier);
        } else {
            warn!(
                "Initial sync requested but no nodes available for client {}",
                msg.client_identifier
            );
        }

        Ok(())
    }
}

impl Handler<AddNode> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNode, _ctx: &mut Self::Context) -> Self::Result {
        self.add_node(msg.node);
        Ok(())
    }
}

impl Handler<RemoveNode> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNode, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_node(msg.node_id);
        Ok(())
    }
}

impl Handler<AddEdge> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.add_edge(msg.edge);
        Ok(())
    }
}

impl Handler<RemoveEdge> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_edge(&msg.edge_id);
        Ok(())
    }
}

impl Handler<GetNodeMap> for GraphServiceActor {
    type Result = Result<std::sync::Arc<HashMap<u32, Node>>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Self::Context) -> Self::Result {
        Ok(Arc::clone(&self.node_map)) 
    }
}

///
impl Handler<GetPhysicsState> for GraphServiceActor {
    type Result = Result<PhysicsState, String>;

    fn handle(&mut self, _msg: GetPhysicsState, _ctx: &mut Self::Context) -> Self::Result {
        
        let avg_ke = if !self.kinetic_energy_history.is_empty() {
            self.kinetic_energy_history.iter().sum::<f32>()
                / self.kinetic_energy_history.len() as f32
        } else {
            0.0
        };

        let state_name = match self.current_state {
            AutoBalanceState::Stable => "stable",
            AutoBalanceState::Spreading => "spreading",
            AutoBalanceState::Clustering => "clustering",
            AutoBalanceState::Bouncing => "bouncing",
            AutoBalanceState::Oscillating => "oscillating",
            AutoBalanceState::Adjusting => "adjusting",
        };

        Ok(PhysicsState {
            is_settled: self.current_state == AutoBalanceState::Stable && self.stable_count > 30,
            stable_frame_count: self.stable_count,
            kinetic_energy: avg_ke,
            current_state: state_name.to_string(),
        })
    }
}

impl Handler<BuildGraphFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, ctx: &mut Self::Context) -> Self::Result {
        info!(
            "BuildGraphFromMetadata handler called with {} metadata entries",
            msg.metadata.len()
        );
        
        let result = self.build_from_metadata(msg.metadata, ctx);

        
        if result.is_ok() {
            
            if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
                info!("Graph data prepared for GPU physics");

                
                gpu_compute_addr.do_send(InitializeGPU {
                    graph: Arc::clone(&self.graph_data),
                    graph_service_addr: Some(ctx.address()),
                    physics_orchestrator_addr: None,
                    gpu_manager_addr: None,
                });
                info!("Sent GPU initialization request to GPU compute actor");

                
                gpu_compute_addr.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(&self.graph_data),
                });
                info!("Sent initial graph data to GPU compute actor");
            } else {
                info!("No GPU compute address available - skipping GPU initialization");
            }
        }

        result
    }
}

impl Handler<AddNodesFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNodesFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.add_nodes_from_metadata(msg.metadata)
    }
}

impl Handler<UpdateNodeFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodeFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_from_metadata(msg.metadata_id, msg.metadata)
    }
}

impl Handler<RemoveNodeByMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNodeByMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_node_by_metadata(msg.metadata_id)
    }
}

impl Handler<StartSimulation> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StartSimulation, ctx: &mut Self::Context) -> Self::Result {
        self.start_simulation_loop(ctx);
        Ok(())
    }
}

impl Handler<StopSimulation> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: StopSimulation, _ctx: &mut Self::Context) -> Self::Result {
        self.simulation_running.store(false, Ordering::SeqCst);
        Ok(())
    }
}

impl Handler<UpdateNodePosition> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePosition, _ctx: &mut Self::Context) -> Self::Result {
        
        if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&msg.node_id) {
            
            let new_position = glam_to_vec3data(msg.position);
            let new_velocity = glam_to_vec3data(msg.velocity);

            node.data.x = new_position.x;
            node.data.y = new_position.y;
            node.data.z = new_position.z;
            node.data.vx = new_velocity.x;
            node.data.vy = new_velocity.y;
            node.data.vz = new_velocity.z;

            
        } else {
            debug!("Received update for unknown node ID: {}", msg.node_id);
            return Err(format!("Unknown node ID: {}", msg.node_id));
        }

        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        for node_in_graph_data in &mut graph_data_mut.nodes {
            
            if node_in_graph_data.id == msg.node_id {
                
                let pos = glam_to_vec3data(msg.position);
                node_in_graph_data.data.x = pos.x;
                node_in_graph_data.data.y = pos.y;
                node_in_graph_data.data.z = pos.z;

                
                let vel = glam_to_vec3data(msg.velocity);
                node_in_graph_data.data.vx = vel.x;
                node_in_graph_data.data.vy = vel.y;
                node_in_graph_data.data.vz = vel.z;
                break;
            }
        }

        Ok(())
    }
}

impl Handler<SimulationStep> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: SimulationStep, ctx: &mut Self::Context) -> Self::Result {
        
        self.run_simulation_step(ctx);
        Ok(())
    }
}

impl Handler<GetAutoBalanceNotifications> for GraphServiceActor {
    type Result = Result<Vec<AutoBalanceNotification>, String>;

    fn handle(
        &mut self,
        msg: GetAutoBalanceNotifications,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        match self.auto_balance_notifications.lock() {
            Ok(notifications) => {
                let filtered_notifications = if let Some(since) = msg.since_timestamp {
                    notifications
                        .iter()
                        .filter(|n| n.timestamp > since)
                        .cloned()
                        .collect()
                } else {
                    notifications.clone()
                };

                Ok(filtered_notifications)
            }
            _ => Err("Failed to access notifications".to_string()),
        }
    }
}

impl Handler<UpdateGraphData> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, ctx: &mut Self::Context) -> Self::Result {
        info!(
            "Updating graph data with {} nodes, {} edges",
            msg.graph_data.nodes.len(),
            msg.graph_data.edges.len()
        );

        
        self.graph_data = msg.graph_data;

        
        Arc::make_mut(&mut self.node_map).clear();
        for node in &self.graph_data.nodes {
            
            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
        }

        
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone, ctx);

        
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            info!("Sending loaded graph data to GPU physics");

            
            gpu_compute_addr.do_send(InitializeGPU {
                graph: Arc::clone(&self.graph_data),
                graph_service_addr: Some(ctx.address()),
                physics_orchestrator_addr: None,
                gpu_manager_addr: None,
            });
            info!("Sent GPU initialization request to GPU compute actor");

            
            gpu_compute_addr.do_send(UpdateGPUGraphData {
                graph: Arc::clone(&self.graph_data),
            });
            info!("Sent loaded graph data to GPU compute actor");
        } else {
            warn!("GPU compute actor not available, physics simulation won't be initialized");
        }

        info!("Graph data updated successfully with constraint generation and GPU initialization");
        Ok(())
    }
}

impl Handler<ReloadGraphFromDatabase> for GraphServiceActor {
    type Result = ResponseActFuture<Self, Result<(), String>>;

    fn handle(&mut self, _msg: ReloadGraphFromDatabase, _ctx: &mut Self::Context) -> Self::Result {
        let kg_repo = self.kg_repo.clone();

        Box::pin(
            async move {
                
                match kg_repo.load_graph().await {
                    Ok(graph_data) => {
                        info!(
                            "ReloadGraphFromDatabase: Loaded {} nodes from database",
                            graph_data.nodes.len()
                        );
                        Ok(graph_data)
                    }
                    Err(e) => {
                        error!("ReloadGraphFromDatabase: Failed to load graph: {}", e);
                        Err(format!("Failed to load graph from database: {}", e))
                    }
                }
            }
            .into_actor(self)
            .map(|result, actor, ctx| {
                match result {
                    Ok(graph_data) => {
                        
                        actor.graph_data = graph_data;

                        
                        Arc::make_mut(&mut actor.node_map).clear();
                        for node in &actor.graph_data.nodes {
                            Arc::make_mut(&mut actor.node_map).insert(node.id, node.clone());
                        }

                        
                        if let Some(ref gpu_addr) = actor.gpu_compute_addr {
                            gpu_addr.do_send(UpdateGPUGraphData {
                                graph: Arc::clone(&actor.graph_data),
                            });
                            info!("ReloadGraphFromDatabase: Updated GPU with fresh graph data");
                        }

                        info!("ReloadGraphFromDatabase: Successfully reloaded graph from database");
                        Ok(())
                    }
                    Err(e) => Err(e),
                }
            }),
        )
    }
}

impl Handler<UpdateBotsGraph> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) -> Self::Result {
        
        if msg.agents.is_empty() {
            debug!("No agents to update - skipping bots graph broadcast");
            return;
        }

        
        let mut nodes = vec![];
        let mut edges = vec![];

        
        let bot_id_offset = 10000;

        
        let mut existing_positions: HashMap<String, (Vec3Data, Vec3Data)> = HashMap::new();
        for node in &self.bots_graph_data.nodes {
            existing_positions.insert(
                node.metadata_id.clone(),
                (node.data.position(), node.data.velocity()),
            );
        }

        
        for (i, agent) in msg.agents.iter().enumerate() {
            let node_id = bot_id_offset + i as u32;

            
            let mut node = Node::new_with_id(agent.id.clone(), Some(node_id));

            if let Some((saved_position, saved_velocity)) = existing_positions.get(&agent.id) {
                
                node.data.x = saved_position.x;
                node.data.y = saved_position.y;
                node.data.z = saved_position.z;
                node.data.vx = saved_velocity.x;
                node.data.vy = saved_velocity.y;
                node.data.vz = saved_velocity.z;
            } else {
                
                let physics = crate::config::dev_config::physics();
                use rand::rngs::{OsRng, StdRng};
                use rand::{Rng, SeedableRng};
                let mut rng = StdRng::from_seed(OsRng.gen());

                let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
                let phi = rng.gen::<f32>() * std::f32::consts::PI;
                let radius =
                    physics.initial_radius_min + rng.gen::<f32>() * physics.initial_radius_range;

                node.data.x = radius * phi.sin() * theta.cos();
                node.data.y = radius * phi.sin() * theta.sin();
                node.data.z = radius * phi.cos();

                node.data.vx = rng.gen_range(-0.5..0.5);
                node.data.vy = rng.gen_range(-0.5..0.5);
                node.data.vz = rng.gen_range(-0.5..0.5);
            }

            
            node.color = Some(match agent.agent_type.as_str() {
                "coordinator" => "#FF6B6B".to_string(),
                "researcher" => "#4ECDC4".to_string(),
                "coder" => "#45B7D1".to_string(),
                "analyst" => "#FFA07A".to_string(),
                "architect" => "#98D8C8".to_string(),
                "tester" => "#F7DC6F".to_string(),
                _ => "#95A5A6".to_string(),
            });

            node.label = agent.name.clone();
            node.size = Some(20.0 + (agent.workload * 25.0)); 

            
            node.metadata
                .insert("agent_type".to_string(), agent.agent_type.clone());
            node.metadata
                .insert("status".to_string(), agent.status.clone());
            node.metadata
                .insert("cpu_usage".to_string(), agent.cpu_usage.to_string());
            node.metadata
                .insert("memory_usage".to_string(), agent.memory_usage.to_string());
            node.metadata
                .insert("health".to_string(), agent.health.to_string());
            node.metadata
                .insert("is_agent".to_string(), "true".to_string()); 

            nodes.push(node);
        }

        
        for (i, source_agent) in msg.agents.iter().enumerate() {
            for (j, target_agent) in msg.agents.iter().enumerate() {
                if i != j {
                    let source_node_id = bot_id_offset + i as u32;
                    let target_node_id = bot_id_offset + j as u32;

                    
                    let communication_intensity = if source_agent.agent_type == "coordinator"
                        || target_agent.agent_type == "coordinator"
                    {
                        0.8 
                    } else if source_agent.status == "active" && target_agent.status == "active" {
                        0.5 
                    } else {
                        0.2 
                    };

                    
                    if communication_intensity > 0.1 {
                        let mut edge =
                            Edge::new(source_node_id, target_node_id, communication_intensity);
                        
                        let metadata = edge.metadata.get_or_insert_with(HashMap::new);
                        metadata.insert(
                            "communication_type".to_string(),
                            "agent_collaboration".to_string(),
                        );
                        metadata
                            .insert("intensity".to_string(), communication_intensity.to_string());
                        edges.push(edge);
                    }
                }
            }
        }

        
        let bots_graph_data_mut = Arc::make_mut(&mut self.bots_graph_data);
        bots_graph_data_mut.nodes = nodes;
        bots_graph_data_mut.edges = edges;

        info!("Updated bots graph with {} agents and {} edges - data will be broadcast in next physics cycle",
             msg.agents.len(), self.bots_graph_data.edges.len());

        
        
        
        
        
        
        
        
        
        
        
        
        

        debug!("Agent graph data updated ({} nodes). Physics loop will broadcast with AGENT_NODE_FLAG.",
               self.bots_graph_data.nodes.len());

        
        
        if self.bots_graph_data.nodes.len() > 0 {
            if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
                gpu_compute_addr.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(&self.bots_graph_data),
                });
                info!("Sent updated bots graph data ({} nodes) to GPU compute actor for physics simulation",
                      self.bots_graph_data.nodes.len());
            } else {
                warn!("No GPU compute address available - bots will use initial positions only");
            }
        }
    }
}

impl Handler<GetBotsGraphData> for GraphServiceActor {
    type Result = Result<std::sync::Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetBotsGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(Arc::clone(&self.bots_graph_data)) 
    }
}

// Handler for storing GPU compute actor address
impl Handler<StoreGPUComputeAddress> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, msg: StoreGPUComputeAddress, _ctx: &mut Self::Context) -> Self::Result {
        info!("Storing GPU compute actor address in GraphServiceActor");
        self.gpu_compute_addr = msg.addr;
        if self.gpu_compute_addr.is_some() {
            info!("GPU compute actor address stored - waiting for GPU initialization");
        } else {
            warn!("GPU compute actor address is None - physics will not be available");
        }
    }
}

// Handler for initializing GPU connection after system startup
impl Handler<InitializeGPUConnection> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, msg: InitializeGPUConnection, ctx: &mut Self::Context) -> Self::Result {
        info!("Initializing GPU connection after system startup");

        if let Some(gpu_manager) = msg.gpu_manager {
            let _gpu_manager_clone = gpu_manager.clone();
            let _self_addr = ctx.address();

            
            self.gpu_compute_addr = Some(gpu_manager.clone());
            info!("[GraphServiceActor] Stored GPUManagerActor address for GPU coordination");


            if !self.graph_data.nodes.is_empty() {
                info!("Sending initial graph data to GPU via GPUManager");
                gpu_manager.do_send(InitializeGPU {
                    graph: Arc::clone(&self.graph_data),
                    graph_service_addr: Some(ctx.address()),
                    physics_orchestrator_addr: None,
                    gpu_manager_addr: None,
                });


                gpu_manager.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(&self.graph_data),
                });

                self.gpu_init_in_progress = true;
                info!("GPU initialization in progress - waiting for GPUInitialized confirmation message");
            }
        } else {
            warn!("No GPU manager provided for initialization");
        }
    }
}

// Handler for GPU initialization notification
impl Handler<GPUInitialized> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, _msg: GPUInitialized, _ctx: &mut Self::Context) -> Self::Result {
        info!("✅ GPU initialization CONFIRMED - GPUInitialized message received");
        self.gpu_initialized = true;
        self.gpu_init_in_progress = false;

        
        info!("Physics simulation is now ready:");
        info!("  - GPU initialized: {}", self.gpu_initialized);
        info!("  - Physics enabled: {}", self.simulation_params.enabled);
        info!("  - Node count: {}", self.graph_data.nodes.len());
        info!("  - Edge count: {}", self.graph_data.edges.len());
    }
}

impl Handler<UpdateSimulationParams> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateSimulationParams, _ctx: &mut Self::Context) -> Self::Result {
        if crate::utils::logging::is_debug_enabled() {
            info!("[GRAPH ACTOR] === UpdateSimulationParams RECEIVED ===");
            info!("[GRAPH ACTOR] OLD physics values:");
            info!("  - repel_k: {} (was)", self.simulation_params.repel_k);
            info!("  - damping: {:.3} (was)", self.simulation_params.damping);
            info!("  - dt: {:.3} (was)", self.simulation_params.dt);
            info!("  - spring_k: {:.3} (was)", self.simulation_params.spring_k);
            info!("  - spring_k: {:.3} (was)", self.simulation_params.spring_k);
            info!(
                "  - max_velocity: {:.3} (was)",
                self.simulation_params.max_velocity
            );
            info!("  - enabled: {} (was)", self.simulation_params.enabled);
            info!(
                "  - auto_balance: {} (was)",
                self.simulation_params.auto_balance
            );

            info!("[GRAPH ACTOR] NEW physics values:");
            info!("  - repel_k: {} (new)", msg.params.repel_k);
            info!("  - damping: {:.3} (new)", msg.params.damping);
            info!("  - dt: {:.3} (new)", msg.params.dt);
            info!("  - spring_k: {:.3} (new)", msg.params.spring_k);
            info!("  - spring_k: {:.3} (new)", msg.params.spring_k);
            info!("  - max_velocity: {:.3} (new)", msg.params.max_velocity);
            info!("  - enabled: {} (new)", msg.params.enabled);
            info!("  - auto_balance: {} (new)", msg.params.auto_balance);
        }

        
        let auto_balance_just_enabled =
            !self.simulation_params.auto_balance && msg.params.auto_balance;

        self.simulation_params = msg.params.clone();
        
        self.target_params = msg.params.clone();

        
        if auto_balance_just_enabled {
            info!("[AUTO-BALANCE] Auto-balance enabled - starting adaptive tuning from current values");

            
            self.auto_balance_history.clear();
            self.stable_count = 0;

            info!("[AUTO-BALANCE] Will adaptively tune from current settings - repel_k: {:.3}, damping: {:.3}",
                  self.simulation_params.repel_k, self.simulation_params.damping);
        }

        
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            if crate::utils::logging::is_debug_enabled() {
                info!("Forwarding params to ForceComputeActor");
            }
            gpu_addr.do_send(msg);
        }

        if crate::utils::logging::is_debug_enabled() {
            info!("=== GraphServiceActor physics params update COMPLETE ===");
        }
        Ok(())
    }
}

impl Handler<RequestPositionSnapshot> for GraphServiceActor {
    type Result = Result<PositionSnapshot, String>;

    fn handle(&mut self, msg: RequestPositionSnapshot, _ctx: &mut Self::Context) -> Self::Result {
        let mut knowledge_nodes = Vec::new();
        let mut agent_nodes = Vec::new();

        
        if msg.include_knowledge_graph {
            for node in &self.graph_data.nodes {
                
                if node.metadata.get("is_agent").map_or(false, |v| v == "true") {
                    continue;
                }

                let node_data =
                    BinaryNodeDataClient::new(node.id, node.data.position(), node.data.velocity());

                knowledge_nodes.push((node.id, node_data));
            }
        }

        
        if msg.include_agent_graph {
            for node in &self.bots_graph_data.nodes {
                let node_data =
                    BinaryNodeDataClient::new(node.id, node.data.position(), node.data.velocity());

                agent_nodes.push((node.id, node_data));
            }
        }

        Ok(PositionSnapshot {
            knowledge_nodes,
            agent_nodes,
            timestamp: std::time::Instant::now(),
        })
    }
}

// Advanced Physics and Constraint Message Handlers

impl Handler<UpdateAdvancedParams> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateAdvancedParams, _ctx: &mut Self::Context) -> Self::Result {
        self.advanced_params = msg.params.clone();

        
        self.stress_solver =
            crate::physics::stress_majorization::StressMajorizationSolver::from_advanced_params(
                &msg.params,
            );

        
        
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            let update_msg = crate::actors::messages::UpdateAdvancedParams {
                params: msg.params.clone(),
            };
            gpu_addr.do_send(update_msg);
        }

        info!("Updated advanced physics parameters");
        Ok(())
    }
}

impl Handler<UpdateConstraints> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateConstraints, ctx: &mut Self::Context) -> Self::Result {
        self.handle_constraint_update(msg.constraint_data, ctx)
    }
}

impl Handler<GetConstraints> for GraphServiceActor {
    type Result = Result<ConstraintSet, String>;

    fn handle(&mut self, _msg: GetConstraints, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.constraint_set.clone())
    }
}

impl Handler<TriggerStressMajorization> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: TriggerStressMajorization,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        self.execute_stress_majorization_step();
        info!("Manually triggered stress majorization optimization");
        Ok(())
    }
}

impl Handler<RegenerateSemanticConstraints> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(
        &mut self,
        _msg: RegenerateSemanticConstraints,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        
        self.constraint_set
            .set_group_active("semantic_dynamic", false);
        self.constraint_set
            .set_group_active("domain_clustering", false);
        self.constraint_set
            .set_group_active("clustering_dynamic", false);

        
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone, ctx);

        
        if self.last_semantic_analysis.is_some() {
            self.update_dynamic_constraints(ctx);
        }

        info!(
            "Regenerated semantic constraints: {} total constraints",
            self.constraint_set.constraints.len()
        );
        Ok(())
    }
}

impl Handler<SetAdvancedGPUContext> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, _msg: SetAdvancedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        
        
        self.gpu_init_in_progress = false; 
        info!("Advanced GPU context initialization signal received");
    }
}

// StoreAdvancedGPUContext handler removed - GPU context now managed by ForceComputeActor

impl Handler<ResetGPUInitFlag> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, _msg: ResetGPUInitFlag, _ctx: &mut Self::Context) -> Self::Result {
        self.gpu_init_in_progress = false;
        debug!("GPU initialization flag reset");
    }
}

impl Handler<ComputeShortestPaths> for GraphServiceActor {
    type Result = Result<crate::ports::gpu_semantic_analyzer::PathfindingResult, String>;

    fn handle(&mut self, msg: ComputeShortestPaths, _ctx: &mut Self::Context) -> Self::Result {
        use crate::ports::gpu_semantic_analyzer::PathfindingResult;
        use std::time::Instant;

        
        let source_id = msg.source_node_id;
        let start_time = Instant::now();

        
        if !self.node_map.contains_key(&source_id) {
            return Err(format!("Source node {} not found", source_id));
        }

        
        let mut distances: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        let mut predecessors: std::collections::HashMap<u32, u32> =
            std::collections::HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let mut priority_queue = std::collections::BinaryHeap::new();

        
        distances.insert(source_id, 0.0);
        priority_queue.push(std::cmp::Reverse((
            ordered_float::OrderedFloat(0.0_f32),
            source_id,
        )));

        
        let mut adjacency: std::collections::HashMap<u32, Vec<(u32, f32)>> =
            std::collections::HashMap::new();
        for edge in &self.graph_data.edges {
            
            adjacency
                .entry(edge.source)
                .or_insert_with(Vec::new)
                .push((edge.target, edge.weight));
            adjacency
                .entry(edge.target)
                .or_insert_with(Vec::new)
                .push((edge.source, edge.weight));
        }

        
        while let Some(std::cmp::Reverse((current_dist, current_node))) = priority_queue.pop() {
            let current_dist = current_dist.into_inner();
            if visited.contains(&current_node) {
                continue;
            }

            visited.insert(current_node);

            
            if let Some(neighbors) = adjacency.get(&current_node) {
                for &(neighbor_id, edge_weight) in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    let new_dist = current_dist + edge_weight;
                    let current_neighbor_dist = distances.get(&neighbor_id).copied();

                    if current_neighbor_dist.is_none() || new_dist < current_neighbor_dist.unwrap()
                    {
                        distances.insert(neighbor_id, new_dist);
                        predecessors.insert(neighbor_id, current_node);
                        priority_queue.push(std::cmp::Reverse((
                            ordered_float::OrderedFloat(new_dist),
                            neighbor_id,
                        )));
                    }
                }
            }
        }

        
        let mut paths: std::collections::HashMap<u32, Vec<u32>> = std::collections::HashMap::new();
        for &target_id in distances.keys() {
            if target_id == source_id {
                paths.insert(target_id, vec![source_id]);
                continue;
            }

            let mut path = vec![target_id];
            let mut current = target_id;

            while let Some(&pred) = predecessors.get(&current) {
                path.push(pred);
                current = pred;
                if current == source_id {
                    break;
                }
            }

            path.reverse();
            paths.insert(target_id, path);
        }

        let computation_time = start_time.elapsed().as_secs_f32() * 1000.0;

        info!(
            "SSSP computed from node {}: {} reachable nodes out of {} in {:.2}ms",
            source_id,
            distances.len(),
            self.node_map.len(),
            computation_time
        );

        Ok(PathfindingResult {
            source_node: source_id,
            distances,
            paths,
            computation_time_ms: computation_time,
        })
    }
}

impl Handler<PhysicsPauseMessage> for GraphServiceActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: PhysicsPauseMessage, _ctx: &mut Self::Context) -> Self::Result {
        if msg.pause {
            self.simulation_params.is_physics_paused = true;
            info!("[AUTO-PAUSE] Physics manually paused: {}", msg.reason);
        } else {
            self.resume_physics_if_paused(msg.reason);
        }
        Ok(())
    }
}

impl Handler<NodeInteractionMessage> for GraphServiceActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: NodeInteractionMessage, _ctx: &mut Self::Context) -> Self::Result {
        
        if self
            .simulation_params
            .auto_pause_config
            .resume_on_interaction
        {
            let reason = match msg.interaction_type {
                NodeInteractionType::Dragged => format!("Node {} dragged", msg.node_id),
                NodeInteractionType::Selected => format!("Node {} selected", msg.node_id),
                NodeInteractionType::Released => format!("Node {} released", msg.node_id),
            };
            self.resume_physics_if_paused(reason);
        }

        
        if let (Some(_position), NodeInteractionType::Dragged) =
            (msg.position, &msg.interaction_type)
        {
            
            if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&msg.node_id) {
                
                
                
                node.data.vx = 0.0;
                node.data.vy = 0.0;
                node.data.vz = 0.0;
            }
        }

        Ok(())
    }
}

impl Handler<ForceResumePhysics> for GraphServiceActor {
    type Result = Result<(), VisionFlowError>;

    fn handle(&mut self, msg: ForceResumePhysics, _ctx: &mut Self::Context) -> Self::Result {
        self.resume_physics_if_paused(msg.reason);
        Ok(())
    }
}

impl Handler<GetEquilibriumStatus> for GraphServiceActor {
    type Result = Result<bool, VisionFlowError>;

    fn handle(&mut self, _msg: GetEquilibriumStatus, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.simulation_params.is_physics_paused)
    }
}

// ============================================================================
// BATCH OPERATION HANDLERS - Optimized data ingestion pipeline
// ============================================================================

impl Handler<BatchAddNodes> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BatchAddNodes, _ctx: &mut Self::Context) -> Self::Result {
        self.batch_add_nodes(msg.nodes)
    }
}

impl Handler<BatchAddEdges> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BatchAddEdges, _ctx: &mut Self::Context) -> Self::Result {
        self.batch_add_edges(msg.edges)
    }
}

impl Handler<BatchGraphUpdate> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BatchGraphUpdate, _ctx: &mut Self::Context) -> Self::Result {
        self.batch_graph_update(
            msg.nodes,
            msg.edges,
            msg.remove_node_ids,
            msg.remove_edge_ids,
        )
    }
}

impl Handler<FlushUpdateQueue> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: FlushUpdateQueue, _ctx: &mut Self::Context) -> Self::Result {
        self.flush_update_queue()
    }
}

impl Handler<ConfigureUpdateQueue> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: ConfigureUpdateQueue, _ctx: &mut Self::Context) -> Self::Result {
        self.configure_update_queue(
            msg.max_operations,
            msg.flush_interval_ms,
            msg.enable_auto_flush,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::metadata::{FileMetadata, MetadataStore};
    use crate::types::vec3::Vec3Data;
    use chrono::Utc;
    use std::collections::HashMap;

    
    
    #[actix::test]
    async fn test_position_preservation_across_rebuilds() {
        
        let mut actor = GraphServiceActor::new();
        let mut ctx = actix::Context::new();

        
        let mut metadata = MetadataStore::new();
        metadata.insert(
            "file1.md".to_string(),
            FileMetadata {
                file_name: "file1.md".to_string(),
                file_size: 1000,
                node_size: 100.0,
                hyperlink_count: 5,
                sha1: "abc123".to_string(),
                node_id: "1".to_string(),
                last_modified: Utc::now(),
                last_content_change: None,
                last_commit: None,
                change_count: Some(0),
                file_blob_sha: None,
                perplexity_link: "".to_string(),
                last_perplexity_process: None,
                topic_counts: HashMap::new(),
            },
        );
        metadata.insert(
            "file2.md".to_string(),
            FileMetadata {
                file_name: "file2.md".to_string(),
                file_size: 2000,
                node_size: 200.0,
                hyperlink_count: 10,
                sha1: "def456".to_string(),
                node_id: "2".to_string(),
                last_modified: Utc::now(),
                last_content_change: None,
                last_commit: None,
                change_count: Some(0),
                file_blob_sha: None,
                perplexity_link: "".to_string(),
                last_perplexity_process: None,
                topic_counts: HashMap::new(),
            },
        );

        
        assert!(actor
            .build_from_metadata(metadata.clone(), &mut ctx)
            .is_ok());

        
        let initial_positions: HashMap<String, (Vec3Data, Vec3Data)> = actor
            .node_map
            .values()
            .map(|node| {
                (
                    node.metadata_id.clone(),
                    (node.data.position(), node.data.velocity()),
                )
            })
            .collect();

        assert_eq!(
            initial_positions.len(),
            2,
            "Should have 2 nodes after first build"
        );

        
        let modified_position = Vec3Data::new(10.0, 20.0, 30.0);
        let modified_velocity = Vec3Data::new(1.0, 2.0, 3.0);

        for node in Arc::make_mut(&mut actor.node_map).values_mut() {
            if node.metadata_id == "file1" {
                node.data.x = modified_position.x;
                node.data.y = modified_position.y;
                node.data.z = modified_position.z;
                node.data.vx = modified_velocity.x;
                node.data.vy = modified_velocity.y;
                node.data.vz = modified_velocity.z;
            }
        }

        
        for node in &mut Arc::make_mut(&mut actor.graph_data).nodes {
            if node.metadata_id == "file1" {
                node.data.x = modified_position.x;
                node.data.y = modified_position.y;
                node.data.z = modified_position.z;
                node.data.vx = modified_velocity.x;
                node.data.vy = modified_velocity.y;
                node.data.vz = modified_velocity.z;
            }
        }

        
        assert!(actor
            .build_from_metadata(metadata.clone(), &mut ctx)
            .is_ok());

        
        let file1_node = actor
            .node_map
            .values()
            .find(|node| node.metadata_id == "file1")
            .expect("file1 node should exist after rebuild");

        assert_eq!(file1_node.data.x, 10.0, "Position X should be preserved");
        assert_eq!(file1_node.data.y, 20.0, "Position Y should be preserved");
        assert_eq!(file1_node.data.z, 30.0, "Position Z should be preserved");
        assert_eq!(file1_node.data.vx, 1.0, "Velocity X should be preserved");
        assert_eq!(file1_node.data.vy, 2.0, "Velocity Y should be preserved");
        assert_eq!(file1_node.data.vz, 3.0, "Velocity Z should be preserved");

        
        let file2_node = actor
            .node_map
            .values()
            .find(|node| node.metadata_id == "file2")
            .expect("file2 node should exist after rebuild");

        let original_file2_pos = initial_positions.get("file2").unwrap().0;
        assert_eq!(
            file2_node.data.position(),
            original_file2_pos,
            "file2 position should be preserved"
        );
    }

    
    #[actix::test]
    async fn test_new_nodes_get_initial_positions() {
        let mut actor = GraphServiceActor::new();
        let mut ctx = actix::Context::new();

        
        let mut metadata1 = MetadataStore::new();
        metadata1.insert(
            "file1.md".to_string(),
            FileMetadata {
                file_name: "file1.md".to_string(),
                file_size: 1000,
                node_size: 100.0,
                hyperlink_count: 5,
                sha1: "abc123".to_string(),
                node_id: "1".to_string(),
                last_modified: Utc::now(),
                last_content_change: None,
                last_commit: None,
                change_count: Some(0),
                file_blob_sha: None,
                perplexity_link: "".to_string(),
                last_perplexity_process: None,
                topic_counts: HashMap::new(),
            },
        );

        assert!(actor.build_from_metadata(metadata1, &mut ctx).is_ok());
        assert_eq!(
            actor.node_map.len(),
            1,
            "Should have 1 node after first build"
        );

        
        let mut metadata2 = MetadataStore::new();
        metadata2.insert(
            "file1.md".to_string(),
            FileMetadata {
                file_name: "file1.md".to_string(),
                file_size: 1000,
                node_size: 100.0,
                hyperlink_count: 5,
                sha1: "abc123".to_string(),
                node_id: "1".to_string(),
                last_modified: Utc::now(),
                last_content_change: None,
                last_commit: None,
                change_count: Some(0),
                file_blob_sha: None,
                perplexity_link: "".to_string(),
                last_perplexity_process: None,
                topic_counts: HashMap::new(),
            },
        );
        metadata2.insert(
            "file2.md".to_string(),
            FileMetadata {
                file_name: "file2.md".to_string(),
                file_size: 2000,
                node_size: 200.0,
                hyperlink_count: 10,
                sha1: "def456".to_string(),
                node_id: "2".to_string(),
                last_modified: Utc::now(),
                last_content_change: None,
                last_commit: None,
                change_count: Some(0),
                file_blob_sha: None,
                perplexity_link: "".to_string(),
                last_perplexity_process: None,
                topic_counts: HashMap::new(),
            },
        );

        assert!(actor.build_from_metadata(metadata2, &mut ctx).is_ok());
        assert_eq!(
            actor.node_map.len(),
            2,
            "Should have 2 nodes after second build"
        );

        
        let file2_node = actor
            .node_map
            .values()
            .find(|node| node.metadata_id == "file2")
            .expect("file2 node should exist");

        let pos = file2_node.data.position();
        let distance_from_origin = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
        assert!(
            distance_from_origin > 0.1,
            "New node should not be at origin, distance: {}",
            distance_from_origin
        );
    }
}
