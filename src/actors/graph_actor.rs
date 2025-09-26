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
//! // Traditional usage (unchanged)
//! let actor = GraphServiceActor::new(client_manager, gpu_compute_addr);
//!
//! // Enhanced features
//! actor.update_advanced_physics_params(advanced_params)?;
//! actor.trigger_stress_optimization()?;
//! let status = actor.get_semantic_analysis_status();
//! ```

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::Duration;
use log::{debug, info, warn, error, trace};
use glam::Vec3;
use crate::types::Vec3Data;
 
use crate::actors::messages::*;
use crate::errors::VisionFlowError;
// use crate::utils::binary_protocol; // Unused
use crate::actors::client_manager_actor::ClientManagerActor;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::{MetadataStore, FileMetadata};
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::{BinaryNodeData, BinaryNodeDataClient, glam_to_vec3data}; // Added glam_to_vec3data
// Using the modular GPU system's ForceComputeActor for physics computation
use crate::actors::gpu::GPUManagerActor;
use crate::models::simulation_params::SimulationParams;
use crate::config::AutoBalanceConfig;

// Advanced physics and AI modules
use crate::models::constraints::{ConstraintSet, Constraint, AdvancedParams};
use crate::services::semantic_analyzer::{SemanticAnalyzer, SemanticFeatures};
use crate::services::edge_generation::{AdvancedEdgeGenerator, EdgeGenerationConfig};
use crate::utils::unified_gpu_compute::{UnifiedGPUCompute};
// use crate::models::simulation_params::SimParams; // Unused
use crate::physics::stress_majorization::StressMajorizationSolver;
use std::sync::Mutex;

pub struct GraphServiceActor {
    graph_data: Arc<GraphData>, // Changed to Arc<GraphData>
    node_map: Arc<HashMap<u32, Node>>, // Changed to Arc for shared access
    gpu_compute_addr: Option<Addr<GPUManagerActor>>, // GPUManagerActor for coordinated GPU computation
    client_manager: Addr<ClientManagerActor>,
    simulation_running: AtomicBool,
    shutdown_complete: Arc<AtomicBool>,
    next_node_id: AtomicU32,
    bots_graph_data: Arc<GraphData>, // Changed to Arc for shared access
    simulation_params: SimulationParams, // Physics simulation parameters
    
    // Advanced hybrid solver components - GPU context managed by ForceComputeActor
    gpu_init_in_progress: bool, // Flag to prevent multiple initialization attempts
    gpu_initialized: bool,  // Track if GPU has been successfully initialized
    constraint_set: ConstraintSet,
    semantic_analyzer: SemanticAnalyzer,
    edge_generator: AdvancedEdgeGenerator,
    stress_solver: StressMajorizationSolver,
    semantic_features_cache: HashMap<String, SemanticFeatures>,
    
    // Advanced physics parameters
    advanced_params: AdvancedParams,
    
    // Control flow state
    stress_step_counter: u32,
    constraint_update_counter: u32,
    last_semantic_analysis: Option<std::time::Instant>,
    
    // Auto-balance tracking
    settings_addr: Option<Addr<crate::actors::settings_actor::SettingsActor>>,
    auto_balance_history: Vec<f32>, // Track max distances to detect minima
    stable_count: u32, // Count stable frames to detect settling
    auto_balance_notifications: Arc<Mutex<Vec<AutoBalanceNotification>>>, // Store notifications for REST API
    kinetic_energy_history: Vec<f32>, // Track kinetic energy for stability detection
    last_adjustment_time: std::time::Instant, // Cooldown tracking
    current_state: AutoBalanceState, // Track current auto-balance state for hysteresis
    frames_since_last_broadcast: Option<u32>, // Track frames since last position broadcast (deprecated - use time-based instead)
    last_broadcast_time: Option<std::time::Instant>, // Time-based broadcast tracking to fix 10-second delay issue
    initial_positions_sent: bool, // Track if initial positions have been sent to clients
    
    // Smooth parameter transitions
    target_params: SimulationParams, // Target physics parameters
    param_transition_rate: f32, // How fast to transition (0.0 - 1.0)
    
    // Position change tracking to avoid unnecessary updates
    previous_positions: HashMap<u32, crate::types::vec3::Vec3Data>, // Track previous positions for change detection
}

/// Auto-balance state tracking for hysteresis prevention
#[derive(Debug, Clone, PartialEq)]
enum AutoBalanceState {
    Stable,         // System is in equilibrium
    Spreading,      // Nodes are spreading out
    Clustering,     // Nodes are clustering together
    Bouncing,       // Nodes are bouncing off boundaries
    Oscillating,    // System is oscillating between states
    Adjusting,      // Currently making parameter adjustments
}

// Auto-balance notification structure
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoBalanceNotification {
    pub message: String,
    pub timestamp: i64,
    pub severity: String, // "info", "warning", "success"
}

impl GraphServiceActor {
    fn smooth_transition_params(&mut self) {
        // Smoothly transition current params to target params
        let rate = self.param_transition_rate;
        
        // Use exponential smoothing for each parameter
        self.simulation_params.repel_k = self.simulation_params.repel_k * (1.0 - rate) + self.target_params.repel_k * rate;
        self.simulation_params.damping = self.simulation_params.damping * (1.0 - rate) + self.target_params.damping * rate;
        self.simulation_params.max_velocity = self.simulation_params.max_velocity * (1.0 - rate) + self.target_params.max_velocity * rate;
        self.simulation_params.spring_k = self.simulation_params.spring_k * (1.0 - rate) + self.target_params.spring_k * rate;
        self.simulation_params.viewport_bounds = self.simulation_params.viewport_bounds * (1.0 - rate) + self.target_params.viewport_bounds * rate;
        
        // New CUDA kernel parameters - smooth transitions
        self.simulation_params.max_repulsion_dist = self.simulation_params.max_repulsion_dist * (1.0 - rate) + self.target_params.max_repulsion_dist * rate;
        self.simulation_params.boundary_force_strength = self.simulation_params.boundary_force_strength * (1.0 - rate) + self.target_params.boundary_force_strength * rate;
        self.simulation_params.cooling_rate = self.simulation_params.cooling_rate * (1.0 - rate) + self.target_params.cooling_rate * rate;
        self.simulation_params.spring_k = self.simulation_params.spring_k * (1.0 - rate) + self.target_params.spring_k * rate;
        
        // For boolean values, switch immediately when more than halfway
        if (self.target_params.enable_bounds as i32 - self.simulation_params.enable_bounds as i32).abs() > 0 {
            self.simulation_params.enable_bounds = self.target_params.enable_bounds;
        }
    }
    
    fn set_target_params(&mut self, new_params: SimulationParams) {
        self.target_params = new_params;
        // Optionally increase transition rate for urgent changes
        let damping_change = (self.target_params.damping - self.simulation_params.damping).abs();
        let repulsion_cutoff_change = (self.target_params.max_repulsion_dist - self.simulation_params.max_repulsion_dist).abs();
        let grid_cell_change = (self.target_params.boundary_force_strength - self.simulation_params.boundary_force_strength).abs();
        
        if damping_change > 0.3 || repulsion_cutoff_change > 20.0 || grid_cell_change > 0.5 {
            self.param_transition_rate = 0.2; // Faster transition for large changes
        } else {
            self.param_transition_rate = 0.1; // Normal transition rate
        }
    }
    
    
    /// Determines auto-balance state with hysteresis bands to prevent rapid switching
    fn determine_auto_balance_state(
        &self,
        max_distance: f32,
        boundary_nodes: u32,
        total_nodes: usize,
        has_numerical_instability: bool,
        has_spatial_issues: bool,
        config: &crate::config::AutoBalanceConfig,
    ) -> AutoBalanceState {
        // Priority 1: Critical issues
        if has_numerical_instability {
            return AutoBalanceState::Adjusting;
        }
        
        // Priority 2: Bouncing nodes (at boundaries)
        if boundary_nodes as f32 > (total_nodes as f32 * config.bouncing_node_percentage) {
            return AutoBalanceState::Bouncing;
        }
        
        // Priority 3: Oscillation detection
        if self.auto_balance_history.len() >= config.oscillation_detection_frames {
            let recent = &self.auto_balance_history[self.auto_balance_history.len() - config.oscillation_detection_frames..];
            let changes = recent.windows(2)
                .filter(|w| (w[0] - w[1]).abs() > config.oscillation_change_threshold)
                .count();
            if changes > config.min_oscillation_changes {
                return AutoBalanceState::Oscillating;
            }
        }
        
        // Priority 4: Distance-based states with hysteresis
        match self.current_state {
            AutoBalanceState::Spreading => {
                // When spreading, require going below threshold minus hysteresis to switch
                if max_distance < (config.spreading_distance_threshold - config.spreading_hysteresis_buffer) {
                    if max_distance < (config.clustering_distance_threshold + config.clustering_hysteresis_buffer) {
                        AutoBalanceState::Clustering
                    } else {
                        AutoBalanceState::Stable
                    }
                } else {
                    AutoBalanceState::Spreading // Stay in spreading state
                }
            },
            AutoBalanceState::Clustering => {
                // When clustering, require going above threshold plus hysteresis to switch
                if max_distance > (config.clustering_distance_threshold + config.clustering_hysteresis_buffer) {
                    if max_distance > (config.spreading_distance_threshold - config.spreading_hysteresis_buffer) {
                        AutoBalanceState::Spreading
                    } else {
                        AutoBalanceState::Stable
                    }
                } else {
                    AutoBalanceState::Clustering // Stay in clustering state
                }
            },
            _ => {
                // From stable or other states, use normal thresholds
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
    
    /// Applies gradual parameter adjustments based on state
    fn apply_gradual_adjustment(&mut self, state: AutoBalanceState, config: &crate::config::AutoBalanceConfig) -> bool {
        let mut adjustment_made = false;
        let adjustment_rate = config.parameter_adjustment_rate;
        
        match state {
            AutoBalanceState::Spreading => {
                // Gradually increase attraction and center gravity
                let mut new_target = self.target_params.clone();
                
                let attraction_factor = 1.0 + adjustment_rate;
                new_target.spring_k = (self.simulation_params.spring_k * attraction_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));
                
                let spring_factor = 1.0 + adjustment_rate * 0.5; // Smaller spring adjustment
                new_target.spring_k = (self.simulation_params.spring_k * spring_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));
                
                self.set_target_params(new_target);
                self.send_auto_balance_notification("Gradual adjustment: Increasing attraction to counter spreading");
                adjustment_made = true;
            },
            AutoBalanceState::Clustering => {
                // Gradually increase repulsion
                let mut new_target = self.target_params.clone();
                
                let repulsion_factor = 1.0 + adjustment_rate;
                new_target.repel_k = (self.simulation_params.repel_k * repulsion_factor)
                    .max(self.simulation_params.repel_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.repel_k * (1.0 + config.max_adjustment_factor));
                
                // Slightly reduce attraction
                let attraction_factor = 1.0 - adjustment_rate * 0.5;
                new_target.spring_k = (self.simulation_params.spring_k * attraction_factor)
                    .max(self.simulation_params.spring_k * (1.0 + config.min_adjustment_factor))
                    .min(self.simulation_params.spring_k * (1.0 + config.max_adjustment_factor));
                
                self.set_target_params(new_target);
                self.send_auto_balance_notification("Gradual adjustment: Increasing repulsion to counter clustering");
                adjustment_made = true;
            },
            AutoBalanceState::Bouncing => {
                // Gradual damping increase and velocity reduction
                let mut new_target = self.target_params.clone();
                
                let damping_factor = 1.0 + adjustment_rate * 0.5;
                new_target.damping = (self.simulation_params.damping * damping_factor).min(0.99);
                
                let velocity_factor = 1.0 - adjustment_rate * 0.5;
                new_target.max_velocity = (self.simulation_params.max_velocity * velocity_factor).max(1.0);
                
                self.set_target_params(new_target);
                self.send_auto_balance_notification("Gradual adjustment: Increasing damping to reduce bouncing");
                adjustment_made = true;
            },
            AutoBalanceState::Oscillating => {
                // Aggressive damping to stop oscillation
                let mut new_target = self.target_params.clone();
                new_target.damping = (self.simulation_params.damping * 1.2).min(0.98);
                new_target.max_velocity = self.simulation_params.max_velocity * 0.7;
                
                // Also slow down the parameter transition rate temporarily
                self.param_transition_rate = config.parameter_dampening_factor;
                
                self.set_target_params(new_target);
                self.send_auto_balance_notification("Emergency adjustment: Stopping oscillation with increased damping");
                adjustment_made = true;
            },
            AutoBalanceState::Adjusting => {
                // Handle numerical instability with conservative parameters
                let mut new_target = self.target_params.clone();
                new_target.dt = (self.simulation_params.dt * 0.8).max(0.001);
                new_target.damping = (self.simulation_params.damping * 1.1).min(0.99);
                
                self.set_target_params(new_target);
                self.send_auto_balance_notification("Emergency adjustment: Fixing numerical instability");
                adjustment_made = true;
            },
            AutoBalanceState::Stable => {
                // Gradually return to baseline parameters
                self.param_transition_rate = config.parameter_dampening_factor;
                // No immediate adjustment needed
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
        
        // Determine severity based on message content
        let severity = if message.contains("disabled") || message.contains("failed") {
            "warning"
        } else if message.contains("stable") || message.contains("found") {
            "success"
        } else {
            "info"
        }.to_string();
        
        // Store notification for REST API retrieval
        let notification = AutoBalanceNotification {
            message: message.to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            severity,
        };
        
        if let Ok(mut notifications) = self.auto_balance_notifications.lock() {
            notifications.push(notification);
            // Keep only last 50 notifications
            if notifications.len() > 50 {
                let drain_count = notifications.len() - 50;
                notifications.drain(0..drain_count);
            }
        }
    }
    
    fn notify_settings_update(&self) {
        // Send updated physics parameters back to settings system
        // This will trigger UI updates
        info!("[AUTO-BALANCE] Notifying settings system of parameter changes");
        
        // Send message to settings actor to update and propagate to UI
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
            
            let update_msg = crate::actors::messages::UpdatePhysicsFromAutoBalance { 
                physics_update 
            };
            settings_addr.do_send(update_msg);
        }
    }
    
    pub fn new(
        client_manager: Addr<ClientManagerActor>,
        gpu_compute_addr: Option<Addr<GPUManagerActor>>,
        settings_addr: Option<Addr<crate::actors::settings_actor::SettingsActor>>,
    ) -> Self {
        let advanced_params = AdvancedParams::default();
        let semantic_analyzer = SemanticAnalyzer::new(
            crate::services::semantic_analyzer::SemanticAnalyzerConfig::default()
        );
        let edge_generator = AdvancedEdgeGenerator::new(
            EdgeGenerationConfig::default()
        );
        let stress_solver = StressMajorizationSolver::from_advanced_params(&advanced_params);
        
        // Initialize with logseq (knowledge graph) physics from settings
        // This will be updated when settings are loaded, but provides better defaults
        let simulation_params = {
            // Try to get settings from the global config
            if let Ok(settings_yaml) = std::fs::read_to_string("/app/settings.yaml")
                .or_else(|_| std::fs::read_to_string("data/settings.yaml")) {
                if let Ok(settings) = serde_yaml::from_str::<crate::config::AppFullSettings>(&settings_yaml) {
                    // Use logseq physics as the default for knowledge graph
                    let physics = settings.get_physics("logseq");
                    SimulationParams::from(physics)
                } else {
                    warn!("Failed to parse settings.yaml, using defaults");
                    SimulationParams::default()
                }
            } else {
                info!("Settings file not found at startup, using defaults (will be updated when settings load)");
                SimulationParams::default()
            }
        };
        
        // Clone simulation_params for target_params before moving it
        let target_params = simulation_params.clone();
        
        Self {
            graph_data: Arc::new(GraphData::new()), // Changed to Arc::new
            node_map: Arc::new(HashMap::new()), // Changed to Arc::new for shared access
            gpu_compute_addr,
            client_manager,
            simulation_running: AtomicBool::new(false),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            next_node_id: AtomicU32::new(1),
            bots_graph_data: Arc::new(GraphData::new()), // Changed to Arc::new for shared access
            simulation_params, // Use logseq physics from settings
            
            // Initialize advanced components - GPU context managed by ForceComputeActor
            gpu_init_in_progress: false,
            gpu_initialized: false,  // GPU not yet initialized
            constraint_set: ConstraintSet::default(),
            semantic_analyzer,
            edge_generator,
            stress_solver,
            semantic_features_cache: HashMap::new(),
            advanced_params,
            
            // Initialize control state
            stress_step_counter: 0,
            constraint_update_counter: 0,
            last_semantic_analysis: None,
            
            // Auto-balance tracking
            settings_addr,
            auto_balance_history: Vec::with_capacity(60), // Track last 60 frames (1 second at 60fps)
            stable_count: 0,
            auto_balance_notifications: Arc::new(Mutex::new(Vec::new())),
            kinetic_energy_history: Vec::with_capacity(60), // Track kinetic energy
            last_adjustment_time: std::time::Instant::now(), // Initialize cooldown tracking
            current_state: AutoBalanceState::Stable, // Start in stable state
            frames_since_last_broadcast: None,
            last_broadcast_time: None, // Initialize time-based broadcast tracking
            initial_positions_sent: false,
            
            // Smooth parameter transitions - initialize with actual loaded settings
            target_params,
            param_transition_rate: 0.1, // 10% per frame for smooth transitions
            
            // Position change tracking
            previous_positions: HashMap::new(),
        }
    }

    pub fn get_graph_data(&self) -> &GraphData { // Returns a reference to the inner GraphData
        &self.graph_data // Dereferences Arc<GraphData> to &GraphData
    }

    pub fn get_node_map(&self) -> &HashMap<u32, Node> {
        &self.node_map
    }

    pub fn add_node(&mut self, node: Node) {
        let node_id = node.id; // Store the ID before moving node
        
        // Update node_map
        Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        // Add to graph data if not already present
        if !graph_data_mut.nodes.iter().any(|n| n.id == node.id) {
            graph_data_mut.nodes.push(node);
        } else {
            // Update existing node
            if let Some(existing) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node.id) {
                *existing = node; // Move node here instead of cloning
            }
        }
        
        debug!("Added/updated node: {}", node_id);
    }

    pub fn remove_node(&mut self, node_id: u32) {
        // Remove from node_map
        Arc::make_mut(&mut self.node_map).remove(&node_id);
        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        // Remove from graph data
        graph_data_mut.nodes.retain(|n| n.id != node_id);
        
        // Remove related edges
        graph_data_mut.edges.retain(|e| e.source != node_id && e.target != node_id);
        
        // Clean up position tracking
        self.previous_positions.remove(&node_id);
        
        debug!("Removed node: {}", node_id);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        let edge_id = edge.id.clone(); // Store the ID before moving edge
        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        // Add to graph data if not already present
        if !graph_data_mut.edges.iter().any(|e| e.id == edge.id) {
            graph_data_mut.edges.push(edge);
        } else {
            // Update existing edge
            if let Some(existing) = graph_data_mut.edges.iter_mut().find(|e| e.id == edge.id) {
                *existing = edge; // Move edge here instead of cloning
            }
        }
        
        debug!("Added/updated edge: {}", edge_id);
    }

    pub fn remove_edge(&mut self, edge_id: &str) {
        Arc::make_mut(&mut self.graph_data).edges.retain(|e| e.id != edge_id);
        debug!("Removed edge: {}", edge_id);
    }

    /// Build graph from metadata while preserving existing node positions
    /// 
    /// This method addresses a critical issue where node positions were reset every time
    /// BuildGraphFromMetadata was called (e.g., when clients connected). The fix ensures
    /// that existing nodes maintain their positions across rebuilds, while new nodes
    /// still get proper initial positions.
    /// 
    /// # Position Preservation Strategy
    /// 1. Save existing positions before clearing the node_map
    /// 2. Create new nodes with generated positions 
    /// 3. Restore saved positions for nodes that existed before
    /// 4. Allow physics simulation to continue from preserved positions
    /// 
    /// # Args
    /// * `metadata` - The metadata store containing file information
    /// 
    /// # Returns
    /// * `Ok(())` if the graph was built successfully
    /// * `Err(String)` if there was an error during graph construction
    pub fn build_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        let mut new_graph_data = GraphData::new();
        
        // BREADCRUMB: Save existing node positions before clearing node_map
        // This preserves positions across rebuilds, preventing position reset on client connections
        let mut existing_positions: HashMap<String, (crate::types::vec3::Vec3Data, crate::types::vec3::Vec3Data)> = HashMap::new();
        
        // Save positions from existing nodes indexed by metadata_id
        for node in self.node_map.values() {
            existing_positions.insert(node.metadata_id.clone(), (node.data.position(), node.data.velocity()));
            debug!("Saved position for existing node '{}': ({}, {}, {})", 
                   node.metadata_id, node.data.x, node.data.y, node.data.z);
        }
        debug!("Total existing positions saved: {}", existing_positions.len());
        
        Arc::make_mut(&mut self.node_map).clear();
        self.semantic_features_cache.clear();

        // Phase 1: Build nodes with semantic analysis
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
            // Note: flags field not available in BinaryNodeDataClient
            // Using node_id low bit as flag if needed: node.data.node_id |= 1;

            // BREADCRUMB: Restore existing position if this node was previously created
            // This ensures positions persist across BuildGraphFromMetadata calls
            debug!("Looking for existing position for metadata_id: '{}'", metadata_id_val);
            debug!("Available keys in existing_positions: {:?}", existing_positions.keys().collect::<Vec<_>>());
            
            if let Some((saved_position, saved_velocity)) = existing_positions.get(&metadata_id_val) {
                node.data.x = saved_position.x;
                node.data.y = saved_position.y;
                node.data.z = saved_position.z;
                node.data.vx = saved_velocity.x;
                node.data.vy = saved_velocity.y;
                node.data.vz = saved_velocity.z;
                debug!("Restored position for node '{}': ({}, {}, {})", 
                       metadata_id_val, saved_position.x, saved_position.y, saved_position.z);
            } else {
                debug!("New node '{}' will use generated position: ({}, {}, {})",
                       metadata_id_val, node.data.x, node.data.y, node.data.z);
            }

            // Enhanced metadata with semantic features
            node.metadata.insert("fileName".to_string(), file_meta_data.file_name.clone());
            node.metadata.insert("fileSize".to_string(), file_meta_data.file_size.to_string());
            node.metadata.insert("nodeSize".to_string(), file_meta_data.node_size.to_string());
            node.metadata.insert("hyperlinkCount".to_string(), file_meta_data.hyperlink_count.to_string());
            node.metadata.insert("sha1".to_string(), file_meta_data.sha1.clone());
            node.metadata.insert("lastModified".to_string(), file_meta_data.last_modified.to_rfc3339());
            if !file_meta_data.perplexity_link.is_empty() {
                node.metadata.insert("perplexityLink".to_string(), file_meta_data.perplexity_link.clone());
            }
            if let Some(last_process) = file_meta_data.last_perplexity_process {
                node.metadata.insert("lastPerplexityProcess".to_string(), last_process.to_rfc3339());
            }
            node.metadata.insert("metadataId".to_string(), metadata_id_val.clone());
            
            // Extract semantic features
            let features = self.semantic_analyzer.analyze_metadata(file_meta_data);
            self.semantic_features_cache.insert(metadata_id_val, features);

            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
            new_graph_data.nodes.push(node);
        }
        
        info!("Phase 1 complete: Processed {} nodes from metadata", node_count);
        info!("new_graph_data now contains {} nodes", new_graph_data.nodes.len());
        info!("node_map now contains {} entries", self.node_map.len());

        // Phase 2: Generate enhanced edges with multi-modal similarities  
        info!("Phase 2: Generating enhanced edges with multi-modal similarities");
        info!("Semantic features cache contains {} entries", self.semantic_features_cache.len());
        for (id, features) in &self.semantic_features_cache {
            debug!("Semantic feature ID: '{}' (topics: {})", id, features.topics.len());
        }
        
        let enhanced_edges = self.edge_generator.generate(&self.semantic_features_cache);
        info!("Generated {} enhanced edges", enhanced_edges.len());
        
        // Convert enhanced edges to basic edges and add topic-based edges
        let mut edge_map: HashMap<(u32, u32), f32> = HashMap::new();
        
        // Add semantic similarity edges
        for enhanced_edge in &enhanced_edges {
            debug!("Processing enhanced edge: {} -> {} (weight: {})", 
                enhanced_edge.source, enhanced_edge.target, enhanced_edge.weight);
            
            // Find node IDs from metadata IDs
            if let (Some(source_node), Some(target_node)) = (
                self.node_map.values().find(|n| n.metadata_id == enhanced_edge.source),
                self.node_map.values().find(|n| n.metadata_id == enhanced_edge.target)
            ) {
                let edge_key = if source_node.id < target_node.id { 
                    (source_node.id, target_node.id) 
                } else { 
                    (target_node.id, source_node.id) 
                };
                // Use semantic weight as primary, topic count as secondary
                *edge_map.entry(edge_key).or_insert(0.0) += enhanced_edge.weight;
            }
        }
        
        // Add traditional topic-based edges (with lower weight)
        for (source_filename_ext, source_meta) in &metadata {
            let source_metadata_id = source_filename_ext.trim_end_matches(".md");
            if let Some(source_node) = self.node_map.values().find(|n| n.metadata_id == source_metadata_id) {
                for (target_filename_ext, count) in &source_meta.topic_counts {
                    let target_metadata_id = target_filename_ext.trim_end_matches(".md");
                    if let Some(target_node) = self.node_map.values().find(|n| n.metadata_id == target_metadata_id) {
                        if source_node.id != target_node.id {
                            let edge_key = if source_node.id < target_node.id { 
                                (source_node.id, target_node.id) 
                            } else { 
                                (target_node.id, source_node.id) 
                            };
                            // Lower weight for topic counts to not override semantic similarity
                            *edge_map.entry(edge_key).or_insert(0.0) += (*count as f32) * 0.3;
                        }
                    }
                }
            }
        }

        // Create edges from the combined map
        for ((source_id, target_id), weight) in edge_map {
            new_graph_data.edges.push(Edge::new(source_id, target_id, weight));
        }
        
        // Phase 3: Generate initial semantic constraints from the built graph
        info!("Phase 3: Generating initial semantic constraints");
        
        // Store the graph data first so constraint generation can access it
        self.graph_data = Arc::new(new_graph_data.clone());
        
        // Generate constraints based on semantic analysis
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone);
        
        // Phase 4: Initialize advanced GPU context if needed (async context not available in message handler)
        // Note: Advanced GPU context initialization will be attempted on first physics step
        if self.gpu_compute_addr.is_none() {
            trace!("Advanced GPU context will be initialized on first physics step");
        }
        
        // Update graph data metadata and timestamp
        Arc::make_mut(&mut self.graph_data).metadata = metadata.clone();
        self.last_semantic_analysis = Some(std::time::Instant::now());
        
        info!("Built enhanced graph: {} nodes, {} edges, {} constraints",
              self.graph_data.nodes.len(), self.graph_data.edges.len(), self.constraint_set.constraints.len());
        
        // Note: GPU initialization will be handled by the message handler that calls this method
        
        Ok(())
    }
    
    /// Helper methods for the hybrid solver orchestration
    
    fn prepare_node_positions(&self) -> Vec<(f32, f32, f32)> {
        self.graph_data.nodes.iter().map(|node| {
            (
                node.data.x,
                node.data.y,
                node.data.z,
            )
        }).collect()
    }
    
    
    fn execute_stress_majorization_step(&mut self) {
        if self.graph_data.nodes.len() < 3 {
            return; // Skip for very small graphs
        }
        
        // Skip stress majorization if GPU compute is not available
        if self.gpu_compute_addr.is_none() {
            trace!("Skipping stress majorization - no GPU context available");
            return;
        }
        
        let mut graph_data_clone = self.graph_data.as_ref().clone(); // Still need to clone here for stress solver which modifies the data
        
        match self.stress_solver.optimize(&mut graph_data_clone, &self.constraint_set) {
            Ok(result) => {
                if result.converged || result.final_stress < f32::INFINITY {
                    // Apply optimized positions back to main graph
                    let graph_data_mut = Arc::make_mut(&mut self.graph_data);
                    for (i, node) in graph_data_mut.nodes.iter_mut().enumerate() {
                        if let Some(optimized_node) = graph_data_clone.nodes.get(i) {
                            // Validate positions before applying
                            let new_x = optimized_node.data.x;
                            let new_y = optimized_node.data.y;
                            let new_z = optimized_node.data.z;
                            
                            if new_x.is_finite() && new_y.is_finite() && new_z.is_finite() {
                                // Calculate displacement for safety clamping
                                let old_pos = Vec3Data::new(node.data.x, node.data.y, node.data.z);
                                let displacement_x = new_x - old_pos.x;
                                let displacement_y = new_y - old_pos.y;
                                let displacement_z = new_z - old_pos.z;
                                let displacement_magnitude = (displacement_x * displacement_x + 
                                                            displacement_y * displacement_y + 
                                                            displacement_z * displacement_z).sqrt();
                                
                                // Calculate layout extent for displacement clamping (5% max displacement)
                                let layout_extent = self.simulation_params.viewport_bounds.max(1000.0);
                                let max_displacement = layout_extent * 0.05;
                                
                                let (final_x, final_y, final_z) = if displacement_magnitude > max_displacement {
                                    // Clamp displacement to safe range
                                    let scale = max_displacement / displacement_magnitude;
                                    (
                                        old_pos.x + displacement_x * scale,
                                        old_pos.y + displacement_y * scale,
                                        old_pos.z + displacement_z * scale,
                                    )
                                } else {
                                    (new_x, new_y, new_z)
                                };
                                
                                // Apply boundary constraints with bounded AABB domain
                                if self.simulation_params.enable_bounds && self.simulation_params.viewport_bounds > 0.0 {
                                    let boundary_limit = self.simulation_params.viewport_bounds;
                                    node.data.x = final_x.clamp(-boundary_limit, boundary_limit);
                                    node.data.y = final_y.clamp(-boundary_limit, boundary_limit);
                                    node.data.z = final_z.clamp(-boundary_limit, boundary_limit);
                                } else {
                                    // Apply default AABB bounds to prevent position explosions
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
                    
                    // Update node_map as well with validated positions
                    for node in &graph_data_mut.nodes {
                        if let Some(node_in_map) = Arc::make_mut(&mut self.node_map).get_mut(&node.id) {
                            // Use the already validated and clamped positions
                            node_in_map.data.x = node.data.x;
                            node_in_map.data.y = node.data.y;
                            node_in_map.data.z = node.data.z;
                        }
                    }
                    
                    trace!("Stress majorization step completed: {} iterations, stress = {:.6}", 
                           result.iterations, result.final_stress);
                }
            }
            Err(e) => {
                error!("Stress majorization step failed: {}", e);
            }
        }
    }
    
    fn update_dynamic_constraints(&mut self) {
        // Update dynamic constraints based on semantic analysis
        trace!("Updating dynamic constraints based on semantic analysis");
        
        // Only update if we have recent semantic analysis
        if self.last_semantic_analysis.is_none() {
            return;
        }
        
        // Clear dynamic constraints (keep manually added ones)
        self.constraint_set.set_group_active("semantic_dynamic", false);
        
        // Generate new semantic constraints
        if let Ok(constraints) = self.generate_dynamic_semantic_constraints() {
            let constraint_count = constraints.len();
            for constraint in constraints {
                self.constraint_set.add_to_group("semantic_dynamic", constraint);
            }
            trace!("Updated {} dynamic semantic constraints", constraint_count);
        } else {
            trace!("Failed to generate dynamic semantic constraints");
        }
        
        // Re-cluster nodes based on current positions and semantic features
        if let Ok(clustering_constraints) = self.generate_clustering_constraints() {
            self.constraint_set.set_group_active("clustering_dynamic", false);
            let constraint_count = clustering_constraints.len();
            for constraint in clustering_constraints {
                self.constraint_set.add_to_group("clustering_dynamic", constraint);
            }
            trace!("Updated {} dynamic clustering constraints", constraint_count);
        } else {
            trace!("Failed to generate dynamic clustering constraints");
        }
        
        // Upload updated constraints to GPU
        self.upload_constraints_to_gpu();
    }
    
    fn generate_initial_semantic_constraints(&mut self, graph_data: &std::sync::Arc<GraphData>) {
        // Generate domain-based clustering constraints from semantic analysis
        let mut domain_clusters: HashMap<String, Vec<u32>> = HashMap::new();
        
        for node in &graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                if !features.domains.is_empty() {
                    let domain_key = format!("{:?}", features.domains[0]);
                    domain_clusters.entry(domain_key).or_insert_with(Vec::new).push(node.id);
                }
            }
        }
        
        // Clear existing initial constraints
        self.constraint_set.set_group_active("domain_clustering", false);
        
        // Create clustering constraints for domains with multiple files
        for (domain, node_ids) in domain_clusters {
            if node_ids.len() >= 2 {
                let cluster_constraint = Constraint::cluster(
                    node_ids, 
                    domain.len() as f32, // Use domain as cluster ID
                    0.6 // Medium clustering strength
                );
                self.constraint_set.add_to_group("domain_clustering", cluster_constraint);
            }
        }
        
        // Upload constraints to GPU if available
        self.upload_constraints_to_gpu();
        
        info!("Generated {} initial semantic constraints", 
              self.constraint_set.active_constraints().len());
    }
    
    fn generate_dynamic_semantic_constraints(&self) -> Result<Vec<Constraint>, Box<dyn std::error::Error>> {
        let mut constraints = Vec::new();
        
        // Create separation constraints for high-importance nodes
        let high_importance_nodes: Vec<_> = self.semantic_features_cache
            .iter()
            .filter(|(_, features)| features.importance_score > 0.7)
            .filter_map(|(id, _)| self.node_map.values().find(|n| n.metadata_id == *id).map(|n| n.id))
            .collect();
        
        for i in 0..high_importance_nodes.len() {
            for j in i+1..high_importance_nodes.len() {
                let constraint = Constraint::separation(
                    high_importance_nodes[i],
                    high_importance_nodes[j],
                    100.0 // Minimum separation for important nodes
                );
                constraints.push(constraint);
            }
        }
        
        Ok(constraints)
    }
    
    fn generate_clustering_constraints(&self) -> Result<Vec<Constraint>, Box<dyn std::error::Error>> {
        let mut constraints = Vec::new();
        
        // Group nodes by file type
        let mut type_clusters: HashMap<String, Vec<u32>> = HashMap::new();
        
        for node in &self.graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                type_clusters.entry(features.content.language.clone())
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }
        
        // Create clustering constraints for each file type
        for (file_type, node_ids) in type_clusters {
            if node_ids.len() >= 2 {
                let constraint = Constraint::cluster(
                    node_ids,
                    file_type.len() as f32, // Use type length as cluster ID
                    0.4 // Moderate clustering strength
                );
                constraints.push(constraint);
            }
        }
        
        Ok(constraints)
    }
    
    /// Upload current constraints to GPU via ForceComputeActor
    fn upload_constraints_to_gpu(&mut self) {
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            // Convert constraints to GPU format
            let active_constraints = self.constraint_set.active_constraints();
            let constraint_data: Vec<crate::models::constraints::ConstraintData> = 
                active_constraints.iter().map(|c| c.to_gpu_format()).collect();
                
            // Send constraints to ForceComputeActor
            let upload_msg = crate::actors::messages::UploadConstraintsToGPU {
                constraint_data: constraint_data.clone(),
            };
            
            let gpu_addr_clone = gpu_addr.clone();
            actix::spawn(async move {
                if let Err(e) = gpu_addr_clone.send(upload_msg).await {
                    error!("Failed to send constraints to ForceComputeActor: {}", e);
                } else {
                    trace!("Successfully sent {} constraints to ForceComputeActor", constraint_data.len());
                }
            });
        } else {
            trace!("No GPU compute actor available for constraint upload");
        }
    }
    
    /// Handle control frames for constraint updates
    pub fn handle_constraint_update(&mut self, constraint_data: serde_json::Value) -> Result<(), String> {
        match constraint_data.get("action").and_then(|v| v.as_str()) {
            Some("add_fixed_position") => {
                if let (Some(node_id), Some(x), Some(y), Some(z)) = (
                    constraint_data.get("node_id").and_then(|v| v.as_u64()).map(|v| v as u32),
                    constraint_data.get("x").and_then(|v| v.as_f64()).map(|v| v as f32),
                    constraint_data.get("y").and_then(|v| v.as_f64()).map(|v| v as f32),
                    constraint_data.get("z").and_then(|v| v.as_f64()).map(|v| v as f32),
                ) {
                    let constraint = Constraint::fixed_position(node_id, x, y, z);
                    self.constraint_set.add_to_group("user_fixed", constraint);
                    info!("Added fixed position constraint for node {}", node_id);
                }
            }
            Some("toggle_clustering") => {
                if let Some(enabled) = constraint_data.get("enabled").and_then(|v| v.as_bool()) {
                    self.constraint_set.set_group_active("domain_clustering", enabled);
                    info!("Toggled domain clustering: {}", enabled);
                }
            }
            Some("update_separation_factor") => {
                if let Some(factor) = constraint_data.get("factor").and_then(|v| v.as_f64()).map(|v| v as f32) {
                    self.advanced_params.separation_factor = factor;
                    info!("Updated separation factor to {}", factor);
                }
            }
            Some("enable_hierarchical") => {
                if let Some(enabled) = constraint_data.get("enabled").and_then(|v| v.as_bool()) {
                    self.advanced_params.hierarchical_mode = enabled;
                    if enabled {
                        // Add hierarchical constraints based on directory structure
                        self.generate_hierarchical_constraints();
                    }
                    info!("Set hierarchical mode: {}", enabled);
                }
            }
            _ => {
                warn!("Unknown constraint update action: {:?}", constraint_data.get("action"));
                return Err("Unknown constraint action".to_string());
            }
        }
        
        // Upload updated constraints to GPU after any modification
        self.upload_constraints_to_gpu();
        
        Ok(())
    }
    
    fn generate_hierarchical_constraints(&mut self) {
        // Create layer constraints based on directory depth
        let mut depth_layers: HashMap<u32, Vec<u32>> = HashMap::new();
        
        for node in &self.graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                depth_layers.entry(features.structural.directory_depth)
                    .or_insert_with(Vec::new)
                    .push(node.id);
            }
        }
        
        // Create alignment constraints for each depth layer
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
                self.constraint_set.add_to_group("hierarchical_layers", constraint);
            }
        }
        
        info!("Generated hierarchical layer constraints for {} depths", depth_layers.len());
    }

    /// Helper method to detect spatial hashing effectiveness issues
    fn detect_spatial_hashing_issues(&self, positions: &[(f32, f32, f32)], config: &AutoBalanceConfig) -> (bool, f32) {
        if positions.len() < 2 {
            return (false, 1.0);
        }
        
        let current_grid_cell_size = self.simulation_params.max_repulsion_dist; // Used as grid_cell_size proxy
        let mut clustering_detected = false;
        let mut efficiency_score = 1.0;
        
        // Calculate average inter-node distance to assess grid efficiency
        let mut total_distance = 0.0f32;
        let mut distance_count = 0;
        
        for i in 0..positions.len() {
            for j in i+1..std::cmp::min(i+10, positions.len()) { // Sample to avoid O(n²) complexity
                let pos1 = positions[i];
                let pos2 = positions[j];
                let dist = ((pos1.0 - pos2.0).powi(2) + (pos1.1 - pos2.1).powi(2) + (pos1.2 - pos2.2).powi(2)).sqrt();
                total_distance += dist;
                distance_count += 1;
            }
        }
        
        let avg_distance = if distance_count > 0 { total_distance / distance_count as f32 } else { 1.0 };
        
        // If average distance is much smaller than grid cell size, we have clustering issues
        if avg_distance < current_grid_cell_size * 0.5 {
            clustering_detected = true;
            efficiency_score = avg_distance / current_grid_cell_size;
        }
        
        // Check for excessive clustering (too many nodes in small areas)
        let cluster_density = positions.len() as f32 / (avg_distance * avg_distance);
        if cluster_density > config.cluster_density_threshold {
            clustering_detected = true;
            efficiency_score = efficiency_score.min(0.3);
        }
        
        (clustering_detected, efficiency_score)
    }
    
    /// Helper method to detect numerical instability
    fn detect_numerical_instability(&self, positions: &[(f32, f32, f32)], config: &AutoBalanceConfig) -> bool {
        // Check for NaN or infinite positions
        for &(x, y, z) in positions {
            if !x.is_finite() || !y.is_finite() || !z.is_finite() {
                return true;
            }
        }
        
        // Check for excessive velocities in kinetic energy history
        if let Some(&recent_ke) = self.kinetic_energy_history.last() {
            if recent_ke > config.numerical_instability_threshold {
                // Check if kinetic energy is growing exponentially
                if self.kinetic_energy_history.len() >= 5 {
                    let last_5: Vec<f32> = self.kinetic_energy_history.iter().rev().take(5).cloned().collect();
                    let is_growing = last_5.windows(2).all(|w| w[0] > w[1] * 1.5);
                    if is_growing {
                        return true;
                    }
                }
            }
        }
        
        false
    }

    pub fn update_node_positions(&mut self, positions: Vec<(u32, BinaryNodeData)>) {
        // CRITICAL FIX: Skip position updates if physics is paused
        // This prevents the graph from jumping when it's in a settled/stable state
        if self.simulation_params.is_physics_paused {
            debug!("Physics is paused, skipping position update for {} nodes", positions.len());
            return;
        }
        
        let mut updated_count = 0;
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        for (node_id, mut position_data) in positions {
            // CRITICAL FIX: Validate and clamp positions to prevent extreme values
            // This prevents z-axis from going to -99.99 or other boundary issues
            const MAX_COORD: f32 = 500.0;
            const MIN_Z: f32 = -50.0;
            const MAX_Z: f32 = 50.0;
            
            // Clamp positions to reasonable bounds
            position_data.x = position_data.x.clamp(-MAX_COORD, MAX_COORD);
            position_data.y = position_data.y.clamp(-MAX_COORD, MAX_COORD);
            position_data.z = position_data.z.clamp(MIN_Z, MAX_Z);
            
            // Detect and warn about extreme positions
            if position_data.z.abs() > 45.0 {
                debug!("Node {} has extreme z position: {}, clamped to range [{}, {}]", 
                    node_id, position_data.z, MIN_Z, MAX_Z);
            }
            
            // Update in node_map
            if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&node_id) {
                node.data.x = position_data.x;
                node.data.y = position_data.y;
                node.data.z = position_data.z;
                node.data.vx = position_data.vx;
                node.data.vy = position_data.vy;
                node.data.vz = position_data.vz;
                updated_count += 1;
            }
            
            // Update in graph_data.nodes
            if let Some(node) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node_id) {
                node.data.x = position_data.x;
                node.data.y = position_data.y;
                node.data.z = position_data.z;
                node.data.vx = position_data.vx;
                node.data.vy = position_data.vy;
                node.data.vz = position_data.vz;
            }
        }
        
        // Position sampling disabled to reduce log volume
        
        // Check for extreme positions, boundary nodes, and calculate metrics including kinetic energy
        let config = &self.simulation_params.auto_balance_config;
        let mut extreme_count = 0;
        let mut boundary_nodes = 0;
        let mut max_distance = 0.0f32;
        let mut total_distance = 0.0f32;
        let mut positions = Vec::new();
        let mut total_kinetic_energy = 0.0f32;
        
        for (_, node) in self.node_map.iter() {
            let dist = node.data.x.abs()
                .max(node.data.y.abs())
                .max(node.data.z.abs());
            max_distance = max_distance.max(dist);
            total_distance += dist;
            positions.push(dist);
            
            // Calculate kinetic energy: KE = 0.5 * mass * velocity^2
            // Assuming unit mass for simplicity (mass = 1)
            let velocity_squared = node.data.vx * node.data.vx +
                                  node.data.vy * node.data.vy +
                                  node.data.vz * node.data.vz;
            total_kinetic_energy += 0.5 * velocity_squared;
            
            // FIXED: Percentage-based boundary detection relative to viewport_bounds
            let viewport_bounds = self.simulation_params.viewport_bounds;
            let boundary_min_threshold = viewport_bounds * (config.boundary_min_distance / 100.0); // 90% of bounds
            let boundary_max_threshold = viewport_bounds * (config.boundary_max_distance / 100.0); // 100% of bounds
            
            if dist > config.extreme_distance_threshold {
                extreme_count += 1;
            } else if dist >= boundary_min_threshold && dist <= boundary_max_threshold {
                // Count nodes at boundary using percentage of viewport_bounds
                boundary_nodes += 1;
            }
        }
        
        let avg_distance = if !self.node_map.is_empty() {
            total_distance / self.node_map.len() as f32
        } else {
            0.0
        };
        
        // Auto-pause equilibrium detection
        self.check_and_handle_equilibrium(total_kinetic_energy, self.node_map.len());
        
        // IMPROVED Auto-balance system with hysteresis, cooldown, and gradual adjustments
        // Fixed oscillation issues with dampening and state tracking
        if self.simulation_params.auto_balance {
            // CRITICAL FIX: Skip auto-balance if graph is already settled/stable
            // This prevents the graph from jumping after it has settled
            if self.stable_count > 30 {
                debug!("Graph is stable (stable_count: {}), skipping auto-balance", self.stable_count);
                return;
            }
            
            // Normalize kinetic energy by number of nodes
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
            
            // Track history for minima detection
            self.auto_balance_history.push(max_distance);
            if self.auto_balance_history.len() > 60 {
                self.auto_balance_history.remove(0);
            }
            
            // Track kinetic energy history
            self.kinetic_energy_history.push(avg_kinetic_energy);
            if self.kinetic_energy_history.len() > 60 {
                self.kinetic_energy_history.remove(0);
            }
            
            // Check cooldown periods to prevent rapid adjustments
            let now = std::time::Instant::now();
            let config = &self.simulation_params.auto_balance_config;
            
            // Check if enough time has passed since last adjustment
            let adjustment_cooldown_duration = std::time::Duration::from_millis(config.adjustment_cooldown_ms);
            let time_since_last_adjustment = now.duration_since(self.last_adjustment_time);
            
            if time_since_last_adjustment >= adjustment_cooldown_duration {
                    
                // Prepare position data for advanced analysis
                let position_data: Vec<(f32, f32, f32)> = self.node_map.values()
                    .map(|node| (node.data.x, node.data.y, node.data.z))
                    .collect();
                
                // Detect spatial hashing issues and numerical instability
                let (has_spatial_issues, efficiency_score) = self.detect_spatial_hashing_issues(&position_data, config);
                let has_numerical_instability = self.detect_numerical_instability(&position_data, config);
                
                // Determine current state with hysteresis bands to prevent rapid switching
                let new_state = self.determine_auto_balance_state(
                    max_distance, 
                    boundary_nodes, 
                    self.node_map.len(),
                    has_numerical_instability,
                    has_spatial_issues,
                    config
                );
                
                // Check for stability (equilibrium detection)
                let is_stable = if self.auto_balance_history.len() >= 30 && self.kinetic_energy_history.len() >= 30 {
                    // Check position variance
                    let recent_avg = self.auto_balance_history[self.auto_balance_history.len()-30..]
                        .iter().sum::<f32>() / 30.0;
                    let position_variance = self.auto_balance_history[self.auto_balance_history.len()-30..]
                        .iter()
                        .map(|x| (x - recent_avg).powi(2))
                        .sum::<f32>() / 30.0;
                    
                    // Check kinetic energy (should be low for stable state)
                    let recent_ke = self.kinetic_energy_history[self.kinetic_energy_history.len()-30..]
                        .iter().sum::<f32>() / 30.0;
                    let ke_variance = self.kinetic_energy_history[self.kinetic_energy_history.len()-30..]
                        .iter()
                        .map(|x| (x - recent_ke).powi(2))
                        .sum::<f32>() / 30.0;
                    
                    // System is stable if both position variance is low AND kinetic energy is low
                    let ke_threshold = 0.01; // Low kinetic energy threshold  
                    let ke_variance_threshold = 0.001; // Very low KE variance threshold
                    position_variance < config.stability_variance_threshold && 
                    recent_ke < ke_threshold && 
                    ke_variance < ke_variance_threshold
                } else {
                    false
                };
                
                // Apply gradual adjustments based on current state
                let new_state_clone = new_state.clone();
                if new_state != self.current_state || new_state != AutoBalanceState::Stable {
                    // TODO: Fix borrow checker issue with apply_gradual_adjustment
                    // let adjustment_made = self.apply_gradual_adjustment(new_state, config);
                    
                    info!("[AUTO-BALANCE] State transition: {:?} -> {:?} (max_distance: {:.1}, boundary: {}/{})", 
                          self.current_state, new_state_clone, max_distance, boundary_nodes, self.node_map.len());
                }
                
                // Handle stability detection (for UI updates)  
                if is_stable && new_state_clone == AutoBalanceState::Stable {
                    // We've found a stable minima
                    self.stable_count += 1;
                    
                    // After being stable for configured frames, update UI and save
                    if self.stable_count == config.stability_frame_count {
                        info!("[AUTO-BALANCE] Stable equilibrium found at {:.1} units - updating UI sliders", max_distance);
                        
                        // Send success notification to client
                        self.send_auto_balance_notification("Auto-Balance: Stable equilibrium achieved!");
                        
                        // Update UI sliders and potentially save to settings.yaml
                        self.notify_settings_update();
                        
                        // Reset stability counter to avoid repeated updates
                        self.stable_count = 181; // Set to a value that prevents repeated notifications
                    } else if self.stable_count < 180 {
                        debug!("[AUTO-BALANCE] Stability detected for {} frames (need 180 for UI update)", self.stable_count);
                    }
                } else {
                    // Not stable yet, reset counter
                    if self.stable_count > 0 && self.stable_count < 180 {
                        debug!("[AUTO-BALANCE] Lost stability after {} frames", self.stable_count);
                    }
                    self.stable_count = 0;
                }
            }
        }
        
        // FIXED: Broadcast position updates to WebSocket clients
        // This was the critical missing piece causing the settling issue
        
        // Check if we should broadcast positions (avoid spam during stable periods)
        let now = std::time::Instant::now();
        let should_broadcast = if let Some(last_time) = self.last_broadcast_time {
            // During stable/settled periods, reduce broadcast frequency to save bandwidth
            // But still broadcast occasionally to ensure clients stay updated
            let stable_broadcast_interval = std::time::Duration::from_millis(1000); // 1 second during stable
            let active_broadcast_interval = std::time::Duration::from_millis(50); // 20Hz during active
            
            let is_stable = self.current_state == AutoBalanceState::Stable && self.stable_count > 30;
            let required_interval = if is_stable { stable_broadcast_interval } else { active_broadcast_interval };
            
            now.duration_since(last_time) >= required_interval
        } else {
            // First broadcast or no previous broadcast time
            true
        };
        
        // CRITICAL FIX: Always broadcast if we haven't sent initial positions to clients
        // This ensures new clients get data immediately regardless of settled state
        let force_broadcast = !self.initial_positions_sent;
        
        if should_broadcast || force_broadcast {
            // Create binary position data for all nodes
            let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
            
            for (node_id, node) in self.node_map.iter() {
                position_data.push((*node_id, BinaryNodeDataClient::new(
                    *node_id,
                    node.data.position(),
                    node.data.velocity(),
                )));
            }
            
            // Broadcast to all connected clients via client manager
            if !position_data.is_empty() {
                let binary_data = crate::utils::binary_protocol::encode_node_data(&position_data);
                
                // Send to client manager for broadcasting
                self.client_manager.do_send(crate::actors::messages::BroadcastNodePositions {
                    positions: binary_data,
                });
                
                // Update broadcast time and mark initial positions as sent
                self.last_broadcast_time = Some(now);
                if !self.initial_positions_sent {
                    self.initial_positions_sent = true;
                    info!("Sent initial positions to clients ({} nodes)", position_data.len());
                } else if force_broadcast {
                    info!("Force broadcast positions to new clients ({} nodes)", position_data.len());
                }
                
                if crate::utils::logging::is_debug_enabled() && !force_broadcast {
                    debug!("Broadcast positions: {} nodes, stable: {}", 
                           position_data.len(), 
                           self.current_state == AutoBalanceState::Stable);
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

        // Defer interval scheduling to ensure proper runtime context
        ctx.run_later(Duration::from_millis(100), |actor, ctx| {
            // Now start the simulation interval from within the actor's execution context
            ctx.run_interval(Duration::from_millis(16), |actor, ctx| {
                if !actor.simulation_running.load(Ordering::SeqCst) {
                    return;
                }

                actor.run_simulation_step(ctx);
            });
        });
    }

    fn run_simulation_step(&mut self, ctx: &mut Context<Self>) {
        // Check if GPU is ready before running physics
        if !self.gpu_initialized && self.gpu_compute_addr.is_some() {
            // GPU is expected but not yet initialized - skip physics
            if self.gpu_init_in_progress {
                // Silently skip - we're already waiting for initialization
                return;
            }
            warn!("Skipping physics simulation - waiting for GPU initialization");
            return;
        }
        
        // Apply smooth parameter transitions if auto-balance is enabled
        // Fixed oscillation issues with improved dampening
        if self.simulation_params.auto_balance {
            self.smooth_transition_params();
        }
        
        // Increment counters for periodic operations
        self.stress_step_counter += 1;
        self.constraint_update_counter += 1;
        
        // Execute periodic stress-majorization projection
        // DISABLED: Causing node jumps every few seconds
        // if self.stress_step_counter >= self.advanced_params.stress_step_interval_frames {
        //     self.execute_stress_majorization_step();
        //     self.stress_step_counter = 0;
        // }
        
        // Update constraints periodically based on semantic analysis
        // DISABLED: Causing node jumps every 2 seconds
        // if self.constraint_update_counter >= 120 { // Every 2 seconds at 60 FPS
        //     self.update_dynamic_constraints();
        //     self.constraint_update_counter = 0;
        // }
        
        // Initialize advanced GPU context if needed
        if self.gpu_compute_addr.is_none() && !self.gpu_init_in_progress && !self.graph_data.nodes.is_empty() {
            // Mark initialization as in progress
            self.gpu_init_in_progress = true;
            
            // Attempt to initialize advanced GPU context
            let graph_data_clone = Arc::clone(&self.graph_data);
            let self_addr = ctx.address();
            
            actix::spawn(async move {
                // Add initialization delay as suggested
                info!("Waiting 2 seconds before GPU initialization to ensure system is ready...");
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                
                // Enhanced PTX loading with proper fallback logic
                let ptx_content = match crate::utils::ptx::load_ptx().await {
                    Ok(content) => {
                        info!("PTX content loaded successfully");
                        content
                    },
                    Err(e) => {
                        error!("Failed to load PTX content: {}", e);
                        error!("PTX load error details: {:?}", e);
                        // Reset the init flag so we can retry
                        self_addr.do_send(ResetGPUInitFlag {});
                        return;
                    }
                };
                
                // For CSR format, each undirected edge becomes 2 directed edges
                let num_directed_edges = graph_data_clone.edges.len() * 2;
                info!("Creating UnifiedGPUCompute with {} nodes and {} directed edges (from {} undirected edges)", 
                      graph_data_clone.nodes.len(), num_directed_edges, graph_data_clone.edges.len());
                info!("PTX content size: {} bytes", ptx_content.len());
                
                match UnifiedGPUCompute::new(
                    graph_data_clone.nodes.len(),
                    num_directed_edges,
                    &ptx_content,
                ) {
                    Ok(context) => {
                        info!("✅ Successfully initialized advanced GPU context with {} nodes and {} edges", 
                              graph_data_clone.nodes.len(), num_directed_edges);
                        info!("GPU physics simulation is now active for knowledge graph");
                        // GPU context now managed by ForceComputeActor, no need to store locally
                    }
                    Err(e) => {
                        error!("❌ Failed to initialize advanced GPU context: {}", e);
                        error!("GPU Details: {} nodes, {} directed edges, PTX size: {} bytes", 
                               graph_data_clone.nodes.len(), num_directed_edges, ptx_content.len());
                        error!("Full error: {:?}", e);
                        
                        // Log specific error details
                        let error_str = e.to_string();
                        if error_str.contains("PTX") {
                            error!("PTX compilation or loading issue detected");
                        } else if error_str.contains("memory") {
                            error!("GPU memory allocation issue - may need to reduce graph size");
                        } else if error_str.contains("device") {
                            error!("CUDA device issue - check GPU availability");
                        }
                        
                        // Reset the flag on failure so we can retry later
                        self_addr.do_send(ResetGPUInitFlag {});
                    }
                }
            });
        }
        
        // Use advanced GPU compute only - legacy path removed
        if self.gpu_compute_addr.is_some() && self.gpu_initialized {
            self.run_advanced_gpu_step(ctx);
        } else if self.gpu_compute_addr.is_none() {
            warn!("No GPU compute context available for physics simulation");
        }
        // If GPU is expected but not initialized, we already returned above
    }
    
    fn run_advanced_gpu_step(&mut self, ctx: &mut Context<Self>) {
        // Only log physics step execution when debug is enabled
        // DISABLED: This was logging every 16ms causing log flooding and crashes
        // if crate::utils::logging::is_debug_enabled() {
        //     info!("[GPU STEP] === Starting physics simulation step ===");
        //     info!("[GPU STEP] Current physics parameters:");
        //     info!("  - repel_k: {} (node spreading force)", self.simulation_params.repel_k);
        //     info!("  - damping: {:.3} (velocity reduction, 1.0 = frozen)", self.simulation_params.damping);
        //     info!("  - dt: {:.3} (simulation speed)", self.simulation_params.dt);
        //     info!("  - spring_k: {:.3} (edge tension)", self.simulation_params.spring_k);
        //     info!("  - spring_k: {:.3} (unused - for future clustering)", self.simulation_params.spring_k);
        //     info!("  - center_gravity_k: {:.3} (center pull force)", self.simulation_params.center_gravity_k);
        //     info!("  - max_velocity: {:.3} (explosion prevention)", self.simulation_params.max_velocity);
        //     info!("  - enabled: {} (is physics on?)", self.simulation_params.enabled);
        //     info!("  - auto_balance: {} (auto-tuning enabled?)", self.simulation_params.auto_balance);
        // }
        
        if !self.simulation_params.enabled {
            if crate::utils::logging::is_debug_enabled() {
                info!("[GPU STEP] Physics disabled - skipping simulation");
            }
            return;
        }
        
        // Check if physics is paused due to equilibrium
        if self.simulation_params.is_physics_paused {
            if crate::utils::logging::is_debug_enabled() {
                trace!("[GPU STEP] Physics paused (equilibrium reached) - skipping simulation");
            }
            return;
        }
        
        // Physics parameters loaded
        
        // Delegate GPU computation to ForceComputeActor
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            // Send ComputeForces message to ForceComputeActor
            let gpu_addr_clone = gpu_addr.clone();
            let ctx_addr = Context::address(ctx).recipient();
            
            actix::spawn(async move {
                match gpu_addr_clone.send(crate::actors::messages::ComputeForces).await {
                    Ok(Ok(())) => {
                        // GPU computation successful, now get the updated node data
                        match gpu_addr_clone.send(crate::actors::messages::GetNodeData).await {
                            Ok(Ok(node_data)) => {
                                // Send positions back to GraphServiceActor for processing
                                let update_msg = crate::actors::messages::UpdateNodePositions {
                                    positions: node_data.iter().enumerate()
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
            });
            
            // Return early - the async block will handle position updates
            return;
        }
        
        // No GPU compute actor available - skip GPU computation
        trace!("No GPU compute actor available for physics simulation");
    }
    // Legacy GPU step removed - only advanced GPU compute is supported
    // All physics computation uses the unified GPU kernel

    /// Update advanced physics parameters
    pub fn update_advanced_physics_params(&mut self, params: AdvancedParams) -> Result<(), String> {
        self.advanced_params = params.clone();
        self.stress_solver = StressMajorizationSolver::from_advanced_params(&params);
        
        // Update GPU parameters via ForceComputeActor
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            let update_msg = crate::actors::messages::UpdateSimulationParams {
                params: self.simulation_params.clone(),
            };
            gpu_addr.do_send(update_msg);
        }
        
        info!("Updated advanced physics parameters via public API");
        Ok(())
    }
    
    /// Get current constraint set (read-only access)
    pub fn get_constraint_set(&self) -> &ConstraintSet {
        &self.constraint_set
    }
    
    /// Manually trigger stress majorization optimization
    pub fn trigger_stress_optimization(&mut self) -> Result<(), String> {
        self.execute_stress_majorization_step();
        info!("Manually triggered stress majorization via public API");
        Ok(())
    }
    
    /// Check if advanced GPU context is available
    pub fn has_advanced_gpu(&self) -> bool {
        self.gpu_compute_addr.is_some()
    }
    
    /// Get semantic analysis status
    pub fn get_semantic_analysis_status(&self) -> (usize, Option<std::time::Duration>) {
        let feature_count = self.semantic_features_cache.len();
        let age = self.last_semantic_analysis.map(|t| t.elapsed());
        (feature_count, age)
    }

    /// Calculate Communication Intensity between two agents based on:
    /// - Agent types and their collaboration patterns
    /// - Activity levels (active tasks)
    /// - Performance metrics (success rates)
    fn calculate_communication_intensity(
        &self,
        source_type: &crate::types::claude_flow::AgentType,
        target_type: &crate::types::claude_flow::AgentType,
        source_active_tasks: u32,
        target_active_tasks: u32,
        source_success_rate: f32,
        target_success_rate: f32,
    ) -> f32 {
        // Base communication intensity based on agent type relationships
        let base_intensity = match (source_type, target_type) {
            // Coordinator has high communication with all agent types
            (crate::types::claude_flow::AgentType::Coordinator, _) |
            (_, crate::types::claude_flow::AgentType::Coordinator) => 0.9,

            // High collaboration pairs
            (crate::types::claude_flow::AgentType::Coder, crate::types::claude_flow::AgentType::Tester) |
            (crate::types::claude_flow::AgentType::Tester, crate::types::claude_flow::AgentType::Coder) => 0.8,

            (crate::types::claude_flow::AgentType::Researcher, crate::types::claude_flow::AgentType::Analyst) |
            (crate::types::claude_flow::AgentType::Analyst, crate::types::claude_flow::AgentType::Researcher) => 0.7,

            (crate::types::claude_flow::AgentType::Architect, crate::types::claude_flow::AgentType::Coder) |
            (crate::types::claude_flow::AgentType::Coder, crate::types::claude_flow::AgentType::Architect) => 0.7,

            // Medium collaboration pairs
            (crate::types::claude_flow::AgentType::Architect, crate::types::claude_flow::AgentType::Analyst) |
            (crate::types::claude_flow::AgentType::Analyst, crate::types::claude_flow::AgentType::Architect) => 0.6,

            (crate::types::claude_flow::AgentType::Reviewer, crate::types::claude_flow::AgentType::Coder) |
            (crate::types::claude_flow::AgentType::Coder, crate::types::claude_flow::AgentType::Reviewer) => 0.6,

            (crate::types::claude_flow::AgentType::Optimizer, crate::types::claude_flow::AgentType::Analyst) |
            (crate::types::claude_flow::AgentType::Analyst, crate::types::claude_flow::AgentType::Optimizer) => 0.6,
            
            // Default moderate communication for other pairs
            _ => 0.4,
        };
        
        // Activity factor: agents with more active tasks communicate more
        let max_tasks = std::cmp::max(source_active_tasks, target_active_tasks);
        let activity_factor = if max_tasks > 0 {
            1.0 + (max_tasks as f32 * 0.1).min(0.5) // Cap at 50% boost
        } else {
            0.7 // Reduce for inactive agents
        };
        
        // Performance factor: higher success rates lead to more collaboration
        let avg_success_rate = (source_success_rate + target_success_rate) / 200.0; // Convert to 0-1 range
        let performance_factor = 0.5 + avg_success_rate * 0.5; // Range: 0.5 to 1.0
        
        // Calculate final intensity with all factors
        let final_intensity = base_intensity * activity_factor * performance_factor;
        
        // Clamp to reasonable range
        final_intensity.min(1.0).max(0.0)
    }
    
    /// Check if the physics simulation has reached equilibrium and handle auto-pause
    fn check_and_handle_equilibrium(&mut self, total_kinetic_energy: f32, node_count: usize) {
        if !self.simulation_params.auto_pause_config.enabled || node_count == 0 {
            return;
        }
        
        // Calculate average velocity from kinetic energy
        // KE = 0.5 * m * v^2, assuming m = 1, so v = sqrt(2 * KE)
        let avg_kinetic_energy = total_kinetic_energy / node_count as f32;
        let avg_velocity = (2.0 * avg_kinetic_energy).sqrt();
        
        let config = &self.simulation_params.auto_pause_config;
        
        // Check if we're in equilibrium state
        let is_in_equilibrium = avg_velocity < config.equilibrium_velocity_threshold 
            && avg_kinetic_energy < config.equilibrium_energy_threshold;
        
        if is_in_equilibrium {
            // Increment stability counter
            self.simulation_params.equilibrium_stability_counter += 1;
            
            // Check if we've been stable for enough frames
            if self.simulation_params.equilibrium_stability_counter >= config.equilibrium_check_frames {
                if !self.simulation_params.is_physics_paused && config.pause_on_equilibrium {
                    // Pause physics (not disable it)
                    self.simulation_params.is_physics_paused = true;
                    
                    if crate::utils::logging::is_debug_enabled() {
                        info!("[AUTO-PAUSE] Physics paused - equilibrium reached (avg_velocity: {:.4}, avg_energy: {:.4})", 
                              avg_velocity, avg_kinetic_energy);
                    }
                    
                    // Broadcast pause notification to clients
                    let pause_msg = PhysicsPauseMessage {
                        pause: true,
                        reason: format!("Equilibrium reached (vel: {:.4}, energy: {:.4})", avg_velocity, avg_kinetic_energy),
                    };
                    
                    // Send to client manager for broadcast
                    self.client_manager.do_send(BroadcastMessage {
                        message: format!("{{\"type\": \"physics_paused\", \"reason\": \"{}\"}}", pause_msg.reason),
                    });
                }
            }
        } else {
            // CRITICAL FIX: Only reset stability counter if physics is not already paused
            // This prevents the graph from un-settling after it has reached equilibrium
            if !self.simulation_params.is_physics_paused {
                // Reset stability counter if not in equilibrium and physics is still running
                self.simulation_params.equilibrium_stability_counter = 0;
            }
            // If physics is paused and we're no longer in equilibrium, keep it paused
            // This prevents auto-resume which could cause the jumping behavior
        }
        
        if crate::utils::logging::is_debug_enabled() {
            trace!("[AUTO-PAUSE] Equilibrium check: velocity={:.4}, energy={:.4}, stable_frames={}/{}, paused={}", 
                   avg_velocity, avg_kinetic_energy, 
                   self.simulation_params.equilibrium_stability_counter, 
                   config.equilibrium_check_frames,
                   self.simulation_params.is_physics_paused);
        }
    }
    
    /// Resume physics if paused, usually triggered by user interaction
    fn resume_physics_if_paused(&mut self, reason: String) {
        if self.simulation_params.is_physics_paused {
            self.simulation_params.is_physics_paused = false;
            self.simulation_params.equilibrium_stability_counter = 0;
            
            if crate::utils::logging::is_debug_enabled() {
                info!("[AUTO-PAUSE] Physics resumed: {}", reason);
            }
            
            // Broadcast resume notification to clients
            let resume_msg = PhysicsPauseMessage {
                pause: false,
                reason: reason.clone(),
            };
            
            self.client_manager.do_send(BroadcastMessage {
                message: format!("{{\"type\": \"physics_resumed\", \"reason\": \"{}\"}}", reason),
            });
        }
    }
    
    /// Add new nodes from metadata without rebuilding the entire graph
    pub fn add_nodes_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        debug!("Adding {} new nodes incrementally", metadata.len());
        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        for (filename_with_ext, file_meta_data) in &metadata {
            let metadata_id_val = filename_with_ext.trim_end_matches(".md").to_string();
            
            // Check if node already exists
            if self.node_map.values().any(|n| n.metadata_id == metadata_id_val) {
                debug!("Node {} already exists, skipping", metadata_id_val);
                continue;
            }
            
            let node_id_val = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            let mut node = Node::new_with_id(metadata_id_val.clone(), Some(node_id_val));
            node.label = file_meta_data.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(file_meta_data.file_size as u64);
            // Note: flags field not available in BinaryNodeDataClient
            // Using node_id low bit as flag if needed: node.data.node_id |= 1;

            // Enhanced metadata
            node.metadata.insert("fileName".to_string(), file_meta_data.file_name.clone());
            node.metadata.insert("fileSize".to_string(), file_meta_data.file_size.to_string());
            node.metadata.insert("nodeSize".to_string(), file_meta_data.node_size.to_string());
            node.metadata.insert("hyperlinkCount".to_string(), file_meta_data.hyperlink_count.to_string());
            node.metadata.insert("sha1".to_string(), file_meta_data.sha1.clone());
            node.metadata.insert("lastModified".to_string(), file_meta_data.last_modified.to_rfc3339());
            node.metadata.insert("metadataId".to_string(), metadata_id_val.clone());
            
            // Extract semantic features
            let features = self.semantic_analyzer.analyze_metadata(file_meta_data);
            self.semantic_features_cache.insert(metadata_id_val, features);
            
            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
            graph_data_mut.nodes.push(node);
        }
        
        info!("Added {} new nodes incrementally", metadata.len());
        Ok(())
    }
    
    /// Update a single node from metadata
    pub fn update_node_from_metadata(&mut self, metadata_id: String, metadata: FileMetadata) -> Result<(), String> {
        debug!("Updating node {} incrementally", metadata_id);
        
        // Find and update node in node_map
        let mut node_found = false;
        if let Some(node) = Arc::make_mut(&mut self.node_map).values_mut().find(|n| n.metadata_id == metadata_id) {
            node.label = metadata.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(metadata.file_size as u64);
            
            // Update metadata
            node.metadata.insert("fileName".to_string(), metadata.file_name.clone());
            node.metadata.insert("fileSize".to_string(), metadata.file_size.to_string());
            node.metadata.insert("nodeSize".to_string(), metadata.node_size.to_string());
            node.metadata.insert("hyperlinkCount".to_string(), metadata.hyperlink_count.to_string());
            node.metadata.insert("sha1".to_string(), metadata.sha1.clone());
            node.metadata.insert("lastModified".to_string(), metadata.last_modified.to_rfc3339());
            
            node_found = true;
        }
        
        if !node_found {
            return Err(format!("Node {} not found for update", metadata_id));
        }
        
        // Update corresponding node in graph_data
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        if let Some(node) = graph_data_mut.nodes.iter_mut().find(|n| n.metadata_id == metadata_id) {
            node.label = metadata.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(metadata.file_size as u64);
        }
        
        // Update semantic features
        let features = self.semantic_analyzer.analyze_metadata(&metadata);
        self.semantic_features_cache.insert(metadata_id.clone(), features);
        
        info!("Updated node {} incrementally", metadata_id);
        Ok(())
    }
    
    /// Remove a node by metadata ID
    pub fn remove_node_by_metadata(&mut self, metadata_id: String) -> Result<(), String> {
        debug!("Removing node {} incrementally", metadata_id);
        
        // Find node to remove
        let node_id = self.node_map.values()
            .find(|n| n.metadata_id == metadata_id)
            .map(|n| n.id);
        
        if let Some(node_id) = node_id {
            // Remove from node_map
            Arc::make_mut(&mut self.node_map).remove(&node_id);
            
            // Remove from graph_data
            let graph_data_mut = Arc::make_mut(&mut self.graph_data);
            graph_data_mut.nodes.retain(|n| n.id != node_id);
            graph_data_mut.edges.retain(|e| e.source != node_id && e.target != node_id);
            
            // Remove from semantic features cache
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
        // Optional GPU smoke test on start (set GPU_SMOKE_ON_START=1)
        if std::env::var("GPU_SMOKE_ON_START").ok().as_deref() == Some("1") {
            let report = crate::utils::gpu_diagnostics::ptx_module_smoke_test();
            info!("{}", report);
        }
        self.start_simulation_loop(ctx);
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        self.simulation_running.store(false, Ordering::SeqCst);
        self.shutdown_complete.store(true, Ordering::SeqCst);
        info!("GraphServiceActor stopped");
    }
}

// Message handlers
impl Handler<GetGraphData> for GraphServiceActor {
    type Result = Result<std::sync::Arc<GraphData>, String>;
 
    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("DEBUG_VERIFICATION: GraphServiceActor handling GetGraphData with Arc reference (NO CLONE!).");
        Ok(std::sync::Arc::clone(&self.graph_data)) // Returns Arc reference, no data cloning!
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

    fn handle(&mut self, msg: crate::actors::messages::ForcePositionBroadcast, _ctx: &mut Self::Context) -> Self::Result {
        info!("Force broadcasting positions: {}", msg.reason);
        
        // Create binary position data for all nodes regardless of settled state
        let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
        
        for (node_id, node) in self.node_map.iter() {
            position_data.push((*node_id, BinaryNodeDataClient::new(
                *node_id,
                node.data.position(),
                node.data.velocity(),
            )));
        }
        
        // Broadcast to all connected clients via client manager
        if !position_data.is_empty() {
            let binary_data = crate::utils::binary_protocol::encode_node_data(&position_data);
            
            // Send to client manager for broadcasting
            self.client_manager.do_send(crate::actors::messages::BroadcastNodePositions {
                positions: binary_data,
            });
            
            // Update broadcast time and ensure initial positions flag is set
            self.last_broadcast_time = Some(std::time::Instant::now());
            self.initial_positions_sent = true;
            
            info!("Force broadcast complete: {} nodes sent (reason: {})", position_data.len(), msg.reason);
        } else {
            warn!("Force broadcast requested but no position data available (reason: {})", msg.reason);
        }
        
        Ok(())
    }
}

impl Handler<InitialClientSync> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: InitialClientSync, _ctx: &mut Self::Context) -> Self::Result {
        info!("Initial client sync requested by {} from {}", msg.client_identifier, msg.trigger_source);
        
        // Force broadcast current positions to ensure new client gets synchronized
        // This provides immediate feedback after REST endpoint returns graph structure
        let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
        
        for (node_id, node) in self.node_map.iter() {
            position_data.push((*node_id, BinaryNodeDataClient::new(
                *node_id,
                node.data.position(),
                node.data.velocity(),
            )));
        }
        
        if !position_data.is_empty() {
            let binary_data = crate::utils::binary_protocol::encode_node_data(&position_data);
            
            // Send to client manager for broadcasting to all clients
            self.client_manager.do_send(crate::actors::messages::BroadcastNodePositions {
                positions: binary_data,
            });
            
            // Update tracking flags
            self.last_broadcast_time = Some(std::time::Instant::now());
            self.initial_positions_sent = true;
            
            info!("Initial sync broadcast complete: {} nodes sent for client {}", position_data.len(), msg.client_identifier);
        } else {
            warn!("Initial sync requested but no nodes available for client {}", msg.client_identifier);
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
        Ok(Arc::clone(&self.node_map)) // Return Arc reference, no cloning of HashMap data!
    }
}

impl Handler<BuildGraphFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, ctx: &mut Self::Context) -> Self::Result {
        info!("BuildGraphFromMetadata handler called with {} metadata entries", msg.metadata.len());
        // Build the graph from metadata
        let result = self.build_from_metadata(msg.metadata);
        
        // If successful, handle GPU initialization
        if result.is_ok() {
            // Send data to appropriate GPU context
            if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
                info!("Graph data prepared for GPU physics");
                
                // First initialize GPU with GraphServiceActor address for notification
                gpu_compute_addr.do_send(InitializeGPU {
                    graph: Arc::clone(&self.graph_data),
                    graph_service_addr: Some(ctx.address()),
                    gpu_manager_addr: None,  // Will be filled by GPUManagerActor
                });
                info!("Sent GPU initialization request to GPU compute actor");
                
                // Then update the graph data  
                gpu_compute_addr.do_send(UpdateGPUGraphData { graph: Arc::clone(&self.graph_data) });
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
        // Update node in the node map
        if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&msg.node_id) {
            // Update position and velocity using direct field access
            let new_position = glam_to_vec3data(msg.position);
            let new_velocity = glam_to_vec3data(msg.velocity);

            node.data.x = new_position.x;
            node.data.y = new_position.y;
            node.data.z = new_position.z;
            node.data.vx = new_velocity.x;
            node.data.vy = new_velocity.y;
            node.data.vz = new_velocity.z;

            // Note: mass and flags fields not available in BinaryNodeDataClient
        } else {
            debug!("Received update for unknown node ID: {}", msg.node_id);
            return Err(format!("Unknown node ID: {}", msg.node_id));
        }
        
        // Update corresponding node in graph
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        for node_in_graph_data in &mut graph_data_mut.nodes { // Iterate over mutable graph_data
            if node_in_graph_data.id == msg.node_id {
                // Update position components directly
                let pos = glam_to_vec3data(msg.position);
                node_in_graph_data.data.x = pos.x;
                node_in_graph_data.data.y = pos.y;
                node_in_graph_data.data.z = pos.z;

                // Update velocity components directly
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
        // Just run one simulation step
        self.run_simulation_step(ctx);
        Ok(())
    }
}

impl Handler<GetAutoBalanceNotifications> for GraphServiceActor {
    type Result = Result<Vec<AutoBalanceNotification>, String>;

    fn handle(&mut self, msg: GetAutoBalanceNotifications, _ctx: &mut Self::Context) -> Self::Result {
        match self.auto_balance_notifications.lock() { Ok(notifications) => {
            let filtered_notifications = if let Some(since) = msg.since_timestamp {
                notifications.iter()
                    .filter(|n| n.timestamp > since)
                    .cloned()
                    .collect()
            } else {
                notifications.clone()
            };
            
            Ok(filtered_notifications)
        } _ => {
            Err("Failed to access notifications".to_string())
        }}
    }
}

impl Handler<UpdateGraphData> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, ctx: &mut Self::Context) -> Self::Result {
        info!("Updating graph data with {} nodes, {} edges",
              msg.graph_data.nodes.len(), msg.graph_data.edges.len());
        
        // Update graph data with the provided Arc
        self.graph_data = msg.graph_data;
        
        // Rebuild node map
        Arc::make_mut(&mut self.node_map).clear();
        for node in &self.graph_data.nodes { // Dereferences Arc for iteration
            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
        }
        
        // Generate initial semantic constraints for the new graph data
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone);
        
        // Send data to GPU compute if available
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            info!("Sending loaded graph data to GPU physics");
            
            // First initialize GPU with GraphServiceActor address for notification
            gpu_compute_addr.do_send(InitializeGPU {
                graph: Arc::clone(&self.graph_data),
                graph_service_addr: Some(ctx.address()),
                gpu_manager_addr: None,  // Will be filled by GPUManagerActor
            });
            info!("Sent GPU initialization request to GPU compute actor");
            
            // Then update the graph data  
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: Arc::clone(&self.graph_data) });
            info!("Sent loaded graph data to GPU compute actor");
        } else {
            warn!("GPU compute actor not available, physics simulation won't be initialized");
        }
        
        info!("Graph data updated successfully with constraint generation and GPU initialization");
        Ok(())
    }
}

impl Handler<UpdateBotsGraph> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) -> Self::Result {
        // This logic converts `Agent` objects into `Node` and `Edge` objects
        let mut nodes = vec![];
        let mut edges = vec![];
        
        // Use a high ID range (starting at 10000) to avoid conflicts with main graph
        let bot_id_offset = 10000;
        
        // Preserve existing agent positions to prevent re-randomization on every update
        let mut existing_positions: HashMap<String, (Vec3Data, Vec3Data)> = HashMap::new();
        for node in &self.bots_graph_data.nodes {
            existing_positions.insert(node.metadata_id.clone(), (node.data.position(), node.data.velocity()));
        }

        // Create nodes for each agent
        for (i, agent) in msg.agents.iter().enumerate() {
            let node_id = bot_id_offset + i as u32;
            
            // Create a node for each agent
            let mut node = Node::new_with_id(agent.id.clone(), Some(node_id));
            
            if let Some((saved_position, saved_velocity)) = existing_positions.get(&agent.id) {
                // Restore existing position
                node.data.x = saved_position.x;
                node.data.y = saved_position.y;
                node.data.z = saved_position.z;
                node.data.vx = saved_velocity.x;
                node.data.vy = saved_velocity.y;
                node.data.vz = saved_velocity.z;
            } else {
                // Generate random initial positions for new agents
                let physics = crate::config::dev_config::physics();
                use rand::{Rng, SeedableRng};
                use rand::rngs::{StdRng, OsRng};
                let mut rng = StdRng::from_seed(OsRng.gen());

                let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
                let phi = rng.gen::<f32>() * std::f32::consts::PI;
                let radius = physics.initial_radius_min + rng.gen::<f32>() * physics.initial_radius_range;

                node.data.x = radius * phi.sin() * theta.cos();
                node.data.y = radius * phi.sin() * theta.sin();
                node.data.z = radius * phi.cos();

                node.data.vx = rng.gen_range(-0.5..0.5);
                node.data.vy = rng.gen_range(-0.5..0.5);
                node.data.vz = rng.gen_range(-0.5..0.5);
            }
            
            // Set node properties based on agent type
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
            node.size = Some(20.0 + (agent.workload * 25.0)); // Size based on workload
            
            // Add metadata including agent flag for GPU physics
            node.metadata.insert("agent_type".to_string(), agent.agent_type.clone());
            node.metadata.insert("status".to_string(), agent.status.clone());
            node.metadata.insert("cpu_usage".to_string(), agent.cpu_usage.to_string());
            node.metadata.insert("memory_usage".to_string(), agent.memory_usage.to_string());
            node.metadata.insert("health".to_string(), agent.health.to_string());
            node.metadata.insert("is_agent".to_string(), "true".to_string()); // Agent node flag
            
            nodes.push(node);
        }
        
        // Create edges based on agent types and status
        for (i, source_agent) in msg.agents.iter().enumerate() {
            for (j, target_agent) in msg.agents.iter().enumerate() {
                if i != j {
                    let source_node_id = bot_id_offset + i as u32;
                    let target_node_id = bot_id_offset + j as u32;
                    
                    // Simple communication intensity based on agent types
                    let communication_intensity = if source_agent.agent_type == "coordinator" || target_agent.agent_type == "coordinator" {
                        0.8 // Coordinators have strong connections
                    } else if source_agent.status == "active" && target_agent.status == "active" {
                        0.5 // Active agents communicate
                    } else {
                        0.2 // Default weak connection
                    };
                    
                    // Only create edges for significant communication
                    if communication_intensity > 0.1 {
                        let mut edge = Edge::new(source_node_id, target_node_id, communication_intensity);
                        // Correctly handle Option<HashMap>
                        let metadata = edge.metadata.get_or_insert_with(HashMap::new);
                        metadata.insert(
                            "communication_type".to_string(),
                            "agent_collaboration".to_string(),
                        );
                        metadata.insert("intensity".to_string(), communication_intensity.to_string());
                        edges.push(edge);
                    }
                }
            }
        }
        
        // Update the bots graph data  
        let bots_graph_data_mut = Arc::make_mut(&mut self.bots_graph_data);
        bots_graph_data_mut.nodes = nodes;
        bots_graph_data_mut.edges = edges;
        
        info!("Updated bots graph with {} agents and {} edges - sending optimized position updates to WebSocket clients",
             msg.agents.len(), self.bots_graph_data.edges.len());
        
        // Send BINARY WebSocket updates for agents (position, velocity, SSSP, control bits)
        // This is the HIGH-SPEED data path - metadata goes via REST
        let mut position_data: Vec<(u32, BinaryNodeData)> = Vec::new();
        let mut agent_node_ids: Vec<u32> = Vec::new();

        // Collect agent positions for binary protocol (34 bytes per node)
        for node in &self.bots_graph_data.nodes {
            // Create binary node data with position, velocity, SSSP
            let mut binary_node = BinaryNodeDataClient::new(
                node.id,
                glam_to_vec3data(Vec3::new(node.data.x, node.data.y, node.data.z)),
                glam_to_vec3data(Vec3::new(node.data.vx, node.data.vy, node.data.vz)),
            );

            position_data.push((node.id, binary_node));

            // Track agent IDs for control bit encoding
            agent_node_ids.push(node.id);
        }

        let binary_size = if !position_data.is_empty() {
            // Use binary protocol with control bits for agent identification
            // This encodes: node_id (2 bytes with flags), position (12 bytes),
            // velocity (12 bytes), SSSP distance (4 bytes), SSSP parent (4 bytes)
            let binary_data = crate::utils::binary_protocol::encode_node_data_with_flags(
                &position_data,
                &agent_node_ids  // These will have the agent control bit set
            );

            // Send via BINARY WebSocket protocol
            self.client_manager.do_send(crate::actors::messages::BroadcastNodePositions {
                positions: binary_data.clone(),
            });

            // Log metrics
            let binary_size = binary_data.len();
            let nodes_sent = position_data.len();
            info!("Sent BINARY agent update: {} nodes, {} bytes total, {} bytes/node",
                  nodes_sent, binary_size, binary_size / nodes_sent);

            binary_size
        } else {
            0
        };

        // DO NOT send graph structure over WebSocket - this belongs in REST!
        // WebSocket is ONLY for high-speed variable data:
        // - Position/Velocity (sent above via binary)
        // - SSSP data (included in binary)
        // - Voice/Audio streams
        // - Control bits for node types
        //
        // Everything else (edges, metadata, agent details) goes via REST:
        // - GET /api/bots/data - Full agent list with metadata
        // - GET /api/bots/status - Agent telemetry
        // - POST /api/bots/submit-task - Submit tasks
        // - GET /api/bots/task-status/{id} - Task status

        debug!("Agent binary positions sent. Graph structure available via REST /api/bots/data");

        info!("Sent optimized graph update: {} nodes, {} edges ({} bytes)",
              position_data.len(),
              self.bots_graph_data.edges.len(),
              binary_size);

        // Send the updated bots graph data to GPU for force-directed positioning
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            gpu_compute_addr.do_send(UpdateGPUGraphData {
                graph: Arc::clone(&self.bots_graph_data)
            });
            info!("Sent updated bots graph data to GPU compute actor");
        } else {
            warn!("No GPU compute address available - bots will remain at initial positions");
        }

        // Remove CPU physics calculations for agent graph - delegate to GPU
        // The GPU will use the edge weights (communication intensity) for spring forces
    }
}

impl Handler<GetBotsGraphData> for GraphServiceActor {
    type Result = Result<std::sync::Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetBotsGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(Arc::clone(&self.bots_graph_data)) // Return Arc reference, no cloning of GraphData!
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
            let gpu_manager_clone = gpu_manager.clone();
            let self_addr = ctx.address();
            
            // Store the GPUManagerActor address directly
            self.gpu_compute_addr = Some(gpu_manager.clone());
            info!("[GraphServiceActor] Stored GPUManagerActor address for GPU coordination");

            // Initialize GPU with current graph data if we have any
            if !self.graph_data.nodes.is_empty() {
                info!("Sending initial graph data to GPU via GPUManager");
                gpu_manager.do_send(InitializeGPU {
                    graph: Arc::clone(&self.graph_data),
                    graph_service_addr: Some(ctx.address()),
                    gpu_manager_addr: None,  // Will be filled by GPUManagerActor
                });

                // Also send the graph data update
                gpu_manager.do_send(UpdateGPUGraphData {
                    graph: Arc::clone(&self.graph_data),
                });

                self.gpu_initialized = true;
                info!("GPU initialization messages sent via GPUManager");
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
        info!("GPU has been successfully initialized - enabling physics simulation");
        self.gpu_initialized = true;
        self.gpu_init_in_progress = false;
        
        // Log current state
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
            info!("  - max_velocity: {:.3} (was)", self.simulation_params.max_velocity);
            info!("  - enabled: {} (was)", self.simulation_params.enabled);
            info!("  - auto_balance: {} (was)", self.simulation_params.auto_balance);
            
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
        
        // Check if auto-balance is being turned on for the first time
        let auto_balance_just_enabled = !self.simulation_params.auto_balance && msg.params.auto_balance;
        
        self.simulation_params = msg.params.clone();
        // CRITICAL: Also update target_params so smooth transitions work correctly
        self.target_params = msg.params.clone();
        
        // If auto-balance was just enabled, reset tracking state for fresh start
        if auto_balance_just_enabled {
            info!("[AUTO-BALANCE] Auto-balance enabled - starting adaptive tuning from current values");
            
            // Reset history and stability counter for fresh start
            self.auto_balance_history.clear();
            self.stable_count = 0;
            
            info!("[AUTO-BALANCE] Will adaptively tune from current settings - repel_k: {:.3}, damping: {:.3}", 
                  self.simulation_params.repel_k, self.simulation_params.damping);
        }
        
        // Update GPU compute actor if available
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
        
        // Collect knowledge graph positions if requested
        if msg.include_knowledge_graph {
            for node in &self.graph_data.nodes {
                // Skip agent nodes in main graph
                if node.metadata.get("is_agent").map_or(false, |v| v == "true") {
                    continue;
                }

                let node_data = BinaryNodeDataClient::new(
                    node.id,
                    node.data.position(),
                    node.data.velocity(),
                );
                
                knowledge_nodes.push((node.id, node_data));
            }
        }
        
        // Collect agent graph positions if requested
        if msg.include_agent_graph {
            for node in &self.bots_graph_data.nodes {
                let node_data = BinaryNodeDataClient::new(
                    node.id,
                    node.data.position(),
                    node.data.velocity(),
                );
                
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
        
        // Update stress solver configuration
        self.stress_solver = crate::physics::stress_majorization::StressMajorizationSolver::from_advanced_params(&msg.params);
        
        // Update advanced GPU context if available
        // Update GPU parameters via ForceComputeActor
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
    
    fn handle(&mut self, msg: UpdateConstraints, _ctx: &mut Self::Context) -> Self::Result {
        self.handle_constraint_update(msg.constraint_data)
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
    
    fn handle(&mut self, _msg: TriggerStressMajorization, _ctx: &mut Self::Context) -> Self::Result {
        self.execute_stress_majorization_step();
        info!("Manually triggered stress majorization optimization");
        Ok(())
    }
}

impl Handler<RegenerateSemanticConstraints> for GraphServiceActor {
    type Result = Result<(), String>;
    
    fn handle(&mut self, _msg: RegenerateSemanticConstraints, _ctx: &mut Self::Context) -> Self::Result {
        // Clear existing semantic constraints
        self.constraint_set.set_group_active("semantic_dynamic", false);
        self.constraint_set.set_group_active("domain_clustering", false);
        self.constraint_set.set_group_active("clustering_dynamic", false);
        
        // Regenerate initial semantic constraints  
        let graph_data_clone = Arc::clone(&self.graph_data);
        self.generate_initial_semantic_constraints(&graph_data_clone);
        
        // Regenerate dynamic constraints if semantic analysis is available
        if self.last_semantic_analysis.is_some() {
            self.update_dynamic_constraints();
        }
        
        info!("Regenerated semantic constraints: {} total constraints", 
              self.constraint_set.constraints.len());
        Ok(())
    }
}

impl Handler<SetAdvancedGPUContext> for GraphServiceActor {
    type Result = ();
    
    fn handle(&mut self, _msg: SetAdvancedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        // The GPU context initialization signal has been received
        // The actual context is managed separately in the actor that created it
        self.gpu_init_in_progress = false; // Reset the flag
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
    type Result = Result<std::collections::HashMap<u32, Option<f32>>, String>;

    fn handle(&mut self, msg: ComputeShortestPaths, _ctx: &mut Self::Context) -> Self::Result {
        // Implement Dijkstra's algorithm for SSSP computation
        let source_id = msg.source_node_id;

        // Check if source node exists
        if !self.node_map.contains_key(&source_id) {
            return Err(format!("Source node {} not found", source_id));
        }

        // Initialize distances
        let mut distances = std::collections::HashMap::new();
        let mut visited = std::collections::HashSet::new();
        let mut priority_queue = std::collections::BinaryHeap::new();

        // Set all distances to infinity initially
        for node_id in self.node_map.keys() {
            distances.insert(*node_id, None);
        }

        // Source node has distance 0
        distances.insert(source_id, Some(0.0));
        priority_queue.push(std::cmp::Reverse((ordered_float::OrderedFloat(0.0_f32), source_id)));

        // Build adjacency list from edges
        let mut adjacency: std::collections::HashMap<u32, Vec<(u32, f32)>> = std::collections::HashMap::new();
        for edge in &self.graph_data.edges {
            // For undirected graph, add both directions
            adjacency.entry(edge.source).or_insert_with(Vec::new).push((edge.target, edge.weight));
            adjacency.entry(edge.target).or_insert_with(Vec::new).push((edge.source, edge.weight));
        }

        // Dijkstra's algorithm
        while let Some(std::cmp::Reverse((current_dist, current_node))) = priority_queue.pop() {
            let current_dist = current_dist.into_inner();
            if visited.contains(&current_node) {
                continue;
            }

            visited.insert(current_node);

            // Check neighbors
            if let Some(neighbors) = adjacency.get(&current_node) {
                for &(neighbor_id, edge_weight) in neighbors {
                    if visited.contains(&neighbor_id) {
                        continue;
                    }

                    let new_dist = current_dist + edge_weight;
                    let current_neighbor_dist = distances.get(&neighbor_id).and_then(|d| *d);

                    if current_neighbor_dist.is_none() || new_dist < current_neighbor_dist.unwrap() {
                        distances.insert(neighbor_id, Some(new_dist));
                        priority_queue.push(std::cmp::Reverse((ordered_float::OrderedFloat(new_dist), neighbor_id)));
                    }
                }
            }
        }

        info!("SSSP computed from node {}: {} reachable nodes out of {}",
              source_id,
              distances.values().filter(|d| d.is_some()).count(),
              distances.len());

        Ok(distances)
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
        // Resume physics on interaction if configured to do so
        if self.simulation_params.auto_pause_config.resume_on_interaction {
            let reason = match msg.interaction_type {
                NodeInteractionType::Dragged => format!("Node {} dragged", msg.node_id),
                NodeInteractionType::Selected => format!("Node {} selected", msg.node_id),
                NodeInteractionType::Released => format!("Node {} released", msg.node_id),
            };
            self.resume_physics_if_paused(reason);
        }

        // Update node position if provided (for dragging)
        if let (Some(position), NodeInteractionType::Dragged) = (msg.position, &msg.interaction_type) {
            // Update node position in node_map
            if let Some(node) = Arc::make_mut(&mut self.node_map).get_mut(&msg.node_id) {
                // FIXME: Type mismatch - commented for compilation
                // node.data.position = crate::utils::socket_flow_messages::glam_to_vec3data(position);
                // Reset velocity to avoid physics conflicts during drag
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::metadata::{MetadataStore, FileMetadata};
    use crate::types::vec3::Vec3Data;
    use std::collections::HashMap;
    use chrono::Utc;

    /// Test that node positions are preserved across multiple BuildGraphFromMetadata calls
    /// This addresses the issue where positions were reset every time a client connected
    #[test]
    fn test_position_preservation_across_rebuilds() {
        // Create a test GraphServiceActor
        let mut actor = GraphServiceActor::new();
        
        // Create initial metadata
        let mut metadata = MetadataStore::new();
        metadata.insert("file1.md".to_string(), FileMetadata {
            file_name: "file1.md".to_string(),
            file_size: 1000,
            node_size: 100,
            hyperlink_count: 5,
            sha1: "abc123".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "".to_string(),
            last_perplexity_process: None,
            topic_counts: HashMap::new(),
        });
        metadata.insert("file2.md".to_string(), FileMetadata {
            file_name: "file2.md".to_string(),
            file_size: 2000,
            node_size: 200,
            hyperlink_count: 10,
            sha1: "def456".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "".to_string(),
            last_perplexity_process: None,
            topic_counts: HashMap::new(),
        });

        // First build - nodes will get initial positions
        assert!(actor.build_from_metadata(metadata.clone()).is_ok());
        
        // Store the positions after first build
        let initial_positions: HashMap<String, (Vec3Data, Vec3Data)> = actor.node_map
            .values()
            .map(|node| (node.metadata_id.clone(), (node.data.position(), node.data.velocity())))
            .collect();
            
        assert_eq!(initial_positions.len(), 2, "Should have 2 nodes after first build");
        
        // Modify node positions to simulate physics simulation
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
        
        // Update graph_data to match node_map changes
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

        // Second build - should preserve modified positions
        assert!(actor.build_from_metadata(metadata.clone()).is_ok());
        
        // Verify positions were preserved
        let file1_node = actor.node_map.values()
            .find(|node| node.metadata_id == "file1")
            .expect("file1 node should exist after rebuild");
            
        assert_eq!(file1_node.data.x, 10.0, "Position X should be preserved");
        assert_eq!(file1_node.data.y, 20.0, "Position Y should be preserved");
        assert_eq!(file1_node.data.z, 30.0, "Position Z should be preserved");
        assert_eq!(file1_node.data.vx, 1.0, "Velocity X should be preserved");
        assert_eq!(file1_node.data.vy, 2.0, "Velocity Y should be preserved");
        assert_eq!(file1_node.data.vz, 3.0, "Velocity Z should be preserved");
        
        // Verify file2 node kept its original position (since we didn't modify it)
        let file2_node = actor.node_map.values()
            .find(|node| node.metadata_id == "file2")
            .expect("file2 node should exist after rebuild");
            
        let original_file2_pos = initial_positions.get("file2").unwrap().0;
        assert_eq!(file2_node.data.position(), original_file2_pos, "file2 position should be preserved");
    }
    
    /// Test that new nodes still get proper initial positions
    #[test]
    fn test_new_nodes_get_initial_positions() {
        let mut actor = GraphServiceActor::new();
        
        // First build with one file
        let mut metadata1 = MetadataStore::new();
        metadata1.insert("file1.md".to_string(), FileMetadata {
            file_name: "file1.md".to_string(),
            file_size: 1000,
            node_size: 100,
            hyperlink_count: 5,
            sha1: "abc123".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "".to_string(),
            last_perplexity_process: None,
            topic_counts: HashMap::new(),
        });
        
        assert!(actor.build_from_metadata(metadata1).is_ok());
        assert_eq!(actor.node_map.len(), 1, "Should have 1 node after first build");
        
        // Second build with additional file  
        let mut metadata2 = MetadataStore::new();
        metadata2.insert("file1.md".to_string(), FileMetadata {
            file_name: "file1.md".to_string(),
            file_size: 1000,
            node_size: 100,
            hyperlink_count: 5,
            sha1: "abc123".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "".to_string(),
            last_perplexity_process: None,
            topic_counts: HashMap::new(),
        });
        metadata2.insert("file2.md".to_string(), FileMetadata {
            file_name: "file2.md".to_string(),
            file_size: 2000,
            node_size: 200,
            hyperlink_count: 10,
            sha1: "def456".to_string(),
            last_modified: Utc::now(),
            perplexity_link: "".to_string(),
            last_perplexity_process: None,
            topic_counts: HashMap::new(),
        });
        
        assert!(actor.build_from_metadata(metadata2).is_ok());
        assert_eq!(actor.node_map.len(), 2, "Should have 2 nodes after second build");
        
        // Verify the new node has a non-zero position (not at origin)
        let file2_node = actor.node_map.values()
            .find(|node| node.metadata_id == "file2")
            .expect("file2 node should exist");
            
        let pos = file2_node.data.position();
        let distance_from_origin = (pos.x * pos.x + pos.y * pos.y + pos.z * pos.z).sqrt();
        assert!(distance_from_origin > 0.1, "New node should not be at origin, distance: {}", distance_from_origin);
    }
}
