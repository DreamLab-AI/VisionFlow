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
//! actor.update_advanced_physics_params(advanced_params);
//! actor.trigger_stress_optimization()?;
//! let status = actor.get_semantic_analysis_status();
//! ```

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::Duration;
use log::{debug, info, warn, error, trace};
 
use crate::actors::messages::*;
use crate::utils::binary_protocol;
use crate::actors::client_manager_actor::ClientManagerActor;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::{BinaryNodeData, glam_to_vec3data}; // Added glam_to_vec3data
use crate::actors::gpu_compute_actor::GPUComputeActor;
use crate::models::simulation_params::SimulationParams;
use crate::config::AutoBalanceConfig;

// Advanced physics and AI modules
use crate::models::constraints::{ConstraintSet, Constraint, AdvancedParams};
use crate::services::semantic_analyzer::{SemanticAnalyzer, SemanticFeatures};
use crate::services::edge_generation::{AdvancedEdgeGenerator, EdgeGenerationConfig};
use crate::utils::unified_gpu_compute::{UnifiedGPUCompute};
use crate::models::simulation_params::SimParams;
use crate::physics::stress_majorization::StressMajorizationSolver;
use std::sync::Mutex;

pub struct GraphServiceActor {
    graph_data: Arc<GraphData>, // Changed to Arc<GraphData>
    node_map: HashMap<u32, Node>,
    gpu_compute_addr: Option<Addr<GPUComputeActor>>, // Re-enable for physics updates
    client_manager: Addr<ClientManagerActor>,
    simulation_running: AtomicBool,
    shutdown_complete: Arc<AtomicBool>,
    next_node_id: AtomicU32,
    bots_graph_data: GraphData, // Add a new field for the bots graph
    simulation_params: SimulationParams, // Physics simulation parameters
    
    // Advanced hybrid solver components
    advanced_gpu_context: Option<UnifiedGPUCompute>,
    gpu_init_in_progress: bool, // Flag to prevent multiple initialization attempts
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
    
    // Smooth parameter transitions
    target_params: SimulationParams, // Target physics parameters
    param_transition_rate: f32, // How fast to transition (0.0 - 1.0)
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
        self.simulation_params.attraction_k = self.simulation_params.attraction_k * (1.0 - rate) + self.target_params.attraction_k * rate;
        
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
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
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
        
        Self {
            graph_data: Arc::new(GraphData::new()), // Changed to Arc::new
            node_map: HashMap::new(),
            gpu_compute_addr,
            client_manager,
            simulation_running: AtomicBool::new(false),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            next_node_id: AtomicU32::new(1),
            bots_graph_data: GraphData::new(),
            simulation_params, // Use logseq physics from settings
            
            // Initialize advanced components
            advanced_gpu_context: None,
            gpu_init_in_progress: false,
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
            
            // Smooth parameter transitions
            target_params: SimulationParams::default(),
            param_transition_rate: 0.1, // 10% per frame for smooth transitions
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
        self.node_map.insert(node.id, node.clone());
        
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
        self.node_map.remove(&node_id);
        
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        // Remove from graph data
        graph_data_mut.nodes.retain(|n| n.id != node_id);
        
        // Remove related edges
        graph_data_mut.edges.retain(|e| e.source != node_id && e.target != node_id);
        
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

    pub fn build_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        let mut new_graph_data = GraphData::new();
        self.node_map.clear();
        self.semantic_features_cache.clear();

        // Phase 1: Build nodes with semantic analysis
        info!("Phase 1: Building nodes with semantic analysis");
        for (filename_with_ext, file_meta_data) in &metadata {
            let node_id_val = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            let metadata_id_val = filename_with_ext.trim_end_matches(".md").to_string();
            
            let mut node = Node::new_with_id(metadata_id_val.clone(), Some(node_id_val));
            node.label = file_meta_data.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(file_meta_data.file_size as u64);
            node.data.flags = 1;

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

            self.node_map.insert(node.id, node.clone());
            new_graph_data.nodes.push(node);
        }

        // Phase 2: Generate enhanced edges with multi-modal similarities  
        info!("Phase 2: Generating enhanced edges with multi-modal similarities");
        let enhanced_edges = self.edge_generator.generate(&self.semantic_features_cache);
        
        // Convert enhanced edges to basic edges and add topic-based edges
        let mut edge_map: HashMap<(u32, u32), f32> = HashMap::new();
        
        // Add semantic similarity edges
        for enhanced_edge in enhanced_edges {
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
        
        // Phase 3: DISABLED - Don't automatically generate constraints
        // Constraints should only be enabled explicitly through the control center
        info!("Phase 3: Skipping automatic constraint generation (prevents bouncing)");
        
        // Phase 4: Initialize advanced GPU context if needed (async context not available in message handler)
        // Note: Advanced GPU context initialization will be attempted on first physics step
        if self.advanced_gpu_context.is_none() {
            trace!("Advanced GPU context will be initialized on first physics step");
        }
        
        new_graph_data.metadata = metadata.clone();
        self.graph_data = Arc::new(new_graph_data);
        self.last_semantic_analysis = Some(std::time::Instant::now());
        
        info!("Built enhanced graph: {} nodes, {} edges, {} constraints",
              self.graph_data.nodes.len(), self.graph_data.edges.len(), self.constraint_set.constraints.len());
        
        // Send data to appropriate GPU context
        if self.advanced_gpu_context.is_some() {
            info!("Graph data prepared for advanced GPU physics");
        } else if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            let graph_data_for_gpu = (*self.graph_data).clone();
            // First initialize GPU
            gpu_compute_addr.do_send(InitializeGPU { graph: graph_data_for_gpu.clone() });
            info!("Sent GPU initialization request to GPU compute actor");
            // Then update the graph data
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data_for_gpu });
            info!("Sent initial graph data to legacy GPU compute actor");
        }
        
        Ok(())
    }
    
    /// Helper methods for the hybrid solver orchestration
    
    fn prepare_node_positions(&self) -> Vec<(f32, f32, f32)> {
        self.graph_data.nodes.iter().map(|node| {
            (
                node.data.position.x,
                node.data.position.y,
                node.data.position.z,
            )
        }).collect()
    }
    
    
    fn execute_stress_majorization_step(&mut self) {
        if self.graph_data.nodes.len() < 3 {
            return; // Skip for very small graphs
        }
        
        // Skip stress majorization if GPU compute is not available
        if self.advanced_gpu_context.is_none() && self.gpu_compute_addr.is_none() {
            trace!("Skipping stress majorization - no GPU context available");
            return;
        }
        
        let mut graph_data_clone = (*self.graph_data).clone();
        
        match self.stress_solver.optimize(&mut graph_data_clone, &self.constraint_set) {
            Ok(result) => {
                if result.converged || result.final_stress < f32::INFINITY {
                    // Apply optimized positions back to main graph
                    let graph_data_mut = Arc::make_mut(&mut self.graph_data);
                    for (i, node) in graph_data_mut.nodes.iter_mut().enumerate() {
                        if let Some(optimized_node) = graph_data_clone.nodes.get(i) {
                            // Validate positions before applying
                            let new_x = optimized_node.data.position.x;
                            let new_y = optimized_node.data.position.y;
                            let new_z = optimized_node.data.position.z;
                            
                            if new_x.is_finite() && new_y.is_finite() && new_z.is_finite() {
                                // Use physics boundary settings if enabled, otherwise no clamping
                                if self.simulation_params.enable_bounds && self.simulation_params.viewport_bounds > 0.0 {
                                    let boundary_limit = self.simulation_params.viewport_bounds;
                                    node.data.position.x = new_x.clamp(-boundary_limit, boundary_limit);
                                    node.data.position.y = new_y.clamp(-boundary_limit, boundary_limit);
                                    node.data.position.z = new_z.clamp(-boundary_limit, boundary_limit);
                                } else {
                                    // No boundary constraints
                                    node.data.position.x = new_x;
                                    node.data.position.y = new_y;
                                    node.data.position.z = new_z;
                                }
                            } else {
                                warn!("Skipping invalid position from stress majorization for node {}: ({}, {}, {})", 
                                      node.id, new_x, new_y, new_z);
                            }
                        }
                    }
                    
                    // Update node_map as well with validated positions
                    for node in &graph_data_mut.nodes {
                        if let Some(node_in_map) = self.node_map.get_mut(&node.id) {
                            // Use the already validated and clamped positions
                            node_in_map.data.position.x = node.data.position.x;
                            node_in_map.data.position.y = node.data.position.y;
                            node_in_map.data.position.z = node.data.position.z;
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
        // DISABLED: Dynamic constraints cause bouncing behavior
        // Constraints should only be enabled explicitly through the control center
        info!("Skipping dynamic constraint updates (prevents bouncing)");
        return;
        
        // Original code disabled to prevent automatic constraint generation:
        /*
        // Only update if we have recent semantic analysis
        if self.last_semantic_analysis.is_none() {
            return;
        }
        
        // Clear dynamic constraints (keep manually added ones)
        self.constraint_set.set_group_active("semantic_dynamic", false);
        
        // Generate new semantic constraints
        if let Ok(constraints) = self.generate_dynamic_semantic_constraints() {
            for constraint in constraints {
                self.constraint_set.add_to_group("semantic_dynamic", constraint);
            }
            trace!("Updated dynamic semantic constraints");
        }
        
        // Re-cluster nodes based on current positions and semantic features
        if let Ok(clustering_constraints) = self.generate_clustering_constraints() {
            self.constraint_set.set_group_active("clustering_dynamic", false);
            for constraint in clustering_constraints {
                self.constraint_set.add_to_group("clustering_dynamic", constraint);
            }
            trace!("Updated dynamic clustering constraints");
        }
        */
    }
    
    fn generate_initial_semantic_constraints(&mut self, graph_data: &GraphData) {
        // DISABLED: Boundary constraints cause bouncing behavior
        // Don't automatically generate any constraints - they should only be enabled through control center
        info!("Skipping automatic boundary constraint generation (prevents bouncing)");
        return;  // Exit early to prevent any constraint generation
        
        // The code below is disabled to prevent boundary constraints from causing bouncing
        
        // Generate domain-based clustering
        let mut domain_clusters: HashMap<String, Vec<u32>> = HashMap::new();
        
        for node in &graph_data.nodes {
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                if !features.domains.is_empty() {
                    let domain_key = format!("{:?}", features.domains[0]);
                    domain_clusters.entry(domain_key).or_insert_with(Vec::new).push(node.id);
                }
            }
        }
        
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
        
        info!("Generated {} initial constraints", self.constraint_set.constraints.len());
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
        let mut updated_count = 0;
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        
        for (node_id, position_data) in positions {
            // Update in node_map
            if let Some(node) = self.node_map.get_mut(&node_id) {
                node.data.position = position_data.position;
                node.data.velocity = position_data.velocity;
                updated_count += 1;
            }
            
            // Update in graph_data.nodes
            if let Some(node) = graph_data_mut.nodes.iter_mut().find(|n| n.id == node_id) {
                node.data.position = position_data.position;
                node.data.velocity = position_data.velocity;
            }
        }
        
        // Sample first few nodes to check positions
        if updated_count > 0 {
            let sample_size = std::cmp::min(3, updated_count);
            for i in 0..sample_size {
                if let Some(node) = self.node_map.get(&(i as u32)) {
                    info!("Node {} position: ({:.2}, {:.2}, {:.2})", 
                        i, node.data.position.x, node.data.position.y, node.data.position.z);
                }
            }
        }
        
        // Check for extreme positions, boundary nodes, and calculate metrics including kinetic energy
        let config = &self.simulation_params.auto_balance_config;
        let mut extreme_count = 0;
        let mut boundary_nodes = 0;
        let mut max_distance = 0.0f32;
        let mut total_distance = 0.0f32;
        let mut positions = Vec::new();
        let mut total_kinetic_energy = 0.0f32;
        
        for (_, node) in self.node_map.iter() {
            let dist = node.data.position.x.abs()
                .max(node.data.position.y.abs())
                .max(node.data.position.z.abs());
            max_distance = max_distance.max(dist);
            total_distance += dist;
            positions.push(dist);
            
            // Calculate kinetic energy: KE = 0.5 * mass * velocity^2
            // Assuming unit mass for simplicity (mass = 1)
            let velocity_squared = node.data.velocity.x * node.data.velocity.x + 
                                  node.data.velocity.y * node.data.velocity.y + 
                                  node.data.velocity.z * node.data.velocity.z;
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
        
        // Auto-tune physics if enabled
        if self.simulation_params.auto_balance {
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
            
            // Check if enough time has passed since last auto-balance
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            
            static mut LAST_AUTO_BALANCE: u64 = 0;
            let interval = self.simulation_params.auto_balance_interval_ms as u64;
            
            unsafe {
                if now - LAST_AUTO_BALANCE >= interval {
                    LAST_AUTO_BALANCE = now;
                    
                    // Prepare position data for advanced analysis
                    let position_data: Vec<(f32, f32, f32)> = self.node_map.values()
                        .map(|node| (node.data.position.x, node.data.position.y, node.data.position.z))
                        .collect();
                    
                    // Detect spatial hashing issues
                    let config = &self.simulation_params.auto_balance_config;
                    let (has_spatial_issues, efficiency_score) = self.detect_spatial_hashing_issues(&position_data, config);
                    let has_numerical_instability = self.detect_numerical_instability(&position_data, config);
                    
                    // Detect if we've found a minima (stable state) using both position and kinetic energy
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
                        // Use hardcoded thresholds for now, can be moved to config later
                        let ke_threshold = 0.01; // Low kinetic energy threshold  
                        let ke_variance_threshold = 0.001; // Very low KE variance threshold
                        position_variance < config.stability_variance_threshold && 
                        recent_ke < ke_threshold && 
                        ke_variance < ke_variance_threshold
                    } else {
                        false
                    };
                    
                    // Detect bouncing: many nodes at boundary OR oscillating history
                    let is_bouncing = boundary_nodes as f32 > (self.node_map.len() as f32 * config.bouncing_node_percentage) || 
                                     (self.auto_balance_history.len() >= config.oscillation_detection_frames && {
                                         // Check for oscillation in recent history
                                         let recent = &self.auto_balance_history[self.auto_balance_history.len()-config.oscillation_detection_frames..];
                                         let changes = recent.windows(2)
                                             .filter(|w| (w[0] - w[1]).abs() > config.oscillation_change_threshold)
                                             .count();
                                         changes > config.min_oscillation_changes  // Too many changes means oscillating
                                     });
                    
                    if is_bouncing {
                        info!("[AUTO-BALANCE] Bouncing detected! Boundary nodes: {}/{}, max distance: {:.0}", 
                              boundary_nodes, self.node_map.len(), max_distance);
                        self.stable_count = 0;
                        
                        // Check for complete deadlock (all nodes stuck with no movement)
                        let is_deadlocked = boundary_nodes == self.node_map.len() && avg_kinetic_energy < 0.001;
                        
                        info!("[DEADLOCK-CHECK] Boundary nodes: {}/{}, Kinetic energy: {:.6}, Deadlocked: {}", 
                              boundary_nodes, self.node_map.len(), avg_kinetic_energy, is_deadlocked);
                        
                        if is_deadlocked {
                            // Recovery mode: Gradually restore forces to allow movement
                            info!("[AUTO-BALANCE] DEADLOCK DETECTED! All nodes stuck. Initiating recovery...");
                            
                            let mut new_target = self.target_params.clone();
                            // Use stronger values to break out of boundary stuck state
                            new_target.repel_k = 2.5;  // Strong repulsion to push nodes apart
                            new_target.damping = 0.7;  // Lower damping for more movement
                            new_target.max_velocity = 10.0;  // Higher velocity to escape boundaries
                            new_target.spring_k = 1.0;  // Strong springs to pull connected nodes
                            // Keep existing boundary settings from simulation_params
                            // Don't override enable_bounds or viewport_bounds
                            
                            // Use faster transition for deadlock recovery
                            self.param_transition_rate = 0.5;  // 50% per frame for faster recovery
                            
                            self.set_target_params(new_target);
                            self.send_auto_balance_notification("Adaptive Balancing: Recovering from deadlock");
                        } else {
                            // Normal bouncing stabilization - use gentler reduction
                            let mut new_target = self.target_params.clone();
                            new_target.repel_k = (self.simulation_params.repel_k * 0.8).max(0.1);  // Less extreme reduction
                            new_target.damping = (self.simulation_params.damping * 1.05).min(0.95);  // Gradual damping increase
                            new_target.max_velocity = (self.simulation_params.max_velocity * 0.8).max(0.5);  // Keep some velocity
                            new_target.spring_k = (self.simulation_params.spring_k * 0.9).max(0.1);
                            // Keep existing boundary settings from simulation_params
                            // Don't override enable_bounds or viewport_bounds
                            
                            self.set_target_params(new_target);
                            self.send_auto_balance_notification("Adaptive Balancing: Stabilizing bouncing nodes");
                        }
                        
                        info!("[AUTO-BALANCE] Applied parameters - repel_k: {:.3}, damping: {:.3}, max_velocity: {:.3}", 
                              self.target_params.repel_k, 
                              self.target_params.damping,
                              self.target_params.max_velocity);
                        
                    } else if extreme_count > 0 || max_distance > config.spreading_distance_threshold {
                        // Nodes spreading too far - INCREASE ATTRACTION and CENTER GRAVITY
                        info!("[AUTO-BALANCE] Nodes spreading (max: {:.0}), increasing attractive and centering forces", max_distance);
                        self.stable_count = 0;
                        
                        // Calculate how much to increase attraction based on spread
                        let spread_ratio = (max_distance / config.spreading_distance_threshold).min(3.0);
                        let attraction_boost = 1.0 + (spread_ratio - 1.0) * 0.2; // Up to 40% increase
                        let gravity_boost = 1.0 + (spread_ratio - 1.0) * 0.5; // Up to 100% increase for gravity
                        
                        let mut new_target = self.target_params.clone();
                        // CORRECT: Increase attractive forces to pull nodes together
                        new_target.spring_k = (self.simulation_params.spring_k * attraction_boost).min(0.1);
                        new_target.attraction_k = (self.simulation_params.attraction_k * attraction_boost).min(0.5);
                        
                        // NEW: Adjust center gravity through cooling_rate proxy (center_gravity_k equivalent)
                        new_target.cooling_rate = (self.simulation_params.cooling_rate * gravity_boost)
                            .max(config.center_gravity_min)
                            .min(config.center_gravity_max);
                        
                        // Slightly reduce max velocity to allow settling
                        new_target.max_velocity = (self.simulation_params.max_velocity * 0.9).max(1.0);
                        // Keep repulsion stable or slightly increase to maintain structure
                        new_target.repel_k = (self.simulation_params.repel_k * 1.05).min(100.0);
                        
                        // DO NOT force enable boundaries - respect user settings
                        // Boundaries should only be enabled if explicitly set in config
                        // Keep existing enable_bounds setting from simulation_params
                        
                        info!("[AUTO-BALANCE] Increased containment - spring_k: {:.3}, attraction_k: {:.3}, center_gravity: {:.6}, repel_k: {:.3}", 
                              new_target.spring_k, new_target.attraction_k, new_target.cooling_rate, new_target.repel_k);
                        
                        self.set_target_params(new_target);
                        
                        // Send notification to client
                        self.send_auto_balance_notification("Adaptive Balancing: Increasing containment forces to prevent spreading");
                    } else if extreme_count == 0 && max_distance < config.clustering_distance_threshold {
                        // Nodes too clustered - increase repulsion AND reduce attraction
                        info!("[AUTO-BALANCE] Nodes clustered at {:.1} units, adjusting forces", max_distance);
                        
                        let mut new_target = self.target_params.clone();
                        // Increase repulsion to push nodes apart
                        new_target.repel_k = (self.simulation_params.repel_k * 1.3).min(100.0);
                        // ALSO reduce attraction to allow expansion
                        new_target.spring_k = (self.simulation_params.spring_k * 0.8).max(0.001);
                        new_target.attraction_k = (self.simulation_params.attraction_k * 0.8).max(0.01);
                        // Allow higher velocity for expansion
                        new_target.max_velocity = (self.simulation_params.max_velocity * 1.1).min(10.0);
                        
                        info!("[AUTO-BALANCE] Adjusted clustering - repel_k: {:.3}, spring_k: {:.3}, attraction_k: {:.3}", 
                              new_target.repel_k, new_target.spring_k, new_target.attraction_k);
                        
                        self.set_target_params(new_target);
                        self.stable_count = 0; // Reset stability counter
                        
                        // Send notification to client
                        self.send_auto_balance_notification("Adaptive Balancing: Expanding clustered nodes");
                    } else if has_numerical_instability {
                        // Critical: Numerical instability detected - emergency parameter adjustment
                        info!("[AUTO-BALANCE] NUMERICAL INSTABILITY detected - emergency parameter adjustment");
                        self.stable_count = 0;
                        
                        let mut new_target = self.target_params.clone();
                        
                        // Increase repulsion softening to prevent division by zero
                        new_target.cooling_rate = (self.simulation_params.cooling_rate * 10.0).min(config.repulsion_softening_max);
                        
                        // Reduce time step and forces for numerical stability
                        new_target.dt = (self.simulation_params.dt * 0.5).max(0.001);
                        new_target.max_force = (self.simulation_params.max_force * 0.8).max(1.0);
                        new_target.max_velocity = (self.simulation_params.max_velocity * 0.7).max(0.1);
                        
                        // Increase damping for stability
                        new_target.damping = (self.simulation_params.damping * 1.2).min(0.99);
                        
                        self.param_transition_rate = 0.3; // Fast emergency transition
                        self.set_target_params(new_target);
                        self.send_auto_balance_notification("Adaptive Balancing: Emergency - Fixing numerical instability");
                        
                    } else if has_spatial_issues && efficiency_score < config.spatial_hash_efficiency_threshold {
                        // Spatial hashing is inefficient - adjust grid parameters
                        info!("[AUTO-BALANCE] Spatial hashing inefficiency detected (score: {:.2})", efficiency_score);
                        self.stable_count = 0;
                        
                        let mut new_target = self.target_params.clone();
                        
                        // Adjust grid cell size based on average inter-node distance
                        let avg_distance = if !position_data.is_empty() {
                            let mut total_dist = 0.0f32;
                            let mut count = 0;
                            for i in 0..std::cmp::min(position_data.len(), 20) {
                                for j in i+1..std::cmp::min(i+5, position_data.len()) {
                                    let pos1 = position_data[i];
                                    let pos2 = position_data[j];
                                    let dist = ((pos1.0 - pos2.0).powi(2) + (pos1.1 - pos2.1).powi(2) + (pos1.2 - pos2.2).powi(2)).sqrt();
                                    total_dist += dist;
                                    count += 1;
                                }
                            }
                            if count > 0 { total_dist / count as f32 } else { 10.0 }
                        } else { 10.0 };
                        
                        // Set grid cell size to approximately 2x average distance for optimal spatial hashing
                        let optimal_grid_cell_size = (avg_distance * 2.0)
                            .max(config.grid_cell_size_min)
                            .min(config.grid_cell_size_max);
                        
                        // Update repulsion cutoff to match grid efficiency
                        new_target.max_repulsion_dist = optimal_grid_cell_size * 1.5; // Slightly larger than grid cell
                        
                        // Adjust repulsion softening based on clustering
                        if efficiency_score < 0.2 {
                            // Severe clustering - increase softening
                            new_target.cooling_rate = (self.simulation_params.cooling_rate * 2.0).min(config.repulsion_softening_max);
                        }
                        
                        info!("[AUTO-BALANCE] Adjusted spatial parameters - grid_cell_proxy: {:.1}, repulsion_cutoff: {:.1}", 
                              optimal_grid_cell_size, new_target.max_repulsion_dist);
                        
                        self.set_target_params(new_target);
                        self.send_auto_balance_notification("Adaptive Balancing: Optimizing spatial hashing efficiency");
                        
                    } else if is_stable {
                        // We've found a stable minima
                        self.stable_count += 1;
                        
                        // After being stable for configured frames, update UI and save
                        if self.stable_count == config.stability_frame_count {
                            info!("[AUTO-BALANCE] Stable minima found at {:.1} units - updating UI sliders", max_distance);
                            
                            // Send success notification to client
                            self.send_auto_balance_notification("Adaptive Balancing: Stable configuration found!");
                            
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

        // Start the simulation interval
        ctx.run_interval(Duration::from_millis(16), |actor, ctx| {
            if !actor.simulation_running.load(Ordering::SeqCst) {
                return;
            }

            actor.run_simulation_step(ctx);
        });
    }

    fn run_simulation_step(&mut self, ctx: &mut Context<Self>) {
        // Apply smooth parameter transitions if auto-balance is enabled
        if self.simulation_params.auto_balance {
            self.smooth_transition_params();
        }
        
        // Increment counters for periodic operations
        self.stress_step_counter += 1;
        self.constraint_update_counter += 1;
        
        // Execute periodic stress-majorization projection
        if self.stress_step_counter >= self.advanced_params.stress_step_interval_frames {
            self.execute_stress_majorization_step();
            self.stress_step_counter = 0;
        }
        
        // Update constraints periodically based on semantic analysis
        if self.constraint_update_counter >= 120 { // Every 2 seconds at 60 FPS
            self.update_dynamic_constraints();
            self.constraint_update_counter = 0;
        }
        
        // Initialize advanced GPU context if needed
        if self.advanced_gpu_context.is_none() && !self.gpu_init_in_progress && !self.graph_data.nodes.is_empty() {
            // Mark initialization as in progress
            self.gpu_init_in_progress = true;
            
            // Attempt to initialize advanced GPU context
            let graph_data_clone = (*self.graph_data).clone();
            let self_addr = ctx.address();
            
            actix::spawn(async move {
                // Load the PTX file content from the single location where build.rs places it
                // In production, this is copied to /app/src/utils/ptx/
                // In development, it's at src/utils/ptx/ relative to the workspace
                let ptx_path = if std::path::Path::new("/app/src/utils/ptx/visionflow_unified.ptx").exists() {
                    "/app/src/utils/ptx/visionflow_unified.ptx"  // Production path
                } else {
                    "src/utils/ptx/visionflow_unified.ptx"  // Development path
                };
                
                let ptx_content = match std::fs::read_to_string(ptx_path) {
                    Ok(content) => {
                        info!("Successfully loaded PTX file from: {}", ptx_path);
                        content
                    },
                    Err(e) => {
                        error!("Failed to load PTX file from {}: {}", ptx_path, e);
                        error!("PTX should be generated by build.rs during compilation");
                        return;
                    }
                };
                
                // For CSR format, each undirected edge becomes 2 directed edges
                let num_directed_edges = graph_data_clone.edges.len() * 2;
                info!("Creating UnifiedGPUCompute with {} nodes and {} directed edges (from {} undirected edges)", 
                      graph_data_clone.nodes.len(), num_directed_edges, graph_data_clone.edges.len());
                match UnifiedGPUCompute::new(
                    graph_data_clone.nodes.len(),
                    num_directed_edges,
                    &ptx_content,
                ) {
                    Ok(context) => {
                        info!("Successfully initialized advanced GPU context");
                        self_addr.do_send(SetAdvancedGPUContext { context });
                    }
                    Err(e) => {
                        warn!("Failed to initialize advanced GPU context: {}", e);
                        // Reset the flag on failure so we can retry later
                        self_addr.do_send(ResetGPUInitFlag);
                    }
                }
            });
        }
        
        // Use advanced GPU compute only - legacy path removed
        if self.advanced_gpu_context.is_some() {
            self.run_advanced_gpu_step(ctx);
        } else {
            warn!("No GPU compute context available for physics simulation");
        }
    }
    
    fn run_advanced_gpu_step(&mut self, _ctx: &mut Context<Self>) {
        // Only log physics step execution when debug is enabled
        if crate::utils::logging::is_debug_enabled() {
            info!("[GPU STEP] === Starting physics simulation step ===");
            info!("[GPU STEP] Current physics parameters:");
            info!("  - repel_k: {} (node spreading force)", self.simulation_params.repel_k);
            info!("  - damping: {:.3} (velocity reduction, 1.0 = frozen)", self.simulation_params.damping);
            info!("  - dt: {:.3} (simulation speed)", self.simulation_params.dt);
            info!("  - spring_k: {:.3} (edge tension)", self.simulation_params.spring_k);
            info!("  - attraction_k: {:.3} (unused - for future clustering)", self.simulation_params.attraction_k);
            info!("  - center_gravity_k: {:.3} (center pull force)", self.simulation_params.center_gravity_k);
            info!("  - max_velocity: {:.3} (explosion prevention)", self.simulation_params.max_velocity);
            info!("  - enabled: {} (is physics on?)", self.simulation_params.enabled);
            info!("  - auto_balance: {} (auto-tuning enabled?)", self.simulation_params.auto_balance);
        }
        
        if !self.simulation_params.enabled {
            if crate::utils::logging::is_debug_enabled() {
                info!("[GPU STEP] Physics disabled - skipping simulation");
            }
            return;
        }
        
        // Physics parameters loaded
        
        // Prepare node positions for unified GPU compute
        let positions = self.prepare_node_positions();
        
        // Update GPU with enhanced node data
        let mut positions_to_update = Vec::new();
        let active_constraints_count;
        let iteration_count;
        
        if let Some(ref mut gpu_context) = self.advanced_gpu_context {
            let positions_x: Vec<f32> = positions.iter().map(|p| p.0).collect();
            let positions_y: Vec<f32> = positions.iter().map(|p| p.1).collect();
            let positions_z: Vec<f32> = positions.iter().map(|p| p.2).collect();
            
            // Upload positions to GPU
            if let Err(e) = gpu_context.upload_positions(&positions_x, &positions_y, &positions_z) {
                error!("Failed to upload positions to unified GPU: {}", e);
                return;
            }
            
            let mut row_offsets = vec![0];
            let mut col_indices = vec![];
            let mut weights = vec![];
            let mut adj = vec![vec![]; self.graph_data.nodes.len()];
            let node_indices: HashMap<u32, usize> = self.graph_data.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

            for edge in &self.graph_data.edges {
                if let (Some(&src_idx), Some(&dst_idx)) = (node_indices.get(&edge.source), node_indices.get(&edge.target)) {
                    adj[src_idx].push((dst_idx as i32, edge.weight));
                    adj[dst_idx].push((src_idx as i32, edge.weight));
                }
            }

            for i in 0..self.graph_data.nodes.len() {
                for (neighbor, weight) in &adj[i] {
                    col_indices.push(*neighbor);
                    weights.push(*weight);
                }
                row_offsets.push(col_indices.len() as i32);
            }
            
            // Upload edges to GPU
            if let Err(e) = gpu_context.upload_edges_csr(&row_offsets, &col_indices, &weights) {
                error!("Failed to upload edges to unified GPU: {}", e);
                error!("  row_offsets.len() = {}, col_indices.len() = {}, weights.len() = {}", 
                       row_offsets.len(), col_indices.len(), weights.len());
                return;
            }
        
            let sim_params = crate::utils::unified_gpu_compute::SimParams::from(&self.simulation_params);
            // Set GPU parameters
            
            // Execute GPU physics step
            match gpu_context.execute(sim_params) {
                Ok(()) => {
                    let mut host_pos_x = vec![0.0; self.graph_data.nodes.len()];
                    let mut host_pos_y = vec![0.0; self.graph_data.nodes.len()];
                    let mut host_pos_z = vec![0.0; self.graph_data.nodes.len()];
                    gpu_context.download_positions(&mut host_pos_x, &mut host_pos_y, &mut host_pos_z).unwrap();

                    let node_ids: Vec<u32> = self.graph_data.nodes.iter().map(|n| n.id).collect();
                    
                    for (index, node_id) in node_ids.iter().enumerate() {
                        let binary_node = BinaryNodeData {
                            position: crate::types::vec3::Vec3Data {
                                x: host_pos_x[index],
                                y: host_pos_y[index],
                                z: host_pos_z[index]
                            },
                            velocity: crate::types::vec3::Vec3Data::zero(),
                            mass: 1,
                            flags: 0,
                            padding: [0, 0],
                        };
                        positions_to_update.push((*node_id, binary_node));
                    }
                    
                    if !positions_to_update.is_empty() {
                        // Broadcast to clients
                        let binary_data = binary_protocol::encode_node_data(&positions_to_update);
                        self.client_manager.do_send(BroadcastNodePositions {
                            positions: binary_data
                        });
                    }
                }
                Err(e) => {
                    error!("Unified GPU physics step failed: {}", e);
                }
            }
            active_constraints_count = 0;
            
            iteration_count = self.stress_step_counter;
            
            // Log progress periodically
            if iteration_count % 60 == 0 {
                trace!("Advanced physics step completed (iteration {}, {} constraints active)", 
                      iteration_count, active_constraints_count);
            }
        } else {
            return;
        }
        
        // Update local positions after GPU context is released
        if !positions_to_update.is_empty() {
            self.update_node_positions(positions_to_update);
        }
    }
    // Legacy GPU step removed - only advanced GPU compute is supported
    // All physics computation uses the unified GPU kernel

    /// Update advanced physics parameters
    pub fn update_advanced_physics_params(&mut self, params: AdvancedParams) {
        self.advanced_params = params.clone();
        self.stress_solver = StressMajorizationSolver::from_advanced_params(&params);
        
        // Update advanced GPU context if available
        if let Some(ref mut gpu_context) = self.advanced_gpu_context {
            let sim_params = SimParams::from(&self.simulation_params);
            gpu_context.set_params(sim_params);
        }
        
        info!("Updated advanced physics parameters via public API");
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
        self.advanced_gpu_context.is_some()
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
}

impl Actor for GraphServiceActor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        info!("GraphServiceActor started");
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
    type Result = Result<GraphData, String>; // Result type changed from Arc<GraphData>
 
    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("DEBUG_VERIFICATION: GraphServiceActor handling GetGraphData with OWNED data clone strategy.");
        Ok((*self.graph_data).clone()) // Clones the GraphData itself
    }
}

impl Handler<UpdateNodePositions> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_positions(msg.positions);
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
    type Result = Result<HashMap<u32, Node>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Self::Context) -> Self::Result {
        Ok(self.node_map.clone())
    }
}

impl Handler<BuildGraphFromMetadata> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.build_from_metadata(msg.metadata)
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
        if let Some(node) = self.node_map.get_mut(&msg.node_id) {
            // Preserve existing mass and flags
            let original_mass = node.data.mass;
            let original_flags = node.data.flags;
            
            node.data.position = glam_to_vec3data(msg.position);
            node.data.velocity = glam_to_vec3data(msg.velocity);
            
            // Restore mass and flags
            node.data.mass = original_mass;
            node.data.flags = original_flags;
        } else {
            debug!("Received update for unknown node ID: {}", msg.node_id);
            return Err(format!("Unknown node ID: {}", msg.node_id));
        }
        
        // Update corresponding node in graph
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        for node_in_graph_data in &mut graph_data_mut.nodes { // Iterate over mutable graph_data
            if node_in_graph_data.id == msg.node_id {
                // Preserve mass and flags
                let original_mass = node_in_graph_data.data.mass;
                let original_flags = node_in_graph_data.data.flags;
                
                node_in_graph_data.data.position = glam_to_vec3data(msg.position);
                node_in_graph_data.data.velocity = glam_to_vec3data(msg.velocity);
                
                // Restore mass and flags
                node_in_graph_data.data.mass = original_mass;
                node_in_graph_data.data.flags = original_flags;
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
        if let Ok(notifications) = self.auto_balance_notifications.lock() {
            let filtered_notifications = if let Some(since) = msg.since_timestamp {
                notifications.iter()
                    .filter(|n| n.timestamp > since)
                    .cloned()
                    .collect()
            } else {
                notifications.clone()
            };
            
            Ok(filtered_notifications)
        } else {
            Err("Failed to access notifications".to_string())
        }
    }
}

impl Handler<UpdateGraphData> for GraphServiceActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("Updating graph data with {} nodes, {} edges",
              msg.graph_data.nodes.len(), msg.graph_data.edges.len());
        
        // Update graph data by creating a new Arc
        self.graph_data = Arc::new(msg.graph_data);
        
        // Rebuild node map
        self.node_map.clear();
        for node in &self.graph_data.nodes { // Dereferences Arc for iteration
            self.node_map.insert(node.id, node.clone());
        }
        
        info!("Graph data updated successfully");
        Ok(())
    }
}

impl Handler<UpdateBotsGraph> for GraphServiceActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) -> Self::Result {
        // This logic converts `AgentStatus` objects into `Node` and `Edge` objects
        let mut nodes = vec![];
        let mut edges = vec![];
        
        // Use a high ID range (starting at 10000) to avoid conflicts with main graph
        let bot_id_offset = 10000;
        
        // Create nodes for each agent
        for (i, agent) in msg.agents.iter().enumerate() {
            let node_id = bot_id_offset + i as u32;
            
            // Create a node for each agent
            let mut node = Node::new_with_id(agent.agent_id.clone(), Some(node_id));
            
            // Set node properties based on agent status
            node.color = Some(match agent.profile.agent_type {
                crate::types::claude_flow::AgentType::Coordinator => "#FF6B6B".to_string(),
                crate::types::claude_flow::AgentType::Researcher => "#4ECDC4".to_string(),
                crate::types::claude_flow::AgentType::Coder => "#45B7D1".to_string(),
                crate::types::claude_flow::AgentType::Analyst => "#FFA07A".to_string(),
                crate::types::claude_flow::AgentType::Architect => "#98D8C8".to_string(),
                crate::types::claude_flow::AgentType::Tester => "#F7DC6F".to_string(),
                _ => "#95A5A6".to_string(),
            });
            
            node.label = agent.profile.name.clone();
            node.size = Some(20.0 + (agent.active_tasks_count as f32 * 5.0)); // Size based on activity
            
            // Add metadata including agent flag for GPU physics
            node.metadata.insert("agent_type".to_string(), format!("{:?}", agent.profile.agent_type));
            node.metadata.insert("status".to_string(), agent.status.clone());
            node.metadata.insert("active_tasks".to_string(), agent.active_tasks_count.to_string());
            node.metadata.insert("completed_tasks".to_string(), agent.completed_tasks_count.to_string());
            node.metadata.insert("is_agent".to_string(), "true".to_string()); // Agent node flag
            
            nodes.push(node);
        }
        
        // Create edges based on communication intensity
        for (i, source_agent) in msg.agents.iter().enumerate() {
            for (j, target_agent) in msg.agents.iter().enumerate() {
                if i != j {
                    let source_node_id = bot_id_offset + i as u32;
                    let target_node_id = bot_id_offset + j as u32;
                    
                    // Calculate Communication Intensity
                    let communication_intensity = self.calculate_communication_intensity(
                        &source_agent.profile.agent_type,
                        &target_agent.profile.agent_type,
                        source_agent.active_tasks_count,
                        target_agent.active_tasks_count,
                        source_agent.success_rate,
                        target_agent.success_rate,
                    );
                    
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
        self.bots_graph_data.nodes = nodes;
        self.bots_graph_data.edges = edges;
        
        info!("Updated bots graph with {} agents and {} communication edges", 
             msg.agents.len(), self.bots_graph_data.edges.len());
        
        // Send bots-full-update WebSocket message with complete agent data
        let bots_full_update = serde_json::json!({
            "type": "bots-full-update",
            "agents": msg.agents.iter().map(|agent| {
                serde_json::json!({
                    "id": agent.agent_id,
                    "type": format!("{:?}", agent.profile.agent_type),
                    "status": agent.status,
                    "name": agent.profile.name,
                    "cpuUsage": agent.cpu_usage,
                    "memoryUsage": agent.memory_usage,
                    "health": agent.health,
                    "workload": agent.activity,
                    "capabilities": agent.profile.capabilities,
                    "currentTask": agent.current_task.as_ref().map(|t| t.description.clone()),
                    "tasksActive": agent.tasks_active,
                    "tasksCompleted": agent.completed_tasks_count,
                    "successRate": agent.success_rate,
                    "tokens": agent.token_usage.total,
                    "tokenRate": agent.token_usage.token_rate,
                    "activity": agent.activity,
                    "swarmId": agent.swarm_id,
                    "agentMode": agent.agent_mode,
                    "parentQueenId": agent.parent_queen_id,
                    "processingLogs": agent.processing_logs,
                    "createdAt": agent.timestamp.to_rfc3339(),
                    "age": chrono::Utc::now().timestamp_millis() - agent.timestamp.timestamp_millis(),
                })
            }).collect::<Vec<_>>(),
            "swarmMetrics": {
                "totalAgents": msg.agents.len(),
                "activeAgents": msg.agents.iter().filter(|a| a.status == "active").count(),
                "totalTasks": msg.agents.iter().map(|a| a.tasks_active).sum::<u32>(),
                "completedTasks": msg.agents.iter().map(|a| a.completed_tasks_count).sum::<u32>(),
                "avgSuccessRate": msg.agents.iter().map(|a| a.success_rate).sum::<f32>() as f64 / msg.agents.len().max(1) as f64,
                "totalTokens": msg.agents.iter().map(|a| a.token_usage.total).sum::<u64>(),
            },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        // Broadcast the full update to all connected clients
        self.client_manager.do_send(BroadcastMessage {
            message: bots_full_update.to_string(),
        });
        
        // Remove CPU physics calculations for agent graph - delegate to GPU
        // The GPU will use the edge weights (communication intensity) for spring forces
    }
}

impl Handler<GetBotsGraphData> for GraphServiceActor {
    type Result = Result<GraphData, String>;

    fn handle(&mut self, _msg: GetBotsGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.bots_graph_data.clone())
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
            info!("  - attraction_k: {:.3} (was)", self.simulation_params.attraction_k);
            info!("  - max_velocity: {:.3} (was)", self.simulation_params.max_velocity);
            info!("  - enabled: {} (was)", self.simulation_params.enabled);
            info!("  - auto_balance: {} (was)", self.simulation_params.auto_balance);
            
            info!("[GRAPH ACTOR] NEW physics values:");
            info!("  - repel_k: {} (new)", msg.params.repel_k);
            info!("  - damping: {:.3} (new)", msg.params.damping);
            info!("  - dt: {:.3} (new)", msg.params.dt);
            info!("  - spring_k: {:.3} (new)", msg.params.spring_k);
            info!("  - attraction_k: {:.3} (new)", msg.params.attraction_k);
            info!("  - max_velocity: {:.3} (new)", msg.params.max_velocity);
            info!("  - enabled: {} (new)", msg.params.enabled);
            info!("  - auto_balance: {} (new)", msg.params.auto_balance);
        }
        
        // Check if auto-balance is being turned on for the first time
        let auto_balance_just_enabled = !self.simulation_params.auto_balance && msg.params.auto_balance;
        
        self.simulation_params = msg.params.clone();
        
        // If auto-balance was just enabled, reset tracking state for fresh start
        if auto_balance_just_enabled {
            info!("[AUTO-BALANCE] Auto-balance enabled - starting adaptive tuning from current values");
            
            // Reset history and stability counter for fresh start
            self.auto_balance_history.clear();
            self.stable_count = 0;
            
            info!("[AUTO-BALANCE] Will adaptively tune from current settings - repel_k: {:.3}, damping: {:.3}", 
                  self.simulation_params.repel_k, self.simulation_params.damping);
        }
        
        // Update the advanced GPU context if available
        if let Some(ref mut gpu_context) = self.advanced_gpu_context {
            let sim_params = SimParams::from(&self.simulation_params);
            if crate::utils::logging::is_debug_enabled() {
                info!("Updating advanced_gpu_context with new params");
            }
            gpu_context.set_params(sim_params);
        }
        
        // Also update GPU compute actor if available
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            if crate::utils::logging::is_debug_enabled() {
                info!("Forwarding params to GPUComputeActor");
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
                
                let node_data = BinaryNodeData {
                    position: node.data.position.clone(),
                    velocity: node.data.velocity.clone(),
                    mass: node.data.mass,
                    flags: node.data.flags | 0x40, // Set knowledge graph flag
                    padding: node.data.padding,
                };
                
                knowledge_nodes.push((node.id, node_data));
            }
        }
        
        // Collect agent graph positions if requested
        if msg.include_agent_graph {
            for node in &self.bots_graph_data.nodes {
                let node_data = BinaryNodeData {
                    position: node.data.position.clone(),
                    velocity: node.data.velocity.clone(),
                    mass: node.data.mass,
                    flags: node.data.flags | 0x80, // Set agent flag
                    padding: node.data.padding,
                };
                
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
        if let Some(ref mut gpu_context) = self.advanced_gpu_context {
            let sim_params = SimParams::from(&self.simulation_params);
            gpu_context.set_params(sim_params);
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
        
        // DISABLED: Don't regenerate constraints automatically
        // Constraints should only be enabled through control center
        info!("Skipping automatic constraint regeneration (prevents bouncing)");
        
        info!("Regenerated semantic constraints: {} total constraints", 
              self.constraint_set.constraints.len());
        Ok(())
    }
}

impl Handler<SetAdvancedGPUContext> for GraphServiceActor {
    type Result = ();
    
    fn handle(&mut self, msg: SetAdvancedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        self.advanced_gpu_context = Some(msg.context);
        self.gpu_init_in_progress = false; // Reset the flag
        info!("Advanced GPU context successfully initialized and set");
    }
}

impl Handler<ResetGPUInitFlag> for GraphServiceActor {
    type Result = ();
    
    fn handle(&mut self, _msg: ResetGPUInitFlag, _ctx: &mut Self::Context) -> Self::Result {
        self.gpu_init_in_progress = false;
        debug!("GPU initialization flag reset");
    }
}
