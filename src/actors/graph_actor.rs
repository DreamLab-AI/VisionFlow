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

// Advanced physics and AI modules
use crate::models::constraints::{ConstraintSet, Constraint, AdvancedParams};
use crate::services::semantic_analyzer::{SemanticAnalyzer, SemanticFeatures};
use crate::services::edge_generation::{AdvancedEdgeGenerator, EdgeGenerationConfig};
use crate::utils::advanced_gpu_compute::{AdvancedGPUContext, EnhancedBinaryNodeData};
use crate::physics::stress_majorization::{StressMajorizationSolver, StressMajorizationConfig};

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
    advanced_gpu_context: Option<AdvancedGPUContext>,
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
}

impl GraphServiceActor {
    pub fn new(
        client_manager: Addr<ClientManagerActor>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) -> Self {
        let advanced_params = AdvancedParams::default();
        let semantic_analyzer = SemanticAnalyzer::new(
            crate::services::semantic_analyzer::SemanticAnalyzerConfig::default()
        );
        let edge_generator = AdvancedEdgeGenerator::new(
            EdgeGenerationConfig::default()
        );
        let stress_solver = StressMajorizationSolver::from_advanced_params(&advanced_params);
        
        Self {
            graph_data: Arc::new(GraphData::new()), // Changed to Arc::new
            node_map: HashMap::new(),
            gpu_compute_addr,
            client_manager,
            simulation_running: AtomicBool::new(false),
            shutdown_complete: Arc::new(AtomicBool::new(false)),
            next_node_id: AtomicU32::new(1),
            bots_graph_data: GraphData::new(),
            simulation_params: SimulationParams::default(), // Initialize with default physics
            
            // Initialize advanced components
            advanced_gpu_context: None,
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
        
        // Phase 3: Generate initial constraints based on semantic clustering
        info!("Phase 3: Generating initial constraints based on semantic features");
        self.generate_initial_semantic_constraints(&new_graph_data);
        
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
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data_for_gpu });
            info!("Sent initial graph data to legacy GPU compute actor");
        }
        
        Ok(())
    }
    
    /// Helper methods for the hybrid solver orchestration
    
    fn prepare_enhanced_nodes(&self) -> Vec<EnhancedBinaryNodeData> {
        self.graph_data.nodes.iter().enumerate().map(|(idx, node)| {
            let basic_data = BinaryNodeData {
                position: node.data.position.clone(),
                velocity: node.data.velocity.clone(),
                mass: node.data.mass,
                flags: node.data.flags,
                padding: [0, 0],
            };
            
            let mut enhanced = EnhancedBinaryNodeData::from(basic_data);
            
            // Enhance with semantic features if available
            if let Some(features) = self.semantic_features_cache.get(&node.metadata_id) {
                enhanced.importance_score = features.importance_score;
                enhanced.semantic_weight = features.domains.len() as f32 / 5.0; // Normalize
                enhanced.temporal_weight = features.temporal.modification_frequency;
                enhanced.structural_weight = features.structural.complexity_score / 10.0; // Normalize
                
                // Set node type based on primary domain
                enhanced.node_type = if !features.domains.is_empty() {
                    match features.domains[0] {
                        crate::services::semantic_analyzer::KnowledgeDomain::ComputerScience => 1,
                        crate::services::semantic_analyzer::KnowledgeDomain::Mathematics => 2,
                        crate::services::semantic_analyzer::KnowledgeDomain::DataScience => 3,
                        crate::services::semantic_analyzer::KnowledgeDomain::WebDevelopment => 4,
                        _ => 0,
                    }
                } else { 0 };
            }
            
            enhanced
        }).collect()
    }
    
    fn execute_stress_majorization_step(&mut self) {
        if self.graph_data.nodes.len() < 3 {
            return; // Skip for very small graphs
        }
        
        let mut graph_data_clone = (*self.graph_data).clone();
        
        match self.stress_solver.optimize(&mut graph_data_clone, &self.constraint_set) {
            Ok(result) => {
                if result.converged || result.final_stress < f32::INFINITY {
                    // Apply optimized positions back to main graph
                    let graph_data_mut = Arc::make_mut(&mut self.graph_data);
                    for (i, node) in graph_data_mut.nodes.iter_mut().enumerate() {
                        if let Some(optimized_node) = graph_data_clone.nodes.get(i) {
                            node.data.x = optimized_node.data.x;
                            node.data.y = optimized_node.data.y;
                            node.data.z = optimized_node.data.z;
                        }
                    }
                    
                    // Update node_map as well
                    for node in &graph_data_mut.nodes {
                        if let Some(node_in_map) = self.node_map.get_mut(&node.id) {
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
    }
    
    fn generate_initial_semantic_constraints(&mut self, graph_data: &GraphData) {
        // Generate boundary constraints
        let boundary = Constraint::boundary(
            graph_data.nodes.iter().map(|n| n.id).collect(),
            -1000.0, 1000.0,
            -1000.0, 1000.0,
            -1000.0, 1000.0,
        );
        self.constraint_set.add_to_group("boundary", boundary);
        
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
        for (depth, node_ids) in depth_layers {
            if node_ids.len() >= 2 {
                let z_position = -(depth as f32) * self.advanced_params.layer_separation;
                let constraint = Constraint {
                    kind: crate::models::constraints::ConstraintKind::AlignmentDepth,
                    node_indices: node_ids,
                    params: vec![z_position],
                    weight: 0.8,
                    active: true,
                };
                self.constraint_set.add_to_group("hierarchical_layers", constraint);
            }
        }
        
        info!("Generated hierarchical layer constraints for {} depths", depth_layers.len());
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
        
        debug!("Updated positions for {} nodes", updated_count);
    }

    fn start_simulation_loop(&mut self, ctx: &mut Context<Self>) {
        if self.simulation_running.load(Ordering::SeqCst) {
            warn!("Simulation already running");
            return;
        }

        self.simulation_running.store(true, Ordering::SeqCst);
        info!("Starting physics simulation loop");

        // Start the simulation interval
        ctx.run_interval(Duration::from_millis(16), |actor, ctx| {
            if !actor.simulation_running.load(Ordering::SeqCst) {
                return;
            }

            actor.run_simulation_step(ctx);
        });
    }

    fn run_simulation_step(&mut self, ctx: &mut Context<Self>) {
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
        if self.advanced_gpu_context.is_none() && !self.graph_data.nodes.is_empty() {
            // Attempt to initialize advanced GPU context
            let graph_data_clone = (*self.graph_data).clone();
            let self_addr = ctx.address();
            let simulation_params = self.simulation_params.clone();
            let advanced_params = self.advanced_params.clone();
            
            actix::spawn(async move {
                match AdvancedGPUContext::new(
                    graph_data_clone.nodes.len() as u32,
                    graph_data_clone.edges.len() as u32,
                    simulation_params,
                    advanced_params,
                ).await {
                    Ok(context) => {
                        self_addr.do_send(SetAdvancedGPUContext { context });
                    }
                    Err(e) => {
                        warn!("Failed to initialize advanced GPU context: {}", e);
                    }
                }
            });
        }
        
        // Use advanced GPU compute if available, otherwise fallback to legacy
        if let Some(ref mut advanced_gpu) = self.advanced_gpu_context {
            self.run_advanced_gpu_step(advanced_gpu, ctx);
        } else if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            self.run_legacy_gpu_step(gpu_compute_addr, ctx);
        } else {
            warn!("No GPU compute context available for physics simulation");
        }
    }
    
    fn run_advanced_gpu_step(&mut self, gpu_context: &mut AdvancedGPUContext, _ctx: &mut Context<Self>) {
        // Convert nodes to enhanced format with semantic features
        let enhanced_nodes = self.prepare_enhanced_nodes();
        
        // Update GPU with enhanced node data
        if let Err(e) = gpu_context.update_node_data(enhanced_nodes) {
            error!("Failed to update advanced GPU node data: {}", e);
            return;
        }
        
        // Execute GPU physics step with constraints
        let active_constraints: Vec<Constraint> = self.constraint_set.active_constraints()
            .into_iter().cloned().collect();
        
        if let Err(e) = gpu_context.step_with_constraints(&active_constraints) {
            error!("Advanced GPU physics step failed: {}", e);
            return;
        }
        
        // Get results and update local state
        match gpu_context.get_legacy_node_data() {
            Ok(node_data) => {
                let node_ids: Vec<u32> = self.graph_data.nodes.iter().map(|n| n.id).collect();
                let mut positions = Vec::new();
                
                for (index, data) in node_data.iter().enumerate() {
                    if let Some(node_id) = node_ids.get(index) {
                        positions.push((*node_id, data.clone()));
                    }
                }
                
                if !positions.is_empty() {
                    // Update local positions
                    self.update_node_positions(positions.clone());
                    
                    // Broadcast to clients
                    let binary_data = binary_protocol::encode_node_data(&positions);
                    self.client_manager.do_send(BroadcastNodePositions { 
                        positions: binary_data 
                    });
                }
            }
            Err(e) => {
                error!("Failed to get advanced GPU results: {}", e);
            }
        }
        
        // Log progress periodically
        if gpu_context.iteration_count % 60 == 0 {
            trace!("Advanced physics step completed (iteration {}, {} constraints active)", 
                  gpu_context.iteration_count, active_constraints.len());
        }
    }
    
    fn run_legacy_gpu_step(&mut self, gpu_compute_addr: &Addr<GPUComputeActor>, ctx: &mut Context<Self>) {
        let node_count = self.graph_data.nodes.len();
        
        // Only log periodically to avoid spam
        static mut LOG_COUNTER: u32 = 0;
        unsafe {
            if LOG_COUNTER % 60 == 0 {
                info!("GraphServiceActor: Fallback to legacy GPU for {} nodes", node_count);
            }
            LOG_COUNTER += 1;
        }
        
        // Send graph data to GPU
        let graph_data_for_gpu = crate::models::graph::GraphData {
            nodes: self.graph_data.nodes.clone(),
            edges: self.graph_data.edges.clone(),
            metadata: self.graph_data.metadata.clone(),
            id_to_metadata: self.graph_data.id_to_metadata.clone(),
        };
        
        gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data_for_gpu });
        gpu_compute_addr.do_send(ComputeForces);
        
        // Handle results asynchronously 
        let gpu_addr = gpu_compute_addr.clone();
        let client_manager = self.client_manager.clone();
        let node_ids: Vec<u32> = self.graph_data.nodes.iter().map(|n| n.id).collect();
        let self_addr = ctx.address();
        
        actix::spawn(async move {
            tokio::time::sleep(Duration::from_millis(5)).await;
            
            match gpu_addr.send(GetNodeData).await {
                Ok(Ok(node_data)) => {
                    let mut positions = Vec::new();
                    for (index, data) in node_data.iter().enumerate() {
                        if let Some(node_id) = node_ids.get(index) {
                            positions.push((*node_id, data.clone()));
                        }
                    }
                    
                    if !positions.is_empty() {
                        self_addr.do_send(UpdateNodePositions { 
                            positions: positions.clone() 
                        });
                        
                        let binary_data = binary_protocol::encode_node_data(&positions);
                        client_manager.do_send(BroadcastNodePositions { 
                            positions: binary_data 
                        });
                    }
                }
                Ok(Err(e)) => warn!("Failed to get legacy GPU node data: {}", e),
                Err(e) => warn!("Failed to communicate with legacy GPU actor: {}", e),
            }
        });
    }

    // CPU physics removed - all physics now handled by GPU compute actor
    // BREADCRUMB: CPU fallback was removed as GPU is always available in our architecture

    /// Update advanced physics parameters
    pub fn update_advanced_physics_params(&mut self, params: AdvancedParams) {
        self.advanced_params = params.clone();
        self.stress_solver = StressMajorizationSolver::from_advanced_params(&params);
        
        // Update advanced GPU context if available
        if let Some(ref mut gpu_context) = self.advanced_gpu_context {
            gpu_context.update_advanced_params(params);
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
        source_type: &crate::services::claude_flow::AgentType,
        target_type: &crate::services::claude_flow::AgentType,
        source_active_tasks: u32,
        target_active_tasks: u32,
        source_success_rate: f32,
        target_success_rate: f32,
    ) -> f32 {
        // Base communication intensity based on agent type relationships
        let base_intensity = match (source_type, target_type) {
            // Coordinator has high communication with all agent types
            (crate::services::claude_flow::AgentType::Coordinator, _) |
            (_, crate::services::claude_flow::AgentType::Coordinator) => 0.9,
            
            // High collaboration pairs
            (crate::services::claude_flow::AgentType::Coder, crate::services::claude_flow::AgentType::Tester) |
            (crate::services::claude_flow::AgentType::Tester, crate::services::claude_flow::AgentType::Coder) => 0.8,
            
            (crate::services::claude_flow::AgentType::Researcher, crate::services::claude_flow::AgentType::Analyst) |
            (crate::services::claude_flow::AgentType::Analyst, crate::services::claude_flow::AgentType::Researcher) => 0.7,
            
            (crate::services::claude_flow::AgentType::Architect, crate::services::claude_flow::AgentType::Coder) |
            (crate::services::claude_flow::AgentType::Coder, crate::services::claude_flow::AgentType::Architect) => 0.7,
            
            // Medium collaboration pairs
            (crate::services::claude_flow::AgentType::Architect, crate::services::claude_flow::AgentType::Analyst) |
            (crate::services::claude_flow::AgentType::Analyst, crate::services::claude_flow::AgentType::Architect) => 0.6,
            
            (crate::services::claude_flow::AgentType::Reviewer, crate::services::claude_flow::AgentType::Coder) |
            (crate::services::claude_flow::AgentType::Coder, crate::services::claude_flow::AgentType::Reviewer) => 0.6,
            
            (crate::services::claude_flow::AgentType::Optimizer, crate::services::claude_flow::AgentType::Analyst) |
            (crate::services::claude_flow::AgentType::Analyst, crate::services::claude_flow::AgentType::Optimizer) => 0.6,
            
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
                crate::services::claude_flow::AgentType::Coordinator => "#FF6B6B".to_string(),
                crate::services::claude_flow::AgentType::Researcher => "#4ECDC4".to_string(),
                crate::services::claude_flow::AgentType::Coder => "#45B7D1".to_string(),
                crate::services::claude_flow::AgentType::Analyst => "#FFA07A".to_string(),
                crate::services::claude_flow::AgentType::Architect => "#98D8C8".to_string(),
                crate::services::claude_flow::AgentType::Tester => "#F7DC6F".to_string(),
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
                        source_agent.success_rate as f32,
                        target_agent.success_rate as f32,
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
                    "tasksCompleted": agent.performance_metrics.tasks_completed,
                    "successRate": agent.performance_metrics.success_rate,
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
                "completedTasks": msg.agents.iter().map(|a| a.performance_metrics.tasks_completed).sum::<u32>(),
                "avgSuccessRate": msg.agents.iter().map(|a| a.performance_metrics.success_rate).sum::<f64>() / msg.agents.len().max(1) as f64,
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
        info!("GraphServiceActor updating physics simulation parameters");
        self.simulation_params = msg.params.clone();
        
        // Also update GPU compute actor if available
        if let Some(ref gpu_addr) = self.gpu_compute_addr {
            gpu_addr.do_send(msg);
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
            gpu_context.update_advanced_params(msg.params);
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
        
        // Regenerate based on current graph state
        self.generate_initial_semantic_constraints(&(*self.graph_data).clone());
        self.update_dynamic_constraints();
        
        info!("Regenerated semantic constraints: {} total constraints", 
              self.constraint_set.constraints.len());
        Ok(())
    }
}

impl Handler<SetAdvancedGPUContext> for GraphServiceActor {
    type Result = ();
    
    fn handle(&mut self, msg: SetAdvancedGPUContext, _ctx: &mut Self::Context) -> Self::Result {
        self.advanced_gpu_context = Some(msg.context);
        info!("Advanced GPU context successfully initialized and set");
    }
}
