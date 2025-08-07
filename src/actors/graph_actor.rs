//! Graph Service Actor to replace Arc<RwLock<GraphService>>

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::time::Duration;
use log::{debug, info, warn};
// use actix::fut::WrapFuture; // Unused import
 
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
}

impl GraphServiceActor {
    pub fn new(
        client_manager: Addr<ClientManagerActor>,
        gpu_compute_addr: Option<Addr<GPUComputeActor>>,
    ) -> Self {
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
        let mut new_graph_data = GraphData::new(); // Create a new GraphData instance
        self.node_map.clear(); // Clear node_map separately

        // Build nodes from metadata
        // Assuming metadata is MetadataStore which is HashMap<String, crate::models::metadata::Metadata>
        for (filename_with_ext, file_meta_data) in &metadata {
            let node_id_val = self.next_node_id.fetch_add(1, Ordering::SeqCst);
            let metadata_id_val = filename_with_ext.trim_end_matches(".md").to_string();
            
            let mut node = Node::new_with_id(metadata_id_val.clone(), Some(node_id_val));
            node.label = file_meta_data.file_name.trim_end_matches(".md").to_string();
            node.set_file_size(file_meta_data.file_size as u64);
            node.data.flags = 1;

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
            node.metadata.insert("metadataId".to_string(), metadata_id_val);

            // Add to new_graph_data and self.node_map
            self.node_map.insert(node.id, node.clone());
            new_graph_data.nodes.push(node);
        }

        // Build edges from topic counts
        let mut edge_map: HashMap<(u32, u32), f32> = HashMap::new();
        for (source_filename_ext, source_meta) in &metadata {
            let source_metadata_id = source_filename_ext.trim_end_matches(".md");
            if let Some(source_node) = self.node_map.values().find(|n| n.metadata_id == source_metadata_id) {
                for (target_filename_ext, count) in &source_meta.topic_counts {
                    let target_metadata_id = target_filename_ext.trim_end_matches(".md");
                    if let Some(target_node) = self.node_map.values().find(|n| n.metadata_id == target_metadata_id) {
                        if source_node.id != target_node.id {
                            let edge_key = if source_node.id < target_node.id { (source_node.id, target_node.id) } else { (target_node.id, source_node.id) };
                            *edge_map.entry(edge_key).or_insert(0.0) += *count as f32;
                        }
                    }
                }
            }
        }

        for ((source_id, target_id), weight) in edge_map {
            new_graph_data.edges.push(Edge::new(source_id, target_id, weight));
        }
        
        // Populate metadata in new_graph_data (assuming metadata is MetadataStore)
        new_graph_data.metadata = metadata.clone(); // Clone the entire store

        self.graph_data = Arc::new(new_graph_data); // Replace the old Arc with the new one
        
        info!("Built graph from metadata: {} nodes, {} edges",
              self.graph_data.nodes.len(), self.graph_data.edges.len());
        
        // Send the graph data to GPU compute actor
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            let graph_data_for_gpu = (*self.graph_data).clone();
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data_for_gpu });
            info!("Sent initial graph data to GPU compute actor");
        }
        
        Ok(())
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
        // Use GPU compute actor if available
        if let Some(ref gpu_compute_addr) = self.gpu_compute_addr {
            // Send graph data to GPU
            let graph_data_for_gpu = crate::models::graph::GraphData {
                nodes: self.graph_data.nodes.clone(),
                edges: self.graph_data.edges.clone(),
                metadata: self.graph_data.metadata.clone(),
                id_to_metadata: self.graph_data.id_to_metadata.clone(),
            };
            
            gpu_compute_addr.do_send(UpdateGPUGraphData { graph: graph_data_for_gpu });
            
            // Request computation
            gpu_compute_addr.do_send(ComputeForces);
            
            // Request node data back
            let gpu_addr = gpu_compute_addr.clone();
            let client_manager = self.client_manager.clone();
            let node_ids: Vec<u32> = self.graph_data.nodes.iter().map(|n| n.id).collect();
            
            // Get self address to send position updates back
            let self_addr = ctx.address();
            
            // Spawn async task to get results and broadcast
            actix::spawn(async move {
                // Small delay to let GPU compute
                tokio::time::sleep(Duration::from_millis(5)).await;
                
                // Get the computed positions
                match gpu_addr.send(GetNodeData).await {
                    Ok(Ok(node_data)) => {
                        // Convert to position updates with node IDs
                        let mut positions = Vec::new();
                        for (index, data) in node_data.iter().enumerate() {
                            if let Some(node_id) = node_ids.get(index) {
                                positions.push((*node_id, data.clone()));
                            }
                        }
                        
                        if !positions.is_empty() {
                            // Update local node positions
                            self_addr.do_send(UpdateNodePositions { 
                                positions: positions.clone() 
                            });
                            
                            // Encode and broadcast
                            let binary_data = binary_protocol::encode_node_data(&positions);
                            client_manager.do_send(BroadcastNodePositions { 
                                positions: binary_data 
                            });
                        }
                    }
                    Ok(Err(e)) => {
                        warn!("Failed to get GPU node data: {}", e);
                    }
                    Err(e) => {
                        warn!("Failed to communicate with GPU actor: {}", e);
                    }
                }
            });
        } else {
            // No GPU available, skip this frame
            warn!("No GPU compute actor available for physics simulation");
        }
    }

    fn calculate_layout_cpu(&self) -> Result<Vec<(u32, BinaryNodeData)>, String> {
        // Proper CPU physics simulation using actual physics parameters
        let mut updated_positions = Vec::new();
        let nodes = &self.graph_data.nodes;
        let edges = &self.graph_data.edges;
        
        // Use the actual physics parameters
        let params = &self.simulation_params;
        let dt = params.time_step;
        
        for (i, node) in nodes.iter().enumerate() {
            let mut force_x = 0.0f32;
            let mut force_y = 0.0f32;
            let mut force_z = 0.0f32;
            
            // Current position and velocity
            let pos = &node.data.position;
            let vel = &node.data.velocity;
            
            // Apply repulsion forces from other nodes
            for (j, other) in nodes.iter().enumerate() {
                if i != j {
                    let other_pos = &other.data.position;
                    let dx = pos.x - other_pos.x;
                    let dy = pos.y - other_pos.y;
                    let dz = pos.z - other_pos.z;
                    
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    let dist = dist_sq.sqrt().max(0.1); // Avoid division by zero
                    
                    if dist < params.max_repulsion_distance {
                        // Repulsion force
                        let force_magnitude = params.repulsion / (dist_sq + 1.0);
                        force_x += (dx / dist) * force_magnitude;
                        force_y += (dy / dist) * force_magnitude;
                        force_z += (dz / dist) * force_magnitude;
                    }
                }
            }
            
            // Apply spring forces from edges
            for edge in edges {
                let other_idx = if edge.source == node.id {
                    nodes.iter().position(|n| n.id == edge.target)
                } else if edge.target == node.id {
                    nodes.iter().position(|n| n.id == edge.source)
                } else {
                    None
                };
                
                if let Some(other_idx) = other_idx {
                    if other_idx != i {
                        let other_pos = &nodes[other_idx].data.position;
                        let dx = other_pos.x - pos.x;
                        let dy = other_pos.y - pos.y;
                        let dz = other_pos.z - pos.z;
                        
                        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                        
                        // Spring force
                        let force_magnitude = params.spring_strength * (dist - 50.0); // Rest length of 50
                        force_x += (dx / dist) * force_magnitude * edge.weight;
                        force_y += (dy / dist) * force_magnitude * edge.weight;
                        force_z += (dz / dist) * force_magnitude * edge.weight;
                    }
                }
            }
            
            // Apply boundary forces if enabled
            if params.enable_bounds {
                let bounds = params.viewport_bounds;
                let boundary_force = 10.0;
                
                if pos.x.abs() > bounds * 0.9 {
                    force_x -= pos.x.signum() * boundary_force;
                }
                if pos.y.abs() > bounds * 0.9 {
                    force_y -= pos.y.signum() * boundary_force;
                }
                if pos.z.abs() > bounds * 0.9 {
                    force_z -= pos.z.signum() * boundary_force;
                }
            }
            
            // Update velocity with damping
            let mut new_data = node.data.clone();
            new_data.velocity.x = (vel.x + force_x * dt) * params.damping;
            new_data.velocity.y = (vel.y + force_y * dt) * params.damping;
            new_data.velocity.z = (vel.z + force_z * dt) * params.damping;
            
            // Update position
            new_data.position.x = pos.x + new_data.velocity.x * dt;
            new_data.position.y = pos.y + new_data.velocity.y * dt;
            new_data.position.z = pos.z + new_data.velocity.z * dt;
            
            updated_positions.push((node.id, new_data));
        }
        
        Ok(updated_positions)
    }

    fn encode_node_positions(&self, positions: &[(u32, BinaryNodeData)]) -> Result<Vec<u8>, String> {
        // Now binary_protocol expects (u32, BinaryNodeData) directly
        Ok(binary_protocol::encode_node_data(positions))
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
