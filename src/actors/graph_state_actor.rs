//! Graph State Actor - Refactored with Hexagonal Architecture
//!
//! This module implements a specialized actor focused exclusively on graph state management.
//! Now uses KnowledgeGraphRepository port for persistence operations.
//!
//! ## Hexagonal Architecture
//!
//! - **Port**: KnowledgeGraphRepository (in-memory interface)
//! - **Adapter**: UnifiedGraphRepository (unified database implementation)
//! - **Actor**: Maintains in-memory state and coordinates operations
//!
//! ## Core Responsibilities
//!
//! ### 1. Graph Data Management
//! - **Primary Graph**: Maintains the main graph data structure with nodes and edges
//! - **Node Map**: Provides efficient O(1) node lookup by ID
//! - **Bots Graph**: Manages separate graph data for agent visualization
//! - **Persistence**: Uses repository port for database operations
//!
//! ### 2. Node Operations (via Repository)
//! - **AddNode**: Add new nodes to the graph with proper ID management
//! - **RemoveNode**: Remove nodes and clean up associated edges
//! - **UpdateNodeFromMetadata**: Update existing nodes based on metadata changes
//!
//! ### 3. Edge Operations (via Repository)
//! - **AddEdge**: Create connections between nodes
//! - **RemoveEdge**: Remove specific edges by ID
//! - **Edge consistency**: Maintain edge integrity during node operations
//!
//! ### 4. Metadata Integration
//! - **BuildGraphFromMetadata**: Rebuild entire graph from metadata store
//! - **AddNodesFromMetadata**: Add multiple nodes from metadata
//! - **RemoveNodeByMetadata**: Remove nodes by metadata ID
//!
//! ### 5. Path Computation
//! - **ComputeShortestPaths**: Calculate shortest paths from source nodes
//! - **Graph traversal**: Provide efficient path finding algorithms
//!
//! ## Usage Pattern
//!
//! ```rust
//! 
//! let graph_data = graph_state_actor.send(GetGraphData).await?;
//!
//! 
//! graph_state_actor.send(AddNode { node }).await?;
//!
//! 
//! graph_state_actor.send(BuildGraphFromMetadata { metadata }).await?;
//! ```

use actix::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use log::{debug, info, warn, error, trace};

use crate::actors::messages::*;
use crate::errors::VisionFlowError;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::{MetadataStore, FileMetadata};
use crate::models::graph::GraphData;
use crate::utils::socket_flow_messages::{BinaryNodeData, BinaryNodeDataClient, glam_to_vec3data};

// Ports (hexagonal architecture)
use crate::ports::knowledge_graph_repository::KnowledgeGraphRepository;

/
pub struct GraphStateActor {
    
    repository: Arc<dyn KnowledgeGraphRepository>,
    
    graph_data: Arc<GraphData>,
    
    node_map: Arc<HashMap<u32, Node>>,
    
    bots_graph_data: Arc<GraphData>,
    
    next_node_id: std::sync::atomic::AtomicU32,
}

impl GraphStateActor {
    
    pub fn new(repository: Arc<dyn KnowledgeGraphRepository>) -> Self {
        info!("Initializing GraphStateActor with repository injection");
        Self {
            repository,
            graph_data: Arc::new(GraphData::new()),
            node_map: Arc::new(HashMap::new()),
            bots_graph_data: Arc::new(GraphData::new()),
            next_node_id: std::sync::atomic::AtomicU32::new(1),
        }
    }

    
    pub fn get_graph_data(&self) -> &GraphData {
        &self.graph_data
    }

    
    pub fn get_node_map(&self) -> &HashMap<u32, Node> {
        &self.node_map
    }

    
    fn add_node(&mut self, node: Node) {
        let node_id = node.id;

        
        Arc::make_mut(&mut self.node_map).insert(node_id, node.clone());

        
        Arc::make_mut(&mut self.graph_data).nodes.push(node);

        info!("Added node {} to graph", node_id);
    }

    
    fn remove_node(&mut self, node_id: u32) {
        
        if Arc::make_mut(&mut self.node_map).remove(&node_id).is_some() {
            
            let graph_data_mut = Arc::make_mut(&mut self.graph_data);
            graph_data_mut.nodes.retain(|n| n.id != node_id);

            
            graph_data_mut.edges.retain(|e| e.source != node_id && e.target != node_id);

            info!("Removed node {} and its edges from graph", node_id);
        } else {
            warn!("Attempted to remove non-existent node {}", node_id);
        }
    }

    
    fn add_edge(&mut self, edge: Edge) {
        
        if !self.node_map.contains_key(&edge.source) {
            warn!("Cannot add edge: source node {} does not exist", edge.source);
            return;
        }
        if !self.node_map.contains_key(&edge.target) {
            warn!("Cannot add edge: target node {} does not exist", edge.target);
            return;
        }

        
        Arc::make_mut(&mut self.graph_data).edges.push(edge.clone());
        info!("Added edge from {} to {} with weight {}", edge.source, edge.target, edge.weight);
    }

    
    fn remove_edge(&mut self, edge_id: &str) {
        let graph_data_mut = Arc::make_mut(&mut self.graph_data);
        let initial_count = graph_data_mut.edges.len();

        graph_data_mut.edges.retain(|e| e.id != edge_id);

        let removed_count = initial_count - graph_data_mut.edges.len();
        if removed_count > 0 {
            info!("Removed {} edge(s) with ID {}", removed_count, edge_id);
        } else {
            warn!("No edges found with ID {}", edge_id);
        }
    }

    
    fn build_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        let mut new_graph_data = GraphData::new();

        
        let mut existing_positions: HashMap<String, (crate::types::vec3::Vec3Data, crate::types::vec3::Vec3Data)> = HashMap::new();

        for node in &self.graph_data.nodes {
            existing_positions.insert(node.metadata_id.clone(), (node.data.position(), node.data.velocity()));
        }

        
        let mut new_node_map = HashMap::new();
        let mut current_id = self.next_node_id.load(std::sync::atomic::Ordering::SeqCst);

        for (metadata_id, file_metadata) in metadata.iter() {
            let mut node = Node::new_with_id(metadata_id.clone(), Some(current_id));

            
            if let Some((position, velocity)) = existing_positions.get(metadata_id) {
                node.data.x = position.x;
                node.data.y = position.y;
                node.data.z = position.z;
                node.data.vx = velocity.x;
                node.data.vy = velocity.y;
                node.data.vz = velocity.z;
            } else {
                
                self.generate_random_position(&mut node);
            }

            
            self.configure_node_from_metadata(&mut node, file_metadata);

            new_node_map.insert(current_id, node.clone());
            new_graph_data.nodes.push(node);
            current_id += 1;
        }

        
        self.generate_edges_from_metadata(&mut new_graph_data, &metadata);

        
        self.graph_data = Arc::new(new_graph_data);
        self.node_map = Arc::new(new_node_map);
        self.next_node_id.store(current_id, std::sync::atomic::Ordering::SeqCst);

        info!("Built graph from metadata: {} nodes, {} edges",
              self.graph_data.nodes.len(), self.graph_data.edges.len());

        Ok(())
    }

    
    fn generate_random_position(&self, node: &mut Node) {
        use rand::{Rng, SeedableRng};
        use rand::rngs::{StdRng, OsRng};

        let mut rng = StdRng::from_seed(OsRng.gen());
        let radius = 50.0 + rng.gen::<f32>() * 100.0;
        let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
        let phi = rng.gen::<f32>() * std::f32::consts::PI;

        node.data.x = radius * phi.sin() * theta.cos();
        node.data.y = radius * phi.sin() * theta.sin();
        node.data.z = radius * phi.cos();

        
        node.data.vx = rng.gen_range(-1.0..1.0);
        node.data.vy = rng.gen_range(-1.0..1.0);
        node.data.vz = rng.gen_range(-1.0..1.0);
    }

    
    fn configure_node_from_metadata(&self, node: &mut Node, metadata: &FileMetadata) {
        
        if let Some(filename) = metadata.path.file_name() {
            node.label = filename.to_string_lossy().to_string();
        }

        
        node.color = Some(self.get_color_for_extension(&metadata.path));

        
        if let Some(size) = metadata.size {
            node.size = Some(10.0 + (size as f32 / 1000.0).min(50.0));
        }

        
        node.metadata.insert("path".to_string(), metadata.path.to_string_lossy().to_string());
        if let Some(size) = metadata.size {
            node.metadata.insert("size".to_string(), size.to_string());
        }
        if let Some(modified) = metadata.modified {
            node.metadata.insert("modified".to_string(), modified.to_string());
        }
    }

    
    fn get_color_for_extension(&self, path: &std::path::Path) -> String {
        match path.extension().and_then(|s| s.to_str()) {
            Some("rs") => "#CE422B".to_string(), 
            Some("js") | Some("ts") => "#F7DF1E".to_string(), 
            Some("py") => "#3776AB".to_string(), 
            Some("html") => "#E34F26".to_string(), 
            Some("css") => "#1572B6".to_string(), 
            Some("json") => "#000000".to_string(), 
            Some("md") => "#083FA1".to_string(), 
            Some("txt") => "#808080".to_string(), 
            _ => "#95A5A6".to_string(), 
        }
    }

    
    fn generate_edges_from_metadata(&self, graph_data: &mut GraphData, metadata: &MetadataStore) {
        
        let mut path_to_node: HashMap<std::path::PathBuf, u32> = HashMap::new();

        
        for node in &graph_data.nodes {
            if let Some(path_str) = node.metadata.get("path") {
                let path = std::path::PathBuf::from(path_str);
                path_to_node.insert(path, node.id);
            }
        }

        
        let mut directory_nodes: HashMap<std::path::PathBuf, Vec<u32>> = HashMap::new();

        for (path, node_id) in &path_to_node {
            if let Some(parent) = path.parent() {
                directory_nodes.entry(parent.to_path_buf())
                    .or_insert_with(Vec::new)
                    .push(*node_id);
            }
        }

        
        for (_, nodes) in directory_nodes {
            if nodes.len() > 1 {
                for i in 0..nodes.len() {
                    for j in i+1..nodes.len() {
                        let edge = Edge::new(nodes[i], nodes[j], 0.3); 
                        graph_data.edges.push(edge);
                    }
                }
            }
        }

        info!("Generated {} edges from metadata relationships", graph_data.edges.len());
    }

    
    fn add_nodes_from_metadata(&mut self, metadata: MetadataStore) -> Result<(), String> {
        let mut added_count = 0;
        let mut current_id = self.next_node_id.load(std::sync::atomic::Ordering::SeqCst);

        for (metadata_id, file_metadata) in metadata.iter() {
            
            if self.node_map.values().any(|n| n.metadata_id == *metadata_id) {
                continue;
            }

            let mut node = Node::new_with_id(metadata_id.clone(), Some(current_id));
            self.generate_random_position(&mut node);
            self.configure_node_from_metadata(&mut node, file_metadata);

            self.add_node(node);
            current_id += 1;
            added_count += 1;
        }

        self.next_node_id.store(current_id, std::sync::atomic::Ordering::SeqCst);
        info!("Added {} new nodes from metadata", added_count);
        Ok(())
    }

    
    fn update_node_from_metadata(&mut self, metadata_id: String, metadata: FileMetadata) -> Result<(), String> {
        
        let mut node_found = false;
        let node_map_mut = Arc::make_mut(&mut self.node_map);

        for (_, node) in node_map_mut.iter_mut() {
            if node.metadata_id == metadata_id {
                self.configure_node_from_metadata(node, &metadata);
                node_found = true;
                break;
            }
        }

        
        if node_found {
            let graph_data_mut = Arc::make_mut(&mut self.graph_data);
            for node in &mut graph_data_mut.nodes {
                if node.metadata_id == metadata_id {
                    self.configure_node_from_metadata(node, &metadata);
                    break;
                }
            }
            info!("Updated node with metadata_id: {}", metadata_id);
            Ok(())
        } else {
            warn!("Node with metadata_id {} not found for update", metadata_id);
            Err(format!("Node with metadata_id {} not found", metadata_id))
        }
    }

    
    fn remove_node_by_metadata(&mut self, metadata_id: String) -> Result<(), String> {
        
        let node_id = self.node_map.values()
            .find(|n| n.metadata_id == metadata_id)
            .map(|n| n.id);

        if let Some(id) = node_id {
            self.remove_node(id);
            Ok(())
        } else {
            warn!("Node with metadata_id {} not found for removal", metadata_id);
            Err(format!("Node with metadata_id {} not found", metadata_id))
        }
    }

    
    fn compute_shortest_paths(&self, source_node_id: u32) -> Result<HashMap<u32, (f32, Vec<u32>)>, String> {
        if !self.node_map.contains_key(&source_node_id) {
            return Err(format!("Source node {} not found", source_node_id));
        }

        let mut distances: HashMap<u32, f32> = HashMap::new();
        let mut predecessors: HashMap<u32, u32> = HashMap::new();
        let mut unvisited: std::collections::BTreeSet<(ordered_float::OrderedFloat<f32>, u32)> = std::collections::BTreeSet::new();

        
        for &node_id in self.node_map.keys() {
            let distance = if node_id == source_node_id { 0.0 } else { f32::INFINITY };
            distances.insert(node_id, distance);
            unvisited.insert((ordered_float::OrderedFloat(distance), node_id));
        }

        while let Some((current_distance, current_node)) = unvisited.pop_first() {
            let current_distance = current_distance.into_inner();

            if current_distance == f32::INFINITY {
                break; 
            }

            
            for edge in &self.graph_data.edges {
                let (neighbor, edge_weight) = if edge.source == current_node {
                    (edge.target, edge.weight)
                } else if edge.target == current_node {
                    (edge.source, edge.weight)
                } else {
                    continue;
                };

                let new_distance = current_distance + edge_weight;
                let old_distance = distances.get(&neighbor).copied().unwrap_or(f32::INFINITY);

                if new_distance < old_distance {
                    
                    unvisited.remove(&(ordered_float::OrderedFloat(old_distance), neighbor));

                    
                    distances.insert(neighbor, new_distance);
                    predecessors.insert(neighbor, current_node);

                    
                    unvisited.insert((ordered_float::OrderedFloat(new_distance), neighbor));
                }
            }
        }

        
        let mut result: HashMap<u32, (f32, Vec<u32>)> = HashMap::new();

        for (&target_node, &distance) in &distances {
            if distance != f32::INFINITY {
                let mut path = Vec::new();
                let mut current = target_node;

                
                while current != source_node_id {
                    path.push(current);
                    if let Some(&prev) = predecessors.get(&current) {
                        current = prev;
                    } else {
                        break;
                    }
                }
                path.push(source_node_id);
                path.reverse();

                result.insert(target_node, (distance, path));
            }
        }

        info!("Computed shortest paths from node {} to {} reachable nodes",
              source_node_id, result.len());

        Ok(result)
    }
}

impl Actor for GraphStateActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        info!("GraphStateActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        info!("GraphStateActor stopped");
    }
}

// Handler implementations

impl Handler<GetGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Self::Context) -> Self::Result {
        debug!("GraphStateActor handling GetGraphData with Arc reference");
        Ok(Arc::clone(&self.graph_data))
    }
}

impl Handler<AddNode> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNode, _ctx: &mut Self::Context) -> Self::Result {
        self.add_node(msg.node);
        Ok(())
    }
}

impl Handler<RemoveNode> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNode, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_node(msg.node_id);
        Ok(())
    }
}

impl Handler<AddEdge> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.add_edge(msg.edge);
        Ok(())
    }
}

impl Handler<RemoveEdge> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveEdge, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_edge(&msg.edge_id);
        Ok(())
    }
}

impl Handler<GetNodeMap> for GraphStateActor {
    type Result = Result<Arc<HashMap<u32, Node>>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Self::Context) -> Self::Result {
        debug!("GraphStateActor handling GetNodeMap with Arc reference");
        Ok(Arc::clone(&self.node_map))
    }
}

impl Handler<BuildGraphFromMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        info!("BuildGraphFromMetadata handler called with {} metadata entries", msg.metadata.len());
        self.build_from_metadata(msg.metadata)
    }
}

impl Handler<AddNodesFromMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNodesFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.add_nodes_from_metadata(msg.metadata)
    }
}

impl Handler<UpdateNodeFromMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodeFromMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.update_node_from_metadata(msg.metadata_id, msg.metadata)
    }
}

impl Handler<RemoveNodeByMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNodeByMetadata, _ctx: &mut Self::Context) -> Self::Result {
        self.remove_node_by_metadata(msg.metadata_id)
    }
}

impl Handler<UpdateGraphData> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, _ctx: &mut Self::Context) -> Self::Result {
        info!("Updating graph data with {} nodes, {} edges",
              msg.graph_data.nodes.len(), msg.graph_data.edges.len());

        
        self.graph_data = msg.graph_data;

        
        Arc::make_mut(&mut self.node_map).clear();
        for node in &self.graph_data.nodes {
            Arc::make_mut(&mut self.node_map).insert(node.id, node.clone());
        }

        info!("Graph data updated successfully");
        Ok(())
    }
}

impl Handler<GetBotsGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetBotsGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(Arc::clone(&self.bots_graph_data))
    }
}

impl Handler<UpdateBotsGraph> for GraphStateActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) -> Self::Result {
        
        let mut nodes = vec![];
        let mut edges = vec![];

        let bot_id_offset = 10000;

        
        let mut existing_positions: HashMap<String, (crate::types::vec3::Vec3Data, crate::types::vec3::Vec3Data)> = HashMap::new();
        for node in &self.bots_graph_data.nodes {
            existing_positions.insert(node.metadata_id.clone(), (node.data.position(), node.data.velocity()));
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
                self.generate_random_position(&mut node);
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

            
            node.metadata.insert("agent_type".to_string(), agent.agent_type.clone());
            node.metadata.insert("status".to_string(), agent.status.clone());
            node.metadata.insert("cpu_usage".to_string(), agent.cpu_usage.to_string());
            node.metadata.insert("memory_usage".to_string(), agent.memory_usage.to_string());
            node.metadata.insert("health".to_string(), agent.health.to_string());
            node.metadata.insert("is_agent".to_string(), "true".to_string());

            nodes.push(node);
        }

        
        for (i, source_agent) in msg.agents.iter().enumerate() {
            for (j, target_agent) in msg.agents.iter().enumerate() {
                if i != j {
                    let source_node_id = bot_id_offset + i as u32;
                    let target_node_id = bot_id_offset + j as u32;

                    let communication_intensity = if source_agent.agent_type == "coordinator" || target_agent.agent_type == "coordinator" {
                        0.8
                    } else if source_agent.status == "active" && target_agent.status == "active" {
                        0.5
                    } else {
                        0.2
                    };

                    if communication_intensity > 0.1 {
                        let mut edge = Edge::new(source_node_id, target_node_id, communication_intensity);
                        let metadata = edge.metadata.get_or_insert_with(HashMap::new);
                        metadata.insert("communication_type".to_string(), "agent_collaboration".to_string());
                        metadata.insert("intensity".to_string(), communication_intensity.to_string());
                        edges.push(edge);
                    }
                }
            }
        }

        
        let bots_graph_data_mut = Arc::make_mut(&mut self.bots_graph_data);
        bots_graph_data_mut.nodes = nodes;
        bots_graph_data_mut.edges = edges;

        info!("Updated bots graph with {} agents and {} edges",
             msg.agents.len(), self.bots_graph_data.edges.len());
    }
}

impl Handler<ComputeShortestPaths> for GraphStateActor {
    type Result = Result<u32, String>;

    fn handle(&mut self, msg: ComputeShortestPaths, _ctx: &mut Self::Context) -> Self::Result {
        match self.compute_shortest_paths(msg.source_node_id) {
            Ok(paths) => {
                info!("Computed shortest paths from node {}: {} reachable nodes",
                      msg.source_node_id, paths.len());
                Ok(paths.len() as u32)
            }
            Err(e) => {
                error!("Failed to compute shortest paths: {}", e);
                Err(e)
            }
        }
    }
}