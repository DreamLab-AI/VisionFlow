//! GraphStateActor - Pure state management for graph data
//!
//! This actor manages the canonical graph state with thread-safe access
//! and change tracking. It provides no physics computation or external
//! actor dependencies - only pure state operations.

use actix::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::utils::socket_flow_messages::BinaryNodeData;
use crate::models::metadata::MetadataStore;

// Import legacy messages for compatibility
use crate::actors::messages::{
    GetGraphData as LegacyGetGraphData,
    AddNode,
    AddNodesFromMetadata,
    BuildGraphFromMetadata,
    UpdateGraphData,
    InitializeGPUConnection,
    GetNodeMap,
    GetPhysicsState,
    GetAutoBalanceNotifications,
    UpdateBotsGraph,
    GetBotsGraphData,
    UpdateSimulationParams,
    ComputeShortestPaths,
    AddEdge as LegacyAddEdge,
    RemoveNode,
    UpdateNodePositions as LegacyUpdateNodePositions,
    RequestPositionSnapshot,
    PositionSnapshot,
    SimulationStep,
};
use crate::actors::graph_actor::{PhysicsState, AutoBalanceNotification};
use crate::models::simulation_params::SimulationParams;

// ============================================================================
// Message Definitions
// ============================================================================

#[derive(Message)]
#[rtype(result = "Result<Arc<GraphData>, String>")]
pub struct GetGraphData;

#[derive(Message)]
#[rtype(result = "Result<Vec<u32>, String>")]
pub struct AddNodes {
    pub nodes: Vec<Node>,
}

#[derive(Message)]
#[rtype(result = "Result<Vec<String>, String>")]
pub struct AddEdges {
    pub edges: Vec<Edge>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct UpdateNodePositions {
    pub updates: Vec<(u32, BinaryNodeData)>,
}

#[derive(Message)]
#[rtype(result = "Result<HashSet<u32>, String>")]
pub struct GetDirtyNodes;

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct ClearDirtyNodes;

#[derive(Message)]
#[rtype(result = "u64")]
pub struct GetVersion;

#[derive(Message)]
#[rtype(result = "Result<Vec<Node>, String>")]
pub struct GetNodes {
    pub node_ids: Vec<u32>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct RemoveNodes {
    pub node_ids: Vec<u32>,
}

#[derive(Message)]
#[rtype(result = "Result<(), String>")]
pub struct Clear;

// ============================================================================
// Actor Definition
// ============================================================================

pub struct GraphStateActor {
    /// Thread-safe graph data store
    graph_data: Arc<RwLock<GraphData>>,

    /// Fast node lookup by ID
    node_index: HashMap<u32, usize>,

    /// Tracks nodes with changes since last clear
    dirty_nodes: HashSet<u32>,

    /// Version counter for optimistic locking
    version: u64,
}

impl GraphStateActor {
    pub fn new() -> Self {
        Self {
            graph_data: Arc::new(RwLock::new(GraphData::new())),
            node_index: HashMap::new(),
            dirty_nodes: HashSet::new(),
            version: 0,
        }
    }

    /// Increment version and return new value
    fn bump_version(&mut self) -> u64 {
        self.version = self.version.wrapping_add(1);
        self.version
    }

    /// Mark node as dirty
    fn mark_dirty(&mut self, node_id: u32) {
        self.dirty_nodes.insert(node_id);
    }

    /// Rebuild node index from current graph data
    fn rebuild_index(&mut self) -> Result<(), String> {
        self.node_index.clear();

        let graph = self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        for (idx, node) in graph.nodes.iter().enumerate() {
            self.node_index.insert(node.id, idx);
        }

        Ok(())
    }
}

impl Default for GraphStateActor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor for GraphStateActor {
    type Context = Context<Self>;

    fn started(&mut self, _ctx: &mut Self::Context) {
        log::info!("GraphStateActor started");
    }

    fn stopped(&mut self, _ctx: &mut Self::Context) {
        log::info!("GraphStateActor stopped");
    }
}

// ============================================================================
// Message Handlers
// ============================================================================

impl Handler<GetGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(Arc::new(self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?
            .clone()))
    }
}

impl Handler<AddNodes> for GraphStateActor {
    type Result = Result<Vec<u32>, String>;

    fn handle(&mut self, msg: AddNodes, _ctx: &mut Context<Self>) -> Self::Result {
        let mut added_ids = Vec::with_capacity(msg.nodes.len());

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for node in msg.nodes {
                let node_id = node.id;

                // Check if node already exists
                if self.node_index.contains_key(&node_id) {
                    return Err(format!("Node with ID {} already exists", node_id));
                }

                // Add to graph
                let idx = graph.nodes.len();
                graph.nodes.push(node);

                // Update index
                self.node_index.insert(node_id, idx);

                added_ids.push(node_id);
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for &node_id in &added_ids {
            self.mark_dirty(node_id);
        }
        self.bump_version();

        Ok(added_ids)
    }
}

impl Handler<AddEdges> for GraphStateActor {
    type Result = Result<Vec<String>, String>;

    fn handle(&mut self, msg: AddEdges, _ctx: &mut Context<Self>) -> Self::Result {
        let mut added_ids = Vec::with_capacity(msg.edges.len());

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for edge in msg.edges {
                // Validate source and target nodes exist
                if !self.node_index.contains_key(&edge.source) {
                    return Err(format!("Source node {} does not exist", edge.source));
                }
                if !self.node_index.contains_key(&edge.target) {
                    return Err(format!("Target node {} does not exist", edge.target));
                }

                let edge_id = edge.id.clone();
                graph.edges.push(edge);

                added_ids.push(edge_id);
            }
        } // Drop write lock

        self.bump_version();

        Ok(added_ids)
    }
}

impl Handler<UpdateNodePositions> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateNodePositions, _ctx: &mut Context<Self>) -> Self::Result {
        let mut dirty_nodes = Vec::new();

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for (node_id, binary_data) in msg.updates {
                // Find node index
                let idx = self.node_index.get(&node_id)
                    .ok_or_else(|| format!("Node {} not found", node_id))?;

                // Update node data
                if let Some(node) = graph.nodes.get_mut(*idx) {
                    node.data = binary_data;
                    dirty_nodes.push(node_id);
                } else {
                    return Err(format!("Node index {} out of bounds", idx));
                }
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for node_id in dirty_nodes {
            self.mark_dirty(node_id);
        }
        self.bump_version();

        Ok(())
    }
}

impl Handler<GetDirtyNodes> for GraphStateActor {
    type Result = Result<HashSet<u32>, String>;

    fn handle(&mut self, _msg: GetDirtyNodes, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(self.dirty_nodes.clone())
    }
}

impl Handler<ClearDirtyNodes> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: ClearDirtyNodes, _ctx: &mut Context<Self>) -> Self::Result {
        self.dirty_nodes.clear();
        Ok(())
    }
}

impl Handler<GetVersion> for GraphStateActor {
    type Result = u64;

    fn handle(&mut self, _msg: GetVersion, _ctx: &mut Context<Self>) -> Self::Result {
        self.version
    }
}

impl Handler<GetNodes> for GraphStateActor {
    type Result = Result<Vec<Node>, String>;

    fn handle(&mut self, msg: GetNodes, _ctx: &mut Context<Self>) -> Self::Result {
        let graph = self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        let mut nodes = Vec::with_capacity(msg.node_ids.len());

        for node_id in msg.node_ids {
            let idx = self.node_index.get(&node_id)
                .ok_or_else(|| format!("Node {} not found", node_id))?;

            if let Some(node) = graph.nodes.get(*idx) {
                nodes.push(node.clone());
            } else {
                return Err(format!("Node index {} out of bounds", idx));
            }
        }

        Ok(nodes)
    }
}

impl Handler<RemoveNodes> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNodes, _ctx: &mut Context<Self>) -> Self::Result {
        let mut graph = self.graph_data.write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        // Collect indices to remove
        let mut indices_to_remove: Vec<usize> = Vec::new();
        for node_id in &msg.node_ids {
            if let Some(&idx) = self.node_index.get(node_id) {
                indices_to_remove.push(idx);
            }
        }

        // Sort in reverse order to remove from end first
        indices_to_remove.sort_unstable_by(|a, b| b.cmp(a));

        // Remove nodes
        for idx in indices_to_remove {
            if idx < graph.nodes.len() {
                graph.nodes.remove(idx);
            }
        }

        // Remove associated edges
        let node_ids_set: HashSet<u32> = msg.node_ids.iter().copied().collect();
        graph.edges.retain(|edge| {
            !node_ids_set.contains(&edge.source) && !node_ids_set.contains(&edge.target)
        });

        // Rebuild index after removal
        drop(graph);
        self.rebuild_index()?;

        // Remove from dirty set
        for node_id in &msg.node_ids {
            self.dirty_nodes.remove(node_id);
        }

        self.bump_version();

        Ok(())
    }
}

impl Handler<Clear> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: Clear, _ctx: &mut Context<Self>) -> Self::Result {
        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            graph.nodes.clear();
            graph.edges.clear();
            graph.metadata.clear();
            graph.id_to_metadata.clear();
        } // Drop write lock

        self.node_index.clear();
        self.dirty_nodes.clear();
        self.bump_version();

        Ok(())
    }
}

// ============================================================================
// Legacy Message Handlers for Backward Compatibility
// ============================================================================

impl Handler<LegacyGetGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: LegacyGetGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        Ok(Arc::new(self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?
            .clone()))
    }
}

impl Handler<AddNodesFromMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNodesFromMetadata, _ctx: &mut Context<Self>) -> Self::Result {
        let mut nodes = Vec::new();

        // Convert metadata to nodes
        // MetadataStore is HashMap<String, Metadata> where key is filename
        for (filename, _metadata) in msg.metadata.iter() {
            // Create node with filename as both metadata_id and label
            let mut node = Node::new(filename.clone());
            node.label = filename.clone(); // Set label so it displays in UI
            nodes.push(node);
        }

        // Add nodes via existing handler logic
        if nodes.is_empty() {
            return Ok(());
        }

        let mut added_ids = Vec::new();

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for node in nodes {
                let node_id = node.id;

                // Skip if node already exists
                if self.node_index.contains_key(&node_id) {
                    continue;
                }

                let idx = graph.nodes.len();
                graph.nodes.push(node);
                self.node_index.insert(node_id, idx);
                added_ids.push(node_id);
            }

            // Create edges between new nodes and connect to existing graph
            if !added_ids.is_empty() {
                // Get all existing node IDs before the new additions
                let existing_ids: Vec<u32> = self.node_index.keys()
                    .filter(|id| !added_ids.contains(id))
                    .copied()
                    .collect();

                // Connect new nodes to each other
                for i in 0..added_ids.len() {
                    for j in 1..=2 {
                        if i + j < added_ids.len() {
                            let edge = Edge::new(added_ids[i], added_ids[i + j], 1.0);
                            graph.edges.push(edge);
                        }
                    }
                }

                // Connect new nodes to existing graph (2-3 edges each to existing nodes)
                for &new_id in &added_ids {
                    let num_connections = existing_ids.len().min(3);
                    for i in 0..num_connections {
                        if i < existing_ids.len() {
                            let edge = Edge::new(new_id, existing_ids[i], 1.0);
                            graph.edges.push(edge);
                        }
                    }
                }

                log::info!("Added {} new nodes and created edges to integrate with existing {} nodes",
                          added_ids.len(), existing_ids.len());
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for node_id in added_ids {
            self.mark_dirty(node_id);
        }
        self.bump_version();

        Ok(())
    }
}

impl Handler<BuildGraphFromMetadata> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: BuildGraphFromMetadata, _ctx: &mut Context<Self>) -> Self::Result {
        // BuildGraphFromMetadata clears existing graph and builds from metadata
        // This is different from AddNodesFromMetadata which adds to existing graph

        // Clear existing graph first
        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            graph.nodes.clear();
            graph.edges.clear();
            graph.metadata.clear();
            graph.id_to_metadata.clear();
        } // Drop write lock

        self.node_index.clear();
        self.dirty_nodes.clear();

        // Now add nodes from metadata (reuse AddNodesFromMetadata logic)
        let mut nodes = Vec::new();

        for (filename, _metadata) in msg.metadata.iter() {
            let mut node = Node::new(filename.clone());
            node.label = filename.clone(); // Set label so it displays in UI
            nodes.push(node);
        }

        if nodes.is_empty() {
            return Ok(());
        }

        let mut added_ids = Vec::new();

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for node in nodes {
                let node_id = node.id;
                let idx = graph.nodes.len();
                graph.nodes.push(node);
                self.node_index.insert(node_id, idx);
                added_ids.push(node_id);
            }

            // Create basic edges between nodes (fully connected for visualization)
            // For now, create edges between consecutive nodes for a minimal graph structure
            if added_ids.len() > 1 {
                for i in 0..added_ids.len() {
                    // Connect each node to next 2-3 nodes for a sparse mesh
                    for j in 1..=2 {
                        if i + j < added_ids.len() {
                            let edge = Edge::new(added_ids[i], added_ids[i + j], 1.0);
                            graph.edges.push(edge);
                        }
                    }
                }
                log::info!("Created {} edges between {} nodes", graph.edges.len(), added_ids.len());
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for node_id in added_ids {
            self.mark_dirty(node_id);
        }
        self.bump_version();

        Ok(())
    }
}

impl Handler<UpdateGraphData> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: UpdateGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        // Replace entire graph with new data
        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            // Extract data from Arc
            let new_graph_data = (*msg.graph_data).clone();

            // Replace all graph data
            *graph = new_graph_data;
        } // Drop write lock

        // Rebuild index from new graph
        self.rebuild_index()?;

        // Mark all nodes as dirty
        {
            let graph_read = self.graph_data.read()
                .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

            for node in &graph_read.nodes {
                self.dirty_nodes.insert(node.id);
            }
        }

        self.bump_version();

        Ok(())
    }
}

impl Handler<InitializeGPUConnection> for GraphStateActor {
    type Result = ();

    fn handle(&mut self, _msg: InitializeGPUConnection, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't manage GPU connections
        // This is a no-op for pure state management
        log::debug!("GraphStateActor: GPU connection message received (no-op for pure state)");
    }
}

impl Handler<GetNodeMap> for GraphStateActor {
    type Result = Result<Arc<HashMap<u32, Node>>, String>;

    fn handle(&mut self, _msg: GetNodeMap, _ctx: &mut Context<Self>) -> Self::Result {
        let graph = self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        let mut node_map = HashMap::new();
        for node in &graph.nodes {
            node_map.insert(node.id, node.clone());
        }

        Ok(Arc::new(node_map))
    }
}

impl Handler<GetPhysicsState> for GraphStateActor {
    type Result = Result<PhysicsState, String>;

    fn handle(&mut self, _msg: GetPhysicsState, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't manage physics state
        // Return default state (consult PhysicsOrchestratorActor for actual physics)
        Ok(PhysicsState {
            is_settled: false,
            stable_frame_count: 0,
            kinetic_energy: 0.0,
            current_state: "Not managed by GraphStateActor".to_string(),
        })
    }
}

impl Handler<GetAutoBalanceNotifications> for GraphStateActor {
    type Result = Result<Vec<AutoBalanceNotification>, String>;

    fn handle(&mut self, _msg: GetAutoBalanceNotifications, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't manage auto-balance notifications
        // Return empty list (consult PhysicsOrchestratorActor for actual notifications)
        Ok(Vec::new())
    }
}

impl Handler<AddNode> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: AddNode, _ctx: &mut Context<Self>) -> Self::Result {
        let node_id = msg.node.id;

        // Check if node already exists
        if self.node_index.contains_key(&node_id) {
            return Err(format!("Node with ID {} already exists", node_id));
        }

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            let idx = graph.nodes.len();
            graph.nodes.push(msg.node);
            self.node_index.insert(node_id, idx);
        } // Drop write lock

        self.mark_dirty(node_id);
        self.bump_version();

        Ok(())
    }
}

impl Handler<UpdateBotsGraph> for GraphStateActor {
    type Result = ();

    fn handle(&mut self, msg: UpdateBotsGraph, _ctx: &mut Context<Self>) -> Self::Result {
        let mut added_ids = Vec::new();

        // Add nodes (ignore errors for backward compatibility)
        if let Ok(mut graph) = self.graph_data.write() {
            for agent in &msg.agents {
                // Create node from agent data
                let mut node = Node::new(agent.id.clone());
                node.label = agent.name.clone();
                node.node_type = Some(agent.agent_type.clone());
                node.data.x = agent.x;
                node.data.y = agent.y;
                node.data.z = agent.z;

                let node_id = node.id;

                // Skip if already exists
                if self.node_index.contains_key(&node_id) {
                    continue;
                }

                let idx = graph.nodes.len();
                graph.nodes.push(node);
                self.node_index.insert(node_id, idx);
                added_ids.push(node_id);
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for node_id in added_ids {
            self.mark_dirty(node_id);
        }
        self.bump_version();
    }
}

impl Handler<GetBotsGraphData> for GraphStateActor {
    type Result = Result<Arc<GraphData>, String>;

    fn handle(&mut self, _msg: GetBotsGraphData, _ctx: &mut Context<Self>) -> Self::Result {
        // Same as GetGraphData
        Ok(Arc::new(self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?
            .clone()))
    }
}

impl Handler<UpdateSimulationParams> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: UpdateSimulationParams, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't manage simulation parameters
        // This should be sent to PhysicsOrchestratorActor instead
        log::debug!("GraphStateActor: Received UpdateSimulationParams (no-op, delegate to PhysicsOrchestratorActor)");
        Ok(())
    }
}

impl Handler<ComputeShortestPaths> for GraphStateActor {
    type Result = Result<HashMap<u32, Option<f32>>, String>;

    fn handle(&mut self, _msg: ComputeShortestPaths, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't compute shortest paths
        // This should be sent to SemanticProcessorActor instead
        Err("GraphStateActor doesn't compute paths. Use SemanticProcessorActor instead.".to_string())
    }
}

impl Handler<LegacyAddEdge> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: LegacyAddEdge, _ctx: &mut Context<Self>) -> Self::Result {
        // Validate source and target nodes exist
        if !self.node_index.contains_key(&msg.edge.source) {
            return Err(format!("Source node {} does not exist", msg.edge.source));
        }
        if !self.node_index.contains_key(&msg.edge.target) {
            return Err(format!("Target node {} does not exist", msg.edge.target));
        }

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            graph.edges.push(msg.edge);
        } // Drop write lock

        self.bump_version();

        Ok(())
    }
}

impl Handler<RemoveNode> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: RemoveNode, _ctx: &mut Context<Self>) -> Self::Result {
        // Find node index
        let idx = self.node_index.get(&msg.node_id)
            .ok_or_else(|| format!("Node {} not found", msg.node_id))?;

        let mut graph = self.graph_data.write()
            .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

        // Remove node
        if *idx < graph.nodes.len() {
            graph.nodes.remove(*idx);
        }

        // Remove associated edges
        graph.edges.retain(|edge| {
            edge.source != msg.node_id && edge.target != msg.node_id
        });

        // Drop lock before rebuild
        drop(graph);

        // Rebuild index
        self.rebuild_index()?;

        // Remove from dirty set
        self.dirty_nodes.remove(&msg.node_id);
        self.bump_version();

        Ok(())
    }
}

impl Handler<LegacyUpdateNodePositions> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, msg: LegacyUpdateNodePositions, _ctx: &mut Context<Self>) -> Self::Result {
        let mut dirty_nodes = Vec::new();

        {
            let mut graph = self.graph_data.write()
                .map_err(|e| format!("Failed to acquire write lock: {}", e))?;

            for (node_id, binary_data) in msg.positions {
                // Find node index
                let idx = self.node_index.get(&node_id)
                    .ok_or_else(|| format!("Node {} not found", node_id))?;

                // Update node data
                if let Some(node) = graph.nodes.get_mut(*idx) {
                    node.data = binary_data;
                    dirty_nodes.push(node_id);
                } else {
                    return Err(format!("Node index {} out of bounds", idx));
                }
            }
        } // Drop write lock

        // Mark as dirty and bump version
        for node_id in dirty_nodes {
            self.mark_dirty(node_id);
        }
        self.bump_version();

        Ok(())
    }
}

impl Handler<RequestPositionSnapshot> for GraphStateActor {
    type Result = Result<PositionSnapshot, String>;

    fn handle(&mut self, msg: RequestPositionSnapshot, _ctx: &mut Context<Self>) -> Self::Result {
        let graph = self.graph_data.read()
            .map_err(|e| format!("Failed to acquire read lock: {}", e))?;

        let mut knowledge_nodes = Vec::new();
        let mut agent_nodes = Vec::new();

        if msg.include_knowledge_graph {
            // Include only knowledge nodes (exclude agent, ontology types)
            for node in &graph.nodes {
                let is_agent = node.node_type.as_ref()
                    .map_or(false, |t| t.contains("agent") || t.contains("bot"));
                let is_ontology = node.node_type.as_ref()
                    .map_or(false, |t| t.contains("ontology") || t.contains("owl"));

                if !is_agent && !is_ontology {
                    knowledge_nodes.push((node.id, node.data));
                }
            }
        }

        if msg.include_agent_graph {
            // Filter for agent nodes (node_type contains "agent" or "bot")
            for node in &graph.nodes {
                if let Some(ref node_type) = node.node_type {
                    if node_type.contains("agent") || node_type.contains("bot") {
                        agent_nodes.push((node.id, node.data));
                    }
                }
            }
        }

        Ok(PositionSnapshot {
            knowledge_nodes,
            agent_nodes,
            timestamp: std::time::Instant::now(),
        })
    }
}

impl Handler<SimulationStep> for GraphStateActor {
    type Result = Result<(), String>;

    fn handle(&mut self, _msg: SimulationStep, _ctx: &mut Context<Self>) -> Self::Result {
        // GraphStateActor doesn't run simulation steps
        // This should be sent to PhysicsOrchestratorActor instead
        log::debug!("GraphStateActor: Received SimulationStep (no-op, delegate to PhysicsOrchestratorActor)");
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[actix::test]
    async fn test_add_and_get_nodes() {
        let actor = GraphStateActor::new().start();

        let node1 = Node::new("test1.md".to_string());
        let node2 = Node::new("test2.md".to_string());
        let node1_id = node1.id;
        let node2_id = node2.id;

        // Add nodes
        let result = actor.send(AddNodes {
            nodes: vec![node1, node2],
        }).await;

        assert!(result.is_ok());
        let ids = result.unwrap();
        assert!(ids.is_ok());
        let added_ids = ids.unwrap();
        assert_eq!(added_ids.len(), 2);

        // Get nodes
        let result = actor.send(GetNodes {
            node_ids: vec![node1_id, node2_id],
        }).await;

        assert!(result.is_ok());
        let nodes = result.unwrap();
        assert!(nodes.is_ok());
        let fetched_nodes = nodes.unwrap();
        assert_eq!(fetched_nodes.len(), 2);
    }

    #[actix::test]
    async fn test_update_positions() {
        let actor = GraphStateActor::new().start();

        let node = Node::new("test.md".to_string());
        let node_id = node.id;

        // Add node
        let _ = actor.send(AddNodes {
            nodes: vec![node],
        }).await;

        // Update position
        let new_data = BinaryNodeData {
            node_id,
            x: 10.0,
            y: 20.0,
            z: 30.0,
            vx: 1.0,
            vy: 2.0,
            vz: 3.0,
        };

        let result = actor.send(UpdateNodePositions {
            updates: vec![(node_id, new_data)],
        }).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());

        // Verify position was updated
        let nodes = actor.send(GetNodes {
            node_ids: vec![node_id],
        }).await.unwrap().unwrap();

        assert_eq!(nodes[0].data.x, 10.0);
        assert_eq!(nodes[0].data.y, 20.0);
        assert_eq!(nodes[0].data.z, 30.0);
    }

    #[actix::test]
    async fn test_dirty_tracking() {
        let actor = GraphStateActor::new().start();

        let node = Node::new("test.md".to_string());
        let node_id = node.id;

        // Add node
        let _ = actor.send(AddNodes {
            nodes: vec![node],
        }).await;

        // Check dirty nodes
        let dirty = actor.send(GetDirtyNodes).await.unwrap().unwrap();
        assert!(dirty.contains(&node_id));

        // Clear dirty nodes
        let _ = actor.send(ClearDirtyNodes).await;

        // Verify cleared
        let dirty = actor.send(GetDirtyNodes).await.unwrap().unwrap();
        assert!(dirty.is_empty());
    }

    #[actix::test]
    async fn test_version_tracking() {
        let actor = GraphStateActor::new().start();

        let v1 = actor.send(GetVersion).await.unwrap();
        assert_eq!(v1, 0);

        let node = Node::new("test.md".to_string());
        let _ = actor.send(AddNodes {
            nodes: vec![node],
        }).await;

        let v2 = actor.send(GetVersion).await.unwrap();
        assert_eq!(v2, 1);
    }

    #[actix::test]
    async fn test_remove_nodes() {
        let actor = GraphStateActor::new().start();

        let node = Node::new("test.md".to_string());
        let node_id = node.id;

        // Add node
        let _ = actor.send(AddNodes {
            nodes: vec![node],
        }).await;

        // Remove node
        let result = actor.send(RemoveNodes {
            node_ids: vec![node_id],
        }).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());

        // Verify node was removed
        let result = actor.send(GetNodes {
            node_ids: vec![node_id],
        }).await;

        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    #[actix::test]
    async fn test_clear() {
        let actor = GraphStateActor::new().start();

        let node1 = Node::new("test1.md".to_string());
        let node2 = Node::new("test2.md".to_string());

        // Add nodes
        let _ = actor.send(AddNodes {
            nodes: vec![node1, node2],
        }).await;

        // Clear
        let result = actor.send(Clear).await;
        assert!(result.is_ok());
        assert!(result.unwrap().is_ok());

        // Verify empty
        let dirty = actor.send(GetDirtyNodes).await.unwrap().unwrap();
        assert!(dirty.is_empty());

        let graph = actor.send(GetGraphData).await.unwrap().unwrap();
        let g = graph.read().unwrap();
        assert_eq!(g.nodes.len(), 0);
        assert_eq!(g.edges.len(), 0);
    }

    #[actix::test]
    async fn test_add_edges() {
        let actor = GraphStateActor::new().start();

        let node1 = Node::new("test1.md".to_string());
        let node2 = Node::new("test2.md".to_string());
        let node1_id = node1.id;
        let node2_id = node2.id;

        // Add nodes
        let _ = actor.send(AddNodes {
            nodes: vec![node1, node2],
        }).await;

        // Add edge
        let edge = Edge::new(node1_id, node2_id, 1.0);
        let result = actor.send(AddEdges {
            edges: vec![edge],
        }).await;

        assert!(result.is_ok());
        let ids = result.unwrap();
        assert!(ids.is_ok());
        let added_ids = ids.unwrap();
        assert_eq!(added_ids.len(), 1);

        // Verify edge exists
        let graph = actor.send(GetGraphData).await.unwrap().unwrap();
        let g = graph.read().unwrap();
        assert_eq!(g.edges.len(), 1);
        assert_eq!(g.edges[0].source, node1_id);
        assert_eq!(g.edges[0].target, node2_id);
    }

    #[actix::test]
    async fn test_add_edge_invalid_nodes() {
        let actor = GraphStateActor::new().start();

        // Try to add edge with non-existent nodes
        let edge = Edge::new(999, 1000, 1.0);
        let result = actor.send(AddEdges {
            edges: vec![edge],
        }).await;

        assert!(result.is_ok());
        let ids = result.unwrap();
        assert!(ids.is_err());
    }
}
