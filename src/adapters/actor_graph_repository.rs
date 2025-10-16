// ActorGraphRepository - Adapter wrapping GraphStateActor
// Implements GraphRepository port for hexagonal architecture
// Future: Add #[derive(HexAdapter)] when Hexser available

use async_trait::async_trait;
use actix::Addr;
use std::sync::Arc;
use std::collections::HashSet;

use crate::ports::graph_repository::{GraphRepository, Result};
use crate::actors::graph_state_actor::GraphStateActor;
use crate::actors::messages::{
    GetGraphData, AddNode, AddEdge, RemoveNode, UpdateNodePositions,
    GetNodeMap,
};
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::utils::socket_flow_messages::BinaryNodeData;

/// Adapter that wraps GraphStateActor to implement GraphRepository trait
pub struct ActorGraphRepository {
    graph_state_actor: Addr<GraphStateActor>,
    dirty_nodes: Arc<std::sync::RwLock<HashSet<u32>>>,
    version: Arc<std::sync::atomic::AtomicU64>,
}

impl ActorGraphRepository {
    /// Create new adapter wrapping a GraphStateActor address
    pub fn new(graph_state_actor: Addr<GraphStateActor>) -> Self {
        Self {
            graph_state_actor,
            dirty_nodes: Arc::new(std::sync::RwLock::new(HashSet::new())),
            version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        self.graph_state_actor
            .send(GetGraphData)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))?
    }

    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>> {
        let mut node_ids = Vec::with_capacity(nodes.len());

        for node in nodes {
            let node_id = node.id;
            self.graph_state_actor
                .send(AddNode { node })
                .await
                .map_err(|e| format!("Actor mailbox error: {}", e))??;

            node_ids.push(node_id);

            // Track dirty node
            if let Ok(mut dirty) = self.dirty_nodes.write() {
                dirty.insert(node_id);
            }
        }

        // Increment version
        self.version.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(node_ids)
    }

    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>> {
        let mut edge_ids = Vec::with_capacity(edges.len());

        for edge in edges {
            let edge_id = edge.id.clone();
            self.graph_state_actor
                .send(AddEdge { edge })
                .await
                .map_err(|e| format!("Actor mailbox error: {}", e))??;

            edge_ids.push(edge_id);
        }

        // Increment version
        self.version.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(edge_ids)
    }

    async fn update_positions(&self, updates: Vec<(u32, BinaryNodeData)>) -> Result<()> {
        self.graph_state_actor
            .send(UpdateNodePositions { positions: updates.clone() })
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))??;

        // Track dirty nodes
        if let Ok(mut dirty) = self.dirty_nodes.write() {
            for (node_id, _) in updates {
                dirty.insert(node_id);
            }
        }

        Ok(())
    }

    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>> {
        self.dirty_nodes
            .read()
            .map(|set| set.clone())
            .map_err(|e| format!("Lock poisoned: {}", e))
    }

    async fn clear_dirty_nodes(&self) -> Result<()> {
        self.dirty_nodes
            .write()
            .map(|mut set| set.clear())
            .map_err(|e| format!("Lock poisoned: {}", e))
    }

    async fn get_version(&self) -> Result<u64> {
        Ok(self.version.load(std::sync::atomic::Ordering::SeqCst))
    }

    async fn get_nodes(&self, node_ids: Vec<u32>) -> Result<Vec<Node>> {
        let node_map = self.graph_state_actor
            .send(GetNodeMap)
            .await
            .map_err(|e| format!("Actor mailbox error: {}", e))??;

        let nodes = node_ids
            .into_iter()
            .filter_map(|id| node_map.get(&id).cloned())
            .collect();

        Ok(nodes)
    }

    async fn remove_nodes(&self, node_ids: Vec<u32>) -> Result<()> {
        for node_id in node_ids {
            self.graph_state_actor
                .send(RemoveNode { node_id })
                .await
                .map_err(|e| format!("Actor mailbox error: {}", e))??;
        }

        // Increment version
        self.version.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        // Get all nodes and remove them
        let graph = self.get_graph().await?;
        let node_ids: Vec<u32> = graph.nodes.iter().map(|n| n.id).collect();
        self.remove_nodes(node_ids).await?;

        // Clear dirty tracking
        self.clear_dirty_nodes().await?;

        Ok(())
    }
}
