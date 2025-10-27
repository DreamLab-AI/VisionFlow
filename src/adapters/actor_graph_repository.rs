//! Actor-based Graph Repository Adapter
//!
//! Implements GraphRepository port using the existing GraphServiceActor.
//! This allows gradual migration - queries use CQRS while actor handles writes.

use actix::Addr;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::actors::graph_actor::{AutoBalanceNotification, GraphServiceActor, PhysicsState};
use crate::actors::messages as actor_msgs;
use crate::errors::VisionFlowError;
use crate::models::constraints::ConstraintSet;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::graph_repository::{
    GraphRepository, GraphRepositoryError, PathfindingParams, PathfindingResult, Result,
};
use glam::Vec3;

/// Actor-based implementation of GraphRepository
///
/// This adapter delegates all operations to the GraphServiceActor,
/// providing a bridge between the CQRS query layer and the existing
/// actor-based architecture.
pub struct ActorGraphRepository {
    actor_addr: Addr<GraphServiceActor>,
}

impl ActorGraphRepository {
    /// Create a new ActorGraphRepository with the given actor address
    pub fn new(actor_addr: Addr<GraphServiceActor>) -> Self {
        Self { actor_addr }
    }
}

#[async_trait]
impl GraphRepository for ActorGraphRepository {
    // === Write Operations (Commands) ===

    /// Add nodes to the graph
    ///
    /// Sends individual AddNode messages for each node.
    /// Returns the IDs of successfully added nodes.
    async fn add_nodes(&self, nodes: Vec<Node>) -> Result<Vec<u32>> {
        let mut added_ids = Vec::with_capacity(nodes.len());

        for node in nodes {
            let node_id = node.id;

            self.actor_addr
                .send(actor_msgs::AddNode { node })
                .await
                .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
                .map_err(GraphRepositoryError::AccessError)?;

            added_ids.push(node_id);
        }

        Ok(added_ids)
    }

    /// Add edges to the graph
    ///
    /// Sends individual AddEdge messages for each edge.
    /// Returns the IDs of successfully added edges.
    async fn add_edges(&self, edges: Vec<Edge>) -> Result<Vec<String>> {
        let mut added_ids = Vec::with_capacity(edges.len());

        for edge in edges {
            let edge_id = edge.id.clone();

            self.actor_addr
                .send(actor_msgs::AddEdge { edge })
                .await
                .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
                .map_err(GraphRepositoryError::AccessError)?;

            added_ids.push(edge_id);
        }

        Ok(added_ids)
    }

    /// Update node positions (from physics simulation)
    ///
    /// Sends batched position updates to the actor.
    /// Note: Converts from simple (f32, f32, f32) tuple to BinaryNodeDataClient
    async fn update_positions(
        &self,
        updates: Vec<(u32, crate::ports::graph_repository::BinaryNodeData)>,
    ) -> Result<()> {
        use crate::types::Vec3Data;
        use crate::utils::socket_flow_messages::BinaryNodeDataClient;

        // Convert from simple tuple to BinaryNodeDataClient format
        let positions: Vec<(u32, BinaryNodeDataClient)> = updates
            .into_iter()
            .map(|(id, (x, y, z))| {
                let pos = Vec3Data { x, y, z };
                let vel = Vec3Data {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
                (id, BinaryNodeDataClient::new(id, pos, vel))
            })
            .collect();

        self.actor_addr
            .send(actor_msgs::UpdateNodePositions { positions })
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Clear dirty node tracking
    ///
    /// Note: No-op until dirty tracking is implemented in the actor.
    async fn clear_dirty_nodes(&self) -> Result<()> {
        // TODO: Implement dirty node clearing in GraphServiceActor
        // For now, this is a no-op
        Ok(())
    }

    // === Read Operations (Queries) - CQRS Pattern ===

    /// Get the current graph state
    ///
    /// Returns an Arc clone of the graph data - no deep copy needed.
    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        self.actor_addr
            .send(actor_msgs::GetGraphData)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get the node map (id -> Node)
    ///
    /// Returns an Arc clone of the node map - no deep copy needed.
    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
        self.actor_addr
            .send(actor_msgs::GetNodeMap)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get current physics state
    ///
    /// Returns settlement/physics state for client optimization.
    async fn get_physics_state(&self) -> Result<PhysicsState> {
        self.actor_addr
            .send(actor_msgs::GetPhysicsState)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get node positions as a vector of (id, position) tuples
    ///
    /// Extracts positions from the node map since there's no dedicated message.
    async fn get_node_positions(&self) -> Result<Vec<(u32, Vec3)>> {
        let node_map = self.get_node_map().await?;

        let positions: Vec<(u32, Vec3)> = node_map
            .iter()
            .map(|(id, node)| (*id, Vec3::new(node.data.x, node.data.y, node.data.z)))
            .collect();

        Ok(positions)
    }

    /// Get the graph data for bots/pathfinding
    ///
    /// Returns an Arc clone - no deep copy needed.
    async fn get_bots_graph(&self) -> Result<Arc<GraphData>> {
        self.actor_addr
            .send(actor_msgs::GetBotsGraphData)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get current constraint set
    ///
    /// Returns the active constraint set from physics simulation.
    async fn get_constraints(&self) -> Result<ConstraintSet> {
        self.actor_addr
            .send(actor_msgs::GetConstraints)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get auto-balance notifications
    ///
    /// Returns notifications since a given timestamp (or all if None).
    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>> {
        self.actor_addr
            .send(actor_msgs::GetAutoBalanceNotifications {
                since_timestamp: None,
            })
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)
    }

    /// Get equilibrium status
    ///
    /// Returns true if the graph has reached equilibrium (settled state).
    async fn get_equilibrium_status(&self) -> Result<bool> {
        self.actor_addr
            .send(actor_msgs::GetEquilibriumStatus)
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(|e: VisionFlowError| GraphRepositoryError::AccessError(e.to_string()))
    }

    /// Compute shortest paths between nodes
    ///
    /// Uses GPU-accelerated pathfinding if available.
    async fn compute_shortest_paths(&self, params: PathfindingParams) -> Result<PathfindingResult> {
        use crate::ports::gpu_semantic_analyzer::PathfindingResult as GpuPathfindingResult;

        // Send compute message with source node
        let gpu_result: GpuPathfindingResult = self
            .actor_addr
            .send(actor_msgs::ComputeShortestPaths {
                source_node_id: params.start_node,
            })
            .await
            .map_err(|e| GraphRepositoryError::AccessError(format!("Mailbox error: {}", e)))?
            .map_err(GraphRepositoryError::AccessError)?;

        // Extract the specific path from source to end node
        let path = gpu_result
            .paths
            .get(&params.end_node)
            .cloned()
            .unwrap_or_default();

        let total_distance = gpu_result
            .distances
            .get(&params.end_node)
            .copied()
            .unwrap_or(f32::INFINITY);

        Ok(PathfindingResult {
            path,
            total_distance,
        })
    }

    /// Get nodes that have changed since last sync
    ///
    /// Note: The actor system doesn't currently track dirty nodes.
    /// This returns an empty set until dirty tracking is implemented.
    async fn get_dirty_nodes(&self) -> Result<HashSet<u32>> {
        // TODO: Implement dirty node tracking in GraphServiceActor
        // For now, return empty set as the actor doesn't expose this
        Ok(HashSet::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Integration tests require a running actor system
    // Unit tests are limited since this is purely a delegation adapter

    #[test]
    fn test_repository_construction() {
        // Cannot construct without actor system, but verify type compiles
        // This is a compile-time test
    }
}
