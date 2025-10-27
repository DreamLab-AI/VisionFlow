//! Unit tests for CQRS Query Handlers (Phase 1D)
//!
//! Tests all 8 query handlers with mock repository implementations

use hexser::{HexResult, QueryHandler};
use std::collections::HashMap;
use std::sync::Arc;

use crate::actors::graph_actor::{AutoBalanceNotification, PhysicsState};
use crate::application::graph::queries::*;
use crate::models::constraints::ConstraintSet;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::Metadata;
use crate::models::node::Node;
use crate::ports::graph_repository::{
    GraphRepository, GraphRepositoryError, PathfindingParams, PathfindingResult, Result,
};
use crate::types::vec3::Vec3Data;
use crate::utils::socket_flow_messages::BinaryNodeDataClient;

// ============================================================================
// MOCK REPOSITORY IMPLEMENTATION
// ============================================================================

/// Mock implementation of GraphRepository for testing
struct MockGraphRepository {
    graph_data: Arc<GraphData>,
    node_map: Arc<HashMap<u32, Node>>,
    physics_state: PhysicsState,
    constraints: ConstraintSet,
    notifications: Vec<AutoBalanceNotification>,
    equilibrium: bool,
}

impl MockGraphRepository {
    fn new() -> Self {
        // Create test graph data
        let mut nodes = Vec::new();
        let mut node_map = HashMap::new();

        for i in 1..=5 {
            let node = Node {
                id: i,
                metadata_id: format!("test_meta_{}", i),
                label: format!("Test Node {}", i),
                data: BinaryNodeDataClient::new(
                    i,
                    Vec3Data {
                        x: i as f32 * 10.0,
                        y: i as f32 * 10.0,
                        z: i as f32 * 10.0,
                    },
                    Vec3Data {
                        x: 0.0,
                        y: 0.0,
                        z: 0.0,
                    },
                ),
                metadata: HashMap::new(),
                node_type: Some("default".to_string()),
                size: Some(1.0),
                color: Some("#FFFFFF".to_string()),
                weight: Some(1.0),
                group: Some("test".to_string()),
            };
            node_map.insert(i, node.clone());
            nodes.push(node);
        }

        let mut edges = Vec::new();
        for i in 1..=4 {
            edges.push(Edge {
                id: format!("edge_{}_{}", i, i + 1),
                source: i,
                target: i + 1,
                weight: Some(1.0),
                label: Some(format!("Edge {}-{}", i, i + 1)),
                edge_type: Some("default".to_string()),
                color: None,
                width: None,
            });
        }

        let graph_data = Arc::new(GraphData {
            nodes,
            edges,
            metadata: HashMap::new(),
        });

        let physics_state = PhysicsState {
            is_settled: false,
            stable_frame_count: 10,
            kinetic_energy: 0.5,
        };

        let notifications = vec![AutoBalanceNotification {
            timestamp: 1000,
            parameter_name: "repulsion_strength".to_string(),
            old_value: 100.0,
            new_value: 150.0,
            reason: "Test adjustment".to_string(),
        }];

        Self {
            graph_data,
            node_map: Arc::new(node_map),
            physics_state,
            constraints: ConstraintSet::default(),
            notifications,
            equilibrium: false,
        }
    }

    fn with_settled_physics(mut self) -> Self {
        self.physics_state.is_settled = true;
        self.physics_state.stable_frame_count = 100;
        self.physics_state.kinetic_energy = 0.001;
        self.equilibrium = true;
        self
    }
}

#[async_trait::async_trait]
impl GraphRepository for MockGraphRepository {
    async fn add_nodes(&self, _nodes: Vec<Node>) -> Result<Vec<u32>> {
        Ok(vec![])
    }

    async fn add_edges(&self, _edges: Vec<Edge>) -> Result<Vec<String>> {
        Ok(vec![])
    }

    async fn update_positions(&self, _updates: Vec<(u32, (f32, f32, f32))>) -> Result<()> {
        Ok(())
    }

    async fn clear_dirty_nodes(&self) -> Result<()> {
        Ok(())
    }

    async fn get_graph(&self) -> Result<Arc<GraphData>> {
        Ok(self.graph_data.clone())
    }

    async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
        Ok(self.node_map.clone())
    }

    async fn get_physics_state(&self) -> Result<PhysicsState> {
        Ok(self.physics_state.clone())
    }

    async fn get_node_positions(&self) -> Result<Vec<(u32, glam::Vec3)>> {
        Ok(vec![])
    }

    async fn get_bots_graph(&self) -> Result<Arc<GraphData>> {
        Ok(self.graph_data.clone())
    }

    async fn get_constraints(&self) -> Result<ConstraintSet> {
        Ok(self.constraints.clone())
    }

    async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>> {
        Ok(self.notifications.clone())
    }

    async fn get_equilibrium_status(&self) -> Result<bool> {
        Ok(self.equilibrium)
    }

    async fn compute_shortest_paths(&self, params: PathfindingParams) -> Result<PathfindingResult> {
        // Simple mock: return path from start to end
        Ok(PathfindingResult {
            path: vec![params.start_node, params.end_node],
            total_distance: 10.0,
        })
    }

    async fn get_dirty_nodes(&self) -> Result<std::collections::HashSet<u32>> {
        Ok(std::collections::HashSet::new())
    }
}

// ============================================================================
// QUERY HANDLER TESTS
// ============================================================================

#[test]
fn test_get_graph_data_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetGraphDataHandler::new(mock_repo);

    let result = handler.handle(GetGraphData);

    assert!(result.is_ok());
    let graph_data = result.unwrap();
    assert_eq!(graph_data.nodes.len(), 5);
    assert_eq!(graph_data.edges.len(), 4);
}

#[test]
fn test_get_node_map_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetNodeMapHandler::new(mock_repo);

    let result = handler.handle(GetNodeMap);

    assert!(result.is_ok());
    let node_map = result.unwrap();
    assert_eq!(node_map.len(), 5);
    assert!(node_map.contains_key(&1));
    assert!(node_map.contains_key(&5));
}

#[test]
fn test_get_physics_state_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetPhysicsStateHandler::new(mock_repo);

    let result = handler.handle(GetPhysicsState);

    assert!(result.is_ok());
    let physics_state = result.unwrap();
    assert_eq!(physics_state.is_settled, false);
    assert_eq!(physics_state.stable_frame_count, 10);
    assert!((physics_state.kinetic_energy - 0.5).abs() < 0.001);
}

#[test]
fn test_get_physics_state_handler_settled() {
    let mock_repo =
        Arc::new(MockGraphRepository::new().with_settled_physics()) as Arc<dyn GraphRepository>;
    let handler = GetPhysicsStateHandler::new(mock_repo);

    let result = handler.handle(GetPhysicsState);

    assert!(result.is_ok());
    let physics_state = result.unwrap();
    assert_eq!(physics_state.is_settled, true);
    assert_eq!(physics_state.stable_frame_count, 100);
    assert!(physics_state.kinetic_energy < 0.01);
}

#[test]
fn test_get_auto_balance_notifications_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetAutoBalanceNotificationsHandler::new(mock_repo);

    let query = GetAutoBalanceNotifications {
        since_timestamp: None,
    };
    let result = handler.handle(query);

    assert!(result.is_ok());
    let notifications = result.unwrap();
    assert_eq!(notifications.len(), 1);
    assert_eq!(notifications[0].parameter_name, "repulsion_strength");
    assert_eq!(notifications[0].old_value, 100.0);
    assert_eq!(notifications[0].new_value, 150.0);
}

#[test]
fn test_get_auto_balance_notifications_handler_with_timestamp() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetAutoBalanceNotificationsHandler::new(mock_repo);

    let query = GetAutoBalanceNotifications {
        since_timestamp: Some(500),
    };
    let result = handler.handle(query);

    assert!(result.is_ok());
    let notifications = result.unwrap();
    // Mock always returns all notifications, but in real impl would filter
    assert_eq!(notifications.len(), 1);
}

#[test]
fn test_get_bots_graph_data_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetBotsGraphDataHandler::new(mock_repo);

    let result = handler.handle(GetBotsGraphData);

    assert!(result.is_ok());
    let graph_data = result.unwrap();
    assert_eq!(graph_data.nodes.len(), 5);
    assert_eq!(graph_data.edges.len(), 4);
}

#[test]
fn test_get_constraints_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetConstraintsHandler::new(mock_repo);

    let result = handler.handle(GetConstraints);

    assert!(result.is_ok());
    let _constraints = result.unwrap();
    // ConstraintSet has default implementation, just verify it doesn't error
}

#[test]
fn test_get_equilibrium_status_handler_not_settled() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = GetEquilibriumStatusHandler::new(mock_repo);

    let result = handler.handle(GetEquilibriumStatus);

    assert!(result.is_ok());
    let equilibrium = result.unwrap();
    assert_eq!(equilibrium, false);
}

#[test]
fn test_get_equilibrium_status_handler_settled() {
    let mock_repo =
        Arc::new(MockGraphRepository::new().with_settled_physics()) as Arc<dyn GraphRepository>;
    let handler = GetEquilibriumStatusHandler::new(mock_repo);

    let result = handler.handle(GetEquilibriumStatus);

    assert!(result.is_ok());
    let equilibrium = result.unwrap();
    assert_eq!(equilibrium, true);
}

#[test]
fn test_compute_shortest_paths_handler() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = ComputeShortestPathsHandler::new(mock_repo);

    let query = ComputeShortestPaths {
        start_node: 1,
        end_node: 5,
        max_depth: Some(10),
    };
    let result = handler.handle(query);

    assert!(result.is_ok());
    let pathfinding_result = result.unwrap();
    assert_eq!(pathfinding_result.path, vec![1, 5]);
    assert_eq!(pathfinding_result.total_distance, 10.0);
}

#[test]
fn test_compute_shortest_paths_handler_no_max_depth() {
    let mock_repo = Arc::new(MockGraphRepository::new()) as Arc<dyn GraphRepository>;
    let handler = ComputeShortestPathsHandler::new(mock_repo);

    let query = ComputeShortestPaths {
        start_node: 2,
        end_node: 4,
        max_depth: None,
    };
    let result = handler.handle(query);

    assert!(result.is_ok());
    let pathfinding_result = result.unwrap();
    assert!(!pathfinding_result.path.is_empty());
}

// ============================================================================
// INTEGRATION TESTS WITH ACTUAL REPOSITORY
// ============================================================================

#[cfg(test)]
mod handler_integration_tests {
    use super::*;

    // These tests would use ActorGraphRepository with a real actor system
    // For now they are placeholders showing the structure

    #[test]
    #[ignore = "Requires running actor system"]
    fn test_get_graph_data_with_actor_repository() {
        // TODO: Implement with ActorGraphRepository and running actor
    }

    #[test]
    #[ignore = "Requires running actor system"]
    fn test_concurrent_query_execution() {
        // TODO: Test parallel query execution with real repository
    }
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

#[test]
fn test_handler_error_propagation() {
    // Mock repository that returns errors
    struct ErrorMockRepository;

    #[async_trait::async_trait]
    impl GraphRepository for ErrorMockRepository {
        async fn add_nodes(&self, _nodes: Vec<Node>) -> Result<Vec<u32>> {
            Err(GraphRepositoryError::AccessError("Test error".to_string()))
        }

        async fn add_edges(&self, _edges: Vec<Edge>) -> Result<Vec<String>> {
            Ok(vec![])
        }

        async fn update_positions(&self, _updates: Vec<(u32, (f32, f32, f32))>) -> Result<()> {
            Ok(())
        }

        async fn clear_dirty_nodes(&self) -> Result<()> {
            Ok(())
        }

        async fn get_graph(&self) -> Result<Arc<GraphData>> {
            Err(GraphRepositoryError::AccessError(
                "Graph access failed".to_string(),
            ))
        }

        async fn get_node_map(&self) -> Result<Arc<HashMap<u32, Node>>> {
            Err(GraphRepositoryError::AccessError(
                "Node map access failed".to_string(),
            ))
        }

        async fn get_physics_state(&self) -> Result<PhysicsState> {
            Err(GraphRepositoryError::AccessError(
                "Physics state access failed".to_string(),
            ))
        }

        async fn get_node_positions(&self) -> Result<Vec<(u32, glam::Vec3)>> {
            Ok(vec![])
        }

        async fn get_bots_graph(&self) -> Result<Arc<GraphData>> {
            Ok(Arc::new(GraphData::default()))
        }

        async fn get_constraints(&self) -> Result<ConstraintSet> {
            Ok(ConstraintSet::default())
        }

        async fn get_auto_balance_notifications(&self) -> Result<Vec<AutoBalanceNotification>> {
            Ok(vec![])
        }

        async fn get_equilibrium_status(&self) -> Result<bool> {
            Ok(false)
        }

        async fn compute_shortest_paths(
            &self,
            _params: PathfindingParams,
        ) -> Result<PathfindingResult> {
            Err(GraphRepositoryError::AccessError(
                "Pathfinding failed".to_string(),
            ))
        }

        async fn get_dirty_nodes(&self) -> Result<std::collections::HashSet<u32>> {
            Ok(std::collections::HashSet::new())
        }
    }

    let error_repo = Arc::new(ErrorMockRepository) as Arc<dyn GraphRepository>;

    // Test GetGraphData error handling
    let handler = GetGraphDataHandler::new(error_repo.clone());
    let result = handler.handle(GetGraphData);
    assert!(result.is_err());

    // Test GetNodeMap error handling
    let handler = GetNodeMapHandler::new(error_repo.clone());
    let result = handler.handle(GetNodeMap);
    assert!(result.is_err());

    // Test GetPhysicsState error handling
    let handler = GetPhysicsStateHandler::new(error_repo.clone());
    let result = handler.handle(GetPhysicsState);
    assert!(result.is_err());

    // Test ComputeShortestPaths error handling
    let handler = ComputeShortestPathsHandler::new(error_repo.clone());
    let result = handler.handle(ComputeShortestPaths {
        start_node: 1,
        end_node: 2,
        max_depth: None,
    });
    assert!(result.is_err());
}
