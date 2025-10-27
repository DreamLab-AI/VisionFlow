// tests/cqrs/integration_tests.rs
//! Integration tests for CQRS layer
//!
//! Tests end-to-end command and query execution with actual repository implementations.

use anyhow::Result;
use std::sync::Arc;
use visionflow::cqrs::bus::{CommandBus, QueryBus};
use visionflow::cqrs::commands::*;
use visionflow::cqrs::handlers::*;
use visionflow::cqrs::queries::*;
use visionflow::models::node::Node;
use visionflow::services::SqliteKnowledgeGraphRepository;

#[tokio::test]
async fn test_add_and_get_node() -> Result<()> {
    // Setup
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Create a node
    let mut node = Node::default();
    node.label = "Test Node".to_string();
    node.x = 1.0;
    node.y = 2.0;
    node.z = 3.0;

    let add_cmd = AddNodeCommand { node: node.clone() };
    let node_id = command_bus.execute(add_cmd).await?;

    // Retrieve the node
    let get_query = GetNodeQuery { node_id };
    let retrieved_node = query_bus.execute(get_query).await?;

    assert!(retrieved_node.is_some());
    let retrieved = retrieved_node.unwrap();
    assert_eq!(retrieved.label, "Test Node");
    assert_eq!(retrieved.x, 1.0);

    Ok(())
}

#[tokio::test]
async fn test_batch_add_nodes() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let cmd_handler = GraphCommandHandler::new(repo.clone());
    command_bus.register(Box::new(cmd_handler)).await;

    // Create multiple nodes
    let mut nodes = Vec::new();
    for i in 0..5 {
        let mut node = Node::default();
        node.label = format!("Node {}", i);
        nodes.push(node);
    }

    let add_cmd = AddNodesCommand { nodes };
    let node_ids = command_bus.execute(add_cmd).await?;

    assert_eq!(node_ids.len(), 5);

    Ok(())
}

#[tokio::test]
async fn test_search_nodes() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Add test nodes
    let mut node1 = Node::default();
    node1.label = "Apple".to_string();
    let mut node2 = Node::default();
    node2.label = "Application".to_string();
    let mut node3 = Node::default();
    node3.label = "Banana".to_string();

    command_bus.execute(AddNodeCommand { node: node1 }).await?;
    command_bus.execute(AddNodeCommand { node: node2 }).await?;
    command_bus.execute(AddNodeCommand { node: node3 }).await?;

    // Search for nodes containing "App"
    let search_query = SearchNodesQuery {
        label_pattern: "App".to_string(),
    };
    let results = query_bus.execute(search_query).await?;

    assert_eq!(results.len(), 2);

    Ok(())
}

#[tokio::test]
async fn test_update_node() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Create node
    let mut node = Node::default();
    node.label = "Original".to_string();
    let add_cmd = AddNodeCommand { node: node.clone() };
    let node_id = command_bus.execute(add_cmd).await?;

    // Update node
    node.id = node_id;
    node.label = "Updated".to_string();
    let update_cmd = UpdateNodeCommand { node };
    command_bus.execute(update_cmd).await?;

    // Verify update
    let get_query = GetNodeQuery { node_id };
    let retrieved = query_bus.execute(get_query).await?.unwrap();
    assert_eq!(retrieved.label, "Updated");

    Ok(())
}

#[tokio::test]
async fn test_remove_node() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Create node
    let mut node = Node::default();
    node.label = "To Delete".to_string();
    let add_cmd = AddNodeCommand { node };
    let node_id = command_bus.execute(add_cmd).await?;

    // Remove node
    let remove_cmd = RemoveNodeCommand { node_id };
    command_bus.execute(remove_cmd).await?;

    // Verify removal
    let get_query = GetNodeQuery { node_id };
    let result = query_bus.execute(get_query).await?;
    assert!(result.is_none());

    Ok(())
}

#[tokio::test]
async fn test_graph_statistics() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Add multiple nodes
    for i in 0..10 {
        let mut node = Node::default();
        node.label = format!("Node {}", i);
        command_bus.execute(AddNodeCommand { node }).await?;
    }

    // Get statistics
    let stats_query = GetGraphStatsQuery;
    let stats = query_bus.execute(stats_query).await?;

    assert_eq!(stats.node_count, 10);

    Ok(())
}

#[tokio::test]
async fn test_command_validation() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let cmd_handler = GraphCommandHandler::new(repo);
    command_bus.register(Box::new(cmd_handler)).await;

    // Try to add node with empty label (should fail validation)
    let mut node = Node::default();
    node.label = "".to_string();
    let add_cmd = AddNodeCommand { node };

    let result = command_bus.execute(add_cmd).await;
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_update_positions() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Create nodes
    let mut node = Node::default();
    node.label = "Node 1".to_string();
    let node_id = command_bus.execute(AddNodeCommand { node }).await?;

    // Update position
    let positions = vec![(node_id, 10.0, 20.0, 30.0)];
    let update_cmd = UpdatePositionsCommand { positions };
    command_bus.execute(update_cmd).await?;

    // Verify position
    let get_query = GetNodeQuery { node_id };
    let node = query_bus.execute(get_query).await?.unwrap();
    assert_eq!(node.x, 10.0);
    assert_eq!(node.y, 20.0);
    assert_eq!(node.z, 30.0);

    Ok(())
}

#[tokio::test]
async fn test_clear_graph() -> Result<()> {
    let repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").await?);
    repo.initialize().await?;

    let command_bus = CommandBus::new();
    let query_bus = QueryBus::new();

    let cmd_handler = GraphCommandHandler::new(repo.clone());
    let query_handler = GraphQueryHandler::new(repo.clone());

    command_bus.register(Box::new(cmd_handler)).await;
    query_bus.register(Box::new(query_handler)).await;

    // Add nodes
    for i in 0..5 {
        let mut node = Node::default();
        node.label = format!("Node {}", i);
        command_bus.execute(AddNodeCommand { node }).await?;
    }

    // Clear graph
    let clear_cmd = ClearGraphCommand;
    command_bus.execute(clear_cmd).await?;

    // Verify cleared
    let stats_query = GetGraphStatsQuery;
    let stats = query_bus.execute(stats_query).await?;
    assert_eq!(stats.node_count, 0);

    Ok(())
}
