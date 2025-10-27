// tests/ports/test_knowledge_graph_repository.rs
//! Contract tests for KnowledgeGraphRepository port

use super::mocks::MockKnowledgeGraphRepository;
use glam::Vec3;
use webxr::models::{Edge, Node};
use webxr::ports::KnowledgeGraphRepository;

fn create_test_node(id: u32, label: &str) -> Node {
    Node {
        id,
        label: label.to_string(),
        position: Vec3::new(0.0, 0.0, 0.0),
        velocity: Vec3::ZERO,
        color: "#3498db".to_string(),
        size: 1.0,
        metadata_id: format!("node-{}", id),
        ..Default::default()
    }
}

fn create_test_edge(source: u32, target: u32) -> Edge {
    Edge {
        id: String::new(),
        source,
        target,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_add_get_node() {
    let repo = MockKnowledgeGraphRepository::new();

    let node = create_test_node(0, "Test Node");
    let node_id = repo.add_node(&node).await.unwrap();

    assert!(node_id > 0);

    let loaded = repo.get_node(node_id).await.unwrap().unwrap();
    assert_eq!(loaded.label, "Test Node");
}

#[tokio::test]
async fn test_batch_add_nodes() {
    let repo = MockKnowledgeGraphRepository::new();

    let nodes = vec![
        create_test_node(0, "Node 1"),
        create_test_node(0, "Node 2"),
        create_test_node(0, "Node 3"),
    ];

    let ids = repo.batch_add_nodes(nodes).await.unwrap();
    assert_eq!(ids.len(), 3);

    // Verify all nodes were added
    let loaded_nodes = repo.get_nodes(ids).await.unwrap();
    assert_eq!(loaded_nodes.len(), 3);
}

#[tokio::test]
async fn test_update_node() {
    let repo = MockKnowledgeGraphRepository::new();

    let node = create_test_node(0, "Original");
    let node_id = repo.add_node(&node).await.unwrap();

    let mut updated = repo.get_node(node_id).await.unwrap().unwrap();
    updated.label = "Updated".to_string();

    repo.update_node(&updated).await.unwrap();

    let loaded = repo.get_node(node_id).await.unwrap().unwrap();
    assert_eq!(loaded.label, "Updated");
}

#[tokio::test]
async fn test_remove_node() {
    let repo = MockKnowledgeGraphRepository::new();

    let node = create_test_node(0, "To Remove");
    let node_id = repo.add_node(&node).await.unwrap();

    assert!(repo.get_node(node_id).await.unwrap().is_some());

    repo.remove_node(node_id).await.unwrap();

    assert!(repo.get_node(node_id).await.unwrap().is_none());
}

#[tokio::test]
async fn test_search_nodes_by_label() {
    let repo = MockKnowledgeGraphRepository::new();

    repo.add_node(&create_test_node(0, "Rust Programming")).await.unwrap();
    repo.add_node(&create_test_node(0, "Python Programming")).await.unwrap();
    repo.add_node(&create_test_node(0, "Rust Best Practices")).await.unwrap();

    let rust_nodes = repo.search_nodes_by_label("Rust").await.unwrap();
    assert_eq!(rust_nodes.len(), 2);

    let programming_nodes = repo.search_nodes_by_label("Programming").await.unwrap();
    assert_eq!(programming_nodes.len(), 2);
}

#[tokio::test]
async fn test_add_get_edges() {
    let repo = MockKnowledgeGraphRepository::new();

    let node1_id = repo.add_node(&create_test_node(0, "Node 1")).await.unwrap();
    let node2_id = repo.add_node(&create_test_node(0, "Node 2")).await.unwrap();

    let edge = create_test_edge(node1_id, node2_id);
    let edge_id = repo.add_edge(&edge).await.unwrap();

    assert!(!edge_id.is_empty());

    let edges = repo.get_node_edges(node1_id).await.unwrap();
    assert_eq!(edges.len(), 1);
}

#[tokio::test]
async fn test_get_neighbors() {
    let repo = MockKnowledgeGraphRepository::new();

    let node1_id = repo.add_node(&create_test_node(0, "Node 1")).await.unwrap();
    let node2_id = repo.add_node(&create_test_node(0, "Node 2")).await.unwrap();
    let node3_id = repo.add_node(&create_test_node(0, "Node 3")).await.unwrap();

    repo.add_edge(&create_test_edge(node1_id, node2_id)).await.unwrap();
    repo.add_edge(&create_test_edge(node1_id, node3_id)).await.unwrap();

    let neighbors = repo.get_neighbors(node1_id).await.unwrap();
    assert_eq!(neighbors.len(), 2);
}

#[tokio::test]
async fn test_batch_update_positions() {
    let repo = MockKnowledgeGraphRepository::new();

    let node1_id = repo.add_node(&create_test_node(0, "Node 1")).await.unwrap();
    let node2_id = repo.add_node(&create_test_node(0, "Node 2")).await.unwrap();

    let positions = vec![(node1_id, 10.0, 20.0, 30.0), (node2_id, 15.0, 25.0, 35.0)];

    repo.batch_update_positions(positions).await.unwrap();

    let node1 = repo.get_node(node1_id).await.unwrap().unwrap();
    assert_eq!(node1.position, Vec3::new(10.0, 20.0, 30.0));

    let node2 = repo.get_node(node2_id).await.unwrap().unwrap();
    assert_eq!(node2.position, Vec3::new(15.0, 25.0, 35.0));
}

#[tokio::test]
async fn test_get_statistics() {
    let repo = MockKnowledgeGraphRepository::new();

    let node1_id = repo.add_node(&create_test_node(0, "Node 1")).await.unwrap();
    let node2_id = repo.add_node(&create_test_node(0, "Node 2")).await.unwrap();

    repo.add_edge(&create_test_edge(node1_id, node2_id)).await.unwrap();

    let stats = repo.get_statistics().await.unwrap();
    assert_eq!(stats.node_count, 2);
    assert_eq!(stats.edge_count, 1);
    assert!(stats.average_degree > 0.0);
}

#[tokio::test]
async fn test_clear_graph() {
    let repo = MockKnowledgeGraphRepository::new();

    repo.add_node(&create_test_node(0, "Node 1")).await.unwrap();
    repo.add_node(&create_test_node(0, "Node 2")).await.unwrap();

    repo.clear_graph().await.unwrap();

    let stats = repo.get_statistics().await.unwrap();
    assert_eq!(stats.node_count, 0);
    assert_eq!(stats.edge_count, 0);
}

#[tokio::test]
async fn test_transaction_support() {
    let repo = MockKnowledgeGraphRepository::new();

    // Test transaction methods exist and don't error
    repo.begin_transaction().await.unwrap();
    repo.add_node(&create_test_node(0, "Node")).await.unwrap();
    repo.commit_transaction().await.unwrap();

    repo.begin_transaction().await.unwrap();
    repo.rollback_transaction().await.unwrap();
}

#[tokio::test]
async fn test_health_check() {
    let repo = MockKnowledgeGraphRepository::new();
    assert!(repo.health_check().await.unwrap());
}
