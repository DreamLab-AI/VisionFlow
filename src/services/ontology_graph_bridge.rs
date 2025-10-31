// src/services/ontology_graph_bridge.rs
//! Ontology to Knowledge Graph Bridge Service
//!
//! Synchronizes data between ontology.db (OWL classes) and knowledge_graph.db (visualization nodes)
//! This bridge ensures that ontology classes are visible in the graph visualization.

use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::adapters::sqlite_knowledge_graph_repository::SqliteKnowledgeGraphRepository;
use crate::adapters::sqlite_ontology_repository::SqliteOntologyRepository;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;

pub struct OntologyGraphBridge {
    ontology_repo: Arc<SqliteOntologyRepository>,
    graph_repo: Arc<SqliteKnowledgeGraphRepository>,
}

impl OntologyGraphBridge {
    pub fn new(
        ontology_repo: Arc<SqliteOntologyRepository>,
        graph_repo: Arc<SqliteKnowledgeGraphRepository>,
    ) -> Self {
        Self {
            ontology_repo,
            graph_repo,
        }
    }

    /// Synchronize ontology data to knowledge graph database
    /// Converts OWL classes to graph nodes and class hierarchies to edges
    pub async fn sync_ontology_to_graph(&self) -> Result<SyncStats, String> {
        info!("[OntologyBridge] Starting ontology → knowledge graph synchronization");

        let mut stats = SyncStats::default();

        // Get all OWL classes from ontology repository
        let classes = self
            .ontology_repo
            .get_classes()
            .await
            .map_err(|e| format!("Failed to fetch ontology classes: {}", e))?;

        info!(
            "[OntologyBridge] Fetched {} OWL classes from ontology.db",
            classes.len()
        );

        if classes.is_empty() {
            warn!("[OntologyBridge] No ontology classes found to sync");
            return Ok(stats);
        }

        // Convert OWL classes to graph nodes
        let mut nodes: Vec<Node> = Vec::new();
        let mut node_id_map = std::collections::HashMap::new();
        let mut next_id: u32 = 1;

        for class in &classes {
            let node_id = next_id;
            next_id += 1;

            // Map class IRI to node ID for edge creation
            node_id_map.insert(class.iri.clone(), node_id);

            let node = Node {
                id: node_id,
                label: class.label.clone().unwrap_or_else(|| {
                    // Extract label from IRI if not provided
                    class
                        .iri
                        .split(&['#', '/'][..])
                        .last()
                        .unwrap_or(&class.iri)
                        .to_string()
                }),
                x: 0.0, // Will be positioned by physics engine
                y: 0.0,
                z: 0.0,
                vx: 0.0,
                vy: 0.0,
                vz: 0.0,
                mass: 1.0,
                metadata_id: Some(class.iri.clone()),
                shape: "sphere".to_string(),
                size: 1.0,
                color: if class.is_deprecated { "#666666" } else { "#4A90E2" }.to_string(),
                description: class.comment.clone(),
            };

            debug!(
                "[OntologyBridge] Converted class {} → node {} ({})",
                class.iri, node_id, node.label
            );
            nodes.push(node);
            stats.nodes_created += 1;
        }

        // Get class hierarchies and convert to edges
        let mut edges: Vec<Edge> = Vec::new();
        let mut edge_id = 1;

        for class in &classes {
            if let Some(parent_iri) = &class.parent_class_iri {
                // Find node IDs for source and target
                if let (Some(&source_id), Some(&target_id)) = (
                    node_id_map.get(&class.iri),
                    node_id_map.get(parent_iri),
                ) {
                    let edge = Edge {
                        id: format!("edge_{}", edge_id),
                        source: source_id,
                        target: target_id,
                        label: Some("subClassOf".to_string()),
                        edge_type: "hierarchy".to_string(),
                        weight: 1.0,
                        metadata: None,
                    };

                    debug!(
                        "[OntologyBridge] Created edge: {} → {} (subClassOf)",
                        class.iri, parent_iri
                    );
                    edges.push(edge);
                    edge_id += 1;
                    stats.edges_created += 1;
                } else {
                    warn!(
                        "[OntologyBridge] Skipping edge for {}: parent {} not found in node map",
                        class.iri, parent_iri
                    );
                }
            }
        }

        info!(
            "[OntologyBridge] Converted {} classes → {} nodes, {} edges",
            classes.len(),
            nodes.len(),
            edges.len()
        );

        // Create graph data structure
        let mut graph_data = GraphData::new();
        graph_data.nodes = nodes;
        graph_data.edges = edges;

        // Save to knowledge graph database
        info!(
            "[OntologyBridge] Saving graph with {} nodes and {} edges to knowledge_graph.db",
            graph_data.nodes.len(),
            graph_data.edges.len()
        );

        self.graph_repo
            .save_graph(&graph_data)
            .await
            .map_err(|e| format!("Failed to save graph: {}", e))?;

        info!(
            "[OntologyBridge] ✅ Successfully synced ontology to knowledge graph: {} nodes, {} edges",
            stats.nodes_created, stats.edges_created
        );

        Ok(stats)
    }

    /// Clear knowledge graph database before sync (optional)
    pub async fn clear_graph(&self) -> Result<(), String> {
        info!("[OntologyBridge] Clearing knowledge_graph.db");
        self.graph_repo
            .clear_graph()
            .await
            .map_err(|e| format!("Failed to clear graph: {}", e))?;
        info!("[OntologyBridge] ✅ Knowledge graph cleared");
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct SyncStats {
    pub nodes_created: usize,
    pub edges_created: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let onto_repo = Arc::new(SqliteOntologyRepository::new(":memory:").unwrap());
        let graph_repo = Arc::new(SqliteKnowledgeGraphRepository::new(":memory:").unwrap());
        let bridge = OntologyGraphBridge::new(onto_repo, graph_repo);

        // Test sync with empty database
        let stats = bridge.sync_ontology_to_graph().await.unwrap();
        assert_eq!(stats.nodes_created, 0);
        assert_eq!(stats.edges_created, 0);
    }
}
