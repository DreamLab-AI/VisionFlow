// src/cqrs/queries/graph_queries.rs
//! Knowledge Graph Queries
//!
//! Read operations for the knowledge graph repository.
//! All queries are immutable and do not modify state.

use crate::cqrs::types::{Query, Result};
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::ports::knowledge_graph_repository::GraphStatistics;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GetNodeQuery {
    pub node_id: u32,
}

impl Query for GetNodeQuery {
    type Result = Option<Node>;

    fn name(&self) -> &'static str {
        "GetNode"
    }
}

#[derive(Debug, Clone)]
pub struct GetNodesQuery {
    pub node_ids: Vec<u32>,
}

impl Query for GetNodesQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "GetNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.node_ids.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node ID"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GetAllNodesQuery;

impl Query for GetAllNodesQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "GetAllNodes"
    }
}

#[derive(Debug, Clone)]
pub struct SearchNodesQuery {
    pub label_pattern: String,
}

impl Query for SearchNodesQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "SearchNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.label_pattern.is_empty() {
            return Err(anyhow::anyhow!("Label pattern cannot be empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GetNodesByMetadataQuery {
    pub metadata_id: String,
}

impl Query for GetNodesByMetadataQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "GetNodesByMetadata"
    }

    fn validate(&self) -> Result<()> {
        if self.metadata_id.is_empty() {
            return Err(anyhow::anyhow!("Metadata ID cannot be empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GetNodeEdgesQuery {
    pub node_id: u32,
}

impl Query for GetNodeEdgesQuery {
    type Result = Vec<Edge>;

    fn name(&self) -> &'static str {
        "GetNodeEdges"
    }
}

#[derive(Debug, Clone)]
pub struct GetEdgesBetweenQuery {
    pub source_id: u32,
    pub target_id: u32,
}

impl Query for GetEdgesBetweenQuery {
    type Result = Vec<Edge>;

    fn name(&self) -> &'static str {
        "GetEdgesBetween"
    }
}

#[derive(Debug, Clone)]
pub struct GetNeighborsQuery {
    pub node_id: u32,
    pub max_depth: Option<usize>, 
}

impl Query for GetNeighborsQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "GetNeighbors"
    }
}

#[derive(Debug, Clone)]
pub struct CountNodesQuery;

impl Query for CountNodesQuery {
    type Result = usize;

    fn name(&self) -> &'static str {
        "CountNodes"
    }
}

#[derive(Debug, Clone)]
pub struct CountEdgesQuery;

impl Query for CountEdgesQuery {
    type Result = usize;

    fn name(&self) -> &'static str {
        "CountEdges"
    }
}

#[derive(Debug, Clone)]
pub struct GetGraphStatsQuery;

impl Query for GetGraphStatsQuery {
    type Result = GraphStatistics;

    fn name(&self) -> &'static str {
        "GetGraphStats"
    }
}

#[derive(Debug, Clone)]
pub struct LoadGraphQuery;

impl Query for LoadGraphQuery {
    type Result = Arc<GraphData>;

    fn name(&self) -> &'static str {
        "LoadGraph"
    }
}

#[derive(Debug, Clone)]
pub struct QueryNodesQuery {
    pub query: String,
}

impl Query for QueryNodesQuery {
    type Result = Vec<Node>;

    fn name(&self) -> &'static str {
        "QueryNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.query.is_empty() {
            return Err(anyhow::anyhow!("Query string cannot be empty"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GraphHealthCheckQuery;

impl Query for GraphHealthCheckQuery {
    type Result = bool;

    fn name(&self) -> &'static str {
        "GraphHealthCheck"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_node_query() {
        let query = GetNodeQuery { node_id: 1 };
        assert_eq!(query.name(), "GetNode");
        assert!(query.validate().is_ok());
    }

    #[test]
    fn test_search_nodes_validation() {
        let query = SearchNodesQuery {
            label_pattern: "test".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = SearchNodesQuery {
            label_pattern: "".to_string(),
        };
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_get_nodes_validation() {
        let query = GetNodesQuery { node_ids: vec![1] };
        assert!(query.validate().is_ok());

        let query = GetNodesQuery { node_ids: vec![] };
        assert!(query.validate().is_err());
    }

    #[test]
    fn test_query_nodes_validation() {
        let query = QueryNodesQuery {
            query: "color = red".to_string(),
        };
        assert!(query.validate().is_ok());

        let query = QueryNodesQuery {
            query: "".to_string(),
        };
        assert!(query.validate().is_err());
    }
}
