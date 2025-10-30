// src/cqrs/commands/graph_commands.rs
//! Knowledge Graph Commands
//!
//! Write operations for the knowledge graph repository.
//! All commands are immutable and validated before execution.

use crate::cqrs::types::{Command, Result};
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::node::Node;

/// Add a single node to the knowledge graph
#[derive(Debug, Clone)]
pub struct AddNodeCommand {
    pub node: Node,
}

impl Command for AddNodeCommand {
    type Result = u32; // Returns assigned node ID

    fn name(&self) -> &'static str {
        "AddNode"
    }

    fn validate(&self) -> Result<()> {
        if self.node.label.is_empty() {
            return Err(anyhow::anyhow!("Node label cannot be empty"));
        }
        Ok(())
    }
}

/// Batch add multiple nodes atomically
#[derive(Debug, Clone)]
pub struct AddNodesCommand {
    pub nodes: Vec<Node>,
}

impl Command for AddNodesCommand {
    type Result = Vec<u32>; // Returns assigned node IDs

    fn name(&self) -> &'static str {
        "AddNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node"));
        }
        for node in &self.nodes {
            if node.label.is_empty() {
                return Err(anyhow::anyhow!("All nodes must have labels"));
            }
        }
        Ok(())
    }
}

/// Update an existing node
#[derive(Debug, Clone)]
pub struct UpdateNodeCommand {
    pub node: Node,
}

impl Command for UpdateNodeCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateNode"
    }

    fn validate(&self) -> Result<()> {
        if self.node.label.is_empty() {
            return Err(anyhow::anyhow!("Node label cannot be empty"));
        }
        Ok(())
    }
}

/// Batch update multiple nodes atomically
#[derive(Debug, Clone)]
pub struct UpdateNodesCommand {
    pub nodes: Vec<Node>,
}

impl Command for UpdateNodesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.nodes.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node"));
        }
        Ok(())
    }
}

/// Remove a node by ID
#[derive(Debug, Clone)]
pub struct RemoveNodeCommand {
    pub node_id: u32,
}

impl Command for RemoveNodeCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveNode"
    }
}

/// Batch remove multiple nodes atomically
#[derive(Debug, Clone)]
pub struct RemoveNodesCommand {
    pub node_ids: Vec<u32>,
}

impl Command for RemoveNodesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveNodes"
    }

    fn validate(&self) -> Result<()> {
        if self.node_ids.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one node ID"));
        }
        Ok(())
    }
}

/// Add an edge between two nodes
#[derive(Debug, Clone)]
pub struct AddEdgeCommand {
    pub edge: Edge,
}

impl Command for AddEdgeCommand {
    type Result = String; // Returns assigned edge ID

    fn name(&self) -> &'static str {
        "AddEdge"
    }

    fn validate(&self) -> Result<()> {
        if self.edge.id.is_empty() {
            return Err(anyhow::anyhow!("Edge ID cannot be empty"));
        }
        Ok(())
    }
}

/// Batch add multiple edges atomically
#[derive(Debug, Clone)]
pub struct AddEdgesCommand {
    pub edges: Vec<Edge>,
}

impl Command for AddEdgesCommand {
    type Result = Vec<String>; // Returns assigned edge IDs

    fn name(&self) -> &'static str {
        "AddEdges"
    }

    fn validate(&self) -> Result<()> {
        if self.edges.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one edge"));
        }
        Ok(())
    }
}

/// Update an existing edge
#[derive(Debug, Clone)]
pub struct UpdateEdgeCommand {
    pub edge: Edge,
}

impl Command for UpdateEdgeCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdateEdge"
    }
}

/// Remove an edge by ID
#[derive(Debug, Clone)]
pub struct RemoveEdgeCommand {
    pub edge_id: String,
}

impl Command for RemoveEdgeCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveEdge"
    }

    fn validate(&self) -> Result<()> {
        if self.edge_id.is_empty() {
            return Err(anyhow::anyhow!("Edge ID cannot be empty"));
        }
        Ok(())
    }
}

/// Batch remove multiple edges atomically
#[derive(Debug, Clone)]
pub struct RemoveEdgesCommand {
    pub edge_ids: Vec<String>,
}

impl Command for RemoveEdgesCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "RemoveEdges"
    }

    fn validate(&self) -> Result<()> {
        if self.edge_ids.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one edge ID"));
        }
        Ok(())
    }
}

/// Save complete graph structure
#[derive(Debug, Clone)]
pub struct SaveGraphCommand {
    pub graph: GraphData,
}

impl Command for SaveGraphCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "SaveGraph"
    }
}

/// Clear all graph data
#[derive(Debug, Clone)]
pub struct ClearGraphCommand;

impl Command for ClearGraphCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "ClearGraph"
    }
}

/// Update node positions in batch (for physics simulation)
#[derive(Debug, Clone)]
pub struct UpdatePositionsCommand {
    pub positions: Vec<(u32, f32, f32, f32)>, // (node_id, x, y, z)
}

impl Command for UpdatePositionsCommand {
    type Result = ();

    fn name(&self) -> &'static str {
        "UpdatePositions"
    }

    fn validate(&self) -> Result<()> {
        if self.positions.is_empty() {
            return Err(anyhow::anyhow!("Must provide at least one position"));
        }
        for (node_id, x, y, z) in &self.positions {
            if x.is_nan() || y.is_nan() || z.is_nan() {
                return Err(anyhow::anyhow!(
                    "Position coordinates cannot be NaN for node {}",
                    node_id
                ));
            }
            if x.is_infinite() || y.is_infinite() || z.is_infinite() {
                return Err(anyhow::anyhow!(
                    "Position coordinates cannot be infinite for node {}",
                    node_id
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_node_validation() {
        let mut node = Node::default();
        node.label = "Test".to_string();
        let cmd = AddNodeCommand { node: node.clone() };
        assert!(cmd.validate().is_ok());

        node.label = "".to_string();
        let cmd = AddNodeCommand { node };
        assert!(cmd.validate().is_err());
    }

    #[test]
    fn test_add_nodes_validation() {
        let cmd = AddNodesCommand { nodes: vec![] };
        assert!(cmd.validate().is_err());

        let mut node = Node::default();
        node.label = "Test".to_string();
        let cmd = AddNodesCommand { nodes: vec![node] };
        assert!(cmd.validate().is_ok());
    }

    #[test]
    fn test_update_positions_validation() {
        let cmd = UpdatePositionsCommand {
            positions: vec![(1, 1.0, 2.0, 3.0)],
        };
        assert!(cmd.validate().is_ok());

        let cmd = UpdatePositionsCommand {
            positions: vec![(1, f32::NAN, 2.0, 3.0)],
        };
        assert!(cmd.validate().is_err());

        let cmd = UpdatePositionsCommand {
            positions: vec![(1, f32::INFINITY, 2.0, 3.0)],
        };
        assert!(cmd.validate().is_err());
    }
}
