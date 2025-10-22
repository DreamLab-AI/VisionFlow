//! Graph type definitions for multi-agent systems

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GraphType {
    /// Standard node-edge graph
    Standard,
    /// Multi-agent system graph
    MultiAgent,
    /// Force-directed layout graph
    ForceDirected,
    /// Hierarchical graph structure
    Hierarchical,
    /// Network topology graph
    Network,
    /// OWL/RDF ontology graph with semantic constraints
    Ontology,
}

impl Default for GraphType {
    fn default() -> Self {
        GraphType::Standard
    }
}

impl std::fmt::Display for GraphType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphType::Standard => write!(f, "standard"),
            GraphType::MultiAgent => write!(f, "multi-agent"),
            GraphType::ForceDirected => write!(f, "force-directed"),
            GraphType::Hierarchical => write!(f, "hierarchical"),
            GraphType::Network => write!(f, "network"),
            GraphType::Ontology => write!(f, "ontology"),
        }
    }
}

impl std::str::FromStr for GraphType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "standard" => Ok(GraphType::Standard),
            "multi-agent" | "multiagent" => Ok(GraphType::MultiAgent),
            "force-directed" | "forcedirected" => Ok(GraphType::ForceDirected),
            "hierarchical" => Ok(GraphType::Hierarchical),
            "network" => Ok(GraphType::Network),
            "ontology" => Ok(GraphType::Ontology),
            _ => Err(format!("Unknown graph type: {}", s)),
        }
    }
}
