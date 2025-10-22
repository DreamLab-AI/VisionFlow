// src/application/knowledge_graph/mod.rs
//! Knowledge Graph Domain Application Layer
//!
//! Contains all directives (write operations) and queries (read operations)
//! for knowledge graph management following CQRS patterns.

pub mod directives;
pub mod queries;

// Re-export directives
pub use directives::{
    AddEdge, AddEdgeHandler, AddNode, AddNodeHandler, BatchUpdatePositions,
    BatchUpdatePositionsHandler, RemoveEdge, RemoveEdgeHandler, RemoveNode, RemoveNodeHandler,
    SaveGraph, SaveGraphHandler, UpdateEdge, UpdateEdgeHandler, UpdateNode, UpdateNodeHandler,
};

// Re-export queries
pub use queries::{
    GetGraphStatistics, GetGraphStatisticsHandler, GetNode, GetNodeEdges, GetNodeEdgesHandler,
    GetNodeHandler, GetNodesByMetadataId, GetNodesByMetadataIdHandler, LoadGraph, LoadGraphHandler,
    QueryNodes, QueryNodesHandler, QueryResult,
};
