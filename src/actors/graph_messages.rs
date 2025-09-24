use actix::prelude::*;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::graph::GraphData;
use crate::models::metadata::FileMetadata;
use std::collections::HashMap;

/// Message-based graph operations to replace Arc::make_mut() patterns
#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct AddNode {
    pub node: Node,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct RemoveNode {
    pub node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct AddEdge {
    pub edge: Edge,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct RemoveEdge {
    pub edge_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct UpdateMetadata {
    pub metadata: FileMetadata,
}

#[derive(Message)]
#[rtype(result = "Result<Node, Box<dyn std::error::Error>>")]
pub struct GetNode {
    pub node_id: u32,
}

#[derive(Message)]
#[rtype(result = "Result<GraphData, Box<dyn std::error::Error>>")]
pub struct GetGraphData;

#[derive(Message)]
#[rtype(result = "Result<HashMap<u32, Node>, Box<dyn std::error::Error>>")]
pub struct GetNodeMap;

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct ClearGraph;

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct BatchUpdateNodes {
    pub nodes: Vec<Node>,
}

#[derive(Message)]
#[rtype(result = "Result<(), Box<dyn std::error::Error>>")]
pub struct UpdateNodePosition {
    pub node_id: u32,
    pub position: (f32, f32, f32),
}