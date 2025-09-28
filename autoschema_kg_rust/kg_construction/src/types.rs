//! Core types for knowledge graph construction

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Node types in the knowledge graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NodeType {
    Entity,
    Event,
    Relation,
}

impl std::fmt::Display for NodeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeType::Entity => write!(f, "entity"),
            NodeType::Event => write!(f, "event"),
            NodeType::Relation => write!(f, "relation"),
        }
    }
}

impl std::str::FromStr for NodeType {
    type Err = crate::error::KgConstructionError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "entity" => Ok(NodeType::Entity),
            "event" => Ok(NodeType::Event),
            "relation" => Ok(NodeType::Relation),
            _ => Err(crate::error::KgConstructionError::InvalidNodeType(s.to_string())),
        }
    }
}

/// A concept node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub node: String,
    pub conceptualized_node: Vec<String>,
    pub node_type: NodeType,
}

/// Batched data for processing
#[derive(Debug, Clone)]
pub struct BatchedData {
    pub events: Vec<Vec<String>>,
    pub entities: Vec<Vec<String>>,
    pub relations: Vec<Vec<String>>,
}

/// Processing configuration
#[derive(Debug, Clone)]
pub struct ProcessingConfig {
    pub output_directory: String,
    pub filename_pattern: String,
    pub max_workers: usize,
    pub batch_size: usize,
    pub language: String,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            output_directory: "processed_data".to_string(),
            filename_pattern: "kg".to_string(),
            max_workers: 4,
            batch_size: 32,
            language: "en".to_string(),
        }
    }
}

/// Graph traversal configuration and utilities
#[derive(Debug, Clone)]
pub struct GraphTraversal {
    pub max_neighbors: usize,
    pub context_window: usize,
}

impl Default for GraphTraversal {
    fn default() -> Self {
        Self {
            max_neighbors: 2,
            context_window: 5,
        }
    }
}

/// Statistics tracking for processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Statistics {
    pub total_nodes_processed: usize,
    pub total_batches_processed: usize,
    pub unique_concepts_generated: usize,
    pub processing_time_ms: u128,
    pub events_processed: usize,
    pub entities_processed: usize,
    pub relations_processed: usize,
    pub errors_encountered: usize,
    pub concepts_by_type: HashMap<String, usize>,
}

/// LLM response with usage statistics
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub text: String,
    pub usage: Option<TokenUsage>,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

/// Concept mapping for nodes to concepts
#[derive(Debug, Clone, Default)]
pub struct ConceptMapping {
    pub node_to_concepts: HashMap<String, Vec<String>>,
    pub relation_to_concepts: HashMap<String, Vec<String>>,
    pub all_concepts: std::collections::HashSet<String>,
}

/// Edge representation for CSV output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub start_id: String,
    pub end_id: String,
    pub relation: String,
    pub edge_type: String,
}

/// Node representation for CSV output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: String,
    pub name: String,
    pub label: String,
    pub node_type: Option<NodeType>,
    pub concepts: Vec<String>,
    pub synsets: Vec<String>,
}

/// Shard configuration for distributed processing
#[derive(Debug, Clone)]
pub struct ShardConfig {
    pub shard_idx: usize,
    pub num_shards: usize,
    pub shuffle_data: bool,
}

impl Default for ShardConfig {
    fn default() -> Self {
        Self {
            shard_idx: 0,
            num_shards: 1,
            shuffle_data: true,
        }
    }
}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult {
    pub batch_type: NodeType,
    pub processed_nodes: usize,
    pub generated_concepts: Vec<ConceptNode>,
    pub processing_time_ms: u128,
    pub errors: Vec<String>,
}