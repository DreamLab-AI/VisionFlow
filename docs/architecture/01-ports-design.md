# Port Layer Design - Hexagonal Architecture

## Overview

This document defines all port traits (interfaces) for the hexagonal architecture migration. Ports define **what** the application needs without specifying **how** it's implemented.

## Database Ports

### SettingsRepository

**Purpose**: Provides access to application, user, and developer configuration settings.

```rust
// src/ports/settings_repository.rs

use async_trait::async_trait;
use std::collections::HashMap;
use crate::config::{AppFullSettings, PhysicsSettings};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum SettingValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(JsonValue),
}

#[async_trait]
pub trait SettingsRepository: Send + Sync {
    /// Get a single setting by key (supports both camelCase and snake_case)
    async fn get_setting(&self, key: &str) -> Result<Option<SettingValue>, String>;

    /// Set a single setting by key
    async fn set_setting(&self, key: &str, value: SettingValue, description: Option<&str>) -> Result<(), String>;

    /// Get batch of settings by keys
    async fn get_settings_batch(&self, keys: &[String]) -> Result<HashMap<String, SettingValue>, String>;

    /// Set batch of settings atomically
    async fn set_settings_batch(&self, updates: HashMap<String, SettingValue>) -> Result<(), String>;

    /// Load complete application settings
    async fn load_all_settings(&self) -> Result<Option<AppFullSettings>, String>;

    /// Save complete application settings
    async fn save_all_settings(&self, settings: &AppFullSettings) -> Result<(), String>;

    /// Get physics settings for a specific profile (e.g., "logseq", "ontology")
    async fn get_physics_settings(&self, profile_name: &str) -> Result<PhysicsSettings, String>;

    /// Save physics settings for a specific profile
    async fn save_physics_settings(&self, profile_name: &str, settings: &PhysicsSettings) -> Result<(), String>;

    /// List all available physics profiles
    async fn list_physics_profiles(&self) -> Result<Vec<String>, String>;

    /// Delete a physics profile
    async fn delete_physics_profile(&self, profile_name: &str) -> Result<(), String>;

    /// Clear cache (for implementations with caching)
    async fn clear_cache(&self) -> Result<(), String>;
}
```

### KnowledgeGraphRepository

**Purpose**: Manages the main knowledge graph structure parsed from local markdown files.

```rust
// src/ports/knowledge_graph_repository.rs

use async_trait::async_trait;
use std::sync::Arc;
use crate::models::graph::GraphData;
use crate::models::node::Node;
use crate::models::edge::Edge;
use crate::models::metadata::MetadataStore;

#[async_trait]
pub trait KnowledgeGraphRepository: Send + Sync {
    /// Load complete graph structure from database
    async fn load_graph(&self) -> Result<Arc<GraphData>, String>;

    /// Save complete graph structure to database
    async fn save_graph(&self, graph: &GraphData) -> Result<(), String>;

    /// Add a single node to the graph
    async fn add_node(&self, node: &Node) -> Result<u32, String>; // Returns assigned node ID

    /// Update an existing node
    async fn update_node(&self, node: &Node) -> Result<(), String>;

    /// Remove a node by ID
    async fn remove_node(&self, node_id: u32) -> Result<(), String>;

    /// Get a node by ID
    async fn get_node(&self, node_id: u32) -> Result<Option<Node>, String>;

    /// Get nodes by metadata ID
    async fn get_nodes_by_metadata_id(&self, metadata_id: &str) -> Result<Vec<Node>, String>;

    /// Add an edge between two nodes
    async fn add_edge(&self, edge: &Edge) -> Result<String, String>; // Returns assigned edge ID

    /// Update an existing edge
    async fn update_edge(&self, edge: &Edge) -> Result<(), String>;

    /// Remove an edge by ID
    async fn remove_edge(&self, edge_id: &str) -> Result<(), String>;

    /// Get all edges connected to a node
    async fn get_node_edges(&self, node_id: u32) -> Result<Vec<Edge>, String>;

    /// Batch update node positions (for physics simulation)
    async fn batch_update_positions(&self, positions: Vec<(u32, f32, f32, f32)>) -> Result<(), String>;

    /// Query nodes by properties (e.g., "color = red", "size > 10")
    async fn query_nodes(&self, query: &str) -> Result<Vec<Node>, String>;

    /// Get graph statistics
    async fn get_statistics(&self) -> Result<GraphStatistics, String>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub connected_components: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}
```

### OntologyRepository

**Purpose**: Manages the ontology graph structure parsed from GitHub markdown files, including OWL classes, properties, and inference results.

```rust
// src/ports/ontology_repository.rs

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::models::graph::GraphData;

#[async_trait]
pub trait OntologyRepository: Send + Sync {
    /// Load ontology graph structure
    async fn load_ontology_graph(&self) -> Result<Arc<GraphData>, String>;

    /// Save ontology graph structure
    async fn save_ontology_graph(&self, graph: &GraphData) -> Result<(), String>;

    /// Add an OWL class definition
    async fn add_owl_class(&self, class: &OwlClass) -> Result<String, String>; // Returns class IRI

    /// Get an OWL class by IRI
    async fn get_owl_class(&self, iri: &str) -> Result<Option<OwlClass>, String>;

    /// List all OWL classes
    async fn list_owl_classes(&self) -> Result<Vec<OwlClass>, String>;

    /// Add an OWL property definition
    async fn add_owl_property(&self, property: &OwlProperty) -> Result<String, String>;

    /// Get an OWL property by IRI
    async fn get_owl_property(&self, iri: &str) -> Result<Option<OwlProperty>, String>;

    /// List all OWL properties
    async fn list_owl_properties(&self) -> Result<Vec<OwlProperty>, String>;

    /// Add an axiom (e.g., SubClassOf, EquivalentClass)
    async fn add_axiom(&self, axiom: &OwlAxiom) -> Result<u64, String>; // Returns axiom ID

    /// Get all axioms for a class
    async fn get_class_axioms(&self, class_iri: &str) -> Result<Vec<OwlAxiom>, String>;

    /// Store inference results
    async fn store_inference_results(&self, results: &InferenceResults) -> Result<(), String>;

    /// Get latest inference results
    async fn get_inference_results(&self) -> Result<Option<InferenceResults>, String>;

    /// Validate ontology consistency
    async fn validate_ontology(&self) -> Result<ValidationReport, String>;

    /// Query ontology using SPARQL-like syntax
    async fn query_ontology(&self, query: &str) -> Result<Vec<HashMap<String, String>>, String>;

    /// Get ontology metrics
    async fn get_metrics(&self) -> Result<OntologyMetrics, String>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwlClass {
    pub iri: String,
    pub label: Option<String>,
    pub description: Option<String>,
    pub parent_classes: Vec<String>,
    pub properties: HashMap<String, String>,
    pub source_file: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwlProperty {
    pub iri: String,
    pub label: Option<String>,
    pub property_type: PropertyType,
    pub domain: Vec<String>,
    pub range: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum PropertyType {
    ObjectProperty,
    DataProperty,
    AnnotationProperty,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OwlAxiom {
    pub id: Option<u64>,
    pub axiom_type: AxiomType,
    pub subject: String,
    pub object: String,
    pub annotations: HashMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum AxiomType {
    SubClassOf,
    EquivalentClass,
    DisjointWith,
    ObjectPropertyAssertion,
    DataPropertyAssertion,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceResults {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub inferred_axioms: Vec<OwlAxiom>,
    pub inference_time_ms: u64,
    pub reasoner_version: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OntologyMetrics {
    pub class_count: usize,
    pub property_count: usize,
    pub axiom_count: usize,
    pub max_depth: usize,
    pub average_branching_factor: f32,
}
```

## GPU Acceleration Ports

### GpuPhysicsAdapter

**Purpose**: Provides GPU-accelerated physics simulation capabilities.

```rust
// src/ports/gpu_physics_adapter.rs

use async_trait::async_trait;
use std::sync::Arc;
use crate::models::graph::GraphData;
use crate::models::simulation_params::SimulationParams;
use crate::models::constraints::ConstraintSet;

#[async_trait]
pub trait GpuPhysicsAdapter: Send + Sync {
    /// Initialize GPU with graph data
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<(), String>;

    /// Execute a single physics simulation step
    async fn simulate_step(&mut self, params: &SimulationParams) -> Result<PhysicsStepResult, String>;

    /// Update graph data on GPU
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<(), String>;

    /// Upload constraints to GPU
    async fn upload_constraints(&mut self, constraints: &ConstraintSet) -> Result<(), String>;

    /// Clear all constraints
    async fn clear_constraints(&mut self) -> Result<(), String>;

    /// Update simulation parameters
    async fn update_parameters(&mut self, params: &SimulationParams) -> Result<(), String>;

    /// Get current positions from GPU
    async fn get_positions(&self) -> Result<Vec<(u32, f32, f32, f32)>, String>;

    /// Set specific node positions (for user interaction)
    async fn set_node_position(&mut self, node_id: u32, x: f32, y: f32, z: f32) -> Result<(), String>;

    /// Get physics statistics
    async fn get_statistics(&self) -> Result<PhysicsStatistics, String>;

    /// Check if GPU is available and initialized
    fn is_available(&self) -> bool;

    /// Get GPU device information
    fn get_device_info(&self) -> GpuDeviceInfo;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhysicsStepResult {
    pub iteration: u64,
    pub kinetic_energy: f32,
    pub potential_energy: f32,
    pub total_energy: f32,
    pub convergence_delta: f32,
    pub execution_time_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PhysicsStatistics {
    pub total_steps: u64,
    pub average_step_time_ms: f32,
    pub current_fps: f32,
    pub gpu_memory_used_mb: f32,
    pub gpu_utilization_percent: f32,
    pub convergence_rate: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub compute_capability: String,
    pub total_memory_mb: usize,
    pub available_memory_mb: usize,
    pub cuda_cores: Option<usize>,
}
```

### GpuSemanticAnalyzer

**Purpose**: Provides GPU-accelerated semantic analysis, clustering, and pathfinding.

```rust
// src/ports/gpu_semantic_analyzer.rs

use async_trait::async_trait;
use std::sync::Arc;
use std::collections::HashMap;
use crate::models::graph::GraphData;
use crate::models::constraints::ConstraintSet;

#[async_trait]
pub trait GpuSemanticAnalyzer: Send + Sync {
    /// Initialize semantic analyzer with graph data
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<(), String>;

    /// Perform GPU-accelerated community detection
    async fn detect_communities(&mut self, algorithm: ClusteringAlgorithm) -> Result<CommunityDetectionResult, String>;

    /// Compute shortest paths from a source node (GPU-accelerated SSSP)
    async fn compute_shortest_paths(&mut self, source_node_id: u32) -> Result<PathfindingResult, String>;

    /// Compute all-pairs shortest paths
    async fn compute_all_pairs_shortest_paths(&mut self) -> Result<HashMap<(u32, u32), Vec<u32>>, String>;

    /// Generate semantic constraints based on graph analysis
    async fn generate_semantic_constraints(&mut self, config: SemanticConstraintConfig) -> Result<ConstraintSet, String>;

    /// Perform stress majorization layout optimization
    async fn optimize_layout(&mut self, constraints: &ConstraintSet, max_iterations: usize) -> Result<OptimizationResult, String>;

    /// Analyze node importance (PageRank, centrality, etc.)
    async fn analyze_node_importance(&mut self, algorithm: ImportanceAlgorithm) -> Result<HashMap<u32, f32>, String>;

    /// Update graph data for analysis
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<(), String>;

    /// Get semantic analysis statistics
    async fn get_statistics(&self) -> Result<SemanticStatistics, String>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ClusteringAlgorithm {
    Louvain,
    LabelPropagation,
    ConnectedComponents,
    HierarchicalClustering { min_cluster_size: usize },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommunityDetectionResult {
    pub clusters: HashMap<u32, usize>, // node_id -> cluster_id
    pub cluster_sizes: HashMap<usize, usize>, // cluster_id -> size
    pub modularity: f32,
    pub computation_time_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PathfindingResult {
    pub source_node: u32,
    pub distances: HashMap<u32, f32>, // node_id -> distance
    pub paths: HashMap<u32, Vec<u32>>, // node_id -> path (sequence of nodes)
    pub computation_time_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticConstraintConfig {
    pub similarity_threshold: f32,
    pub enable_clustering_constraints: bool,
    pub enable_importance_constraints: bool,
    pub enable_topic_constraints: bool,
    pub max_constraints: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OptimizationResult {
    pub converged: bool,
    pub iterations: u32,
    pub final_stress: f32,
    pub convergence_delta: f32,
    pub computation_time_ms: f32,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ImportanceAlgorithm {
    PageRank { damping: f32, max_iterations: usize },
    Betweenness,
    Closeness,
    Eigenvector,
    Degree,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SemanticStatistics {
    pub total_analyses: u64,
    pub average_clustering_time_ms: f32,
    pub average_pathfinding_time_ms: f32,
    pub cache_hit_rate: f32,
    pub gpu_memory_used_mb: f32,
}
```

## Ontology Inference Port

### InferenceEngine

**Purpose**: Provides ontology reasoning and inference capabilities using whelk-rs.

```rust
// src/ports/inference_engine.rs

use async_trait::async_trait;
use crate::ports::ontology_repository::{OwlClass, OwlAxiom, InferenceResults};

#[async_trait]
pub trait InferenceEngine: Send + Sync {
    /// Load ontology for reasoning
    async fn load_ontology(&mut self, classes: Vec<OwlClass>, axioms: Vec<OwlAxiom>) -> Result<(), String>;

    /// Perform inference to derive new axioms
    async fn infer(&mut self) -> Result<InferenceResults, String>;

    /// Check if a specific axiom is entailed by the ontology
    async fn is_entailed(&self, axiom: &OwlAxiom) -> Result<bool, String>;

    /// Get all inferred subclass relationships
    async fn get_subclass_hierarchy(&self) -> Result<Vec<(String, String)>, String>;

    /// Classify instances into classes
    async fn classify_instance(&self, instance_iri: &str) -> Result<Vec<String>, String>;

    /// Check ontology consistency
    async fn check_consistency(&self) -> Result<bool, String>;

    /// Explain why an axiom is entailed
    async fn explain_entailment(&self, axiom: &OwlAxiom) -> Result<Vec<OwlAxiom>, String>;

    /// Clear loaded ontology
    async fn clear(&mut self) -> Result<(), String>;

    /// Get inference engine statistics
    async fn get_statistics(&self) -> Result<InferenceStatistics, String>;
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InferenceStatistics {
    pub loaded_classes: usize,
    pub loaded_axioms: usize,
    pub inferred_axioms: usize,
    pub last_inference_time_ms: u64,
    pub total_inferences: u64,
}
```

## Summary

This port layer design provides:

1. **Database Repositories** for the three separate databases (settings, knowledge_graph, ontology)
2. **GPU Adapters** for physics simulation and semantic analysis
3. **Inference Engine** for ontology reasoning

All ports are:
- **Async-first** for non-blocking I/O
- **Thread-safe** (`Send + Sync`)
- **Well-documented** with clear purpose and contracts
- **Fully specified** with complete type definitions (no TODOs or stubs)

These interfaces will be implemented by adapters in the next design document.
