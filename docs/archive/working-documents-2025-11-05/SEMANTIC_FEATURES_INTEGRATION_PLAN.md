# Semantic Graph Features Integration Plan

**Version:** 1.0
**Date:** November 5, 2025
**Status:** Implementation Ready

---

## Executive Summary

This document outlines the complete integration plan for three major feature sets into VisionFlow:

1. **Semantic Forces** - GPU-accelerated physics with semantic meaning
2. **Typed Ontology System** - Schema-aware nodes and edges with natural language queries
3. **Intelligent Pathfinding** - Query-guided and LLM-assisted graph traversal

**Expected Impact:**
- 10x more intuitive graph layouts (semantic forces)
- Natural language graph interaction (LLM integration)
- 5x more relevant pathfinding results (semantic SSSP)

---

## 1. Semantic Forces Integration

### 1.1 Overview

Enhance VisionFlow's GPU physics engine with semantic meaning, where forces convey information about relationships, hierarchies, and node types rather than just preventing overlap.

**Source Inspiration:** `3d-force-graph` library patterns

### 1.2 Backend Implementation

#### A. Node/Edge Type System

**File:** `src/models/graph_models.rs`

Add type enumerations:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeType {
    Generic,
    Person,
    Organization,
    Project,
    Task,
    Concept,
    Class,      // Ontology class
    Individual, // Ontology individual
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EdgeType {
    Generic,
    Dependency,
    Hierarchy,   // Parent-child
    Association,
    Sequence,    // Temporal or ordered
    SubClassOf,  // Ontology
    InstanceOf,  // Ontology
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticForceConfig {
    pub force_type: SemanticForceType,
    pub strength: f32,
    pub enabled: bool,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SemanticForceType {
    DAGLayout,           // Hierarchical layout
    TypeClustering,      // Group by node type
    Collision,           // Prevent overlap
    AttributeWeighted,   // Custom attribute-based
    EdgeTypeWeighted,    // Different edge types = different spring strengths
}
```

**File:** `src/models/neo4j_models.rs`

Extend Neo4j node/edge models:

```rust
pub struct Neo4jNode {
    pub id: String,
    pub label: String,
    pub node_type: NodeType,  // NEW
    pub properties: HashMap<String, serde_json::Value>,
    pub semantic_weight: f32, // NEW - for semantic forces
    pub hierarchy_level: Option<i32>, // NEW - for DAG layout
}

pub struct Neo4jEdge {
    pub source: String,
    pub target: String,
    pub edge_type: EdgeType,  // NEW
    pub weight: f32,
    pub semantic_strength: f32, // NEW - semantic spring strength
}
```

#### B. CUDA Semantic Force Kernels

**File:** `src/gpu/kernels/semantic_forces.cu`

```cuda
// 1. DAG Hierarchical Layout Kernel
__global__ void apply_dag_force(
    const int* node_hierarchy_levels,
    const int* node_types,
    float3* positions,
    float3* velocities,
    const int num_nodes,
    const DAGConfig config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int my_level = node_hierarchy_levels[idx];
    if (my_level < 0) return; // Not part of DAG

    // Apply strong vertical/radial force to hierarchy level
    float3 target_position;
    switch (config.direction) {
        case DAG_TOP_DOWN:
            target_position = make_float3(
                positions[idx].x,  // Keep x
                config.level_distance * my_level, // Lock y to level
                positions[idx].z   // Keep z
            );
            break;
        case DAG_RADIAL:
            // Arrange in concentric circles by level
            float angle = /* distribute around circle */;
            float radius = config.level_distance * my_level;
            target_position = make_float3(
                radius * cos(angle),
                0.0f,
                radius * sin(angle)
            );
            break;
        case DAG_LEFT_RIGHT:
            target_position = make_float3(
                config.level_distance * my_level, // Lock x to level
                positions[idx].y,  // Keep y
                positions[idx].z   // Keep z
            );
            break;
    }

    // Apply spring force towards target position
    float3 force = (target_position - positions[idx]) * config.strength;
    velocities[idx] += force;
}

// 2. Type Clustering Kernel
__global__ void apply_type_clustering(
    const int* node_types,
    float3* positions,
    float3* velocities,
    const int num_nodes,
    const TypeClusterConfig config
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    int my_type = node_types[idx];
    float3 cluster_center = config.type_centers[my_type];

    // Attract to type cluster center
    float3 to_center = cluster_center - positions[idx];
    float distance = length(to_center);

    if (distance > 0.01f) {
        float3 force = (to_center / distance) * config.clustering_strength;
        velocities[idx] += force;
    }

    // Reduce repulsion between same-type nodes
    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;
        if (node_types[j] != my_type) continue;

        float3 diff = positions[idx] - positions[j];
        float dist = length(diff);
        if (dist < config.same_type_radius && dist > 0.01f) {
            // Weaker repulsion for same type
            float3 repulsion = (diff / dist) * (config.same_type_repulsion / (dist * dist));
            velocities[idx] += repulsion;
        }
    }
}

// 3. Collision Detection Kernel
__global__ void apply_collision_force(
    const float* node_radii,
    float3* positions,
    float3* velocities,
    const int num_nodes,
    const float collision_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    float my_radius = node_radii[idx];

    for (int j = 0; j < num_nodes; j++) {
        if (j == idx) continue;

        float3 diff = positions[idx] - positions[j];
        float dist = length(diff);
        float min_dist = my_radius + node_radii[j];

        // If nodes are overlapping
        if (dist < min_dist && dist > 0.01f) {
            // Push apart with force proportional to overlap
            float overlap = min_dist - dist;
            float3 push_force = (diff / dist) * overlap * collision_strength;
            velocities[idx] += push_force;
        }
    }
}

// 4. Attribute-Weighted Spring Force Kernel
__global__ void apply_attribute_weighted_springs(
    const int* edge_sources,
    const int* edge_targets,
    const float* edge_weights,
    const float* edge_semantic_strengths,
    float3* positions,
    float3* velocities,
    const int num_edges,
    const float spring_base_strength
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    int src = edge_sources[idx];
    int tgt = edge_targets[idx];

    float3 diff = positions[tgt] - positions[src];
    float dist = length(diff);

    // Semantic strength modifies ideal spring length and force
    float semantic_strength = edge_semantic_strengths[idx];
    float ideal_length = edge_weights[idx] * semantic_strength;

    if (dist > 0.01f) {
        float displacement = dist - ideal_length;
        float3 force = (diff / dist) * displacement * spring_base_strength * semantic_strength;

        // Apply to both nodes (Newton's third law)
        atomicAdd(&velocities[src].x, force.x);
        atomicAdd(&velocities[src].y, force.y);
        atomicAdd(&velocities[src].z, force.z);

        atomicAdd(&velocities[tgt].x, -force.x);
        atomicAdd(&velocities[tgt].y, -force.y);
        atomicAdd(&velocities[tgt].z, -force.z);
    }
}
```

**File:** `src/gpu/semantic_physics.rs`

Rust wrapper for CUDA kernels:

```rust
pub struct SemanticPhysicsEngine {
    gpu_context: Arc<GpuContext>,
    force_configs: HashMap<SemanticForceType, SemanticForceConfig>,
    hierarchy_calculator: HierarchyCalculator,
}

impl SemanticPhysicsEngine {
    pub fn apply_semantic_forces(
        &self,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> Result<Vec<Vector3<f32>>, GpuError> {
        let mut velocity_updates = vec![Vector3::zero(); nodes.len()];

        // 1. Apply DAG forces if enabled
        if self.is_enabled(SemanticForceType::DAGLayout) {
            let hierarchy_levels = self.hierarchy_calculator
                .calculate_levels(nodes, edges)?;
            self.apply_dag_kernel(&hierarchy_levels, &mut velocity_updates)?;
        }

        // 2. Apply type clustering if enabled
        if self.is_enabled(SemanticForceType::TypeClustering) {
            self.apply_type_clustering_kernel(nodes, &mut velocity_updates)?;
        }

        // 3. Apply collision forces if enabled
        if self.is_enabled(SemanticForceType::Collision) {
            self.apply_collision_kernel(nodes, &mut velocity_updates)?;
        }

        // 4. Apply attribute-weighted springs if enabled
        if self.is_enabled(SemanticForceType::AttributeWeighted) {
            self.apply_attribute_springs_kernel(nodes, edges, &mut velocity_updates)?;
        }

        Ok(velocity_updates)
    }
}
```

#### C. Hierarchy Calculator

**File:** `src/gpu/hierarchy_calculator.rs`

```rust
pub struct HierarchyCalculator {
    cache: Arc<RwLock<HashMap<String, i32>>>,
}

impl HierarchyCalculator {
    pub fn calculate_levels(
        &self,
        nodes: &[GraphNode],
        edges: &[GraphEdge],
    ) -> Result<Vec<i32>, GraphError> {
        // Build adjacency list
        let graph = self.build_directed_graph(nodes, edges);

        // Detect cycles (DAG validation)
        if self.has_cycle(&graph) {
            return Err(GraphError::NotADAG);
        }

        // Topological sort + level assignment
        let levels = self.assign_levels_topological(&graph)?;

        Ok(levels)
    }

    fn assign_levels_topological(&self, graph: &DirectedGraph) -> Result<Vec<i32>, GraphError> {
        let mut levels = vec![-1; graph.node_count()];
        let mut in_degree = vec![0; graph.node_count()];

        // Calculate in-degrees
        for edges in graph.adjacency.values() {
            for &target in edges {
                in_degree[target] += 1;
            }
        }

        // BFS level assignment
        let mut queue: VecDeque<usize> = in_degree.iter()
            .enumerate()
            .filter(|(_, &deg)| deg == 0)
            .map(|(idx, _)| idx)
            .collect();

        // Root nodes are level 0
        for &node in &queue {
            levels[node] = 0;
        }

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = graph.adjacency.get(&node) {
                for &neighbor in neighbors {
                    in_degree[neighbor] -= 1;

                    // Level = max(parent_levels) + 1
                    levels[neighbor] = levels[neighbor].max(levels[node] + 1);

                    if in_degree[neighbor] == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        Ok(levels)
    }
}
```

### 1.3 API Endpoints

**File:** `src/handlers/semantic_forces_handler.rs`

```rust
#[derive(Debug, Deserialize)]
pub struct ConfigureSemanticForceRequest {
    pub force_type: SemanticForceType,
    pub strength: f32,
    pub enabled: bool,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SemanticForceStatus {
    pub enabled_forces: Vec<SemanticForceType>,
    pub dag_mode: Option<DAGMode>,
    pub hierarchy_levels: HashMap<String, i32>,
}

pub async fn configure_semantic_force(
    app_state: web::Data<AppState>,
    req: web::Json<ConfigureSemanticForceRequest>,
) -> Result<HttpResponse, ActixError> {
    let config = SemanticForceConfig {
        force_type: req.force_type.clone(),
        strength: req.strength,
        enabled: req.enabled,
        parameters: req.parameters.clone(),
    };

    app_state.semantic_physics.update_config(config).await?;

    Ok(HttpResponse::Ok().json(json!({
        "status": "success",
        "force_type": req.force_type,
        "enabled": req.enabled
    })))
}

pub async fn get_semantic_force_status(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, ActixError> {
    let status = app_state.semantic_physics.get_status().await?;
    Ok(HttpResponse::Ok().json(status))
}

pub async fn calculate_hierarchy(
    app_state: web::Data<AppState>,
) -> Result<HttpResponse, ActixError> {
    let hierarchy_levels = app_state.semantic_physics
        .calculate_and_cache_hierarchy().await?;

    Ok(HttpResponse::Ok().json(json!({
        "hierarchy_levels": hierarchy_levels,
        "node_count": hierarchy_levels.len()
    })))
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/semantic-forces")
            .route("/configure", web::post().to(configure_semantic_force))
            .route("/status", web::get().to(get_semantic_force_status))
            .route("/hierarchy", web::post().to(calculate_hierarchy))
    );
}
```

### 1.4 Frontend Integration

**File:** `client/src/api/semanticForces.ts`

```typescript
export interface SemanticForceConfig {
  forceType: SemanticForceType;
  strength: number;
  enabled: boolean;
  parameters: Record<string, any>;
}

export enum SemanticForceType {
  DAGLayout = 'DAGLayout',
  TypeClustering = 'TypeClustering',
  Collision = 'Collision',
  AttributeWeighted = 'AttributeWeighted',
  EdgeTypeWeighted = 'EdgeTypeWeighted',
}

export const semanticForcesApi = {
  configure: async (config: SemanticForceConfig): Promise<void> => {
    await axios.post('/api/semantic-forces/configure', config);
  },

  getStatus: async (): Promise<SemanticForceStatus> => {
    const response = await axios.get('/api/semantic-forces/status');
    return response.data;
  },

  calculateHierarchy: async (): Promise<Record<string, number>> => {
    const response = await axios.post('/api/semantic-forces/hierarchy');
    return response.data.hierarchy_levels;
  },
};
```

**File:** `client/src/features/physics/components/SemanticForceControls.tsx`

```tsx
export const SemanticForceControls: React.FC = () => {
  const [dagEnabled, setDagEnabled] = useState(false);
  const [dagMode, setDagMode] = useState<'td' | 'radial' | 'lr'>('td');
  const [clusteringEnabled, setClusteringEnabled] = useState(false);
  const [collisionEnabled, setCollisionEnabled] = useState(true);

  const handleDAGToggle = async () => {
    const newEnabled = !dagEnabled;
    setDagEnabled(newEnabled);

    await semanticForcesApi.configure({
      forceType: SemanticForceType.DAGLayout,
      strength: 10.0,
      enabled: newEnabled,
      parameters: {
        direction: dagMode,
        levelDistance: 100.0,
      },
    });
  };

  const handleCalculateHierarchy = async () => {
    const levels = await semanticForcesApi.calculateHierarchy();
    console.log('Hierarchy levels:', levels);
    // Update visualization
  };

  return (
    <div className="semantic-force-controls">
      <h3>Semantic Forces</h3>

      <div className="force-section">
        <label>
          <input
            type="checkbox"
            checked={dagEnabled}
            onChange={handleDAGToggle}
          />
          DAG Hierarchical Layout
        </label>

        {dagEnabled && (
          <select value={dagMode} onChange={(e) => setDagMode(e.target.value as any)}>
            <option value="td">Top-Down</option>
            <option value="radial">Radial</option>
            <option value="lr">Left-Right</option>
          </select>
        )}
      </div>

      <div className="force-section">
        <label>
          <input
            type="checkbox"
            checked={clusteringEnabled}
            onChange={() => setClusteringEnabled(!clusteringEnabled)}
          />
          Type Clustering
        </label>
      </div>

      <div className="force-section">
        <label>
          <input
            type="checkbox"
            checked={collisionEnabled}
            onChange={() => setCollisionEnabled(!collisionEnabled)}
          />
          Collision Detection
        </label>
      </div>

      <button onClick={handleCalculateHierarchy}>
        Calculate Hierarchy
      </button>
    </div>
  );
};
```

---

## 2. Typed Ontology & Natural Language Queries

### 2.1 Overview

Add formal types to nodes and edges, expose schema to LLMs, and enable natural language graph queries.

**Source Inspiration:** `graph_RAG` SPARQL integration

### 2.2 Backend Implementation

#### A. Schema Exposure Service

**File:** `src/services/schema_service.rs`

```rust
#[derive(Debug, Serialize)]
pub struct GraphSchema {
    pub node_types: Vec<NodeTypeSchema>,
    pub edge_types: Vec<EdgeTypeSchema>,
    pub properties: HashMap<String, PropertySchema>,
}

#[derive(Debug, Serialize)]
pub struct NodeTypeSchema {
    pub type_name: String,
    pub description: String,
    pub properties: Vec<String>,
    pub count: usize,
}

pub struct SchemaService {
    neo4j_adapter: Arc<Neo4jAdapter>,
    schema_cache: Arc<RwLock<Option<GraphSchema>>>,
}

impl SchemaService {
    pub async fn extract_schema(&self) -> Result<GraphSchema, ServiceError> {
        // Query Neo4j for all node types
        let node_types = self.extract_node_types().await?;
        let edge_types = self.extract_edge_types().await?;
        let properties = self.extract_properties().await?;

        let schema = GraphSchema {
            node_types,
            edge_types,
            properties,
        };

        // Cache for LLM context
        *self.schema_cache.write().await = Some(schema.clone());

        Ok(schema)
    }

    pub async fn get_llm_context(&self) -> Result<String, ServiceError> {
        let schema = self.schema_cache.read().await.clone()
            .ok_or(ServiceError::SchemaNotLoaded)?;

        // Format for LLM consumption
        let context = format!(
            "Graph Database Schema:\n\n\
             Node Types:\n{}\n\n\
             Edge Types:\n{}\n\n\
             Properties:\n{}",
            self.format_node_types(&schema.node_types),
            self.format_edge_types(&schema.edge_types),
            self.format_properties(&schema.properties)
        );

        Ok(context)
    }
}
```

#### B. Natural Language Query Handler

**File:** `src/handlers/nl_query_handler.rs`

```rust
#[derive(Debug, Deserialize)]
pub struct NaturalLanguageQueryRequest {
    pub query: String,
    pub llm_provider: Option<String>, // "openai", "anthropic", etc.
}

#[derive(Debug, Serialize)]
pub struct NaturalLanguageQueryResponse {
    pub interpreted_query: String,
    pub cypher_query: String,
    pub results: Vec<serde_json::Value>,
    pub confidence: f32,
}

pub async fn handle_nl_query(
    app_state: web::Data<AppState>,
    req: web::Json<NaturalLanguageQueryRequest>,
) -> Result<HttpResponse, ActixError> {
    // 1. Get schema for LLM context
    let schema_context = app_state.schema_service.get_llm_context().await?;

    // 2. Call LLM to translate natural language to Cypher
    let llm_prompt = format!(
        "{}\n\nUser Query: {}\n\nGenerate a Cypher query to answer this question.",
        schema_context,
        req.query
    );

    let cypher_query = app_state.llm_service
        .generate_cypher(llm_prompt, req.llm_provider.as_deref())
        .await?;

    // 3. Execute Cypher query
    let results = app_state.neo4j_adapter
        .execute_cypher(&cypher_query)
        .await?;

    // 4. Calculate confidence based on result count and query complexity
    let confidence = calculate_confidence(&results, &cypher_query);

    Ok(HttpResponse::Ok().json(NaturalLanguageQueryResponse {
        interpreted_query: req.query.clone(),
        cypher_query,
        results,
        confidence,
    }))
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/nl-query")
            .route("", web::post().to(handle_nl_query))
            .route("/schema", web::get().to(get_schema))
    );
}
```

### 2.3 Frontend Integration

**File:** `client/src/features/query/components/NaturalLanguageQuery.tsx`

```tsx
export const NaturalLanguageQuery: React.FC = () => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await axios.post('/api/nl-query', { query });
      setResults(response.data);
    } catch (error) {
      console.error('Query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="nl-query-panel">
      <h3>Ask a Question About Your Graph</h3>
      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g., Show me all high-priority tasks connected to the VisionFlow project"
          rows={3}
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Querying...' : 'Ask Question'}
        </button>
      </form>

      {results && (
        <div className="query-results">
          <div className="interpreted">
            <strong>Cypher Query:</strong>
            <pre>{results.cypher_query}</pre>
          </div>
          <div className="confidence">
            Confidence: {(results.confidence * 100).toFixed(1)}%
          </div>
          <div className="results">
            <strong>Results ({results.results.length}):</strong>
            {/* Render results */}
          </div>
        </div>
      )}
    </div>
  );
};
```

---

## 3. Intelligent Pathfinding Integration

### 3.1 Overview

Enhance existing hybrid SSSP with semantic weighting and implement new traversal algorithms.

**Source Inspiration:** `knowledge-graph-traversal-semantic-rag-research` algorithms

### 3.2 Backend Implementation

#### A. Semantic SSSP Enhancement

**File:** `src/algorithms/semantic_sssp.rs`

```rust
pub struct SemanticSSP {
    base_sssp: HybridSSP,
    similarity_calculator: Arc<SimilarityCalculator>,
    query_context: Option<String>,
}

impl SemanticSSP {
    pub fn calculate_path(
        &self,
        source: NodeId,
        target: NodeId,
        query: Option<&str>,
    ) -> Result<SemanticPath, PathError> {
        // Calculate base shortest path
        let base_path = self.base_sssp.calculate(source, target)?;

        if let Some(query_text) = query {
            // Recalculate with semantic weighting
            let semantic_path = self.calculate_semantic_weighted_path(
                source,
                target,
                query_text,
                &base_path
            )?;

            Ok(semantic_path)
        } else {
            Ok(SemanticPath {
                nodes: base_path.nodes,
                edges: base_path.edges,
                total_cost: base_path.cost,
                semantic_relevance: 0.0,
                query_similarities: HashMap::new(),
            })
        }
    }

    fn calculate_semantic_weighted_path(
        &self,
        source: NodeId,
        target: NodeId,
        query: &str,
        base_path: &Path,
    ) -> Result<SemanticPath, PathError> {
        // Embed query
        let query_embedding = self.similarity_calculator.embed(query)?;

        // For each node in path, calculate similarity to query
        let mut query_similarities = HashMap::new();
        let mut modified_costs = Vec::new();

        for (i, edge) in base_path.edges.iter().enumerate() {
            let node = &base_path.nodes[i + 1];

            // Calculate node similarity to query
            let node_embedding = self.get_node_embedding(node)?;
            let similarity = cosine_similarity(&query_embedding, &node_embedding);
            query_similarities.insert(node.id.clone(), similarity);

            // Modify edge weight: new_weight = original_weight + (1.0 - similarity)
            let original_weight = edge.weight;
            let semantic_weight = original_weight + (1.0 - similarity);
            modified_costs.push(semantic_weight);
        }

        // Re-run pathfinding with modified weights
        let semantic_path = self.find_path_with_weights(
            source,
            target,
            &modified_costs
        )?;

        let avg_relevance = query_similarities.values().sum::<f32>()
            / query_similarities.len() as f32;

        Ok(SemanticPath {
            nodes: semantic_path.nodes,
            edges: semantic_path.edges,
            total_cost: semantic_path.total_cost,
            semantic_relevance: avg_relevance,
            query_similarities,
        })
    }
}
```

#### B. Query-Guided Traversal

**File:** `src/algorithms/query_traversal.rs`

```rust
pub struct QueryTraversal {
    graph: Arc<GraphRepository>,
    similarity_calculator: Arc<SimilarityCalculator>,
    max_hops: usize,
    min_sentences: usize,
}

impl QueryTraversal {
    pub async fn traverse(
        &self,
        query: &str,
        anchor_node: NodeId,
    ) -> Result<TraversalResult, TraversalError> {
        let query_embedding = self.similarity_calculator.embed(query)?;

        let mut current_node = anchor_node;
        let mut visited = HashSet::new();
        let mut extracted_content = Vec::new();
        let mut path = Vec::new();
        let mut hop_count = 0;

        visited.insert(current_node.clone());
        path.push(current_node.clone());

        // Extract content from anchor
        let anchor_content = self.extract_node_content(&current_node).await?;
        extracted_content.extend(anchor_content);

        while extracted_content.len() < self.min_sentences && hop_count < self.max_hops {
            hop_count += 1;

            // Get connected nodes
            let neighbors = self.graph.get_neighbors(&current_node).await?;

            // Calculate similarity to query for each neighbor
            let mut neighbor_scores: Vec<(NodeId, f32)> = Vec::new();
            for neighbor in neighbors {
                if visited.contains(&neighbor) {
                    continue;
                }

                let neighbor_embedding = self.get_node_embedding(&neighbor).await?;
                let similarity = cosine_similarity(&query_embedding, &neighbor_embedding);
                neighbor_scores.push((neighbor, similarity));
            }

            // Sort by similarity (highest first)
            neighbor_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            if neighbor_scores.is_empty() {
                break;
            }

            // Move to most similar neighbor
            let (best_neighbor, best_similarity) = neighbor_scores[0].clone();
            current_node = best_neighbor.clone();
            visited.insert(best_neighbor.clone());
            path.push(best_neighbor.clone());

            // Extract content
            let content = self.extract_node_content(&best_neighbor).await?;
            extracted_content.extend(content);
        }

        Ok(TraversalResult {
            path,
            content: extracted_content,
            hop_count,
            avg_relevance: self.calculate_avg_relevance(&extracted_content, &query_embedding),
        })
    }
}
```

### 3.3 API Endpoints

**File:** `src/handlers/semantic_pathfinding_handler.rs`

```rust
#[derive(Debug, Deserialize)]
pub struct SemanticPathRequest {
    pub source: String,
    pub target: String,
    pub query: Option<String>,
    pub algorithm: PathfindingAlgorithm,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PathfindingAlgorithm {
    SemanticSSP,
    QueryTraversal,
    ChunkTraversal,
    LLMGuided,
}

pub async fn find_semantic_path(
    app_state: web::Data<AppState>,
    req: web::Json<SemanticPathRequest>,
) -> Result<HttpResponse, ActixError> {
    let result = match req.algorithm {
        PathfindingAlgorithm::SemanticSSP => {
            app_state.semantic_ssp
                .calculate_path(&req.source, &req.target, req.query.as_deref())
                .await?
        }
        PathfindingAlgorithm::QueryTraversal => {
            let query = req.query.as_ref()
                .ok_or(ActixError::BadRequest("Query required for query traversal"))?;
            app_state.query_traversal
                .traverse(query, &req.source)
                .await?
        }
        PathfindingAlgorithm::ChunkTraversal => {
            app_state.chunk_traversal
                .traverse(&req.source, &req.target)
                .await?
        }
        PathfindingAlgorithm::LLMGuided => {
            let query = req.query.as_ref()
                .ok_or(ActixError::BadRequest("Query required for LLM-guided traversal"))?;
            app_state.llm_traversal
                .traverse(query, &req.source, &req.target)
                .await?
        }
    };

    Ok(HttpResponse::Ok().json(result))
}

pub fn configure_routes(cfg: &mut web::ServiceConfig) {
    cfg.service(
        web::scope("/semantic-pathfinding")
            .route("/find-path", web::post().to(find_semantic_path))
            .route("/algorithms", web::get().to(list_algorithms))
    );
}
```

### 3.4 Frontend Integration

**File:** `client/src/features/pathfinding/components/SemanticPathfinder.tsx`

```tsx
export const SemanticPathfinder: React.FC = () => {
  const [sourceNode, setSourceNode] = useState('');
  const [targetNode, setTargetNode] = useState('');
  const [query, setQuery] = useState('');
  const [algorithm, setAlgorithm] = useState<PathfindingAlgorithm>('semantic_ssp');
  const [path, setPath] = useState<any>(null);

  const handleFindPath = async () => {
    const response = await axios.post('/api/semantic-pathfinding/find-path', {
      source: sourceNode,
      target: targetNode,
      query: query || null,
      algorithm,
    });

    setPath(response.data);
    // Highlight path in visualization
  };

  return (
    <div className="semantic-pathfinder">
      <h3>Intelligent Pathfinding</h3>

      <div className="path-inputs">
        <input
          placeholder="Source Node ID"
          value={sourceNode}
          onChange={(e) => setSourceNode(e.target.value)}
        />
        <input
          placeholder="Target Node ID"
          value={targetNode}
          onChange={(e) => setTargetNode(e.target.value)}
        />
      </div>

      <div className="query-input">
        <textarea
          placeholder="Optional: Describe what path you're looking for"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          rows={2}
        />
      </div>

      <div className="algorithm-select">
        <label>Algorithm:</label>
        <select value={algorithm} onChange={(e) => setAlgorithm(e.target.value as any)}>
          <option value="semantic_ssp">Semantic Shortest Path</option>
          <option value="query_traversal">Query-Guided Traversal</option>
          <option value="chunk_traversal">Similarity Exploration</option>
          <option value="llm_guided">AI-Guided (Premium)</option>
        </select>
      </div>

      <button onClick={handleFindPath}>Find Path</button>

      {path && (
        <div className="path-results">
          <div>Path Length: {path.nodes.length} nodes</div>
          <div>Hops: {path.hop_count}</div>
          <div>Relevance: {(path.avg_relevance * 100).toFixed(1)}%</div>
          {/* Visualize path */}
        </div>
      )}
    </div>
  );
};
```

---

## 4. Implementation Phases

### Phase 1: Type System & Schema (Week 1)
- [ ] Add NodeType/EdgeType enums to models
- [ ] Migrate Neo4j schema to include types
- [ ] Implement SchemaService
- [ ] Add schema API endpoint
- [ ] Test type system

### Phase 2: Semantic Forces (Week 2-3)
- [ ] Implement CUDA kernels (DAG, clustering, collision)
- [ ] Create SemanticPhysicsEngine wrapper
- [ ] Add HierarchyCalculator
- [ ] Implement API endpoints
- [ ] Add frontend controls
- [ ] Test and tune forces

### Phase 3: Natural Language Queries (Week 3-4)
- [ ] Implement LLMService for Cypher generation
- [ ] Add NL query handler
- [ ] Create frontend query interface
- [ ] Test with various query types
- [ ] Optimize prompt engineering

### Phase 4: Semantic Pathfinding (Week 4-5)
- [ ] Enhance SSSP with semantic weighting
- [ ] Implement QueryTraversal algorithm
- [ ] Implement ChunkTraversal algorithm
- [ ] Add LLM-guided traversal (optional)
- [ ] Create pathfinding API
- [ ] Add frontend pathfinder UI
- [ ] Performance testing

### Phase 5: Integration & Documentation (Week 6)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Create comprehensive documentation
- [ ] User guides and examples
- [ ] Deploy and monitor

---

## 5. Testing Strategy

### Unit Tests
- CUDA kernel correctness
- Algorithm correctness
- Type system validation

### Integration Tests
- End-to-end semantic force application
- Natural language query accuracy
- Pathfinding result quality

### Performance Tests
- CUDA kernel performance
- Large graph handling (100k+ nodes)
- Query response times

---

## 6. Documentation Requirements

### User Documentation
- Semantic forces guide with examples
- Natural language query tutorial
- Intelligent pathfinding guide

### Developer Documentation
- CUDA kernel implementation details
- Algorithm descriptions
- API reference

### Architecture Documentation
- System architecture updates
- Data flow diagrams
- Integration points

---

**Status:** Ready for Implementation
**Estimated Effort:** 6 weeks
**Priority:** High
**Risk Level:** Medium

---

This integration will transform VisionFlow from a general-purpose graph visualizer into an intelligent, semantic knowledge exploration platform.
