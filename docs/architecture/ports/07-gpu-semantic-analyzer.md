# GpuSemanticAnalyzer Port

## Purpose

The **GpuSemanticAnalyzer** port provides GPU-accelerated semantic analysis, clustering, and pathfinding algorithms for knowledge graphs. It includes community detection, shortest path computation, and node importance analysis.

## Location

- **Trait Definition**: `src/ports/gpu_semantic_analyzer.rs`
- **Adapter Implementation**: `src/adapters/cuda_semantic_analyzer.rs`

## Interface

```rust
#[async_trait]
pub trait GpuSemanticAnalyzer: Send + Sync {
    // Initialization
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<()>;
    async fn update_graph_data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    // Community detection
    async fn detect_communities(&mut self, algorithm: ClusteringAlgorithm) -> Result<CommunityDetectionResult>;

    // Pathfinding
    async fn compute_shortest_paths(&mut self, source_node_id: u32) -> Result<PathfindingResult>;
    async fn compute_sssp_distances(&mut self, source_node_id: u32) -> Result<Vec<f32>>;
    async fn compute_all_pairs_shortest_paths(&mut self) -> Result<HashMap<(u32, u32), Vec<u32>>>;
    async fn compute_landmark_apsp(&mut self, num_landmarks: usize) -> Result<Vec<Vec<f32>>>;

    // Semantic constraints
    async fn generate_semantic_constraints(&mut self, config: SemanticConstraintConfig) -> Result<ConstraintSet>;

    // Layout optimization
    async fn optimize_layout(&mut self, constraints: &ConstraintSet, max_iterations: usize) -> Result<OptimizationResult>;

    // Node importance
    async fn analyze_node_importance(&mut self, algorithm: ImportanceAlgorithm) -> Result<HashMap<u32, f32>>;

    // Cache management
    async fn invalidate_pathfinding_cache(&mut self) -> Result<()>;

    // Statistics
    async fn get_statistics(&self) -> Result<SemanticStatistics>;
}
```

## Types

### ClusteringAlgorithm

Community detection algorithms:

```rust
pub enum ClusteringAlgorithm {
    Louvain,
    LabelPropagation,
    ConnectedComponents,
    HierarchicalClustering { min_cluster_size: usize },
}
```

### CommunityDetectionResult

Clustering results:

```rust
pub struct CommunityDetectionResult {
    pub clusters: HashMap<u32, usize>,        // node_id -> cluster_id
    pub cluster_sizes: HashMap<usize, usize>, // cluster_id -> size
    pub modularity: f32,
    pub computation_time_ms: f32,
}
```

### PathfindingResult

SSSP results:

```rust
pub struct PathfindingResult {
    pub source_node: u32,
    pub distances: HashMap<u32, f32>,  // node_id -> distance
    pub paths: HashMap<u32, Vec<u32>>, // node_id -> path
    pub computation_time_ms: f32,
}
```

### ImportanceAlgorithm

Node centrality algorithms:

```rust
pub enum ImportanceAlgorithm {
    PageRank { damping: f32, max_iterations: usize },
    Betweenness,
    Closeness,
    Eigenvector,
    Degree,
}
```

## Usage Examples

### Community Detection

```rust
let mut analyzer: Box<dyn GpuSemanticAnalyzer> = Box::new(CudaSemanticAnalyzer::new()?);

// Initialize with graph
let graph = Arc::new(load_graph().await?);
analyzer.initialize(graph).await?;

// Run Louvain clustering
let result = analyzer.detect_communities(ClusteringAlgorithm::Louvain).await?;

println!("Found {} clusters with modularity {}", result.cluster_sizes.len(), result.modularity);
for (node_id, cluster_id) in &result.clusters {
    println!("Node {} → Cluster {}", node_id, cluster_id);
}

// Label propagation (faster, less accurate)
let result = analyzer.detect_communities(ClusteringAlgorithm::LabelPropagation).await?;
```

### Single-Source Shortest Paths

```rust
// Compute SSSP from node 0
let result = analyzer.compute_shortest_paths(0).await?;

println!("Shortest paths from node 0:");
for (target_id, distance) in &result.distances {
    if let Some(path) = result.paths.get(target_id) {
        println!("  → Node {}: distance = {}, path = {:?}", target_id, distance, path);
    }
}

// Or get just distances (faster)
let distances = analyzer.compute_sssp_distances(0).await?;
println!("Distance to node 5: {}", distances[5]);
```

### All-Pairs Shortest Paths

```rust
// Exact APSP (slower, for small graphs)
let paths = analyzer.compute_all_pairs_shortest_paths().await?;

for ((source, target), path) in &paths {
    println!("Path {} → {}: {:?}", source, target, path);
}

// Landmark-based approximation (faster, for large graphs)
let num_landmarks = (graph.nodes.len() as f32).sqrt() as usize;
let distance_matrix = analyzer.compute_landmark_apsp(num_landmarks).await?;

println!("Approximate distance from node 0 to node 5: {}", distance_matrix[0][5]);
```

### Node Importance Analysis

```rust
// PageRank
let importance = analyzer.analyze_node_importance(
    ImportanceAlgorithm::PageRank {
        damping: 0.85,
        max_iterations: 100,
    }
).await?;

let mut ranked: Vec<_> = importance.iter().collect();
ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

println!("Top 10 most important nodes:");
for (node_id, score) in ranked.iter().take(10) {
    println!("  Node {}: {:.4}", node_id, score);
}

// Betweenness centrality
let betweenness = analyzer.analyze_node_importance(ImportanceAlgorithm::Betweenness).await?;

// Degree centrality (fastest)
let degree = analyzer.analyze_node_importance(ImportanceAlgorithm::Degree).await?;
```

### Semantic Constraints Generation

```rust
// Generate constraints for layout optimization
let config = SemanticConstraintConfig {
    similarity_threshold: 0.7,
    enable_clustering_constraints: true,
    enable_importance_constraints: true,
    enable_topic_constraints: false,
    max_constraints: 1000,
};

let constraints = analyzer.generate_semantic_constraints(config).await?;
println!("Generated {} constraints", constraints.len());
```

### Layout Optimization

```rust
// Optimize layout with constraints
let result = analyzer.optimize_layout(&constraints, 100).await?;

println!("Optimization result:");
println!("  Converged: {}", result.converged);
println!("  Iterations: {}", result.iterations);
println!("  Final stress: {}", result.final_stress);
println!("  Time: {}ms", result.computation_time_ms);
```

## Implementation Notes

### CUDA SSSP Kernel

GPU implementation of single-source shortest paths:

```cuda
// src/cuda/sssp_compact.cu
__global__ void sssp_kernel(
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const float* csr_weights,
    float* distances,
    int* updated,
    int num_nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    float my_dist = distances[tid];
    int start = csr_row_ptr[tid];
    int end = csr_row_ptr[tid + 1];

    for (int i = start; i < end; i++) {
        int neighbor = csr_col_idx[i];
        float edge_weight = csr_weights[i];
        float new_dist = my_dist + edge_weight;

        // Atomic min for relaxation
        atomicMin_float(&distances[neighbor], new_dist);
        *updated = 1;
    }
}
```

### Louvain Clustering

Community detection using modularity optimization:

```rust
async fn detect_communities_louvain(&mut self) -> Result<CommunityDetectionResult> {
    let start = Instant::now();

    // Phase 1: Local modularity optimization
    let mut communities = self.initialize_communities();
    let mut improved = true;

    while improved {
        improved = false;
        for node in 0..self.num_nodes {
            let best_community = self.find_best_community(node, &communities);
            if best_community != communities[node] {
                communities[node] = best_community;
                improved = true;
            }
        }
    }

    // Phase 2: Graph coarsening
    let coarse_graph = self.build_coarse_graph(&communities);

    // Compute modularity
    let modularity = self.compute_modularity(&communities);

    // Count cluster sizes
    let mut cluster_sizes = HashMap::new();
    for &community_id in &communities {
        *cluster_sizes.entry(community_id).or_insert(0) += 1;
    }

    Ok(CommunityDetectionResult {
        clusters: communities.iter().enumerate()
            .map(|(i, &c)| (i as u32, c))
            .collect(),
        cluster_sizes,
        modularity,
        computation_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}
```

### PageRank Implementation

GPU-accelerated PageRank:

```cuda
__global__ void pagerank_kernel(
    const int* csr_row_ptr,
    const int* csr_col_idx,
    const float* old_scores,
    float* new_scores,
    const int* out_degrees,
    int num_nodes,
    float damping
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    float sum = 0.0f;
    int start = csr_row_ptr[tid];
    int end = csr_row_ptr[tid + 1];

    // Sum contributions from incoming edges
    for (int i = start; i < end; i++) {
        int neighbor = csr_col_idx[i];
        int degree = out_degrees[neighbor];
        sum += old_scores[neighbor] / degree;
    }

    new_scores[tid] = (1.0f - damping) / num_nodes + damping * sum;
}
```

## Performance Benchmarks

Target performance (CUDA adapter, RTX 3090):

**Community Detection**:
- Louvain (1,000 nodes): < 50ms
- Louvain (10,000 nodes): < 200ms
- Label Propagation (10,000 nodes): < 50ms

**Pathfinding**:
- SSSP (1,000 nodes): < 10ms
- SSSP (10,000 nodes): < 50ms
- Landmark APSP (10,000 nodes, 100 landmarks): < 500ms

**Node Importance**:
- PageRank (1,000 nodes, 100 iter): < 50ms
- PageRank (10,000 nodes, 100 iter): < 200ms
- Degree centrality: < 5ms

**GPU Speedup vs CPU**:
- SSSP: 20-50x faster
- PageRank: 30-100x faster
- Louvain: 10-30x faster

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum GpuSemanticAnalyzerError {
    #[error("GPU not available")]
    GpuNotAvailable,

    #[error("Analysis error: {0}")]
    AnalysisError(String),

    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    #[error("Algorithm not supported: {0}")]
    UnsupportedAlgorithm(String),

    #[error("CUDA error: {0}")]
    CudaError(String),
}
```

## Testing

### Mock Implementation

```rust
pub struct MockGpuSemanticAnalyzer {
    graph: Option<Arc<GraphData>>,
    cache: HashMap<u32, Vec<f32>>,
}

#[async_trait]
impl GpuSemanticAnalyzer for MockGpuSemanticAnalyzer {
    async fn compute_shortest_paths(&mut self, source_node_id: u32) -> Result<PathfindingResult> {
        // Simple BFS for testing
        let graph = self.graph.as_ref().ok_or(GpuSemanticAnalyzerError::InvalidGraph("".into()))?;

        let mut distances = HashMap::new();
        let mut paths = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(source_node_id, 0.0);
        paths.insert(source_node_id, vec![source_node_id]);
        queue.push_back(source_node_id);

        while let Some(node) = queue.pop_front() {
            for edge in &graph.edges {
                if edge.source == node {
                    let dist = distances[&node] + edge.weight;
                    if !distances.contains_key(&edge.target) || dist < distances[&edge.target] {
                        distances.insert(edge.target, dist);
                        let mut path = paths[&node].clone();
                        path.push(edge.target);
                        paths.insert(edge.target, path);
                        queue.push_back(edge.target);
                    }
                }
            }
        }

        Ok(PathfindingResult {
            source_node: source_node_id,
            distances,
            paths,
            computation_time_ms: 1.0,
        })
    }
}
```

## References

- **Graph Algorithms**: https://en.wikipedia.org/wiki/Graph_algorithm
- **PageRank**: https://en.wikipedia.org/wiki/PageRank
- **Louvain Method**: https://en.wikipedia.org/wiki/Louvain_method
- **GPU Graph Analytics**: https://gunrock.github.io/docs/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
