---
layout: default
title: "GpuSemanticAnalyzer Port"
parent: Ports
grand_parent: Architecture
nav_order: 99
---

# GpuSemanticAnalyzer Port

## Purpose

The **GpuSemanticAnalyzer** port provides GPU-accelerated semantic analysis, clustering, and pathfinding algorithms for knowledge graphs. It includes community detection, shortest path computation, and node importance analysis.

## Location

- **Trait Definition**: `src/ports/gpu-semantic-analyzer.rs`
- **Adapter Implementation**: `src/adapters/cuda-semantic-analyzer.rs`

## Interface

```rust
#[async-trait]
pub trait GpuSemanticAnalyzer: Send + Sync {
    // Initialization
    async fn initialize(&mut self, graph: Arc<GraphData>) -> Result<()>;
    async fn update-graph-data(&mut self, graph: Arc<GraphData>) -> Result<()>;

    // Community detection
    async fn detect-communities(&mut self, algorithm: ClusteringAlgorithm) -> Result<CommunityDetectionResult>;

    // Pathfinding
    async fn compute-shortest-paths(&mut self, source-node-id: u32) -> Result<PathfindingResult>;
    async fn compute-sssp-distances(&mut self, source-node-id: u32) -> Result<Vec<f32>>;
    async fn compute-all-pairs-shortest-paths(&mut self) -> Result<HashMap<(u32, u32), Vec<u32>>>;
    async fn compute-landmark-apsp(&mut self, num-landmarks: usize) -> Result<Vec<Vec<f32>>>;

    // Semantic constraints
    async fn generate-semantic-constraints(&mut self, config: SemanticConstraintConfig) -> Result<ConstraintSet>;

    // Layout optimization
    async fn optimize-layout(&mut self, constraints: &ConstraintSet, max-iterations: usize) -> Result<OptimizationResult>;

    // Node importance
    async fn analyze-node-importance(&mut self, algorithm: ImportanceAlgorithm) -> Result<HashMap<u32, f32>>;

    // Cache management
    async fn invalidate-pathfinding-cache(&mut self) -> Result<()>;

    // Statistics
    async fn get-statistics(&self) -> Result<SemanticStatistics>;
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
    HierarchicalClustering { min-cluster-size: usize },
}
```

### CommunityDetectionResult

Clustering results:

```rust
pub struct CommunityDetectionResult {
    pub clusters: HashMap<u32, usize>,        // node-id -> cluster-id
    pub cluster-sizes: HashMap<usize, usize>, // cluster-id -> size
    pub modularity: f32,
    pub computation-time-ms: f32,
}
```

### PathfindingResult

SSSP results:

```rust
pub struct PathfindingResult {
    pub source-node: u32,
    pub distances: HashMap<u32, f32>,  // node-id -> distance
    pub paths: HashMap<u32, Vec<u32>>, // node-id -> path
    pub computation-time-ms: f32,
}
```

### ImportanceAlgorithm

Node centrality algorithms:

```rust
pub enum ImportanceAlgorithm {
    PageRank { damping: f32, max-iterations: usize },
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
let graph = Arc::new(load-graph().await?);
analyzer.initialize(graph).await?;

// Run Louvain clustering
let result = analyzer.detect-communities(ClusteringAlgorithm::Louvain).await?;

println!("Found {} clusters with modularity {}", result.cluster-sizes.len(), result.modularity);
for (node-id, cluster-id) in &result.clusters {
    println!("Node {} → Cluster {}", node-id, cluster-id);
}

// Label propagation (faster, less accurate)
let result = analyzer.detect-communities(ClusteringAlgorithm::LabelPropagation).await?;
```

### Single-Source Shortest Paths

```rust
// Compute SSSP from node 0
let result = analyzer.compute-shortest-paths(0).await?;

println!("Shortest paths from node 0:");
for (target-id, distance) in &result.distances {
    if let Some(path) = result.paths.get(target-id) {
        println!("  → Node {}: distance = {}, path = {:?}", target-id, distance, path);
    }
}

// Or get just distances (faster)
let distances = analyzer.compute-sssp-distances(0).await?;
println!("Distance to node 5: {}", distances[5]);
```

### All-Pairs Shortest Paths

```rust
// Exact APSP (slower, for small graphs)
let paths = analyzer.compute-all-pairs-shortest-paths().await?;

for ((source, target), path) in &paths {
    println!("Path {} → {}: {:?}", source, target, path);
}

// Landmark-based approximation (faster, for large graphs)
let num-landmarks = (graph.nodes.len() as f32).sqrt() as usize;
let distance-matrix = analyzer.compute-landmark-apsp(num-landmarks).await?;

println!("Approximate distance from node 0 to node 5: {}", distance-matrix[0][5]);
```

### Node Importance Analysis

```rust
// PageRank
let importance = analyzer.analyze-node-importance(
    ImportanceAlgorithm::PageRank {
        damping: 0.85,
        max-iterations: 100,
    }
).await?;

let mut ranked: Vec<-> = importance.iter().collect();
ranked.sort-by(|a, b| b.1.partial-cmp(a.1).unwrap());

println!("Top 10 most important nodes:");
for (node-id, score) in ranked.iter().take(10) {
    println!("  Node {}: {:.4}", node-id, score);
}

// Betweenness centrality
let betweenness = analyzer.analyze-node-importance(ImportanceAlgorithm::Betweenness).await?;

// Degree centrality (fastest)
let degree = analyzer.analyze-node-importance(ImportanceAlgorithm::Degree).await?;
```

### Semantic Constraints Generation

```rust
// Generate constraints for layout optimization
let config = SemanticConstraintConfig {
    similarity-threshold: 0.7,
    enable-clustering-constraints: true,
    enable-importance-constraints: true,
    enable-topic-constraints: false,
    max-constraints: 1000,
};

let constraints = analyzer.generate-semantic-constraints(config).await?;
println!("Generated {} constraints", constraints.len());
```

### Layout Optimization

```rust
// Optimize layout with constraints
let result = analyzer.optimize-layout(&constraints, 100).await?;

println!("Optimization result:");
println!("  Converged: {}", result.converged);
println!("  Iterations: {}", result.iterations);
println!("  Final stress: {}", result.final-stress);
println!("  Time: {}ms", result.computation-time-ms);
```

## Implementation Notes

### CUDA SSSP Kernel

GPU implementation of single-source shortest paths:

```cuda
// src/cuda/sssp-compact.cu
--global-- void sssp-kernel(
    const int* csr-row-ptr,
    const int* csr-col-idx,
    const float* csr-weights,
    float* distances,
    int* updated,
    int num-nodes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num-nodes) return;

    float my-dist = distances[tid];
    int start = csr-row-ptr[tid];
    int end = csr-row-ptr[tid + 1];

    for (int i = start; i < end; i++) {
        int neighbor = csr-col-idx[i];
        float edge-weight = csr-weights[i];
        float new-dist = my-dist + edge-weight;

        // Atomic min for relaxation
        atomicMin-float(&distances[neighbor], new-dist);
        *updated = 1;
    }
}
```

### Louvain Clustering

Community detection using modularity optimization:

```rust
async fn detect-communities-louvain(&mut self) -> Result<CommunityDetectionResult> {
    let start = Instant::now();

    // Phase 1: Local modularity optimization
    let mut communities = self.initialize-communities();
    let mut improved = true;

    while improved {
        improved = false;
        for node in 0..self.num-nodes {
            let best-community = self.find-best-community(node, &communities);
            if best-community != communities[node] {
                communities[node] = best-community;
                improved = true;
            }
        }
    }

    // Phase 2: Graph coarsening
    let coarse-graph = self.build-coarse-graph(&communities);

    // Compute modularity
    let modularity = self.compute-modularity(&communities);

    // Count cluster sizes
    let mut cluster-sizes = HashMap::new();
    for &community-id in &communities {
        *cluster-sizes.entry(community-id).or-insert(0) += 1;
    }

    Ok(CommunityDetectionResult {
        clusters: communities.iter().enumerate()
            .map(|(i, &c)| (i as u32, c))
            .collect(),
        cluster-sizes,
        modularity,
        computation-time-ms: start.elapsed().as-secs-f32() * 1000.0,
    })
}
```

### PageRank Implementation

GPU-accelerated PageRank:

```cuda
--global-- void pagerank-kernel(
    const int* csr-row-ptr,
    const int* csr-col-idx,
    const float* old-scores,
    float* new-scores,
    const int* out-degrees,
    int num-nodes,
    float damping
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num-nodes) return;

    float sum = 0.0f;
    int start = csr-row-ptr[tid];
    int end = csr-row-ptr[tid + 1];

    // Sum contributions from incoming edges
    for (int i = start; i < end; i++) {
        int neighbor = csr-col-idx[i];
        int degree = out-degrees[neighbor];
        sum += old-scores[neighbor] / degree;
    }

    new-scores[tid] = (1.0f - damping) / num-nodes + damping * sum;
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

#[async-trait]
impl GpuSemanticAnalyzer for MockGpuSemanticAnalyzer {
    async fn compute-shortest-paths(&mut self, source-node-id: u32) -> Result<PathfindingResult> {
        // Simple BFS for testing
        let graph = self.graph.as-ref().ok-or(GpuSemanticAnalyzerError::InvalidGraph("".into()))?;

        let mut distances = HashMap::new();
        let mut paths = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(source-node-id, 0.0);
        paths.insert(source-node-id, vec![source-node-id]);
        queue.push-back(source-node-id);

        while let Some(node) = queue.pop-front() {
            for edge in &graph.edges {
                if edge.source == node {
                    let dist = distances[&node] + edge.weight;
                    if !distances.contains-key(&edge.target) || dist < distances[&edge.target] {
                        distances.insert(edge.target, dist);
                        let mut path = paths[&node].clone();
                        path.push(edge.target);
                        paths.insert(edge.target, path);
                        queue.push-back(edge.target);
                    }
                }
            }
        }

        Ok(PathfindingResult {
            source-node: source-node-id,
            distances,
            paths,
            computation-time-ms: 1.0,
        })
    }
}
```

---

---

## Related Documentation

- [GpuPhysicsAdapter Port](06-gpu-physics-adapter.md)
- [InferenceEngine Port](05-inference-engine.md)
- [OntologyRepository Port](04-ontology-repository.md)
- [Stress Majorization for GPU-Accelerated Graph Layout](../stress-majorization.md)
- [Semantic Physics Architecture](../semantic-physics.md)

## References

- **Graph Algorithms**: https://en.wikipedia.org/wiki/Graph-algorithm
- **PageRank**: https://en.wikipedia.org/wiki/PageRank
- **Louvain Method**: https://en.wikipedia.org/wiki/Louvain-method
- **GPU Graph Analytics**: https://gunrock.github.io/docs/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Phase**: 1.3 - Hexagonal Architecture Ports Layer
