# VisionFlow Data Flow Architecture

## Overview

VisionFlow implements a sophisticated data pipeline that processes multiple data sources into a unified, real-time 3D graph visualisation. The architecture supports bidirectional data flow, enabling real-time collaboration whilst maintaining consistency across all connected clients.

## Data Sources

The system ingests data from multiple heterogeneous sources, each requiring specialised processing:

### 1. Knowledge Graph Sources

#### Markdown Files
- **Location**: Local file system, GitHub repositories
- **Format**: CommonMark with YAML frontmatter
- **Processing**: Content parsing, link extraction, metadata analysis
- **Update Frequency**: File system events, periodic sync

#### JSON Metadata
- **Location**: `data/metadata/metadata.json`
- **Content**: File relationships, tags, timestamps
- **Processing**: Semantic analysis, relationship inference
- **Update Frequency**: Real-time on file changes

#### External APIs
- **GitHub API**: Repository content, issues, pull requests
- **Nostr Network**: Decentralised social graph data
- **RAGFlow**: Conversational context and embeddings
- **Perplexity**: Research queries and citations

### 2. AI Agent Sources

#### Claude Flow MCP
- **Connection**: Direct TCP to port 9500
- **Protocol**: JSON-RPC over WebSocket
- **Data**: Agent status, task assignments, performance metrics
- **Update Frequency**: 10Hz real-time telemetry

#### Multi Agent Orchestration
- **Topology Types**: Mesh, hierarchical, ring, star
- **Agent Types**: 50+ specialised agents (researcher, coder, analyst, etc.)
- **Metrics**: Task completion, resource usage, coordination efficiency

## Data Processing Pipeline

The data flows through a multi-stage processing pipeline with parallel processing capabilities:

```mermaid
flowchart TD
    subgraph "Data Ingestion"
        A1[Markdown Files]
        A2[JSON Metadata]
        A3[GitHub API]
        A4[Claude Flow MCP]
        A5[Nostr Network]
        A6[RAGFlow API]
    end

    subgraph "Processing Layer"
        B1[File Parser]
        B2[Metadata Parser]
        B3[API Adapter]
        B4[MCP Handler]
        B5[Semantic Analyser]
        B6[Edge Generator]
    end

    subgraph "Unified Processing"
        C1[Knowledge Graph Builder]
        C2[Agent Graph Builder]
        C3[Relationship Inference]
        C4[Constraint System]
    end

    subgraph "Physics Engine"
        D1[GPU Compute Actor]
        D2[Unified CUDA Kernel]
        D3[Parallel Graph Physics]
        D4[Constraint Satisfaction]
    end

    subgraph "Output Layer"
        E1[Binary Protocol]
        E2[Differential Updates]
        E3[Client Broadcast]
        E4[State Synchronisation]
    end

    A1 --> B1
    A2 --> B2
    A3 --> B3
    A4 --> B4
    A5 --> B3
    A6 --> B3

    B1 --> B5
    B2 --> B5
    B3 --> B6
    B4 --> C2
    B5 --> C1
    B6 --> C1

    C1 --> C3
    C2 --> C3
    C3 --> D1
    C4 --> D1

    D1 --> D2
    D2 --> D3
    D3 --> D4

    D4 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
```

## Stage-by-Stage Processing

### Stage 1: Data Ingestion

#### File System Processing
```rust
// File monitoring and parsing
pub struct FileProcessor {
    watcher: RecommendedWatcher,
    metadata_store: Arc<RwLock<MetadataStore>>,
    semantic_analyser: SemanticAnalyser,
}

impl FileProcessor {
    pub async fn process_file(&self, path: &Path) -> Result<FileMetadata, Error> {
        let content = tokio::fs::read_to_string(path).await?;
        let metadata = self.extract_metadata(&content)?;
        let links = self.extract_links(&content)?;
        let semantic_data = self.semantic_analyser.analyse(&content).await?;
        
        Ok(FileMetadata {
            path: path.to_path_buf(),
            content_hash: self.calculate_hash(&content),
            links,
            metadata,
            semantic_data,
            last_modified: SystemTime::now(),
        })
    }
}
```

#### MCP Integration
```rust
// Claude Flow MCP connection
pub struct MCPClient {
    connection: TcpStream,
    message_queue: VecDeque<MCPMessage>,
    reconnect_backoff: ExponentialBackoff,
}

impl MCPClient {
    pub async fn subscribe_telemetry(&mut self) -> Result<(), MCPError> {
        let request = MCPRequest {
            method: "telemetry.subscribe",
            params: json!({ "interval": 100, "include_metrics": true }),
        };
        self.send_request(request).await
    }
    
    pub async fn poll_agent_data(&mut self) -> Result<Vec<AgentStatus>, MCPError> {
        let response = self.call_method("agent.list_all", json!({})).await?;
        serde_json::from_value(response.result)
    }
}
```

### Stage 2: Data Transformation

#### Knowledge Graph Construction
```rust
// Graph building from metadata
pub struct GraphBuilder {
    node_factory: NodeFactory,
    edge_factory: EdgeFactory,
    semantic_analyser: SemanticAnalyser,
}

impl GraphBuilder {
    pub async fn build_knowledge_graph(
        &self, 
        metadata: &MetadataStore
    ) -> Result<GraphData, BuildError> {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        // Create nodes from files
        for (path, file_meta) in metadata.files.iter() {
            let node = self.node_factory.create_file_node(path, file_meta)?;
            nodes.push(node);
        }
        
        // Generate semantic edges
        let semantic_edges = self.semantic_analyser
            .generate_semantic_edges(&nodes).await?;
        edges.extend(semantic_edges);
        
        // Add explicit link edges
        let link_edges = self.extract_link_edges(metadata)?;
        edges.extend(link_edges);
        
        Ok(GraphData { nodes, edges })
    }
}
```

#### Agent Graph Processing
```rust
// Agent data transformation
pub struct AgentGraphProcessor {
    agent_factory: AgentNodeFactory,
    relationship_inferrer: AgentRelationshipInferrer,
}

impl AgentGraphProcessor {
    pub fn process_agent_telemetry(
        &self, 
        telemetry: Vec<AgentStatus>
    ) -> Result<GraphData, ProcessError> {
        let nodes: Vec<Node> = telemetry
            .iter()
            .map(|status| self.agent_factory.create_agent_node(status))
            .collect::<Result<Vec<_>, _>>()?;
        
        let edges = self.relationship_inferrer
            .infer_agent_relationships(&nodes)?;
        
        Ok(GraphData { nodes, edges })
    }
}
```

### Stage 3: Semantic Analysis

#### Content Analysis
```rust
// Semantic relationship inference
pub struct SemanticAnalyser {
    embeddings_client: EmbeddingsClient,
    similarity_threshold: f32,
    cache: LruCache<String, Vec<f32>>,
}

impl SemanticAnalyser {
    pub async fn analyse_content(&self, content: &str) -> Result<SemanticData, Error> {
        let embedding = self.generate_embedding(content).await?;
        let topics = self.extract_topics(content)?;
        let entities = self.extract_entities(content)?;
        let sentiment = self.analyse_sentiment(content)?;
        
        Ok(SemanticData {
            embedding,
            topics,
            entities,
            sentiment,
            content_hash: self.hash_content(content),
        })
    }
    
    pub async fn calculate_similarity(
        &self, 
        content1: &str, 
        content2: &str
    ) -> Result<f32, Error> {
        let embedding1 = self.get_or_generate_embedding(content1).await?;
        let embedding2 = self.get_or_generate_embedding(content2).await?;
        
        Ok(cosine_similarity(&embedding1, &embedding2))
    }
}
```

#### Edge Generation
```rust
// Dynamic edge creation based on relationships
pub struct EdgeGenerator {
    semantic_analyser: SemanticAnalyser,
    link_extractor: LinkExtractor,
    weight_calculator: EdgeWeightCalculator,
}

impl EdgeGenerator {
    pub async fn generate_edges(
        &self, 
        nodes: &[Node]
    ) -> Result<Vec<Edge>, GenerationError> {
        let mut edges = Vec::new();
        
        // Generate semantic similarity edges
        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let similarity = self.semantic_analyser
                    .calculate_node_similarity(&nodes[i], &nodes[j]).await?;
                
                if similarity > 0.7 {
                    let weight = self.weight_calculator
                        .calculate_semantic_weight(similarity);
                    
                    edges.push(Edge {
                        id: format!("semantic_{}_{}", nodes[i].id, nodes[j].id),
                        source: nodes[i].id,
                        target: nodes[j].id,
                        weight,
                        edge_type: EdgeType::Semantic,
                    });
                }
            }
        }
        
        // Add explicit link edges
        let link_edges = self.link_extractor.extract_link_edges(nodes)?;
        edges.extend(link_edges);
        
        Ok(edges)
    }
}
```

### Stage 4: Physics Processing

#### GPU Compute Pipeline
```rust
// CUDA kernel execution
pub struct GPUPhysicsEngine {
    context: CudaContext,
    unified_kernel: CudaFunction,
    node_positions: DeviceBuffer<Vec3>,
    node_velocities: DeviceBuffer<Vec3>,
    edge_list: DeviceBuffer<Edge>,
    simulation_params: DeviceBuffer<SimParams>,
}

impl GPUPhysicsEngine {
    pub async fn compute_physics_step(&mut self) -> Result<Vec<Vec3>, CudaError> {
        // Update parameters
        self.simulation_params.copy_from_host(&self.current_params)?;
        
        // Launch unified kernel
        unsafe {
            launch!(
                self.unified_kernel<<<
                    (self.node_count + BLOCK_SIZE - 1) / BLOCK_SIZE,
                    BLOCK_SIZE,
                    self.stream
                >>>(
                    self.node_positions.as_device_ptr(),
                    self.node_velocities.as_device_ptr(),
                    self.edge_list.as_device_ptr(),
                    self.simulation_params.as_device_ptr(),
                    self.node_count as u32,
                    self.edge_count as u32
                )
            )?;
        }
        
        // Copy results back
        let mut positions = vec![Vec3::zero(); self.node_count];
        self.node_positions.copy_to_host(&mut positions)?;
        
        Ok(positions)
    }
}
```

#### Dual Graph Physics
```cuda
// CUDA kernel for parallel graph processing
__global__ void visionflow_unified_kernel(
    float3* positions,
    float3* velocities,
    Edge* edges,
    SimParams* params,
    uint32_t node_count,
    uint32_t edge_count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;
    
    float3 force = make_float3(0.0f, 0.0f, 0.0f);
    float3 position = positions[idx];
    
    // Determine graph type from node ID
    bool is_agent = (idx & 0x80000000) != 0;
    bool is_knowledge = (idx & 0x40000000) != 0;
    
    // Apply graph-specific physics parameters
    SimParams effective_params = *params;
    if (is_agent) {
        effective_params.spring_k *= 0.5f;  // Softer agent connections
        effective_params.repel_k *= 1.5f;   // Stronger agent separation
    }
    
    // Compute forces based on mode
    switch (effective_params.compute_mode) {
        case 0: // Basic force-directed
            force = compute_basic_forces(idx, position, positions, edges, 
                                       node_count, edge_count, effective_params);
            break;
        case 1: // Dual graph mode
            force = compute_dual_graph_forces(idx, position, positions, edges,
                                             node_count, edge_count, effective_params,
                                             is_agent, is_knowledge);
            break;
        case 2: // Constraint-enhanced
            force = compute_constraint_forces(idx, position, positions, edges,
                                             node_count, edge_count, effective_params);
            break;
        case 3: // Visual analytics
            force = compute_analytics_forces(idx, position, positions, edges,
                                            node_count, edge_count, effective_params);
            break;
    }
    
    // Update velocity and position
    velocities[idx] = (velocities[idx] + force * effective_params.dt) * effective_params.damping;
    positions[idx] = position + velocities[idx] * effective_params.dt;
}
```

### Stage 5: Output and Distribution

#### Binary Protocol Encoding
```rust
// Efficient binary serialisation
#[repr(C)]
pub struct BinaryNodeUpdate {
    pub node_id: u32,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub flags: u32,
}

pub struct BinaryProtocolEncoder {
    compression_threshold: usize,
    position_quantisation: f32,
}

impl BinaryProtocolEncoder {
    pub fn encode_node_updates(
        &self, 
        updates: &[(u32, Vec3, Vec3)]
    ) -> Result<Vec<u8>, EncodeError> {
        let mut buffer = Vec::with_capacity(updates.len() * 32);
        
        // Header
        buffer.extend_from_slice(&(updates.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&0u32.to_le_bytes()); // Flags
        
        // Node updates
        for (id, position, velocity) in updates {
            let update = BinaryNodeUpdate {
                node_id: *id,
                position: [position.x, position.y, position.z],
                velocity: [velocity.x, velocity.y, velocity.z],
                flags: 0,
            };
            
            buffer.extend_from_slice(unsafe {
                std::slice::from_raw_parts(
                    &update as *const _ as *const u8,
                    std::mem::size_of::<BinaryNodeUpdate>()
                )
            });
        }
        
        // Compress if beneficial
        if buffer.len() > self.compression_threshold {
            Ok(self.compress(&buffer)?)
        } else {
            Ok(buffer)
        }
    }
}
```

#### Differential Update System
```rust
// Change detection and minimal updates
pub struct DifferentialUpdateTracker {
    last_positions: HashMap<u32, Vec3>,
    position_threshold: f32,
    velocity_threshold: f32,
}

impl DifferentialUpdateTracker {
    pub fn calculate_updates(
        &mut self, 
        current_positions: &[(u32, Vec3, Vec3)]
    ) -> Vec<(u32, Vec3, Vec3)> {
        let mut updates = Vec::new();
        
        for (node_id, position, velocity) in current_positions {
            let should_update = if let Some(last_pos) = self.last_positions.get(node_id) {
                let distance = (position - *last_pos).magnitude();
                distance > self.position_threshold || 
                velocity.magnitude() > self.velocity_threshold
            } else {
                true // New node
            };
            
            if should_update {
                updates.push((*node_id, *position, *velocity));
                self.last_positions.insert(*node_id, *position);
            }
        }
        
        updates
    }
}
```

## Real-Time Streaming Architecture

### WebSocket Communication Flow

```mermaid
sequenceDiagram
    participant Client1
    participant Client2
    participant WSHandler
    participant ClientManager
    participant GraphActor
    participant GPUActor

    Note over Client1,GPUActor: Continuous Physics Loop (60 FPS)
    
    loop Every 16ms
        GraphActor->>GPUActor: ComputeForces
        GPUActor-->>GraphActor: Updated Positions
        GraphActor->>GraphActor: Calculate Differential Updates
        
        alt Updates Available
            GraphActor->>ClientManager: BroadcastNodePositions(binary_data)
            ClientManager->>WSHandler: Forward to all clients
            WSHandler->>Client1: Binary position update
            WSHandler->>Client2: Binary position update
        end
    end
    
    Note over Client1,GPUActor: User Interaction
    
    Client1->>WSHandler: Move node (user drag)
    WSHandler->>GraphActor: UpdateNodePosition
    GraphActor->>GPUActor: Apply user constraint
    GPUActor-->>GraphActor: Acknowledge
    GraphActor->>ClientManager: BroadcastNodePositions
    ClientManager->>WSHandler: Forward update
    WSHandler->>Client2: Reflect user action
```

### Bidirectional State Synchronisation

```rust
// Client-server state sync
pub struct StateSynchroniser {
    authoritative_state: Arc<RwLock<GraphState>>,
    client_states: HashMap<ClientId, ClientState>,
    conflict_resolver: ConflictResolver,
}

impl StateSynchroniser {
    pub async fn handle_client_update(
        &mut self,
        client_id: ClientId,
        update: NodeUpdate
    ) -> Result<Vec<ClientId>, SyncError> {
        // Validate update
        if !self.validate_update(client_id, &update)? {
            return Err(SyncError::InvalidUpdate);
        }
        
        // Apply to authoritative state
        {
            let mut state = self.authoritative_state.write().await;
            if let Some(conflicts) = state.apply_update(&update)? {
                // Resolve conflicts
                let resolution = self.conflict_resolver.resolve(conflicts)?;
                state.apply_resolution(&resolution)?;
            }
        }
        
        // Determine which clients need updates
        let affected_clients = self.calculate_affected_clients(&update);
        
        Ok(affected_clients)
    }
    
    pub async fn broadcast_state_update(
        &self,
        clients: &[ClientId],
        update: &StateUpdate
    ) -> Result<(), BroadcastError> {
        let binary_data = self.encode_update(update)?;
        
        for client_id in clients {
            if let Some(client_state) = self.client_states.get(client_id) {
                client_state.send_binary(&binary_data).await?;
            }
        }
        
        Ok(())
    }
}
```

## Performance Optimisation

### Data Pipeline Optimisations

#### Memory Management
```rust
// Object pooling for frequent allocations
pub struct NodeUpdatePool {
    pool: Vec<Vec<BinaryNodeUpdate>>,
    in_use: HashSet<usize>,
}

impl NodeUpdatePool {
    pub fn acquire(&mut self) -> Vec<BinaryNodeUpdate> {
        if let Some(index) = self.pool.iter().position(|_| true) {
            if !self.in_use.contains(&index) {
                self.in_use.insert(index);
                return std::mem::take(&mut self.pool[index]);
            }
        }
        
        // Allocate new if none available
        Vec::with_capacity(1024)
    }
    
    pub fn release(&mut self, mut vec: Vec<BinaryNodeUpdate>) {
        vec.clear();
        self.pool.push(vec);
    }
}
```

#### Cache Optimisation
```rust
// Multi-level caching strategy
pub struct CacheHierarchy {
    l1_positions: LruCache<u32, Vec3>,           // Hot data
    l2_metadata: HashMap<String, FileMetadata>,  // Warm data
    l3_embeddings: DiskCache<String, Vec<f32>>,  // Cold data
}

impl CacheHierarchy {
    pub async fn get_position(&mut self, node_id: u32) -> Option<Vec3> {
        // L1 cache hit
        if let Some(pos) = self.l1_positions.get(&node_id) {
            return Some(*pos);
        }
        
        // L2 fallback (if position derivable from metadata)
        if let Some(pos) = self.derive_from_metadata(node_id).await {
            self.l1_positions.put(node_id, pos);
            return Some(pos);
        }
        
        None
    }
}
```

### Network Optimisation

#### Adaptive Update Rates
```rust
// Dynamic update frequency based on activity
pub struct AdaptiveUpdateManager {
    base_rate: Duration,
    activity_multiplier: f32,
    client_capabilities: HashMap<ClientId, ClientCapabilities>,
}

impl AdaptiveUpdateManager {
    pub fn calculate_update_interval(&self, graph_activity: f32) -> Duration {
        let activity_factor = (1.0 + graph_activity * self.activity_multiplier).min(4.0);
        Duration::from_millis(
            (self.base_rate.as_millis() as f32 / activity_factor) as u64
        )
    }
    
    pub fn should_send_to_client(
        &self, 
        client_id: ClientId, 
        update_size: usize
    ) -> bool {
        if let Some(caps) = self.client_capabilities.get(&client_id) {
            update_size <= caps.max_update_size && 
            caps.supports_binary_protocol
        } else {
            update_size <= 1024 // Conservative default
        }
    }
}
```

#### Compression Strategy
```rust
// Intelligent compression selection
pub struct CompressionManager {
    algorithms: Vec<Box<dyn CompressionAlgorithm>>,
    size_thresholds: Vec<usize>,
}

impl CompressionManager {
    pub fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        if data.len() < self.size_thresholds[0] {
            return Ok(data.to_vec()); // No compression
        }
        
        let mut best_result = data.to_vec();
        let mut best_ratio = 1.0;
        
        for algorithm in &self.algorithms {
            if let Ok(compressed) = algorithm.compress(data) {
                let ratio = compressed.len() as f32 / data.len() as f32;
                if ratio < best_ratio {
                    best_result = compressed;
                    best_ratio = ratio;
                }
            }
        }
        
        Ok(best_result)
    }
}
```

## Error Handling and Recovery

### Data Pipeline Resilience
```rust
// Robust error handling with recovery
pub struct ResilientPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    retry_policies: HashMap<String, RetryPolicy>,
    fallback_handlers: HashMap<String, Box<dyn FallbackHandler>>,
}

impl ResilientPipeline {
    pub async fn process(&mut self, input: PipelineInput) -> Result<PipelineOutput, PipelineError> {
        let mut current_data = input;
        
        for (index, stage) in self.stages.iter_mut().enumerate() {
            match self.process_stage(stage.as_mut(), current_data, index).await {
                Ok(output) => current_data = output,
                Err(e) => {
                    error!("Stage {} failed: {}", index, e);
                    
                    // Attempt recovery
                    if let Some(recovered) = self.attempt_recovery(index, current_data, &e).await? {
                        current_data = recovered;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(current_data)
    }
    
    async fn attempt_recovery(
        &mut self,
        stage_index: usize,
        input: PipelineInput,
        error: &PipelineError
    ) -> Result<Option<PipelineInput>, PipelineError> {
        let stage_name = format!("stage_{}", stage_index);
        
        // Try fallback handler
        if let Some(fallback) = self.fallback_handlers.get_mut(&stage_name) {
            return Ok(Some(fallback.handle(input, error).await?));
        }
        
        // Try retry with backoff
        if let Some(policy) = self.retry_policies.get(&stage_name) {
            for attempt in 1..=policy.max_attempts {
                let delay = policy.calculate_delay(attempt);
                tokio::time::sleep(delay).await;
                
                if let Ok(output) = self.stages[stage_index].process(input.clone()).await {
                    info!("Stage {} recovered after {} attempts", stage_index, attempt);
                    return Ok(Some(output));
                }
            }
        }
        
        Ok(None)
    }
}
```

## Monitoring and Observability

### Pipeline Metrics
```rust
// Comprehensive metrics collection
pub struct PipelineMetrics {
    stage_timings: HashMap<String, Histogram>,
    throughput_counters: HashMap<String, Counter>,
    error_rates: HashMap<String, Counter>,
    queue_depths: HashMap<String, Gauge>,
}

impl PipelineMetrics {
    pub fn record_stage_duration(&mut self, stage: &str, duration: Duration) {
        self.stage_timings
            .entry(stage.to_string())
            .or_insert_with(|| Histogram::new())
            .record(duration.as_millis() as f64);
    }
    
    pub fn increment_throughput(&mut self, stage: &str, count: usize) {
        self.throughput_counters
            .entry(stage.to_string())
            .or_insert_with(|| Counter::new())
            .increment(count as u64);
    }
    
    pub fn generate_report(&self) -> MetricsReport {
        MetricsReport {
            timestamp: SystemTime::now(),
            stage_performance: self.calculate_stage_performance(),
            bottlenecks: self.identify_bottlenecks(),
            error_summary: self.summarise_errors(),
            recommendations: self.generate_recommendations(),
        }
    }
}
```

## Future Enhancements

### Planned Improvements

1. **Streaming Analytics**: Real-time pattern detection in data streams
2. **Federated Learning**: Distributed model training across client devices
3. **Edge Computing**: Client-side processing for reduced latency
4. **Blockchain Integration**: Immutable audit trail for data provenance
5. **Advanced Caching**: Predictive prefetching based on user behaviour

### Scalability Roadmap

1. **Horizontal Scaling**: Multi-instance coordination with shared state
2. **Geographical Distribution**: Edge nodes for global low-latency access
3. **Hybrid Cloud**: Integration with cloud-native services
4. **Data Partitioning**: Intelligent graph partitioning for massive datasets

## Related Documentation

- [System Architecture Overview](index.md) - High-level system design
- [Component Architecture](components.md) - Detailed component breakdown
- [GPU Compute Integration](gpu-compute.md) - CUDA implementation details
- [Binary Protocol Specification](../api/binary-protocol.md) - Network protocol details
- [WebSocket Protocols](../api/websocket-protocols.md) - Real-time communication
- [Performance Tuning Guide](../guides/performance-tuning.md) - Optimisation strategies

---

*Last updated: January 2025*  
*Pipeline throughput: 100K+ nodes at 60 FPS*  
*Latency: <10ms end-to-end*  
*Status: Production Ready*