---
title: Analytics Visualization Extension Design
description: **Agent**: Analytics Visualization Agent **Date**: 2025-11-28 **Task**: Protocol V3 Extension - Clustering, Anomaly Detection, Community Visualization
category: explanation
tags:
  - architecture
  - rest
  - websocket
  - neo4j
  - rust
updated-date: 2025-12-18
difficulty-level: advanced
---


# Analytics Visualization Extension Design

**Agent**: Analytics Visualization Agent
**Date**: 2025-11-28
**Task**: Protocol V3 Extension - Clustering, Anomaly Detection, Community Visualization

---

## Executive Summary

VisionFlow currently uses **Protocol V2** (36 bytes/node) for binary WebSocket updates. This document proposes extending to **Protocol V3** (48 bytes/node) to support advanced analytics visualization:

- **K-means Clustering** - Color-coded cluster visualization
- **Anomaly Detection** - Red glow effect for outliers (LOF algorithm)
- **Community Detection** - Louvain modularity visualization

---

## Current State Analysis

### 1. Binary Protocol V2 (36 bytes/node)

**File**: `/home/devuser/workspace/project/src/utils/binary_protocol.rs`

**Current Structure**:
```rust
pub struct WireNodeDataItemV2 {
    pub id: u32,                // 4 bytes - with type flags
    pub position: Vec3Data,     // 12 bytes (x, y, z)
    pub velocity: Vec3Data,     // 12 bytes (vx, vy, vz)
    pub sssp_distance: f32,     // 4 bytes - pathfinding
    pub sssp_parent: i32,       // 4 bytes - pathfinding
}
// Total: 36 bytes per node
```

**Node Type Flags** (in `id` field):
- Bit 31: Agent node flag (`0x80000000`)
- Bit 30: Knowledge node flag (`0x40000000`)
- Bits 26-28: Ontology type flags (Class/Individual/Property)
- Bits 0-29: Actual node ID (supports up to 1,073,741,823 nodes)

**Performance**:
- 100K nodes = 3.6 MB (vs 18 MB JSON = 80% savings)
- Latency: 10ms total (1.2ms serialize + 8ms network + 0.8ms deserialize)
- WebSocket endpoint: `ws://localhost:9090/ws?token=JWT`

### 2. Protocol V3 (48 bytes/node) - ALREADY IMPLEMENTED

**The protocol extension V3 is ALREADY fully implemented in the codebase!**

**File**: `/home/devuser/workspace/project/src/utils/binary_protocol.rs` (Lines 65-77)

```rust
pub struct WireNodeDataItemV3 {
    pub id: u32,                // 4 bytes
    pub position: Vec3Data,     // 12 bytes
    pub velocity: Vec3Data,     // 12 bytes
    pub sssp_distance: f32,     // 4 bytes
    pub sssp_parent: i32,       // 4 bytes
    pub cluster_id: u32,        // 4 bytes - NEW
    pub anomaly_score: f32,     // 4 bytes - NEW (0.0-1.0)
    pub community_id: u32,      // 4 bytes - NEW
}
// Total: 48 bytes per node
```

**Encoding Function** (Lines 475-553):
```rust
pub fn encode_node_data_with_analytics(
    nodes: &[(u32, BinaryNodeData)],
    agent_node_ids: &[u32],
    knowledge_node_ids: &[u32],
    ontology_class_ids: &[u32],
    ontology_individual_ids: &[u32],
    ontology_property_ids: &[u32],
    analytics: &HashMap<u32, (u32, f32, u32)>, // (cluster_id, anomaly_score, community_id)
) -> Vec<u8>
```

**Decoding Function** (Lines 846-996):
```rust
fn decode_node_data_v3(data: &[u8]) -> Result<Vec<(u32, BinaryNodeData)>, String>
```

### 3. Clustering Implementation (GPU-Accelerated)

**File**: `/home/devuser/workspace/project/src/actors/gpu/clustering_actor.rs`

**ClusteringActor** provides:
- **K-means clustering** (GPU-accelerated via CUDA)
- **Community detection** (Label Propagation + Louvain algorithms)
- **Cluster statistics** (silhouette score, coherence, sizes)

**Output Format**:
```rust
pub struct ClusteringStats {
    pub total_clusters: usize,
    pub average_cluster_size: f32,
    pub largest_cluster_size: usize,
    pub smallest_cluster_size: usize,
    pub silhouette_score: f32,
    pub computation_time_ms: u64,
}

pub struct Cluster {
    pub id: String,
    pub label: String,
    pub node_count: u32,
    pub coherence: f32,
    pub color: String,           // Hex color (e.g., "#4F46E5")
    pub keywords: Vec<String>,
    pub centroid: Option<[f32; 3]>,
    pub nodes: Vec<u32>,
}
```

**K-means GPU Pipeline**:
1. `RunKMeans` message → ClusteringActor
2. GPU CUDA kernel: `run_kmeans_clustering_with_metrics()`
3. Returns: `(assignments, centroids, inertia, iterations, converged)`
4. Converts to `Cluster` structs with colors and metadata

**Community Detection** (Louvain):
```rust
pub struct Community {
    pub id: String,
    pub nodes: Vec<u32>,
    pub internal_edges: usize,
    pub external_edges: usize,
    pub density: f32,
}

pub struct CommunityDetectionStats {
    pub total_communities: usize,
    pub modularity: f32,
    pub average_community_size: f32,
    pub largest_community: usize,
    pub smallest_community: usize,
    pub computation_time_ms: u64,
}
```

### 4. Anomaly Detection (LOF - Local Outlier Factor)

**File**: `/home/devuser/workspace/project/src/actors/gpu/anomaly_detection_actor.rs`

**AnomalyDetectionActor** provides:
- GPU-accelerated LOF (Local Outlier Factor) algorithm
- Real-time anomaly scoring (0.0-1.0 range)
- Configurable sensitivity and thresholds

**Output**:
```rust
pub struct AnomalyResult {
    pub node_id: u32,
    pub score: f32,              // 0.0 = normal, 1.0 = max anomaly
    pub is_anomaly: bool,
    pub neighbors: Vec<u32>,
    pub local_density: f32,
}
```

**REST API Endpoints** (already implemented):
- `POST /api/analytics/anomaly/toggle` - Enable/disable detection
- `GET /api/analytics/anomaly/current` - Get current anomalies

### 5. Client-Side TypeScript API

**File**: `/home/devuser/workspace/project/client/src/api/analyticsApi.ts`

**AnalyticsAPI** provides:
```typescript
export class AnalyticsAPI {
  // Clustering
  async runClustering(request: ClusteringRequest): Promise<string>
  async getTaskStatus(taskId: string): Promise<AnalysisTask>
  async getAnalysisResults(taskId: string): Promise<any>

  // Anomaly Detection
  async configureAnomalyDetection(config: AnomalyDetectionConfig): Promise<boolean>
  async getCurrentAnomalies(): Promise<any[]>

  // WebSocket Subscriptions
  subscribeToTask(taskId: string, callback: (task: AnalysisTask) => void): () => void

  // Performance Monitoring
  async getPerformanceStats(): Promise<GPUPerformanceStats>
  async getGPUStatus(): Promise<{ gpu_available, features, performance }>
}
```

**WebSocket Analytics Updates**:
- `ws://localhost:9090/analytics/ws`
- Real-time clustering progress
- Anomaly alerts (critical/high severity)
- GPU metrics (utilization, memory, temperature)

---

## Protocol V3 Extension Requirements

### Binary Format (48 bytes total)

```
┌──────────┬───────────────────────────────────────────┐
│ Offset   │ Field (Type)                               │
├──────────┼───────────────────────────────────────────┤
│ [0-3]    │ Node ID (u32) - with type flags           │
│ [4-7]    │ Position X (f32)                          │
│ [8-11]   │ Position Y (f32)                          │
│ [12-15]  │ Position Z (f32)                          │
│ [16-19]  │ Velocity X (f32)                          │
│ [20-23]  │ Velocity Y (f32)                          │
│ [24-27]  │ Velocity Z (f32)                          │
│ [28-31]  │ SSSP Distance (f32) - pathfinding         │
│ [32-35]  │ SSSP Parent (i32) - pathfinding           │
│ [36-39]  │ Cluster ID (u32) - K-means cluster        │ ← NEW
│ [40-43]  │ Anomaly Score (f32) - LOF score 0.0-1.0   │ ← NEW
│ [44-47]  │ Community ID (u32) - Louvain community    │ ← NEW
└──────────┴───────────────────────────────────────────┘

Total: 48 bytes per node
100K nodes = 4.8 MB (vs V2 = 3.6 MB, +33% bandwidth)
```

### Field Specifications

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| **Cluster ID** | u32 | 0 - 4,294,967,295 | K-means cluster assignment (0 = no cluster) |
| **Anomaly Score** | f32 | 0.0 - 1.0 | LOF anomaly score (0.0=normal, 1.0=extreme outlier) |
| **Community ID** | u32 | 0 - 4,294,967,295 | Louvain community assignment (0 = no community) |

### Usage Pattern

**Server**: Create analytics HashMap and encode with V3
```rust
use std::collections::HashMap;

// Run clustering, anomaly detection, community detection
let kmeans_result = gpu_compute.run_kmeans_clustering_with_metrics(5, 100, 0.001, 42)?;
let anomaly_result = gpu_compute.run_anomaly_detection_lof(k_neighbors, threshold)?;
let community_result = gpu_compute.run_louvain_community_detection(100, 1.0, 42)?;

// Build analytics HashMap: node_id -> (cluster_id, anomaly_score, community_id)
let mut analytics: HashMap<u32, (u32, f32, u32)> = HashMap::new();
for (node_id, cluster_id) in kmeans_result.0.iter().enumerate() {
    let anomaly_score = anomaly_result.get(node_id).map(|a| a.score).unwrap_or(0.0);
    let community_id = community_result.0.get(node_id).copied().unwrap_or(0);
    analytics.insert(node_id as u32, (*cluster_id as u32, anomaly_score, community_id as u32));
}

// Encode with Protocol V3
let binary_data = encode_node_data_with_analytics(
    &nodes,
    &agent_node_ids,
    &knowledge_node_ids,
    &ontology_class_ids,
    &ontology_individual_ids,
    &ontology_property_ids,
    &analytics,
);

// Send via WebSocket (binary message)
session.binary(binary_data).await?;
```

**Client**: Decode Protocol V3 and render analytics
```typescript
interface NodeUpdateV3 {
    id: number;
    position: [number, number, number];
    velocity: [number, number, number];
    sssp_distance: number;
    sssp_parent: number;
    cluster_id: number;          // NEW
    anomaly_score: number;       // NEW (0.0-1.0)
    community_id: number;        // NEW
}

class BinaryProtocolV3Parser {
    parseNodeUpdates(buffer: ArrayBuffer): NodeUpdateV3[] {
        const view = new DataView(buffer);
        const version = view.getUint8(0);  // Should be 3

        if (version !== 3) {
            throw new Error(`Expected Protocol V3, got V${version}`);
        }

        const nodeCount = (view.byteLength - 1) / 48;
        const updates: NodeUpdateV3[] = [];

        for (let i = 0; i < nodeCount; i++) {
            const offset = 1 + i * 48;  // +1 for version byte

            updates.push({
                id: view.getUint32(offset + 0, true),
                position: [
                    view.getFloat32(offset + 4, true),
                    view.getFloat32(offset + 8, true),
                    view.getFloat32(offset + 12, true),
                ],
                velocity: [
                    view.getFloat32(offset + 16, true),
                    view.getFloat32(offset + 20, true),
                    view.getFloat32(offset + 24, true),
                ],
                sssp_distance: view.getFloat32(offset + 28, true),
                sssp_parent: view.getInt32(offset + 32, true),
                cluster_id: view.getUint32(offset + 36, true),         // NEW
                anomaly_score: view.getFloat32(offset + 40, true),     // NEW
                community_id: view.getUint32(offset + 44, true),       // NEW
            });
        }

        return updates;
    }
}
```

---

## Client Rendering Requirements

### 1. Cluster Color-Coding

**Cluster Colors** (from ClusteringActor):
```rust
// ClusteringActor generates colors using golden ratio hue distribution
fn generate_cluster_color(cluster_id: usize) -> [f32; 3] {
    let hue = (cluster_id as f32 * 137.5) % 360.0; // Golden angle
    let saturation = 0.7 + (rng.gen::<f32>() * 0.3);
    let value = 0.8 + (rng.gen::<f32>() * 0.2);
    // Convert HSV to RGB
}
```

**Three.js Material Application**:
```typescript
function updateNodeClusterColor(node: THREE.Mesh, clusterId: number, clusterColors: Map<number, string>) {
    if (clusterId === 0) {
        // No cluster - use default color
        (node.material as THREE.MeshStandardMaterial).color.set(0x808080);
        return;
    }

    const hexColor = clusterColors.get(clusterId) || '#808080';
    (node.material as THREE.MeshStandardMaterial).color.set(hexColor);
    (node.material as THREE.MeshStandardMaterial).emissive.set(hexColor);
    (node.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.2;
}
```

### 2. Cluster Boundary Visualization (Convex Hulls)

**Convex Hull Computation** (QuickHull algorithm):
```typescript
import { ConvexGeometry } from 'three/examples/jsm/geometries/ConvexGeometry';

function createClusterBoundary(clusterNodes: THREE.Vector3[], clusterId: number, clusterColor: string): THREE.Mesh {
    const geometry = new ConvexGeometry(clusterNodes);
    const material = new THREE.MeshBasicMaterial({
        color: clusterColor,
        transparent: true,
        opacity: 0.15,
        side: THREE.DoubleSide,
        depthWrite: false,
    });

    const hull = new THREE.Mesh(geometry, material);
    hull.name = `cluster-boundary-${clusterId}`;
    hull.renderOrder = -1;  // Render behind nodes

    return hull;
}

// Update hulls when cluster assignments change
function updateClusterBoundaries(scene: THREE.Scene, updates: NodeUpdateV3[]) {
    // Group nodes by cluster_id
    const clusterMap = new Map<number, THREE.Vector3[]>();

    updates.forEach(node => {
        if (node.cluster_id > 0) {
            if (!clusterMap.has(node.cluster_id)) {
                clusterMap.set(node.cluster_id, []);
            }
            clusterMap.get(node.cluster_id)!.push(new THREE.Vector3(...node.position));
        }
    });

    // Remove old hulls
    scene.children.filter(c => c.name.startsWith('cluster-boundary-')).forEach(hull => {
        scene.remove(hull);
        hull.geometry.dispose();
        (hull.material as THREE.Material).dispose();
    });

    // Create new hulls (only for clusters with 4+ nodes)
    clusterMap.forEach((positions, clusterId) => {
        if (positions.length >= 4) {
            const hull = createClusterBoundary(positions, clusterId, clusterColors.get(clusterId)!);
            scene.add(hull);
        }
    });
}
```

### 3. Anomaly Red Glow Effect

**Shader-Based Glow** (Three.js post-processing):
```typescript
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass';

function setupAnomalyGlow(renderer: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.Camera) {
    const composer = new EffectComposer(renderer);
    composer.addPass(new RenderPass(scene, camera));

    const bloomPass = new UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        1.5,   // strength
        0.4,   // radius
        0.85   // threshold
    );
    composer.addPass(bloomPass);

    return composer;
}

function updateNodeAnomalyGlow(node: THREE.Mesh, anomalyScore: number) {
    const material = node.material as THREE.MeshStandardMaterial;

    if (anomalyScore > 0.5) {  // Threshold for "high anomaly"
        // Red glow intensity based on score
        const intensity = (anomalyScore - 0.5) * 2.0;  // Map 0.5-1.0 to 0.0-1.0

        material.emissive.set(0xff0000);  // Red
        material.emissiveIntensity = intensity * 2.0;  // Up to 2x brightness

        // Pulsing animation for critical anomalies
        if (anomalyScore > 0.8) {
            const pulse = Math.sin(Date.now() * 0.005) * 0.3 + 0.7;
            material.emissiveIntensity = intensity * 2.0 * pulse;
        }

        // Increase node size for visibility
        node.scale.setScalar(1.0 + anomalyScore * 0.5);
    } else {
        // Normal node - no glow
        material.emissive.set(0x000000);
        material.emissiveIntensity = 0.0;
        node.scale.setScalar(1.0);
    }
}
```

**Anomaly Layer Rendering** (selective bloom):
```typescript
// Render anomalies to separate layer for selective bloom
const ANOMALY_LAYER = 1;

function setupSelectiveBloom() {
    const anomalyLayer = new THREE.Layers();
    anomalyLayer.set(ANOMALY_LAYER);

    // Configure camera to see anomaly layer
    camera.layers.enable(ANOMALY_LAYER);

    // Bloom only affects anomaly layer
    const bloomComposer = new EffectComposer(renderer);
    bloomComposer.renderToScreen = false;

    const bloomPass = new UnrealBloomPass(/* ... */);
    bloomComposer.addPass(new RenderPass(scene, camera));
    bloomComposer.addPass(bloomPass);

    return { bloomComposer, anomalyLayer };
}

function updateNodeAnomalyLayer(node: THREE.Mesh, anomalyScore: number) {
    if (anomalyScore > 0.5) {
        node.layers.enable(ANOMALY_LAYER);
    } else {
        node.layers.disable(ANOMALY_LAYER);
    }
}
```

### 4. Community Visualization (Louvain)

**Community Clustering** (similar to K-means but modularity-based):
```typescript
interface CommunityMetadata {
    id: number;
    modularity: number;
    density: f32;
    internal_edges: number;
    external_edges: number;
    color: string;
}

// Fetch community metadata from API
async function fetchCommunityMetadata(taskId: string): Promise<Map<number, CommunityMetadata>> {
    const result = await analyticsAPI.getAnalysisResults(taskId);
    const communities = new Map<number, CommunityMetadata>();

    result.communities.forEach((comm: any) => {
        communities.set(parseInt(comm.id), {
            id: parseInt(comm.id),
            modularity: result.modularity,
            density: comm.density,
            internal_edges: comm.internal_edges,
            external_edges: comm.external_edges,
            color: generateCommunityColor(parseInt(comm.id)),
        });
    });

    return communities;
}

// Render community boundaries (larger, more diffuse than clusters)
function createCommunityBoundary(communityNodes: THREE.Vector3[], communityId: number, metadata: CommunityMetadata): THREE.Mesh {
    const geometry = new ConvexGeometry(communityNodes);

    // Expand boundary by 10% for visual separation
    geometry.scale(1.1, 1.1, 1.1);

    const material = new THREE.MeshBasicMaterial({
        color: metadata.color,
        transparent: true,
        opacity: 0.08 + metadata.density * 0.12,  // Denser = more visible
        side: THREE.DoubleSide,
        depthWrite: false,
        wireframe: true,
        wireframeLinewidth: 2,
    });

    const hull = new THREE.Mesh(geometry, material);
    hull.name = `community-boundary-${communityId}`;
    hull.renderOrder = -2;  // Behind cluster boundaries

    return hull;
}

// Color scheme: Use spectral gradient for modularity
function generateCommunityColor(communityId: number): string {
    const colors = [
        '#8B00FF',  // Violet
        '#0080FF',  // Blue
        '#00FFFF',  // Cyan
        '#00FF80',  // Spring green
        '#FFFF00',  // Yellow
        '#FF8000',  // Orange
        '#FF0000',  // Red
    ];
    return colors[communityId % colors.len()];
}
```

**Community Labels** (Three.js CSS2DRenderer):
```typescript
import { CSS2DRenderer, CSS2DObject } from 'three/examples/jsm/renderers/CSS2DRenderer';

function createCommunityLabel(communityId: number, centroid: THREE.Vector3, metadata: CommunityMetadata): CSS2DObject {
    const div = document.createElement('div');
    div.className = 'community-label';
    div.style.color = metadata.color;
    div.style.fontSize = '14px';
    div.style.fontWeight = 'bold';
    div.style.textShadow = '0 0 4px rgba(0,0,0,0.8)';
    div.textContent = `Community ${communityId} (${metadata.internal_edges} edges, ρ=${metadata.density.toFixed(2)})`;

    const label = new CSS2DObject(div);
    label.position.copy(centroid);
    label.name = `community-label-${communityId}`;

    return label;
}
```

---

## Neo4j Schema Additions

### Current Schema (V2)

**File**: `/home/devuser/workspace/project/docs/neo4j-rich-ontology-schema-v2.md`

**Existing Node Properties** (30+ indexes):
- Core: `iri`, `term_id`, `preferred_term`, `label`
- Classification: `source_domain`, `version`, `class_type`
- Quality: `status`, `maturity`, `quality_score`, `authority_score`
- OWL2: `owl_physicality`, `owl_role`
- Source: `file_sha1`, `source_file`, `markdown_content`

### Required Additions for Analytics

**Node Labels**:
```cypher
// Add analytics properties to existing nodes
MATCH (n)
SET n.cluster_id = $cluster_id,
    n.anomaly_score = $anomaly_score,
    n.community_id = $community_id,
    n.last_clustered = datetime(),
    n.last_anomaly_check = datetime();

// Create indexes for analytics queries
CREATE INDEX node_cluster_id IF NOT EXISTS FOR (n:Node) ON (n.cluster_id);
CREATE INDEX node_anomaly_score IF NOT EXISTS FOR (n:Node) ON (n.anomaly_score);
CREATE INDEX node_community_id IF NOT EXISTS FOR (n:Node) ON (n.community_id);
CREATE INDEX node_high_anomaly IF NOT EXISTS FOR (n:Node) ON (n.anomaly_score) WHERE n.anomaly_score > 0.5;
```

**Analytics Metadata Nodes**:
```cypher
// Store clustering metadata
CREATE (:ClusteringRun {
    run_id: $run_id,
    algorithm: $algorithm,
    num_clusters: $num_clusters,
    silhouette_score: $silhouette_score,
    timestamp: datetime(),
    parameters: $parameters
});

// Store community detection metadata
CREATE (:CommunityDetectionRun {
    run_id: $run_id,
    algorithm: 'louvain',
    num_communities: $num_communities,
    modularity: $modularity,
    timestamp: datetime()
});

// Link nodes to clustering runs
MATCH (n:Node), (c:ClusteringRun {run_id: $run_id})
WHERE n.cluster_id IS NOT NULL
CREATE (n)-[:CLUSTERED_IN {cluster_id: n.cluster_id}]->(c);
```

**Anomaly Tracking**:
```cypher
// Create anomaly event nodes
CREATE (:AnomalyEvent {
    event_id: $event_id,
    node_id: $node_id,
    score: $score,
    severity: $severity,
    detection_method: 'LOF',
    timestamp: datetime(),
    resolved: false
});

// Link to affected nodes
MATCH (n:Node {id: $node_id}), (a:AnomalyEvent {event_id: $event_id})
CREATE (n)-[:HAS_ANOMALY]->(a);
```

**Queries for Analytics Visualization**:
```cypher
// Get all nodes in a cluster with positions
MATCH (n:Node)
WHERE n.cluster_id = $cluster_id
RETURN n.id, n.x, n.y, n.z, n.cluster_id, n.anomaly_score;

// Get high-severity anomalies
MATCH (n:Node)
WHERE n.anomaly_score > 0.8
RETURN n.id, n.anomaly_score, n.preferred_term
ORDER BY n.anomaly_score DESC;

// Get community structure
MATCH (n:Node)
WHERE n.community_id IS NOT NULL
WITH n.community_id AS community, collect(n) AS nodes, count(n) AS size
RETURN community, size, nodes[0..10] AS sample_nodes
ORDER BY size DESC;

// Get inter-community edges
MATCH (n1:Node)-[r]-(n2:Node)
WHERE n1.community_id <> n2.community_id
RETURN n1.community_id, n2.community_id, count(r) AS edge_count
ORDER BY edge_count DESC;
```

---

## Integration Summary

### Server-Side (Rust)

1. **ClusteringActor** already provides:
   - GPU K-means clustering
   - Louvain community detection
   - Output: `Cluster` structs with colors, centroids, nodes

2. **AnomalyDetectionActor** already provides:
   - GPU LOF anomaly detection
   - Output: `AnomalyResult` with scores per node

3. **Protocol V3** already implements:
   - `encode_node_data_with_analytics()` function
   - 48-byte binary format
   - Version byte = 3

4. **Missing Integration**:
   - WebSocket handler needs to call `encode_node_data_with_analytics()` instead of `encode_node_data_with_types()`
   - Build analytics HashMap from ClusteringActor + AnomalyDetectionActor results
   - Add Protocol V3 negotiation (query param: `protocol=v3`)

### Client-Side (TypeScript)

1. **BinaryProtocolV3Parser** (NEW):
   - Decode 48-byte messages
   - Extract `cluster_id`, `anomaly_score`, `community_id`

2. **Rendering Pipelines** (NEW):
   - Cluster color-coding (material.color)
   - Convex hull boundaries (ConvexGeometry)
   - Anomaly red glow (selective bloom + emissive)
   - Community boundaries (wireframe hulls)
   - Community labels (CSS2DRenderer)

3. **Analytics UI** (already exists):
   - `/client/src/api/analyticsApi.ts` provides full REST API
   - WebSocket subscriptions for real-time updates

### Neo4j Integration

1. **Add Properties** (3 new fields):
   - `cluster_id: u32`
   - `anomaly_score: f32`
   - `community_id: u32`

2. **Add Indexes** (3 new indexes):
   - `node_cluster_id`
   - `node_anomaly_score`
   - `node_community_id`

3. **Analytics Queries**:
   - Cluster membership
   - Anomaly detection
   - Community structure
   - Inter-community edges

---

## Performance Impact

### Bandwidth

**Protocol V2** (36 bytes):
- 100K nodes = 3.6 MB

**Protocol V3** (48 bytes):
- 100K nodes = 4.8 MB
- **+33% bandwidth increase**

**Mitigation**:
- Enable WebSocket compression (`permessage-deflate`)
- Compression typically achieves 2-3x reduction
- Net result: ~2.4 MB compressed (vs 1.8 MB for V2 compressed)

### Latency

**Additional Processing**:
- Clustering: 50-200ms (GPU K-means for 100K nodes)
- Anomaly Detection: 30-150ms (GPU LOF)
- Community Detection: 100-500ms (Louvain)

**Total**: +180-850ms one-time cost when analytics enabled

**Real-time Updates**:
- No additional latency (analytics pre-computed)
- Binary encoding: +0.2ms for extra 12 bytes/node

### Client Rendering

**Additional GPU Load**:
- Convex hull computation: 5-20ms per cluster (CPU-bound)
- Bloom post-processing: 2-5ms per frame (GPU-bound)
- Material updates: <1ms (cached)

**Optimization**:
- Compute hulls once, cache geometry
- Update bloom pass only when anomalies change
- Use instanced rendering for cluster boundaries

---

## Recommendations

### Phase 1: Enable Protocol V3 (Immediate)

1. Update WebSocket handler to use `encode_node_data_with_analytics()`
2. Add Protocol V3 client decoder
3. Test with existing clustering/anomaly endpoints

### Phase 2: Client Rendering (Week 1)

1. Implement cluster color-coding
2. Add anomaly red glow effect
3. Create cluster boundary hulls (convex geometry)

### Phase 3: Neo4j Integration (Week 2)

1. Add analytics properties to schema
2. Create indexes for clustering/anomaly queries
3. Implement analytics persistence layer

### Phase 4: Community Visualization (Week 3)

1. Implement Louvain community boundaries
2. Add community labels
3. Create community analytics dashboard

### Phase 5: Optimization (Week 4)

1. Enable WebSocket compression
2. Implement hull caching
3. Add selective bloom rendering
4. Performance profiling and tuning

---

## Files Modified

### Rust (Server)
- ✅ `/src/utils/binary_protocol.rs` - **ALREADY DONE** (Protocol V3 implemented)
- ✅ `/src/actors/gpu/clustering_actor.rs` - **ALREADY DONE** (K-means + Louvain)
- ✅ `/src/actors/gpu/anomaly_detection_actor.rs` - **ALREADY DONE** (LOF)
- ⚠️ `/src/handlers/realtime_websocket_handler.rs` - **NEEDS UPDATE** (use V3 encoding)
- ⚠️ `/src/adapters/neo4j_graph_repository.rs` - **NEEDS UPDATE** (add analytics properties)

### TypeScript (Client)
- 🆕 `/client/src/services/BinaryProtocolV3Parser.ts` - **NEW FILE**
- 🆕 `/client/src/features/analytics/rendering/ClusterRenderer.ts` - **NEW FILE**
- 🆕 `/client/src/features/analytics/rendering/AnomalyRenderer.ts` - **NEW FILE**
- 🆕 `/client/src/features/analytics/rendering/CommunityRenderer.ts` - **NEW FILE**
- ⚠️ `/client/src/services/WebSocketService.ts` - **NEEDS UPDATE** (add V3 support)
- ⚠️ `/client/src/features/graph/managers/graphDataManager.ts` - **NEEDS UPDATE** (apply analytics)

### Database
- 🆕 `/migrations/004_analytics_properties.sql` - **NEW FILE**
- 🆕 `/docs/neo4j-analytics-schema.md` - **NEW FILE**

---

## Conclusion

**Protocol V3 is already fully implemented on the server side!** The remaining work is:

1. **Integration** (2-3 days):
   - Connect WebSocket handler to V3 encoding
   - Build analytics HashMap from GPU actors
   - Add protocol version negotiation

2. **Client Rendering** (1 week):
   - Decode V3 binary format
   - Implement cluster/anomaly/community visualization
   - Add Three.js post-processing effects

3. **Neo4j Schema** (2-3 days):
   - Add 3 new properties + indexes
   - Create analytics metadata nodes
   - Implement persistence layer

4. **Testing & Optimization** (1 week):
   - Performance profiling
   - WebSocket compression
   - Hull caching
   - End-to-end validation

**Total Estimated Timeline**: 3-4 weeks to production-ready analytics visualization.

---

**Agent**: Analytics Visualization Agent
**Status**: Analysis Complete ✅
**Next Steps**: Prioritize Phase 1 (Protocol V3 integration) as foundation for all other work.
