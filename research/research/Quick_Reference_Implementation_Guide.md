# Quick Reference: Constraint-Based 3D Ontology Visualization

**For Rapid Implementation Lookup** | Companion to Academic Research Survey

---

## 🎯 Decision Tree (Start Here)

```
What do you need RIGHT NOW?

├─ "Show me the fastest path to working demo"
│  └─> Section 1: MVP Implementation (2-4 weeks)
│
├─ "What constraint should I implement first?"
│  └─> Section 2: Constraint Priority Matrix
│
├─ "How do I structure my Rust code?"
│  └─> Section 3: Architecture Patterns
│
├─ "My layout looks like a hairball, help!"
│  └─> Section 4: Troubleshooting Hairballs
│
└─ "What's the best library for X?"
   └─> Section 5: Technology Quick Picks
```

---

## Section 1: MVP Implementation Paths

### Path A: Fastest Demo (2-4 weeks)

**Stack**: 3d-force-graph (JavaScript) + Rust backend (wasm)

```
Week 1: Basic Integration
├─ Install: npm install 3d-force-graph
├─ Load ontology: hornedowl → extract classes/properties
├─ Convert to JSON: { nodes: [...], links: [...] }
└─ Render: const graph = ForceGraph3D()(elem).graphData(data);

Week 2: Semantic Constraints (JavaScript layer)
├─ Z-axis hierarchy: graph.nodeThreeObject(node => positionByDepth(node))
├─ Type clustering: Custom force: d3.forceRadial(radius, x, y)
└─ Edge weighting: link.strength = relationshipWeight(link.type)

Week 3: OWL Reasoning (Rust wasm)
├─ whelk reasoning → inferred axioms
├─ Extract part-of relationships
├─ Generate containment constraints
└─ Export to JSON → JavaScript consumes

Week 4: Polish
├─ Node colors by ontology module
├─ Edge bundling (3d-force-graph supports via link curvature)
├─ Expandable nodes (toggle children visibility)
└─ Camera controls + UI panel
```

**Code Snippet** (JavaScript integration):
```javascript
import ForceGraph3D from '3d-force-graph';

// Load Rust-processed ontology
const ontologyData = await fetch('/api/ontology/graph').then(r => r.json());

const graph = ForceGraph3D()
  .graphData(ontologyData)
  .nodeLabel('name')
  .nodeColor(node => colorByType(node.type))
  .linkDirectionalArrowLength(3.5)
  .d3Force('charge', d3.forceManyBody().strength(-120))
  .d3Force('z-hierarchy', () => {
    ontologyData.nodes.forEach(node => {
      node.fz = node.depth * 30; // Pin Z by hierarchy
    });
  });
```

### Path B: High Performance (3-4 months)

**Stack**: Custom Rust + wgpu + three-d

```
Month 1: Foundation
├─ Setup wgpu rendering pipeline
├─ Implement Barnes-Hut octree (hardest part!)
│  Reference: GraphPU source code
├─ Basic force-directed simulation
└─ Camera controls (arcball rotation)

Month 2: Constraint System
├─ ComposableForceManager trait system
├─ HierarchicalZAxisForce
├─ TypeClusteringForce
├─ NonOverlapCollision (spatial hashing)
└─ Reasoner integration (offline step)

Month 3: Advanced Features
├─ GRIP multiscale initialization
├─ Edge bundling (3D force-directed bundling)
├─ Expandable nodes with animation
└─ LOD rendering (frustum culling, label hiding)

Month 4: Optimization & Polish
├─ GPU compute shaders for forces
├─ Instanced rendering for nodes/edges
├─ Persistent Homology barcode UI
└─ User testing + refinement
```

**Critical Rust Dependencies**:
```toml
[dependencies]
wgpu = "0.19"
three-d = "0.16"
petgraph = "0.6"
hornedowl = "0.1"
rayon = "1.8" # Parallel CPU fallback
serde = { version = "1.0", features = ["derive"] }
```

---

## Section 2: Constraint Priority Matrix

| Priority | Constraint Type | Impact on Hairball | Impl. Difficulty | Code Example |
|----------|----------------|-------------------|------------------|--------------|
| **1 (DO FIRST)** | Z-Axis Hierarchy | ⭐⭐⭐⭐⭐ 80% improvement | Easy (1 day) | `node.fz = depth * 30` |
| **2** | Pinned Constellation | ⭐⭐⭐⭐ 65% improvement | Easy (2 days) | `if (isUpperLevel) node.fx/fy/fz = PRESET` |
| **3** | Type Clustering | ⭐⭐⭐⭐ 60% improvement | Medium (3 days) | `d3.forceRadial(r, cx, cy).strength(2.0)` |
| **4** | Non-Overlap | ⭐⭐⭐ 40% improvement | Medium (4 days) | Spatial hashing collision |
| **5** | Reasoner-Inferred | ⭐⭐⭐⭐⭐ 75% for complex ontologies | High (1 week) | whelk → containment constraints |
| **6** | Multiscale GRIP | ⭐⭐⭐⭐ 70% for large graphs | High (1-2 weeks) | Progressive coarsening |
| **7** | Edge Bundling | ⭐⭐⭐ 50% visual clarity | High (1-2 weeks) | Force-directed bundling |
| **8** | Persistent Homology | ⭐⭐⭐⭐ 80% for user control | Very High (2-3 weeks) | GUDHI integration + barcode UI |

**Rule of Thumb**: Implement top 5 for production system. Add 6-8 only if you have >5,000 nodes.

---

## Section 3: Architecture Patterns

### Pattern 1: Composable Force Manager

**Problem**: Need to combine multiple constraint forces flexibly

**Solution**:
```rust
pub trait ConstraintForce: Send + Sync {
    fn compute_forces(&self, positions: &[Vec3], graph: &OntologyGraph) -> Vec<Vec3>;
    fn name(&self) -> &str;
    fn enabled(&self) -> bool { true }
}

pub struct ForceManager {
    forces: Vec<Box<dyn ConstraintForce>>,
}

impl ForceManager {
    pub fn add(&mut self, force: Box<dyn ConstraintForce>) {
        self.forces.push(force);
    }

    pub fn compute_total(&self, positions: &[Vec3], graph: &OntologyGraph) -> Vec<Vec3> {
        let mut total = vec![Vec3::ZERO; positions.len()];

        for force in &self.forces {
            if !force.enabled() { continue; }

            let partial = force.compute_forces(positions, graph);
            for (i, f) in partial.iter().enumerate() {
                total[i] += f;
            }
        }

        total
    }
}
```

**Usage**:
```rust
let mut manager = ForceManager::new();
manager.add(Box::new(BarnesHutRepulsion { theta: 0.5 }));
manager.add(Box::new(HierarchicalZAxisForce { strength: 5.0, z_scale: 3.0 }));
manager.add(Box::new(TypeClusteringForce::new(&graph, 2.0)));

// Simulation loop
loop {
    let forces = manager.compute_total(&positions, &graph);
    update_positions(&mut positions, &forces, dt);
}
```

### Pattern 2: GPU Force Computation

**Problem**: CPU force calculation too slow for >1,000 nodes

**Solution** (WGSL compute shader):
```wgsl
// forces.wgsl
struct Node {
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
}

@group(0) @binding(0) var<storage, read> nodes: array<Node>;
@group(0) @binding(1) var<storage, read_write> forces: array<vec3<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&nodes) { return; }

    var total_force = vec3<f32>(0.0);
    let pos = nodes[i].position;

    // Repulsion (Barnes-Hut would go here)
    for (var j = 0u; j < arrayLength(&nodes); j++) {
        if i == j { continue; }
        let delta = pos - nodes[j].position;
        let dist = length(delta);
        if dist > 0.0 {
            total_force += normalize(delta) * (params.repulsion / (dist * dist));
        }
    }

    // Custom constraint forces
    total_force += compute_hierarchy_force(pos, i);
    total_force += compute_clustering_force(pos, i);

    forces[i] = total_force;
}
```

**Rust dispatch**:
```rust
// Dispatch GPU compute
compute_pass.set_pipeline(&force_compute_pipeline);
compute_pass.set_bind_group(0, &bind_group, &[]);
compute_pass.dispatch_workgroups((node_count + 255) / 256, 1, 1);
```

### Pattern 3: Expandable Node State Machine

**Problem**: Complex state transitions when nodes expand/collapse

**Solution**:
```rust
pub enum NodeState {
    Collapsed,
    Expanding { start_time: Instant, duration: Duration },
    Expanded,
    Collapsing { start_time: Instant, duration: Duration },
}

impl NodeState {
    pub fn update(&mut self, now: Instant) -> StateChange {
        match self {
            Self::Expanding { start_time, duration } => {
                let elapsed = now.duration_since(*start_time);
                if elapsed >= *duration {
                    *self = Self::Expanded;
                    StateChange::CompletedExpansion
                } else {
                    StateChange::None
                }
            }
            Self::Collapsing { start_time, duration } => {
                let elapsed = now.duration_since(*start_time);
                if elapsed >= *duration {
                    *self = Self::Collapsed;
                    StateChange::CompletedCollapse
                } else {
                    StateChange::None
                }
            }
            _ => StateChange::None
        }
    }

    pub fn animation_progress(&self, now: Instant) -> f32 {
        match self {
            Self::Expanding { start_time, duration } |
            Self::Collapsing { start_time, duration } => {
                let elapsed = now.duration_since(*start_time).as_secs_f32();
                let total = duration.as_secs_f32();
                (elapsed / total).min(1.0)
            }
            _ => 1.0
        }
    }
}

pub fn ease_in_out_cubic(t: f32) -> f32 {
    if t < 0.5 {
        4.0 * t * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(3) / 2.0
    }
}
```

---

## Section 4: Troubleshooting Hairballs

### Symptom 1: "Everything clumps in the center"

**Diagnosis**: Repulsion force too weak OR graph disconnected

**Fix**:
```rust
// Increase repulsion
repulsion_strength *= 2.0;

// OR add global gravity to origin (prevents drift)
let gravity = -position.normalize() * 0.1;
```

### Symptom 2: "Nodes fly apart to infinity"

**Diagnosis**: Repulsion too strong OR no attractive forces

**Fix**:
```rust
// Add damping
velocity *= 0.95; // Each frame

// OR reduce repulsion
repulsion_strength *= 0.5;
```

### Symptom 3: "Layout looks random, no structure"

**Diagnosis**: Missing semantic constraints

**Fix** (Priority order):
1. ✅ Add Z-axis hierarchy force
2. ✅ Add type clustering
3. ✅ Pin upper-level ontology classes

### Symptom 4: "Nodes overlap heavily"

**Diagnosis**: No collision detection

**Fix** (Spatial hashing):
```rust
fn resolve_collisions(positions: &mut [Vec3], radii: &[f32]) {
    const CELL_SIZE: f32 = 5.0;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

    // Bin nodes into grid cells
    for (i, pos) in positions.iter().enumerate() {
        let cell = (
            (pos.x / CELL_SIZE).floor() as i32,
            (pos.y / CELL_SIZE).floor() as i32,
            (pos.z / CELL_SIZE).floor() as i32,
        );
        grid.entry(cell).or_default().push(i);
    }

    // Check collisions only within same cell + 26 neighbors
    for (cell, indices) in grid.iter() {
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    let neighbor_cell = (cell.0 + di, cell.1 + dj, cell.2 + dk);
                    if let Some(neighbor_indices) = grid.get(&neighbor_cell) {
                        for &i in indices {
                            for &j in neighbor_indices {
                                if i >= j { continue; }
                                resolve_pair(positions, radii, i, j);
                            }
                        }
                    }
                }
            }
        }
    }
}

fn resolve_pair(positions: &mut [Vec3], radii: &[f32], i: usize, j: usize) {
    let delta = positions[j] - positions[i];
    let dist = delta.length();
    let min_dist = radii[i] + radii[j];

    if dist < min_dist && dist > 0.0 {
        let overlap = min_dist - dist;
        let correction = delta.normalize() * (overlap / 2.0);
        positions[i] -= correction;
        positions[j] += correction;
    }
}
```

### Symptom 5: "Can't see hierarchy despite Z-axis force"

**Diagnosis**: Z-scale too small OR other forces too strong

**Fix**:
```rust
// Increase Z-scale (more vertical separation)
hierarchical_force.z_scale = 10.0; // Was 3.0

// OR reduce XY forces
xy_forces_strength *= 0.5;
```

---

## Section 5: Technology Quick Picks

### Rendering Library (Choose One)

| Library | Best For | Learning Curve |
|---------|----------|----------------|
| **3d-force-graph** (JS) | Fastest MVP, 3D + VR support | Low (1 day) |
| **three-d** (Rust) | Full control, custom shaders | Medium (1 week) |
| **GraphPU** (Rust) | Massive scale (50K+ nodes) | High (fork & modify) |

**Recommendation**: Start with 3d-force-graph, migrate to three-d if need custom constraints.

### Graph Data Structure

**Winner**: `petgraph` (Rust)

```rust
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Direction;

let mut graph = Graph::<NodeData, EdgeData>::new();

let a = graph.add_node(NodeData { name: "ClassA", depth: 0 });
let b = graph.add_node(NodeData { name: "ClassB", depth: 1 });
graph.add_edge(a, b, EdgeData { relationship: "subClassOf" });

// Traverse
for neighbor in graph.neighbors_directed(a, Direction::Outgoing) {
    println!("Child: {:?}", graph[neighbor]);
}
```

### OWL Reasoning

**Winner**: `whelk` (Rust, OWL 2 EL profile, fast)

```bash
# Command-line usage
whelk --classify ontology.owl > inferred.owl

# Or as library (coming soon)
```

**Alternative**: Call Java reasoners via CLI (Elk, HermiT)
```rust
use std::process::Command;

let output = Command::new("java")
    .args(&["-jar", "elk.jar", "ontology.owl"])
    .output()?;

let inferred = String::from_utf8(output.stdout)?;
```

### GPU Compute

**Winner**: `wgpu` (cross-platform WebGPU)

```rust
let device = /* wgpu device creation */;
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("Force Compute Shader"),
    source: wgpu::ShaderSource::Wgsl(include_str!("forces.wgsl").into()),
});

let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("Force Pipeline"),
    layout: Some(&pipeline_layout),
    module: &shader,
    entry_point: "compute_forces",
});
```

### Persistent Homology

**Winner**: GUDHI (C++) via FFI

```rust
// Rust FFI wrapper (simplified)
#[link(name = "gudhi")]
extern "C" {
    fn compute_persistence(
        edges: *const Edge,
        num_edges: usize,
        out_bars: *mut PersistenceBar
    ) -> usize;
}

pub fn compute_ph(graph: &Graph) -> Vec<PersistenceBar> {
    let edges = graph_to_edge_list(graph);
    let mut bars = vec![PersistenceBar::default(); edges.len()];

    unsafe {
        let count = compute_persistence(
            edges.as_ptr(),
            edges.len(),
            bars.as_mut_ptr()
        );
        bars.truncate(count);
    }

    bars
}
```

---

## Section 6: Parameter Cheat Sheet

### Good Starting Values

```rust
// Force strengths (adjust based on graph size)
const REPULSION: f32 = 100.0;        // Higher = more spread out
const EDGE_ATTRACTION: f32 = 0.01;   // Higher = tighter clustering
const HIERARCHY_Z: f32 = 5.0;        // Higher = more vertical separation
const TYPE_CLUSTERING: f32 = 2.0;    // Higher = tighter type groups

// Spatial
const Z_SCALE: f32 = 3.0;            // Units per hierarchy level
const NODE_RADIUS: f32 = 1.0;        // For collision detection

// Simulation
const DAMPING: f32 = 0.95;           // Velocity decay (0.9-0.99)
const TIME_STEP: f32 = 0.016;        // ~60fps
const MAX_ITERATIONS: usize = 500;   // Until convergence

// Barnes-Hut
const THETA: f32 = 0.5;              // Accuracy vs speed (0.3-0.7)
```

### Relationship-Specific Spring Constants

```rust
fn spring_constant(relationship: &str) -> f32 {
    match relationship {
        "rdfs:subClassOf" => 10.0,   // Strong, short
        "partOf" => 5.0,              // Moderate
        "associatedWith" => 1.0,      // Weak, long
        _ => 2.0                      // Default
    }
}
```

---

## Section 7: Performance Targets

### Frame Rate (Interactive Exploration)

| Node Count | CPU Only | GPU Accelerated | Acceptable? |
|------------|----------|-----------------|-------------|
| 100 | 60fps | 60fps | ✅ Excellent |
| 1,000 | 30fps | 60fps | ✅ Good |
| 5,000 | 5fps | 45-60fps | ✅ Acceptable |
| 10,000 | 1fps | 30fps | ⚠️ Marginal |
| 20,000 | 0.1fps | 10-20fps | ❌ Layout only |

**Rule**: Freeze layout after convergence for >5,000 nodes (static navigation OK).

### Layout Quality

| Metric | Excellent | Good | Poor |
|--------|-----------|------|------|
| Stress | < 0.1 | 0.1-0.2 | > 0.2 |
| Distance Correlation | > 0.90 | 0.80-0.90 | < 0.80 |
| Edge Crossings (vs baseline) | -70% | -50% | -20% |

### Memory Usage (Estimate)

```
Per Node: ~200 bytes (position, velocity, metadata)
Per Edge: ~50 bytes (source, target, weight)

1,000 nodes, 2,000 edges: ~300 KB
10,000 nodes, 25,000 edges: ~3.3 MB
100,000 nodes, 300,000 edges: ~35 MB
```

GPU: Add 2-4× overhead for buffers, octrees, shader data.

---

## Section 8: Copy-Paste Recipes

### Recipe 1: Minimal Working 3d-force-graph

```html
<!DOCTYPE html>
<html>
<head>
    <style>body { margin: 0; }</style>
    <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
    <div id="graph"></div>
    <script>
        const data = {
            nodes: [
                { id: 'A', depth: 0, type: 'root' },
                { id: 'B', depth: 1, type: 'branch' },
                { id: 'C', depth: 1, type: 'branch' },
                { id: 'D', depth: 2, type: 'leaf' }
            ],
            links: [
                { source: 'A', target: 'B', type: 'subClassOf' },
                { source: 'A', target: 'C', type: 'subClassOf' },
                { source: 'B', target: 'D', type: 'subClassOf' }
            ]
        };

        const graph = ForceGraph3D()
            .graphData(data)
            .nodeLabel('id')
            .nodeColor(node => ({
                root: '#ff0000',
                branch: '#00ff00',
                leaf: '#0000ff'
            }[node.type]))
            .d3Force('charge', d3.forceManyBody().strength(-120))
            .d3VelocityDecay(0.4);

        // Z-axis hierarchy constraint
        graph.d3Force('z-hierarchy', () => {
            data.nodes.forEach(node => {
                node.fz = node.depth * 30; // Pin Z coordinate
            });
        });

        graph(document.getElementById('graph'));
    </script>
</body>
</html>
```

### Recipe 2: Rust Semantic Spring Constants

```rust
use petgraph::graph::Graph;
use std::collections::HashMap;

pub struct SemanticForceDirected {
    graph: Graph<NodeData, EdgeData>,
    positions: Vec<Vec3>,
    velocities: Vec<Vec3>,
    spring_constants: HashMap<String, f32>,
}

impl SemanticForceDirected {
    pub fn new(graph: Graph<NodeData, EdgeData>) -> Self {
        let mut spring_constants = HashMap::new();
        spring_constants.insert("rdfs:subClassOf".to_string(), 10.0);
        spring_constants.insert("partOf".to_string(), 5.0);
        spring_constants.insert("associatedWith".to_string(), 1.0);

        let node_count = graph.node_count();
        Self {
            graph,
            positions: vec![Vec3::ZERO; node_count],
            velocities: vec![Vec3::ZERO; node_count],
            spring_constants,
        }
    }

    pub fn step(&mut self, dt: f32) {
        let mut forces = vec![Vec3::ZERO; self.positions.len()];

        // Edge attraction (semantic weighted)
        for edge in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge).unwrap();
            let edge_data = &self.graph[edge];

            let k = self.spring_constants
                        .get(&edge_data.relationship)
                        .copied()
                        .unwrap_or(2.0);

            let delta = self.positions[target.index()] - self.positions[source.index()];
            let distance = delta.length();
            let force = delta.normalize() * k * (distance - 1.0);

            forces[source.index()] += force;
            forces[target.index()] -= force;
        }

        // Repulsion (all pairs - use Barnes-Hut for large graphs!)
        for i in 0..self.positions.len() {
            for j in (i+1)..self.positions.len() {
                let delta = self.positions[j] - self.positions[i];
                let dist = delta.length().max(0.1);
                let force = delta.normalize() * (100.0 / (dist * dist));

                forces[i] -= force;
                forces[j] += force;
            }
        }

        // Integrate
        for i in 0..self.positions.len() {
            self.velocities[i] += forces[i] * dt;
            self.velocities[i] *= 0.95; // Damping
            self.positions[i] += self.velocities[i] * dt;
        }
    }
}
```

### Recipe 3: WASM Bridge (Rust → JavaScript)

```rust
// lib.rs
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<NodeData>,
    pub links: Vec<EdgeData>,
}

#[wasm_bindgen]
pub fn process_ontology(owl_content: &str) -> JsValue {
    // Parse OWL using hornedowl
    let ontology = hornedowl::io::rdf::reader::read_string(owl_content).unwrap();

    // Extract graph structure
    let mut nodes = Vec::new();
    let mut links = Vec::new();

    for class in ontology.classes() {
        nodes.push(NodeData {
            id: class.to_string(),
            depth: compute_depth(class, &ontology),
            class_type: infer_type(class, &ontology),
        });
    }

    for axiom in ontology.axioms() {
        if let Some((source, target, rel)) = extract_relationship(axiom) {
            links.push(EdgeData { source, target, relationship: rel });
        }
    }

    let graph_data = GraphData { nodes, links };
    serde_wasm_bindgen::to_value(&graph_data).unwrap()
}
```

**JavaScript usage**:
```javascript
import init, { process_ontology } from './pkg/ontology_processor.js';

async function loadOntology() {
    await init(); // Initialize WASM

    const owlText = await fetch('ontology.owl').then(r => r.text());
    const graphData = process_ontology(owlText);

    // Render with 3d-force-graph
    ForceGraph3D().graphData(graphData)(document.getElementById('graph'));
}
```

---

## Section 9: Testing Checklist

### Functional Tests

- [ ] Graph loads without errors (1,000-node test file)
- [ ] All nodes visible (check node count matches ontology)
- [ ] Edges render correctly (spot-check 10 random relationships)
- [ ] Hierarchy visible on Z-axis (manually inspect 5 parent-child pairs)
- [ ] Type clustering evident (visual inspection of class groups)
- [ ] Collision detection works (no overlapping nodes at rest)
- [ ] Expandable nodes animate smoothly (no stuttering)
- [ ] Camera controls responsive (orbit, pan, zoom)

### Performance Tests

- [ ] 1,000 nodes: 60fps sustained
- [ ] 5,000 nodes: 30fps minimum
- [ ] Layout converges in <2 minutes for 5,000 nodes
- [ ] Memory usage stable (no leaks after 10 expand/collapse cycles)
- [ ] GPU utilization >60% during layout computation

### Visual Quality Tests

- [ ] Stress < 0.15 (use Kamada-Kawai formula)
- [ ] Distance correlation > 0.85 (Pearson of graph vs Euclidean distance)
- [ ] Edge crossings reduced by >50% vs baseline force-directed
- [ ] No "flying nodes" (all nodes within reasonable bounds)
- [ ] Semantic meaning clear (domain experts can identify clusters)

### User Acceptance Tests (5 domain experts)

- [ ] Can find parent class in <10 seconds (avg)
- [ ] Can trace relationship path in <15 seconds (avg)
- [ ] Can identify cluster theme in <20 seconds (avg)
- [ ] Cognitive load <50/100 (NASA TLX)
- [ ] Preference ranking: New system > baseline (majority)

---

## Section 10: Common Gotchas

### Gotcha 1: Euler Integration Instability

**Problem**: Simulation explodes (nodes fly to infinity)

**Cause**: Time step too large OR forces too strong

**Fix**:
```rust
const MAX_FORCE: f32 = 50.0;
fn clamp_forces(forces: &mut [Vec3]) {
    for force in forces {
        if force.length() > MAX_FORCE {
            *force = force.normalize() * MAX_FORCE;
        }
    }
}
```

### Gotcha 2: Integer Overflow in Octree

**Problem**: Barnes-Hut octree construction crashes on large graphs

**Cause**: Using `u32` for child pointers, exceeds 4 billion nodes

**Fix**: Use `usize` for all node indices/pointers

### Gotcha 3: Floating-Point Atomics Missing

**Problem**: GPU shader needs atomic add for f32, but only i32/u32 atomic ops available

**Fix**: Simulate with bit-casting
```wgsl
fn atomic_add_f32(addr: ptr<storage, atomic<u32>>, value: f32) {
    var old_bits = atomicLoad(addr);
    var new_bits: u32;

    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_value = old_value + value;
        new_bits = bitcast<u32>(new_value);

        let exchanged = atomicCompareExchangeWeak(addr, old_bits, new_bits);
        if exchanged.exchanged { break; }
        old_bits = exchanged.old_value;
    }
}
```

### Gotcha 4: Reasoner Timeout on Large Ontologies

**Problem**: whelk hangs on 10K+ class ontology

**Cause**: OWL 2 DL reasoning is undecidable, exponential worst-case

**Fix**: Use OWL 2 EL or RL profile (polynomial time)
```bash
whelk --profile EL ontology.owl
```

### Gotcha 5: Z-Fighting on Stratified Layers

**Problem**: Nodes at same depth flicker (Z-buffer conflict)

**Cause**: Exact Z-coordinate overlap

**Fix**: Add small jitter
```rust
node.z = depth as f32 * Z_SCALE + random::<f32>() * 0.1;
```

---

**End of Quick Reference**

For detailed research background, see companion document: `Academic_Research_Survey.md`

**Last Updated**: October 2025
