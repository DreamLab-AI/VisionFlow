# Implementation Roadmap: Constraint-Based 3D Ontology Visualization

**Companion to Academic Research Survey & Quick Reference Guide**

---

## Executive Decision Matrix

**Choose your implementation path based on constraints:**

| Your Situation | Recommended Path | Timeline | Expected Result |
|----------------|-----------------|----------|-----------------|
| **Need proof-of-concept in 2 weeks** | Path A: JavaScript MVP | 2-4 weeks | 1K-5K nodes, semantic constraints, web-based |
| **Need production system for 5K+ nodes** | Path B: Hybrid Rust/JS | 2-3 months | 10K nodes @30fps, GPU-ready, extensible |
| **Need maximum performance (10K-50K nodes)** | Path C: Full Rust/GPU | 3-4 months | 50K nodes @10-20fps, cutting-edge constraints |
| **Research project (publishing goal)** | Path D: GNN + PH Advanced | 6-8 months | State-of-art, novel contributions, paper-worthy |

---

## Path A: JavaScript MVP (2-4 Weeks)

**Stack**: 3d-force-graph + Rust backend (WebAssembly)

### Week 1: Foundation
**Goal**: Basic 3D graph rendering with OWL loading

**Monday-Tuesday: Setup & Data Pipeline**
- [ ] Initialize project: `npm create vite@latest ontology-viz -- --template vanilla`
- [ ] Install: `npm install 3d-force-graph three d3-force-3d`
- [ ] Create Rust workspace: `cargo new --lib ontology-processor`
- [ ] Add wasm-bindgen: `cargo add wasm-bindgen serde wasm-bindgen`
- [ ] Setup build: `wasm-pack build --target web`

**Wednesday-Thursday: OWL Parsing**
```rust
// ontology-processor/src/lib.rs
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct Node {
    pub id: String,
    pub label: String,
    pub depth: usize,
    pub node_type: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Link {
    pub source: String,
    pub target: String,
    pub relation: String,
}

#[derive(Serialize, Deserialize)]
pub struct GraphData {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
}

#[wasm_bindgen]
pub fn parse_ontology(owl_rdf: &str) -> Result<JsValue, JsValue> {
    // Use hornedowl or rio parsers
    // For MVP, use simple RDF/XML parsing

    let mut nodes = Vec::new();
    let mut links = Vec::new();

    // Parse classes
    // Extract rdfs:subClassOf relationships
    // Compute depths via BFS

    let graph = GraphData { nodes, links };
    Ok(serde_wasm_bindgen::to_value(&graph)
        .map_err(|e| JsValue::from_str(&e.to_string()))?)
}
```

**Friday: Integration Test**
```javascript
// main.js
import init, { parse_ontology } from './pkg/ontology_processor.js';
import ForceGraph3D from '3d-force-graph';

async function main() {
    await init();

    const owlText = await fetch('test-ontology.owl').then(r => r.text());
    const graphData = parse_ontology(owlText);

    const graph = ForceGraph3D()
        .graphData(graphData)
        .nodeLabel('label')
        .nodeColor(() => '#4a90e2')
        .linkColor(() => '#999999');

    graph(document.getElementById('graph'));
}

main();
```

**Deliverable**: Can load & render simple OWL ontology (e.g., Pizza.owl)

---

### Week 2: Semantic Constraints (JavaScript Layer)
**Goal**: Hierarchical Z-axis + type-based clustering

**Monday: Z-Axis Hierarchy**
```javascript
const graph = ForceGraph3D()
    .graphData(graphData)
    .nodeLabel('label')
    .nodeColor(node => {
        const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6'];
        return colors[node.depth % colors.length];
    })
    .d3Force('charge', d3.forceManyBody().strength(-200))
    .d3VelocityDecay(0.3);

// Z-axis stratification
graph.d3Force('z-hierarchy', () => {
    graphData.nodes.forEach(node => {
        node.fz = node.depth * 40; // Pin Z coordinate by depth
    });
});
```

**Tuesday-Wednesday: Type-Based Clustering**
```javascript
// Group nodes by type
const typeGroups = {};
graphData.nodes.forEach(node => {
    if (!typeGroups[node.node_type]) {
        typeGroups[node.node_type] = [];
    }
    typeGroups[node.node_type].push(node);
});

// Create radial forces for each type
Object.entries(typeGroups).forEach(([type, nodes], index) => {
    const angle = (index / Object.keys(typeGroups).length) * 2 * Math.PI;
    const radius = 100;
    const cx = Math.cos(angle) * radius;
    const cy = Math.sin(angle) * radius;

    graph.d3Force(`cluster-${type}`, d3.forceRadial(
        node => nodes.includes(node) ? 30 : 0,
        cx,
        cy
    ).strength(0.5));
});
```

**Thursday: Semantic Edge Weighting**
```javascript
// Relationship-specific spring constants
const relationshipStrengths = {
    'rdfs:subClassOf': 2.0,
    'partOf': 1.5,
    'associatedWith': 0.5,
};

graph.d3Force('link', d3.forceLink()
    .id(d => d.id)
    .distance(link => {
        const strength = relationshipStrengths[link.relation] || 1.0;
        return 50 / strength; // Inverse: stronger = shorter
    })
    .strength(link => {
        return relationshipStrengths[link.relation] || 1.0;
    })
);
```

**Friday: UI Controls**
```html
<!-- index.html -->
<div id="controls">
    <label>
        Z-Scale: <input type="range" id="z-scale" min="10" max="100" value="40">
        <span id="z-scale-value">40</span>
    </label>
    <label>
        Cluster Strength: <input type="range" id="cluster-strength" min="0" max="2" step="0.1" value="0.5">
        <span id="cluster-strength-value">0.5</span>
    </label>
    <button id="reset-camera">Reset Camera</button>
</div>
```

```javascript
// Controls logic
document.getElementById('z-scale').addEventListener('input', (e) => {
    const zScale = parseFloat(e.target.value);
    document.getElementById('z-scale-value').textContent = zScale;

    graphData.nodes.forEach(node => {
        node.fz = node.depth * zScale;
    });
    graph.graphData(graphData); // Trigger re-render
});
```

**Deliverable**: Interactive 3D graph with visible hierarchy and type clustering

---

### Week 3: Expandable Nodes & Polish
**Goal**: Click to expand/collapse, visual refinements

**Monday-Tuesday: Expandable Node Logic**
```javascript
// Track expanded state
const expandedNodes = new Set();

graph
    .nodeThreeObject(node => {
        const geometry = new THREE.SphereGeometry(
            expandedNodes.has(node.id) ? 8 : 5
        );
        const material = new THREE.MeshLambertMaterial({
            color: node.color,
            transparent: true,
            opacity: 0.9
        });
        return new THREE.Mesh(geometry, material);
    })
    .onNodeClick(node => {
        if (expandedNodes.has(node.id)) {
            // Collapse
            expandedNodes.delete(node.id);
            hideChildren(node);
        } else {
            // Expand
            expandedNodes.add(node.id);
            showChildren(node);
        }
        graph.graphData(graphData); // Refresh
    });

function showChildren(node) {
    graphData.links
        .filter(link => link.source === node.id)
        .forEach(link => {
            const child = graphData.nodes.find(n => n.id === link.target);
            if (child) child.visible = true;
        });
}

function hideChildren(node) {
    graphData.links
        .filter(link => link.source === node.id)
        .forEach(link => {
            const child = graphData.nodes.find(n => n.id === link.target);
            if (child) {
                child.visible = false;
                hideChildren(child); // Recursive
            }
        });
}
```

**Wednesday: Edge Bundling (Simplified)**
```javascript
graph
    .linkCurvature(link => {
        // Curve edges between different clusters
        const sourceType = graphData.nodes.find(n => n.id === link.source)?.node_type;
        const targetType = graphData.nodes.find(n => n.id === link.target)?.node_type;
        return sourceType !== targetType ? 0.3 : 0;
    })
    .linkDirectionalArrowLength(3.5)
    .linkDirectionalArrowRelPos(1);
```

**Thursday: Performance Optimization**
```javascript
// Freeze layout after convergence
let tickCount = 0;
const MAX_TICKS = 300;

graph.onEngineStop(() => {
    console.log('Layout converged');
    tickCount = MAX_TICKS;
});

graph.d3Force('center', d3.forceCenter());

// Manual tick limit
const interval = setInterval(() => {
    tickCount++;
    if (tickCount >= MAX_TICKS) {
        graph.pauseAnimation();
        clearInterval(interval);
    }
}, 16); // ~60fps
```

**Friday: Visual Polish**
- Node labels on hover
- Edge thickness by importance
- Bloom post-processing effect (via Three.js UnrealBloomPass)
- Loading spinner during layout computation

**Deliverable**: Polished, interactive demo ready for stakeholder review

---

### Week 4: Documentation & Deployment
**Goal**: Deployment + usage docs

**Monday-Tuesday: Deployment**
```bash
# Build for production
npm run build

# Deploy to Netlify/Vercel
netlify deploy --prod --dir=dist

# Or Docker container
docker build -t ontology-viz .
docker run -p 8080:80 ontology-viz
```

**Wednesday-Thursday: Documentation**
Create README with:
- Installation instructions
- Sample ontology files
- Keyboard shortcuts
- Performance tips (graph size limits)
- Architecture diagram

**Friday: User Testing**
- 5 domain experts try the system
- Collect feedback on usability
- Measure task completion times
- Identify pain points for v2

**Deliverable**: Deployed MVP + documentation

---

## Path B: Hybrid Rust/JS (2-3 Months)

**Stack**: Rust constraint solver + 3d-force-graph renderer

### Month 1: Rust Constraint Engine

**Week 1-2: Core Force Simulation**
```rust
// constraint-engine/src/simulation.rs
use nalgebra::Vector3;
use rayon::prelude::*;

pub struct ForceSimulation {
    positions: Vec<Vector3<f32>>,
    velocities: Vec<Vector3<f32>>,
    forces: Vec<Box<dyn Force>>,
}

pub trait Force: Send + Sync {
    fn compute(&self, positions: &[Vector3<f32>]) -> Vec<Vector3<f32>>;
}

impl ForceSimulation {
    pub fn step(&mut self, dt: f32) {
        // Parallel force computation
        let total_forces: Vec<Vector3<f32>> = self.forces
            .par_iter()
            .map(|force| force.compute(&self.positions))
            .reduce(
                || vec![Vector3::zeros(); self.positions.len()],
                |mut acc, forces| {
                    for (i, f) in forces.iter().enumerate() {
                        acc[i] += f;
                    }
                    acc
                }
            );

        // Euler integration
        for i in 0..self.positions.len() {
            self.velocities[i] += total_forces[i] * dt;
            self.velocities[i] *= 0.95; // Damping
            self.positions[i] += self.velocities[i] * dt;
        }
    }
}
```

**Week 3: Semantic Constraints**
```rust
// constraint-engine/src/forces/hierarchical.rs
pub struct HierarchicalZForce {
    depths: Vec<usize>,
    z_scale: f32,
    strength: f32,
}

impl Force for HierarchicalZForce {
    fn compute(&self, positions: &[Vector3<f32>]) -> Vec<Vector3<f32>> {
        positions.iter().enumerate().map(|(i, pos)| {
            let target_z = self.depths[i] as f32 * self.z_scale;
            let error = target_z - pos.z;
            Vector3::new(0.0, 0.0, error * self.strength)
        }).collect()
    }
}
```

**Week 4: WASM Bindings**
```rust
// constraint-engine/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct SimulationHandle {
    sim: ForceSimulation,
}

#[wasm_bindgen]
impl SimulationHandle {
    #[wasm_bindgen(constructor)]
    pub fn new(node_count: usize) -> Self {
        let mut sim = ForceSimulation::new(node_count);
        sim.add_force(Box::new(BarnesHutRepulsion::new()));
        Self { sim }
    }

    pub fn add_hierarchy_constraint(&mut self, depths: Vec<usize>, z_scale: f32) {
        self.sim.add_force(Box::new(HierarchicalZForce {
            depths,
            z_scale,
            strength: 5.0,
        }));
    }

    pub fn step(&mut self, dt: f32) {
        self.sim.step(dt);
    }

    pub fn get_positions(&self) -> Vec<f32> {
        // Flatten Vec<Vector3> to Vec<f32> for JS
        self.sim.positions
            .iter()
            .flat_map(|v| vec![v.x, v.y, v.z])
            .collect()
    }
}
```

---

### Month 2: Advanced Constraints + Reasoner Integration

**Week 5: OWL Reasoner Integration**
```rust
// Use whelk or call ELK via process
use std::process::Command;

pub fn compute_inferred_containments(owl_path: &str) -> Vec<(String, String)> {
    // Run ELK reasoner
    let output = Command::new("java")
        .args(&["-jar", "elk.jar", "--classify", owl_path])
        .output()
        .expect("Failed to run reasoner");

    // Parse output for part-of inferences
    let inferred = String::from_utf8(output.stdout).unwrap();
    parse_part_of_axioms(&inferred)
}

fn parse_part_of_axioms(rdf: &str) -> Vec<(String, String)> {
    // Parse RDF/XML or Turtle for:
    // ?part rdfs:subClassOf [owl:onProperty partOf; owl:someValuesFrom ?whole]
    vec![] // Placeholder
}
```

**Week 6-7: Multiscale GRIP Initialization**
```rust
pub fn grip_coarsen(graph: &Graph, levels: usize) -> Vec<Vec<NodeId>> {
    let mut hierarchies = vec![graph.node_indices().collect()];

    for level in 0..levels {
        let current = &hierarchies[level];
        let next = maximal_independent_set(graph, current, 2_usize.pow(level as u32));
        hierarchies.push(next);
    }

    hierarchies
}

pub fn grip_layout(graph: &Graph) -> Vec<Vector3<f32>> {
    let hierarchies = grip_coarsen(graph, 5);

    // Layout coarsest level
    let mut positions = force_directed(&hierarchies.last().unwrap());

    // Progressive refinement
    for level in (0..hierarchies.len()-1).rev() {
        let added_nodes = &hierarchies[level];
        initialize_from_neighbors(&mut positions, added_nodes);
        local_refinement(&mut positions, added_nodes, 50);
    }

    positions
}
```

**Week 8: Testing & Benchmarking**
- Unit tests for each force
- Integration tests: 1K, 5K, 10K node graphs
- Benchmark: positions/sec throughput
- Memory profiling

---

### Month 3: GPU Acceleration + Polish

**Week 9-10: GPU Barnes-Hut**
```rust
// Use wgpu for compute shaders
use wgpu;

pub struct GPUBarnesHut {
    device: wgpu::Device,
    queue: wgpu::Queue,
    octree_pipeline: wgpu::ComputePipeline,
}

impl GPUBarnesHut {
    pub fn compute_forces(&self, positions: &[Vector3<f32>]) -> Vec<Vector3<f32>> {
        // 1. Build octree on GPU
        // 2. Traverse octree in compute shader
        // 3. Read back forces

        // See GraphPU implementation for full details
        vec![] // Placeholder
    }
}
```

**Week 11: Performance Optimization**
- Profile with `cargo flamegraph`
- Optimize hot paths (likely octree construction)
- Parallel Barnes-Hut via rayon
- SIMD vectorization for force accumulation

**Week 12: Integration & Deployment**
- Wire Rust WASM module to 3d-force-graph
- Dockerfile for deployment
- CI/CD pipeline (GitHub Actions)
- Load testing (stress test with 10K-50K nodes)

**Deliverable**: Production system handling 10K nodes @30fps

---

## Path C: Full Rust/GPU (3-4 Months)

**Stack**: Custom renderer (three-d or bevy) + wgpu compute

### Month 1: Rendering Foundation

**Week 1-2: three-d Setup**
```rust
// main.rs
use three_d::*;

fn main() {
    let window = Window::new(WindowSettings {
        title: "Ontology Visualizer".to_string(),
        max_size: Some((1920, 1080)),
        ..Default::default()
    }).unwrap();

    let context = window.gl();

    let mut camera = Camera::new_perspective(
        window.viewport(),
        vec3(0.0, 0.0, 200.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        degrees(45.0),
        0.1,
        1000.0,
    );

    let mut control = OrbitControl::new(*camera.target(), 1.0, 100.0);

    // Load ontology & initialize simulation
    let graph = load_ontology("ontology.owl");
    let mut simulation = ForceSimulation::new(graph.node_count());

    window.render_loop(move |frame_input| {
        control.handle_events(&mut camera, &mut frame_input.events);

        // Simulation step
        simulation.step(0.016);

        // Render nodes & edges
        render_graph(&context, &camera, &simulation, &graph);

        FrameOutput::default()
    });
}
```

**Week 3: Instanced Rendering**
```rust
fn render_nodes(context: &Context, camera: &Camera, positions: &[Vector3<f32>]) {
    let sphere = Gm::new(
        Mesh::new(context, &CpuMesh::sphere(16)),
        ColorMaterial {
            color: Color::new(0.3, 0.5, 0.8, 1.0),
            ..Default::default()
        },
    );

    // Instance buffer for positions
    let instances = Instances {
        translations: positions.iter()
            .map(|p| vec3(p.x, p.y, p.z))
            .collect(),
        ..Default::default()
    };

    sphere.render_instances(camera, &instances);
}
```

**Week 4: Edge Rendering (Line Instances)**
```rust
fn render_edges(context: &Context, camera: &Camera, graph: &Graph, positions: &[Vector3<f32>]) {
    for edge in graph.edges() {
        let source_pos = positions[edge.source()];
        let target_pos = positions[edge.target()];

        let line = Gm::new(
            Mesh::new(context, &CpuMesh::cylinder(1)),
            ColorMaterial::default(),
        );

        // Transform cylinder to connect source→target
        let midpoint = (source_pos + target_pos) / 2.0;
        let direction = target_pos - source_pos;
        let length = direction.magnitude();

        line.set_transformation(
            Mat4::from_translation(midpoint) *
            Mat4::from_quat(Quat::from_arc(vec3(0.0, 1.0, 0.0), direction.normalize(), None)) *
            Mat4::from_nonuniform_scale(0.2, length / 2.0, 0.2)
        );

        line.render(camera);
    }
}
```

---

### Month 2: GPU Compute Pipeline

**Week 5-6: Compute Shader Infrastructure**
```wgsl
// forces.wgsl
struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
}

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn compute_forces(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= arrayLength(&particles) { return; }

    var force = vec3<f32>(0.0);

    // Repulsion (Barnes-Hut would be separate pass)
    for (var j = 0u; j < arrayLength(&particles); j++) {
        if i == j { continue; }
        force += compute_repulsion(particles[i].position, particles[j].position);
    }

    // Update velocity & position
    particles[i].velocity += force * params.dt;
    particles[i].velocity *= params.damping;
    particles[i].position += particles[i].velocity * params.dt;
}
```

**Week 7-8: Barnes-Hut Octree on GPU**
- Extremely complex implementation
- Reference GraphPU source code
- Consider using existing CUDA libraries via bindings
- Fallback: CPU Barnes-Hut with rayon parallelism

---

### Month 3-4: Advanced Features + Optimization

**Week 9: Persistent Homology Integration**
```rust
// Use GUDHI via FFI
#[link(name = "gudhi")]
extern "C" {
    fn compute_persistence_0d(
        edges: *const [usize; 2],
        weights: *const f32,
        num_edges: usize,
        out_diagram: *mut PersistencePair,
    ) -> usize;
}

pub struct PersistencePair {
    pub birth: f32,
    pub death: f32,
    pub cluster_id: usize,
}

pub fn compute_ph(graph: &Graph) -> Vec<PersistencePair> {
    // Convert graph to weighted edge list
    // Call GUDHI
    // Parse results
    vec![]
}
```

**Week 10-11: Interactive Barcode UI**
```rust
// Use egui for UI overlay
use egui;

fn render_barcode_ui(ctx: &egui::Context, persistence: &[PersistencePair]) {
    egui::Window::new("Persistent Homology").show(ctx, |ui| {
        for (i, pair) in persistence.iter().enumerate() {
            let persistence_value = pair.death - pair.birth;

            ui.horizontal(|ui| {
                let checkbox = ui.checkbox(&mut selected[i], "");
                if checkbox.clicked() {
                    toggle_cluster(pair.cluster_id);
                }

                ui.label(format!("Cluster {}: {:.2}", pair.cluster_id, persistence_value));

                // Bar visualization
                let bar_length = persistence_value * 100.0;
                ui.add(egui::ProgressBar::new(bar_length).show_percentage());
            });
        }
    });
}
```

**Week 12-13: Optimization**
- Frame pacing (<16ms frame time)
- LOD for distant nodes
- Frustum culling
- Occlusion culling (if needed)

**Week 14-16: Testing & Deployment**
- Cross-platform testing (Windows, macOS, Linux)
- Performance benchmarking suite
- User acceptance testing
- Documentation

**Deliverable**: Standalone native application, 50K nodes @10-20fps

---

## Path D: Research/Advanced (6-8 Months)

**For Publishing/Novel Contributions**

### Month 1-2: Foundation (Same as Path C)

### Month 3-4: GNN Implementation

**Dataset Preparation**
- Collect 20-50 ontologies from BioPortal, OBO Foundry
- Generate ground-truth layouts (offline optimization)
- Split: 70% train, 15% validation, 15% test

**StructureNet Adaptation**
```python
# Using PyTorch Geometric
import torch
from torch_geometric.nn import GCNConv

class OntologyEncoder(torch.nn.Module):
    def __init__(self, node_features=64, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 3)  # Output: 3D position

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        position = self.fc(x)  # Predict (x, y, z)
        return position

# Training loop
model = OntologyEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()

        pred_positions = model(batch.x, batch.edge_index)
        ground_truth = batch.positions

        # Loss: combination of spatial coherence + graph distance preservation
        loss = (
            0.5 * mse_loss(pred_positions, ground_truth) +
            0.3 * graph_distance_loss(pred_positions, batch) +
            0.2 * collision_penalty(pred_positions)
        )

        loss.backward()
        optimizer.step()
```

**Export to Rust**
```python
# Export to ONNX
torch.onnx.export(model, dummy_input, "ontology_encoder.onnx")
```

```rust
// Load in Rust via tract
use tract_onnx::prelude::*;

let model = tract_onnx::onnx()
    .model_for_path("ontology_encoder.onnx")?
    .into_optimized()?
    .into_runnable()?;

pub fn predict_positions(graph: &Graph) -> Vec<Vector3<f32>> {
    let input = prepare_graph_tensor(graph);
    let result = model.run(tvec!(input.into()))?;
    parse_positions(result)
}
```

---

### Month 5-6: Novel Contributions

**Research Questions to Explore**:
1. Does GNN+PH hybrid outperform either alone?
2. Optimal constraint weight learning from user behavior?
3. Temporal ontology layouts (maintaining spatial memory across versions)?

**Experimental Design**:
- Controlled user study (n=30 participants)
- A/B testing: Baseline vs GNN vs GNN+PH
- Measure: task completion time, accuracy, cognitive load
- Collect telemetry: interaction patterns, zoom levels, camera paths

**Novel Algorithm**: Adaptive Constraint Weighting
```python
# Reinforcement learning for constraint weights
import gym
from stable_baselines3 import PPO

class LayoutEnv(gym.Env):
    def __init__(self, ontology):
        self.ontology = ontology
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(5,))  # 5 constraint weights
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,))

    def step(self, action):
        # action = [w_hierarchy, w_clustering, w_repulsion, w_attraction, w_semantic]
        layout = compute_layout(self.ontology, weights=action)

        # Reward: negative stress + user task success rate
        reward = -compute_stress(layout) + user_task_success(layout)

        obs = extract_layout_features(layout)
        done = True
        return obs, reward, done, {}

# Train
model = PPO("MlpPolicy", LayoutEnv(ontology), verbose=1)
model.learn(total_timesteps=10000)

# Use learned weights
optimal_weights = model.predict(obs)[0]
```

---

### Month 7-8: Publication

**Paper Structure**:
1. **Introduction**: The "ball of string" problem in ontology visualization
2. **Related Work**: Survey of constraint-based methods (cite 20-30 papers)
3. **Methodology**: GNN architecture, PH integration, adaptive weighting
4. **Experiments**: Benchmark datasets, user study protocol
5. **Results**: Quantitative metrics + qualitative insights
6. **Discussion**: Limitations, future work
7. **Conclusion**: Contributions summary

**Target Venues**:
- IEEE VIS (Deadline: March)
- EuroVis (Deadline: December)
- CHI (Deadline: September)
- ISWC (Semantic Web Conference, Deadline: April)

**Deliverable**: Conference paper submission + open-source implementation

---

## Common Milestones Across All Paths

### Milestone 1: "Hello World" (Week 1)
- [ ] Can load & render basic graph (100 nodes)
- [ ] Camera controls work
- [ ] No crashes on test data

### Milestone 2: "Semantic Constraints" (Week 2-4)
- [ ] Z-axis hierarchy visible
- [ ] Type-based clusters evident
- [ ] Visual quality improved vs baseline

### Milestone 3: "Scale" (Week 4-8)
- [ ] 1,000 nodes render smoothly
- [ ] 5,000 nodes acceptable performance
- [ ] Barnes-Hut or equivalent optimization working

### Milestone 4: "Production Ready" (Week 8-12)
- [ ] Deployment pipeline
- [ ] User documentation
- [ ] 5+ stakeholder demos successful

### Milestone 5: "Advanced Features" (Week 12-16, optional)
- [ ] Expandable nodes
- [ ] PH integration OR GNN priors
- [ ] Novel research contribution

---

## Risk Mitigation

### Risk 1: Barnes-Hut Implementation Too Complex

**Mitigation**:
- Use existing library (ForceAtlas2 port)
- Fallback to CPU parallel implementation (rayon)
- Accept lower node count target (5K instead of 50K)

### Risk 2: Reasoner Doesn't Scale

**Mitigation**:
- Use OWL 2 EL profile (polynomial time)
- Pre-compute inferences offline
- Cache results for known ontologies

### Risk 3: GNN Training Fails to Converge

**Mitigation**:
- Start with simpler architecture (GCN, not StructureNet)
- Use pre-trained embeddings (OWL2Vec)
- Fallback to handcrafted constraints

### Risk 4: User Study Recruitment

**Mitigation**:
- Partner with domain experts early
- Offer co-authorship on publications
- Use crowdsourcing platforms (Prolific) as backup

---

## Success Metrics

### Technical Metrics
- [ ] Frame rate: ≥30fps for 5K nodes
- [ ] Layout quality: Stress <0.15
- [ ] Distance correlation: >0.85
- [ ] Edge crossings: 50%+ reduction vs baseline

### User Metrics (n=10 domain experts)
- [ ] Task completion time: <baseline by 20%+
- [ ] Accuracy: >80% on relationship questions
- [ ] Cognitive load (NASA TLX): <50/100
- [ ] Preference: 70%+ prefer new system

### Project Metrics
- [ ] On-time delivery (±2 weeks acceptable)
- [ ] On-budget (if commercial project)
- [ ] Stakeholder satisfaction (≥4/5 rating)
- [ ] Adoption rate (if deploying to users)

---

**End of Roadmap**

For research details, see: `Academic_Research_Survey.md`
For implementation snippets, see: `Quick_Reference_Implementation_Guide.md`
