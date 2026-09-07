---
id: VW-03
title: Force layout — Barnes-Hut, SIMD and CSR simulation lifecycle
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
adrs: []
sources:
  - ../vowl-wasm/src/layout/mod.rs
  - ../vowl-wasm/src/layout/simulation.rs
  - ../vowl-wasm/src/layout/quadtree.rs
  - ../vowl-wasm/src/layout/force.rs
  - ../vowl-wasm/src/layout/simd.rs
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/examples/barnes_hut_benchmark.rs
verified_commit: 65e2d1e78
---

## VW-03.1 `ForceSimulation` one tick — `LayoutAlgorithm` impl
```mermaid
sequenceDiagram
    autonumber
    participant C as caller (WebVowl::tick / run_simulation)
    participant FS as ForceSimulation::tick<br/>src/layout/simulation.rs:268 impl LayoutAlgorithm
    participant CF as calculate_forces<br/>src/layout/simulation.rs:131
    participant AF as apply_forces<br/>src/layout/simulation.rs:239
    C->>FS: tick(&mut graph)
    FS->>FS: guard clause: if is_finished() return early<br/>src/layout/simulation.rs:269-271
    FS->>CF: calculate_forces(graph)
    CF-->>FS: HashMap<node_id, Vector2<f64>>
    FS->>AF: apply_forces(graph, &forces)
    AF->>AF: skip nodes where visual.fixed<br/>src/layout/simulation.rs:242
    AF->>AF: update velocity by force times alpha, apply damping, integrate position<br/>src/layout/simulation.rs:244-251
    FS->>FS: decay alpha, increment iteration counter<br/>src/layout/simulation.rs:287-288
```
- `run(graph, iterations)` (src/layout/simulation.rs:318) calls `initialize` at src/layout/simulation.rs:319 once then loops `tick` up to `iterations` times, breaking early on `is_finished()` (src/layout/simulation.rs:322-323).
- `is_finished()` is `alpha < config.alpha_min` (src/layout/simulation.rs:332) — a pure read, not a latched flag (contrast with `CsrSimulation`, VW-03.5).

## VW-03.2 `calculate_forces` — repulsion / attraction / centring
```mermaid
flowchart TB
    START["calculate_forces(graph)<br/>src/layout/simulation.rs:131"] --> BR{"n > 50 AND<br/>config.use_barnes_hut?<br/>src/layout/simulation.rs:150"}
    BR -->|yes| QT["QuadTree::build(&nodes)<br/>then calculate_force() per node<br/>src/layout/simulation.rs:155-164"]
    BR -->|no| BATCH["calculate_batch_repulsion(positions, charge)<br/>SIMD/scalar batch, O(n²)<br/>src/layout/simulation.rs:172"]
    QT --> ATTR["attraction along edges:<br/>calculate_attraction(pos1,pos2,link_distance,link_strength) * alpha<br/>src/layout/simulation.rs:181-190"]
    BATCH --> ATTR
    ATTR --> CENTER["calculate_batch_center_force(positions, center, center_strength)<br/>src/layout/simulation.rs:225-228"]
    CENTER --> SUM["forces: HashMap<node_id, Vector2<f64>><br/>summed per node"]
```
- Barnes-Hut only engages above the `n > 50` threshold (src/layout/simulation.rs:150) — small graphs always take the O(n²) batch path.

## VW-03.3 `QuadTree` — build, insert/subdivide, Barnes-Hut recursion
```mermaid
flowchart TB
    B["QuadTree::build(nodes)<br/>src/layout/quadtree.rs:126"] --> INS["insert(node_ref) per node<br/>src/layout/quadtree.rs:166"]
    INS --> COM["update center_of_mass, total_mass<br/>src/layout/quadtree.rs:168-173"]
    COM --> HASCH{"children already exist?"}
    HASCH -->|yes| DELEGATE["delegate to child quadrant<br/>get_quadrant(x,y)<br/>src/layout/quadtree.rs:178-181"]
    HASCH -->|no| PUSH["push into self.nodes<br/>src/layout/quadtree.rs:184"]
    PUSH --> CAP{"nodes.len() > capacity?"}
    CAP -->|yes| SUB["subdivide(): 4 children,<br/>redistribute existing nodes<br/>src/layout/quadtree.rs:193-238"]
    CAP -->|no| DONE1["leaf, unchanged"]
    B --> CF["calculate_force(node_pos, node_id, theta, strength)<br/>src/layout/quadtree.rs:266"]
    CF --> LEAF{"self.children.is_none()?<br/>src/layout/quadtree.rs:279"}
    LEAF -->|yes, 1 node = self| ZERO["return zero (self-interaction)<br/>src/layout/quadtree.rs:280-282"]
    LEAF -->|"yes, other leaves"| EXACT["calculate_exact_force<br/>src/layout/quadtree.rs:285, detail in VW-03.4"]
    LEAF -->|no| THETA{"s/distance < theta?<br/>src/layout/quadtree.rs:300"}
    THETA -->|yes| APPROX["approximate via center_of_mass,<br/>strength*total_mass/distance²<br/>src/layout/quadtree.rs:301-303"]
    THETA -->|no| RECURSE["recurse into all 4 children,<br/>sum their forces<br/>src/layout/quadtree.rs:307-311"]
```
- `theta` (typical 0.5–0.9, doc comment src/layout/quadtree.rs:261) trades layout accuracy for speed: smaller theta forces more recursion (exact), larger theta approximates more aggressively.

## VW-03.4 Leaf-force SIMD/scalar dispatch
```mermaid
flowchart LR
    LEAF["calculate_exact_force(node_pos, node_id, strength)<br/>src/layout/quadtree.rs:323"] --> FILT["sources = leaf nodes minus self<br/>src/layout/quadtree.rs:330-334"]
    FILT --> GATE{"feature: simd AND<br/>sources.len() >= 4?<br/>src/layout/quadtree.rs:341,343"}
    GATE -->|yes, wasm32| SIMDW["unsafe calculate_leaf_force_simd()<br/>src/layout/simd.rs:386, target_feature simd128"]
    GATE -->|yes, non-wasm32| SIMDN["calculate_leaf_force_simd() scalar shim<br/>src/layout/simd.rs:479"]
    GATE -->|no| SCALAR["scalar fold over sources:<br/>strength*mass/distance_sq, deterministic<br/>perturbation for coincident points<br/>src/layout/quadtree.rs:377-389"]
    SIMDW --> OUT["Vector2<f64> force"]
    SIMDN --> OUT
    SCALAR --> OUT
```
- INVARIANT: coincident nodes (`distance_sq < 0.0001`) never divide by zero — both the scalar leaf path (src/layout/quadtree.rs:384-387) and the CSR tick (VW-03.5) substitute a deterministic sine/cosine-derived nudge.
- `is_simd_available()` differs by target: `#[cfg(target_arch = "wasm32")]` (src/layout/simd.rs:516) probes the runtime feature; the non-wasm32 stub (src/layout/simd.rs:525) always returns `false`.

## VW-03.5 `CsrSimulation::tick` — deterministic O(n²) worker physics
```mermaid
flowchart TB
    T["tick()<br/>src/layout/csr_sim.rs:204"] --> FIN{"finished or n==0?"}
    FIN -->|yes| RET["return, no-op<br/>src/layout/csr_sim.rs:205-207"]
    FIN -->|no| REP["repulsion: O(n²) Coulomb over<br/>every unordered pair once<br/>src/layout/csr_sim.rs:213-238"]
    REP --> COIN{"d² < 1e-6 (coincident)?"}
    COIN -->|yes| NUDGE["index-derived sin/cos perturbation<br/>src/layout/csr_sim.rs:219-226"]
    COIN -->|no| REP
    NUDGE --> SPRING["springs: Hooke toward link_distance<br/>over edge_a/edge_b/edge_type<br/>src/layout/csr_sim.rs:241-259"]
    SPRING --> INTEG["centre gravity + symplectic-Euler<br/>integration with damping, per node<br/>src/layout/csr_sim.rs:262-282"]
    INTEG --> DECAY["alpha *= 1 - alpha_decay<br/>src/layout/csr_sim.rs:284"]
    DECAY --> FLOOR{"alpha < alpha_min?"}
    FLOOR -->|yes| LATCH["alpha = alpha_min; finished = true<br/>src/layout/csr_sim.rs:285-287"]
    FLOOR -->|no| DONE["tick complete, positions updated in place"]
```
- DIVERGENCE: unlike `ForceSimulation` (no Barnes-Hut option), `CsrSimulation` is always O(n²) — the crate-level doc (src/layout/csr_sim.rs:9-11) scopes this to ≤1,500-node tiers (T1) and calls Barnes-Hut for this path "a follow-up slot", not yet implemented.
- `finished` is a latched `bool` (src/layout/csr_sim.rs:82,297) set once the alpha floor is hit — `tick()` after that point is a guaranteed no-op, not merely a small-alpha update.

## VW-03.6 NGG1 tier load — bytes to seeded simulation
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS worker: loadCsr(bytes)
    participant NX as NggExplorer::load_csr<br/>src/bindings/explorer.rs:40
    participant CS as CsrSimulation::load_csr<br/>src/layout/csr_sim.rs:182
    participant N1 as ngg1::Ngg1 parser<br/>src/ngg1.rs
    JS->>NX: loadCsr(bytes: &[u8])
    NX->>CS: self.sim.load_csr(bytes)
    CS->>N1: decode header (32B) + node records (24B stride)<br/>NGG1_HEADER_SIZE=32, NGG1_NODE_STRIDE=24<br/>src/ngg1.rs:21-24
    N1-->>CS: Ngg1<'_> view (borrowed, no realloc)
    CS->>CS: load(&g): seed positions, build CSR springs<br/>src/layout/csr_sim.rs:191
    CS-->>NX: Result<(), Ngg1Error>
    NX-->>JS: Result<(), JsValue> (mapped via to_string)<br/>src/bindings/explorer.rs:42-44
```
- The reader is allocation-free for node/CSR field reads: no `unsafe`, decoded with `from_le_bytes` because the input slice from JS is not guaranteed 4-byte aligned (src/ngg1.rs:9-13).
- The node record is fixed at **24 bytes**, not 20 — the crate doc corrects a self-inconsistent brief (src/ngg1.rs:15-16, `N_DEGREE` offset 20 at src/ngg1.rs:31).
- Canonical spec `publishing-tools/WasmVOWL/FORMAT-NGG1.md` lives in the consumer repo (visionGraph, EXTERNAL — see VG-* format docs); this reader must stay byte-compatible with its TypeScript counterpart `modern/src/lib/ngg1.ts` (src/ngg1.rs:3-4).

## VW-03.7 Alpha annealing — shared shape, two independent implementations
```mermaid
stateDiagram-v2
    [*] --> Running: alpha = config.alpha (1.0)
    Running --> Running: tick() decays alpha *= (1 - alpha_decay)
    Running --> Finished: alpha < alpha_min
    Finished --> Running: CsrSimulation only — reheat(alpha)<br/>src/layout/csr_sim.rs:368 / set_param("reheat")
    note right of Finished
      ForceSimulation.is_finished() recomputes
      from alpha every call (no reheat API).
      CsrSimulation.finished is a latched bool,
      cleared only by reheat().
    end note
```
- DOC-DRIFT: only `CsrSimulation`/`NggExplorer` expose a `reheat` (src/layout/csr_sim.rs:368, src/bindings/explorer.rs:106) — `WebVowl`/`ForceSimulation` have no equivalent JS-reachable re-anneal call; a settled `WebVowl` graph can only be restarted via a fresh `initSimulation()`.

## VW-03.8 `LayoutAlgorithm` trait vs the two concrete engines
```mermaid
classDiagram
    class LayoutAlgorithm {
        <<trait>>
        +initialize(graph) Result
        +tick(graph) Result
        +run(graph, iterations) Result
        +is_finished() bool
        +alpha() f64
        src/layout/mod.rs:17
    }
    class ForceSimulation {
        -LayoutConfig config
        -f64 alpha
        -usize iteration
        -DebugFlags debug_flags
        src/layout/simulation.rs:20
    }
    class CsrSimulation {
        -SimConfig config
        -Vec~f32~ positions
        -Vec~f32~ velocities
        -bool finished
        src/layout/csr_sim.rs:69
        note "not a LayoutAlgorithm impl — own tick()/is_finished()/alpha() inherent methods, f32 not f64"
    }
    LayoutAlgorithm <|.. ForceSimulation
    ForceSimulation --> "used by" WebVowl
    CsrSimulation --> "used by" NggExplorer
```
- `CsrSimulation` deliberately does not implement `LayoutAlgorithm`: it operates on flat `Vec<f32>` CSR buffers rather than a `VowlGraph`, and exposes `f32` alpha (src/layout/csr_sim.rs:303) where the trait's `alpha()` is `f64` (src/layout/mod.rs:30) — the two engines share a design (annealing, Hooke springs, Coulomb repulsion) but not a common Rust interface.
