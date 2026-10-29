---
id: VW-03
title: Force layout — Barnes-Hut, SIMD and CSR simulation lifecycle
area: vowl-wasm
governing:
  - ../vowl-wasm/README.md
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
adrs: [vowl-wasm:ADR-001]
sources:
  - ../vowl-wasm/src/layout/mod.rs
  - ../vowl-wasm/Cargo.toml
  - ../vowl-wasm/src/layout/simulation.rs
  - ../vowl-wasm/src/layout/quadtree.rs
  - ../vowl-wasm/src/layout/simd.rs
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
verified_commit: 89d1a0a32eec16e79a7fa507eaa6f8fcb92e4f0b
---

## VW-03.1 `ForceSimulation` one tick — `LayoutAlgorithm` impl
```mermaid
sequenceDiagram
    autonumber
    participant C as caller (WebVowl::tick / run_simulation)
    participant FS as ForceSimulation::tick<br/>src/layout/simulation.rs:276 impl LayoutAlgorithm
    participant CF as calculate_forces<br/>src/layout/simulation.rs:139
    participant AF as apply_forces<br/>src/layout/simulation.rs:247
    C->>FS: tick(&mut graph)
    FS->>FS: guard clause: if is_finished() return early<br/>src/layout/simulation.rs:277-279
    FS->>CF: calculate_forces(graph)
    CF-->>FS: HashMap<node_id, Vector2<f64>>
    FS->>AF: apply_forces(graph, &forces)
    AF->>AF: skip nodes where visual.fixed<br/>src/layout/simulation.rs:250
    AF->>AF: update velocity by force times alpha, apply damping, integrate position<br/>src/layout/simulation.rs:252-259
    FS->>FS: decay alpha, increment iteration counter<br/>src/layout/simulation.rs:295-296
```
- `run(graph, iterations)` (src/layout/simulation.rs:326) calls `initialize` at src/layout/simulation.rs:327 once then loops `tick` up to `iterations` times, breaking early on `is_finished()` (src/layout/simulation.rs:330-331).
- `is_finished()` is `alpha < config.alpha_min` (src/layout/simulation.rs:340) — a pure read, not a latched flag (contrast with `CsrSimulation`, VW-03.5).

## VW-03.2 `calculate_forces` — repulsion / attraction / centring
```mermaid
flowchart TB
    START["calculate_forces(graph)<br/>src/layout/simulation.rs:139"] --> BR{"n > 50 AND<br/>config.use_barnes_hut?<br/>src/layout/simulation.rs:158"}
    BR -->|yes| QT["QuadTree::build(&nodes)<br/>then calculate_force() per node<br/>src/layout/simulation.rs:163-172"]
    BR -->|no| BATCH["calculate_batch_repulsion(positions, charge)<br/>SIMD/scalar batch, O(n²)<br/>src/layout/simulation.rs:180"]
    QT --> ATTR["attraction along edges:<br/>calculate_attraction(pos1,pos2,link_distance,link_strength) * alpha<br/>src/layout/simulation.rs:189-198"]
    BATCH --> ATTR
    ATTR --> CENTER["calculate_batch_center_force(positions, center, center_strength)<br/>src/layout/simulation.rs:233-236"]
    CENTER --> SUM["forces: HashMap<node_id, Vector2<f64>><br/>summed per node"]
```
- Barnes-Hut only engages above the `n > 50` threshold (src/layout/simulation.rs:158) — small graphs always take the O(n²) batch path.

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

## VW-03.5 `CsrSimulation::tick` — deterministic worker physics, Barnes-Hut by default
```mermaid
flowchart TB
    T["tick()<br/>src/layout/csr_sim.rs:366"] --> FIN{"finished or n==0?"}
    FIN -->|yes| RET["return, no-op<br/>src/layout/csr_sim.rs:367-369"]
    FIN -->|no| REP["repulsion_forces dispatches on config.repulsion,<br/>default BarnesHut theta 0.9<br/>src/layout/csr_sim.rs:375 and src/layout/csr_sim.rs:429-433 and src/layout/csr_sim.rs:91"]
    REP -->|"BarnesHut(theta)"| BH["one QuadTree::build per tick, one stack walk per node,<br/>a cell collapses to its centre of charge when width² &lt; theta² · d²<br/>and the node is outside it<br/>src/layout/csr_sim.rs:473-535 and src/layout/csr_sim.rs:518"]
    REP -->|"Exact, theta &lt;= 0"| EX["O(n²) Coulomb over every unordered pair once<br/>src/layout/csr_sim.rs:437-471"]
    BH --> COIN{"d² < 1e-6 (coincident)?"}
    EX --> COIN
    COIN -->|yes| NUDGE["index-derived sin/cos perturbation, same pair rule on both paths<br/>src/layout/csr_sim.rs:108 and src/layout/csr_sim.rs:447-452"]
    COIN -->|no| SPRING
    NUDGE --> SPRING["springs: Hooke toward link_distance<br/>over edge_a/edge_b/edge_type<br/>src/layout/csr_sim.rs:377-397"]
    SPRING --> INTEG["centre gravity + symplectic-Euler<br/>integration with damping, per node<br/>src/layout/csr_sim.rs:399-418"]
    INTEG --> DECAY["alpha *= 1 - alpha_decay<br/>src/layout/csr_sim.rs:420"]
    DECAY --> FLOOR{"alpha < alpha_min?"}
    FLOOR -->|yes| LATCH["alpha = alpha_min; finished = true<br/>src/layout/csr_sim.rs:421-424"]
    FLOOR -->|no| DONE["tick complete, positions updated in place"]
    N1["DEBT: ForceSimulation and CsrSimulation each carry their own Barnes-Hut quadtree, a node-reference tree in quadtree.rs and an arena tree over the flat position buffer in csr_sim.rs, so a fix to one does not reach the other<br/>src/layout/quadtree.rs:126 and src/layout/csr_sim.rs:152-161"]
    REP --- N1
```
- Since ER-20 `CsrSimulation` is no longer always O(n²): the module doc now states Barnes-Hut is the default and `Repulsion::Exact`, `set_theta(0.0)` or `set_param("theta", 0.0)` select the exact loop, which runs unchanged for comparison (src/layout/csr_sim.rs:9-17, src/layout/csr_sim.rs:605, src/layout/csr_sim.rs:633). The default `SimConfig` is still described as tuned for a scoped tier of at most 1,500 nodes (src/layout/csr_sim.rs:82).
- `finished` is a latched `bool` (src/layout/csr_sim.rs:245,538) set once the alpha floor is hit — `tick()` after that point is a guaranteed no-op, not merely a small-alpha update.

## VW-03.6 NGG1 tier load — bytes to seeded simulation
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS worker: loadCsr(bytes)
    participant NX as NggExplorer::load_csr<br/>src/bindings/explorer.rs:40
    participant CS as CsrSimulation::load_csr<br/>src/layout/csr_sim.rs:344
    participant N1 as ngg1::Ngg1 parser, re-exported from ontomark-ngg1<br/>src/ngg1.rs:5
    JS->>NX: loadCsr(bytes: &[u8])
    NX->>CS: self.sim.load_csr(bytes)
    CS->>N1: Ngg1::parse(bytes), the codec is the ontomark-ngg1 crate<br/>src/layout/csr_sim.rs:345 and src/ngg1.rs:5
    N1-->>CS: Ngg1<'_> view (borrowed, no realloc)
    CS->>CS: load(&g): seed positions, build CSR springs<br/>src/layout/csr_sim.rs:353
    CS-->>NX: Result<(), Ngg1Error>
    NX-->>JS: Result<(), JsValue> (mapped via to_string)<br/>src/bindings/explorer.rs:42-44
```
- The reader is no longer in this repository: `src/ngg1.rs` is one `pub use ontomark_ngg1::*;` so the `vowl_wasm::ngg1` path and every public name keep working (src/ngg1.rs:1-5), and the crate depends on `ontomark-ngg1` 0.1.0 from crates.io behind the `ngg1` feature (Cargo.toml:45-46, Cargo.toml:67). Header size, node stride, field offsets and the specification are therefore owned by ontomark, not by `vowl-wasm` (docs/adr/ADR-001-shared-ontology-rendering-contract.md:26).
- **Open:** the byte layout this diagram used to cite (32-byte header, 24-byte node record, `N_DEGREE` at offset 20) cannot be re-read from this repository at 89d1a0a; it is not confirmed here and the VW-03 stamp makes no claim about it.
- `CsrSimulation::load_csr` maps a parse failure to `Ngg1Error` and `NggExplorer::load_csr` maps that to a string for JS (src/layout/csr_sim.rs:344, src/bindings/explorer.rs:40-44).

## VW-03.7 Alpha annealing — shared shape, two independent implementations
```mermaid
stateDiagram-v2
    [*] --> Running: alpha = config.alpha (1.0)
    Running --> Running: tick() decays alpha *= (1 - alpha_decay)
    Running --> Finished: alpha < alpha_min
    Finished --> Running: reheat(alpha), both engines<br/>src/layout/csr_sim.rs:618 and src/layout/simulation.rs:71 / set_param("reheat") src/layout/csr_sim.rs:634
    note right of Finished
      ForceSimulation.is_finished() recomputes
      from alpha every call, so its reheat only raises alpha
      (src/layout/simulation.rs:71-75).
      CsrSimulation.finished is a latched bool,
      cleared only by reheat().
      CsrSimulation does not implement LayoutAlgorithm<br/>
      (trait alpha() f64, src/layout/mod.rs:30) — it exposes<br/>
      its own f32 alpha() over flat Vec~f32~ CSR buffers,<br/>
      not a VowlGraph (src/layout/csr_sim.rs:544).
    end note
```
- Both engines can now be re-annealed from JS: `ForceSimulation::reheat` clamps alpha to `(0, 1]` and ignores a non-finite or non-positive value (src/layout/simulation.rs:71-75), `CsrSimulation::reheat` raises alpha to at least `alpha_min` and clears `finished` (src/layout/csr_sim.rs:618-621); the `WebVowl` binding exposes `reheat` (src/bindings/mod.rs:133-136) and `NggExplorer` exposes `reheat` and `setTheta` (src/bindings/explorer.rs:111-118). This closes the earlier gap where a settled `WebVowl` graph could only restart through a fresh `initSimulation()`.
- `ForceSimulation` implements `LayoutAlgorithm` (src/layout/mod.rs:17); `CsrSimulation` deliberately does not — the two engines share a design (annealing, Hooke springs, Coulomb repulsion) but not a common Rust interface.

## VW-03.9 Seeding — `ForceSimulation::initialize` vs `CsrSimulation::load`
```mermaid
flowchart TB
    subgraph FS["ForceSimulation::initialize<br/>src/layout/simulation.rs:269"]
        FI1["initialize_positions(graph)<br/>src/layout/simulation.rs:89"] --> FI2{"preserve_seeded_positions AND<br/>node already placed?<br/>src/layout/simulation.rs:111-112"}
        FI2 -->|no| FI3["seed on circle, radius=seed_radius<br/>angle += TAU/node_count<br/>src/layout/simulation.rs:113-114"]
        FI2 -->|yes| FI4["keep caller's x,y"]
        FI3 --> FI5["vx,vy reset to 0 regardless<br/>src/layout/simulation.rs:118-119"]
        FI4 --> FI5
        FI5 --> FI6["alpha = config.alpha; iteration = 0<br/>src/layout/simulation.rs:270-271"]
    end
    subgraph CS["CsrSimulation::load_csr → load<br/>src/layout/csr_sim.rs:344,353"]
        CI1["Ngg1::parse(bytes)<br/>src/layout/csr_sim.rs:345"] --> CI2["positions = g.seed_positions()<br/>src/layout/csr_sim.rs:355"]
        CI2 --> CI3["velocities = zeroed Vec, len n*2<br/>src/layout/csr_sim.rs:356"]
        CI3 --> CI4["edge_a/edge_b/edge_type from g.edges()<br/>src/layout/csr_sim.rs:357-360"]
        CI4 --> CI5["alpha = config.alpha;<br/>finished = (n == 0)<br/>src/layout/csr_sim.rs:361-362"]
    end
```
- `ForceSimulation` seeds a fresh circular layout unless a caller opted into `preserve_seeded_positions`; `CsrSimulation` always takes its seed positions straight from the NGG1 tier's own `seed_positions()` — there is no circular-seed path on the CSR side.
- Both zero velocity/alpha state on entry, so a re-`initialize`/re-`load` is a clean restart, not a resume (contrast `reheat`, VW-03.7).

## VW-03.10 Engine and tuning-preset choice
```mermaid
flowchart TB
    N["graph to lay out"] --> SRC{"source is an NGG1 tier<br/>(worker-loaded, scoped tier of at most 1,500 nodes)?<br/>src/layout/csr_sim.rs:82"}
    SRC -->|yes| CSR["CsrSimulation<br/>flat Vec&lt;f32&gt; CSR buffers,<br/>Barnes-Hut repulsion by default, exact on theta &lt;= 0<br/>src/layout/csr_sim.rs:91"]
    SRC -->|no, VowlGraph| FSIM["ForceSimulation<br/>LayoutAlgorithm impl<br/>src/layout/simulation.rs:20"]
    FSIM --> SIZE{"graph size vs<br/>LayoutConfig preset?"}
    SIZE -->|"small/medium,<br/>default()"| DFLT["link_distance 30, charge -30,<br/>use_barnes_hut true (n&gt;50 engages it)<br/>src/layout/mod.rs:99-111"]
    SIZE -->|"large, dense<br/>ontology, large_graph()"| BIG["link_distance 250, charge -2000,<br/>center_strength 0.001, seed_radius 2000<br/>src/layout/mod.rs:138-144"]
    CSR --> TUNE["SimConfig: charge 2000, link_distance 60,<br/>damping 0.6, center_gravity 0.02<br/>src/layout/csr_sim.rs:84-95"]
```
- `large_graph()`'s doc comment (src/layout/mod.rs:118-121) records it was tuned empirically on a 1,155-node ontology and deliberately over-spreads a small graph, which is why it is a named preset rather than the default; Barnes-Hut engagement above `n > 50` is detailed in VW-03.2.
- `CsrSimulation` has no size-tiered preset — its `SimConfig::default()` is the only configuration, commented as tuned for the scoped tier of at most 1,500 nodes (src/layout/csr_sim.rs:82-95); its repulsion strategy is the one field a caller can switch at run time (src/layout/csr_sim.rs:70-71, src/layout/csr_sim.rs:605-611).
