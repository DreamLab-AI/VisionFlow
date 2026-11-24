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
  - ../vowl-wasm/src/layout/force.rs
  - ../vowl-wasm/tests/webvowl_layout_test.rs
  - ../vowl-wasm/tests/ngg_explorer_layout_test.rs
  - ../vowl-wasm/src/layout/quadtree.rs
  - ../vowl-wasm/src/layout/simd.rs
  - ../vowl-wasm/src/layout/csr_sim.rs
  - ../vowl-wasm/tests/ngg1_wasm.test.mjs
  - ../vowl-wasm/tests/ontomark_ngg1_migration.rs
  - ../vowl-wasm/src/ngg1.rs
  - ../vowl-wasm/src/bindings/mod.rs
  - ../vowl-wasm/src/bindings/explorer.rs
  - ../vowl-wasm/docs/adr/ADR-001-shared-ontology-rendering-contract.md
verified_commit: 0779174612a3942b15331356422f58bf8379912b
---

**Repair note — 2026-11-03:** selected counterexamples below have a [scoped successor verification](06-layout-and-release-repairs.md). Historical probe receipts retain their original revisions; source references were refreshed on 14 November 2026 after reviewing the plain-array loader, documentation and lint changes. The successor states the repaired contracts and acceptance boundaries.

## VW-03.1 `ForceSimulation` one tick — `LayoutAlgorithm` impl
```mermaid
sequenceDiagram
    autonumber
    participant C as caller (WebVowl::tick / run_simulation)
    participant FS as ForceSimulation::tick<br/>src/layout/simulation.rs:346 impl LayoutAlgorithm
    participant CF as calculate_forces<br/>src/layout/simulation.rs:161
    participant AF as apply_forces<br/>src/layout/simulation.rs:310
    participant RC as recenter<br/>src/layout/simulation.rs:226
    C->>FS: tick(&mut graph)
    FS->>FS: guard clause: if is_finished() return early<br/>src/layout/simulation.rs:347-349
    FS->>CF: calculate_forces(graph)
    CF-->>FS: Vec of Vector2, one per node in index order<br/>src/layout/simulation.rs:161
    FS->>AF: apply_forces(graph, &forces)
    AF->>AF: skip nodes where visual.fixed<br/>src/layout/simulation.rs:318-320
    AF->>AF: v = (v + alpha F) times velocity_decay, alpha applied only here<br/>src/layout/simulation.rs:321-322
    AF->>AF: drop a non-finite velocity, cap speed at MAX_STEP_FACTOR 2.0 times max(link_distance, seed_radius)<br/>src/layout/simulation.rs:311-316 and src/layout/simulation.rs:323-328
    FS->>RC: recenter(graph)
    RC->>RC: translate movable nodes so the centroid moves center_strength of the way to center<br/>src/layout/simulation.rs:236-245
    FS->>FS: decay alpha, increment iteration counter<br/>src/layout/simulation.rs:366-367
    Note over FS,RC: CHANGED at 0779174 (0.3.0, commit 3a8eb4b): alpha was applied twice to springs and the centre was a per-node pull,<br/>now alpha is applied once in apply_forces and centring is a d3 forceCenter translation, src/layout/simulation.rs:304-306,361-363
```
- `run(graph, iterations)` (src/layout/simulation.rs:397) calls `initialize` at src/layout/simulation.rs:398 once then loops `tick` up to `iterations` times, breaking early on `is_finished()` (src/layout/simulation.rs:401-402). Every call reseeds, including fixed nodes unless the native preservation option applies; repeated `runSimulation(n)` is not a continuation API (src/layout/simulation.rs:110-140).
- `is_finished()` is `alpha < config.alpha_min` (src/layout/simulation.rs:411) — a pure read, not a latched flag (contrast with `CsrSimulation`, VW-03.5). `reset()` returns alpha and the iteration count to their start values but keeps the configuration (src/layout/simulation.rs:87-96); `WebVowl::loadOntology` calls it (VW-03.7, VW-04.1).

## VW-03.2 `calculate_forces` — repulsion and degree-normalised springs
```mermaid
flowchart TB
    START["calculate_forces(graph)<br/>src/layout/simulation.rs:161"] --> BR{"n > 50 AND<br/>config.use_barnes_hut?<br/>src/layout/simulation.rs:175"}
    BR -->|yes| QT["QuadTree::build(&nodes)<br/>then calculate_force() per node<br/>src/layout/simulation.rs:180-191"]
    BR -->|no| BATCH["calculate_batch_repulsion(positions, charge)<br/>SIMD/scalar batch, O(n²)<br/>src/layout/simulation.rs:193"]
    QT --> LINK["add_link_forces: each edge once, acting on both ends<br/>src/layout/simulation.rs:196 and src/layout/simulation.rs:262"]
    BATCH --> LINK
    LINK --> STIFF["stiffness = link_strength / min(degree source, degree target),<br/>correction shared by degree, self-loops skipped<br/>src/layout/simulation.rs:268-277 and src/layout/simulation.rs:291-298"]
    STIFF --> SUM["forces: Vec of Vector2 in node index order<br/>src/layout/simulation.rs:216"]
    SIGN["many-body sign: negative charge repels on every path,<br/>magnitude held below one unit, MIN_DISTANCE_SQ<br/>src/layout/force.rs:19 and src/layout/force.rs:27-41"] --- BATCH
    N1["CHANGED at 0779174 (0.3.0): springs ran per outgoing neighbour on the source only, at unit stiffness per edge,<br/>the exact kernel attracted at short range, and centring was summed here as a per-node force.<br/>Centring is now recenter after integration, see VW-03.1"]
    LINK --- N1
    T1["INVARIANT: every real fixture settles into a bounded layout through the WebVowl API with default settings<br/>tests/webvowl_layout_test.rs:229"]
    SUM --- T1
```
- Barnes-Hut only engages above the `n > 50` threshold (src/layout/simulation.rs:175) — small graphs always take the O(n²) batch path. `runSimulation` settles every fixture too (tests/webvowl_layout_test.rs:263).

## VW-03.3 `QuadTree` — build, insert/subdivide, Barnes-Hut recursion
```mermaid
flowchart TB
    B["QuadTree::build(nodes)<br/>src/layout/quadtree.rs:129"] --> INS["insert(node_ref) per node<br/>src/layout/quadtree.rs:169"]
    INS --> COM["update center_of_mass, total_mass<br/>src/layout/quadtree.rs:171-176"]
    COM --> HASCH{"children already exist?"}
    HASCH -->|yes| DELEGATE["delegate to child quadrant<br/>get_quadrant(x,y)<br/>src/layout/quadtree.rs:181-184"]
    HASCH -->|no| PUSH["push into self.nodes<br/>src/layout/quadtree.rs:187"]
    PUSH --> CAP{"nodes.len() > capacity AND depth < 32?<br/>src/layout/quadtree.rs:189-194"}
    CAP -->|yes| SUB["subdivide(): 4 children,<br/>redistribute existing nodes<br/>src/layout/quadtree.rs:198-251"]
    CAP -->|no| DONE1["leaf, unchanged"]
    B --> CF["calculate_force(node_pos, node_id, theta, strength)<br/>src/layout/quadtree.rs:274"]
    CF --> LEAF{"self.children.is_none()?<br/>src/layout/quadtree.rs:287"}
    LEAF -->|yes, 1 node = self| ZERO["return zero (self-interaction)<br/>src/layout/quadtree.rs:288-290"]
    LEAF -->|"yes, other leaves"| EXACT["calculate_exact_force<br/>src/layout/quadtree.rs:293, detail in VW-03.4"]
    LEAF -->|no| THETA{"s/distance < theta?<br/>src/layout/quadtree.rs:308"}
    THETA -->|yes| APPROX["approximate via center_of_mass,<br/>strength*total_mass/max(distance², 1)<br/>src/layout/quadtree.rs:309-312"]
    THETA -->|no| RECURSE["recurse into all 4 children,<br/>sum their forces<br/>src/layout/quadtree.rs:316-320"]
```
- **Failure boundary:** RESOLVED at vowl-wasm 2aff199 — insertion used to have no depth or minimum-cell guard, so 51 nodes pinned at one coordinate trapped `WebVowl.tick()` during repeated subdivision. Each cell now records its depth and stops subdividing at depth 32, keeping coincident points in a bounded leaf (src/layout/quadtree.rs:189-194, src/layout/quadtree.rs:240-242), and a test covers 51 coincident nodes (src/layout/quadtree.rs:660). The separate CSR tree has a maximum depth of 40 (src/layout/csr_sim.rs:156, src/layout/csr_sim.rs:259).
- `theta` (typical 0.5–0.9, doc comment src/layout/quadtree.rs:269) trades layout accuracy for speed: smaller theta forces more recursion (exact), larger theta approximates more aggressively.

## VW-03.4 Leaf-force SIMD/scalar dispatch
```mermaid
flowchart LR
    LEAF["calculate_exact_force(node_pos, node_id, strength)<br/>src/layout/quadtree.rs:332"] --> FILT["sources = leaf nodes minus self<br/>src/layout/quadtree.rs:339-343"]
    FILT --> GATE{"feature: simd AND<br/>sources.len() >= 4?<br/>src/layout/quadtree.rs:350,352"}
    GATE -->|yes, wasm32| SIMDW["unsafe calculate_leaf_force_simd()<br/>src/layout/simd.rs:390, target_feature simd128"]
    GATE -->|yes, non-wasm32| SIMDN["calculate_leaf_force_simd() scalar shim<br/>src/layout/simd.rs:486"]
    GATE -->|no| SCALAR["scalar fold over sources:<br/>strength*mass/max(distance_sq, 1) toward the source, deterministic<br/>perturbation for coincident points<br/>src/layout/quadtree.rs:386-402"]
    SIMDW --> OUT["Vector2<f64> force"]
    SIMDN --> OUT
    SCALAR --> OUT
```
- INVARIANT: coincident nodes (`distance_sq < 0.0001`) never divide by zero — both the scalar leaf path (src/layout/quadtree.rs:395-398) and the CSR tick (VW-03.5) substitute a deterministic sine/cosine-derived nudge.
- `is_simd_available()` differs by target: `#[cfg(target_arch = "wasm32")]` (src/layout/simd.rs:523) returns the compile-time `cfg!(feature = "simd")`; it does not probe browser support; the non-wasm32 stub (src/layout/simd.rs:532) always returns `false`.

## VW-03.5 `CsrSimulation::tick` — deterministic worker physics, Barnes-Hut by default
```mermaid
flowchart TB
    T["tick()<br/>src/layout/csr_sim.rs:476"] --> FIN{"finished or n==0?"}
    FIN -->|yes| RET["return, no-op<br/>src/layout/csr_sim.rs:477-479"]
    FIN -->|no| REP["repulsion_forces dispatches on config.repulsion,<br/>default BarnesHut theta 0.9<br/>src/layout/csr_sim.rs:485 and src/layout/csr_sim.rs:687-692 and src/layout/csr_sim.rs:108"]
    REP -->|"BarnesHut(theta)"| BH["one QuadTree::build per tick, one stack walk per node,<br/>a cell collapses to its centre of charge when width² &lt; theta² · d²<br/>and the node is outside it; both magnitudes use d².max(soft2) at src/layout/csr_sim.rs:765 and src/layout/csr_sim.rs:780<br/>src/layout/csr_sim.rs:732-794 and src/layout/csr_sim.rs:778"]
    REP -->|"Exact, theta &lt;= 0"| EX["O(n²) Coulomb over every unordered pair once, magnitude charge / d².max(soft2)<br/>src/layout/csr_sim.rs:695-729 and src/layout/csr_sim.rs:720"]
    BH --> COIN{"d² < 1e-6 (coincident)?"}
    EX --> COIN
    COIN -->|yes| NUDGE["index-derived sin/cos perturbation, same pair rule on both paths<br/>src/layout/csr_sim.rs:162 and src/layout/csr_sim.rs:706-711"]
    COIN -->|no| SPRING
    NUDGE --> SPRING["springs: Hooke toward link_distance<br/>over edge_a/edge_b/edge_type<br/>src/layout/csr_sim.rs:487-507"]
    SPRING --> CG{"componentGravity > 0?<br/>src/layout/csr_sim.rs:511"}
    CG -->|yes| CGF["component_gravity_forces: rigid pull of each connected component toward center<br/>src/layout/csr_sim.rs:512 and src/layout/csr_sim.rs:588, see VW-03.11"]
    CG -->|no| INTEG
    CGF --> INTEG["centre gravity + damped velocity update per node; a pinned node is skipped (src/layout/csr_sim.rs:522-524)<br/>and speed is clamped to MAX_STEP_FACTOR times link_distance (src/layout/csr_sim.rs:534-539)<br/>src/layout/csr_sim.rs:515-542"]
    INTEG --> COL{"collideRadius > 0?<br/>src/layout/csr_sim.rs:546"}
    COL -->|yes| COLF["collide() collideIterations times on predicted positions, then the same speed cap<br/>src/layout/csr_sim.rs:547-560, see VW-03.11"]
    COL -->|no| POS
    COLF --> POS["integrate: positions += velocities, pinned skipped<br/>src/layout/csr_sim.rs:562-568"]
    N2["CHANGED at 0779174 (0.4.0, PR #6): velocity update and position integration were one loop,<br/>they are now split so collision can act between them. Both new branches are skipped, not zeroed, when off<br/>src/layout/csr_sim.rs:509-510"]
    COL --- N2
    POS --> DECAY["alpha *= 1 - alpha_decay<br/>src/layout/csr_sim.rs:570"]
    DECAY --> FLOOR{"alpha < alpha_min?"}
    FLOOR -->|yes| LATCH["alpha = alpha_min; finished = true<br/>src/layout/csr_sim.rs:571-574"]
    FLOOR -->|no| DONE["tick complete, positions updated in place"]
    N1["DEBT: ForceSimulation and CsrSimulation each carry their own Barnes-Hut quadtree, a node-reference tree in quadtree.rs and an arena tree over the flat position buffer in csr_sim.rs, so a fix to one does not reach the other<br/>src/layout/quadtree.rs:129 and src/layout/csr_sim.rs:206-215"]
    REP --- N1
```
- Since ER-20 `CsrSimulation` is no longer always O(n²): the module doc now states Barnes-Hut is the default and `Repulsion::Exact`, `set_theta(0.0)` or `set_param("theta", 0.0)` select the exact loop, which runs unchanged for comparison (src/layout/csr_sim.rs:9-17, src/layout/csr_sim.rs:880, src/layout/csr_sim.rs:1020). The default `SimConfig` is still described as tuned for a scoped tier of at most 1,500 nodes (src/layout/csr_sim.rs:99).
- `finished` is a latched `bool` (src/layout/csr_sim.rs:308,573) set once the alpha floor is hit — `tick()` after that point is a guaranteed no-op, not merely a small-alpha update.
- For valid finite configuration, two guards bound motion in a tick since the 0.2.2 pin. Repulsion is softened: below a radius of `SOFTENING_FRACTION` (0.25) times `link_distance` the magnitude is held at its value at that radius (`softening_sq`, src/layout/csr_sim.rs:579-582, constants src/layout/csr_sim.rs:148-150), so a near-coincident seed pair is not flung thousands of units in one tick; and each node's per-tick displacement is capped at `MAX_STEP_FACTOR` (2.0) times `link_distance` (src/layout/csr_sim.rs:520, src/layout/csr_sim.rs:534-539). The test `near_coincident_seeds_stay_in_bounds` pins both for the exact and the Barnes-Hut path (src/layout/csr_sim.rs:1332).
- Drag pinning: `pin(i, x, y)` writes the position, zeroes the velocity and sets `pinned[i]`; a pinned node still exerts forces but is skipped at integration (src/layout/csr_sim.rs:522-524, src/layout/csr_sim.rs:902-913). It returns `false`, changing nothing, for an index outside the scope or a non-finite coordinate. `unpin` and `is_pinned` follow (src/layout/csr_sim.rs:916-929), `NggExplorer` exposes them as `pinNode` and `unpinNode` (src/bindings/explorer.rs:225-233), and every load clears the pins (src/layout/csr_sim.rs:427, src/layout/csr_sim.rs:465). `pin` does not reheat, so a caller pairs it with `reheat` (src/layout/csr_sim.rs:933).

## VW-03.6 NGG1 tier load — bytes to seeded simulation
```mermaid
sequenceDiagram
    autonumber
    participant JS as JS worker: loadCsr(bytes)
    participant NX as NggExplorer::load_csr<br/>src/bindings/explorer.rs:80
    participant CS as CsrSimulation::load_csr<br/>src/layout/csr_sim.rs:452
    participant N1 as ontomark codec<br/>src/ngg1.rs:5
    JS->>NX: loadCsr(bytes: &[u8])
    NX->>CS: self.sim.load_csr(bytes)
    CS->>N1: Ngg1::parse(bytes), the codec is the ontomark-ngg1 crate<br/>src/layout/csr_sim.rs:453 and src/ngg1.rs:5
    N1-->>CS: Ngg1<'_> view (borrowed, no realloc)
    CS->>CS: load(&g): seed positions, build CSR springs<br/>src/layout/csr_sim.rs:461
    CS-->>NX: Result<(), Ngg1Error>
    NX-->>JS: Result<(), JsValue> (mapped via to_string)<br/>src/bindings/explorer.rs:81-83
```
- The reader is no longer in this repository: `src/ngg1.rs` is one `pub use ontomark_ngg1::*;` so the `vowl_wasm::ngg1` path and every public name keep working (src/ngg1.rs:1-5), and the crate depends on `ontomark-ngg1` 0.1.0 from crates.io behind the `ngg1` feature (Cargo.toml:59-60, Cargo.toml:81). Header size, node stride, field offsets and the specification are therefore owned by ontomark, not by `vowl-wasm` (docs/adr/ADR-001-shared-ontology-rendering-contract.md:36).
- The external codec owns byte-layout details. Current native migration tests and the built-WASM golden test exercise compatibility rather than duplicating those constants here (tests/ontomark_ngg1_migration.rs:501-509, tests/ngg1_wasm.test.mjs:24).
- `CsrSimulation::load_csr` maps a parse failure to `Ngg1Error` and `NggExplorer::load_csr` maps that to a string for JS (src/layout/csr_sim.rs:452, src/bindings/explorer.rs:80-83).

## VW-03.7 Alpha annealing — shared shape, two independent implementations
```mermaid
stateDiagram-v2
    [*] --> Running: alpha = config.alpha (1.0)
    Running --> Running: tick() decays alpha *= (1 - alpha_decay)
    Running --> Finished: alpha < alpha_min
    Finished --> Running: reheat(alpha), both engines<br/>src/layout/csr_sim.rs:933 and src/layout/simulation.rs:81 / set_param("reheat") src/layout/csr_sim.rs:1021
    note right of Finished
      Legacy: recomputes from alpha.
      Reheat validates then assigns alpha.
      src/layout/simulation.rs:81-85
      CSR: latched flag; reheat or a
      non-empty load clears it.
      src/layout/csr_sim.rs:471-472
      src/layout/csr_sim.rs:933-939
    end note
```
- Both engines can now be re-annealed from JS: `ForceSimulation::reheat` clamps alpha to `(0, 1]` and ignores a non-finite or non-positive value (src/layout/simulation.rs:81-85), `CsrSimulation::reheat` ignores a non-finite value, assigns `max(value, alpha_min)` and sets `finished` only for an empty graph (src/layout/csr_sim.rs:933-939); the `WebVowl` binding exposes `reheat` (src/bindings/mod.rs:144-147) and `NggExplorer` exposes `reheat` and `setTheta` (src/bindings/explorer.rs:216-239). This closes the earlier gap where a settled `WebVowl` graph could only restart through a fresh `initSimulation()`.
- **Input boundary:** RESOLVED at vowl-wasm 2aff199 for non-finite input — every CSR parameter setter and `reheat` now returns early on a non-finite value, so an infinite reheat no longer produces non-finite positions (src/layout/csr_sim.rs:844-896, src/layout/csr_sim.rs:933-936, dispatch src/layout/csr_sim.rs:1012-1027). Finite out-of-range values are still accepted: the earlier probe showed a negative `alphaMin` prevents settling in 10,000 ticks, and `set_alpha_min` checks only finiteness (src/layout/csr_sim.rs:891-895).
- `ForceSimulation` implements `LayoutAlgorithm` (src/layout/mod.rs:17); `CsrSimulation` deliberately does not — the two engines share a design (annealing, Hooke springs, Coulomb repulsion) but not a common Rust interface.

## VW-03.9 Seeding — `ForceSimulation::initialize` vs `CsrSimulation::load`
```mermaid
flowchart TB
    subgraph FS["Legacy initialise<br/>src/layout/simulation.rs:339"]
        FI1["initialize_positions(graph)<br/>src/layout/simulation.rs:110"] --> FI2{"preserve_seeded_positions AND<br/>node already placed?<br/>src/layout/simulation.rs:132-133"}
        FI2 -->|no| FI3["seed on circle, radius=seed_radius<br/>angle += TAU/node_count<br/>src/layout/simulation.rs:134-135"]
        FI2 -->|yes| FI4["keep caller's x,y"]
        FI3 --> FI5["vx,vy reset to 0 regardless<br/>src/layout/simulation.rs:139-140"]
        FI4 --> FI5
        FI5 --> FI6["alpha = config.alpha; iteration = 0<br/>src/layout/simulation.rs:341-342"]
    end
    subgraph CS["CSR load<br/>src/layout/csr_sim.rs:452,461"]
        CI1["Ngg1::parse(bytes)<br/>src/layout/csr_sim.rs:453"] --> CI2["positions = g.seed_positions()<br/>src/layout/csr_sim.rs:463"]
        CI2 --> CI3["velocities = zeroed Vec, len n*2; pinned = all false<br/>src/layout/csr_sim.rs:464-465"]
        CI3 --> CI4["edge_a/edge_b/edge_type from g.edges()<br/>src/layout/csr_sim.rs:466-469"]
        CI4 --> CI5["alpha = config.alpha;<br/>finished = (n == 0)<br/>src/layout/csr_sim.rs:471-472"]
    end
```
- `ForceSimulation` seeds a fresh circular layout unless a caller opted into `preserve_seeded_positions`; `CsrSimulation` always takes its seed positions straight from the NGG1 tier's own `seed_positions()` — there is no circular-seed path on the CSR side.
- Both zero velocities and restore configured alpha on entry, so re-`initialize`/re-`load` restarts annealing rather than resuming (contrast `reheat`, VW-03.7).

## VW-03.10 Engine and tuning-preset choice
```mermaid
flowchart TB
    N["graph to lay out"] --> SRC{"source is an NGG1 tier<br/>(worker-loaded; defaults tuned for up to 1,500 nodes,<br/>not an enforced size limit)?<br/>src/layout/csr_sim.rs:99"}
    SRC -->|yes| CSR["CsrSimulation<br/>flat Vec&lt;f32&gt; CSR buffers,<br/>Barnes-Hut repulsion by default, exact on theta &lt;= 0<br/>src/layout/csr_sim.rs:108"]
    SRC -->|no, VowlGraph| FSIM["ForceSimulation<br/>LayoutAlgorithm impl<br/>src/layout/simulation.rs:30"]
    FSIM --> SIZE{"graph size vs<br/>LayoutConfig preset?"}
    SIZE -->|"small/medium,<br/>default()"| DFLT["link_distance 30, charge -30,<br/>use_barnes_hut true (n&gt;50 engages it)<br/>src/layout/mod.rs:115-127"]
    SIZE -->|"large, dense<br/>ontology, large_graph()"| BIG["link_distance 250, charge -2000,<br/>center_strength 0.001, seed_radius 2000<br/>src/layout/mod.rs:154-160"]
    CSR --> TUNE["SimConfig: charge 2000, link_distance 60,<br/>damping 0.6, center_gravity 0.02<br/>src/layout/csr_sim.rs:101-112"]
```
- `large_graph()`'s doc comment (src/layout/mod.rs:134-137) records it was tuned empirically on a 1,155-node ontology and deliberately over-spreads a small graph, which is why it is a named preset rather than the default; Barnes-Hut engagement above `n > 50` is detailed in VW-03.2.
- `CsrSimulation` has no size-tiered preset — its `SimConfig::default()` is the only named preset, commented as tuned for the scoped tier of at most 1,500 nodes (src/layout/csr_sim.rs:99-112); callers can change repulsion and numeric force parameters at run time (src/layout/csr_sim.rs:844-885, src/layout/csr_sim.rs:1012-1027).

## VW-03.11 Opt-in readability forces on `CsrSimulation` — collision and component gravity
```mermaid
flowchart TB
    KEYS["setParam keys collideRadius, collideStrength, collideIterations, componentGravity<br/>src/layout/csr_sim.rs:1022-1025"] --> SET["setters ignore non-finite values and clamp:<br/>radius at least 0, strength 0..1, iterations rounded into 1..16, gravity 0..1<br/>src/layout/csr_sim.rs:944-976"]
    SET --> RD["Readability struct kept apart from SimConfig, defaults radius 0, strength 1, iterations 1, gravity 0<br/>src/layout/csr_sim.rs:119 and src/layout/csr_sim.rs:130-140"]
    RD --> COL["collide: quadtree over predicted positions p + v, each overlapping pair pushed apart by strength times overlap,<br/>all of it to the free node when the other is pinned<br/>src/layout/csr_sim.rs:616 and src/layout/csr_sim.rs:668-679"]
    RD --> CGR["component_gravity_forces: every member of a component gets gravity times (center minus centroid)<br/>src/layout/csr_sim.rs:588-610"]
    LOAD["every load relabels connected components by union-find<br/>src/layout/csr_sim.rs:341 and src/layout/csr_sim.rs:432,470"] --> CGR
    G1["INVARIANT: with both forces off the trajectory is byte-identical to 0.3.0,<br/>positions after 50 and 600 ticks match the tests/fixtures/ngg-golden buffers for seven ontologies<br/>tests/ngg_explorer_layout_test.rs:10-13 and tests/ngg_explorer_layout_test.rs:175-186"]
    G2["recommended 21, 0.7, 3, 0.1 leaves no pair closer than 40 and peak extent within 1.25 times the default run<br/>tests/ngg_explorer_layout_test.rs:337-342 and tests/ngg_explorer_layout_test.rs:390-407"]
    COL --- G1
    CGR --- G2
    PKG["the layout tests, fixture ontologies and goldens are excluded from the published crate for their licences<br/>Cargo.toml:20-34"]
    G1 --- PKG
```
- New in 0.4.0 (PR #6). `NggExplorer::setParam` documents the same keys and the starting values for hosts drawing nodes up to 12 px (src/bindings/explorer.rs:189-206); consumers that turn them on are VW-04.6. The legacy `ForceSimulation` has no collision or component force.
