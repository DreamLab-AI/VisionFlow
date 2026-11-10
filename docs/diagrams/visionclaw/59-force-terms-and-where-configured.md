---
id: VC-59
title: Force terms and where each is configured — law, gate, setting or constant, and which pass runs it
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
adrs: [ADR-2029, ADR-2140, ADR-2169]
sources:
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/utils/unified_gpu_compute/memory.rs
  - ../project/src/utils/unified_gpu_compute/semantic.rs
  - ../project/src/models/force_channels.rs
  - ../project/src/config/dev_config.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/crates/visionclaw-domain/src/types/semantic_config.rs
  - ../project/src/services/semantic_type_registry.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

One page per force term on the default path at main 46edc196b: the law the kernel computes, the flag or pointer that switches it on, and the stored setting or code constant it reads. VC-11 draws the step that runs them, VC-58 the edge data they consume. The physics rework planned in November 2026 appears only as register entries.

## For the business

The layout has more sliders than live forces. In the default mode several sliders do nothing except switch a force off when set to zero, and some forces are fixed in code with no slider at all. This page is the map of which control moves which force.

## VC-59.1 Which force pass runs, and what each pass lacks

```mermaid
flowchart TB
    STEP["physics step, force stage<br/>execution.rs:659"] --> PICK{"stability_threshold above 0<br/>execution.rs:661-665"}
    DEF["stability_threshold 1e-4 from dev_config, not a stored setting<br/>dev_config.rs:202"] --> PICK
    PICK -->|"yes, the default"| STAB["force_pass_with_stability_kernel<br/>visionclaw_unified.cu:2228"]
    PICK -->|"no"| PLAIN["force_pass_kernel<br/>visionclaw_unified.cu:460"]
    PLAIN --> GRAV["gravity term, default 0.002<br/>visionclaw_unified.cu:659-663, physics_config.rs:563"]
    STAB -.-> NOGRAV["DEBT: the stability kernel has centering<br/>visionclaw_unified.cu:2409<br/>but no gravity term, so the stored gravity value<br/>does not act on the default pass"]
    STAB --> SEMP["semantic pass when semantic_active<br/>execution.rs:963"]
    PLAIN --> SEMP
```

**What it shows:** with the shipped `stability_threshold` the step runs the stability kernel, which carries repulsion, springs and centering but not the separate gravity term the plain kernel adds. **Why it is this way:** the stability kernel was added to skip settled frames; its force body was copied from the plain kernel without the later gravity term.

**Debt:** `gravity` (default 0.002, `../project/crates/visionclaw-domain/src/types/physics_config.rs:563`) is read only by `force_pass_kernel` (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:659-663`), and the default `stability_threshold` of 1e-4 (`../project/src/config/dev_config.rs:202`) selects the other kernel (`../project/src/utils/unified_gpu_compute/execution.rs:661-665`), so a stored gravity value has no effect on ordinary frames.

## VC-59.2 Pairwise terms: separation and repulsion

```mermaid
flowchart TB
    REPK["repel_k above 0, default 120<br/>force_channels.rs:490-492, physics_config.rs:558"] --> RFLAG["ENABLE_REPULSION<br/>visionclaw_unified.cu:518"]
    RFLAG --> SEP["separation push, max_force times penetration<br/>for dist below separation_radius<br/>visionclaw_unified.cu:560-568"]
    RFLAG --> WHICH{"lin_log_mode and node_degrees pointer present<br/>visionclaw_unified.cu:572"}
    WHICH -->|"yes"| FA2["FA2: scaling_ratio times deg_i+1 times deg_j+1 over dist<br/>visionclaw_unified.cu:573-577<br/>scaling_ratio default 10 physics_config.rs:601"]
    WHICH -->|"no"| COUL["Coulomb: repel_k over dist squared<br/>visionclaw_unified.cu:580<br/>same class times 0.1, other class times 3.0<br/>visionclaw_unified.cu:587-593"]
    FA2 --> CAP["each pair capped at max_force, default 150<br/>visionclaw_unified.cu:601, physics_config.rs:557"]
    COUL --> CAP
    DEG["node_degrees zeroed on resize memory.rs:578<br/>pointer passed once degree weights exist execution.rs:692-694<br/>flag set by upload_degree_weights memory.rs:849"] --> WHICH
```

**What it shows:** in the default LinLog mode repulsion is the FA2 law, but its degree terms read `node_degrees`, a buffer that is zero until an analytics run fills it; the Coulomb branch with its fixed class multipliers runs only when the pointer is absent. **Why it is this way:** ADR-2029 derives the feature word from the Hooke and Coulomb scalars, which predate the LinLog default.

**Tension (ADR-2029 flag rule vs the LinLog default):** `repel_k > 0` switches on ENABLE_REPULSION (`../project/src/models/force_channels.rs:490-492`), and that flag gates the FA2 branch too (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:518`, `:572-577`), which reads `scaling_ratio`, never `repel_k`. Setting the Coulomb slider to zero removes the live FA2 repulsion.

**Debt:** the FA2 degree factor is `node_degrees + 1`, but `node_degrees` is zeroed whenever the buffers resize (`../project/src/utils/unified_gpu_compute/memory.rs:578`) and the pointer is passed as soon as degree *weights* exist (`../project/src/utils/unified_gpu_compute/execution.rs:692-694`, `../project/src/utils/unified_gpu_compute/memory.rs:849`), so until an analytics run writes degrees every pair repels as `scaling_ratio / dist`.

**Tension (ADR-2140 vs code):** the Coulomb branch multiplies by hard-coded 0.1 and 3.0 by class (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:587-593`), constants that ADR-2140 decision 2 rules out (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28`). The branch is dead in the default mode while degree weights exist.

## VC-59.3 Edge terms on the CSR: LinLog, Hooke and the hold overlay

```mermaid
flowchart TB
    SPK["spring_k above 0, default 12<br/>force_channels.rs:493-495, physics_config.rs:559"] --> SFLAG["ENABLE_SPRINGS<br/>visionclaw_unified.cu:618"]
    SFLAG --> SCALE["s = spring_scale of the receiving node<br/>visionclaw_unified.cu:644"]
    SCALE --> MODE{"lin_log_mode, default true<br/>visionclaw_unified.cu:434, physics_config.rs:600"}
    MODE -->|"true"| LL["LinLog: log1pf of dist times w times s<br/>visionclaw_unified.cu:437<br/>reads neither spring_k nor rest_length"]
    MODE -->|"false"| HK["Hooke: spring_k times dist minus rest times w times s<br/>rest_length default 50, optional SSSP rest<br/>visionclaw_unified.cu:439-451, physics_config.rs:570"]
    LL --> HOLD["hold overlay: plus spring_k times dist minus snapshot<br/>only while a node is held<br/>visionclaw_unified.cu:453-456"]
    HK --> HOLD
    W["w = CSR weight, registry strength times 2.0 for typed edges<br/>github_sync_service.rs:1966"] --> LL
```

**What it shows:** the default attraction is LinLog, which uses the edge weight and the population multiplier but not `spring_k` or `rest_length`; `spring_k` acts only as the on switch and inside the hold overlay. **Why it is this way:** LinLog was made the default for modularity (comment at `visionclaw_unified.cu:435-436`); ADR-2169 added the overlay so a held node drags its neighbours.

**Tension (ADR-2029 flag rule vs the LinLog default):** `spring_k > 0` sets ENABLE_SPRINGS (`../project/src/models/force_channels.rs:493-495`), the flag that gates the LinLog branch (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:618`, `:437`) although LinLog never reads `spring_k`. A user zeroing a slider that appears inert removes all edge attraction.

## VC-59.4 Semantic terms on the COO list: attribute spring and relation table

```mermaid
flowchart TB
    ACT["semantic_active, any semantic force enabled<br/>execution.rs:963"] --> AS["apply_attribute_spring_force<br/>semantic_forces.cu:357<br/>enabled by default semantic_config.rs:133"]
    ACT --> DY["apply_dynamic_relationship_force<br/>semantic_forces.cu:492<br/>launched when the table is enabled semantic.rs:448"]
    WT["COO weight = the CSR weight<br/>semantic.rs:189"] --> AS
    AS --> ASL["k = base_spring_k times 1 + w times weight_multiplier<br/>rest = rest_length_max minus w times max minus min<br/>semantic_forces.cu:382-388<br/>defaults 0.01, 0.5, 30, 200 semantic_config.rs:129-132"]
    ASL -.-> NEG["DEBT: w reaches 1.8 for a 0.9 registry strength<br/>semantic_type_registry.rs:284, so rest = 200 minus 306<br/>is negative and the spring attracts at every distance"]
    TAB["per-type strength, rest_length, force_type<br/>semantic_type_registry.rs:689"] --> DY
    DY --> DYL["force_mag = strength times displacement over dist<br/>then times normalize of delta<br/>semantic_forces.cu:561-563"]
    DYL -.-> TWICE["DEBT: divided by dist twice, so the magnitude is<br/>strength times displacement over dist, not Hooke"]
```

**What it shows:** every edge also feeds two semantic springs: the attribute spring, whose stiffness and rest length come from the same doubled weight, and the per-type relation spring from the registry table. **Why it is this way:** the semantic forces were moved into the stored document by ADR-2140 but kept their original kernels.

**Debt:** the relation spring computes `strength * displacement / dist` and multiplies by the normalised direction (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:561-563`), dividing by distance twice; its own comment says Hooke's law (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:560`).

**Debt:** the attribute spring's rest length assumes a weight in 0–1 (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:385-388`), but it is fed the CSR weight (`../project/src/utils/unified_gpu_compute/semantic.rs:189`), which is twice the registry strength (`../project/src/services/github_sync_service.rs:1966`) and reaches 1.8 for a 0.9 entry (`../project/src/services/semantic_type_registry.rs:284`); with the defaults (`../project/crates/visionclaw-domain/src/types/semantic_config.rs:129-132`) the rest length is negative above a weight of about 1.18.

## VC-59.5 One registry strength, three springs

```mermaid
flowchart TB
    STR["registry strength for the edge's label<br/>get_or_register_id github_sync_service.rs:1962"] --> CSRW["1. CSR weight = strength times 2.0<br/>github_sync_service.rs:1966<br/>LinLog spring visionclaw_unified.cu:437"]
    CSRW --> ATT["2. attribute spring stiffness and rest<br/>from the same weight semantic_forces.cu:382-388"]
    STR --> REL["3. relation spring at strength<br/>table semantic_type_registry.rs:689<br/>kernel semantic_forces.cu:492"]
```

**What it shows:** a typed edge is pulled by three laws whose coefficients all derive from one registry strength. **Why it is this way:** each path was added separately; no record decides that one relation should have one spring.

**Open:** no ADR states which of the three springs is meant to carry a relation's strength; the November 2026 rework draft proposes one per-relation path, which is not decided at 46edc196b.
