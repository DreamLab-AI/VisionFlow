---
id: VC-59
title: Force terms and where each is configured — law, gate, setting or constant, and which pass runs it
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
adrs: [ADR-2029, ADR-2140, ADR-2169, ADR-2174, ADR-2204]
sources:
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/utils/unified_gpu_compute/memory.rs
  - ../project/src/utils/unified_gpu_compute/buffers.rs
  - ../project/src/utils/unified_gpu_compute/community.rs
  - ../project/src/utils/unified_gpu_compute/semantic.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/models/force_channels.rs
  - ../project/src/config/dev_config.rs
  - ../project/crates/visionclaw-domain/src/types/physics_config.rs
  - ../project/crates/visionclaw-domain/src/types/semantic_config.rs
  - ../project/src/services/semantic_type_registry.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/crates/visionclaw-domain/data/relations.yaml
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
  - ../project/docs/adr/ADR-2174-layout-forces-are-learned-per-relation-type.md
  - ../project/crates/relational-field/PREREGISTRATION.md
verified_commit: {visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e}
---

## For developers

One page per force term on the default path at main 4900ec60b: the law the kernel computes, the flag or pointer that switches it on, and the stored setting or code constant it reads. VC-11 draws the step that runs them, VC-58 the edge data they consume. Of the physics rework (ADR-2204, proposed), the landed slices are drawn: the FA2 repulsion buffer (D4) and schema-held edge weights (S2); the opt-in weighting modes are off by default and the rest appears only as register entries.

## For the business

The layout has more sliders than live forces. In the default mode several sliders do nothing except switch a force off when set to zero, and some forces are fixed in code with no slider at all. This page is the map of which control moves which force.

## VC-59.1 Which force pass runs, and what each pass lacks

```mermaid
flowchart TB
    STEP["physics step, force stage<br/>execution.rs:659"] --> PICK{"stability_threshold above 0<br/>execution.rs:661-665"}
    DEF["stability_threshold 1e-4 from dev_config, not a stored setting<br/>dev_config.rs:202"] --> PICK
    PICK -->|"yes, the default"| STAB["force_pass_with_stability_kernel<br/>visionclaw_unified.cu:2236"]
    PICK -->|"no"| PLAIN["force_pass_kernel<br/>visionclaw_unified.cu:460"]
    PLAIN --> GRAV["gravity term, default 0.002<br/>visionclaw_unified.cu:659-663, physics_config.rs:668"]
    STAB -.-> NOGRAV["DEBT: the stability kernel has centering<br/>visionclaw_unified.cu:2417<br/>but no gravity term, so the stored gravity value<br/>does not act on the default pass"]
    STAB --> SEMP["semantic pass when semantic_active<br/>execution.rs:961"]
    PLAIN --> SEMP
```

**What it shows:** with the shipped `stability_threshold` the step runs the stability kernel, which carries repulsion, springs and centering but not the separate gravity term the plain kernel adds. **Why it is this way:** the stability kernel was added to skip settled frames; its force body was copied from the plain kernel without the later gravity term.

**Debt:** `gravity` (default 0.002, `../project/crates/visionclaw-domain/src/types/physics_config.rs:668`) is read only by `force_pass_kernel` (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:659-663`), and the default `stability_threshold` of 1e-4 (`../project/src/config/dev_config.rs:202`) selects the other kernel (`../project/src/utils/unified_gpu_compute/execution.rs:661-665`), so a stored gravity value has no effect on ordinary frames.

## VC-59.2 Pairwise terms: separation and repulsion

```mermaid
flowchart TB
    REPK["repel_k above 0, default 120<br/>force_channels.rs:490-492, physics_config.rs:663"] --> RFLAG["ENABLE_REPULSION<br/>visionclaw_unified.cu:518"]
    RFLAG --> SEP["separation push, max_force times penetration<br/>for dist below separation_radius<br/>visionclaw_unified.cu:560-568"]
    RFLAG --> WHICH{"lin_log_mode and repulsion pointer present<br/>kernel argument node_degrees visionclaw_unified.cu:572"}
    WHICH -->|"yes"| FA2["FA2: scaling_ratio times b_i+1 times b_j+1 over dist<br/>visionclaw_unified.cu:573-577<br/>scaling_ratio default 0.1 physics_config.rs:714"]
    WHICH -->|"no"| COUL["Coulomb: repel_k over dist squared<br/>visionclaw_unified.cu:580<br/>same class times 0.1, other class times 3.0<br/>visionclaw_unified.cu:587-593"]
    FA2 --> CAP["each pair capped at max_force, default 150<br/>visionclaw_unified.cu:601, physics_config.rs:662"]
    COUL --> CAP
    DEG["repulsion_strength allocated zeroed buffers.rs:157, reset on graph change buffers.rs:303<br/>pointer passed once uploaded execution.rs:691-695<br/>upload_repulsion memory.rs:324, uniform writes zeros memory.rs:326<br/>called after every degree-weight upload force_compute_actor.rs:1255<br/>repulsionMode default uniform physics_config.rs:148-149"] --> WHICH
```

**What it shows:** in the default LinLog mode repulsion is the FA2 law, whose per-node factor reads `repulsion_strength`; the default `uniform` mode fills it with zeros, so every pair repels as `scaling_ratio / dist`. The Coulomb branch with its fixed class multipliers runs only when the pointer is absent. **Why it is this way:** ADR-2029 derives the feature word from the Hooke and Coulomb scalars, which predate the LinLog default.

**Tension (ADR-2029 flag rule vs the LinLog default):** `repel_k > 0` switches on ENABLE_REPULSION (`../project/src/models/force_channels.rs:490-492`), and that flag gates the FA2 branch too (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:518`, `:572-577`), which reads `scaling_ratio`, never `repel_k`. Setting the Coulomb slider to zero removes the live FA2 repulsion.

RESOLVED at visionclaw 4900ec60 (ADR-2204 D4): the FA2 factor no longer shares a buffer with community detection, which now writes `community_degrees` (`../project/src/utils/unified_gpu_compute/community.rs:145`). Its one producer is `upload_repulsion` (`../project/src/utils/unified_gpu_compute/memory.rs:324`), run for the stored `repulsionMode` after every degree-weight upload (`../project/src/actors/gpu/force_compute_actor.rs:1255`); `uniform` (the default) writes zeros by design and `strength` writes node strength minus one.

**Tension (ADR-2140 vs code):** the Coulomb branch multiplies by hard-coded 0.1 and 3.0 by class (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:587-593`), constants that ADR-2140 decision 2 rules out (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28`). The branch is dead in the default mode once the repulsion buffer is uploaded (`../project/src/utils/unified_gpu_compute/execution.rs:691-695`).

## VC-59.3 Edge terms on the CSR: LinLog, Hooke and the hold overlay

```mermaid
flowchart TB
    SPK["spring_k above 0, default 12<br/>force_channels.rs:493-495, physics_config.rs:664"] --> SFLAG["ENABLE_SPRINGS<br/>visionclaw_unified.cu:618"]
    SFLAG --> SCALE["s = spring_scale of the receiving node<br/>visionclaw_unified.cu:644"]
    SCALE --> MODE{"lin_log_mode, default true<br/>visionclaw_unified.cu:434, physics_config.rs:713"}
    MODE -->|"true"| LL["LinLog: log1pf of dist times w times s<br/>visionclaw_unified.cu:437<br/>reads neither spring_k nor rest_length"]
    MODE -->|"false"| HK["Hooke: spring_k times dist minus rest times w times s<br/>rest_length default 50, optional SSSP rest<br/>visionclaw_unified.cu:439-451, physics_config.rs:675"]
    LL --> HOLD["hold overlay: plus spring_k times dist minus snapshot<br/>only while a node is held<br/>visionclaw_unified.cu:453-456"]
    HK --> HOLD
    W["w = CSR weight, the relation's legacy_weight for typed edges<br/>github_sync_service.rs:2149-2151, unchanged in default legacy weighting<br/>physics_config.rs:152-154"] --> LL
```

**What it shows:** the default attraction is LinLog, which uses the edge weight and the population multiplier but not `spring_k` or `rest_length`; `spring_k` acts only as the on switch and inside the hold overlay. **Why it is this way:** LinLog was made the default for modularity (comment at `visionclaw_unified.cu:435-436`); ADR-2169 added the overlay so a held node drags its neighbours.

**Tension (ADR-2029 flag rule vs the LinLog default):** `spring_k > 0` sets ENABLE_SPRINGS (`../project/src/models/force_channels.rs:493-495`), the flag that gates the LinLog branch (`../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu:618`, `:437`) although LinLog never reads `spring_k`. A user zeroing a slider that appears inert removes all edge attraction.

## VC-59.4 Semantic terms on the COO list: attribute spring and relation table

```mermaid
flowchart TB
    ACT["semantic_active, any semantic force enabled<br/>execution.rs:961"] --> AS["apply_attribute_spring_force<br/>semantic_forces.cu:357<br/>enabled by default semantic_config.rs:133"]
    ACT --> DY["apply_dynamic_relationship_force<br/>semantic_forces.cu:492<br/>launched when the table is enabled semantic.rs:448"]
    WT["COO weight = the edge's stored weight<br/>semantic.rs:189"] --> AS
    AS --> ASL["k = base_spring_k times 1 + w times weight_multiplier<br/>rest = rest_length_max minus w times max minus min<br/>semantic_forces.cu:382-388<br/>defaults 0.01, 0.5, 30, 200 semantic_config.rs:129-132"]
    ASL -.-> NEG["DEBT: w reaches 1.6 for hierarchical<br/>relations.yaml:65, so rest = 200 minus 272<br/>is negative and the spring attracts at every distance"]
    TAB["per-type strength, rest_length, force_type<br/>semantic_type_registry.rs:692"] --> DY
    DY --> DYL["force_mag = strength times displacement over dist<br/>then times normalize of delta<br/>semantic_forces.cu:561-563"]
    DYL -.-> TWICE["DEBT: divided by dist twice, so the magnitude is<br/>strength times displacement over dist, not Hooke"]
```

**What it shows:** every edge also feeds two semantic springs: the attribute spring, whose stiffness and rest length come from the same doubled weight, and the per-type relation spring from the registry table. **Why it is this way:** the semantic forces were moved into the stored document by ADR-2140 but kept their original kernels.

**Debt:** the relation spring computes `strength * displacement / dist` and multiplies by the normalised direction (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:561-563`), dividing by distance twice; its own comment says Hooke's law (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:560`).

**Debt:** the attribute spring's rest length assumes a weight in 0–1 (`../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu:385-388`), but it is fed the edge's stored weight (`../project/src/utils/unified_gpu_compute/semantic.rs:189`), which for a typed edge is the relation's `legacy_weight` (`../project/src/services/github_sync_service.rs:2149-2151`) and reaches 1.6 for `hierarchical` (`../project/crates/visionclaw-domain/data/relations.yaml:65`); with the defaults (`../project/crates/visionclaw-domain/src/types/semantic_config.rs:129-132`) the rest length is negative above a weight of about 1.18.

## VC-59.5 One relation strength, held twice, three springs

```mermaid
flowchart TB
    STR["registry strength for the edge's label<br/>kept equal to legacy_weight over 2 by a test<br/>semantic_type_registry.rs:871-888"] --> CSRW["1. CSR weight = legacy_weight from relations.yaml<br/>github_sync_service.rs:2149-2151<br/>LinLog spring visionclaw_unified.cu:437"]
    CSRW --> ATT["2. attribute spring stiffness and rest<br/>from the same weight semantic_forces.cu:382-388"]
    STR --> REL["3. relation spring at strength<br/>table semantic_type_registry.rs:692<br/>kernel semantic_forces.cu:492"]
```

**What it shows:** a typed edge is pulled by three laws whose coefficients all derive from one strength, now held twice: as `legacy_weight` in the relation schema and as the registry strength, kept in step by a test. **Why it is this way:** each path was added separately; ADR-2204 moved the edge weight to the schema and leaves the registry feeding the relation kernel until its slice S4 (`../project/src/services/semantic_type_registry.rs:867-869`).

**Open (ADR-2174):** no ADR states which of the three springs is meant to carry a relation's strength. ADR-2174 proposes learning per-relation forces offline and writing an integration ADR only if H1 holds (`../project/docs/adr/ADR-2174-layout-forces-are-learned-per-relation-type.md:50-52`). The Result is now on main: H1 is not supported as stated, with gates 1, 3 and 5 failing (`../project/crates/relational-field/PREREGISTRATION.md:263-266`). The three passing gates do not establish typed geometry, and the learned parameters do not reproduce RF geometry in the simulator (`../project/crates/relational-field/PREREGISTRATION.md:408-422`). The live force laws above are unchanged.
