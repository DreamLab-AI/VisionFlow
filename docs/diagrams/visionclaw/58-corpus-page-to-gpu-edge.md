---
id: VC-58
title: Corpus page to GPU edge — what an edge is at each hop, and what it loses
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
adrs: [ADR-2071, ADR-2112, ADR-2114, ADR-2140, ADR-2204]
sources:
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/page_parser.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/crates/visionclaw-domain/src/vault/mod.rs
  - ../project/crates/visionclaw-domain/data/relations.yaml
  - ../project/crates/visionclaw-domain/src/relations.rs
  - ../project/crates/visionclaw-domain/src/weighting.rs
  - ../project/crates/visionclaw-domain/src/models/edge.rs
  - ../project/src/services/inferred_edge_materialiser.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/actors/gpu/edge_csr.rs
  - ../project/src/utils/unified_gpu_compute/memory.rs
  - ../project/src/utils/unified_gpu_compute/semantic.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/gpu/semantic_forces.rs
  - ../project/src/services/semantic_type_registry.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
verified_commit: {visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e}
---

## For developers

Follows one corpus page from the vault walk to the two GPU structures that consume its edges: the CSR spring graph, which keeps only an undirected weight, and the semantic COO list, which keeps a per-edge type id. Read with VC-21 (ingest), VC-11 (the step) and VC-59 (each force term). Drawn at visionclaw 4900ec60, after ADR-2204 slices S1 to S4a: relation labels and edge weights are read from one schema (`relations.yaml`), edge provenance survives the store, and a host-side weighting stage sits between the stored edges and the CSR. In its default profile that stage passes every stored weight through unchanged.

## For the business

A relation written in a page (`is-a`, `requires`, `has-part`) reaches the layout as a spring whose stiffness comes from one declared table. The layout engine never sees which relation it was: two pages joined by `requires` and two joined by a plain link pull together by the same law, only with different strength. A relation the table does not know is now set aside with a warning instead of being given a guessed strength. Any change to the pages or their links now reaches the layout; earlier, only a change in the number of pages did.

## VC-58.1 Page to node: list, fetch, parse, gate

```mermaid
flowchart TB
    LIST["LocalCorpusSource.list_pages<br/>local.rs:230<br/>walk under VAULT_BASE_PATHS, default knowledge,working<br/>local.rs:160, local.rs:21"] --> FETCH["fetch_page: raw markdown<br/>local.rs:262"]
    FETCH --> PARSE["parse_page, frontmatter into a page<br/>github_sync_service.rs:1970<br/>page_parser.rs:73, vault_core call page_parser.rs:87"]
    PARSE -->|"no ontology type"| PLAIN["process_plain_vault_file<br/>github_sync_service.rs:2193"]
    PARSE --> GATE{"page_is_kg_included<br/>github_sync_service.rs:2455<br/>is_kg_included, public or owl_class<br/>vault/mod.rs:168"}
    PLAIN --> GATE
    GATE -->|"excluded"| DROP["page not in the graph<br/>github_sync_service.rs:2001"]
    GATE -->|"included"| NODE["build_node_from_entity<br/>github_sync_service.rs:103<br/>enrich_node_from_frontmatter github_sync_service.rs:2509<br/>node.weight = quality github_sync_service.rs:2544"]
    NODE --> WLC["metadata wikilink_count<br/>github_sync_service.rs:2025-2029"]
    WLC -.-> DRIFTW["DRIFT: the comment calls it the weight signal the GPU<br/>can consume for mass, github_sync_service.rs:2025-2027<br/>but no GPU path reads it: the uploaded masses<br/>come from CSR degree, force_compute_actor.rs:1269-1272"]
```

**What it shows:** the corpus source lists and fetches pages, the vault parser turns frontmatter into a page, and only pages that are public or carry an `owl_class` become nodes. **Why it is this way:** ADR-2114 put every corpus behind one `CorpusSource`, and ADR-2112 made frontmatter the only ontology carrier.

**Drift:** `github_sync_service.rs:2025-2027` describes `wikilink_count` as "the weight signal the GPU can consume for connectivity-based mass", but it is written (`../project/src/services/github_sync_service.rs:2029`) and never read on the GPU path. Connectivity-based mass does exist, from the CSR instead: the class masses first uploaded as 1.0 (`../project/src/actors/gpu/force_compute_actor.rs:1087`) are replaced by `0.5 + 2w`, capped at 5, where `w` is the median-normalised `ln(1 + degree)` of the uploaded CSR (`../project/src/actors/gpu/force_compute_actor.rs:1207-1212`, `../project/src/actors/gpu/force_compute_actor.rs:1269-1272`, `../project/src/actors/gpu/force_compute_actor.rs:1285`). The same degree feeds the degree-weight buffer, and under the opt-in `strength` repulsion mode the CSR weights also fill the repulsion buffer (`../project/src/actors/gpu/force_compute_actor.rs:1254`, `../project/src/utils/unified_gpu_compute/memory.rs:324`). Corrected on 23 November: the 010e1109b drawing said the masses stayed at 1.0, but the degree-based replacement was already in the code at that revision.

## VC-58.2 Edge producers: one schema for labels and weights

```mermaid
flowchart TB
    PAGE["included page"] --> WL["wikilink: explicit_link, schema weight 1.0, no IRI<br/>github_sync_service.rs:2048-2065<br/>relations.yaml:243"]
    PAGE --> EL["elevated_from, schema weight 1.0<br/>github_sync_service.rs:2071, github_sync_service.rs:2080-2081<br/>relations.yaml:254"]
    PAGE --> TY["typed relation: relation_edges<br/>github_sync_service.rs:2112<br/>label from schema.classify, many IRIs to one label<br/>github_sync_service.rs:2129, relations.rs:413<br/>IRI, relation metadata and Asserted provenance kept"]
    TY --> W2["weight = the relation's legacy_weight<br/>github_sync_service.rs:2149-2151<br/>for example dependency 1.4, relations.yaml:84"]
    TY -->|"predicate not in the schema"| QUAR["quarantined: counted warning, no edge<br/>github_sync_service.rs:2135-2146<br/>rdf:type is ignored, relations.yaml:50-51"]
    TY -->|"replaces the wikilink for the same pair"| OVR["edges.remove of the pair's wikilink<br/>github_sync_service.rs:2099"]
    PAGE --> CO["dangling target: no node, co_citation spring<br/>schema weight 0.5 per shared target, relations.yaml:264<br/>at most fanout_threshold referrers, default 3<br/>github_sync_service.rs:588, relations.yaml:45<br/>folded-link bonus 0.1, github_sync_service.rs:525<br/>shared targets in metadata, github_sync_service.rs:617<br/>stub folding github_sync_service.rs:802"]
    PAGE --> DOM["domain spoke: domain_member, schema weight 1.5<br/>domain_membership_edge github_sync_service.rs:2755-2766<br/>relations.yaml:274<br/>materialise_domain_roots github_sync_service.rs:963"]
    PAGE --> INF["inferred: hierarchical plus rdfs:subClassOf<br/>Inferred provenance, schema weight 1.0<br/>build_inferred_edge inferred_edge_materialiser.rs:71-77<br/>relations.yaml:287<br/>selected github_sync_service.rs:1270, called github_sync_service.rs:1512"]
    W2 -.-> T2140["CHANGED: no factor is computed any more<br/>the schema records registry strength times 2 as literals<br/>relations.yaml:20-22<br/>ADR-2140 now scopes applied as stored to the stored settings<br/>ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:235"]
```

**What it shows:** every edge kind the sync writes, the label it carries and its spring weight, all read from `relations.yaml`. Only typed relations carry a predicate IRI and a `relation` name; co-citation edges stand in for links to pages that do not exist. **Why it is this way:** ADR-2204 S2 moved labels and weights into one schema without changing any produced weight (`relations.yaml:19-23`), and a predicate the schema does not know is quarantined instead of being auto-registered with a default (`github_sync_service.rs:2135-2146`).

**Tension (ADR-2140 vs code), changed at visionclaw 4900ec60:** the 010e1109b drawing found every typed edge weight computed as the registry strength times 2.0, against ADR-2140 decision 2 (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28`). That multiplication is gone: the typed weight is the schema's `legacy_weight` (`../project/src/services/github_sync_service.rs:2149-2151`), which records the same product as a literal (`../project/crates/visionclaw-domain/data/relations.yaml:20-22`). The values are still code-held, not stored settings, and the 2026-11-20 amendment narrows "applied exactly as stored" to the stored settings, leaving the decision to ADR-2204 (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:235`).

## VC-58.3 Persist, reload and the hand-off to the GPU actors

```mermaid
sequenceDiagram
    autonumber
    participant SYNC as GitHubSyncService<br/>github_sync_service.rs:2112
    participant OX as OxigraphGraphRepository<br/>oxigraph_graph_repository.rs:533
    participant GS as GraphStateActor<br/>graph_state_actor.rs:965
    participant SUP as GraphServiceSupervisor<br/>graph_service_supervisor.rs:1653
    participant PO as update_graph_data<br/>physics_orchestrator_actor.rs:535
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3436
    SYNC->>OX: write edge, relationshipType, weight, owlProperty, provenance (oxigraph_graph_repository.rs:542, :546, :551, :557)
    Note over OX: an edge with no type is stored as default (:533)<br/>the edge metadata map is not written
    GS->>OX: ReloadGraphFromDatabase, load_graph (oxigraph_graph_repository.rs:862)
    OX-->>GS: edges rebuilt with label, IRI and provenance, metadata only the inferred flag (oxigraph_graph_repository.rs:1526-1534)
    Note over GS: DRIFT: edge.rs:207-208 says reload maps default back to None<br/>the stored label is copied unchanged, oxigraph_graph_repository.rs:1516
    GS->>GS: remap_to_compact_ids, sorted by persistent id, then 0..N-1 (graph_state_actor.rs:205, :1136)
    SUP->>PO: UpdateGraphData (graph_service_supervisor.rs:1653, handler physics_orchestrator_actor.rs:1471)
    Note over PO: forwards when the content fingerprint changed (:539, :548)<br/>node ids plus each edge source, target, type and weight (:1941)
    PO->>FCA: UpdateGPUGraphData (physics_orchestrator_actor.rs:564)
    FCA->>FCA: store pending graph, upload once the GPU context exists (force_compute_actor.rs:3447, :897)
```

**What it shows:** the typed edge survives Oxigraph as label, weight, predicate IRI and provenance, but its metadata map does not, and the orchestrator re-forwards a graph to the GPU whenever its content fingerprint changes. **Why it is this way:** ADR-2204 S1 made provenance survive store and reload; the fingerprint replaced a node-count check that could not see a swapped node or an edge-only change (`physics_orchestrator_actor.rs:1935-1940`).

**Debt:** `metadata["relation"]` and a co-citation edge's shared-target list are set at build time (`../project/src/services/github_sync_service.rs:2159`, `../project/src/services/github_sync_service.rs:617`), but reload restores only the inferred flag (`../project/src/adapters/oxigraph_graph_repository.rs:1631-1639`). Evidence weighting then counts each reloaded co-citation pair as one undamped item (`../project/crates/visionclaw-domain/src/weighting.rs:325-326`) instead of one damped item per shared target.

**Drift:** the comment on `UNTYPED_EDGE_LABEL` says reload maps the stored `default` back to `None` (`../project/crates/visionclaw-domain/src/models/edge.rs:207-208`), but reload copies the stored label into `edge_type` unchanged (`../project/src/adapters/oxigraph_graph_repository.rs:1516`, `../project/src/adapters/oxigraph_graph_repository.rs:1531`).

**Tension:** RESOLVED at visionclaw 4900ec60. At 010e1109b a sync that changed edges but not the node count was never forwarded to the GPU. The gate now compares a content fingerprint (`../project/src/actors/physics_orchestrator_actor.rs:538-539`, `../project/src/actors/physics_orchestrator_actor.rs:548`, `../project/src/actors/physics_orchestrator_actor.rs:1941`).

## VC-58.4 Two GPU shapes of one edge: CSR drops the type, COO keeps it

```mermaid
flowchart TB
    G["pending GraphData<br/>try_upload_pending_graph_data force_compute_actor.rs:897"] --> WEIGH["weighting stage physics_pairs<br/>force_compute_actor.rs:962<br/>Legacy default: stored weights unchanged<br/>weighting.rs:251-252, weighting.rs:532<br/>hierarchy boost and Evidence mode read edge_type<br/>weighting.rs:646"]
    WEIGH --> PAIR["PhysicsPair: source, target, weight only<br/>weighting.rs:217-224"]
    PAIR --> ADJ["build_csr_from_pairs: both directions pushed<br/>edge_csr.rs:64, edge_csr.rs:75-77"]
    ADJ --> CSR["row offsets, col indices, weights<br/>force_compute_actor.rs:967-972"]
    CSR --> INIT["initialize_graph memory.rs:899<br/>upload_edges_csr memory.rs:220<br/>called force_compute_actor.rs:1039"]
    INIT --> SPR["edge_spring_scale visionclaw_unified.cu:422<br/>LinLog log1pf of dist times w times s visionclaw_unified.cu:437<br/>Hooke spring_k times displacement times w times s visionclaw_unified.cu:451"]
    PAIR -.-> DEBTCSR["DEBT: edge_type, predicate IRI and direction<br/>do not reach the CSR, weighting.rs:217-224"]
    G --> SEM["refresh_semantic force_compute_actor.rs:5064<br/>SemanticForcesEngine.initialize semantic_forces.rs:72<br/>edge_type_to_int per edge semantic_forces.rs:103"]
    SEM --> REG["registry id, unknown label gives 0<br/>semantic_type_registry.rs:729-734"]
    SEM --> COO["COO src, dst, type, weight<br/>SemanticInputs.from_engine semantic.rs:160"]
    REG --> TABLE["relationship table build_gpu_buffer<br/>semantic_type_registry.rs:692<br/>set_relationships semantic.rs:318-328<br/>enabled whenever the table is non-empty force_compute_actor.rs:5082-5086"]
    COO --> ATTR["apply_attribute_spring_force semantic_forces.cu:357"]
    TABLE --> DYN["apply_dynamic_relationship_force semantic_forces.cu:492<br/>launched when dynamic_enabled semantic.rs:448"]
    ATTR --> FORCES["added to the force buffers before integration<br/>execution.rs:961"]
    DYN --> FORCES
    SPR --> FORCES
```

**What it shows:** the same edge reaches the device twice. The CSR keeps only an undirected weight for the spring and repulsion pass, after the host weighting stage; the semantic COO list keeps a per-edge registry id that indexes the relationship table. **Why it is this way:** the CSR was built for the force pass and SSSP, which need neighbours and weights only; ADR-2204 put one weighting function in front of it, used by the first upload and by a live re-weight alike (`force_compute_actor.rs:959-961`).

**Debt:** relation type, predicate IRI and direction do not survive into the physics pair the CSR is built from (`../project/crates/visionclaw-domain/src/weighting.rs:217-224`), so the main spring cannot tell `requires` from `has-part` except through the weight the schema gives each label. The weighting stage can read the type first (`../project/crates/visionclaw-domain/src/weighting.rs:646`), but only for the hierarchy boost and Evidence mode, both off in the default profile (`../project/crates/visionclaw-domain/src/weighting.rs:202-204`).

## VC-58.5 Registry ids are positions, not names

```mermaid
flowchart TB
    NEW["SemanticTypeRegistry defaults<br/>register_internal semantic_type_registry.rs:615"] --> D1["dependency, strength 0.6<br/>semantic_type_registry.rs:159-162"]
    NEW --> D2["dependency again, strength 0.7<br/>semantic_type_registry.rs:427-430"]
    D1 --> MAP["uri_map.insert overwrites, configs.push appends<br/>semantic_type_registry.rs:622-624"]
    D2 --> MAP
    MAP --> ORPH["DEBT: the first dependency row stays in the device table<br/>with no name pointing at it"]
    AUTO["the sync no longer registers labels it meets<br/>an unknown predicate is quarantined instead<br/>semantic_type_registry.rs:379-382"] -.-> MAP
    SCH["co_citation and elevated_from are schema labels<br/>with no registry row, so their type id is 0<br/>relations.yaml:251, relations.yaml:262"] --> REG0["edge_type_to_int falls back to 0<br/>semantic_type_registry.rs:729-734"]
    MAP --> CAP["device table capacity 256 types<br/>semantic_forces.cu:66, c_dynamic_relationships semantic_forces.cu:164"]
```

**What it shows:** type ids are allocation order: a default listed twice keeps two rows, and since ADR-2204 nothing is appended during sync, so a schema label without a default row is typed as the generic id 0. **Why it is this way:** ids are rebuilt on every upload (VC-58.4), so nothing stores an id that would need migrating.

**Debt:** `dependency` is registered twice with different strengths (`../project/src/services/semantic_type_registry.rs:160`, `:428`); `register_internal` overwrites the name map but appends the config (`../project/src/services/semantic_type_registry.rs:622-624`), so the 0.6 row is unreachable by name while still uploaded. The schema's own note records the shadowed row (`../project/crates/visionclaw-domain/data/relations.yaml:85`).
