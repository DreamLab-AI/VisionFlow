---
id: VC-58
title: Corpus page to GPU edge — what an edge is at each hop, and what it loses
area: visionclaw
governing:
  - ../project/docs/GPU-wire-abi.md
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
adrs: [ADR-2071, ADR-2112, ADR-2114, ADR-2140]
sources:
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/page_parser.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/crates/visionclaw-domain/src/vault/mod.rs
  - ../project/src/services/inferred_edge_materialiser.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/physics_orchestrator_actor.rs
  - ../project/src/actors/gpu/force_compute_actor.rs
  - ../project/src/utils/unified_gpu_compute/memory.rs
  - ../project/src/utils/unified_gpu_compute/semantic.rs
  - ../project/src/utils/unified_gpu_compute/execution.rs
  - ../project/src/gpu/semantic_forces.rs
  - ../project/src/services/semantic_type_registry.rs
  - ../project/crates/visionclaw-gpu/src/cuda_sources/visionclaw_unified.cu
  - ../project/crates/visionclaw-gpu/src/cuda_sources/semantic_forces.cu
  - ../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md
verified_commit: {visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e}
---

## For developers

Follows one corpus page from the vault walk to the two GPU structures that consume its edges: the CSR spring graph, which keeps only an undirected weight, and the semantic COO list, which keeps a per-edge type id. Read with VC-21 (ingest), VC-11 (the step) and VC-59 (each force term). Drawn from the code at main 46edc196b; the physics rework planned in November 2026 is not drawn here.

## For the business

A relation written in a page (`is-a`, `requires`, `has-part`) reaches the layout as a spring whose stiffness comes from a shared table. The layout engine never sees which relation it was: two pages joined by `requires` and two joined by a plain link pull together by the same law, only with different strength. A change that only adds or removes links does not reach the layout until the number of pages changes.

## VC-58.1 Page to node: list, fetch, parse, gate

```mermaid
flowchart TB
    LIST["LocalCorpusSource.list_pages<br/>local.rs:230<br/>walk under VAULT_BASE_PATHS, default knowledge,working<br/>local.rs:160, local.rs:21"] --> FETCH["fetch_page: raw markdown<br/>local.rs:262"]
    FETCH --> PARSE["parse_page, frontmatter into a page<br/>github_sync_service.rs:1802<br/>page_parser.rs:73, vault_core call page_parser.rs:87"]
    PARSE -->|"no ontology type"| PLAIN["process_plain_vault_file<br/>github_sync_service.rs:2008"]
    PARSE --> GATE{"page_is_kg_included<br/>github_sync_service.rs:2255<br/>is_kg_included, public or owl_class<br/>vault/mod.rs:168"}
    PLAIN --> GATE
    GATE -->|"excluded"| DROP["page not in the graph<br/>github_sync_service.rs:1833"]
    GATE -->|"included"| NODE["build_node_from_entity<br/>github_sync_service.rs:144<br/>enrich_node_from_frontmatter github_sync_service.rs:2309<br/>node.weight = quality github_sync_service.rs:2344"]
    NODE --> WLC["metadata wikilink_count<br/>github_sync_service.rs:1857-1861"]
    WLC -.-> DRIFTW["DRIFT: the comment calls it the weight signal the GPU<br/>can consume for mass, github_sync_service.rs:1857-1859<br/>but no GPU path reads it, and the uploaded class masses<br/>are uniform 1.0, force_compute_actor.rs:1080"]
```

**What it shows:** the corpus source lists and fetches pages, the vault parser turns frontmatter into a page, and only pages that are public or carry an `owl_class` become nodes. **Why it is this way:** ADR-2114 put every corpus behind one `CorpusSource`, and ADR-2112 made frontmatter the only ontology carrier.

**Drift:** `github_sync_service.rs:1857-1859` describes `wikilink_count` as "the weight signal the GPU can consume for connectivity-based mass", but at 46edc196b it is written (`../project/src/services/github_sync_service.rs:1861`) and never read on the GPU path; the class masses uploaded with the graph are all 1.0 (`../project/src/actors/gpu/force_compute_actor.rs:1080`), and degree enters only through the degree-weight buffer (`../project/src/actors/gpu/force_compute_actor.rs:1194-1205`).

## VC-58.2 Edge producers: seven kinds, their labels and weights

```mermaid
flowchart TB
    PAGE["included page"] --> WL["wikilink: explicit_link, weight 1.0, no IRI<br/>github_sync_service.rs:1880-1896"]
    PAGE --> EL["elevated_from, weight 1.0<br/>github_sync_service.rs:1902, github_sync_service.rs:1911-1912"]
    PAGE --> TY["typed relation: relation_edges<br/>github_sync_service.rs:1941<br/>label from predicate_to_edge_type, many IRIs to one label<br/>github_sync_service.rs:2556-2558<br/>IRI and metadata relation kept"]
    TY --> W2["weight = registry strength times 2.0<br/>github_sync_service.rs:1962-1966"]
    TY -->|"replaces the wikilink for the same pair"| OVR["edges.remove of the pair's wikilink<br/>github_sync_service.rs:1929"]
    PAGE --> CO["dangling target: no node, co_citation spring<br/>COCITE_WEIGHT 0.5, folded-link bonus 0.1<br/>github_sync_service.rs:488-489, github_sync_service.rs:580<br/>stub folding github_sync_service.rs:747"]
    PAGE --> DOM["domain spoke: domain_member, weight 1.5<br/>domain_membership_edge github_sync_service.rs:2540-2545<br/>materialise_domain_roots github_sync_service.rs:907"]
    PAGE --> INF["inferred: hierarchical plus rdfs:subClassOf<br/>metadata inferred, weight 1.0<br/>build_inferred_edge inferred_edge_materialiser.rs:68<br/>selected github_sync_service.rs:1209, called github_sync_service.rs:1429"]
    W2 -.-> T2140["TENSION: ADR-2140 says a value is applied exactly as stored,<br/>ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28<br/>the sync doubles every registry strength into the CSR weight"]
```

**What it shows:** every edge kind the sync writes, the label it carries and its spring weight. Only typed relations carry a predicate IRI and a `relation` name; co-citation edges stand in for links to pages that do not exist. **Why it is this way:** the factor 2.0 normalises the registry's 0–1 strength to the 0–2 spring range (comment at `github_sync_service.rs:1963`).

**Tension (ADR-2140 vs code):** ADR-2140 decision 2 says "there is no scale factor, multiplier, clamp or cap between the store and the GPU" (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:28`), while every typed edge's CSR weight is the registry strength times 2.0 (`../project/src/services/github_sync_service.rs:1966`). The registry strength is a code-held default, not a stored setting, so the ADR's own drift note on scope applies (`../project/docs/adr/ADR-2140-physics-forces-have-one-stored-definition-and-no-scale-factors.md:65`).

## VC-58.3 Persist, reload and the hand-off to the GPU actors

```mermaid
sequenceDiagram
    autonumber
    participant SYNC as GitHubSyncService<br/>github_sync_service.rs:1941
    participant OX as OxigraphGraphRepository<br/>oxigraph_graph_repository.rs:533
    participant GS as GraphStateActor<br/>graph_state_actor.rs:984
    participant SUP as GraphServiceSupervisor<br/>graph_service_supervisor.rs:1639
    participant PO as update_graph_data<br/>physics_orchestrator_actor.rs:531
    participant FCA as ForceComputeActor<br/>force_compute_actor.rs:3162
    SYNC->>OX: write edge, relationshipType, weight, owlProperty (oxigraph_graph_repository.rs:542, :546, :551)
    Note over OX: an edge with no type is stored as default (:533)<br/>the edge metadata map is not written
    GS->>OX: ReloadGraphFromDatabase, load_graph (oxigraph_graph_repository.rs:847)
    OX-->>GS: edges rebuilt with metadata None (oxigraph_graph_repository.rs:1512)
    GS->>GS: remap_to_compact_ids, nodes 0..N-1 (graph_state_actor.rs:192)
    SUP->>PO: UpdateGraphData (graph_service_supervisor.rs:1639, handler physics_orchestrator_actor.rs:1464)
    Note over PO: forwards only when the node count changed (:541)
    PO->>FCA: UpdateGPUGraphData (physics_orchestrator_actor.rs:557)
    FCA->>FCA: store pending graph, upload once the GPU context exists (force_compute_actor.rs:3173, :879)
```

**What it shows:** the typed edge survives Oxigraph as label, weight and predicate IRI, but its metadata does not; the orchestrator re-forwards a graph to the GPU only when the node count differs. **Why it is this way:** the count check closed the "UpdateGPUGraphData propagation gap" for a growing graph (comment at `physics_orchestrator_actor.rs:537-540`); it was not written for edge-only changes.

**Debt:** `metadata["relation"]` and `metadata["inferred"]` are set at build time (`../project/src/services/github_sync_service.rs:1975`, `../project/src/services/inferred_edge_materialiser.rs:68`) but every reloaded edge has `metadata: None` (`../project/src/adapters/oxigraph_graph_repository.rs:1512`), so after a restart-time reload an inferred edge is told apart only by its label and IRI.

**Tension:** a sync that adds or removes edges but leaves the node count unchanged is not forwarded to the GPU (`../project/src/actors/physics_orchestrator_actor.rs:541`), so the layout keeps the previous edge set until the next count change or GPU re-initialisation.

## VC-58.4 Two GPU shapes of one edge: CSR drops the type, COO keeps it

```mermaid
flowchart TB
    G["pending GraphData<br/>try_upload_pending_graph_data force_compute_actor.rs:879"] --> ADJ["adjacency lists: target and edge.weight<br/>both directions pushed<br/>force_compute_actor.rs:941-951"]
    ADJ --> CSR["row offsets, col indices, weights<br/>force_compute_actor.rs:961-972"]
    CSR --> INIT["initialize_graph memory.rs:863<br/>upload_edges_csr memory.rs:204<br/>called force_compute_actor.rs:1032"]
    INIT --> SPR["edge_spring_scale visionclaw_unified.cu:422<br/>LinLog log1pf of dist times w times s visionclaw_unified.cu:437<br/>Hooke spring_k times displacement times w times s visionclaw_unified.cu:451"]
    ADJ -.-> DEBTCSR["DEBT: edge_type, predicate IRI and direction<br/>are dropped here, force_compute_actor.rs:947-949"]
    G --> SEM["refresh_semantic force_compute_actor.rs:4773<br/>SemanticForcesEngine.initialize semantic_forces.rs:72<br/>edge_type_to_int per edge semantic_forces.rs:103"]
    SEM --> REG["registry id, unknown label gives 0<br/>semantic_type_registry.rs:726-731"]
    SEM --> COO["COO src, dst, type, weight<br/>SemanticInputs.from_engine semantic.rs:160"]
    REG --> TABLE["relationship table build_gpu_buffer<br/>semantic_type_registry.rs:689<br/>set_relationships semantic.rs:318-328<br/>enabled whenever the table is non-empty force_compute_actor.rs:4791-4795"]
    COO --> ATTR["apply_attribute_spring_force semantic_forces.cu:357"]
    TABLE --> DYN["apply_dynamic_relationship_force semantic_forces.cu:492<br/>launched when dynamic_enabled semantic.rs:448"]
    ATTR --> FORCES["added to the force buffers before integration<br/>execution.rs:963"]
    DYN --> FORCES
    SPR --> FORCES
```

**What it shows:** the same edge reaches the device twice. The CSR keeps only an undirected weight for the spring and repulsion pass; the semantic COO list keeps a per-edge registry id that indexes the relationship table. **Why it is this way:** the CSR was built for the force pass and SSSP, which need neighbours and weights only; the semantic pass was added later with its own buffers (ADR-2140 moved its configuration into the stored document).

**Debt:** relation type, predicate IRI and direction are discarded when the CSR is built (`../project/src/actors/gpu/force_compute_actor.rs:947-949`), so the main spring cannot tell `requires` from `has-part` except through the doubled strength in the weight.

## VC-58.5 Registry ids are positions, not names

```mermaid
flowchart TB
    NEW["SemanticTypeRegistry defaults<br/>register_internal semantic_type_registry.rs:612"] --> D1["dependency, strength 0.6<br/>semantic_type_registry.rs:159-162"]
    NEW --> D2["dependency again, strength 0.7<br/>semantic_type_registry.rs:424-427"]
    D1 --> MAP["uri_map.insert overwrites, configs.push appends<br/>semantic_type_registry.rs:619-621"]
    D2 --> MAP
    MAP --> ORPH["DEBT: the first dependency row stays in the device table<br/>with no name pointing at it"]
    AUTO["labels the defaults do not know<br/>get_or_register_id github_sync_service.rs:1962<br/>default strength 0.5 semantic_type_registry.rs:81"] --> MAP
    MAP --> CAP["device table capacity 256 types<br/>semantic_forces.cu:66, c_dynamic_relationships semantic_forces.cu:164"]
```

**What it shows:** type ids are allocation order: a default listed twice keeps two rows, and labels met during sync are appended after the defaults. **Why it is this way:** ids are rebuilt on every upload (VC-58.4), so nothing stores an id that would need migrating.

**Debt:** `dependency` is registered twice with different strengths (`../project/src/services/semantic_type_registry.rs:160`, `:425`); `register_internal` overwrites the name map but appends the config (`../project/src/services/semantic_type_registry.rs:619-621`), so the 0.6 row is unreachable by name while still uploaded.
