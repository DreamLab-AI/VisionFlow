---
id: VC-20
title: Ontology pipeline - OWL extraction, Oxigraph, Whelk reasoning, governed mutation
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
  - ../project/docs/SECURITY-profiles.md
adrs: [ADR-2004, ADR-2071, ADR-2064, ADR-2066, ADR-2068, ADR-2114, ADR-2116, ADR-2125, ADR-2147, ADR-OM-009]
sources:
  - ../project/src/bin/load_ontology.rs
  - ../project/src/services/owl_extractor_service.rs
  - ../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/page_parser.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs
  - ../project/crates/visionclaw-adapters/src/whelk_inference_engine.rs
  - ../project/src/services/inferred_edge_materialiser.rs
  - ../project/src/services/ontology_class_index.rs
  - ../project/src/handlers/ontology_handler.rs
  - ../project/src/handlers/ontology_derived_handler.rs
  - ../project/src/handlers/ontology_class_count_handler.rs
  - ../project/src/handlers/ontology_agent_handler.rs
  - ../project/src/handlers/api_handler/ontology/mod.rs
  - ../project/src/handlers/api_handler/ontology_physics/mod.rs
  - ../project/src/actors/ontology_actor.rs
  - ../project/src/actors/messages/ontology_messages.rs
  - ../project/src/services/owl_validator.rs
  - ../project/src/inference/mod.rs
  - ../project/src/reasoning/mod.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/crates/vault/src/main.rs
  - ../project/crates/vault/src/certificate.rs
  - ../ontomark/crates/rdfs-certificate/src/lib.rs
  - ../ontomark/crates/ontomark-compile/src/certificate.rs
  - ../ontomark/crates/ontomark-compile/src/build/mod.rs
  - ../project/crates/vault/src/propose.rs
  - ../project/crates/vault/src/whelk.rs
  - ../project/crates/vault/src/build/turtle.rs
  - ../project/crates/vault-core/src/proposal.rs
  - ../project/crates/vault/tests/certificate_oo_cert.rs
  - ../project/crates/visionclaw-ontology/shapes/ontology-class.shacl.ttl
  - ../project/crates/visionclaw-ontology/src/services/jsonld_validator/shacl.rs
  - ../project/src/main.rs
  - ../project/src/handlers/utils.rs
  - ../project/crates/visionclaw-adapters/src/sparql_migrations.rs
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/settings/models.rs
  - ../project/Cargo.toml
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/services/role_store.rs
  - ../project/src/services/ontology_visibility.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/docs/SECURITY-profiles.md
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9, ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2}
---

## For developers

Corpus pages become OWL classes in Oxigraph's named graphs. Whelk runs EL++ reasoning after each sync, and inferred edges are materialised back into the graph. HTTP routes, the `OntologyActor` and the `vault certify` and `vault propose` CLI expose and govern the result. At ec785afa9 the `state-at` read gained a subjects view fed by the new corpus history (VC-20.7, VC-22.13), and the class-IRI mint for saved nodes is now one helper. At 809a859aa every ontology read surface also answers a class hidden from the caller as one never stored (VC-20.16, SECURITY-profiles 0.1.6).

## For the business

The ontology is the structured vocabulary behind the knowledge graph. Pages are turned into classes, a reasoner works out what follows from them, and every change to the vocabulary goes through a signed, reviewable proposal instead of a direct edit.

## VC-20.1 load_ontology bin - CorpusSource-backed corpus loader
```mermaid
sequenceDiagram
    autonumber
    participant Bin as main<br/>src/bin/load_ontology.rs:46
    participant Src as CorpusSource LocalDirectorySource<br/>src/services/corpus_source/local.rs:217
    participant Parser as parse_page project_ontology<br/>src/services/page_parser.rs:73
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:1826
    participant Extr as OwlExtractorService<br/>src/services/owl_extractor_service.rs:52

    Bin->>Src: LocalDirectorySource::new(cli_root, base_paths) or from_env() (lines 55-56, defn local.rs:41, :55)
    Bin->>Src: list_pages() (line 70)
    Src-->>Bin: Vec of CorpusPage
    Bin->>Src: vocabulary(), missing ontology/vocabulary.yaml is an error (lines 75-78)
    loop each page
        Bin->>Bin: source.fetch_page(page) (line 85)
        Bin->>Parser: parse_page(content, path, base_paths) (line 93, defn 73)
        Bin->>Bin: visionclaw_domain::vault::parse(content).is_kg_included() gate (line 96, see VC-21)
    end
    Bin->>Parser: project_ontology(ontology_pages, vocabulary) (line 108, defn 170)
    Parser-->>Bin: OntologyProjection classes axioms
    Bin->>Repo: persist -> save_ontology(classes, [], axioms) (line 115, persist defn 192, save_ontology defn oxigraph_ontology_repository.rs:1826)
    Repo->>Repo: single atomic INSERT DATA into GRAPH_ONTOLOGY, no per-item ASK dedup (oxigraph_ontology_repository.rs:1837, :1848)
    Bin->>Extr: OwlExtractorService::new(repo).extract_all_owl() (lines 130-131)
    Bin->>Repo: get_classes() (line 135, defn oxigraph_ontology_repository.rs:2522)
    Repo-->>Bin: Vec of OwlClass
    Bin->>Bin: exit(1) when ontology_files_parsed==0 or classes_persisted==0 (line 186)
    Note over Bin,Src: RESOLVED ADR-2114 - the bin now reads through the same CorpusSource port as live sync<br/>(LocalDirectorySource over VAULT_ROOT, ADR-2064), instead of a bespoke root-resolution scheme.<br/>The five hardcoded sample OwlClass literals and the per-class add_owl_class loop are gone.
    Note over Src: shared-corpus-layout is a DEFAULT feature (Cargo.toml:273, :275): list_pages validates the roots<br/>through ontomark_compile::profile::resolve_roots first (local.rs:230-232, checked_roots :120-124),<br/>the walk skips child symlinks (local.rs:175-182) and fetch_page refuses a canonical path<br/>outside the roots (local.rs:264-279). Built without the feature, the legacy walk follows links.
```

## VC-20.2 Oxigraph named-graph layout — ontology and knowledge graphs
```mermaid
erDiagram
    GRAPH_ONTOLOGY_ASSERT {
        string iri "urn:ngm:graph:ontology:assert (oxigraph_ontology_repository.rs:48)"
        string contents "OwlClass/OwlProperty/OwlAxiom triples, vc: ns"
    }
    GRAPH_ONTOLOGY_INFERRED {
        string iri "urn:ngm:graph:ontology:inferred (line 49)"
        string contents "whelk-derived SubClassOf/EquivalentClass closure"
    }
    GRAPH_KNOWLEDGE {
        string iri "urn:ngm:graph:knowledge (line 50)"
        string contents "KGNode + KGEdge triples"
    }
    GRAPH_AGENT {
        string iri "urn:ngm:graph:agent (line 51)"
        string contents "agent-flagged nodes"
    }
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_ASSERT : "add_owl_class add_axiom, oxigraph_ontology_repository.rs:1855 and 2612"
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_INFERRED : "store_inference_results DELETE-WHERE+INSERT DATA, oxigraph_ontology_repository.rs:2669 ADR-2004 D9"
    OxigraphGraphRepository ||--o{ GRAPH_KNOWLEDGE : "src/adapters/oxigraph_graph_repository.rs:6"
    OxigraphGraphRepository ||--o{ GRAPH_AGENT : "src/adapters/oxigraph_graph_repository.rs:16"
```

## VC-20.5 Oxigraph named-graph layout — shapes, provenance, cache and migrations graphs
```mermaid
erDiagram
    GRAPH_SHAPES {
        string iri "urn:ngm:graph:shapes (line 56)"
        string contents "SHACL shape triples loaded from ttl at startup"
    }
    GRAPH_PROVENANCE {
        string iri "urn:ngm:graph:provenance (line 57)"
        string contents "append-only PROV-O activity triples, see VC-22"
    }
    GRAPH_CACHE_SSSP {
        string iri "urn:ngm:graph:cache:sssp (oxigraph_ontology_repository.rs:73)"
        string contents "shortest-path cache quads, own sub-domain so CLEAR GRAPH invalidates atomically"
    }
    GRAPH_CACHE_APSP {
        string iri "urn:ngm:graph:cache:apsp (line 74)"
        string contents "all-pairs shortest-path cache quads"
    }
    GRAPH_MIGRATIONS {
        string iri "urn:ngm:graph:migrations (sparql_migrations.rs:44)"
        string contents "applied SPARQL migration ids, ADR-101 D2"
    }
    OxigraphOntologyRepository ||--o{ GRAPH_SHAPES : "load_shacl_shapes, oxigraph_ontology_repository.rs:411"
    OxigraphOntologyRepository ||--o{ GRAPH_PROVENANCE : "emit_provenance, oxigraph_ontology_repository.rs:652, see VC-22"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_SSSP : "own sub-domain, atomic CLEAR GRAPH, line 73"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_APSP : "line 74"
    SparqlMigrations ||--o{ GRAPH_MIGRATIONS : "sparql_migrations.rs:44, excluded from graph-export/GPU paths"
```

## VC-20.3 Whelk EL++ reasoning cycle (github_sync post-sync)
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1469
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:2522
    participant Engine as WhelkInferenceEngine<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:380
    participant Whelk as whelk-rs reasoner::assert<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:460

    Sync->>Repo: get_classes() inside reason_into_shared (github_sync_service.rs:1388, defn oxigraph_ontology_repository.rs:2522)
    Sync->>Repo: get_axioms() (github_sync_service.rs:1392)
    Sync->>Sync: axioms.extend(ngm_property_hierarchy_axioms()) (github_sync_service.rs:1395)
    Sync->>Engine: load_ontology(classes, axioms) into a fresh engine built per run (github_sync_service.rs:1414, :1416, defn whelk_inference_engine.rs:382)
    Engine->>Engine: compute_ontology_checksum(ontology) (whelk_inference_engine.rs:402)
    alt checksum unchanged since last load
        Engine-->>Sync: reuse cached_subsumptions (whelk_inference_engine.rs:414-416)
    else checksum changed
        Engine-->>Sync: cached_subsumptions cleared, fresh reasoning required (whelk_inference_engine.rs:409-413)
    end
    Sync->>Engine: infer() (github_sync_service.rs:1420, defn whelk_inference_engine.rs:432), after which the fresh engine is swapped into the shared one the ontology-agent routes read (github_sync_service.rs:1430)
    alt cached_subsumptions present (line 441)
        Engine-->>Sync: cached InferenceResults, whelk-rs not invoked (line 442-453)
    else no cache
        Engine->>Whelk: translate_ontology(ontology) (line 457)
        Engine->>Whelk: reasoner::assert(whelk_axioms) (line 460)
        Whelk-->>Engine: named_subsumptions() (line 462)
        Engine-->>Sync: InferenceResults inferred_axioms inference_time_ms (line 481-486)
    end
    Note over Engine: DOC-DRIFT - module doc claims complete EL reasoning (whelk_inference_engine.rs:5) but infer() lines 432-488 has no tokio timeout or axiom-count cap
    Sync->>Repo: store_inference_results(results) (github_sync_service.rs:1501, defn oxigraph_ontology_repository.rs:2669)
    Repo->>Repo: DELETE-WHERE GRAPH_ONTOLOGY_INFERRED then atomic INSERT DATA, one SPARQL Update (ADR-2004 D9, ADR-099 D3, oxigraph_ontology_repository.rs:2765-2769)
    Note over Sync: RESOLVED a6de5a421 - the former DIVERGENCE (OntologyReasoningService::infer_axioms ran CustomReasoner, not Whelk)<br/>is gone: that service, its test helper and both of its test files were deleted, the type having been constructed only by<br/>its own test helper. Production reasoning in this cycle is WhelkInferenceEngine only - the engine GitHubSyncService<br/>receives at github_sync_service.rs:303 and owns at :287, reasoning in a fresh one built at :1414, driven at :1416 and :1420 and swapped into the shared engine at :1430
```

**Drift (resolved a6de5a421):** this diagram recorded a second reasoner, `OntologyReasoningService`, that ran `CustomReasoner` beside Whelk and was never constructed outside its own tests; `a6de5a421` deleted the service, its test helper, `tests/ontology_reasoning_integration_test.rs` and `tests/unit/ontology_reasoning_test.rs`, so the only reasoner driven from the sync cycle is the `WhelkInferenceEngine` that `GitHubSyncService` owns (`src/services/github_sync_service.rs:287`, `src/services/github_sync_service.rs:303`, `src/services/github_sync_service.rs:1414`, `src/services/github_sync_service.rs:1416`, `src/services/github_sync_service.rs:1420`, `src/services/github_sync_service.rs:1430`).

**Debt:** RESOLVED at visionclaw 010e1109 — the second `CustomReasoner` home, `OntologyPipelineService::trigger_reasoning`, and its never-called entry point `on_ontology_modified` were deleted with the service and its orphan test by ADR-2192 (`../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md:54`); the sync cycle drawn here is the only reasoning path, and `run_post_sync_reasoning` now takes the stored `SyncSwitches` (`src/services/github_sync_service.rs:1469-1482`).

## VC-20.4 inferred-edge materialisation and class-summary index
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1469
    participant Sel as GitHubSyncService::select_inferred_edges_for_sync<br/>src/services/github_sync_service.rs:1298
    participant Mat as inferred_edge_materialiser<br/>src/services/inferred_edge_materialiser.rs:127
    participant KG as KnowledgeGraphRepository::batch_add_edges<br/>src/services/github_sync_service.rs:1561
    participant Idx as ontology_class_index::maybe_refresh_after_sync<br/>src/services/ontology_class_index.rs:468

    rect rgb(235,235,255)
    Note over Sync,KG: PATH A - pure selection function, called after Whelk infer<br/>(skipped when qualityGates.ontologyReasoning is off, github_sync_service.rs:1476-1482)
    Sync->>Sync: load_graph then mat::asserted_pairs(g.edges) both<br/>directions (github_sync_service.rs:1533, asserted_pairs defn inferred_edge_materialiser.rs:79)
    Sync->>Sel: select_inferred_edges_for_sync(inferred_axioms,<br/>resolver.resolve, asserted) (github_sync_service.rs:1540, defn :1298)
    Sel->>Mat: is_materialisable_subclass_pair drops self, owl#Nothing<br/>child, owl#Thing parent (github_sync_service.rs:1308, defn inferred_edge_materialiser.rs:169)
    Sel->>Mat: immediate_parents_from_subclass_pairs reduces transitive<br/>ancestors to immediate parents (github_sync_service.rs:1317, defn inferred_edge_materialiser.rs:181)
    loop each immediate (child_iri, parent_iri) pair
        Sel->>Sel: resolve(child_iri) resolve(parent_iri),<br/>unresolved_endpoints += 1 per miss (github_sync_service.rs:1327-1328, :1330, :1333)
    end
    Sel->>Mat: select_inferred_edges drops self-loops, asserted pairs,<br/>caps per child at 8 (github_sync_service.rs:1346, defn inferred_edge_materialiser.rs:127)
    Sel->>Mat: build_inferred_edge tags edge_type hierarchical,<br/>owl-property rdfs:subClassOf, metadata inferred=true<br/>(github_sync_service.rs:1354, defn inferred_edge_materialiser.rs:68)
    Sel-->>Sync: InferredEdgeSelection edges considered_axioms<br/>immediate_pairs unresolved_endpoints
    Sync->>KG: batch_add_edges(inferred_edges) (github_sync_service.rs:1561)
    Note over Sync,Sel: ADR-2071 - select_inferred_edges_for_sync delegates every rule to<br/>inferred_edge_materialiser (docstring github_sync_service.rs:1284-1297). Every edge carries<br/>metadata inferred=true, so edge_is_inferred classifies<br/>sync-produced edges onto the inferred channel.
    end
    Note over Sync,KG: PATH B REMOVED - RESOLVED at visionclaw 010e1109: OntologyPipelineService and its SemanticPhysicsConfig were deleted by ADR-2192<br/>(ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md:54), so select_inferred_edges_for_sync is the only caller<br/>(github_sync_service.rs:1292-1293). Writing is gated by SyncSwitches::writes_inferred_edges (github_sync_service.rs:249-251, :1547-1553)
    Sync->>Idx: maybe_refresh_after_sync(classes)<br/>(github_sync_service.rs:780, sync_graphs_with, defn line<br/>468)
    alt ONTOLOGY_CLASS_INDEX_ENABLED unset (default, ontology_class_index.rs:75)
        Idx-->>Sync: skip refresh (line 474)
    else enabled
        Idx->>Idx: refresh_class_index condense_class RuVector memory_store<br/>namespace ontology-classes (line 495, defn 378,<br/>DEFAULT_NAMESPACE line 33)
    end
```

**Debt:** RESOLVED at visionclaw 010e1109 — `POST /api/ontology-physics/enable|disable` no longer pass through the always-true `check_ontology_feature` (only `GET /constraints` does, `src/handlers/api_handler/ontology_physics/mod.rs:149`): they are thin writers of the stored physics keys through `apply_physics_patch` (`src/handlers/api_handler/ontology_physics/mod.rs:58-95`), and the `Settings.ontology_physics` bool is gone (ADR-2192).

**Drift:** RESOLVED at visionclaw 010e1109 — PATH B, which built its own ancestor map and skipped the vacuous-axiom filter, was removed with `OntologyPipelineService` (`../project/docs/adr/ADR-2192-the-reasoner-the-sync-pipeline-and-node-thinning-are-stored-operator-switches.md:54`), so the docstring claim that the paths cannot drift now holds trivially (`src/services/github_sync_service.rs:1284-1297`).

## VC-20.6 SPARQL read path - /ontology/query
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Route as POST /api/ontology/query<br/>src/handlers/api_handler/ontology/mod.rs:1833
    participant Handler as query_ontology<br/>src/handlers/ontology_handler.rs:1019
    participant Validator as validate_read_only_sparql<br/>src/handlers/ontology_handler.rs:885
    participant Clamp as clamp_sparql_limit<br/>src/handlers/ontology_handler.rs:966
    participant CQRS as QueryOntologyHandler
    participant Cap as cap_result_rows<br/>src/handlers/ontology_handler.rs:996

    Client->>Route: POST /ontology/query query
    Note over Route: gated RequireAuth::power_user().mutations_only() (api_handler/ontology/mod.rs:1822)
    Route->>Handler: query_ontology(auth, state, request) (ontology_handler.rs:1019)
    Handler->>Handler: caller_hidden_classes, 503 when the graph cannot be read, then answer_query_ontology (ontology_handler.rs:1025, :1035, see VC-20.16)
    Handler->>Validator: validate_read_only_sparql(query) (ontology_handler.rs:1041, defn :885)
    alt forbidden keyword INSERT DELETE DROP CLEAR LOAD CREATE ADD MOVE COPY WITH SERVICE
        Validator-->>Handler: Err operation not permitted (ontology_handler.rs:911, :917)
        Handler-->>Client: 400 bad_request
    else no read form present
        Validator-->>Handler: Err only SELECT ASK CONSTRUCT DESCRIBE permitted (ontology_handler.rs:930, :932)
        Handler-->>Client: 400 bad_request
    else valid read query
        Validator-->>Handler: Ok
        Handler->>Clamp: clamp_sparql_limit(query) (ontology_handler.rs:1052, defn :966)
        Clamp-->>Handler: LIMIT injected or clamped to MAX_SPARQL_ROWS 10000 (ontology_handler.rs:954, :984)
        Handler->>CQRS: QueryOntologyHandler.handle(QueryOntology query) off the worker (ontology_handler.rs:1059, see VC-20.13)
        CQRS-->>Handler: Vec of HashMap string string
        Handler->>Cap: cap_result_rows(hidden.query_rows(results)) - rows naming a hidden IRI dropped first (ontology_handler.rs:1065, defn :996)
        Cap-->>Handler: rows capped at 10000, byte fence MAX_SPARQL_RESULT_BYTES 8388608, truncated flag (ontology_handler.rs:956, :1000, :1009)
        Handler-->>Client: 200 results rowCount truncated
    end
    Note over Handler: ADR-2004 - handler-level fence mirrors adapter fence sparql_select_json (crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:886)
```

## VC-20.7 ontology HTTP route families
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Rbac as RbacGate on /api scope<br/>project/src/main.rs:1171
    participant OntScope as /api/ontology scope<br/>src/handlers/api_handler/ontology/mod.rs:1819
    participant DerivedScope as /api/ontology/derived scope<br/>src/handlers/ontology_derived_handler.rs:185
    participant CountScope as /api/ontology/class-count scope<br/>src/handlers/ontology_class_count_handler.rs:76
    participant AgentScope as /api/ontology-agent scope<br/>src/handlers/ontology_agent_handler.rs:468

    Client->>Rbac: any /api/* request
    Note over Rbac: required_level (rbac_gate.rs:147-189): safe GETs are public only when RBAC_PUBLIC_READS=1<br/>(fail-closed default, rbac_gate.rs:135-141), every other POST PUT DELETE needs WriteGraph (rbac_gate.rs:188)<br/>BEFORE any scope below is reached. Only /api/auth, client-logs and health probes are exempt (rbac_gate.rs:55-61)
    rect rgb(230,255,230)
    Note over OntScope: registration order matters - derived and class-count scopes register before this broader /ontology scope<br/>(project/src/main.rs:1181-1196), the single-scope collapse is recorded at ontology/mod.rs:1787-1803
    OntScope->>OntScope: POST graph classes properties axioms inference query sparql load load-axioms validate mapping apply, PUT classes iri and properties iri, DELETE cache axioms iri classes iri (ontology/mod.rs:1824-1844)
    OntScope->>OntScope: GET graph inferred classes properties axioms inference validate metrics inferences hierarchy state-at provenance reports id report health ws (ontology/mod.rs:1846-1871)
    Note over OntScope: power_user().mutations_only() gate ontology/mod.rs:1822 - POST PUT DELETE gated at power_user, GETs pass this scope<br/>(still subject to the RbacGate read rule, state-at and provenance also take the AuthenticatedUser extractor, ontology/mod.rs:1861-1867)
    Note over OntScope: GET state-at takes view=subjects since 9fd0e85cc and ab1e636ff (ontology/mod.rs:1446, :1652) - subject IRIs plus the<br/>earliest validFrom (build_state_at_subjects_sparql :1490, map_state_at_subjects :1541). Corpus history now feeds it, see VC-22.13
    end
    rect rgb(255,230,230)
    DerivedScope->>DerivedScope: POST derived, POST derived/regenerate, GET derived which (line 189-191)
    Note over DerivedScope: RequireAuth::power_user() whole scope (line 188), fence detail see VC-22
    end
    rect rgb(230,230,255)
    CountScope->>CountScope: GET class-count (line 77)
    Note over CountScope: no scope-level auth by design (ontology_class_count_handler.rs:25) - public only while RbacGate public reads are on
    end
    rect rgb(255,245,230)
    AgentScope->>AgentScope: POST discover read query traverse validate check, GET status (ontology_agent_handler.rs:471-477)
    AgentScope->>AgentScope: POST ontology-agent/propose, always 410 Gone (ontology_agent_handler.rs:481, handler defn :305)
    Note over AgentScope: ADR-2116 - the scope carries no auth middleware of its own. The /propose sub-scope and its<br/>RequireAuth+RateLimit middleware are gone with the write path they guarded (ontology_agent_handler.rs:298-299, :463-467) —<br/>propose() answers 410 Gone naming vault propose as the replacement. The governed<br/>write door moved to POST /decisions/record - see VC-24. ADR-2127 added /check with the tri-valued<br/>relation check and open-world scope fields on every answer.
    Note over Rbac,AgentScope: CORRECTED - not anonymous: the scope is mounted inside the RbacGate-wrapped /api scope<br/>(project/src/main.rs:1171, :1235), so every POST here, /propose included, needs WriteGraph (Editor+) before the<br/>handler runs (rbac_gate.rs:188), and GET /status is public only while RBAC_PUBLIC_READS=1
    end
    rect rgb(245,245,245)
    Note over OntScope: RESOLVED ADR-2066 - the /api/inference scope is gone. Its seven handlers extracted a web::Data InferenceService<br/>that was never constructed or registered, so every route 500'd at the extractor. inference_handler.rs,<br/>application/inference_service.rs, events/inference_triggers.rs and the main.rs registration were all removed<br/>(project/src/main.rs:1212-1216). The live reasoning path is GitHubSyncService::run_post_sync_reasoning - see VC-20.3
    end
```

**Drift:** ADR-2116 and the handler comment call every `/ontology-agent` route anonymous (`src/handlers/ontology_agent_handler.rs:463-467`), but the scope is mounted inside the `RbacGate`-wrapped `/api` scope (`../project/src/main.rs:1171`, `../project/src/main.rs:1235`) and `required_level` demands `WriteGraph` for every non-safe method outside the auth/health allowlist (`src/middleware/rbac_gate.rs:179-189`). In practice an unsigned caller is refused, while any signed caller passes, because an unassigned pubkey defaults to Editor (`src/services/role_store.rs:366`). Two comments disagree with the code: `src/middleware/rbac_gate.rs:200` calls `RBAC_PUBLIC_READS` default-on against the fail-closed `src/middleware/rbac_gate.rs:135-141`, and `../project/src/main.rs:1238` still says `/decisions/record` is authed like `/propose`.

## VC-20.8 OntologyActor message set

```mermaid
classDiagram
    class OntologyActor
    class InitializeActor
    class JobCompleted
    class LoadOntologyAxioms
    class UpdateOntologyMapping
    class ValidateOntology
    class ApplyInferences
    class GetOntologyReport
    class GetOntologyHealth
    class ClearOntologyCaches
    class TriggerReasoning
    class GetCachedOntologies
    class ApplyMaterializedAxioms

    OntologyActor ..> InitializeActor : Handler ontology_actor.rs 654
    OntologyActor ..> JobCompleted : Handler line 686
    OntologyActor ..> LoadOntologyAxioms : Handler line 696
    OntologyActor ..> UpdateOntologyMapping : Handler line 728
    OntologyActor ..> ValidateOntology : Handler line 738
    OntologyActor ..> ApplyInferences : Handler line 796
    OntologyActor ..> GetOntologyReport : Handler line 818
    OntologyActor ..> GetOntologyHealth : Handler line 853
    OntologyActor ..> ClearOntologyCaches : Handler line 885
    OntologyActor ..> TriggerReasoning : Handler line 909
    OntologyActor ..> GetCachedOntologies : Handler line 932

    note for LoadOntologyAxioms "LoadOntologyAxioms, ClearOntologyCaches,<br/>GetCachedOntologies, GetOntologyHealth now re-exported from<br/>visionclaw_actors::messages::ontology_messages<br/>(ontology_messages.rs:11-17), not defined in this webxr crate file"
    note for ValidateOntology "fields ontology_id, graph_data, mode, job_id<br/>(ontology_messages.rs:37-46), rtype Result of ValidationReport<br/>or String. owl_validator (src/services/owl_validator.rs:2) is a<br/>shim: pub use visionclaw_ontology::services::owl_validator::*<br/>(ADR-090 Phase A4), mirrored by src/inference/mod.rs:2 and<br/>src/reasoning/mod.rs:2 into the same crate.<br/>RESOLVED ADR-2065 - the orphaned second definition<br/>src/services/owl_validator_stubs.rs (a CPU-only<br/>ValidationConfig/PropertyGraph/RdfTriple/ConstraintSummary/<br/>ValidationReport set, no mod declaration or use anywhere,<br/>never compiled) has been deleted; owl_validator is now the<br/>single validation definition."
    note for ApplyInferences "fields rdf_triples, max_depth<br/>(ontology_messages.rs:52-55)"
    note for TriggerReasoning "fields ontology_id i64, source String<br/>(ontology_actor.rs:904-907), handler forwards to ReasoningActor<br/>per comment lines 918-920 which no longer exists"
    note for ApplyMaterializedAxioms "fields axioms, graph_data<br/>(ontology_messages.rs:76-79); NOT handled by OntologyActor -<br/>routed GPUManagerActor gpu_manager_actor.rs 651 to<br/>PhysicsSupervisor physics_supervisor.rs 747 to<br/>OntologyConstraintActor ontology_constraint_actor.rs 445"
    note for OntologyActor "RESOLVED - ProcessOntologyData and<br/>LogseqPage no longer exist in ontology_messages.rs; page<br/>ingest now runs through parse_page/project_ontology over the<br/>vault-core frontmatter model, see VC-20.1/VC-21"
```

## VC-20.13 Every ontology actix handler dispatches its sync CQRS handler off the worker

```mermaid
sequenceDiagram
    autonumber
    participant CL as Client
    participant AW as actix async worker thread
    participant H as get_owl_property<br/>src/handlers/ontology_handler.rs:355
    participant EX as execute_in_thread<br/>src/handlers/utils.rs:1
    participant BT as tokio blocking thread pool<br/>src/handlers/utils.rs:6
    participant QH as GetOwlPropertyHandler.handle<br/>src/handlers/ontology_handler.rs:380

    CL->>AW: GET /api/ontology/properties/{iri}
    AW->>H: handler future polled on the async worker
    H->>EX: execute_in_thread(move || handler.handle(GetOwlProperty))
    EX->>BT: tokio::task::spawn_blocking(f)
    BT->>QH: the sync CQRS handler runs here
    Note over BT,QH: INVARIANT: the CQRS query handlers are sync-over-async - they<br/>enter a nested block_on. Called inline on an actix worker they panic<br/>with "Cannot start a runtime from within a runtime" and drop the<br/>connection. ontology_handler.rs:374-377, add_owl_property at :462
    QH-->>BT: Result<Option<OwlProperty>, _>
    BT-->>EX: JoinHandle resolves
    EX-->>H: Ok(Ok(Some)) / Ok(Ok(None)) / Ok(Err) / Err
    Note over H: The four-arm match is uniform across every sibling in this file -<br/>join error is distinguished from handler error. ontology_handler.rs:387-404
    H-->>CL: 200 property, 404 not found, or 500
    Note over H,QH: get_owl_property and add_owl_property were the LAST two bare<br/>calls in this file, closed 2026-09-16 (ad8b83ad4). The same defect was<br/>fixed for one endpoint on 2026-08-10 and for add_axiom at<br/>ontology_handler.rs:593.
    Note over QH: Regression cover: owl_property_handlers_round_trip_off_the_async_worker<br/>ontology_handler.rs:1313, and the bare call pinned as the panic it is at<br/>ontology_handler.rs:1348. DEBT: the wrapping is load-bearing only while<br/>the CQRS handlers stay sync - the test module says to delete it if they<br/>become async-direct. ontology_handler.rs:1288
```

## VC-20.14 vault certify — the derivation certificate, the shared crate and the Whelk-parity invariant (ADR-2147, ontomark ADR-OM-009)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vault certify<br/>crates/vault/src/main.rs:562
    participant G as build_graph_attributed<br/>crates/vault/src/build/turtle.rs:216
    participant AD as certificate::certify (adapter)<br/>crates/vault/src/certificate.rs:45
    participant CR as rdfs-certificate crate (crates.io 0.1.0)<br/>../ontomark/crates/rdfs-certificate/src/lib.rs:432
    participant W as whelk::reason<br/>crates/vault/src/whelk.rs:505
    participant OO as oo-cert (Lean 4, external, never vendored)

    CLI->>G: the graph vault build publishes (knowledge vault, projected, attributed, public only)
    CLI->>AD: certify(&graph)
    AD->>AD: spell() — every Subject/Term as an N-Triples string crates/vault/src/certificate.rs:21
    AD->>CR: certify(triples) — the same engine ontomark build --certificate-out runs<br/>../ontomark/crates/ontomark-compile/src/certificate.rs:51, build/mod.rs:790
    loop until the worklist is empty
        CR->>CR: insert() indexes and queues a new fact, a derived one is ONE step lib.rs:249<br/>consequences_of() — rdfs2/3/5/7/9/11 in both premise positions lib.rs:287, run_to_fixpoint lib.rs:411
        Note over CR: rule TAB conclusion TAB premise1 TAB premise2 — premises are asserted or EARLIER lines<br/>pinned at lib.rs:884 (ordering) and lib.rs:915 (premise order per rule)
    end
    CR-->>CLI: Certificate (asserted, steps)
    CLI->>CR: write(out, generation) — asserted.tsv, derivations.tsv, asserted.sha256, scope.tsv lib.rs:585<br/>push_line refuses any field carrying a separator lib.rs:795
    CLI->>W: reason(&graph).inferred - Err before reasoning when RELEVANT_TRANSITIVE_CAP is exceeded, crates/vault/src/whelk.rs:514
    CLI->>CR: named_subclass_closure() lib.rs:515
    Note over CLI,W: INVARIANT whelk_parity: certified named closure == Whelk's, exit 1 otherwise<br/>crates/vault/src/main.rs:595, :631-635. Structural today: the emitter puts restrictions only on the<br/>superclass side, so every named entailment is rdfs11. Pinned at crates/vault/src/certificate.rs:92,<br/>ontomark enforces the same parity as a build refusal, ../ontomark/crates/ontomark-compile/src/build/mod.rs:101, :792-797
    CLI->>OO: oo-cert asserted.tsv derivations.tsv (reviewer / CI, OO_CERT_BIN)
    OO-->>CLI: {"ok":true,"theorem":"OOCert.certificate_sound"} or exit 1 naming the first bad line and rule
    Note over OO: Measured 2026-10-20 on visionGraph 1ec5774ad: 351,421 asserted, 731,141 derivations,<br/>certify 1.2 s, oo-cert 9.5 s accept, a forged line refused. tests/certificate_oo_cert.rs:72, :91
```

## VC-20.15 vault propose — the consequence delta on the PatchProposal (ADR-2147)

Current construction and consumer limits are assessed in [VC-41](41-proposal-construction-and-consumer-scope.md), including version metadata in consequence counts and grouped proposals that cannot reach application. This sequence is re-pinned to the stamped revision; the measurements in its last notes are historical.

```mermaid
sequenceDiagram
    autonumber
    participant P as vault propose<br/>crates/vault/src/main.rs:438
    participant B as build_with_certificates<br/>crates/vault/src/propose.rs:524
    participant A as Assessment::of (base, then patched)<br/>crates/vault/src/propose.rs:118
    participant D as consequence_delta<br/>../ontomark/crates/rdfs-certificate/src/lib.rs:635
    participant PP as PatchProposal.consequences<br/>crates/vault-core/src/proposal.rs:155
    participant F as forum 31402

    P->>B: build_with_certificates(vault, corpus, vocab, generation, options)
    B->>A: Assessment::of(base) — validation, conflicts, Whelk, certificate
    B->>A: Assessment::of(patched) — the same four, once
    B->>D: consequence_delta(&before.certificate, &after.certificate)
    Note over D: compares the CLOSURE (asserted ∪ derived) with blank-node-bearing triples counted, not compared<br/>lib.rs:635. Derived-only misreports a consequence that becomes asserted —<br/>cross-vendor audit finding, pinned at lib.rs:1077
    D-->>B: ConsequenceDelta — verdict word, new/lost, by predicate, examples, proposal asserted.sha256<br/>re-exported at crates/vault-core/src/proposal.rs:159
    B->>PP: with_consequences(delta) propose.rs:624, defn crates/vault-core/src/proposal.rs:259
    opt --certificate-dir
        B->>B: Built::write_certificates — base/, proposed/, consequences.json propose.rs:502
    end
    B-->>P: Built { proposal, base, proposed } propose.rs:486
    P->>F: the proposal JSON is the 31402 content, consequences included, the certificate files stay on disk, the digest travels
    Note over PP,F: The delta is advisory and never blocks. Newly unsatisfiable classes and reasoner-cap refusal block<br/>via propose.rs:199,205. Historical ADR-2147 measurements: one rdfs:domain on<br/>vc:requires = 206,476 new consequences, AI Operator is-a Robotics = 6, corpus dry-run 11.8 s
    Note over P: SHACL side-finding of the same spike: ontology-class.shacl.ttl:21 now sh:nodeKind sh:Literal,<br/>the evaluator types @language as rdf:langString shacl.rs:749 and enforces sh:Literal shacl.rs:305, :767
```

## VC-20.16 Ontology reads answer a hidden class as one never stored (per-caller deny-list)

<!-- 2026-11-06 restamp 809a859aa: new. 3d1539ef7 adds services/ontology_visibility.rs and threads HiddenClasses through every /api/ontology read, /query, /sparql and the six /api/ontology-agent read routes; SECURITY-profiles 0.1.6 records it -->
```mermaid
flowchart TB
    subgraph ROUTES["Read surfaces that take the caller's HiddenClasses"]
        OR["/api/ontology GETs: graph, classes, class, axioms, properties, inference, validate, metrics, inferred<br/>src/handlers/ontology_handler.rs:88, :168, :216, :529, :822, :1150, :1226"]
        OM["/api/ontology report, inferences, hierarchy, reports id, state-at, provenance<br/>src/handlers/api_handler/ontology/mod.rs:624, :881, :1088, :1277, :1673, :1741"]
        SQ["POST /query and /sparql, both power-user<br/>src/handlers/ontology_handler.rs:1025, :1106"]
        AG["/api/ontology-agent discover, read, query, traverse, validate, check<br/>src/handlers/ontology_agent_handler.rs:135, :170, :200, :240, :329, :387"]
    end
    ROUTES --> CH["graph_access.caller_hidden_classes<br/>src/handlers/graph_access.rs:200-208"]
    CH --> HC["hidden_class_iris: the owl_class_iri of every node outside the caller's view<br/>src/handlers/graph_access.rs:174-181"]
    CH -->|"graph unreadable"| F503["503, never the unfiltered store<br/>src/handlers/graph_access.rs:207"]
    HC --> EMPTY{"HiddenClasses.is_empty: filter off or full access<br/>src/services/ontology_visibility.rs:35"}
    EMPTY -->|"yes"| PASS["pass-through, untouched"]
    EMPTY -->|"no"| SCRUB["drop the class and every edge, axiom, binding and triple naming it<br/>src/services/ontology_visibility.rs:1-8"]
    SCRUB --> S1["class, classes, axioms<br/>src/services/ontology_visibility.rs:60, :70, :126"]
    SCRUB --> S2["sparql_json drops rows with a hidden binding value, query_rows the same<br/>src/services/ontology_visibility.rs:196-214, :218-223"]
    SCRUB --> S3["validation_report keeps whole-graph totals as computed<br/>src/services/ontology_visibility.rs:266-269"]
    S2 -.-> R1["OPEN: ASK and aggregate replies carry no bindings to drop"]
    HC -.-> R2["OPEN: a node without owl_class_iri never enters the deny-list,<br/>graph_access.rs:179"]
    CC["GET /api/ontology/class-count, no caller, whole-store count<br/>src/handlers/ontology_class_count_handler.rs:40-51"] -.-> R3["TENSION: /metrics counts excluding hidden classes,<br/>ontology_handler.rs:1240-1241"]
```

**Invariant:** a hidden class reads exactly as a class never stored: every listed route resolves the caller's deny-list before touching the store and fails closed with 503 when the graph cannot be read (`src/handlers/graph_access.rs:200-208`), and an empty set is a pass-through (`src/services/ontology_visibility.rs:35`).

**Tension (risk):** `GET /api/ontology/class-count` takes no caller and returns the whole-store `class_count` (`src/handlers/ontology_class_count_handler.rs:40-51`), while `GET /api/ontology/metrics` counts over the store minus the caller's hidden classes (`src/handlers/ontology_handler.rs:1240-1241`). The difference between the two is the number of classes hidden from that caller.

**Open (risk):** the residuals SECURITY-profiles 0.1.6 lists are unfiltered: SPARQL `ASK`, aggregates, queries that do not project the class and literal forms; inferences computed over the whole store on routes outside per-view reasoning; store-wide totals in validation reports (`src/services/ontology_visibility.rs:266-268`); and ontology nodes with no `owl_class_iri`, which `hidden_class_iris` skips (`src/handlers/graph_access.rs:179`). All are recorded at `../project/docs/SECURITY-profiles.md:111`. The 0.1.8 per-view reasoning has since closed the stored-run reads and the `/inferred` run id (`../project/docs/SECURITY-profiles.md:112`).
