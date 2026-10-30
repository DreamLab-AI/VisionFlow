---
id: VC-20
title: Ontology pipeline - OWL extraction, Oxigraph, Whelk reasoning, governed mutation
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2071, ADR-2064, ADR-2066, ADR-2068, ADR-2114, ADR-2116, ADR-2125, ADR-2147, ADR-OM-009]
sources:
  - ../project/src/bin/load_ontology.rs
  - ../project/src/services/owl_extractor_service.rs
  - ../project/src/services/ontology_pipeline_service.rs
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
  - ../project/tests/integration/pipeline_test.rs
verified_commit: {visionclaw: e4ba5eb35ee6715cfd7f2daf1cd9af05e8a18cce, ontomark: 13af42340448022d4d43ac349a8bb5f3ae6d1d47}
---

## VC-20.1 load_ontology bin - CorpusSource-backed corpus loader
```mermaid
sequenceDiagram
    autonumber
    participant Bin as main<br/>src/bin/load_ontology.rs:46
    participant Src as CorpusSource LocalDirectorySource<br/>src/services/corpus_source/local.rs:200
    participant Parser as parse_page project_ontology<br/>src/services/page_parser.rs:73
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:1664
    participant Extr as OwlExtractorService<br/>src/services/owl_extractor_service.rs:52

    Bin->>Src: LocalDirectorySource::new(cli_root, base_paths) or from_env() (lines 55-56, defn local.rs:60,74)
    Bin->>Src: list_pages() (line 70)
    Src-->>Bin: Vec of CorpusPage
    loop each page
        Bin->>Bin: source.fetch_page(page) (line 85)
        Bin->>Parser: parse_page(content, path) (line 93, defn 73)
        Bin->>Bin: visionclaw_domain::vault::parse(content).is_kg_included() gate (line 96, see VC-21)
    end
    Bin->>Parser: project_ontology(ontology_pages, vocabulary) (line 108, defn 170)
    Parser-->>Bin: OntologyProjection classes axioms
    Bin->>Repo: persist -> save_ontology(classes, [], axioms) (line 115, persist defn 192, save_ontology defn 1664)
    Repo->>Repo: single atomic INSERT DATA into GRAPH_ONTOLOGY, no per-item ASK dedup (lines 1675-1686)
    Bin->>Extr: OwlExtractorService::new(repo).extract_all_owl() (lines 130-131)
    Bin->>Repo: get_classes() (line 135, defn 2360)
    Repo-->>Bin: Vec of OwlClass
    Bin->>Bin: exit(1) when ontology_files_parsed==0 or classes_persisted==0 (line 186)
    Note over Bin,Src: RESOLVED ADR-2114 - the bin now reads through the same CorpusSource port as live sync<br/>(LocalDirectorySource over VAULT_ROOT, ADR-2064), instead of a bespoke root-resolution scheme.<br/>The five hardcoded sample OwlClass literals and the per-class add_owl_class loop are gone.
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
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_ASSERT : "add_owl_class add_axiom lines 1693 2456"
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_INFERRED : "store_inference_results DELETE-WHERE+INSERT DATA line 2507 ADR-2004 D9"
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
    OxigraphOntologyRepository ||--o{ GRAPH_SHAPES : "load_shacl_shapes line 410"
    OxigraphOntologyRepository ||--o{ GRAPH_PROVENANCE : "emit_provenance line 651 see VC-22"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_SSSP : "own sub-domain, atomic CLEAR GRAPH, line 73"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_APSP : "line 74"
    SparqlMigrations ||--o{ GRAPH_MIGRATIONS : "sparql_migrations.rs:44, excluded from graph-export/GPU paths"
```

## VC-20.3 Whelk EL++ reasoning cycle (github_sync post-sync)
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1205
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:2360
    participant Engine as WhelkInferenceEngine<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:360
    participant Whelk as whelk-rs reasoner::assert<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:440

    Sync->>Repo: get_classes() (line 1210, defn 2360)
    Sync->>Repo: get_axioms() (line 1215)
    Sync->>Sync: axioms.extend(ngm_property_hierarchy_axioms()) (line 1224)
    Sync->>Engine: load_ontology(classes, axioms) (line 1240, defn line 362)
    Engine->>Engine: compute_ontology_checksum(ontology) (line 382)
    alt checksum unchanged since last load
        Engine-->>Sync: reuse cached_subsumptions (line 395-396)
    else checksum changed
        Engine-->>Sync: cached_subsumptions cleared, fresh reasoning required (line 391-393)
    end
    Sync->>Engine: infer() (line 1245, defn line 412)
    alt cached_subsumptions present (line 421)
        Engine-->>Sync: cached InferenceResults, whelk-rs not invoked (line 422-434)
    else no cache
        Engine->>Whelk: translate_ontology(ontology) (line 437)
        Engine->>Whelk: reasoner::assert(whelk_axioms) (line 440)
        Whelk-->>Engine: named_subsumptions() (line 442)
        Engine-->>Sync: InferenceResults inferred_axioms inference_time_ms (line 461-466)
    end
    Note over Engine: DOC-DRIFT - module doc claims complete EL reasoning (whelk_inference_engine.rs:5) but infer() lines 412-467 has no tokio timeout or axiom-count cap
    Sync->>Repo: store_inference_results(results) (line 1255, defn line 2507)
    Repo->>Repo: DELETE-WHERE GRAPH_ONTOLOGY_INFERRED then atomic INSERT DATA, one SPARQL Update (ADR-2004 D9, ADR-099 D3, lines 2601-2606)
    Note over Sync: RESOLVED a6de5a421 - the former DIVERGENCE (OntologyReasoningService::infer_axioms ran CustomReasoner, not Whelk)<br/>is gone: that service, its test helper and both of its test files were deleted, the type having been constructed only by<br/>its own test helper. Production reasoning in this cycle is WhelkInferenceEngine only - the engine GitHubSyncService<br/>builds at github_sync_service.rs:226 and drives at github_sync_service.rs:1240 and :1245
```

**Drift (resolved a6de5a421):** this diagram recorded a second reasoner, `OntologyReasoningService`, that ran `CustomReasoner` beside Whelk and was never constructed outside its own tests; `a6de5a421` deleted the service, its test helper, `tests/ontology_reasoning_integration_test.rs` and `tests/unit/ontology_reasoning_test.rs`, so the only reasoner driven from the sync cycle is the `WhelkInferenceEngine` that `GitHubSyncService` owns (`src/services/github_sync_service.rs:226`, `src/services/github_sync_service.rs:1240`, `src/services/github_sync_service.rs:1245`).

**Debt:** `CustomReasoner` survives in a second place, `OntologyPipelineService::trigger_reasoning` (`src/services/ontology_pipeline_service.rs:244`), reached only through `on_ontology_modified` (`src/services/ontology_pipeline_service.rs:157`), and a search of `src/` and `crates/` at the stamped commit finds no caller of that method outside `tests/integration/pipeline_test.rs:94`; PATH B in VC-20.4 therefore hangs off an entry point the running server never invokes.

**Drift (doc comment vs callers):** the doc comment on `on_ontology_modified` says it is called automatically by `GitHubSyncService` (`src/services/ontology_pipeline_service.rs:150`), while the sync cycle in this diagram calls the Whelk engine and `select_inferred_edges_for_sync` directly (`src/services/github_sync_service.rs:1294`).

## VC-20.4 inferred-edge materialisation and class-summary index
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1205
    participant Sel as GitHubSyncService::select_inferred_edges_for_sync<br/>src/services/github_sync_service.rs:1126
    participant Mat as inferred_edge_materialiser<br/>src/services/inferred_edge_materialiser.rs:127
    participant KG as KnowledgeGraphRepository::batch_add_edges<br/>src/services/github_sync_service.rs:1309
    participant Pipeline as OntologyPipelineService::materialise_inferred_edges_from_axioms<br/>src/services/ontology_pipeline_service.rs:476
    participant Idx as ontology_class_index::maybe_refresh_after_sync<br/>src/services/ontology_class_index.rs:468

    rect rgb(235,235,255)
    Note over Sync,KG: PATH A - pure selection function, called unconditionally<br/>after Whelk infer
    Sync->>Sync: load_graph then mat::asserted_pairs(g.edges) both<br/>directions (lines 1267 1287, asserted_pairs defn line 79)
    Sync->>Sel: select_inferred_edges_for_sync(inferred_axioms,<br/>resolver.resolve, asserted) (line 1294, defn line 1126)
    Sel->>Mat: is_materialisable_subclass_pair drops self, owl#Nothing<br/>child, owl#Thing parent (line 1136, defn line 169)
    Sel->>Mat: immediate_parents_from_subclass_pairs reduces transitive<br/>ancestors to immediate parents (line 1145, defn line 181)
    loop each immediate (child_iri, parent_iri) pair
        Sel->>Sel: resolve(child_iri) resolve(parent_iri),<br/>unresolved_endpoints += 1 per miss (lines 1154-1168)
    end
    Sel->>Mat: select_inferred_edges drops self-loops, asserted pairs,<br/>caps per child at 8 (line 1174, defn line 127)
    Sel->>Mat: build_inferred_edge tags edge_type hierarchical,<br/>owl-property rdfs:subClassOf, metadata inferred=true (line<br/>1182, defn line 68)
    Sel-->>Sync: InferredEdgeSelection edges considered_axioms<br/>immediate_pairs unresolved_endpoints
    Sync->>KG: batch_add_edges(inferred_edges) (line 1309)
    Note over Sync,Sel: RESOLVED ADR-2071 - select_inferred_edges_for_sync is now<br/>the SHARED pure function (docstring<br/>line 1113-1125): the per-child cap of 8, asserted-pair<br/>suppression and the transitive-to-immediate<br/>reduction apply identically to PATH B, so the two paths<br/>cannot drift. Every edge carries<br/>metadata inferred=true, so edge_is_inferred classifies<br/>sync-produced edges onto the inferred channel.
    end
    rect rgb(255,245,225)
    Note over Pipeline,Mat: PATH B - OntologyPipelineService, gated OFF by default
    opt config.materialise_inferred_edges is true (default false, ontology_pipeline_service.rs:63)
        Pipeline->>Mat: immediate_inferred_parents(child_to_ancestors) (line 506,<br/>defn line 98)
        Pipeline->>Pipeline: resolve_nodes via get_nodes_by_owl_class_iri (lines 512-525)
        Pipeline->>Pipeline: graph_repo.load_graph() current asserted edges (line 531)
        Pipeline->>Mat: materialise(candidates, current.edges, cfg) (line 538,<br/>defn line 202)
        Mat->>Mat: select_inferred_edges drops self-loops, asserted pairs,<br/>caps per child at max_inferred_parents_per_child=8 (lines<br/>127-153, DEFAULT_MAX_INFERRED_PARENTS_PER_CHILD line 37)
        Pipeline->>Pipeline: graph_repo.batch_add_edges(edges) (line 544)
    end
    Note over Pipeline: SemanticPhysicsConfig (ontology_pipeline_service.rs:27)<br/>defaults: constraint_strength 1.0,<br/>use_gpu_constraints true, materialise_inferred_edges false<br/>(lines 61-63). RESOLVED ADR-2068 - the repo-root<br/>ontology_physics.toml was read by no Rust code and has<br/>been deleted — the live toggle is the<br/>Settings.ontology_physics bool<br/>(src/settings/models.rs:58), gating POST<br/>/api/ontology-physics/enable|disable<br/>(api_handler/ontology_physics/mod.rs:107,387) which drives<br/>OntologyConstraintActor - see VC-04.7, VC-11.5
    end
    Sync->>Idx: maybe_refresh_after_sync(classes)<br/>(github_sync_service.rs:663, sync_graphs_with, defn line<br/>468)
    alt ONTOLOGY_CLASS_INDEX_ENABLED unset (default, ontology_class_index.rs:75)
        Idx-->>Sync: skip refresh (line 474)
    else enabled
        Idx->>Idx: refresh_class_index condense_class RuVector memory_store<br/>namespace ontology-classes (line 495, defn 378,<br/>DEFAULT_NAMESPACE line 33)
    end
```

## VC-20.6 SPARQL read path - /ontology/query
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Route as POST /api/ontology/query<br/>src/handlers/api_handler/ontology/mod.rs:1656
    participant Handler as query_ontology<br/>src/handlers/ontology_handler.rs:954
    participant Validator as validate_read_only_sparql<br/>src/handlers/ontology_handler.rs:820
    participant Clamp as clamp_sparql_limit<br/>src/handlers/ontology_handler.rs:901
    participant CQRS as QueryOntologyHandler
    participant Cap as cap_result_rows<br/>src/handlers/ontology_handler.rs:931

    Client->>Route: POST /ontology/query query
    Note over Route: gated RequireAuth::power_user().mutations_only() (api_handler/ontology/mod.rs:1645)
    Route->>Handler: query_ontology(auth, state, request) (ontology_handler.rs:954)
    Handler->>Validator: validate_read_only_sparql(query) (ontology_handler.rs:962, defn :820)
    alt forbidden keyword INSERT DELETE DROP CLEAR LOAD CREATE ADD MOVE COPY WITH SERVICE
        Validator-->>Handler: Err operation not permitted (ontology_handler.rs:846, :852)
        Handler-->>Client: 400 bad_request
    else no read form present
        Validator-->>Handler: Err only SELECT ASK CONSTRUCT DESCRIBE permitted (ontology_handler.rs:865, :867)
        Handler-->>Client: 400 bad_request
    else valid read query
        Validator-->>Handler: Ok
        Handler->>Clamp: clamp_sparql_limit(query) (ontology_handler.rs:973, defn :901)
        Clamp-->>Handler: LIMIT injected or clamped to MAX_SPARQL_ROWS 10000 (ontology_handler.rs:889, :919)
        Handler->>CQRS: QueryOntologyHandler.handle(QueryOntology query) off the worker (ontology_handler.rs:980, see VC-20.13)
        CQRS-->>Handler: Vec of HashMap string string
        Handler->>Cap: cap_result_rows(results) (ontology_handler.rs:986, defn :931)
        Cap-->>Handler: rows capped at 10000, byte fence MAX_SPARQL_RESULT_BYTES 8388608, truncated flag (ontology_handler.rs:891, :935, :944)
        Handler-->>Client: 200 results rowCount truncated
    end
    Note over Handler: ADR-2004 - handler-level fence mirrors adapter fence sparql_select_json (crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:885)
```

## VC-20.7 ontology HTTP route families
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Rbac as RbacGate on /api scope<br/>project/src/main.rs:1137
    participant OntScope as /api/ontology scope<br/>src/handlers/api_handler/ontology/mod.rs:1642
    participant DerivedScope as /api/ontology/derived scope<br/>src/handlers/ontology_derived_handler.rs:185
    participant CountScope as /api/ontology/class-count scope<br/>src/handlers/ontology_class_count_handler.rs:76
    participant AgentScope as /api/ontology-agent scope<br/>src/handlers/ontology_agent_handler.rs:386

    Client->>Rbac: any /api/* request
    rect rgb(230,255,230)
    Note over OntScope: registration order matters (ontology/mod.rs:1619-1630) - derived and class-count scopes register before this broader /ontology scope
    OntScope->>OntScope: POST graph classes properties axioms inference query sparql load load-axioms validate mapping apply, PUT classes iri and properties iri, DELETE cache axioms iri classes iri (lines 1647-1667)
    OntScope->>OntScope: GET graph inferred classes properties axioms inference validate metrics inferences hierarchy state-at provenance reports id report health ws (lines 1669-1694)
    Note over OntScope: power_user().mutations_only() gate line 1645 - POST PUT DELETE gated, all GET public
    end
    rect rgb(255,230,230)
    DerivedScope->>DerivedScope: POST derived, POST derived/regenerate, GET derived which (line 189-191)
    Note over DerivedScope: RequireAuth::power_user() whole scope (line 188), fence detail see VC-22
    end
    rect rgb(230,230,255)
    CountScope->>CountScope: GET class-count (line 77)
    Note over CountScope: unauthenticated by design (ontology_class_count_handler.rs:25)
    end
    rect rgb(255,245,230)
    AgentScope->>AgentScope: POST discover read query traverse validate check, GET status (lines 389-395)
    AgentScope->>AgentScope: POST ontology-agent/propose, always 410 Gone (line 399, handler defn line 239)
    Note over AgentScope: RESOLVED ADR-2116 - every route on this scope is now anonymous. The /propose sub-scope and its<br/>RequireAuth+RateLimit middleware are gone with the write path they guarded (ontology_agent_handler.rs:239-240, :382-386) —<br/>propose() answers 410 Gone naming vault propose as the replacement, needing no auth to reach no service. The governed<br/>write door moved to POST /decisions/record - see VC-24. ADR-2127 added /check with the tri-valued<br/>relation check and open-world scope fields on every answer.
    end
    rect rgb(245,245,245)
    InfScope->>InfScope: POST run batch validate, GET results ontology_id classify ontology_id, DELETE cache ontology_id (formerly inference_handler.rs, deleted)
    Note over OntScope: RESOLVED ADR-2066 - the /api/inference scope is gone. Its seven handlers extracted a web::Data InferenceService<br/>that was never constructed or registered, so every route 500'd at the extractor. inference_handler.rs,<br/>application/inference_service.rs, events/inference_triggers.rs and the main.rs registration were all removed.<br/>The live reasoning path is GitHubSyncService::run_post_sync_reasoning - see VC-20.3
    end
```

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
    participant H as get_owl_property<br/>src/handlers/ontology_handler.rs:427
    participant EX as execute_in_thread<br/>src/handlers/utils.rs:1
    participant BT as tokio blocking thread pool<br/>src/handlers/utils.rs:6
    participant QH as GetOwlPropertyHandler.handle<br/>src/handlers/ontology_handler.rs:442

    CL->>AW: GET /api/ontology/properties/{iri}
    AW->>H: handler future polled on the async worker
    H->>EX: execute_in_thread(move || handler.handle(GetOwlProperty))
    EX->>BT: tokio::task::spawn_blocking(f)
    BT->>QH: the sync CQRS handler runs here
    Note over BT,QH: INVARIANT: the CQRS query handlers are sync-over-async - they<br/>enter a nested block_on. Called inline on an actix worker they panic<br/>with "Cannot start a runtime from within a runtime" and drop the<br/>connection. ontology_handler.rs:436-439, add_owl_property at :512
    QH-->>BT: Result<Option<OwlProperty>, _>
    BT-->>EX: JoinHandle resolves
    EX-->>H: Ok(Ok(Some)) / Ok(Ok(None)) / Ok(Err) / Err
    Note over H: The four-arm match is uniform across every sibling in this file -<br/>join error is distinguished from handler error. ontology_handler.rs:448-465
    H-->>CL: 200 property, 404 not found, or 500
    Note over H,QH: get_owl_property and add_owl_property were the LAST two bare<br/>calls in this file, closed 2026-09-16 (ad8b83ad4). The same defect was<br/>fixed for one endpoint on 2026-08-10 and for add_axiom at<br/>ontology_handler.rs:632.
    Note over QH: Regression cover: owl_property_handlers_round_trip_off_the_async_worker<br/>ontology_handler.rs:1132, and the bare call pinned as the panic it is at<br/>ontology_handler.rs:1167. DEBT: the wrapping is load-bearing only while<br/>the CQRS handlers stay sync - the test module says to delete it if they<br/>become async-direct. ontology_handler.rs:1107
```

## VC-20.14 vault certify — the derivation certificate, the shared crate and the Whelk-parity invariant (ADR-2147, ontomark ADR-OM-009)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as vault certify<br/>crates/vault/src/main.rs:515
    participant G as build_graph_attributed<br/>crates/vault/src/build/turtle.rs:216
    participant AD as certificate::certify (adapter)<br/>crates/vault/src/certificate.rs:45
    participant CR as rdfs-certificate crate (crates.io 0.1.0)<br/>../ontomark/crates/rdfs-certificate/src/lib.rs:432
    participant W as whelk::reason<br/>crates/vault/src/whelk.rs:505
    participant OO as oo-cert (Lean 4, external, never vendored)

    CLI->>G: the graph vault build publishes (knowledge vault, projected, attributed, public only)
    CLI->>AD: certify(&graph)
    AD->>AD: spell() — every Subject/Term as an N-Triples string crates/vault/src/certificate.rs:21
    AD->>CR: certify(triples) — the same engine ontomark build --certificate-out runs<br/>../ontomark/crates/ontomark-compile/src/certificate.rs:51, build/mod.rs:695
    loop until the worklist is empty
        CR->>CR: insert() indexes and queues a new fact, a derived one is ONE step lib.rs:249<br/>consequences_of() — rdfs2/3/5/7/9/11 in both premise positions lib.rs:287, run_to_fixpoint lib.rs:411
        Note over CR: rule TAB conclusion TAB premise1 TAB premise2 — premises are asserted or EARLIER lines<br/>pinned at lib.rs:879 (ordering) and lib.rs:910 (premise order per rule)
    end
    CR-->>CLI: Certificate (asserted, steps)
    CLI->>CR: write(out, generation) — asserted.tsv, derivations.tsv, asserted.sha256, scope.tsv lib.rs:585<br/>push_line refuses any field carrying a separator lib.rs:790
    CLI->>W: reason(&graph).inferred
    CLI->>CR: named_subclass_closure() lib.rs:515
    Note over CLI,W: INVARIANT whelk_parity: certified named closure == Whelk's, exit 1 otherwise<br/>crates/vault/src/main.rs:548. Structural today: the emitter puts restrictions only on the<br/>superclass side, so every named entailment is rdfs11. Pinned at crates/vault/src/certificate.rs:92,<br/>ontomark enforces the same parity as a build refusal, build/mod.rs:101
    CLI->>OO: oo-cert asserted.tsv derivations.tsv (reviewer / CI, OO_CERT_BIN)
    OO-->>CLI: {"ok":true,"theorem":"OOCert.certificate_sound"} or exit 1 naming the first bad line and rule
    Note over OO: Measured 2026-10-20 on visionGraph 1ec5774ad: 351,421 asserted, 731,141 derivations,<br/>certify 1.2 s, oo-cert 9.5 s accept, a forged line refused. tests/certificate_oo_cert.rs:72, :91
```

## VC-20.15 vault propose — the consequence delta on the PatchProposal (ADR-2147)

```mermaid
sequenceDiagram
    autonumber
    participant P as vault propose<br/>crates/vault/src/main.rs:250
    participant B as build_with_certificates<br/>crates/vault/src/propose.rs:524
    participant A as Assessment::of (base, then patched)<br/>crates/vault/src/propose.rs:118
    participant D as consequence_delta<br/>../ontomark/crates/rdfs-certificate/src/lib.rs:630
    participant PP as PatchProposal.consequences<br/>crates/vault-core/src/proposal.rs:155
    participant F as forum 31402

    P->>B: build_with_certificates(vault, corpus, vocab, generation, options)
    B->>A: Assessment::of(base) — validation, conflicts, Whelk, certificate
    B->>A: Assessment::of(patched) — the same four, once
    B->>D: consequence_delta(&before.certificate, &after.certificate)
    Note over D: compares the CLOSURE (asserted ∪ derived) with blank-node-bearing triples counted, not compared<br/>lib.rs:630. Derived-only misreports a consequence that becomes asserted —<br/>cross-vendor audit finding, pinned at lib.rs:1055
    D-->>B: ConsequenceDelta — verdict word, new/lost, by predicate, examples, proposal asserted.sha256<br/>crates/vault-core/src/proposal.rs:176
    B->>PP: with_consequences(delta) propose.rs:611
    opt --certificate-dir
        B->>B: Built::write_certificates — base/, proposed/, consequences.json propose.rs:502
    end
    B-->>P: Built { proposal, base, proposed } propose.rs:486
    P->>F: the proposal JSON is the 31402 content, consequences included, the certificate files stay on disk, the digest travels
    Note over PP,F: INVARIANT (VAULT-corpus-format 12): the delta is shown to the signer and never blocks,<br/>WHELK_INCONSISTENT stays the only reasoning blocker (ADR-2125). Measured: one rdfs:domain on<br/>vc:requires = 206,476 new consequences, AI Operator is-a Robotics = 6, a dry run on the live corpus 11.8 s
    Note over P: SHACL side-finding of the same spike: ontology-class.shacl.ttl:21 now sh:nodeKind sh:Literal,<br/>the evaluator types @language as rdf:langString shacl.rs:749 and enforces sh:Literal shacl.rs:305, :767
```
