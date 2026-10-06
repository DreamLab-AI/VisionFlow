---
id: VC-20
title: Ontology pipeline - OWL extraction, Oxigraph, Whelk reasoning, governed mutation
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2071, ADR-2064, ADR-2066, ADR-2068, ADR-2114, ADR-2116]
sources:
  - ../project/src/bin/load_ontology.rs
  - ../project/src/services/owl_extractor_service.rs
  - ../project/src/services/ontology_pipeline_service.rs
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/page_parser.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs
  - ../project/crates/visionclaw-adapters/src/whelk_inference_engine.rs
  - ../project/src/services/ontology_reasoning_service.rs
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
  - ../project/src/main.rs
  - ../project/src/handlers/utils.rs
  - ../project/crates/visionclaw-adapters/src/sparql_migrations.rs
  - ../project/src/actors/gpu/gpu_manager_actor.rs
  - ../project/src/actors/gpu/ontology_constraint_actor.rs
  - ../project/src/actors/gpu/physics_supervisor.rs
  - ../project/src/settings/models.rs
verified_commit: {visionclaw: af3dff3f25300cf12bceda5650688ec223270eca}
---

## VC-20.1 load_ontology bin - CorpusSource-backed corpus loader
```mermaid
sequenceDiagram
    autonumber
    participant Bin as main<br/>src/bin/load_ontology.rs:46
    participant Src as CorpusSource LocalDirectorySource<br/>src/services/corpus_source/local.rs:194
    participant Parser as parse_page project_ontology<br/>src/services/page_parser.rs:66
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:1667
    participant Extr as OwlExtractorService<br/>src/services/owl_extractor_service.rs:52

    Bin->>Src: LocalDirectorySource::new(cli_root, base_paths) or from_env() (lines 55-56, defn local.rs:54,68)
    Bin->>Src: list_pages() (line 70)
    Src-->>Bin: Vec of CorpusPage
    loop each page
        Bin->>Bin: source.fetch_page(page) (line 87)
        Bin->>Parser: parse_page(content, path) (line 93, defn 66)
        Bin->>Bin: visionclaw_domain::vault::parse(content).is_kg_included() gate (line 96, see VC-21)
    end
    Bin->>Parser: project_ontology(ontology_pages, vocabulary) (line 108, defn 163)
    Parser-->>Bin: OntologyProjection classes axioms
    Bin->>Repo: persist -> save_ontology(classes, [], axioms) (line 115, persist defn 192, save_ontology defn 1667)
    Repo->>Repo: single atomic INSERT DATA into GRAPH_ONTOLOGY, no per-item ASK dedup (lines 1678-1689)
    Bin->>Extr: OwlExtractorService::new(repo).extract_all_owl() (lines 130-131)
    Bin->>Repo: get_classes() (line 135, defn 2361)
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
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_ASSERT : "add_owl_class add_axiom lines 1696 2457"
    OxigraphOntologyRepository ||--o{ GRAPH_ONTOLOGY_INFERRED : "store_inference_results CLEAR+INSERT line 2508 ADR-2004 D9"
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
    OxigraphOntologyRepository ||--o{ GRAPH_PROVENANCE : "emit_provenance line 648 see VC-22"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_SSSP : "own sub-domain, atomic CLEAR GRAPH, line 73"
    OxigraphOntologyRepository ||--o{ GRAPH_CACHE_APSP : "line 74"
    SparqlMigrations ||--o{ GRAPH_MIGRATIONS : "sparql_migrations.rs:44, excluded from graph-export/GPU paths"
```

## VC-20.3 Whelk EL++ reasoning cycle (github_sync post-sync)
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1184
    participant Repo as OxigraphOntologyRepository<br/>crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:2361
    participant Engine as WhelkInferenceEngine<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:360
    participant Whelk as whelk-rs reasoner::assert<br/>crates/visionclaw-adapters/src/whelk_inference_engine.rs:440

    Sync->>Repo: get_classes() (line 1189, defn 2361)
    Sync->>Repo: get_axioms() (line 1194)
    Sync->>Sync: axioms.extend(ngm_property_hierarchy_axioms()) (line 1203)
    Sync->>Engine: load_ontology(classes, axioms) (line 1219, defn line 362)
    Engine->>Engine: compute_ontology_checksum(ontology) (line 382)
    alt checksum unchanged since last load
        Engine-->>Sync: reuse cached_subsumptions (line 395-396)
    else checksum changed
        Engine-->>Sync: cached_subsumptions cleared, fresh reasoning required (line 391-393)
    end
    Sync->>Engine: infer() (line 1224, defn line 412)
    alt cached_subsumptions present (line 421)
        Engine-->>Sync: cached InferenceResults, whelk-rs not invoked (line 422-434)
    else no cache
        Engine->>Whelk: translate_ontology(ontology) (line 437)
        Engine->>Whelk: reasoner::assert(whelk_axioms) (line 440)
        Whelk-->>Engine: named_subsumptions() (line 442)
        Engine-->>Sync: InferenceResults inferred_axioms inference_time_ms (line 461-466)
    end
    Note over Engine: DOC-DRIFT - module doc calls this bounded EL reasoning but infer() lines 412-467 has no tokio timeout or axiom-count cap
    Sync->>Repo: store_inference_results(results) (line 1234, defn line 2508)
    Repo->>Repo: DELETE-WHERE GRAPH_ONTOLOGY_INFERRED then atomic INSERT DATA, one SPARQL Update (ADR-2004 D9, ADR-099 D3, lines 2599-2605)
    Note over Sync: DIVERGENCE - OntologyReasoningService::infer_axioms (ontology_reasoning_service.rs:107) runs CustomReasoner<br/>not this Whelk engine, its WhelkInferenceEngine field is legacy (line 77) and OntologyReasoningService::new is<br/>never called outside tests
```

## VC-20.4 inferred-edge materialisation and class-summary index
```mermaid
sequenceDiagram
    autonumber
    participant Sync as GitHubSyncService::run_post_sync_reasoning<br/>src/services/github_sync_service.rs:1184
    participant Sel as GitHubSyncService::select_inferred_edges_for_sync<br/>src/services/github_sync_service.rs:1105
    participant Mat as inferred_edge_materialiser<br/>src/services/inferred_edge_materialiser.rs:127
    participant KG as KnowledgeGraphRepository::batch_add_edges<br/>src/services/github_sync_service.rs:1288
    participant Pipeline as OntologyPipelineService::materialise_inferred_edges_from_axioms<br/>src/services/ontology_pipeline_service.rs:473
    participant Idx as ontology_class_index::maybe_refresh_after_sync<br/>src/services/ontology_class_index.rs:468

    rect rgb(235,235,255)
    Note over Sync,KG: PATH A - pure selection function, called unconditionally after Whelk infer
    Sync->>Sync: load_graph then mat::asserted_pairs(g.edges) both directions (lines 1246 1266, asserted_pairs defn line 79)
    Sync->>Sel: select_inferred_edges_for_sync(inferred_axioms, resolver.resolve, asserted) (line 1273, defn line 1105)
    Sel->>Mat: is_materialisable_subclass_pair drops self, owl#Nothing child, owl#Thing parent (line 1115, defn line 169)
    Sel->>Mat: immediate_parents_from_subclass_pairs reduces transitive ancestors to immediate parents (line 1124, defn line 181)
    loop each immediate (child_iri, parent_iri) pair
        Sel->>Sel: resolve(child_iri) resolve(parent_iri), unresolved_endpoints += 1 per miss (lines 1133-1147)
    end
    Sel->>Mat: select_inferred_edges drops self-loops, asserted pairs, caps per child at 8 (line 1153, defn line 127)
    Sel->>Mat: build_inferred_edge tags edge_type hierarchical, owl-property rdfs:subClassOf, metadata inferred=true (line 1161, defn line 68)
    Sel-->>Sync: InferredEdgeSelection edges considered_axioms immediate_pairs unresolved_endpoints
    Sync->>KG: batch_add_edges(inferred_edges) (line 1288)
    Note over Sync,Sel: RESOLVED ADR-2071 - select_inferred_edges_for_sync is now the SHARED pure function (docstring<br/>line 1092-1104): the per-child cap of 8, asserted-pair suppression and the transitive-to-immediate<br/>reduction apply identically to PATH B, so the two paths cannot drift. Every edge carries<br/>metadata inferred=true, so edge_is_inferred classifies sync-produced edges onto the inferred channel.
    end
    rect rgb(255,245,225)
    Note over Pipeline,Mat: PATH B - OntologyPipelineService, gated OFF by default
    opt config.materialise_inferred_edges is true (default false, ontology_pipeline_service.rs:63)
        Pipeline->>Mat: immediate_inferred_parents(child_to_ancestors) (line 503, defn line 98)
        Pipeline->>Pipeline: resolve_nodes via get_nodes_by_owl_class_iri (lines 511-521)
        Pipeline->>Pipeline: graph_repo.load_graph() current asserted edges (line 528)
        Pipeline->>Mat: materialise(candidates, current.edges, cfg) (line 535, defn line 202)
        Mat->>Mat: select_inferred_edges drops self-loops, asserted pairs, caps per child at max_inferred_parents_per_child=8 (lines 127-153, DEFAULT_MAX_INFERRED_PARENTS_PER_CHILD line 37)
        Pipeline->>Pipeline: graph_repo.batch_add_edges(edges) (line 541)
    end
    Note over Pipeline: SemanticPhysicsConfig (ontology_pipeline_service.rs:27) defaults: constraint_strength 1.0,<br/>use_gpu_constraints true, materialise_inferred_edges false (lines 61-63). RESOLVED ADR-2068 - the repo-root<br/>ontology_physics.toml was read by no Rust code and has been deleted — the live toggle is the<br/>Settings.ontology_physics bool (src/settings/models.rs:63), gating POST /api/ontology-physics/enable|disable<br/>(api_handler/ontology_physics/mod.rs:107,387) which drives OntologyConstraintActor - see VC-04.7, VC-11.5
    end
    Sync->>Idx: maybe_refresh_after_sync(classes) (github_sync_service.rs:642, sync_graphs_with, defn line 468)
    alt ONTOLOGY_CLASS_INDEX_ENABLED unset (default, ontology_class_index.rs:75)
        Idx-->>Sync: skip refresh (line 474)
    else enabled
        Idx->>Idx: refresh_class_index condense_class RuVector memory_store namespace ontology-classes (line 495, defn 378, DEFAULT_NAMESPACE line 33)
    end
```

## VC-20.6 SPARQL read path - /ontology/query
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Route as POST /api/ontology/query<br/>src/handlers/api_handler/ontology/mod.rs:1668
    participant Handler as query_ontology<br/>src/handlers/ontology_handler.rs:903
    participant Validator as validate_read_only_sparql<br/>src/handlers/ontology_handler.rs:769
    participant Clamp as clamp_sparql_limit<br/>src/handlers/ontology_handler.rs:850
    participant CQRS as QueryOntologyHandler
    participant Cap as cap_result_rows<br/>src/handlers/ontology_handler.rs:880

    Client->>Route: POST /ontology/query query
    Note over Route: gated RequireAuth::power_user().mutations_only() (api_handler/ontology/mod.rs:1657)
    Route->>Handler: query_ontology(auth, state, request) (ontology_handler.rs:903)
    Handler->>Validator: validate_read_only_sparql(query) (ontology_handler.rs:911, defn :769)
    alt forbidden keyword INSERT DELETE DROP CLEAR LOAD CREATE ADD MOVE COPY WITH SERVICE
        Validator-->>Handler: Err operation not permitted (ontology_handler.rs:795, :801)
        Handler-->>Client: 400 bad_request
    else no read form present
        Validator-->>Handler: Err only SELECT ASK CONSTRUCT DESCRIBE permitted (ontology_handler.rs:814, :816)
        Handler-->>Client: 400 bad_request
    else valid read query
        Validator-->>Handler: Ok
        Handler->>Clamp: clamp_sparql_limit(query) (ontology_handler.rs:922, defn :850)
        Clamp-->>Handler: LIMIT injected or clamped to MAX_SPARQL_ROWS 10000 (ontology_handler.rs:838, :868)
        Handler->>CQRS: QueryOntologyHandler.handle(QueryOntology query) off the worker (ontology_handler.rs:928, see VC-20.13)
        CQRS-->>Handler: Vec of HashMap string string
        Handler->>Cap: cap_result_rows(results) (ontology_handler.rs:934, defn :880)
        Cap-->>Handler: rows capped at 10000, byte fence MAX_SPARQL_RESULT_BYTES 8388608, truncated flag (ontology_handler.rs:840, :884, :893)
        Handler-->>Client: 200 results rowCount truncated
    end
    Note over Handler: ADR-2004 - handler-level fence mirrors adapter fence sparql_select_json (crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:882)
```

## VC-20.7 ontology HTTP route families
```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Rbac as RbacGate on /api scope<br/>src/main.rs:1113
    participant OntScope as /api/ontology scope<br/>src/handlers/api_handler/ontology/mod.rs:1654
    participant DerivedScope as /api/ontology/derived scope<br/>src/handlers/ontology_derived_handler.rs:185
    participant CountScope as /api/ontology/class-count scope<br/>src/handlers/ontology_class_count_handler.rs:76
    participant AgentScope as /api/ontology-agent scope<br/>src/handlers/ontology_agent_handler.rs:275

    Client->>Rbac: any /api/* request
    rect rgb(230,255,230)
    Note over OntScope: registration order matters (ontology/mod.rs:1631-1642) - derived and class-count scopes register before this broader /ontology scope
    OntScope->>OntScope: POST graph classes properties axioms inference query sparql load load-axioms validate mapping apply, DELETE cache axioms iri classes iri (lines 1659-1679)
    OntScope->>OntScope: GET graph inferred classes properties axioms inference validate metrics inferences hierarchy state-at provenance reports id report health ws (lines 1681-1706)
    Note over OntScope: power_user().mutations_only() gate line 1657 - POST PUT DELETE gated, all GET public
    end
    rect rgb(255,230,230)
    DerivedScope->>DerivedScope: POST derived, POST derived/regenerate, GET derived which (line 189-191)
    Note over DerivedScope: RequireAuth::power_user() whole scope (line 187), fence detail see VC-22
    end
    rect rgb(230,230,255)
    CountScope->>CountScope: GET class-count (line 77)
    Note over CountScope: unauthenticated by design (ontology_class_count_handler.rs:24)
    end
    rect rgb(255,245,230)
    AgentScope->>AgentScope: POST discover read query traverse validate check, GET status (lines 276-282)
    AgentScope->>AgentScope: POST ontology-agent/propose, always 410 Gone (line 286, handler defn line 239)
    Note over AgentScope: RESOLVED ADR-2116 - every route on this scope is now anonymous. The /propose sub-scope and its<br/>RequireAuth+RateLimit middleware are gone with the write path they guarded (ontology_agent_handler.rs:268-272) —<br/>propose() answers 410 Gone naming vault propose as the replacement, needing no auth to reach no service. The governed<br/>write door moved to POST /decisions/record - see VC-24. ADR-2127 added /check with the tri-valued<br/>relation check and open-world scope fields on every answer.
    end
    rect rgb(245,245,245)
    InfScope->>InfScope: POST run batch validate, GET results ontology_id classify ontology_id, DELETE cache ontology_id (lines 273-285)
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

    OntologyActor ..> InitializeActor : Handler ontology_actor.rs 707
    OntologyActor ..> JobCompleted : Handler line 739
    OntologyActor ..> LoadOntologyAxioms : Handler line 749
    OntologyActor ..> UpdateOntologyMapping : Handler line 781
    OntologyActor ..> ValidateOntology : Handler line 791
    OntologyActor ..> ApplyInferences : Handler line 849
    OntologyActor ..> GetOntologyReport : Handler line 871
    OntologyActor ..> GetOntologyHealth : Handler line 906
    OntologyActor ..> ClearOntologyCaches : Handler line 938
    OntologyActor ..> TriggerReasoning : Handler line 962
    OntologyActor ..> GetCachedOntologies : Handler line 985

    note for LoadOntologyAxioms "LoadOntologyAxioms, ClearOntologyCaches,<br/>GetCachedOntologies, GetOntologyHealth now re-exported from<br/>visionclaw_actors::messages::ontology_messages<br/>(ontology_messages.rs:11-17), not defined in this webxr crate file"
    note for ValidateOntology "fields ontology_id, graph_data, mode, job_id<br/>(ontology_messages.rs:37-46), rtype Result of ValidationReport<br/>or String. owl_validator (src/services/owl_validator.rs:2) is a<br/>shim: pub use visionclaw_ontology::services::owl_validator::*<br/>(ADR-090 Phase A4), mirrored by src/inference/mod.rs:2 and<br/>src/reasoning/mod.rs:2 into the same crate.<br/>RESOLVED ADR-2065 - the orphaned second definition<br/>src/services/owl_validator_stubs.rs (a CPU-only<br/>ValidationConfig/PropertyGraph/RdfTriple/ConstraintSummary/<br/>ValidationReport set, no mod declaration or use anywhere,<br/>never compiled) has been deleted; owl_validator is now the<br/>single validation definition."
    note for ApplyInferences "fields rdf_triples, max_depth<br/>(ontology_messages.rs:52-55)"
    note for TriggerReasoning "fields ontology_id i64, source String<br/>(ontology_actor.rs:957-960), handler forwards to ReasoningActor<br/>per comment lines 971-973 which no longer exists"
    note for ApplyMaterializedAxioms "fields axioms, graph_data<br/>(ontology_messages.rs:76-79); NOT handled by OntologyActor -<br/>routed GPUManagerActor gpu_manager_actor.rs 645 to<br/>PhysicsSupervisor physics_supervisor.rs 732 to<br/>OntologyConstraintActor ontology_constraint_actor.rs 522"
    note for OntologyActor "RESOLVED - ProcessOntologyData and<br/>LogseqPage no longer exist in ontology_messages.rs; page<br/>ingest now runs through parse_page/project_ontology over the<br/>vault-core frontmatter model, see VC-20.1/VC-21"
```

## VC-20.13 Every ontology actix handler dispatches its sync CQRS handler off the worker

```mermaid
sequenceDiagram
    autonumber
    participant CL as Client
    participant AW as actix async worker thread
    participant H as get_owl_property<br/>src/handlers/ontology_handler.rs:409
    participant EX as execute_in_thread<br/>src/handlers/utils.rs:1
    participant BT as tokio blocking thread pool<br/>src/handlers/utils.rs:6
    participant QH as GetOwlPropertyHandler.handle<br/>src/handlers/ontology_handler.rs:416

    CL->>AW: GET /api/ontology/properties/{iri}
    AW->>H: handler future polled on the async worker
    H->>EX: execute_in_thread(move || handler.handle(GetOwlProperty))
    EX->>BT: tokio::task::spawn_blocking(f)
    BT->>QH: the sync CQRS handler runs here
    Note over BT,QH: INVARIANT: the CQRS query handlers are sync-over-async - they<br/>enter a nested block_on. Called inline on an actix worker they panic<br/>with "Cannot start a runtime from within a runtime" and drop the<br/>connection. ontology_handler.rs:419-421, add_owl_property at :488
    QH-->>BT: Result<Option<OwlProperty>, _>
    BT-->>EX: JoinHandle resolves
    EX-->>H: Ok(Ok(Some)) / Ok(Ok(None)) / Ok(Err) / Err
    Note over H: The four-arm match is uniform across every sibling in this file -<br/>join error is distinguished from handler error. ontology_handler.rs:426-441
    H-->>CL: 200 property, 404 not found, or 500
    Note over H,QH: get_owl_property and add_owl_property were the LAST two bare<br/>calls in this file, closed 2026-09-16 (ad8b83ad4). The same defect was<br/>fixed for one endpoint on 2026-08-10 and for add_axiom at<br/>ontology_handler.rs:594.
    Note over QH: Regression cover: owl_property_handlers_round_trip_off_the_async_worker<br/>ontology_handler.rs:1079, and the bare call pinned as the panic it is at<br/>ontology_handler.rs:1111. DEBT: the wrapping is load-bearing only while<br/>the CQRS handlers stay sync - the test module says to delete it if they<br/>become async-direct. ontology_handler.rs:1054
```
