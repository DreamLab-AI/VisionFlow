---
id: VC-07
title: Hexagonal ports, adapters, the CQRS application layer and the crate split
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2005, ADR-2016]
sources:
  - ../project/src/lib.rs
  - ../project/src/ports/mod.rs
  - ../project/src/ports/settings_repository.rs
  - ../project/src/ports/knowledge_graph_repository.rs
  - ../project/src/ports/graph_repository.rs
  - ../project/src/ports/physics_simulator.rs
  - ../project/src/adapters/mod.rs
  - ../project/src/adapters/actor_graph_repository.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/src/adapters/sqlite_enrichment_repository.rs
  - ../project/src/adapters/sqlite_canary_repository.rs
  - ../project/src/adapters/sqlite_kpi_repository.rs
  - ../project/src/adapters/actix_physics_adapter.rs
  - ../project/src/adapters/actix_semantic_adapter.rs
  - ../project/src/adapters/gpu_semantic_analyzer.rs
  - ../project/src/adapters/physics_orchestrator_adapter.rs
  - ../project/src/application/mod.rs
  - ../project/src/application/physics_service.rs
  - ../project/src/application/semantic_service.rs
  - ../project/src/application/events.rs
  - ../project/src/domain/mod.rs
  - ../project/src/domain/broker/mod.rs
  - ../project/src/domain/broker/broker_case.rs
  - ../project/src/domain/broker/broker_decision.rs
  - ../project/src/domain/broker/precedent_registry.rs
  - ../project/src/repositories/mod.rs
  - ../project/src/validation/mod.rs
  - ../project/src/config/mod.rs
  - ../project/Cargo.toml
  - ../project/src/app_state.rs
  - ../project/src/application/graph/queries.rs
  - ../project/src/application/knowledge_graph/directives.rs
  - ../project/src/application/knowledge_graph/queries.rs
  - ../project/src/application/ontology/directives.rs
  - ../project/src/application/ontology/queries.rs
  - ../project/src/application/settings/directives.rs
  - ../project/src/application/settings/queries.rs
  - ../project/src/handlers/utils.rs
  - ../project/src/handlers/ontology_handler.rs
  - ../project/src/handlers/api_handler/mod.rs
  - ../project/crates/visionclaw-contracts/src/lib.rs
  - ../project/crates/visionclaw-contracts/src/agent_action.rs
  - ../project/crates/visionclaw-contracts/src/telemetry.rs
  - ../project/crates/visionclaw-contracts/src/enterprise.rs
  - ../project/crates/visionclaw-contracts/src/github_adapter.rs
  - ../project/crates/visionclaw-contracts/src/version.rs
  - ../project/src/agent_events/schema.rs
  - ../project/sdk/visionflow-contracts/package.json
  - ../project/src/services/corpus_source/mod.rs
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/corpus_source/github.rs
verified_commit: 58f04f2eb272a2707737f2065f8241b931229e81
---

## VC-07.1 The hexagon — ports, adapters and where each canonical type lives
```mermaid
flowchart TB
    subgraph PORTS["src/ports — trait side"]
        P1["graph_repository::GraphRepository<br/>src/ports/graph_repository.rs — 88 lines, LEGACY"]
        P2["physics_simulator::PhysicsSimulator<br/>src/ports/physics_simulator.rs — 63 lines, LEGACY"]
        P3["REMOVED — semantic_analyzer::SemanticAnalyzer<br/>was src/ports/semantic_analyzer.rs, deleted by vc-knowledge<br/>the live seam is visionclaw_domain gpu_semantic_analyzer"]
        P4["knowledge_graph_repository::KnowledgeGraphRepository<br/>src/ports/knowledge_graph_repository.rs — 115 lines"]
        P5["settings_repository::SettingsRepository<br/>src/ports/settings_repository.rs — 14 lines, SHIM ONLY"]
    end
    subgraph SHIM["re-exported from visionclaw-domain by src/ports/mod.rs:26-45"]
        S1["ports::inference_engine::InferenceEngine"]
        S2["ports::ontology_repository::OntologyRepository"]
        S3["ports::gpu_physics_adapter — GpuPhysicsAdapter, PhysicsParameters,<br/>PhysicsStatistics, PhysicsStepResult, NodeForce, GpuDeviceInfo"]
        S4["ports::gpu_semantic_analyzer — GpuSemanticAnalyzer, ClusteringAlgorithm,<br/>CommunityDetectionResult, ImportanceAlgorithm, PathfindingResult,<br/>OptimizationResult, SemanticConstraintConfig, SemanticStatistics"]
    end
    subgraph ADAPT["src/adapters — implementation side"]
        A1["ActorGraphRepository<br/>src/adapters/actor_graph_repository.rs"]
        A2["OxigraphGraphRepository (ADR-2004 canonical)<br/>src/adapters/oxigraph_graph_repository.rs"]
        A3["SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs"]
        A4["SqliteEnrichmentRepository — data/enrichment.sqlite3 (WS-9)"]
        A5["SqliteCanaryRepository — data/liveness.sqlite3 (RES-a)"]
        A6["SqliteKpiRepository — data/kpi.sqlite3 (REC-4 ADR-130 D5)"]
        A7["ActixPhysicsAdapter — cfg(feature gpu)"]
        A8["ActixSemanticAdapter"]
        A9["GpuSemanticAnalyzerAdapter — cfg(feature gpu)"]
        A10["PhysicsOrchestratorAdapter"]
    end
    subgraph XCRATE["shims into visionclaw-adapters — src/adapters/mod.rs:22-35"]
        X1["whelk_inference_engine::WhelkInferenceEngine"]
        X2["messages"]
        X3["oxigraph_ontology_repository::OxigraphOntologyRepository"]
    end
    PORTS --> ADAPT
    SHIM --> ADAPT
    ADAPT --> XCRATE
    N1["INVARIANT (BASELINE) — persistence is Oxigraph (data/oxigraph) plus per-domain SQLite<br/>under DATA_DIR. One Oxigraph store is shared by the ontology and graph repositories.<br/>No networked graph DB. ADR-2004."]
    ADAPT --- N1
    N2["src/repositories/mod.rs is documentation only — all legacy Neo4j and SQL repositories<br/>were removed. Canonical adapters named there are OxigraphOntologyRepository,<br/>OxigraphGraphRepository, SqliteSettingsRepository (ADR-2004)."]
    ADAPT --- N2
    N3["Workspace — crates/graph-cognition-extract is on disk but EMPTY and NOT a workspace<br/>member (orphan directory, Cargo.toml). solid-pod-rs is a crates.io VERSION pin, not a<br/>git rev — the former rev=main resolution error no longer applies."]
    XCRATE --- N3
    N4["DOC-DRIFT — the adapters module doc src/adapters/mod.rs:8-15 lists seven modules as<br/>still in webxr, resolved in Phase A3; they are still there at this commit, so Phase A3<br/>has not landed. The dev build feature set is gpu,ontology,dev-auth — see VC-08."]
    ADAPT --- N4
```

## VC-07.2 ADR-2005 extraction state — what is actually a shim
```mermaid
sequenceDiagram
    autonumber
    participant C as caller in src/
    participant SH as src/ shim module
    participant CR as workspace crate

    Note over C,CR: ADR-2005 hexagonal crate split is PARTIAL. src/ modules that are pure re-export shims —
    C->>SH: use crate::ports::settings_repository::{SettingsRepository, SettingValue, ...}
    SH->>CR: pub use visionclaw_domain::ports::settings_repository::* (src/ports/settings_repository.rs:8)
    Note over SH: 14 lines total, ADR-090 Phase A6 slice 3, also re-exports AppFullSettings (settings_repository.rs:14)
    C->>SH: use crate::validation::*
    SH->>CR: pub use visionclaw_ontology::validation::* (src/validation/mod.rs:2)
    Note over SH: 2 lines total, ADR-090 Phase A4
    C->>SH: use crate::config::*
    SH->>CR: pub use visionclaw_domain::config::{graph_type, validation, visualisation, system, xr, services}<br/>plus AppFullSettings, DeveloperConfig, FeatureFlags, UserPreferences (src/config/mod.rs:21-56)
    Note over SH: 70 lines, ADR-090 Phase A6 slice 3. Six sibling .rs files are orphaned — see VC-09.2
    C->>SH: use crate::adapters::{messages, oxigraph_ontology_repository, whelk_inference_engine}
    SH->>CR: pub use visionclaw_adapters::* (src/adapters/mod.rs:22-35)
    Note over SH: ADR-090 Phase A1+
    C->>SH: use crate::ports::{inference_engine, ontology_repository, gpu_physics_adapter, gpu_semantic_analyzer}
    SH->>CR: pub use visionclaw_domain::ports::* (src/ports/mod.rs:26-45)
    Note over SH: module-path aliases kept so legacy call sites keep resolving
    Note over C,CR: project/src/lib.rs:47-64 re-exports ClientCoordinatorActor, MetadataActor,<br/>OptimizedSettingsActor, AppState, UserSettings from the root crate itself, plus ADR-090<br/>compatibility aliases MetadataStore, ProtectedSettings, SimulationParams from<br/>visionclaw_domain::models — preserving the old visionclaw_server:: API for external callers
    Note over C,CR: DIVERGENCE (BASELINE l.222) — actor extraction incomplete, 25 src/actors/*.rs<br/>vs 11 in crates/visionclaw-actors. The live tree runs from src/. See VC-02.
    Note over C,CR: DIVERGENCE (BASELINE 2026-09-04 crate and supervision closeout, l.277) — ADR-2005<br/>remains partial. The workspace adds converter and integration-test members to the census.
```

## VC-07.5 CQRS application layer — settings and knowledge-graph domains
```mermaid
classDiagram
    class SettingsDirectives {
      <<src/application/settings/directives.rs>>
      SaveAllSettings
      UpdateSetting
      UpdateSettingsBatch
      UpdatePhysicsSettings
      DeletePhysicsProfile
      ClearSettingsCache
    }
    class SettingsQueries {
      <<src/application/settings/queries.rs>>
      LoadAllSettings
      GetSetting
      GetSettingsBatch
      GetPhysicsSettings
      ListPhysicsProfiles
    }
    class KnowledgeGraphDirectives {
      <<src/application/knowledge_graph/directives.rs>>
      AddNode
      UpdateNode
      RemoveNode
      AddEdge
      UpdateEdge
      RemoveEdge
      BatchUpdatePositions
      SaveGraph
    }
    class KnowledgeGraphQueries {
      <<src/application/knowledge_graph/queries.rs>>
      LoadGraph
      GetNode
      GetNodeEdges
      GetNodesByMetadataId
      QueryNodes
      GetGraphStatistics
    }
    class QueryResult {
      <<src/application/knowledge_graph>>
    }
    class Handlers {
      <<one Handler per Directive and Query>>
      SaveAllSettingsHandler
      UpdateSettingHandler
      LoadAllSettingsHandler
      AddNodeHandler
      LoadGraphHandler
      QueryNodesHandler
    }
    SettingsDirectives ..> Handlers
    SettingsQueries ..> Handlers
    KnowledgeGraphDirectives ..> Handlers
    KnowledgeGraphQueries ..> Handlers
    KnowledgeGraphQueries ..> QueryResult
```

## VC-07.6 CQRS application layer — ontology, graph and services
```mermaid
classDiagram
    class OntologyDirectives {
      <<src/application/ontology/directives.rs>>
      AddOwlClass
      UpdateOwlClass
      RemoveOwlClass
      AddOwlProperty
      UpdateOwlProperty
      AddAxiom
      RemoveAxiom
      SaveOntologyGraph
      StoreInferenceResults
    }
    class OntologyQueries {
      <<src/application/ontology/queries.rs>>
      LoadOntologyGraph
      GetOwlClass
      GetOwlProperty
      ListOwlClasses
      ListOwlProperties
      GetClassAxioms
      GetInferenceResults
      GetOntologyMetrics
      QueryOntology
      ValidateOntology
    }
    class GraphQueries {
      <<src/application/graph/queries.rs>>
      GetGraphData
      GetBotsGraphData
      GetNodeMap
      GetPhysicsState
      GetConstraints
      GetEquilibriumStatus
      GetAutoBalanceNotifications
      ComputeShortestPaths
    }
    class PhysicsService {
      <<src/application/physics_service.rs>>
    }
    class SemanticService {
      <<src/application/semantic_service.rs>>
    }
    class InferenceService {
      <<REMOVED ADR-2066 — was src/application/inference_service.rs>>
      never constructed anywhere
      InferenceEvent
      InferenceServiceConfig
    }
    class DomainEvent {
      <<src/application/events.rs>>
    }
    OntologyDirectives ..> DomainEvent
    GraphQueries ..> PhysicsService
    OntologyQueries ..> InferenceService
    SemanticService ..> DomainEvent
    InferenceService ..> DomainEvent
```

## VC-07.7 A CQRS call in flight — GET /api/config
```mermaid
sequenceDiagram
    autonumber
    participant CL as client
    participant H as get_app_config<br/>src/handlers/api_handler/mod.rs:37
    participant EX as execute_in_thread<br/>src/handlers/utils.rs
    participant QH as LoadAllSettingsHandler<br/>src/application/settings
    participant PT as SettingsRepository port
    participant AD as SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs:378

    CL->>H: GET /api/config
    H->>QH: LoadAllSettingsHandler::new(state.settings_repository.clone()) (src/handlers/api_handler/mod.rs:42)
    Note over H,QH: state.settings_repository is Arc<dyn SettingsRepository> — src/app_state.rs:314
    H->>EX: execute_in_thread(move || handler.handle(LoadAllSettings)) (api_handler/mod.rs:44)
    Note over EX: hexser QueryHandler trait — the direct dispatch path, no CQRS bus
    EX->>QH: handle(LoadAllSettings)
    QH->>PT: port call
    PT->>AD: SQLite read
    alt Ok(Ok(Some(settings)))
        AD-->>H: 200 with version, features{ragflow, perplexity, openai, kokoro, whisper},<br/>websocket{minUpdateRate, maxUpdateRate, motionThreshold, motionDamping},<br/>rendering{ambientLightIntensity, enableAmbientOcclusion, backgroundColor},<br/>xr{enabled, roomScale, spaceType} (api_handler/mod.rs:47-71)
    else Ok(Ok(None))
        AD-->>H: warn "No settings found, using defaults" then AppFullSettings::default() (api_handler/mod.rs:73-78)
    end
    Note over H,AD: comment src/application/mod.rs:71 — application services were removed,<br/>handlers use actors directly via CQRS or direct messaging. There is no dispatcher bus.
    Note over EX: INVARIANT — every sync hexser handler enters a nested block_on, so calling one<br/>straight from an actix worker panics with the nested-runtime error Cannot start a runtime from within a runtime.<br/>execute_in_thread (spawn_blocking) is what makes the sync handler safe, and its Ok(Ok)/Ok(Err)/Err<br/>triple is the shape every call site must match — src/handlers/utils.rs
    Note over EX: DEBT — get_owl_property and add_owl_property were the LAST two bare sync calls<br/>and were only dispatched off the worker on 2026-09-16 in ad8b83ad4, and the earlier hierarchy<br/>endpoint had the same defect fixed on 2026-08-10. Nothing in the type system prevents the<br/>next bare call — ontology_handler.rs:424, ontology_handler.rs:489, regression module<br/>ontology_handler.rs:1054 pins the bare call as the panic it is
```

## VC-07.8 Storage-agnostic domain kernel — src/domain/broker
```mermaid
classDiagram
    class BrokerCase {
      <<src/domain/broker/broker_case.rs — 490 lines>>
    }
    class BrokerDecision {
      <<src/domain/broker/broker_decision.rs — 437 lines>>
    }
    class PrecedentRegistry {
      <<src/domain/broker/precedent_registry.rs — 101 lines>>
    }
    class BrokerModule {
      <<src/domain/broker/mod.rs — 43 lines>>
    }
    BrokerModule --> BrokerCase
    BrokerModule --> BrokerDecision
    BrokerModule --> PrecedentRegistry
    class SqliteEnrichmentRepository {
      <<src/adapters/sqlite_enrichment_repository.rs>>
      EnrichmentProposal
      StoredDecision
      EnrichmentStoreError
    }
    SqliteEnrichmentRepository ..> BrokerDecision
```

## VC-07.9 Broker kernel placement and the never-merged BrokerActor
```mermaid
sequenceDiagram
    autonumber
    participant H as REST handlers<br/>enrichment_proposals / broker_inbox / decision
    participant K as domain broker kernel<br/>src/domain/broker
    participant AD as SqliteEnrichmentRepository<br/>src/adapters/sqlite_enrichment_repository.rs
    participant AC as ACSP producer

    Note over K: src/domain/mod.rs:1-5 — pure aggregate and value-object logic with no transport<br/>or persistence dependency. Adapters (SQLite, Oxigraph, the ACSP producer, the REST<br/>handlers) sit OUTSIDE this module and depend inward on it.
    H->>K: aggregate operations on BrokerCase / BrokerDecision
    K-->>H: pure result — no I/O
    H->>AD: persist StoredDecision / EnrichmentProposal
    H->>AC: emit ACSP event
    Note over AC: stateless producer — there is no BrokerActor
    Note over H,AC: DIVERGENCE (BASELINE l.224) — BrokerActor was never merged. main uses a stateless<br/>ACSP producer plus a cherry-picked storage-agnostic domain broker kernel (~936 LOC).<br/>Measured here: broker_case.rs 490 + broker_decision.rs 437 = 927 lines, 1071 including<br/>mod.rs and precedent_registry.rs — consistent with the doc's ~936.
    Note over H,AC: DIVERGENCE (BASELINE ACSP workflow closeout 2026-09-04, l.273) — the retained kernel's<br/>presence does not prove integration into the elevation actor or the inbox DTO.<br/>Current source review does not certify a complete human-approval journey. See VC-05.
    Note over K,AD: ADR-2016 provenance append-only applies to the decision record — see VC-22
```

## VC-07.12 `visionclaw-contracts` — the cross-boundary DTO crate and its actual blast radius
```mermaid
flowchart TB
    LIB["crates/visionclaw-contracts/src/lib.rs:1-33<br/>'single source of truth for every envelope that<br/>crosses a process boundary' — ADR-10"]
    LIB --> M1["agent_action — AgentActionEnvelope, ActionKind<br/>outbound click -> agentbox, ADR-10 D3<br/>agent_action.rs:101 schema_version field"]
    LIB --> M2["telemetry — AgentTelemetryEnvelope<br/>inbound agentbox -> VisionClaw, ADR-10 D1<br/>telemetry.rs:57"]
    LIB --> M3["enterprise — EnterpriseEventEnvelope<br/>inbound forum -> VisionClaw, ADR-10 D5<br/>enterprise.rs:38"]
    LIB --> M4["github_adapter — ParsedMarkdown<br/>GitHub transport <-> ontology domain boundary, ADR-10 D11 + DDD-08<br/>github_adapter.rs:63"]
    LIB --> VER["version — SCHEMA_VERSION = 1, SCHEMA_VERSION_STRING = 'v1'<br/>version.rs:22,26 — single source of the version literal"]
    LIB --> TS["typescript-export feature -> cargo test emits .d.ts to bindings/<br/>published as npm @visionclaw/contracts, sdk/visionflow-contracts/package.json"]

    N1["DIVERGENCE — verified zero current consumers on either side this crate names.<br/>Rust: visionclaw-server does NOT list visionclaw-contracts under [dependencies]<br/>(only [workspace] members, Cargo.toml:2-4) — grep of AgentActionEnvelope /<br/>AgentTelemetryEnvelope / EnterpriseEventEnvelope / ParsedMarkdown across src/<br/>finds no import from this crate anywhere"]
    LIB --- N1
    N2["src/agent_events/schema.rs:61 defines its OWN, INDEPENDENT<br/>AgentActionEnvelope struct — same name, not the crate's type,<br/>no shared definition. A field added to the crate's envelope<br/>changes nothing this struct emits, and vice versa"]
    M1 --- N2
    N3["TS: no client/ or agentbox/ package.json references<br/>@visionclaw/contracts — the generated bindings/ and npm package<br/>exist and pass their own tests (schema_stability.rs, ts_export.rs)<br/>but are not wired into a consumer's build in this checkout"]
    TS --- N3
    N4["Practical blast radius today: editing this crate breaks only its<br/>own tests. It is a real, tested, independently-buildable contract<br/>layer staged ahead of the cross-repo wiring ADR-10 describes —<br/>not yet the thing anything downstream actually depends on."]
    N1 --- N4
    N2 --- N4
    N3 --- N4
    IND["visionclaw-contracts is independently buildable —<br/>cargo build --manifest-path crates/visionclaw-contracts/Cargo.toml"]
    LIB --- IND
```

## VC-07.13 CorpusSource — the ADR-2114 ingest port
```mermaid
flowchart TB
    T["trait CorpusSource — src/services/corpus_source/mod.rs:96<br/>describe, base_paths, list_pages, list_pages_under, fetch_page,<br/>vocabulary (default Ok(None))"]
    T --> L["LocalDirectorySource — src/services/corpus_source/local.rs:42,194<br/>walks VAULT_ROOT (src/services/corpus_source/local.rs:14), default source"]
    T --> G["GitHubSource — src/services/corpus_source/github.rs:15,47<br/>wraps EnhancedContentAPI, pre-existing remote pull"]
    B["source_from_env_with_github — src/services/corpus_source/mod.rs:157<br/>selects by corpus_source_kind() (src/services/corpus_source/mod.rs:193)<br/>over CORPUS_SOURCE, defaulting to local when VAULT_ROOT is set"]
    B --> T
    AS["AppState::new — src/app_state.rs:577-580<br/>corpus_source = source_from_env_with_github(enhanced_content_api)<br/>GitHubSyncService::new(corpus_source, ...)"]
    B --> AS
    N1["INVARIANT — change_marker differs by source: SHA1 blob hash for<br/>GitHub, mtime:size on disk for local (src/services/corpus_source/mod.rs:139)<br/>— equal markers across two listings mean unchanged"]
    L --- N1
    G --- N1
    N2["GitHubSyncService::new took EnhancedContentAPI directly before<br/>ADR-2114; the port now sits between it and the sync pipeline so<br/>parse -> dual graph -> Whelk reasoning runs unchanged over either source"]
    AS --- N2
```
