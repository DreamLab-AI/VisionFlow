---
id: VC-22
title: Data authority, provenance and erasure
area: visionclaw
governing:
  - ../project/docs/DATA-authority-erasure.md
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2004, ADR-2015, ADR-2016, ADR-2017, ADR-2069, ADR-2070]
sources:
  - ../project/docs/DATA-authority-erasure.md
  - ../project/Cargo.toml
  - ../project/src/services/nostr_service.rs
  - ../project/src/handlers/nostr_handler.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/src/adapters/sqlite_canary_repository.rs
  - ../project/src/adapters/sqlite_kpi_repository.rs
  - ../project/src/adapters/sqlite_enrichment_repository.rs
  - ../project/src/adapters/oxigraph_graph_repository.rs
  - ../project/crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs
  - ../project/crates/visionclaw-adapters/src/provenance_emitter.rs
  - ../project/src/services/provenance_writer.rs
  - ../project/src/services/provenance_trace.rs
  - ../project/src/services/intent_match.rs
  - ../project/migrations/sqlite/0006_kpi_agent_event_intent.sql
  - ../project/src/handlers/ingest_writeback_handler.rs
  - ../project/src/handlers/enrichment_proposals_handler.rs
  - ../project/src/handlers/trace_handler.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/src/services/corpus_source/mod.rs
  - ../project/src/services/corpus_source/local.rs
  - ../project/src/services/role_store.rs
  - ../project/src/middleware/rbac_gate.rs
  - ../project/src/main.rs
  - ../project/src/handlers/socket_flow_handler/position_updates.rs
  - ../project/crates/visionclaw-domain/src/utils/visibility_filter.rs
  - ../project/src/handlers/graph_access.rs
  - ../project/src/handlers/socket_flow_handler/types.rs
  - ../project/src/handlers/api_handler/graph/mod.rs
  - ../project/src/services/corpus_history.rs
  - ../project/client/src/services/solidPod/agentMemory.ts
  - ../project/client/src/services/solidPod/ldpClient.ts
  - ../project/client/src/services/SolidPodService.ts
  - ../project/scripts/backup-sqlite.sh
  - ../project/src/adapters/sqlite_share_repository.rs
  - ../project/src/handlers/graph_export_handler.rs
  - ../project/agentbox/management-api/routes/git-bridge.js
  - ../project/docker-compose.unified.yml
  - ../project/src/app_state.rs
  - ../project/src/actors/graph_state_actor.rs
  - ../project/src/services/ontology_mutation_service.rs
verified_commit: {visionclaw: 010e1109bd78f253f0d82382bb94908b5023b1e9, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93}
---

## For developers

This topic records which store is the write master for each data class: four SQLite databases, Oxigraph named graphs, the Solid pod and RuVector. It also covers how provenance is appended, how erasure reaches each store, and how a caller's view of the graph is cut. At ec785afa9 two mechanisms were added: the visible subgraph now scrubs hidden page names out of visible metadata (VC-22.12), and full syncs append public corpus history to the provenance graph (VC-22.13). At 809a859aa graph shares became a fifth durable SQLite store, `data/shares.sqlite3`, backed up as a required member but not reached by per-user erasure (VC-22.14), and the corpus-history writer now takes its public test from `NodeVisibility::of_node`.

## For the business

Every piece of data has one authoritative home, so a correction or a deletion has a known place to land. Private pages must not leak through the names other pages mention, and the history kept for audit contains only what was already public, because that history cannot yet be deleted.

## VC-22.1 Write-master per data class

```mermaid
flowchart LR
    AuthContent["Authored content<br/>public: true / owl-class markdown"] --> CorpusSrc["CorpusSource port (ADR-2114)<br/>local vault default (VAULT_ROOT set), GitHub opt-in<br/>corpus_source/mod.rs:203 corpus_source_kind<br/>github_sync_service.rs:1916 ingest"]
    CorpusSrc -->|"github_sync_service.rs:1090, :1147 rebuild_assert_graph CLEAR+INSERT"| AssertGraph["GRAPH_ONTOLOGY :assert<br/>oxigraph_ontology_repository.rs:48"]
    CorpusSrc -->|"oxigraph_graph_repository.rs:1212 clear_graph drops knowledge and agent graphs"| KnowledgeGraph["GRAPH_KNOWLEDGE<br/>oxigraph_ontology_repository.rs:50"]

    VisIntent["Visibility intent<br/>visibility + owner_pubkey"] --> SettingsDB["settings.sqlite3 settings table<br/>sqlite_settings_repository.rs:61-68"]
    SettingsDB -->|"projected by"| VisFilter["is_dropped_for, private nodes fail closed<br/>visibility_filter.rs:127<br/>filter_enabled visibility_filter.rs:46, PUBKEY_VISIBILITY_FILTER default ON (:34)<br/>position stream position_updates.rs:10"]
    GraphActor["GraphStateActor snapshot<br/>private_nodes from node metadata public and owner_pubkey<br/>graph_state_actor.rs:173-179"] -->|"feeds"| VisFilter

    AssertGraph -.->|"reasoner run: store_inference_results DELETE+INSERT oxigraph_ontology_repository.rs:2669, :2763-2768"| InferredGraph["GRAPH_ONTOLOGY_INFERRED<br/>oxigraph_ontology_repository.rs:49 derived, never primary"]

    OpState["Operational state"] --> SqliteFour["4x SQLite WAL: settings/enrichment/kpi/liveness<br/>src/adapters/sqlite_*_repository.rs"]

    VectorMem["Vector / agent memory"] --> RuVector["RuVector external Postgres<br/>mcp__claude-flow__memory_* -> ruvector-postgres:5432"]

    EventJournal["Event journal / provenance"] --> ProvGraph["GRAPH_PROVENANCE append-only<br/>oxigraph_ontology_repository.rs:57"]

    AuditRBAC["Audit evidence RBAC/auth"] --> RoleTable["user_roles table in settings.sqlite3<br/>role_store.rs:46-52"]

    Sessions["Session cache<br/>NostrService users map"] --> RedisGate{"redis feature AND REDIS_URL?<br/>Cargo.toml:278, nostr_service.rs:149"}
    RedisGate -->|yes| Redis["Optional Redis session persistence<br/>SETEX (set_ex) with token-expiry TTL<br/>nostr_service.rs:287"]
    RedisGate -->|no| Memory["In-memory sessions only<br/>nostr_service.rs:164"]
    Redis -.->|"restore at initialisation"| Sessions
    RedisScope["Not enabled by default; deployment not verified<br/>ADR-2004 all-non-triple-state wording needs scope"] -.-> Redis

    Credentials["Credentials"] --> DotEnv[".env plaintext filesystem"]

    DerivedWB["Derived write-back only<br/>append_derived_quads:733"] --> SummaryGraph["GRAPH_ONTOLOGY_SUMMARY<br/>oxigraph_ontology_repository.rs:63"]
    DerivedWB --> ObservedGraph["GRAPH_ONTOLOGY_OBSERVED<br/>oxigraph_ontology_repository.rs:64"]

    DivLegacy["DIVERGENCE: legacy ADRs assign primacy to Oxigraph 132 / Pod 050-052 / GitHub 051 / RuVector 030 / provenance 033-034-124-128<br/>code resolves as this matrix, legacy prose not reconciled<br/>docs/DATA-authority-erasure.md:99"]
    AssertGraph -.-> DivLegacy
    RuVector -.-> DivLegacy

    DivCreds["PROPOSED ADR-2104: execute the SOPS rollout or formally withdraw ADR-109 - plaintext .env is documented as the interim state either way<br/>a 43MB scripts/sops binary dated 2026-05-09 is present and gitignored, but no .sops.yaml, secrets.enc.yaml or sops-env.sh exists<br/>docs/DATA-authority-erasure.md:134"]
    DotEnv -.-> DivCreds

    DebtClear["DEBT: clear_inferred_graph oxigraph_ontology_repository.rs:697 has no production caller, only its test :3608"]
    InferredGraph -.-> DebtClear

    DivWorkspace["OPEN: which row of this matrix moves, and when, if ADR-2137 (proposed, implementation none, activation inactive) makes<br/>Oxigraph and the other query stores optional rebuildable projections of authored Markdown?<br/>docs/DATA-authority-erasure.md:263"]
    AssertGraph -.-> DivWorkspace
```

## VC-22.2 SQLite schemas (four write-master databases)

```mermaid
erDiagram
    SETTINGS {
        text key PK
        text owner_pubkey PK
        text value
        text description
        integer updated_at
    }
    PHYSICS_PROFILES {
        text profile_name PK
        text owner_pubkey PK
        text settings_json
        integer updated_at
    }
    AUDIT_LOG {
        integer id PK
        integer occurred_at
        text actor_pubkey
        text request_method
        text request_path
        integer status_code
        text detail_json
    }
    SYNC_FILE_METADATA {
        text file_name PK
        text sha1
        integer updated_at
    }
    USER_ROLES {
        text pubkey PK
        text role
        text assigned_by
        integer updated_at
    }
    LIVENESS_CANARIES {
        text canary_id PK
        text description
        text kind
        text owner_repo
        text wave
        text sha_at_registration
        integer registered_at_ms
    }
    CANARY_FIRES {
        integer id PK
        text canary_id FK
        text evidence
        text sha
        integer fired_at_ms
    }
    KPI_AGENT_EVENTS {
        integer id PK
        integer event_id
        integer source_agent_id
        integer action_type
        integer observed_at_ms
        text agent_did
        text handoff_id
        integer token_count
    }
    KPI_SNAPSHOTS {
        integer id PK
        text kpi
        real value
        real confidence
        integer sample_count
        integer computed_at_ms
        text sha
    }
    KPI_LINEAGE {
        integer id PK
        integer snapshot_id FK
        text source_kind
        text source_ref
        real contribution
    }
    ENRICHMENT_PROPOSALS {
        text case_id PK
        text category
        text source_iri
        text proposal_json
        text status
        integer created_at
        integer updated_at
    }
    ENRICHMENT_DECISIONS {
        integer id PK
        text case_id FK
        text outcome
        integer attributed
        text broker_pubkey
        integer writeback_triggered
        integer writeback_committed
        integer writeback_committed_at_ms
        text activity_urn
        text owner_did
        integer decided_at_ms
    }

    ENRICHMENT_PROPOSALS ||--o{ ENRICHMENT_DECISIONS : "case_id (sqlite_enrichment_repository.rs:96)"
    LIVENESS_CANARIES ||--o{ CANARY_FIRES : "canary_id (sqlite_canary_repository.rs:64)"
    KPI_SNAPSHOTS ||--o{ KPI_LINEAGE : "snapshot_id (sqlite_kpi_repository.rs:104)"
    SETTINGS ||--|| USER_ROLES : "same settings.sqlite3 connection (role_store.rs:4-5, main.rs:784-788, settings repo opened at app_state.rs:417-423)"
```

## VC-22.3 Derived-writeback fence (`POST /api/ingest/writeback`, ADR-2015)

```mermaid
sequenceDiagram
    autonumber
    participant GB as GitBridge<br/>agentbox management-api/routes/git-bridge.js:773
    participant WBH as writeback handler<br/>src/handlers/ingest_writeback_handler.rs:75
    participant APD as apply_decision<br/>src/handlers/enrichment_proposals_handler.rs:473
    participant REPO as SqliteEnrichmentRepository::record_decision<br/>src/adapters/sqlite_enrichment_repository.rs:608
    participant ONTO as append_derived_quads<br/>oxigraph_ontology_repository.rs:734

    GB->>WBH: POST /api/ingest/writeback decision block, approve actions only (git-bridge.js:750), route at ingest_writeback_handler.rs:103
    WBH->>WBH: attribution_pubkey approvedBy did:nostr or hex (ingest_writeback_handler.rs:62-70)
    WBH->>APD: apply_decision(case_id, BrokerDecisionRequest, effective_tier None) (ingest_writeback_handler.rs:88-99)
    APD->>APD: gate_tier(declared tier, effective_tier) stricter wins, then check_rationale 422 when refused (enrichment_proposals_handler.rs:508-510, gate_tier :382)
    APD->>REPO: record_decision INSERT decision + UPDATE proposal.status (sqlite_enrichment_repository.rs:608-680)
    alt writeback_triggered AND attributed (enrichment_proposals_handler.rs:573)
        APD->>ONTO: append_derived_summary(owner_did, activity_urn, triples) (enrichment_proposals_handler.rs:576)
        alt graph in DERIVED_FENCE assert or inferred
            ONTO--xAPD: rejected fenced graph not writable via derived path (oxigraph_ontology_repository.rs:748-751)
        else graph is summary or observed
            ONTO->>ONTO: INSERT DATA GRAPH GRAPH_ONTOLOGY_SUMMARY/OBSERVED (oxigraph_ontology_repository.rs:786-795)
            ONTO-->>APD: Ok(quad count)
            APD->>REPO: mark_writeback_committed case_id activity_urn (enrichment_proposals_handler.rs:588-590)
        end
    else unattributed or no writeback trigger
        Note over APD: decision recorded, no KG write - unattributed approval writes no fact (ingest_writeback_handler.rs:21-23)
    end
    APD-->>WBH: HttpResponse (broker:case_decided broadcast)
    Note over ONTO: INVARIANT: only summary/observed writable, assert/inferred rejected in the repo method itself (ADR-2015, oxigraph_ontology_repository.rs:70 DERIVED_FENCE)
```

## VC-22.4 Provenance write — two producers, one append-only graph (ADR-2016)

```mermaid
sequenceDiagram
    autonumber
    participant MUT as emit_proposal_provenance<br/>src/services/ontology_mutation_service.rs:174
    participant EMITTER as reify_activity<br/>crates/visionclaw-adapters/src/provenance_emitter.rs:307
    participant WRITER as build_assertion_version<br/>src/services/provenance_writer.rs:353
    participant T2 as caller transaction spine<br/>src/services/provenance_writer.rs:7-9
    participant STORE as GRAPH_PROVENANCE urn:ngm:graph:provenance<br/>oxigraph_ontology_repository.rs:57

    par Activity-record path (ADR-127 / ADR-2016)
        MUT->>EMITTER: emit_activity_nonfatal(store, record) (provenance_emitter.rs:446, called at ontology_mutation_service.rs:216)
        EMITTER->>EMITTER: build quads Activity/Agent/Entity triad (provenance_emitter.rs:137-251)
        EMITTER->>STORE: store.insert(Quad) x n, INSERT only (provenance_emitter.rs:32-33 header invariant)
    and Assertion-version path (ADR-049, reconciled PRD-022 WS-2)
        WRITER->>WRITER: build_assertion_version pure builder, no store I/O (provenance_writer.rs:353-471)
        WRITER-->>T2: AssertionVersionQuads.provenance_quads (re-exports GRAPH_PROVENANCE at provenance_writer.rs:87)
        T2->>STORE: execute quads in one atomic transaction (caller-owned, T2 per module doc)
    end
    Note over STORE: INVARIANT: append-only - no DELETE, DROP or CLEAR issued against this graph (ADR-2016,<br/>provenance_emitter.rs:32-33), verified by append_only_verified test (provenance_emitter.rs:908)
    Note over WRITER: Retraction only ADDS dl:validTo and deletes nothing - history never removed (provenance_writer.rs:218,478-502)
    Note over EMITTER,WRITER: PROPOSED ADR-2102: keep the append-only invariant and satisfy erasure by crypto-shredding the per-subject key so<br/>quad structure and hash chain survive while plaintext does not - the alternative is to declare provenance out of erasure<br/>scope explicitly - ADR-2102 exists to force that choice rather than leave it to omission
```

## VC-22.5 Unified provenance trace read path (`GET /api/trace`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant TH as unified_trace<br/>src/handlers/trace_handler.rs:35
    participant SVC as ProvenanceTraceService<br/>src/services/provenance_trace.rs:308
    participant KPI as trajectories_since<br/>src/adapters/sqlite_kpi_repository.rs:368
    participant ENR as provenance_decisions_since<br/>src/adapters/sqlite_enrichment_repository.rs:924
    participant JOIN as build_trace<br/>src/services/provenance_trace.rs:186
    participant LH as LivenessHarness<br/>src/handlers/trace_handler.rs:64

    C->>TH: GET /api/trace agent window_ms (trace_handler.rs:76)
    TH->>SVC: ProvenanceTraceService::new(enrichment_repo, kpi_repo) (trace_handler.rs:39-42)
    TH->>SVC: query(window, agent_did) (trace_handler.rs:44)
    SVC->>KPI: trajectories_since(cutoff) (provenance_trace.rs:339)
    SVC->>ENR: provenance_decisions_since(cutoff) (provenance_trace.rs:344)
    opt agent filter present
        SVC->>SVC: retain rows where agent_did matches (provenance_trace.rs:349, :350)
    end
    SVC->>JOIN: build_trace(trajectories, decisions, pod_marks empty, pod_source_available false) (provenance_trace.rs:354)
    JOIN-->>SVC: ProvenanceTrace sources_present sources_absent joins
    alt joins_multiple_source_kinds true
        TH->>LH: observe(CANARY_REC11_TRACE, evidence) (trace_handler.rs:63-65)
    else single source or none
        Note over TH: no canary fire, only one live source kind
    end
    TH-->>C: 200 JSON ProvenanceTrace
    Note over SVC,ENR: this trace reads only the two SQLite-backed sources - it does NOT query the Oxigraph GRAPH_PROVENANCE<br/>append-only graph ADR-2016 covers (ADR-2016 consumer closeout 2026-09-05)
    JOIN->>JOIN: intent_match(declared intent, action type, target urn)<br/>src/services/intent_match.rs:221
    Note over JOIN: ADR-2110 FR5.2: an agent-event record now carries the DECLARED<br/>intent verbatim and a three-valued verdict. null means NO CLAIM WAS MADE,<br/>which is not the same as a failed claim. Decision and git-mark records<br/>carry neither. provenance_trace.rs:206-211, :222-223, :234-235
    Note over KPI: INVARIANT: the declaration is persisted verbatim on kpi_agent_events<br/>rather than re-derived at read time, so a later reading cannot soften it.<br/>migrations/sqlite/0006_kpi_agent_event_intent.sql:5-9, :16, row field at<br/>src/adapters/sqlite_kpi_repository.rs:245
    Note over JOIN: pod_git_mark source is hardcoded pod_source_available false (provenance_trace.rs:357-358), always reported sources_absent here (:243)
```

## VC-22.6 Erasure — `deleteAgentMemory` Pod path (RuVector tombstone gap)

```mermaid
sequenceDiagram
    autonumber
    participant UI as SolidPodService.deleteAgentMemory<br/>client/src/services/SolidPodService.ts:366
    participant MEM as agentMemory.deleteAgentMemory<br/>client/src/services/solidPod/agentMemory.ts:191
    participant LDP as deleteResource<br/>client/src/services/solidPod/ldpClient.ts:201
    participant POD as Solid Pod HTTP server

    UI->>MEM: deleteAgentMemory(podPath, agentId, key) (SolidPodService.ts:368)
    MEM->>MEM: sanitizePreferenceKey(key), build container path (agentMemory.ts:196-197)
    MEM->>LDP: deleteResource(containerPath + safeKey + .jsonld) (agentMemory.ts:198)
    LDP->>POD: fetchWithAuth method DELETE (ldpClient.ts:203)
    alt response ok or 404
        POD-->>LDP: 2xx or 404 (already absent)
        LDP-->>MEM: true
    else other error status
        POD-->>LDP: 4xx/5xx
        LDP-->>MEM: false, logs DELETE failed (ldpClient.ts:205-207)
    end
    MEM-->>UI: boolean deleted-or-absent
    rect rgb(255, 230, 230)
        Note over MEM,POD: DIVERGENCE: no reverse tombstone to RuVector - embedding row persists and stays searchable (docs/DATA-authority-erasure.md:89)
        Note over MEM: this Pod-side delete never calls mcp__claude-flow__memory_delete or any RuVector endpoint - RuVector is<br/>external Postgres reached only via MCP, no delete-propagation hook exists in this path
        Note over MEM: DRIFT: the server now also holds a direct read-only RuVector client (tokio-postgres from RUVECTOR_PG_CONNINFO, crate visionclaw-memory-cloud),<br/>so MCP-only access holds for writes and deletes but no longer for reads, and the embedding row this path leaves behind has a second consumer Cargo.toml:144-149
    end
```

## VC-22.7 SQLite online backup (`scripts/backup-sqlite.sh`, ADR-2017)

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator/cron
    participant SH as backup-sqlite.sh
    participant CT as visionclaw_container
    participant SQL as sqlite3 CLI in container

    OP->>SH: scripts/backup-sqlite.sh (scripts/backup-sqlite.sh:1)
    SH->>SH: detect_mode docker or host (backup-sqlite.sh:138, :147)
    loop for each db in REQUIRED_DBS + OPTIONAL_DBS (backup-sqlite.sh:75-76)
        SH->>CT: docker exec test -f db (backup_one_docker, backup-sqlite.sh:173)
        alt db present
            SH->>SQL: sqlite3 db .backup tmp (ONLINE BACKUP API)
            SQL-->>SH: PRAGMA integrity_check = ok
            SH->>SH: docker cp tmp to DEST/db
            opt VERIFY_RESTORE=1 (default)
                SH->>SH: verify_restore copy to scratch, SELECT count from expected tables (backup-sqlite.sh:111-130)
            end
        else db absent
            alt is_required(db) (backup-sqlite.sh:88)
                SH->>SH: record missing_required (backup-sqlite.sh:223-224)
            else optional db missing
                SH->>SH: record missing_optional, continue (backup-sqlite.sh:226-228), logged after the loop (:248)
            end
        end
    end
    alt missing_required non-empty (backup-sqlite.sh:243-246)
        SH--xOP: die required database(s) missing, refuse to publish incomplete backup set, no manifest written
    else all required present
        SH->>SH: write MANIFEST.txt with sha256 per db
        SH-->>OP: DONE - N databases in DEST
    end
    Note over SH: PROPOSED ADR-2103: Oxigraph gains a point-in-time snapshot as a REQUIRED backup member with a declared RPO from the<br/>schedule and an RTO from a measured restore drill - GRAPH_PROVENANCE is restore-only because it cannot be re-derived<br/>extends ADR-2017, which stays the per-class write-master authority
```

## VC-22.8 Dual-writes: fact fan-out to two stores

```mermaid
sequenceDiagram
    autonumber
    participant SYNC as GitHubSyncService<br/>src/services/github_sync_service.rs:282
    participant KG as GRAPH_KNOWLEDGE<br/>oxigraph_ontology_repository.rs:50
    participant AG as GRAPH_ONTOLOGY assert<br/>oxigraph_ontology_repository.rs:48
    participant APD as apply_decision<br/>src/handlers/enrichment_proposals_handler.rs:473
    participant SQ as enrichment_decisions row<br/>sqlite_enrichment_repository.rs:608
    participant SUM as GRAPH_ONTOLOGY_SUMMARY<br/>oxigraph_ontology_repository.rs:63
    participant PROV as GRAPH_PROVENANCE<br/>oxigraph_ontology_repository.rs:57

    critical content sync fan-out
        SYNC->>KG: batch_add_nodes ingest corpus pages (github_sync_service.rs:1916)
        SYNC->>AG: save_ontology_graph atomic CLEAR GRAPH assert then INSERT DATA (github_sync_service.rs:1147)
    end
    Note over SYNC,AG: authoritative store (the CorpusSource, ADR-2114) commits first via sync run - the Oxigraph projection is regenerated, never hand-edited (docs/DATA-authority-erasure.md:64)

    critical decision fan-out (enrichment_proposals_handler.rs:561-594)
        APD->>SQ: record_decision INSERT+UPDATE one transaction (sqlite_enrichment_repository.rs:608-680)
        APD->>SUM: append_derived_summary INSERT DATA :summary (enrichment_proposals_handler.rs:576, oxigraph_ontology_repository.rs:786-790)
        SUM->>SUM: prov:wasGeneratedBy marker on each subject, written INTO :summary (oxigraph_ontology_repository.rs:822-828)
        Note over SUM,PROV: CORRECTED - this fan-out writes nothing to GRAPH_PROVENANCE: append_derived_summary routes only<br/>through append_derived_quads (oxigraph_ontology_repository.rs:842), and the decision handler emits no PROV activity
    end
    Note over APD,SUM: no cross-store 2PC - the durable decision commits in SQLite first (writeback_triggered=true) - the Oxigraph<br/>write is best-effort, writeback_committed flips only on Ok (enrichment_proposals_handler.rs:576, :589)
```

**Drift:** the approved-decision fan-out leaves nothing in the append-only `GRAPH_PROVENANCE` ledger ADR-2016 describes: the `prov:wasGeneratedBy` marker is a quad in `:summary` (`crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:822-828`), `append_derived_summary` writes only through `append_derived_quads` (`crates/visionclaw-adapters/src/oxigraph_ontology_repository.rs:842`), and `apply_decision` (`src/handlers/enrichment_proposals_handler.rs:473`) emits no PROV activity.

## VC-22.9 RBAC open-by-default posture on a read

```mermaid
sequenceDiagram
    autonumber
    participant C as Anonymous or unassigned client
    participant GATE as RbacGateMiddleware<br/>src/middleware/rbac_gate.rs:258
    participant REQ as required_level<br/>src/middleware/rbac_gate.rs:147
    participant BOOT as main.rs RBAC bootstrap<br/>src/main.rs:775
    participant RS as RoleStore.effective_role<br/>src/services/role_store.rs:359

    Note over BOOT: RBAC_PUBLIC_READS is fail-closed in code - .unwrap_or(false)<br/>(rbac_gate.rs:135-141) - and set on only by compose<br/>(docker-compose.unified.yml:132)
    BOOT->>BOOT: RBAC_ALLOW_OWNERLESS env check (main.rs:798, const at role_store.rs:33)
    alt no Owner assigned AND RBAC_ALLOW_OWNERLESS=1 (main.rs:798, :803)
        BOOT->>BOOT: warn, run owner-less, only POWER_USER_PUBKEYS to Admin fallback applies
    else no Owner assigned and flag unset
        BOOT--xBOOT: FATAL refuse to start, fail-closed (main.rs:810-819)
    end
    C->>GATE: GET /api/graph/data (safe method)
    GATE->>REQ: required_level(GET, routed_path, public_reads=true) - the decoded routed path,<br/>not the raw URI (rbac_gate.rs:69, :262)
    Note over REQ: GET /api/admin/rbac/whoami resolves to Authenticated before the Admin rule, so<br/>any signed-in user may read their own role (rbac_gate.rs:160-166)
    alt public_reads true (rbac_gate.rs:172-175)
        REQ-->>GATE: None - public route, no auth required
        GATE-->>C: 200 passthrough (rbac_gate.rs:264-269)
        Note over C,GATE: since dc7fd3858 the graph handler then cuts the reply to the caller's view -<br/>visible_subgraph at src/handlers/api_handler/graph/mod.rs:352, see VC-22.12
    else public_reads false
        REQ-->>GATE: Some(ReadOnly)
        GATE->>RS: effective_role(pubkey, is_power_user) for a mutating/gated route
        alt explicit assignment exists
            RS-->>GATE: stored UserRole (role_store.rs:361)
        else no assignment AND is_power_user
            RS-->>GATE: Admin (role_store.rs:363-364)
        else no assignment, not power user
            RS-->>GATE: default_role, Editor unless RBAC_DEFAULT_ROLE=viewer (role_store.rs:366)
        else lookup error
            RS-->>GATE: Viewer, fails closed (role_store.rs:369-371)
        end
    end
    rect rgb(255, 230, 230)
    Note over BOOT,RS: CORRECTED ADR-2070 (raised by estate ADR-2087) - the CODE fails closed:<br/>public_reads_enabled() ends .unwrap_or(false) (rbac_gate.rs:135-141) and<br/>main.rs:810-819 refuses to<br/>start owner-less unless RBAC_ALLOW_OWNERLESS is set. The shipped compose inverts<br/>both<br/>(docker-compose.unified.yml:132, :133, with the :-1 default), so an unassigned<br/>pubkey resolves to Editor<br/>(role_store.rs:359). The open posture is ADR-2027's deliberate demo default and<br/>stays.
    end
```

## VC-22.10 Full-sync corpus rebuild, ontology-typed filter and decision re-materialisation (ADR-2114, ADR-050)

```mermaid
sequenceDiagram
    autonumber
    participant OP as Operator (force_full_sync)
    participant SYNC as GitHubSyncService.sync_graphs<br/>src/services/github_sync_service.rs:342
    participant KGREPO as kg_repo.clear_graph<br/>src/adapters/oxigraph_graph_repository.rs:1212
    participant REBUILD as rebuild_assert_graph<br/>src/services/github_sync_service.rs:1090
    participant AG as GRAPH_ONTOLOGY assert
    participant DECREB as rematerialise_decisions<br/>src/services/github_sync_service.rs:1236
    participant DIRECT as add_owl_class/add_axiom<br/>application/ontology/directives.rs (governed write door)

    OP->>SYNC: FORCE_FULL_SYNC=1
    alt force_full_sync true (github_sync_service.rs:465, :742)
        SYNC->>KGREPO: clear_graph() drops GRAPH_KNOWLEDGE and GRAPH_AGENT (github_sync_service.rs:515-519, oxigraph_graph_repository.rs:1215-1216)
        SYNC->>REBUILD: rebuild_assert_graph(stats) (github_sync_service.rs:742-744)
        REBUILD->>REBUILD: filter KG nodes to ontology-typed only, ONTOLOGY_TYPE_KEY metadata (github_sync_service.rs:1106)
        REBUILD->>AG: CLEAR GRAPH assert then INSERT DATA class nodes + class-relation edges (save_ontology_graph, atomic) (github_sync_service.rs:1147)
        Note over REBUILD,AG: this CLEAR wipes the ENTIRE assert graph including runtime OWL classes/axioms added via the governed write door (github_sync_service.rs:1073-1075)
        REBUILD->>DECREB: rematerialise_decisions() ADR-050 read-half, called AFTER the class CLEAR+INSERT (github_sync_service.rs:1163)
        DECREB->>DECREB: list_pages_under(DECISIONS_DIR), decision_page_quads_logged per page (github_sync_service.rs:1237, :1265)
        DECREB->>AG: insert_quads_to_store re-derived dl:DecisionRecord quads, non-fatal on failure (github_sync_service.rs:1276, :1163-1169)
    else incremental sync (SHA1 filter narrowed file list)
        SYNC->>SYNC: clear_graph guard not entered, existing data left intact (github_sync_service.rs:515)
    end
    Note over DIRECT,AG: a class added via add_owl_class/add_axiom between full-syncs is NOT itself in GRAPH_PROVENANCE-protected history for the<br/>assert graph - only re-derivation from the corpus (the visionGraph vault via CorpusSource, ADR-2114) restores it after the next rebuild
    Note over DECREB,AG: attribution is deliberately NOT re-materialised here - the signed PROV-O attribution stays in GRAPH_PROVENANCE (ADR-049)<br/>the corpus decision page carries only the summary (github_sync_service.rs:1155-1162)
    Note over SYNC,AG: PROPOSED ADR-2102: one durable erasure record, five store acknowledgements, and a partial erasure that is recorded<br/>and retryable rather than a silent success - agentbox ADR-2060 is the RuVector-side half and is referenced, not superseded
```

## VC-22.11 SQLite membership contract vs KPI/enrichment sole-source risk

```mermaid
flowchart TD
    Script["backup-sqlite.sh<br/>REQUIRED_DBS default: settings.sqlite3 enrichment.sqlite3 kpi.sqlite3 shares.sqlite3<br/>scripts/backup-sqlite.sh:75"] --> Req["REQUIRED - missing member fails whole run, no manifest written<br/>scripts/backup-sqlite.sh:243-246"]
    Script --> Opt["OPTIONAL_DBS default: liveness.sqlite3 - missing is logged, run still succeeds<br/>scripts/backup-sqlite.sh:76,248"]
    Req --> Verify["VERIFY_RESTORE=1 default: scratch-restore + SELECT count on expected_tables_for<br/>scripts/backup-sqlite.sh:97-106,110-131"]
    Verify --> Manifest["MANIFEST.txt sha256 per db<br/>scripts/backup-sqlite.sh:252-264"]
    Opt -.-> Missing["missing_optional logged, count still gt 0 required<br/>scripts/backup-sqlite.sh:248-249"]

    DriftNote["RESOLVED ADR-2069: the required/optional membership contract is ratified as the design and DATA-authority-erasure.md now describes it - a missing REQUIRED_DBS member (settings/enrichment/kpi/shares, backup-sqlite.sh:75) aborts the run and publishes no manifest (:243-246), a missing OPTIONAL_DBS member (liveness, :76) is recorded (:226-228), logged and the run continues (:248). Oxigraph still has NO point-in-time backup - that bullet stands. DRIFT: DATA-authority-erasure.md:111 cites :250 for that, which is the rc check"]
    Req -.-> DriftNote
    Verify -.-> NoSchema["DEBT: RESOLVED at visionclaw 46edc196 — shares.sqlite3 now has an expected_tables_for arm,<br/>so its restore check runs against shared_graphs and schema_migrations<br/>scripts/backup-sqlite.sh:97-106, :102, :114"]
```

## VC-22.12 Visible subgraph and hidden page-name scrub (ADR-060 M1)

<!-- 2026-11-06 restamp ec785afa9: new. dc7fd3858 gates REST graph reads with the /wss policy, 395b62924 and efec9b27d scrub hidden page names -->
```mermaid
flowchart TB
    CALL["Callers: caller_view_for graph_access.rs:36-41,<br/>REST graph read api_handler/graph/mod.rs:352,<br/>/wss initialGraphLoad socket_flow_handler/types.rs:509"] --> VS["visible_subgraph(graph, pubkey, filter_on)<br/>visibility_filter.rs:171"]
    VS -->|filter off| SAME["graph returned unchanged<br/>visibility_filter.rs:176-178"]
    VS --> KEPT["keep nodes where of_node(n).is_visible_to(caller)<br/>visibility_filter.rs:179-183, of_node :101-115"]
    KEPT -->|"nothing removed"| SAME2["same Arc, no scrub needed<br/>visibility_filter.rs:185-196"]
    KEPT --> PN["PageNames: hidden = removed nodes' metadata_id, label, owl_class_iri,<br/>file_name, page_iri, class_iri, plus orphan metadata rows<br/>visibility_filter.rs:258-276"]
    PN --> IH["is_hidden: in hidden and not shared with a visible page<br/>visibility_filter.rs:282-285"]
    IH --> SN["scrub_node: is_subclass_of and tags only<br/>visibility_filter.rs:367-372, :381-386"]
    IH --> SR["scrub_row: is_subclass_of, belongs_to_domain<br/>visibility_filter.rs:291-292"]
    PN --> TC["topic_counts kept only for VISIBLE page names (allow-list)<br/>visibility_filter.rs:289-290"]
    SN --> OUT["nodes, edges with both ends kept, metadata rows of kept nodes<br/>visibility_filter.rs:200-231"]
    SR --> OUT
    TC --> OUT
    SN -.-> NOT["domain and source_domain left alone by design<br/>visibility_filter.rs:362-366"]
    OUT --> T["test hidden_page_names_are_scrubbed_from_visible_rows_and_nodes<br/>visibility_filter.rs:806"]
```

**Invariant:** a visible node or metadata row never names a removed page in a page-reference key. `scrub_node` covers `is_subclass_of` and `tags` (`crates/visionclaw-domain/src/utils/visibility_filter.rs:372`), and `scrub_row` covers `is_subclass_of`, `belongs_to_domain` and `topic_counts` (`crates/visionclaw-domain/src/utils/visibility_filter.rs:289-292`).

**Tension (risk):** plain `domain` and `source_domain` text is not scrubbed, by design: these keys are frontmatter text, not links, and removing a value that matches a hidden page name would itself reveal the page (`crates/visionclaw-domain/src/utils/visibility_filter.rs:362-366`). A visible page whose `domain` text equals a hidden page's name still shows that name.

**Debt (parallel):** the scrub uses two rules. `topic_counts` keeps only names on the visible allow-list (`crates/visionclaw-domain/src/utils/visibility_filter.rs:289-290`). Every other key removes only names on the hidden block-list (`crates/visionclaw-domain/src/utils/visibility_filter.rs:291-292`, `:379`), so a dangling reference to a page that does not exist survives everywhere except `topic_counts`.

## VC-22.13 Corpus history in the provenance graph (VisionFlow ADR-2022 P2 and P4)

<!-- 2026-11-06 restamp ec785afa9: new. 9fd0e85cc and ab1e636ff record full-sync corpus changes as assertion-versions. ADR-2022 here is the VisionFlow record. VisionClaw's own ADR-2022 is did:nostr hex identity, so it is not listed in adrs -->
```mermaid
sequenceDiagram
    autonumber
    participant RB as rebuild_assert_graph tail<br/>src/services/github_sync_service.rs:1174
    participant VR as verify_and_record_corpus_history<br/>src/services/github_sync_service.rs:1181
    participant CH as corpus_history.verify_and_record<br/>src/services/corpus_history.rs:441
    participant AG as GRAPH_ASSERT
    participant PG as GRAPH_PROVENANCE
    RB->>VR: after the class CLEAR+INSERT and rematerialise_decisions (github_sync_service.rs:1163-1170)
    VR->>VR: resolve_corpus_agent - kill switch, then panel key as did:nostr (corpus_history.rs:112-126)
    VR->>CH: spawn_blocking verify_and_record(store, graph, agent, now)
    CH->>AG: P4 - digest of what the rebuild meant to write vs what assert holds (corpus_history.rs:449-463)
    alt agent Disabled, NoIdentity or Misconfigured
        CH-->>VR: not_recorded_reason set, nothing written (corpus_history.rs:468-484)
    else Ready
        CH->>CH: public_subgraph - NodeVisibility.of_node public only, IRI triples only (corpus_history.rs:140-146, :487)
        CH->>PG: diff against open corpus-sync versions this writer owns (corpus_history.rs:195, :488-489)
        CH->>PG: apply_history - open and close assertion-versions (corpus_history.rs:402, :506)
        CH->>PG: re-read open versions, digest vs public set (corpus_history.rs:514-522)
    end
    CH-->>VR: report - digest, opened, closed, errors, all non-fatal
    Note over CH,PG: INVARIANT private content never enters - the store cannot erase until ADR-2102 lands (corpus_history.rs:22-25)
```

**Invariant:** the corpus-history writer and the wire paths share one public test: `public_subgraph` keeps the nodes `NodeVisibility::of_node` calls public (`src/services/corpus_history.rs:144`, `crates/visionclaw-domain/src/utils/visibility_filter.rs:101`), so private content cannot enter the unerasable provenance graph by a second, diverging rule.

**Open (risk):** the writer records nothing when no panel key is configured (`src/services/corpus_history.rs:474-477`), and surfaces that only as `SyncStatistics.corpus_history_not_recorded` and a warn. Any deployment without an ACSP panel key therefore keeps no corpus history.

## VC-22.14 Graph shares: one durable store, outside erasure

<!-- 2026-11-06 restamp 809a859aa: new. 3a23f4fa5 moves shares from a map per actix worker into data/shares.sqlite3; bcadfb6a5 requires a creator -->
```mermaid
flowchart TB
    subgraph WRITE["Create: POST /api/graph-export/share"]
        SH["share_graph: rate limit only, no sign-in check of its own<br/>src/handlers/graph_export_handler.rs:165, :178-186"]
        SH --> VIEW["snapshot of the caller's own view, private nodes included<br/>src/handlers/graph_export_handler.rs:188"]
        VIEW --> FILE["share file written under data/shared/ first<br/>src/handlers/graph_export_handler.rs:197"]
        FILE --> ROW["row inserted, the file removed if the insert fails<br/>src/handlers/graph_export_handler.rs:204-211"]
        ROW -.->|"no creator"| NC["store refuses: NoCreator, answered as 500<br/>src/adapters/sqlite_share_repository.rs:110-113"]
    end
    subgraph STORE["SqliteShareRepository, data/shares.sqlite3"]
        OPEN["opened once in AppState.new<br/>src/app_state.rs:481-488"]
        TBL["shared_graphs: creator_id, expires_at_ms, access_count<br/>src/adapters/sqlite_share_repository.rs:41-50"]
        CLAIM["claim_access and delete_owned are single statements,<br/>so workers cannot race<br/>src/adapters/sqlite_share_repository.rs:164, :193, :210"]
        OPEN --> TBL --> CLAIM
    end
    ROW --> TBL
    subgraph LIFE["Retention"]
        GONE["expired: answered 410, row and file kept<br/>src/handlers/graph_export_handler.rs:235-238"]
        SWEEP["removed with their files only at the next start-up<br/>src/main.rs:983-999, sqlite_share_repository.rs:243"]
        GONE --> SWEEP
    end
    TBL --> GONE
    TBL --> BK["backup-sqlite.sh backs up shares.sqlite3 as REQUIRED<br/>scripts/backup-sqlite.sh:75"]
    BK -.-> NOFILES["OPEN: data/shared/ files are not in the backup<br/>docs/DATA-authority-erasure.md:131-133"]
    TBL -.-> NOERASE["OPEN: no per-user erasure path reaches shares<br/>docs/DATA-authority-erasure.md:129-131"]
```

**Open (risk):** a share is a copy of its creator's view and can hold the creator's private nodes (`src/handlers/graph_export_handler.rs:188`), it now survives restarts, and no per-user erasure path reaches it, so erasing a user leaves the shares they created and their files (`docs/DATA-authority-erasure.md:114-115`, `:129-131`). A share made without an expiry is kept until its creator deletes it (`docs/DATA-authority-erasure.md:122-123`).

**Debt (loose end):** the refusal worded "Sharing a graph requires a signed-in caller" sits in the plain export handler (`src/handlers/graph_export_handler.rs:142-146`), while `share_graph` has no sign-in check: an anonymous share request snapshots the graph and writes the file before the store refuses it (`src/adapters/sqlite_share_repository.rs:110-113`) and is answered 500, not 401 (`src/handlers/graph_export_handler.rs:205-210`).

**Debt (loose end):** an expired share is refused with 410 but its row and file stay until the next start-up sweep (`src/handlers/graph_export_handler.rs:235-238`, `src/main.rs:983-999`); a long-running server keeps every expired share's private snapshot on disk.

**Drift:** `docs/DATA-authority-erasure.md:120` cites `src/app_state.rs:475-482` for where the share store is built; at this commit those lines are the end of the KPI store opening plus the start of the share store, which is opened at `src/app_state.rs:481-488`.
