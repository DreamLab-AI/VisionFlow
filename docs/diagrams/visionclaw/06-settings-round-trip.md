---
id: VC-06
title: Settings round trip — REST, actors, SQLite adapter and generated client types
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2005, ADR-2011, ADR-2041, ADR-2046, ADR-2047, ADR-2080, ADR-2115, ADR-2140]
sources:
  - ../project/src/settings/api/settings_routes.rs
  - ../project/src/settings/auth_extractor.rs
  - ../project/src/settings/mod.rs
  - ../project/src/settings/models.rs
  - ../project/src/actors/optimized_settings_actor.rs
  - ../project/src/adapters/sqlite_settings_repository.rs
  - ../project/src/ports/settings_repository.rs
  - ../project/src/ports/mod.rs
  - ../project/src/handlers/settings_validation_fix.rs
  - ../project/src/config/mod.rs
  - ../project/src/config/path_accessible_impls.rs
  - ../project/src/bin/generate_types.rs
  - ../project/src/app_state.rs
  - ../project/src/main.rs
  - ../project/src/handlers/nostr_handler.rs
  - ../project/src/middleware/rbac_gate.rs
  - ../project/crates/visionclaw-domain/src/config/graph_type.rs
  - ../project/src/handlers/settings_handler/validation.rs
  - ../project/src/handlers/settings_handler/write_handlers.rs
  - ../project/src/handlers/constraints_handler.rs
  - ../project/src/handlers/api_handler/analytics/sssp_handlers.rs
  - ../project/src/handlers/layout_handler.rs
  - ../project/src/services/nostr_service.rs
  - ../project/client/src/store/websocket/textMessageHandler.ts
  - ../project/client/src/store/settings/settingsHelpers.ts
  - ../project/client/src/features/graph/explore/nodeFilter.ts
  - ../project/client/src/features/graph/explore/__tests__/nodeFilter.test.ts
  - ../project/src/actors/client_filter.rs
  - ../project/xr-client/rust/src/settings_sync.rs
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/docs/BASELINE-architecture.md
  - ../project/AGENTS.md
  - ../project/tests/test_save_endpoint.rs
  - ../project/tests/test_settings_persistence.rs
verified_commit: {visionclaw: 809a859aae9e3d96d4d6b99690378018e48e3ad8}
---

## For developers

This topic follows a settings write from the REST route through `OptimizedSettingsActor`, the GPU actors and the SQLite adapter, then out to other sessions as a `settingsUpdated` broadcast. It also covers the API-key store and the generated client types. Physics has a single write path, `apply_physics_patch` (ADR-2140). Every other bucket is a JSON row under one fixed key.

Three changes since the last stamp. `PUT /api/settings/rendering` now stores a `rendering` row, so lighting and background survive a restart (VC-06.10). A `visual` broadcast also re-reads the visual-bucket paths that sit in no named section. `ProtectedSettingsActor`, its model and both copies are deleted, so `NostrService` is the only API-key store (VC-06.6). Who may write which bucket is decided at the `RbacGate` (VC-03.6).

## For the business

A settings change made in one browser, or by the settings assistant, now reaches every open browser, including look-and-feel changes such as edge opacity. Settings saved on the server are shared. A change one person saves applies to everyone looking at the graph, not only to that person. Only the personal node filter is kept per account.

## VC-06.1 Settings route surface and the actor behind it
```mermaid
flowchart TB
    S["web::scope('/settings') + RateLimit::per_minute(60)<br/>src/main.rs:1176-1178"]
    S --> CFG["settings::api::configure_routes<br/>src/settings/api/settings_routes.rs:1950"]
    CFG --> P["GET|PUT physics settings_routes.rs:1953-1954<br/>POST physics/reset-layout :1955"]
    CFG --> C["GET|PUT constraints settings_routes.rs:1956-1957"]
    CFG --> R["GET|PUT rendering settings_routes.rs:1958-1959"]
    CFG --> NF["GET|PUT node-filter settings_routes.rs:1960-1961"]
    CFG --> QG["GET|PUT quality-gates settings_routes.rs:1962-1963"]
    CFG --> V["GET|PUT visual settings_routes.rs:1964-1965"]
    CFG --> AL["GET all settings_routes.rs:1966"]
    CFG --> PR["POST|GET profiles settings_routes.rs:1967-1968<br/>GET|DELETE profiles/{id} :1969-1970"]
    CFG --> U["nested scope /user<br/>GET|PUT /filter settings_routes.rs:1974-1976"]
    ACT["state.settings_addr : Addr of OptimizedSettingsActor<br/>src/app_state.rs:301, built src/app_state.rs:1072, started :1081"]
    REPO["settings_repo : web::Data of Arc of SqliteSettingsRepository<br/>injected src/main.rs:1116"]
    P --> ACT
    P --> REPO
    N1["INVARIANT — /api/settings writes require the WriteSettings capability at the RbacGate<br/>src/main.rs:1164-1170. Request-time gate behaviour see VC-03.6"]
    S --- N1
    N2["DIVERGENCE — the settings hot-reload watcher is DISABLED, src/app_state.rs:1083-1087<br/>reason recorded in code: it was causing database deadlocks"]
    ACT --- N2
    N3["OptimizedSettingsActor's wider message surface beyond the round trip shown here — ReloadSettings<br/>(optimized_settings_actor.rs:1182) is the dormant hot-reload handler N2's disabled watcher would<br/>have driven, and nothing sends it; GetSettingByPath/GetSettingsByPaths/SetSettingsByPaths (optimized_settings_actor.rs:812,<br/>:843, :902) give granular per-path read/write alongside the full-snapshot GetSettings/UpdateSettings<br/>used above; WarmCacheMessage (optimized_settings_actor.rs:761) and GetPerformanceMetrics/ClearCaches<br/>(optimized_settings_actor.rs:1154, :1169) handle cache warm-up and introspection"]
    ACT --- N3
```

**Invariant:** the actor snapshot is loaded from the repository once, when the actor starts (`../project/src/actors/optimized_settings_actor.rs:700-710`). The only other path that reloads it is `ReloadSettings` (`../project/src/actors/optimized_settings_actor.rs:1201`), and at this revision nothing in `src/` or `crates/` sends that message.

## VC-06.2 PUT /api/settings/physics — the flagship round trip, phase 1 read and merge
```mermaid
sequenceDiagram
    autonumber
    participant CL as client
    participant RG as RbacGate<br/>src/middleware/rbac_gate.rs
    participant H as update_physics_settings<br/>src/settings/api/settings_routes.rs:594-611
    participant AP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:617-622
    participant SA as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs:774
    participant VA as validate_physics_settings and validate_for_gpu<br/>src/settings/api/settings_routes.rs:140, :131

    CL->>RG: PUT /api/settings/physics
    Note over RG: WriteSettings capability required — see VC-03.6
    RG->>H: forward, AuthenticatedUser extracted (src/settings/auth_extractor.rs:70)
    H->>AP: apply_physics_patch(state, repo, body, auth.pubkey) (settings_routes.rs:600-605)
    Note over AP: ADR-2140 decision 3 — the ONE physics write path. POST /sssp/toggle (sssp_handlers.rs:126)<br/>and POST /api/layout/mode (layout_handler.rs:92) call it too (doc settings_routes.rs:613-616)
    AP->>AP: debug-log updated_by and the raw patch (settings_routes.rs:623-626)
    AP->>SA: send(GetSettings)
    Note over AP,SA: single GetSettings call — a full snapshot is fetched ONCE to avoid a TOCTOU race<br/>comment src/settings/api/settings_routes.rs:628
    alt Ok(Ok(settings))
        SA-->>AP: AppFullSettings
    else Ok(Err(e))
        SA-->>AP: Internal "Failed to fetch current settings" (settings_routes.rs:631-637)
    else Err(mailbox)
        SA-->>AP: Internal "Actor communication error" (settings_routes.rs:638-644)
    end
    AP->>AP: current = full_settings.visualisation.graphs.knowledge.physics (settings_routes.rs:648)
    Note over AP: RESOLVED ADR-2041 by ADR-2115 — `knowledge` is the only accepted graph-type<br/>segment — the one-release `logseq` alias has expired and is now rejected everywhere<br/>(crates/visionclaw-domain/src/config/graph_type.rs:1-7). See VC-09.15
    AP->>AP: merge_physics_patch(current, patch) (settings_routes.rs:651, defined :1447-1479)
    Note over AP: normalize_physics_keys maps snake_case and legacy aliases to canonical camelCase (settings_routes.rs:1459). A layoutMode<br/>value is parsed strictly and refused if unknown (settings_routes.rs:1464-1476, ADR-2140). deep_merge_json (settings_routes.rs:1426-1439)<br/>merges NESTED objects recursively (settings_routes.rs:1477), so one nested slider patch no longer resets its sibling fields.<br/>A non-object body leaves the snapshot's physics unchanged (settings_routes.rs:653)
    alt merge fails
        AP-->>CL: 400 "Invalid settings value" (settings_routes.rs:654-659)
    end
    AP->>VA: validate_physics_settings(&new_physics) (settings_routes.rs:664)
    alt validation fails
        VA-->>CL: 400 "Validation failed: ..." (settings_routes.rs:665-669)
    end
    AP->>VA: validate_for_gpu(&new_physics) (settings_routes.rs:674)
    Note over AP,VA: runs SimulationParams::validate BEFORE anything is persisted (doc settings_routes.rs:126-130). The route-level check is<br/>narrower (it never looked at ssspAlpha), so without this a value could pass, be stored, then be refused by the GPU actor<br/>while the response was still 200
    alt GPU validation fails
        VA-->>CL: 400 "Validation failed: GPU rejected these values: ..." (settings_routes.rs:675-679, prefix :136)
    end
    Note over H,AP: a PhysicsPatchError maps to 400 or 500 in into_response (settings_routes.rs:580-586). Continues in VC-06.3
```

## VC-06.3 PUT /api/settings/physics — phase 2 write-back, GPU propagation, persistence
```mermaid
sequenceDiagram
    autonumber
    participant AP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:617
    participant SA as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs:801
    participant GPU as GPUComputeActor<br/>via state.get_gpu_compute_addr()
    participant GM as GPUManagerActor<br/>state.gpu_manager_addr
    participant CC as ClientCoordinatorActor<br/>state.client_manager_addr
    participant GS as GraphServiceSupervisor<br/>state.graph_service_addr
    participant DB as SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs:409

    AP->>AP: full_settings.visualisation.graphs.knowledge.physics = new_physics (settings_routes.rs:683)
    AP->>SA: send(UpdateSettings { settings: full_settings }) (settings_routes.rs:684-689)
    alt Ok(Ok(()))
        AP->>AP: SimulationParams::from(&new_physics) (settings_routes.rs:695-696)
        rect rgb(236,236,244)
        Note over AP,GM: GPU propagation with a startup fallback
        alt state.get_gpu_compute_addr() is Some
            AP->>GPU: send(UpdateSimulationParams { params }) (settings_routes.rs:703-705)
            alt the actor refuses the update
                GPU-->>AP: Internal "GPU update failed" — early return, the snapshot already holds the value (settings_routes.rs:711-717)
            end
        else direct address not yet cached
            Note over AP,GM: fallback window is roughly the first 6s of startup while the async init task completes (settings_routes.rs:723-724)
            alt state.gpu_manager_addr is Some
                AP->>GM: send(UpdateSimulationParams) — routed (settings_routes.rs:726-727)
            else
                AP->>AP: error — physics will NOT propagate to GPU (settings_routes.rs:736)
            end
        end
        opt GPUComputeActor address available
            AP->>GPU: do_send(UpdateClusteringParams { algorithm, resolution, iterations }) (settings_routes.rs:745-750)
            Note over AP,GPU: community-detector params cannot ride in the 172-byte repr-C SimParams<br/>so they are dispatched separately — this is what makes the Physics tab controls live (settings_routes.rs:740-744)
        end
        AP->>CC: do_send(stream_settings_message(&new_physics)) — ConfigureStream with chunk_nodes, move_threshold, ack_timeout_ms (settings_routes.rs:756-758, built :477-483)
        Note over AP,CC: the three stream settings are not SimParams fields either, they go straight to the client coordinator<br/>that builds each client's chunks. If no GPUComputeActor address is cached, clustering parameters wait for its next boot push (settings_routes.rs:752)
        end
        AP->>GS: send(UpdateSimulationParams) (settings_routes.rs:761)
        AP->>GS: send(ForceResumePhysics { reason: "Physics settings updated via API" }) (settings_routes.rs:770-774)
        Note over GS: without ForceResumePhysics a converged system stays paused and the change is invisible (settings_routes.rs:767-768)
        AP->>DB: set_setting("physics", SettingValue::Json(...), Some("Physics simulation settings")) (settings_routes.rs:785-790)
        alt persist fails
            DB-->>AP: warn only — the response still succeeds (settings_routes.rs:793)
        end
        AP->>AP: broadcast_settings_change(state, "physics", updated_by) (settings_routes.rs:807, ADR-2047)
        AP-->>AP: Ok(new_physics), sent as 200 JSON by the route (settings_routes.rs:809, :608)
    else Ok(Err(e))
        SA-->>AP: Internal "Failed to update physics settings" (settings_routes.rs:811-816)
    else Err(mailbox)
        SA-->>AP: Internal "Actor communication error" (settings_routes.rs:818-823)
    end
    Note over AP,DB: RESOLVED ADR-2047 (was DIVERGENCE) — physics now DOES broadcast to other<br/>sessions via broadcast_settings_change (settings_routes.rs:807, comment :804-806 "the one that never announced<br/>itself"). Rendering (settings_routes.rs:1030) also calls broadcast_settings_change — node-filter uses a<br/>separate raw client_manager_addr.do_send(BroadcastMessage) instead (settings_routes.rs:1167-1170), same effect. See VC-06.4
```

**Tension:** a GPU actor that refuses the update makes `apply_physics_patch` return 500 "GPU update failed" (`../project/src/settings/api/settings_routes.rs:711-717`), but `UpdateSettings` has already replaced the actor snapshot (`../project/src/settings/api/settings_routes.rs:684-691`). The early return skips the SQLite write (`:785-790`) and the broadcast (`:807`). The in-memory snapshot then holds a value the GPU refused and the store never saw, and the next PATCH merges onto it. Source tracing cannot show how often the actor refuses a value that has already passed `validate_for_gpu` (`:674`).

## VC-06.4 Broadcast asymmetry — which settings reach other sessions
```mermaid
sequenceDiagram
    autonumber
    participant H1 as update_rendering_settings<br/>src/settings/api/settings_routes.rs:972
    participant H2 as update_node_filter_settings<br/>src/settings/api/settings_routes.rs:1098
    participant H3 as apply_physics_patch<br/>src/settings/api/settings_routes.rs:617
    participant H4 as reset_layout<br/>src/settings/api/settings_routes.rs:1843
    participant H5 as update_visual_settings<br/>src/settings/api/settings_routes.rs:1503
    participant BC as broadcast_settings_change<br/>src/settings/api/settings_routes.rs:453-469
    participant CC as ClientCoordinatorActor<br/>state.client_manager_addr
    participant B as other browser sessions<br/>client/src/store/websocket/textMessageHandler.ts:139

    H1->>BC: "rendering" (settings_routes.rs:1030), after the rendering row is stored (:1004-1016), see VC-06.10
    H3->>BC: "physics" (settings_routes.rs:807), from PUT physics, SSSP toggle and layout mode
    H4->>BC: "physics", "reset-layout", only after ResetPositions succeeded (settings_routes.rs:1927)
    H5->>BC: "visual" after the row is written (settings_routes.rs:1528-1546), new in ec785afa9
    BC->>CC: do_send(BroadcastMessage) (settings_routes.rs:462-464), category-only payload (:454-459)
    CC->>B: WS frame — other sessions converge
    H2->>CC: own inline do_send(BroadcastMessage) (settings_routes.rs:1167-1170), bypasses the emitter
    Note over H2,CC: payload built at settings_routes.rs:1147-1166, fuller than BC's — echoes the actual settings fields,<br/>not just category. Comment "Propagate node filter changes..." at settings_routes.rs:1144-1146
    CC->>B: WS frame
    B->>B: handleSettingsUpdated (textMessageHandler.ts:155)
    Note over B: own echo dropped (textMessageHandler.ts:163), stale timestamp dropped (:172),<br/>nodeFilter applied directly (:181-189), others re-read (:191, :195-218)
    Note over B: "visual" re-reads every visualisation path outside rendering and physics (client/src/store/settings/settingsHelpers.ts:251-263)
    Note over H1,B: RESOLVED ADR-2047 — physics (PUT, reset-layout and the other apply_physics_patch callers) and rendering now ALL emit<br/>through the single broadcast_settings_change emitter (settings_routes.rs:453-469). node-filter keeps its own richer, separate<br/>broadcast. RESOLVED ADR-2080 (vc-clients) — the client now CONSUMES settingsUpdated (textMessageHandler.ts:139-141).<br/>Visual joined in ec785afa9. Still silent: constraints, quality-gates, profiles.
```

**Open:** `reset_layout` writes the canonical defaults to the GPU actors and to SQLite (`../project/src/settings/api/settings_routes.rs:1867-1907`) but never sends `UpdateSettings` to `OptimizedSettingsActor`. `GET /api/settings/physics` reads SQLite first (`:491-496`), so it shows the reset. The next `apply_physics_patch` merges onto the actor snapshot instead (`:629`, `:648`), and that snapshot still holds the pre-reset values until the process restarts, because nothing sends `ReloadSettings`. On source evidence, the first physics PUT after a reset writes every field it does not name back to its pre-reset value. This needs a runtime check to confirm.

**Tension (risk):** every server settings bucket is written under one constant key with no owner scope: `physics` (`../project/src/settings/api/settings_routes.rs:786-787`), `rendering` (`:1005-1006`), `constraints` (`:893-894`), `node_filter` (`:1131-1132`), `quality_gates` (`:1279-1280`) and `visual` (`:1528-1529`). They land in the global layer, so one viewer's save changes every viewer's settings. BASELINE records this as intended (`../project/docs/BASELINE-architecture.md:362-364`). Only `/api/user/filter` is scoped to its owner (`:1745-1751`).

**Drift (code comment vs code):** `CURRENT_OWNER_PUBKEY` is documented as "set by NIP-98 middleware before any handler runs" (`../project/src/adapters/sqlite_settings_repository.rs:119-120`). No middleware sets it. Only `get_all_settings`, the two user-filter handlers and the WebSocket filter save scope it (`:1571`, `:1672`, `:1745`, `../project/src/handlers/socket_flow_handler/filter_auth.rs:382`).

## VC-06.6 API keys — one NostrService store
```mermaid
sequenceDiagram
    autonumber
    participant HP as POST /api/auth/nostr/api-keys<br/>src/handlers/nostr_handler.rs:277
    participant HG as GET /api/auth/nostr/api-keys<br/>src/handlers/nostr_handler.rs:310
    participant NS as NostrService users map<br/>src/services/nostr_service.rs:479

    HP->>NS: update_user_api_keys(caller, keys) (nostr_handler.rs:293, nostr_service.rs:484)
    NS->>NS: user.api_keys = api_keys, persisted to Redis when the feature is on (nostr_service.rs:495-502)
    HG->>NS: get_user(caller) (nostr_handler.rs:319)
    alt user known
        NS-->>HG: 200 user.api_keys with Cache-Control no-store (nostr_handler.rs:320-326)
    else
        NS-->>HG: 404 User not found (nostr_handler.rs:328)
    end
    Note over HP,NS: RESOLVED since 86cdd7635: the POST and the GET use the same NostrService store
    Note over HP,NS: RESOLVED (VisionClaw 7a0674e79): ProtectedSettingsActor (both copies), its ProtectedSettings model,<br/>AppState::get_api_keys and get_nostr_user are deleted, so no second key store is left in src or crates
```

**What it shows.** Saving and reading a user's API keys. **Why it is this way.** The api-keys GET once read an empty `ProtectedSettingsActor`, so a saved key never came back. It now reads the same `NostrService` record the POST writes, the route test pins that (`../project/src/handlers/nostr_handler.rs:370`), and the dormant actor has since been deleted.

## VC-06.7 Hexagonal hop — SettingsRepository port to the SQLite adapter
```mermaid
sequenceDiagram
    autonumber
    participant A as caller<br/>route or OptimizedSettingsActor
    participant P as SettingsRepository trait<br/>src/ports/settings_repository.rs:8
    participant AD as SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs:372
    participant DB as SQLite via tokio_rusqlite

    Note over P: ADR-090 A6 slice 3 — src/ports/settings_repository.rs is a 14-line SHIM.<br/>The canonical trait lives in visionclaw_domain::ports::settings_repository. See VC-07.
    A->>P: get_setting(key)
    P->>AD: get_setting (sqlite_settings_repository.rs:376)
    AD->>DB: SELECT (sqlite_settings_repository.rs:386)
    AD->>AD: decode_setting_value (sqlite_settings_repository.rs:150) — JSON text to SettingValue
    alt row absent
        AD-->>A: Ok(None)
    else decode error
        AD-->>A: map_json_err (sqlite_settings_repository.rs:145) to SettingsRepositoryError
    end
    A->>P: set_setting(key, value, description)
    P->>AD: set_setting (sqlite_settings_repository.rs:409)
    AD->>AD: encode_setting_value (sqlite_settings_repository.rs:155), current_owner_pubkey() (:128)
    AD->>DB: UPSERT
    par other trait methods
        A->>AD: delete_setting (sqlite_settings_repository.rs:445)
    and
        A->>AD: has_setting (sqlite_settings_repository.rs:466)
    and
        A->>AD: get_settings_batch (sqlite_settings_repository.rs:487)
    and
        A->>AD: set_settings_batch (sqlite_settings_repository.rs:572)
    and
        A->>AD: list_settings(prefix) (sqlite_settings_repository.rs:610)
    and
        A->>AD: load_all_settings (sqlite_settings_repository.rs:635) — seeds physics from the physics row (:708, seed_physics_from_blob :1071)<br/>and rendering from the rendering row (:713, seed_rendering_from_blob :1061)
    end
    Note over AD: flatten_json (sqlite_settings_repository.rs:165) / unflatten_pairs (:188) / insert_at_path (:196)<br/>store settings as flat dotted key rows and rebuild the nested JSON on read
    Note over AD,DB: adapter-only surface beyond the port — open (sqlite_settings_repository.rs:241), from_connection (:255), connection (:261),<br/>get_file_sha1s (:270), upsert_file_sha1s (:289), get_sync_config (:317), set_sync_config (:338),<br/>clear_sync_metadata (:359). The corpus-sync SHA1 ledger shares this database. See VC-21.
    Note over AD,DB: errors are normalised by map_db_err (sqlite_settings_repository.rs:140) and map_json_err (:145), the former map_rusqlite_err helper no longer exists
```

**Invariant:** ADR-2140 decision 1 makes the stored `physics` row the source of truth for the snapshot's physics (`../project/src/adapters/sqlite_settings_repository.rs:1046-1054`). The doc comment says `UpdateSettings` persists nothing and `save_all_settings` has no live caller, which is why `apply_physics_patch` writes the row itself.

## VC-06.8 Validation and field-mapping helpers
```mermaid
flowchart TB
    IN["incoming settings JSON"]
    IN --> N1["normalize_physics_keys<br/>src/settings/api/settings_routes.rs:74<br/>snake_case and legacy names to canonical camelCase, call site inside merge_physics_patch :1459"]
    N1 --> V1["validate_physics_settings<br/>src/settings/api/settings_routes.rs:140 (call site :664)"]
    V1 --> G1["validate_for_gpu<br/>src/settings/api/settings_routes.rs:131 (call site :674)<br/>SimulationParams::validate before persistence"]
    G1 --> OUT["PhysicsSettings"]
    subgraph FIX["src/handlers/settings_validation_fix.rs"]
        F1["validate_physics_settings_complete(&Value) settings_validation_fix.rs:5"]
        F2["validate_constraint(&Value) settings_validation_fix.rs:67"]
        F3["convert_to_snake_case_recursive(&mut Value) settings_validation_fix.rs:107"]
        F4["get_complete_field_mappings() -> HashMap settings_validation_fix.rs:151"]
        F5["apply_field_mappings(&mut Value, &mappings) settings_validation_fix.rs:271"]
    end
    subgraph DOM["canonical validators re-exported by src/config/mod.rs:28-31"]
        D1["validate_bloom_glow_settings"]
        D2["validate_hex_color"]
        D3["validate_percentage"]
        D4["validate_port"]
        D5["validate_width_range"]
    end
    V1 -.-> DOM
    FIX -.-> IN
    NOTE["the two families are parallel — settings_routes.rs carries its own normalise/validate pair<br/>(plus validate_for_gpu) and the live physics write path apply_physics_patch calls those only.<br/>settings_validation_fix.rs is imported by other handlers: settings_handler/validation.rs:7,<br/>settings_handler/write_handlers.rs:12 and constraints_handler.rs:9"]
    FIX --- NOTE
```

**Invariant:** the node-filter defaults (off, quality 0.4, authority 0.5, OR) are written in four places, the server model (`../project/src/settings/models.rs:163-182`), the client (`../project/client/src/features/graph/explore/nodeFilter.ts:57-68`), the headset (`../project/xr-client/rust/src/settings_sync.rs:44-55`) and the `filter_update` parser's fallbacks (`../project/src/handlers/socket_flow_handler/filter_auth.rs:17`), and each is now pinned to the parity fixture's default case (`../project/src/actors/client_filter.rs:406`, `../project/client/src/features/graph/explore/__tests__/nodeFilter.test.ts:58`, `../project/xr-client/rust/src/settings_sync.rs:653-654`).

**Open:** `minConnections` and `minMaturity` are stored on the server (`../project/src/settings/models.rs:145-154`) and applied by the desktop client (`../project/client/src/features/graph/explore/nodeFilter.ts:66-67`). The headset `NodeFilter` has no such fields and documents them as desktop-only (`../project/xr-client/rust/src/settings_sync.rs:31-42`), so a floor one viewer saves greys nodes on every desktop and on no headset. No record says whether that split is intended.

## VC-06.9 Generated client types — src/bin/generate_types.rs
```mermaid
sequenceDiagram
    autonumber
    participant B as generate_types::main<br/>src/bin/generate_types.rs:9
    participant T as generate_typescript_interfaces<br/>src/bin/generate_types.rs:64
    participant O as write_ontology_agent_types<br/>src/bin/generate_types.rs:767
    participant F as client/src/types/generated/settings.ts
    participant G as client/src/types/generated/ontology-agent.ts

    B->>T: build the settings interfaces
    T-->>B: a hand-written TypeScript string literal (generate_types.rs:64-569), snake_case fields
    B->>B: convert_to_camel_case(header + interfaces) (generate_types.rs:19, regex pass :777)
    B->>F: fs::write(output_path, &camel_case_code) (src/bin/generate_types.rs:30)
    Note over B,F: output_path is the literal "client/src/types/generated/settings.ts" (generate_types.rs:21)<br/>the parent directory is created if absent (generate_types.rs:22-28)
    B->>F: fs::metadata(output_path) then log the byte size (generate_types.rs:37-38)
    B->>O: write_ontology_agent_types() (generate_types.rs:57)
    O->>G: fs::write — enum unions from the visionclaw_ontology variants, snake_case wire names kept (generate_types.rs:768-772, :615-620)
    Note over T,F: RESOLVED ADR-2041 by ADR-2115 — the settings interfaces carry `knowledge` only (generate_types.rs:284-285),<br/>`logseq` is gone. path_accessible_impls now matches `knowledge` alone, no alias arm<br/>(src/config/path_accessible_impls.rs:159 and :183). Full alias lifecycle see VC-09.15
    Note over F: DRIFT — the output files are gitignored (.gitignore:151 `generated`) and absent from the tree at this stamp,<br/>while project AGENTS.md:13 says generate_types updates client/src/types/ "from Rust structs".<br/>Only the ontology-agent file is derived from Rust values. The settings file is a string the binary carries
    Note over B: server-side YAML is snake_case, the JSON and TS surface is camelCase —<br/>the serde camelCase output is checked and LOGGED at boot, not enforced, src/main.rs:306-346
```

**Drift:** `AGENTS.md:13` (`../project/AGENTS.md:13`) says `cargo run --bin generate_types` "updates `client/src/types/` from Rust structs". The settings half does not read any Rust struct. `generate_typescript_interfaces` returns a fixed raw string (`../project/src/bin/generate_types.rs:64-569`), so a field added to `PhysicsSettings` does not reach `settings.ts` unless someone edits that string by hand. Only `ontology-agent.ts` is built from Rust values (`../project/src/bin/generate_types.rs:575-596`). Even there, the variant tables are kept in step by hand (`:571-574`).

**Debt:** the boot "serde alias" block in `main.rs` (`../project/src/main.rs:314-346`) logs a confirmation when the rendering JSON is camelCase, and logs an error when serialisation fails. Neither outcome stops boot, so "asserted at boot" in the earlier panel was too strong.

## VC-06.10 Rendering row persistence and the visual-section re-read
```mermaid
sequenceDiagram
    autonumber
    participant H as update_rendering_settings<br/>src/settings/api/settings_routes.rs:972
    participant AD as SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs:372
    participant ACT as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs:700
    participant B as other browser sessions<br/>client/src/store/websocket/textMessageHandler.ts:195

    H->>H: validate_rendering_settings, 400 on failure (settings_routes.rs:983-988)
    H->>ACT: GetSettings, then the snapshot's rendering is replaced (settings_routes.rs:990-992)
    H->>AD: set_setting(RENDERING_ROW, Json(row)) (settings_routes.rs:1004-1010), 500 on failure (:1011-1016)
    H->>ACT: UpdateSettings(full_settings) (settings_routes.rs:1017-1023)
    H->>B: broadcast_settings_change rendering (settings_routes.rs:1030)
    Note over H,ACT: TENSION: the row is stored before UpdateSettings, so an actor error returns 500<br/>(settings_routes.rs:1034-1045) while the store already holds the new document, which the next restart loads
    Note over ACT,AD: on restart, started() calls load_all_settings (optimized_settings_actor.rs:710),<br/>which seeds rendering from the row (sqlite_settings_repository.rs:709-713, seed_rendering_from_blob :1061)
    Note over ACT: an all-zero rendering document is replaced by RenderingSettings defaults (optimized_settings_actor.rs:715-722)
    Note over AD: DEBT: the RENDERING_ROW doc lines sit directly under the ADR-2140 doc comment of seed_physics_from_blob<br/>(sqlite_settings_repository.rs:1046-1057), so rustdoc attaches that comment to the const and the fn at :1071 has none
    B->>B: refetchSettingsCategory drops the 2 s cache first, invalidateSettingsCache (textMessageHandler.ts:203)
    B->>B: visual re-reads every visual-bucket path, named sections plus VISUAL_ONLY_SECTION_PATHS<br/>(client/src/store/settings/settingsHelpers.ts:254-263, list :161-165)
```

**What it shows.** A rendering change now survives a restart, and a `visual` broadcast reaches the three visual-bucket subtrees that belong to no named section. **Why it is this way.** The settings audit of 2026-11-06 found that `UpdateSettings` persists nothing, so lighting and background were lost on every restart. The handler now writes the row itself, as `apply_physics_patch` does for physics (`../project/src/adapters/sqlite_settings_repository.rs:1055-1057`).

**Debt (loose end):** two integration test files are entirely commented out: `tests/test_save_endpoint.rs` says it references removed modules (`../project/tests/test_save_endpoint.rs:1-3`), and `tests/test_settings_persistence.rs` says the same (`../project/tests/test_settings_persistence.rs:1-3`); each body is one block comment. Neither test covers any settings route.
