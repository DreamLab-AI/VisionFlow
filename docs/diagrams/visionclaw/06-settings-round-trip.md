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
  - ../project/src/actors/protected_settings_actor.rs
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
  - ../project/crates/visionclaw-domain/src/models/protected_settings.rs
  - ../project/crates/visionclaw-actors/src/protected_settings_actor.rs
  - ../project/client/src/store/websocket/textMessageHandler.ts
  - ../project/AGENTS.md
verified_commit: 86cdd76355823a51bb64d503c005fe277726e724
---

## VC-06.1 Settings route surface and the actor behind it
```mermaid
flowchart TB
    S["web::scope('/settings') + RateLimit::per_minute(60)<br/>src/main.rs:1143-1145"]
    S --> CFG["settings::api::configure_routes<br/>src/settings/api/settings_routes.rs:1906"]
    CFG --> P["GET|PUT physics settings_routes.rs:1909-1910<br/>POST physics/reset-layout :1911"]
    CFG --> C["GET|PUT constraints settings_routes.rs:1912-1913"]
    CFG --> R["GET|PUT rendering settings_routes.rs:1914-1915"]
    CFG --> NF["GET|PUT node-filter settings_routes.rs:1916-1917"]
    CFG --> QG["GET|PUT quality-gates settings_routes.rs:1918-1919"]
    CFG --> V["GET|PUT visual settings_routes.rs:1920-1921"]
    CFG --> AL["GET all settings_routes.rs:1922"]
    CFG --> PR["POST|GET profiles settings_routes.rs:1923-1924<br/>GET|DELETE profiles/{id} :1925-1926"]
    CFG --> U["nested scope /user<br/>GET|PUT /filter settings_routes.rs:1930-1932"]
    ACT["state.settings_addr : Addr of OptimizedSettingsActor<br/>src/app_state.rs:325, built src/app_state.rs:1114, started :1123"]
    REPO["settings_repo : web::Data of Arc of SqliteSettingsRepository<br/>injected src/main.rs:1083"]
    P --> ACT
    P --> REPO
    N1["INVARIANT — /api/settings writes require the WriteSettings capability at the RbacGate<br/>src/main.rs:1131-1137. Request-time gate behaviour see VC-03.6"]
    S --- N1
    N2["DIVERGENCE — the settings hot-reload watcher is DISABLED, src/app_state.rs:1125-1129<br/>reason recorded in code: it was causing database deadlocks"]
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
    participant H as update_physics_settings<br/>src/settings/api/settings_routes.rs:583-600
    participant AP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:606-611
    participant SA as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs:774
    participant VA as validate_physics_settings and validate_for_gpu<br/>src/settings/api/settings_routes.rs:137, :128

    CL->>RG: PUT /api/settings/physics
    Note over RG: WriteSettings capability required — see VC-03.6
    RG->>H: forward, AuthenticatedUser extracted (src/settings/auth_extractor.rs:70)
    H->>AP: apply_physics_patch(state, repo, body, auth.pubkey) (settings_routes.rs:589-594)
    Note over AP: ADR-2140 decision 3 — the ONE physics write path. POST /sssp/toggle (sssp_handlers.rs:126)<br/>and POST /api/layout/mode (layout_handler.rs:92) call it too (doc settings_routes.rs:602-605)
    AP->>AP: debug-log updated_by and the raw patch (settings_routes.rs:612-615)
    AP->>SA: send(GetSettings)
    Note over AP,SA: single GetSettings call — a full snapshot is fetched ONCE to avoid a TOCTOU race<br/>comment src/settings/api/settings_routes.rs:617
    alt Ok(Ok(settings))
        SA-->>AP: AppFullSettings
    else Ok(Err(e))
        SA-->>AP: Internal "Failed to fetch current settings" (settings_routes.rs:620-626)
    else Err(mailbox)
        SA-->>AP: Internal "Actor communication error" (settings_routes.rs:627-633)
    end
    AP->>AP: current = full_settings.visualisation.graphs.knowledge.physics (settings_routes.rs:637)
    Note over AP: RESOLVED ADR-2041 by ADR-2115 — `knowledge` is the only accepted graph-type<br/>segment — the one-release `logseq` alias has expired and is now rejected everywhere<br/>(crates/visionclaw-domain/src/config/graph_type.rs:1-7). See VC-09.15
    AP->>AP: merge_physics_patch(current, patch) (settings_routes.rs:640, defined :1407-1439)
    Note over AP: normalize_physics_keys maps snake_case and legacy aliases to canonical camelCase (settings_routes.rs:1419). A layoutMode<br/>value is parsed strictly and refused if unknown (settings_routes.rs:1424-1436, ADR-2140). deep_merge_json (settings_routes.rs:1386-1399)<br/>merges NESTED objects recursively (settings_routes.rs:1437), so one nested slider patch no longer resets its sibling fields.<br/>A non-object body leaves the snapshot's physics unchanged (settings_routes.rs:642)
    alt merge fails
        AP-->>CL: 400 "Invalid settings value" (settings_routes.rs:643-648)
    end
    AP->>VA: validate_physics_settings(&new_physics) (settings_routes.rs:653)
    alt validation fails
        VA-->>CL: 400 "Validation failed: ..." (settings_routes.rs:654-658)
    end
    AP->>VA: validate_for_gpu(&new_physics) (settings_routes.rs:663)
    Note over AP,VA: runs SimulationParams::validate BEFORE anything is persisted (doc settings_routes.rs:123-127). The route-level check is<br/>narrower (it never looked at ssspAlpha), so without this a value could pass, be stored, then be refused by the GPU actor<br/>while the response was still 200
    alt GPU validation fails
        VA-->>CL: 400 "Validation failed: GPU rejected these values: ..." (settings_routes.rs:664-668, prefix :133)
    end
    Note over H,AP: a PhysicsPatchError maps to 400 or 500 in into_response (settings_routes.rs:569-575). Continues in VC-06.3
```

## VC-06.3 PUT /api/settings/physics — phase 2 write-back, GPU propagation, persistence
```mermaid
sequenceDiagram
    autonumber
    participant AP as apply_physics_patch<br/>src/settings/api/settings_routes.rs:606
    participant SA as OptimizedSettingsActor<br/>src/actors/optimized_settings_actor.rs:801
    participant GPU as GPUComputeActor<br/>via state.get_gpu_compute_addr()
    participant GM as GPUManagerActor<br/>state.gpu_manager_addr
    participant CC as ClientCoordinatorActor<br/>state.client_manager_addr
    participant GS as GraphServiceSupervisor<br/>state.graph_service_addr
    participant DB as SqliteSettingsRepository<br/>src/adapters/sqlite_settings_repository.rs:409

    AP->>AP: full_settings.visualisation.graphs.knowledge.physics = new_physics (settings_routes.rs:672)
    AP->>SA: send(UpdateSettings { settings: full_settings }) (settings_routes.rs:673-678)
    alt Ok(Ok(()))
        AP->>AP: SimulationParams::from(&new_physics) (settings_routes.rs:684-685)
        rect rgb(236,236,244)
        Note over AP,GM: GPU propagation with a startup fallback
        alt state.get_gpu_compute_addr() is Some
            AP->>GPU: send(UpdateSimulationParams { params }) (settings_routes.rs:692-694)
            alt the actor refuses the update
                GPU-->>AP: Internal "GPU update failed" — early return, the snapshot already holds the value (settings_routes.rs:700-706)
            end
        else direct address not yet cached
            Note over AP,GM: fallback window is roughly the first 6s of startup while the async init task completes (settings_routes.rs:712-713)
            alt state.gpu_manager_addr is Some
                AP->>GM: send(UpdateSimulationParams) — routed (settings_routes.rs:715-716)
            else
                AP->>AP: error — physics will NOT propagate to GPU (settings_routes.rs:725)
            end
        end
        opt GPUComputeActor address available
            AP->>GPU: do_send(UpdateClusteringParams { algorithm, resolution, iterations }) (settings_routes.rs:734-739)
            Note over AP,GPU: community-detector params cannot ride in the 172-byte repr-C SimParams<br/>so they are dispatched separately — this is what makes the Physics tab controls live (settings_routes.rs:729-733)
        end
        AP->>CC: do_send(stream_settings_message(&new_physics)) — ConfigureStream with chunk_nodes, move_threshold, ack_timeout_ms (settings_routes.rs:745-747, built :466-472)
        Note over AP,CC: the three stream settings are not SimParams fields either, they go straight to the client coordinator<br/>that builds each client's chunks. If no GPUComputeActor address is cached, clustering parameters wait for its next boot push (settings_routes.rs:741)
        end
        AP->>GS: send(UpdateSimulationParams) (settings_routes.rs:750)
        AP->>GS: send(ForceResumePhysics { reason: "Physics settings updated via API" }) (settings_routes.rs:759-763)
        Note over GS: without ForceResumePhysics a converged system stays paused and the change is invisible (settings_routes.rs:756-757)
        AP->>DB: set_setting("physics", SettingValue::Json(...), Some("Physics simulation settings")) (settings_routes.rs:774-779)
        alt persist fails
            DB-->>AP: warn only — the response still succeeds (settings_routes.rs:782)
        end
        AP->>AP: broadcast_settings_change(state, "physics", updated_by) (settings_routes.rs:796, ADR-2047)
        AP-->>AP: Ok(new_physics), sent as 200 JSON by the route (settings_routes.rs:798, :597)
    else Ok(Err(e))
        SA-->>AP: Internal "Failed to update physics settings" (settings_routes.rs:800-805)
    else Err(mailbox)
        SA-->>AP: Internal "Actor communication error" (settings_routes.rs:807-812)
    end
    Note over AP,DB: RESOLVED ADR-2047 (was DIVERGENCE) — physics now DOES broadcast to other<br/>sessions via broadcast_settings_change (settings_routes.rs:796, comment :793-795 "the one that never announced<br/>itself"). Rendering (settings_routes.rs:994) also calls broadcast_settings_change — node-filter uses a<br/>separate raw client_manager_addr.do_send(BroadcastMessage) instead (settings_routes.rs:1126-1129), same effect. See VC-06.4
```

**Tension:** a GPU actor that refuses the update makes `apply_physics_patch` return 500 "GPU update failed" (`../project/src/settings/api/settings_routes.rs:700-706`), but `UpdateSettings` has already replaced the actor snapshot (`../project/src/settings/api/settings_routes.rs:673-680`). The early return skips the SQLite write (`:774-779`) and the broadcast (`:796`). The in-memory snapshot then holds a value the GPU refused and the store never saw, and the next PATCH merges onto it. Source tracing cannot show how often the actor refuses a value that has already passed `validate_for_gpu` (`:663`).

## VC-06.4 Broadcast asymmetry — which settings reach other sessions
```mermaid
sequenceDiagram
    autonumber
    participant H1 as update_rendering_settings<br/>src/settings/api/settings_routes.rs:961
    participant H2 as update_node_filter_settings<br/>src/settings/api/settings_routes.rs:1062
    participant H3 as apply_physics_patch<br/>src/settings/api/settings_routes.rs:606
    participant H4 as reset_layout<br/>src/settings/api/settings_routes.rs:1799
    participant BC as broadcast_settings_change<br/>src/settings/api/settings_routes.rs:442-458
    participant CC as ClientCoordinatorActor<br/>state.client_manager_addr
    participant B as other browser sessions<br/>client/src/store/websocket/textMessageHandler.ts:138

    H1->>BC: broadcast_settings_change(state, "rendering", pubkey) (settings_routes.rs:994)
    H3->>BC: broadcast_settings_change(state, "physics", updated_by) (settings_routes.rs:796) — PUT physics, SSSP toggle and layout mode all reach it
    H4->>BC: broadcast_settings_change(state, "physics", "reset-layout") (settings_routes.rs:1883) — sent only after ResetPositions succeeded
    BC->>CC: do_send(BroadcastMessage { message }) (settings_routes.rs:451-453) — category-only payload built :443-448, no settings echoed
    CC->>B: WS frame — other sessions converge
    H2->>CC: do_send(BroadcastMessage { message }) (settings_routes.rs:1126-1129) — OWN inline call, bypasses broadcast_settings_change
    Note over H2,CC: payload built at settings_routes.rs:1111-1125, fuller than BC's — echoes the actual settings fields,<br/>not just category. Comment "Propagate node filter changes..." at settings_routes.rs:1108-1110
    CC->>B: WS frame
    B->>B: handleSettingsUpdated (textMessageHandler.ts:154) — own echo dropped (:162), stale timestamp dropped (:171), nodeFilter applied directly (:180-188), others re-read (:190, :194-215)
    Note over H1,B: RESOLVED ADR-2047 — physics (PUT, reset-layout and the other apply_physics_patch callers) and rendering now ALL emit<br/>through the single broadcast_settings_change emitter (settings_routes.rs:442-458). node-filter keeps its own richer, separate<br/>broadcast. RESOLVED ADR-2080 (vc-clients) — the client now CONSUMES settingsUpdated (textMessageHandler.ts:138-140).<br/>Still silent by choice: constraints, quality-gates, visual, profiles.
```

**Open:** `reset_layout` writes the canonical defaults to the GPU actors and to SQLite (`../project/src/settings/api/settings_routes.rs:1823-1863`) but never sends `UpdateSettings` to `OptimizedSettingsActor`. `GET /api/settings/physics` reads SQLite first (`:480-485`), so it shows the reset. The next `apply_physics_patch` merges onto the actor snapshot instead (`:618`, `:637`), and that snapshot still holds the pre-reset values until the process restarts, because nothing sends `ReloadSettings`. On source evidence, the first physics PUT after a reset writes every field it does not name back to its pre-reset value. This needs a runtime check to confirm.

## VC-06.6 ProtectedSettingsActor — API keys, client tokens and user records
```mermaid
sequenceDiagram
    autonumber
    participant HG as GET /api/auth/nostr/api-keys<br/>src/handlers/nostr_handler.rs:310
    participant HP as POST /api/auth/nostr/api-keys<br/>src/handlers/nostr_handler.rs:277
    participant AS as AppState::get_api_keys<br/>src/app_state.rs:1578
    participant PS as ProtectedSettingsActor<br/>src/actors/protected_settings_actor.rs:7
    participant NS as NostrService users map<br/>src/services/nostr_service.rs:484

    Note over PS: started with ProtectedSettings::default() — src/app_state.rs:1157-1159, Addr src/app_state.rs:326
    HG->>AS: state.get_api_keys(caller) (nostr_handler.rs:319)
    AS->>PS: send(GetApiKeys) (app_state.rs:1579-1583)
    PS-->>AS: settings.get_api_keys(pubkey) (src/actors/protected_settings_actor.rs:33-38)
    Note over PS: the users map is empty from boot (Default impl crates/visionclaw-domain/src/models/protected_settings.rs:132),<br/>so this answers default_api_keys (crates/visionclaw-domain/src/models/protected_settings.rs:223-236)
    HP->>NS: nostr_service.update_user_api_keys(caller, keys) (nostr_handler.rs:293)
    NS->>NS: user.api_keys = api_keys, persisted to Redis when the feature is on (nostr_service.rs:495-502)
    Note over HP,NS: the POST writes NostrService, the GET reads ProtectedSettingsActor — the two stores never meet
    Note over PS: handlers with NO sender anywhere in src or crates at this revision: ValidateClientToken (src/actors/protected_settings_actor.rs:49),<br/>StoreClientToken (:65), UpdateUserApiKeys (:81), CleanupExpiredTokens (:97), MergeSettings (:112),<br/>SaveSettings (:127), GetUser (:142)
    Note over PS: DEBT — a second copy of the actor lives at crates/visionclaw-actors/src/protected_settings_actor.rs:7,<br/>AppState starts the src/actors copy. Session-token realm and get_session expiry see VC-03.5 and VC-05
```

**Tension:** `POST /api/auth/nostr/api-keys` stores keys on the `NostrService` user record (`../project/src/handlers/nostr_handler.rs:293`, `../project/src/services/nostr_service.rs:495`). `GET` on the same resource reads `ProtectedSettingsActor` (`../project/src/handlers/nostr_handler.rs:319`, `../project/src/app_state.rs:1578-1586`), which starts from `ProtectedSettings::default()` (`../project/src/app_state.rs:1159`), whose `users` map is empty (`../project/crates/visionclaw-domain/src/models/protected_settings.rs:132`), and has no writer. On source evidence a non-power user never reads back the keys they saved. This needs a runtime check to confirm.

**Drift:** the panel this replaces drew the api-keys routes driving all eight actor messages. At this revision only `GetApiKeys` has a sender, and seven handlers are unreachable.

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
        A->>AD: load_all_settings (sqlite_settings_repository.rs:635) — seeds the snapshot physics from the physics row (:708, seed_physics_from_blob :1050)
    end
    Note over AD: flatten_json (sqlite_settings_repository.rs:165) / unflatten_pairs (:188) / insert_at_path (:196)<br/>store settings as flat dotted key rows and rebuild the nested JSON on read
    Note over AD,DB: adapter-only surface beyond the port — open (sqlite_settings_repository.rs:241), from_connection (:255), connection (:261),<br/>get_file_sha1s (:270), upsert_file_sha1s (:289), get_sync_config (:317), set_sync_config (:338),<br/>clear_sync_metadata (:359). The corpus-sync SHA1 ledger shares this database. See VC-21.
    Note over AD,DB: errors are normalised by map_db_err (sqlite_settings_repository.rs:140) and map_json_err (:145), the former map_rusqlite_err helper no longer exists
```

**Invariant:** ADR-2140 decision 1 makes the stored `physics` row the source of truth for the snapshot's physics (`../project/src/adapters/sqlite_settings_repository.rs:1041-1049`). The doc comment says `UpdateSettings` persists nothing and `save_all_settings` has no live caller, which is why `apply_physics_patch` writes the row itself.

## VC-06.8 Validation and field-mapping helpers
```mermaid
flowchart TB
    IN["incoming settings JSON"]
    IN --> N1["normalize_physics_keys<br/>src/settings/api/settings_routes.rs:71<br/>snake_case and legacy names to canonical camelCase, call site inside merge_physics_patch :1419"]
    N1 --> V1["validate_physics_settings<br/>src/settings/api/settings_routes.rs:137 (call site :653)"]
    V1 --> G1["validate_for_gpu<br/>src/settings/api/settings_routes.rs:128 (call site :663)<br/>SimulationParams::validate before persistence"]
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

## VC-06.9 Generated client types — src/bin/generate_types.rs
```mermaid
sequenceDiagram
    autonumber
    participant B as generate_types::main<br/>src/bin/generate_types.rs:9
    participant T as generate_typescript_interfaces<br/>src/bin/generate_types.rs:64
    participant O as write_ontology_agent_types<br/>src/bin/generate_types.rs:768
    participant F as client/src/types/generated/settings.ts
    participant G as client/src/types/generated/ontology-agent.ts

    B->>T: build the settings interfaces
    T-->>B: a hand-written TypeScript string literal (generate_types.rs:64-570), snake_case fields
    B->>B: convert_to_camel_case(header + interfaces) (generate_types.rs:19, regex pass :778)
    B->>F: fs::write(output_path, &camel_case_code) (src/bin/generate_types.rs:30)
    Note over B,F: output_path is the literal "client/src/types/generated/settings.ts" (generate_types.rs:21)<br/>the parent directory is created if absent (generate_types.rs:22-28)
    B->>F: fs::metadata(output_path) then log the byte size (generate_types.rs:37-38)
    B->>O: write_ontology_agent_types() (generate_types.rs:57)
    O->>G: fs::write — enum unions from the visionclaw_ontology variants, snake_case wire names kept (generate_types.rs:769-773, :616-621)
    Note over T,F: RESOLVED ADR-2041 by ADR-2115 — the settings interfaces carry `knowledge` only (generate_types.rs:285-286),<br/>`logseq` is gone. path_accessible_impls now matches `knowledge` alone, no alias arm<br/>(src/config/path_accessible_impls.rs:159 and :183). Full alias lifecycle see VC-09.15
    Note over F: DRIFT — the output files are gitignored (.gitignore:151 `generated`) and absent from the tree at this stamp,<br/>while project AGENTS.md:13 says generate_types updates client/src/types/ "from Rust structs".<br/>Only the ontology-agent file is derived from Rust values. The settings file is a string the binary carries
    Note over B: server-side YAML is snake_case, the JSON and TS surface is camelCase —<br/>the serde camelCase output is checked and LOGGED at boot, not enforced, src/main.rs:306-346
```

**Drift:** `AGENTS.md:13` (`../project/AGENTS.md:13`) says `cargo run --bin generate_types` "updates `client/src/types/` from Rust structs". The settings half does not read any Rust struct. `generate_typescript_interfaces` returns a fixed raw string (`../project/src/bin/generate_types.rs:64-570`), so a field added to `PhysicsSettings` does not reach `settings.ts` unless someone edits that string by hand. Only `ontology-agent.ts` is built from Rust values (`../project/src/bin/generate_types.rs:576-597`). Even there, the variant tables are kept in step by hand (`:572-575`).

**Debt:** the boot "serde alias" block in `main.rs` (`../project/src/main.rs:314-346`) logs a confirmation when the rendering JSON is camelCase, and logs an error when serialisation fails. Neither outcome stops boot, so "asserted at boot" in the earlier panel was too strong.
