---
id: VC-34
title: Client feature directories — API and WebSocket surface
area: visionclaw
governing:
  - ../project/docs/BASELINE-architecture.md
adrs: [ADR-2041, ADR-2115, ADR-2006, ADR-2074, ADR-2077, ADR-2142, ADR-2189]
sources:
  - ../project/docs/explanation/control-center.md
  - ../project/src/actors/graph_service_supervisor.rs
  - ../project/src/actors/agent_beam_actor.rs
  - ../project/client/src/features/settings/config/settings.ts
  - ../project/client/src/features/control-center/registry/settingsRegistry.ts
  - ../project/client/src/features/control-center/registry/manifest.ts
  - ../project/client/src/features/control-center/registry/groups/motion.ts
  - ../project/client/src/features/control-center/hooks/useSettingField.ts
  - ../project/client/src/store/settingsStore.ts
  - ../project/client/src/store/autoSaveManager.ts
  - ../project/client/src/store/settingsRetryManager.ts
  - ../project/client/src/api/settings/endpoints.ts
  - ../project/client/src/api/layoutApi.ts
  - ../project/crates/visionclaw-domain/src/config/app_settings.rs
  - ../project/src/bin/generate_types.rs
  - ../project/client/scripts/emit-settings-manifest.ts
  - ../project/client/src/features/control-center/registry/settings-manifest.json
  - ../project/client/src/features/ontology/services/jss/contextLoader.ts
  - ../project/client/src/features/ontology/services/JssOntologyService.ts
  - ../project/client/src/features/ontology/hooks/useOntologyStore.ts
  - ../project/client/src/features/ontology/services/jss/schemaParser.ts
  - ../project/client/src/features/ontology/services/jss/classExtractor.ts
  - ../project/client/src/features/ontology/services/jss/axiomHandler.ts
  - ../project/client/src/features/ontology/services/jss/inferenceClient.ts
  - ../project/client/src/store/websocketStore.ts
  - ../project/client/src/features/ontology/hooks/useInferenceService.ts
  - ../project/client/src/features/ontology/services/sparqlService.ts
  - ../project/client/src/features/ontology/services/inferredAxiomsService.ts
  - ../project/client/src/features/ontology/hooks/useHierarchyData.ts
  - ../project/client/src/features/ontology/hooks/useConstraintStats.ts
  - ../project/client/src/features/ontology/hooks/useOntologyWebSocket.ts
  - ../project/src/handlers/socket_flow_handler/filter_auth.rs
  - ../project/src/actors/ontology_actor.rs
  - ../project/client/src/features/control-center/ControlCenter.tsx
  - ../project/client/src/app/MainLayout.tsx
  - ../project/client/src/features/control-center/panels/SettingsPanel.tsx
  - ../project/client/src/features/control-center/panels/SolidPanel.tsx
  - ../project/client/src/features/control-center/panels/OntologyPanel.tsx
  - ../project/client/src/features/solid/components/SolidTabContent.tsx
  - ../project/client/src/features/ontology/components/OntologyTabContent.tsx
  - ../project/client/src/features/control-center/governance/AcspCaseQueue.tsx
  - ../project/client/src/features/control-center/governance/useBrokerCaseQueue.ts
  - ../project/client/src/features/control-center/governance/brokerCaseQueue.ts
  - ../project/client/src/features/control-center/status/useConnectionTelemetry.ts
  - ../project/client/src/features/control-center/status/useSpacePilot.ts
  - ../project/client/src/features/control-center/kpi/useKpiSummary.ts
  - ../project/client/src/features/control-center/macros/useMacro.ts
  - ../project/client/src/features/control-center/echo/echoPulseBus.ts
  - ../project/client/src/features/bots/hooks/useAgentPolling.ts
  - ../project/client/src/features/bots/services/AgentPollingService.ts
  - ../project/client/src/features/bots/contexts/BotsDataContext.tsx
  - ../project/client/src/features/bots/services/BotsWebSocketIntegration.ts
  - ../project/client/src/features/bots/hooks/useAgentActionFeed.ts
  - ../project/client/src/features/bots/components/AgentDetailPanel.tsx
  - ../project/client/src/services/WebSocketEventBus.ts
  - ../project/client/src/features/monitoring/components/HealthDashboard.tsx
  - ../project/client/src/features/monitoring/hooks/useHealthService.ts
  - ../project/client/src/features/analytics/components/ShortestPathControls.tsx
  - ../project/client/src/features/analytics/hooks/useSemanticService.ts
  - ../project/client/src/features/analytics/store/analyticsStore.ts
  - ../project/client/src/features/analytics/store/nodeAnalyticsStore.ts
  - ../project/client/src/store/websocket/binaryProtocol.ts
  - ../project/client/src/features/graph/components/GemNodes.tsx
  - ../project/client/src/features/graph/components/ClusterHulls.tsx
  - ../project/client/src/features/physics/hooks/usePhysicsService.ts
  - ../project/client/src/features/command-palette/CommandRegistry.ts
  - ../project/client/src/features/command-palette/hooks/useCommandPalette.ts
  - ../project/client/src/features/command-palette/defaultCommands.ts
  - ../project/client/src/features/command-palette/components/CommandPalette.tsx
  - ../project/client/src/features/help/HelpRegistry.ts
  - ../project/client/src/features/help/components/HelpProvider.tsx
  - ../project/client/src/features/onboarding/components/OnboardingEventHandler.tsx
  - ../project/client/src/features/onboarding/hooks/useOnboarding.ts
  - ../project/client/src/features/onboarding/flows/defaultFlows.ts
  - ../project/client/src/features/design-system/patterns/MarkdownRenderer.tsx
  - ../project/client/src/features/design-system/animations.ts
  - ../project/client/src/api/workspaceApi.ts
  - ../project/client/src/hooks/useWorkspaces.ts
  - ../project/client/src/features/solid/hooks/useSolidPod.ts
  - ../project/client/src/features/solid/hooks/useSolidContainer.ts
  - ../project/client/src/features/solid/hooks/useSolidResource.ts
  - ../project/client/src/features/solid/components/PodBrowser.tsx
  - ../project/client/src/features/solid/components/PodSettings.tsx
  - ../project/client/src/features/solid/components/ResourceEditor.tsx
  - ../project/client/src/services/SolidPodService.ts
  - ../project/client/src/services/api/UnifiedApiClient.ts
  - ../project/client/src/features/visualisation/components/CommandInput.tsx
  - ../project/client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
  - ../project/client/src/services/jev/jevClient.ts
  - ../project/client/src/services/jev/commandFlow.ts
  - ../project/client/src/features/bots/agentDemo.ts
  - ../project/client/src/features/graph/components/GraphCanvas.tsx
  - ../project/client/src/features/visualisation/components/TransientBeamsLayer.tsx
  - ../project/client/src/features/visualisation/hooks/useTransientBeams.ts
  - ../project/client/src/store/transientBeamStore.ts
  - ../project/client/src/features/voice/pttAgentBinding.ts
  - ../project/client/src/features/voice/usePushToTalkAgentBinding.ts
  - ../project/client/src/services/PushToTalkService.ts
  - ../project/client/src/services/VoiceWebSocketService.ts
  - ../project/client/src/features/control-center/agents/AgentOpsSurface.tsx
  - ../project/client/src/features/control-center/primitives/SettingSlider.tsx
  - ../project/client/src/features/control-center/state/useControlCenterUI.ts
  - ../project/client/src/features/control-center/status/StatusSurface.tsx
  - ../project/client/src/features/design-system/components/SearchInput.tsx
  - ../project/client/src/features/ontology/components/OntologyContribution.tsx
  - ../project/src/services/broker_events.rs
  - ../project/src/main.rs
  - ../project/docker-compose.unified.yml
  - ../project/client/src/features/control-center/state/settingWrites.ts
  - ../project/client/src/store/settings/coreSlice.ts
  - ../project/client/src/store/settings/retiredPaths.ts
  - ../project/client/src/features/graph/overview/__tests__/edgeOpacity.test.ts
  - ../project/client/src/features/control-center/drawer/RightDrawer.tsx
verified_commit: {visionclaw: ec785afa9331be1fa049cee6282a4dcda16ee86d}
---
## For developers

`client/src/features/` holds one directory per product surface, and this topic traces what each one calls on the server: REST through `unifiedApiClient`, WebSocket through `webSocketService`, and settings through the registry and autosave chain. The ec785afa9 refresh picks up the ADR-2189 control-centre rework. Every settings change from a registry row, a dock dial, the Ask box or a Jev patch now goes through one function, `applySettingWrites` (VC-34.1, VC-34.19). The registry regroups its ten source groups into seven sections, and the emitter writes a v2.0 manifest plus two `ia.json` copies (VC-34.2). The memory explorer has moved into the right drawer (VC-34.20). The analytics store's SSSP call targets the surviving `pathfinding/sssp` route (VC-34.12). The section layout, the drawer and the scope badges belong to VC-57, and the explore layer to VC-56.

**Debt (loose end):** the chat parser writes `visualisation.graphs.knowledge.nodes.opacity` (`CommandInput.tsx:313`), and no renderer reads it. Its only client touch is the server-wins merge (`coreSlice.ts:140`). ADR-2189 retired the `visionclaw` twin of this path (`retiredPaths.ts:39`) but not the knowledge one.

**Debt (parallel):** `serverBucketFor` (`manifest.ts:110`) restates the prefix routing of `updateSettingsByPaths` (`endpoints.ts:344-366`) by hand, as its own comment admits (`manifest.ts:105-108`). The scope badges and manifest `server` field depend on the copy, not on the code that routes the PUT.

**Debt (parallel):** shortest path has two client callers on two routes, `analyticsStore.ts:350` (`/api/analytics/pathfinding/sssp`) and `useSemanticService.ts:200` (`/api/semantic/shortest-path`).

## For the business

The client is a set of product surfaces: settings, ontology, governance, agents, analytics, the command bar and voice. Each one talks to the server in its own way. After this change, a setting typed into the Ask box behaves exactly like moving the slider. It is saved the same way, respects the sync switch and moves the layout when it should. "Make the edges fainter" now changes the edges. "Make it transparent" without the word edges still reports success while nothing on screen changes. The settings menu now has seven sections instead of eleven groups. The headset reads the same menu layout from a generated copy.

## VC-34.1 settings — field edit to server PUT
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant SS as SettingSlider<br/>client/src/features/control-center/primitives/SettingSlider.tsx
    participant USF as useSettingField<br/>client/src/features/control-center/hooks/useSettingField.ts:40
    participant SW as applySettingWrites<br/>client/src/features/control-center/state/settingWrites.ts:42
    participant Store as useSettingsStore.updateSettings<br/>client/src/store/settingsStore.ts
    participant ASM as autoSaveManager<br/>client/src/store/autoSaveManager.ts:52
    participant LA as layoutApi.setMode<br/>client/src/api/layoutApi.ts
    participant EP as settingsApi.updateSettingsByPaths<br/>client/src/api/settings/endpoints.ts:331
    participant SRV as Rust settings handlers

    U->>SS: drag slider
    SS->>USF: set(next)
    USF->>SW: applySettingWrites([path, next]) useSettingField.ts:47
    Note over USF,SW: ADR-2189 one writer per setting - the registry row,<br/>the dock dials (useMacro.ts:35-37) and the Ask box (CommandInput.tsx:563)<br/>all call this one function (settingWrites.ts:2-11)
    SW->>Store: updateSettings(draft) one immer batch over every write settingWrites.ts:46-58
    Note over Store: path is the FROZEN backend contract (settingsRegistry.ts:5)
    Store-->>ASM: queueChanges (internal, autoSaveManager.queueChange path -> value)
    SW->>ASM: queueChange(path, value) per write - explicit second queue settingWrites.ts:60-64
    alt path in LAYOUT_SETTING_PATHS, physics.layoutMode or layoutAlgorithm settingWrites.ts:29-32
        SW->>LA: setMode(value, 800) settingWrites.ts:65-68
        LA->>SRV: POST /api/layout/mode with the name unchanged, client/src/api/layoutApi.ts:70-77
        Note over LA: the engine LayoutMode names are the stored physics.layoutMode values (ADR-2140), there is no<br/>remapping layer any more, toEngineLayoutMode is gone client/src/api/layoutApi.ts:47-60
    end
    ASM->>ASM: scheduleFlush - clearTimeout then setTimeout 500ms DEBOUNCE_DELAY autoSaveManager.ts:18, autoSaveManager.ts:92-99
    Note right of ASM: INVARIANT - CLIENT_ONLY_PATHS (auth.nostr.connected, auth.nostr.publicKey) never leave the browser autoSaveManager.ts:34-43
    loop every queueChange before flush
        USF->>ASM: further edits reset the 500ms timer
    end
    ASM->>EP: updateSettingsByPaths(updates) on flush autoSaveManager.ts:124
    EP->>EP: bucket path by prefix - physics/rendering/qualityGates/nodeFilter/constraints/visual/local-only endpoints.ts:344-366
    par physics bucket
        EP->>SRV: PUT /api/settings/physics
    and rendering bucket
        EP->>SRV: PUT /api/settings/rendering
    and qualityGates bucket
        EP->>SRV: PUT /api/settings/quality-gates
    and nodeFilter and constraints buckets
        EP->>SRV: PUT /api/settings/node-filter, PUT /api/settings/constraints
    and visual bucket
        EP->>SRV: PUT /api/settings/visual
    end
    alt any bucket rejects, 401 included
        EP-->>ASM: every update of the flush goes to settingsRetryManager.addFailedUpdate(path, value, message) autoSaveManager.ts:126-136
        ASM->>EP: a single-path retry calls updateSettingByPath, on 401 it dispatches CustomEvent settings-auth-failed<br/>settingsRetryManager.ts:104-108, endpoints.ts:287, endpoints.ts:309-310
    else PUT ok
        SRV-->>EP: 200
    end
    Note over EP,ASM: DEBT: updateSettingByPath catches every error and never rethrows (endpoints.ts:307-315), so the single-path retry<br/>always takes its success branch and deletes the update from the queue (settingsRetryManager.ts:106-108). A failed retry is<br/>dropped silently, and settings-auth-failed has no listener anywhere in client/src. A 401 on a setting reaches no UI
```
## VC-34.2 settings — type generation and QA manifest (build-time, not runtime)
```mermaid
flowchart TD
    RustStruct["AppFullSettings struct<br/>crates/visionclaw-domain/src/config/app_settings.rs:79"] -.->|mirrored by hand, not read| Gen["generate_typescript_interfaces()<br/>src/bin/generate_types.rs:64, a raw-string TS literal :65-568"]
    Bin["cargo run --bin generate_types<br/>main() src/bin/generate_types.rs:9"] --> Gen
    Bin --> OntGen["write_ontology_agent_types()<br/>src/bin/generate_types.rs:767-768, client/src/types/generated/ontology-agent.ts,<br/>serialised from the real ontology enums (wire(), union_of() :590-598)"]
    Debt1["DEBT: the settings interfaces are a hand-maintained string, not derived from AppFullSettings.<br/>A field added to the Rust struct is absent from settings.ts until someone edits the literal.<br/>Only the ontology-agent output is generated from Rust types (src/bin/generate_types.rs:590-600)"] -.-> Gen
    Gen -->|convert_to_camel_case| Out["client/src/types/generated/settings.ts<br/>build output written at src/bin/generate_types.rs:21, :30, gitignored (.gitignore line 151), not in the repository at this stamp"]
    Note1["settingsUIDefinition.ts and unifiedSettingsConfig.ts no longer exist. The chain is<br/>registry/settingsRegistry.ts + registry/manifest.ts (settingsRegistry.ts:6 names the old<br/>file only inside a zero-drift-test description). NOT doc drift - docs/explanation/<br/>control-center.md:88,110,296 already describes unifiedSettingsConfig.ts as deleted. It was<br/>the swarm brief that was stale, not the governing docs. Verified 2026-09-05"]
    Groups["registry/groups/*.ts - ten SOURCE groups where each field is declared<br/>motion, look, labels, quality, atmosphere, intelligence, system, agents, provenance, graphView<br/>(immersion and decisions are gone, client/src/features/control-center/registry/manifest.ts:27-39)<br/>semanticForces.ts is not a group, it feeds motion (registry/groups/motion.ts:4)"] --> ManifestTs["GROUP_DATA<br/>client/src/features/control-center/registry/manifest.ts:27"]
    ManifestTs -->|placeField per field| Sections["SECTION_DATA - the seven ADR-2189 sections, hotkeys 1..7<br/>client/src/features/control-center/registry/manifest.ts:102, buildSections :75"]
    Sections --> Registry["settingsRegistry.ts:30 REGISTRY - SECTION_DATA with icons attached"]
    DriftReg["DRIFT: the header still says the registry assembles GROUP_DATA (settingsRegistry.ts:4)<br/>and manifest.ts:6 says the same, while REGISTRY maps SECTION_DATA (settingsRegistry.ts:30)"] -.-> Registry
    Sections -->|npm run gen:manifest, ts-node| Emit["client/scripts/emit-settings-manifest.ts:24 buildManifest()<br/>and buildIa() written twice, emit-settings-manifest.ts:19-21, :27"]
    Emit --> ManifestJson["registry/settings-manifest.json - version 2.0, 247 settings, 7 sections<br/>each entry now carries section, sourceGroup, subgroup and scope (manifest.ts:136)"]
    Emit --> Ia["registry/ia.json and xr-client/data/ia.json - section, subsection, fields<br/>built by buildIa (manifest.ts:218), see VC-57 for the parity test and scope badges"]
    ManifestJson -->|consumed by| QA["browser-automation coverage phase: assert every testid present+interactive"]
    ADR["ADR-2041 renamed logseq to knowledge, superseded by ADR-2115<br/>which retires the server-side logseq alias entirely -<br/>touches settings.ts, app_settings.rs, generate_types.rs output"] -.-> RustStruct
    ADR -.-> Out
    Note2["settingsStore.ts:59-63 keeps a receive-side migrateGraphSettingsKey<br/>shim for stored localStorage state - ADR-2115 retired only the<br/>server alias, not this client migration. ADR-2189 wraps it in<br/>stripRemovedPaths so menu-deleted paths are dropped on rehydrate"] -.-> Out
```
## VC-34.3 ontology — JSS fetch, PATCH and WebSocket refresh (contextLoader is the shared base)
```mermaid
sequenceDiagram
    autonumber
    participant Svc as JssOntologyService<br/>client/src/features/ontology/services/JssOntologyService.ts:58
    participant CL as contextLoader.fetchWithAuth<br/>client/src/features/ontology/services/jss/contextLoader.ts:37
    participant SP as schemaParser.fetchJsonLd<br/>client/src/features/ontology/services/jss/schemaParser.ts:99
    participant CE as classExtractor.buildHierarchyFromJsonLd<br/>client/src/features/ontology/services/jss/classExtractor.ts
    participant AH as axiomHandler.patchOntology<br/>client/src/features/ontology/services/jss/axiomHandler.ts:39
    participant IC as InferenceClient<br/>client/src/features/ontology/services/jss/inferenceClient.ts:26
    participant WSS as webSocketService.connectSolid<br/>client/src/store/websocketStore.ts
    participant JSS as JSS Solid server

    Svc->>Svc: initialize() - loadIntoStore then connectWebSocket
    Svc->>SP: fetchOntologyJsonLd(options)
    SP->>SP: isCacheValid(cache) check TTL
    alt cache miss or skipCache
        SP->>CL: getOntologyJsonLdUrl() = getOntologyUrl() + "/ontology.jsonld"<br/>contextLoader.ts:29-31 (ADR-2106)
        CL->>CL: nostrAuth.isAuthenticated - dev mode Bearer dev-session-token or NIP-98 signRequest
        CL->>JSS: fetch GET credentials include
        JSS-->>CL: 200 JSON-LD body
    else cache hit
        SP-->>Svc: cached JsonLdOntology
    end
    Note over SP,CL: RESOLVED ADR-2106 (2026-09-06): the embedded pod serves the ontology<br/>container as an LDP listing on a bare GET, not a negotiated single resource -<br/>fetchJsonLd/fetchTurtle now name the two resources explicitly via<br/>getOntologyJsonLdUrl/getOntologyTurtleUrl (see VC-26.13 for the server side<br/>that publishes them). patchOntology below still targets getOntologyUrl()<br/>directly - unchanged by this ADR.
    Svc->>CE: buildHierarchyFromJsonLd(jsonLd)
    CE-->>Svc: OntologyHierarchy
    Svc->>Svc: useOntologyStore.getState().setHierarchy/setMetrics/setLoaded
    Svc->>IC: connectWebSocket()
    IC->>IC: connect - guard on JSS_WS_URL (VITE_JSS_WS_URL) else warn and return
    IC->>WSS: webSocketService.connectSolid()
    IC->>WSS: subscribeSolidResource(ontologyUrl, handleNotification)
    WSS-->>IC: SolidNotification type pub on resource change
    IC->>SP: invalidateCache then fetchJsonLd skipCache true
    IC->>CE: buildHierarchyFromJsonLd(jsonLd)
    IC->>Svc: useOntologyStore.setHierarchy - full_refresh event fanned to onResourceChange callbacks
    Note over Svc,JSS: see VC-26 for the Solid pod data path and VC-33 for auth token issuance
    opt patchOntology(cache, sparqlUpdate) is called
        Svc->>AH: patchOntology(cache, sparqlUpdate) JssOntologyService.ts:146-147
        AH->>CL: fetchWithAuth(getOntologyUrl(), method PATCH) axiomHandler.ts:43-46
        CL->>JSS: PATCH content-type application/sparql-update axiomHandler.ts:47
        JSS-->>AH: PatchResult status axiomHandler.ts:57
    end
    Note over Svc,AH: DEBT: nothing in client/src calls JssOntologyService.patchOntology, the PATCH path is unreachable. The axiom editor,<br/>OntologyContribution.tsx, creates proposals through useOntologyContributionStore (OntologyContribution.tsx:84), which POSTs<br/>/api/ontology/proposals with a plain fetch (useOntologyStore.ts:322, useOntologyStore.ts:336), not a JSS PATCH
    Note over Svc,AH: OPEN: that plain fetch attaches no NIP-98 header in useOntologyStore.ts, so whether a proposal write is authorised<br/>depends on the server route's gate, which this panel does not trace
```
## VC-34.4 ontology — inference REST, SPARQL console and reasoning report
```mermaid
sequenceDiagram
    autonumber
    participant Panel as OntologyPanel / InferencePanel / SparqlConsole<br/>client/src/features/ontology/components/
    participant UIS as useInferenceService<br/>client/src/features/ontology/hooks/useInferenceService.ts:49
    participant Sparql as sparqlService.runSparqlSelect<br/>client/src/features/ontology/services/sparqlService.ts:73
    participant Inferred as inferredAxiomsService<br/>client/src/features/ontology/services/inferredAxiomsService.ts:168
    participant Hier as useHierarchyData<br/>client/src/features/ontology/hooks/useHierarchyData.ts:57
    participant Stats as useConstraintStats<br/>client/src/features/ontology/hooks/useConstraintStats.ts:25
    participant UAC as unifiedApiClient
    participant SRV as Rust /api/inference and /api/ontology handlers

    Panel->>UIS: runInference(request)
    UIS->>UAC: POST /inference/run useInferenceService.ts:62-63
    UAC->>SRV: HTTP POST
    Panel->>UIS: validateOntology(request)
    UIS->>UAC: POST /inference/validate useInferenceService.ts:94-95
    Panel->>UIS: getInferenceResults / getClassification / getConsistencyReport
    UIS->>UAC: GET /inference/results/{ontologyId}, /inference/classify/{ontologyId}, /inference/consistency/{ontologyId}<br/>useInferenceService.ts:127, useInferenceService.ts:158, useInferenceService.ts:189
    opt cache clear
        UIS->>UAC: DELETE /inference/cache/{ontologyId} useInferenceService.ts:219
    end
    Panel->>Sparql: runSparqlSelect(query)
    Sparql->>Sparql: isReadOnlySelect(query) guard - rejects non-SELECT sparqlService.ts:60, sparqlService.ts:74
    Sparql->>UAC: POST /ontology/sparql (SPARQL_ENDPOINT) sparqlService.ts:30, sparqlService.ts:80
    UAC->>SRV: query executed against Oxigraph
    Panel->>Inferred: fetchReasoningReport()
    Inferred->>UAC: GET /ontology/inferred (INFERRED_ENDPOINT) inferredAxiomsService.ts:43, inferredAxiomsService.ts:170
    Note right of Inferred: INVARIANT - urn:ngm:graph:ontology:inferred<br/>is a read-only projection, never hand-edited
    Hier->>Hier: fetch(url) direct, not the shared API client, so no auth interceptor runs - hierarchy tree endpoint useHierarchyData.ts:96
    loop every 5000ms (intervalMs default) useConstraintStats.ts:25, useConstraintStats.ts:45
        Stats->>Stats: poll constraint stats while enabled
    end
```
## VC-34.5 ontology — server-pushed validation over WebSocket
```mermaid
sequenceDiagram
    autonumber
    participant SRV as Rust ontology validation handler
    participant WSS as webSocketService<br/>client/src/store/websocketStore.ts
    participant WSHook as useOntologyWebSocket<br/>client/src/features/ontology/hooks/useOntologyWebSocket.ts:51
    participant Store as useOntologyStore

    SRV-->>WSS: ontology_validation_update message src/handlers/socket_flow_handler/filter_auth.rs:464, src/actors/ontology_actor.rs:368
    WSS-->>WSHook: onMessage callback useOntologyWebSocket.ts:55-56
    WSHook->>Store: setValidating / setViolations / setMetrics useOntologyWebSocket.ts:60-66
    opt ontology_loaded message
        WSHook->>Store: setLoaded(true) and setMetrics when data.success useOntologyWebSocket.ts:75-85
    end
    Note over SRV,WSHook: DEBT: no server code emits ontology_loaded (git grep over src and crates finds only ontology_validation_update),<br/>so the second subscription (useOntologyWebSocket.ts:75-87) never fires and setLoaded comes only from the REST and JSS loads (VC-34.3)
    Note over SRV,Store: see VC-32 for the WebSocket frame/opcode machinery itself
```
## VC-34.6 control-center — shell composition (SettingsPanel hosts Solid and Ontology panels)
```mermaid
flowchart TD
    CC["ControlCenter.tsx:37<br/>ControlCenter component"] --> Dock["GlassDock<br/>primitives/GlassDock.tsx"]
    CC --> MacroBar["MacroBar<br/>macros/MacroBar.tsx"]
    CC --> CmdInput["CommandInput<br/>visualisation/components/CommandInput.tsx"]
    CC --> SP["SettingsPanel<br/>panels/SettingsPanel.tsx:50"]
    SP -->|REGISTRY, GROUP_BY_ID, ALL_FIELDS - seven sections| Reg["settingsRegistry.ts:30"]
    SP -->|PANELS solid+ontology, each opened from a launcher in its home section| Man["manifest.ts:41, SettingsPanel.tsx:115-126, :137"]
    SP --> SolidPanel["SolidPanel<br/>panels/SolidPanel.tsx:11"]
    SP --> OntPanel["OntologyPanel<br/>panels/OntologyPanel.tsx:12"]
    SolidPanel --> SolidTab["SolidTabContent<br/>client/src/features/solid/components/SolidTabContent.tsx"]
    OntPanel --> OntTab["OntologyTabContent<br/>client/src/features/ontology/components/OntologyTabContent.tsx"]
    SP -->|SearchInput| DS["design-system SearchInput.tsx"]
    CC --> Hotkeys["useControlCenterHotkeys, useExclusiveSidePanels, useRevealSetting<br/>client/src/features/control-center/ControlCenter.tsx:81-83"]
    CC --> Drawer["RightDrawer, LegendSlot, ShortcutSheet mounted beside the dock<br/>client/src/features/control-center/ControlCenter.tsx:134-136 - drawer tabs and IA are VC-57"]
    CC -->|SpaceDriver| SPD["SpaceDriverService - see VC-37 for device wire protocol"]
    Note1["AgentOpsSurface and StatusSurface are NOT children of ControlCenter: MainLayout mounts both beside it<br/>(app/MainLayout.tsx:70, :90, :103), and ControlCenter.tsx:14-15 says the status surface reads SpaceDriver and websocket state itself"]
    Note2["AgentOpsSurface hosts the shared agent-demo toggle: useAgentDemo polls GET /agents/demo (4 s) and POSTs {running} to start or stop it, so the state shown is the server's and a headset-started demo reads as running here (agentDemo.ts:56, :73, AgentOpsSurface.tsx:55)"]
```
## VC-34.7 control-center — ACSP governance case queue (ADR-2006)
```mermaid
sequenceDiagram
    autonumber
    participant Q as AcspCaseQueue<br/>client/src/features/control-center/governance/AcspCaseQueue.tsx:168
    participant H as useBrokerCaseQueue<br/>client/src/features/control-center/governance/useBrokerCaseQueue.ts:32
    participant Pure as brokerCaseQueue.ts pure helpers<br/>governance/brokerCaseQueue.ts:80, :148, :277
    participant UAC as unifiedApiClient
    participant WSS as webSocketService<br/>client/src/store/websocketStore.ts
    participant SRV as broker_events.rs handlers (WS-9/WS-12)

    Q->>H: mount
    H->>UAC: getData GET /broker/inbox useBrokerCaseQueue.ts:41
    UAC->>SRV: HTTP GET
    alt unauthenticated or unavailable
        SRV-->>H: error
        H->>H: fail-soft - logger.debug, empty queue, no crash useBrokerCaseQueue.ts:48
    else ok
        SRV-->>H: cases[]
        H->>Pure: toCaseView(c) per case, openCaseIds(inbox)
        H-->>Q: cases, openCount
    end
    H->>WSS: onMessage subscribe (multiplexed graph socket)
    loop live broker events
        SRV-->>WSS: broker:new_case / broker:case_decided
        WSS-->>H: message
        H->>Pure: parseBrokerEvent(message)
        H->>Pure: applyBrokerEvent(openIds, event) - ambient count updates instantly
        H->>H: refresh() - full case list re-fetched for metadata
    end
    Q->>Pure: sortOldestFirst(open cases) and caseAgeLabel(createdAt, now), both called by the component<br/>governance/brokerCaseQueue.ts:233, :251, AcspCaseQueue.tsx:175, AcspCaseQueue.tsx:65
    Note over Q: ADR-2110 FR6.5: the queue sorts OLDEST FIRST and badges each case's<br/>age, so a stalled judgment is visible rather than silently buried.<br/>AcspCaseQueue.tsx:27, :65
    Q->>Q: human types their own rationale into the case card<br/>AcspCaseQueue.tsx:62, :126
    Q->>Pure: canPublishDecision(view.tier, rationale)<br/>governance/brokerCaseQueue.ts:219, called at AcspCaseQueue.tsx:64
    Note over Q,Pure: ADR-2110 FR2.2 INVARIANT: the decision carries the human's text BYTE<br/>FOR BYTE and the UI authors nothing. The former<br/>operator-verb-via-control-centre template is deleted - a rationale the<br/>human did not write is a fabricated judgement (DDD invariant 1).<br/>AcspCaseQueue.tsx:18, :20
    Note over Pure: INVARIANT: the gate counts UNICODE SCALARS, not UTF-16 code units,<br/>because the server's check_rationale counts str chars. Counting code<br/>units would enable publish on ten astral characters and then collect<br/>a 422, which reads as the system losing the typed rationale.<br/>governance/brokerCaseQueue.ts:199, :189
    Note over Q: ADR-2110 FR6: the agent's SELF-ASSESSMENT (declared tier, confidence)<br/>renders BELOW the controls so it cannot anchor the reviewer before<br/>they have read the proposal. AcspCaseQueue.tsx:24
    Q->>H: decide(caseId, outcome, rationale)
    H->>UAC: POST /broker/cases/{caseId}/decide with outcome and reasoning useBrokerCaseQueue.ts:75-78
    UAC->>SRV: HTTP POST WS-9 operator route
    SRV-->>H: success
    H->>H: optimistically close caseId locally then refresh() useBrokerCaseQueue.ts:80-87
    Note right of H: INVARIANT ADR-2006 - ACSP decisions require human approval through this exact route, never auto-decided
    Note right of H: The client gate is a courtesy, not the rule: the same rationale check<br/>runs server-side on the shared decide core and answers 422. See VC-24.3.<br/>governance/brokerCaseQueue.ts:186
```
## VC-34.8 control-center — status telemetry, KPI polling and echo/macro bus
```mermaid
sequenceDiagram
    autonumber
    participant Status as StatusSurface / StatusFlyout<br/>client/src/features/control-center/status/StatusSurface.tsx
    participant WsStat as useWebSocketStatus<br/>client/src/features/control-center/status/useConnectionTelemetry.ts:28
    participant LM as useLayoutMotion<br/>client/src/features/control-center/status/useConnectionTelemetry.ts:61
    participant SPHook as useSpacePilot (control-center)<br/>client/src/features/control-center/status/useSpacePilot.ts:33
    participant Kpi as KpiPanel/useKpiSummary<br/>client/src/features/control-center/kpi/useKpiSummary.ts:25
    participant UAC as unifiedApiClient
    participant Macro as useMacro<br/>client/src/features/control-center/macros/useMacro.ts:47
    participant Echo as echoPulseBus<br/>client/src/features/control-center/echo/echoPulseBus.ts:39
    participant WSS as websocketStore

    Status->>WsStat: subscribe websocketStore connection state
    Status->>LM: read useWebSocketStore statistics for motion readout
    Status->>SPHook: read SpaceDriver connected/deviceName state
    Note right of SPHook: see VC-37 for SpaceDriver<br/>WebHID wire protocol - not redrawn here
    Kpi->>UAC: getData GET /kpi/summary
    UAC-->>Kpi: raw payload
    Kpi->>Kpi: normaliseKpiSummary(raw) -> KpiTileView[]
    Macro->>Macro: applyMacroWrites - delegates to applySettingWrites, the one write path of VC-34.1 (useMacro.ts:35-37)
    Macro->>Echo: emitEchoPulse(detail) - CustomEvent visionclaw:echo-pulse
    Echo-->>Status: EchoPulseLayer subscribeEchoPulse renders transient ring
    Note over Status,WSS: see VC-32 for the underlying WebSocket connection lifecycle
```
## VC-34.9 bots — adaptive REST polling of the agent population
```mermaid
sequenceDiagram
    autonumber
    participant Ctx as BotsDataContext<br/>client/src/features/bots/contexts/BotsDataContext.tsx
    participant Hook as useAgentPolling<br/>client/src/features/bots/hooks/useAgentPolling.ts:128
    participant Svc as AgentPollingService<br/>client/src/features/bots/services/AgentPollingService.ts:79
    participant UAC as unifiedApiClient
    participant SRV as GraphServiceSupervisor to AgentBeamActor

    Ctx->>Hook: mount
    Hook->>Svc: subscribe(callback) then start()
    Svc->>Svc: subscriberCount++ - only subscriber 1 begins poll() AgentPollingService.ts:126-128, useAgentPolling.ts:245-248
    loop poll every currentInterval (2000ms active / 10000ms idle) AgentPollingService.ts:97-98
        Svc->>UAC: getData GET /bots/data AgentPollingService.ts:221
        UAC->>SRV: HTTP GET
        Note right of SRV: ADR-2142 amendment - GetBotsGraphData forwarded by the supervisor<br/>graph_service_supervisor.rs:1815 to AgentBeamActor agent_beam_actor.rs:267<br/>the roster stays SEPARATE from the main graph, /graph/data?graph_type=agent stays empty
        SRV-->>Svc: envelope {success, data:{nodes, edges}} or bare body (older builds)
        Svc->>Svc: use envelope.data only when envelope.data.nodes is truthy, else the whole envelope AgentPollingService.ts:227-228
        Svc->>Svc: hashData(data) - skip callback fan-out if unchanged AgentPollingService.ts:235
        Svc->>Svc: updateActivityLevel - activeRatio>0.2 or hasChanged or tasks pending -> active 2000ms else idle 10000ms AgentPollingService.ts:284, AgentPollingService.ts:293
        alt hasChanged
            Svc->>Hook: callback(data) fan-out to all subscribers
            Hook->>Hook: transformAgentData(data) / isGenuineAgentNode filter useAgentPolling.ts:18, useAgentPolling.ts:24-28, useAgentPolling.ts:149
        end
        alt request throws
            Svc->>Svc: handlePollingError - retryCount++, errorCallbacks, retryDelay 2000 times 2 pow (n-1) AgentPollingService.ts:308-336
            break retryCount >= maxRetries (3) AgentPollingService.ts:100, AgentPollingService.ts:324
                Svc->>Svc: stop() and no retry timer, polling halts AgentPollingService.ts:326-327
            end
        end
    end
    Hook-->>Ctx: unmount -> Svc.stop() - subscriberCount-- only clears timer at zero AgentPollingService.ts:141-152, useAgentPolling.ts:267-270
    Note over Svc: DEBT: the max-retry stop() releases a subscriber slot no subscriber gave up. With two subscribers the count drops to one,<br/>isPolling stays true and no timer runs, so polling is dead, and a later start() sees count 2, not 1, and never restarts it<br/>(AgentPollingService.ts:126-128, AgentPollingService.ts:141-143, AgentPollingService.ts:324-327)
```
## VC-34.10 bots — WebSocket integration, action feed and interrupt
```mermaid
sequenceDiagram
    autonumber
    participant WSI as BotsWebSocketIntegration<br/>client/src/features/bots/services/BotsWebSocketIntegration.ts:9
    participant WSS as webSocketService<br/>client/src/store/websocketStore.ts
    participant Bus as webSocketEventBus<br/>client/src/services/WebSocketEventBus.ts
    participant Feed as useAgentActionFeed<br/>client/src/features/bots/hooks/useAgentActionFeed.ts:78
    participant WSStore as useWebSocketStore<br/>client/src/store/websocketStore.ts
    participant Panel as AgentDetailPanel<br/>client/src/features/bots/components/AgentDetailPanel.tsx:58
    participant UAC as unifiedApiClient
    participant SRV as interrupt_task handler

    WSI->>WSS: onConnectionStatusChange, onMessage, onBinaryMessage<br/>BotsWebSocketIntegration.ts:34, BotsWebSocketIntegration.ts:49, BotsWebSocketIntegration.ts:102
    WSS-->>WSI: message.type == graph-update | botsGraphUpdate | bots-full-update BotsWebSocketIntegration.ts:55-91
    WSI->>Bus: emit message:bots BotsWebSocketIntegration.ts:53
    WSI->>WSI: emit knowledge-graph-update / bots-graph-update / bots-full-update<br/>BotsWebSocketIntegration.ts:57, BotsWebSocketIntegration.ts:80, BotsWebSocketIntegration.ts:91
    Note over WSI,WSS: DEBT: the bots-position-update listener was removed from this class since the last<br/>stamp, and nothing else in client/src listens for it, yet the store still emits it for every<br/>frame with agent ids (store/websocket/binaryProtocol.ts:325, :380). Agents reach the scene<br/>only through the 0x23 registry (VC-32.11). See VC-32 for the frame layout
    WSStore->>Feed: on(agent-action) - the decoded 0x23 batch emitted by the store useAgentActionFeed.ts:93
    Feed->>Feed: drop state frames, newest first into a bounded ring,<br/>agentActionVerb(actionType) for the line useAgentActionFeed.ts:99-108, useAgentActionFeed.ts:22
    Panel->>UAC: POST /bots/interrupt {task_id, agent_id} AgentDetailPanel.tsx:104
    UAC->>SRV: HTTP POST
    alt the server resolves the id to a task and stops it
        SRV-->>Panel: response.message - status done AgentDetailPanel.tsx:111
    else HTTP 422, resolution unresolved or interruptible false
        Panel->>Panel: interruptStatus = not-interruptible<br/>Externally spawned - not interruptible from here<br/>AgentDetailPanel.tsx:21, AgentDetailPanel.tsx:27, AgentDetailPanel.tsx:114
    end
    Note over Panel,SRV: D2 steering - the id is sent as-is, the server resolves task_id or swarm agent_id<br/>to a task before stopping AgentDetailPanel.tsx:93-97. A not-interruptible answer is<br/>remembered and blocks further tries until the selection changes<br/>AgentDetailPanel.tsx:79-80, AgentDetailPanel.tsx:101
```
## VC-34.11 monitoring — health dashboard poll and MCP relay control
```mermaid
sequenceDiagram
    autonumber
    participant HD as HealthDashboard<br/>client/src/features/monitoring/components/HealthDashboard.tsx:25
    participant Hook as useHealthService<br/>client/src/features/monitoring/hooks/useHealthService.ts:48
    participant UAC as unifiedApiClient
    participant SRV as Rust /health handlers

    HD->>Hook: mount, pollHealth default true useHealthService.ts:49, HealthDashboard.tsx:27-28
    Hook->>Hook: fetchHealth() immediately
    par
        Hook->>UAC: GET /health useHealthService.ts:64-65
    and
        Hook->>UAC: GET /health/physics useHealthService.ts:66
    end
    UAC->>SRV: HTTP GET
    SRV-->>Hook: HealthStatus, PhysicsHealth
    loop every pollInterval (default 5000ms)
        Hook->>Hook: setInterval(fetchHealth, pollInterval) useHealthService.ts:85
    end
    HD->>Hook: startMCPRelay()
    Hook->>UAC: POST /health/mcp/start useHealthService.ts:104
    HD->>Hook: getMCPLogs()
    Hook->>UAC: GET /health/mcp/logs useHealthService.ts:132
    Note right of HD: HD composes design-system Card/Badge/Button/Toast primitives - see VC-34.15
```
## VC-34.12 analytics — semantic service REST, SSSP store and wire-decoded per-node analytics
```mermaid
sequenceDiagram
    autonumber
    participant SPC as ShortestPathControls<br/>client/src/features/analytics/components/ShortestPathControls.tsx:38
    participant USS as useSemanticService<br/>client/src/features/analytics/hooks/useSemanticService.ts:85
    participant AS as useAnalyticsStore<br/>client/src/features/analytics/store/analyticsStore.ts:286
    participant UAC as unifiedApiClient
    participant WSBin as handleLegacyBinaryData<br/>client/src/store/websocket/binaryProtocol.ts:352
    participant NAS as nodeAnalyticsStore<br/>client/src/features/analytics/store/nodeAnalyticsStore.ts:35
    participant Gem as GemNodes / ClusterHulls<br/>client/src/features/graph/components/

    USS->>UAC: GET /api/semantic/statistics useSemanticService.ts:100
    USS->>UAC: POST /api/semantic/communities useSemanticService.ts:136
    USS->>UAC: POST /api/semantic/centrality useSemanticService.ts:168
    SPC->>USS: computeShortestPath(request)
    USS->>UAC: POST /api/semantic/shortest-path useSemanticService.ts:200
    USS->>UAC: POST /api/semantic/constraints/generate useSemanticService.ts:232
    USS->>UAC: POST /api/semantic/cache/invalidate useSemanticService.ts:262
    AS->>UAC: POST /api/analytics/pathfinding/sssp {sourceIdx}<br/>old shortest-path route gone server-side analyticsStore.ts:346-352
    AS->>AS: toDistanceMap(result.distances) accepts the whole-graph array<br/>or the gated id-keyed object, keeps finite values below 3.0e38,<br/>unreachable becomes Infinity analyticsStore.ts:83-93, analyticsStore.ts:363-368
    Note over AS,UAC: analyticsStore and useSemanticService still hit distinct endpoints<br/>for the same shortest-path feature - not unified
    Note over AS,UAC: OPEN: sourceIdx is parseInt(sourceNodeId) (analyticsStore.ts:351),<br/>so a non-numeric node id posts NaN, which JSON carries as null<br/>against the server usize sourceIdx
    Note over USS,UAC: DEBT (source-traced, not run): every path here and in analyticsStore leads with /api, and unifiedApiClient prepends<br/>its own /api base (client/src/services/api/UnifiedApiClient.ts:77, :182), so they resolve to /api/api/semantic/... and<br/>/api/api/analytics/pathfinding/sssp, the doubled prefix the repo records as a 404 for the ontology services it already fixed
    WSBin->>NAS: ingest(parsedNodes) - V3/V5 frames only, wire offsets 36/40/44/48 plus sssp_distance@28 store/websocket/binaryProtocol.ts:371-372, nodeAnalyticsStore.ts:51
    Note right of NAS: ADR-03 D7, ADR-031 D2/D6 - stride 5 buffer<br/>[clusterId, anomalyScore, communityId, centrality, ssspDistance]
    Gem->>NAS: getIndexedBuffer(nodeIdToIndexMap) on every 30th frame while an analytic colour scheme or quality gate is on (GemNodes.tsx:706-709),<br/>and every TICK_INTERVAL frames in ClusterHulls.tsx:241-245. nodeAnalyticsStore.ts:96
    Note over WSBin,Gem: see VC-32 for the binary frame layout, VC-15 for GPU-side computation,<br/>VC-31 for the graph feature's own render-pipeline internals (not redrawn here)
```
## VC-34.13 physics — live simulation control (distinct from settings persistence)
```mermaid
sequenceDiagram
    autonumber
    participant Hook as usePhysicsService<br/>client/src/features/physics/hooks/usePhysicsService.ts:86
    participant UAC as unifiedApiClient
    participant SRV as Rust /api/physics handlers

    Hook->>UAC: GET /api/physics/status usePhysicsService.ts:101
    Hook->>UAC: POST /api/physics/start {simulation_id} usePhysicsService.ts:138
    Hook->>UAC: POST /api/physics/stop usePhysicsService.ts:170
    Hook->>UAC: POST /api/physics/parameters (params) usePhysicsService.ts:198
    Hook->>UAC: POST /api/physics/step usePhysicsService.ts:222
    Hook->>UAC: POST /api/physics/reset usePhysicsService.ts:250
    Hook->>UAC: POST /api/physics/optimize usePhysicsService.ts:279
    Hook->>UAC: POST /api/physics/nodes/pin {nodes} usePhysicsService.ts:307
    Hook->>UAC: POST /api/physics/nodes/unpin {node_ids} usePhysicsService.ts:331
    Hook->>UAC: POST /api/physics/forces/apply {forces} usePhysicsService.ts:355
    opt after start, stop, step and reset only
        Hook->>UAC: GET /api/physics/status - refetch to confirm applied state usePhysicsService.ts:147, usePhysicsService.ts:177, usePhysicsService.ts:229, usePhysicsService.ts:257
    end
    Note over Hook,UAC: DEBT: no component calls usePhysicsService outside its tests, and every path leads with /api although<br/>unifiedApiClient prepends /api (client/src/services/api/UnifiedApiClient.ts:77, :182), so each call would reach /api/api/physics/...
    UAC->>SRV: HTTP request
    Note right of Hook: /api/physics/* (this hook) is a live-simulation control<br/>surface, separate from /api/settings/physics (VC-34.1) which persists PhysicsSettings config
```
## VC-34.14 command-palette, help and onboarding — CustomEvent dispatch chain
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CP as CommandPalette.tsx<br/>client/src/features/command-palette/components/CommandPalette.tsx
    participant UCP as useCommandPalette<br/>client/src/features/command-palette/hooks/useCommandPalette.ts:10
    participant Reg as CommandRegistry<br/>client/src/features/command-palette/CommandRegistry.ts:6
    participant Def as defaultCommands.ts<br/>client/src/features/command-palette/defaultCommands.ts
    participant Help as helpRegistry / HelpProvider<br/>client/src/features/help/HelpRegistry.ts:3
    participant Onb as OnboardingEventHandler<br/>client/src/features/onboarding/components/OnboardingEventHandler.tsx:11
    participant Store as useSettingsStore

    U->>CP: Ctrl+K opens palette, types query
    CP->>UCP: filter Reg.getCommands() by query/category
    opt Jev palette-search enabled (server-driven, jevClient.ts:122-124, on under Compose via docker-compose.unified.yml:21, the server routes it through the shared agentbox proxy :7-11 by default, fail-open)
        UCP->>UCP: after a 350 ms debounce, jevRankPalette(query, enabled commands)<br/>client/src/features/command-palette/hooks/useCommandPalette.ts:47-60
        UCP-->>CP: suggestedCommands appended AFTER the lexical results under "Suggested (semantic)", never reordering them<br/>client/src/services/jev/jevClient.ts:156, :182
    end
    U->>CP: select command
    CP->>Reg: command.handler() execute
    alt id nav.help
        Reg->>Def: window.dispatchEvent(show-help) defaultCommands.ts:79, defaultCommands.ts:86
    else id help.search
        Reg->>Def: window.dispatchEvent(search-help) defaultCommands.ts:92, defaultCommands.ts:100
        Note over Def,Help: DEBT: nothing in client/src listens for show-help or search-help (git grep finds only these two dispatches),<br/>so both commands do nothing. HelpProvider does not subscribe to either event
    else id help.tour
        Reg->>Def: window.dispatchEvent(start-tour) defaultCommands.ts:116, defaultCommands.ts:123
        Def->>Onb: window.addEventListener start-tour OnboardingEventHandler.tsx:32
        Onb->>Onb: startFlow(welcomeFlow, forceRestart true) OnboardingEventHandler.tsx:24
    else id settings.reset
        Reg->>Store: resetSettings() then window.location.reload() defaultCommands.ts:162-163
    else id settings.export
        Reg->>Store: exportSettings() - Blob download, client-only defaultCommands.ts:175
    else id settings.import
        Reg->>Store: importSettings(text) then reload defaultCommands.ts:201-202
    else id system.save
        Reg->>Store: flushPendingUpdates() - forces autoSaveManager.forceFlush (VC-34.1) defaultCommands.ts:277
    end
    Note over CP,Onb: help.tour is the only command-palette to onboarding link, and the only help command with a listener -<br/>onboarding otherwise self-contained via localStorage.onboarding.completedFlows
```
## VC-34.15 design-system — presentational primitives, no network calls of its own
```mermaid
flowchart LR
    DS["design-system/components/*.tsx<br/>Alert, Badge, Button, Card, Dialog, Input,<br/>Select, Slider, Switch, Tabs, Toast, Tooltip, SearchInput..."]
    MD["MarkdownRenderer.tsx<br/>client/src/features/design-system/patterns/MarkdownRenderer.tsx"]
    Anim["animations.ts<br/>client/src/features/design-system/animations.ts"]
    HD["monitoring/components/HealthDashboard.tsx:13-17<br/>Card, Label, Button, Badge, Toast"]
    SPC["analytics/components/ShortestPathControls.tsx:2-21<br/>Card, Select, Input, Tabs, Toast, ScrollArea"]
    SP["control-center/panels/SettingsPanel.tsx:25<br/>SearchInput"]
    DS --> HD
    DS --> SPC
    DS --> SP
    Anim -.-> DS
    MD -.-> DS
    Note1["No fetch/axios/unifiedApiClient/WebSocket anywhere under<br/>client/src/features/design-system - a pure component library, 32 importers across other feature dirs"]
```
## VC-34.17 workspace — CRUD REST plus realtime WebSocket (implementation lives outside features/)
```mermaid
sequenceDiagram
    autonumber
    participant Comp as consumer component
    participant Hook as useWorkspaces<br/>client/src/hooks/useWorkspaces.ts:35
    participant API as workspaceApi<br/>client/src/api/workspaceApi.ts:136
    participant UAC as unifiedApiClient
    participant SRV as Rust /workspace handlers
    participant WS as ws://host/ws/workspaces<br/>client/src/hooks/useWorkspaces.ts:363

    Note over Comp,Hook: RESOLVED ADR-2077: the empty client/src/features/workspace/ stub<br/>is deleted. The real implementation always lived in client/src/hooks<br/>and client/src/api and is unchanged.
    Comp->>Hook: mount, initialLoad true
    Hook->>Hook: isValidCache check - 5 min cacheTimeout, skip fetch on hit useWorkspaces.ts:40, useWorkspaces.ts:63-70
    Hook->>API: fetchWorkspaces({page, limit, ...filters})
    API->>UAC: GET /workspace/list?params under the client base /api workspaceApi.ts:5, workspaceApi.ts:80
    UAC->>SRV: HTTP GET
    SRV-->>API: {success, data:{workspaces, total, page, hasMore}}
    API->>API: transformDates - createdAt/updatedAt/lastAccessed to Date workspaceApi.ts:118
    Hook->>WS: new WebSocket ws(s)://host/ws/workspaces
    loop server pushes
        SRV-->>WS: {type: workspace_updated|workspace_created|workspace_deleted, data}
        WS-->>Hook: onmessage - optimisticallyUpdateWorkspace or splice state useWorkspaces.ts:378-397
    end
    alt WS closes with code != 1000
        WS-->>Hook: onclose
        Hook->>Hook: setTimeout(connectWebSocket, 5000) - manual reconnect useWorkspaces.ts:404-409
    end
    Note over Hook,WS: DEBT: no server route serves /ws/workspaces (git grep over src finds none, while the REST scope is mounted at<br/>src/main.rs:1172), so the socket never opens and the 5 s reconnect retries forever. useWorkspaces itself has no caller in client/src
    Comp->>Hook: createWorkspace(data)
    Hook->>API: createWorkspace - validates name length <=100 workspaceApi.ts:173-174
    API->>UAC: POST /workspace/create
    Comp->>Hook: updateWorkspace(id, data)
    Hook->>Hook: optimisticallyUpdateWorkspace(id, data) - applied before network
    Hook->>API: updateWorkspace -> PUT /workspace/{id}
    alt PUT fails
        Hook->>Hook: refresh() - full re-fetch discards optimistic state
    end
    Comp->>Hook: deleteWorkspace / toggleFavorite / archiveWorkspace
    Hook->>API: DELETE /workspace/{id} | POST /workspace/{id}/favorite | POST /workspace/{id}/archive
```
## VC-34.18 solid — feature-owned Pod UI components (own composition only)
```mermaid
flowchart TD
    Tab["SolidTabContent.tsx:48<br/>client/src/features/solid/components/SolidTabContent.tsx"] --> PS["PodSettings.tsx:94"]
    Tab --> PB["PodBrowser.tsx:222"]
    Tab --> RE["ResourceEditor.tsx:134"]
    PS -->|useSolidPod| USP["useSolidPod.ts:32<br/>checkPod, createPod, deletePod, initPod auto-provision"]
    PB -->|useSolidPod + useSolidContainer| USC["useSolidContainer.ts:31<br/>containerPath listing"]
    RE -->|useSolidResource| USR["useSolidResource.ts:30"]
    USP --> SPS["solidPodService<br/>client/src/services/SolidPodService.ts"]
    USC --> SPS
    USR --> SPS
    USP -->|useNostrAuth| Auth["nostrAuth / useNostrAuth"]
    PS -.-> DSComp["design-system Button/Card/Badge/Input/Label - see VC-34.15"]
    PB -.-> DSComp
    RE -.-> DSComp
    Note1["publicUrl() in useSolidPod.ts:16-20 rewrites internal JSS Docker<br/>hostnames (visionclaw-jss/jss/localhost) to the /solid/ public proxy path"]
    Note2["see VC-33 for Nostr NIP-98 auth issuance, VC-26 for the JSS<br/>Solid pod data path itself - not redrawn here"]
```
## VC-34.19 visualisation — CommandInput natural-language command bar
```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant CI as CommandInput.tsx<br/>client/src/features/visualisation/components/CommandInput.tsx
    participant UAC as unifiedApiClient
    participant Pod as SolidPodService<br/>client/src/services/SolidPodService.ts
    participant SRV as Rust handlers

    participant SW as applySettingWrites<br/>client/src/features/control-center/state/settingWrites.ts:42
    U->>CI: types command, submit
    CI->>CI: parseCommandToActions(cmd) - keyword match against lower-cased text CommandInput.tsx:176, CommandInput.tsx:673
    Note over CI,SW: ADR-2189 - every settings action is now a writes list, not a direct PUT or a store mutation,<br/>so chat edits honour the Sync switch and the layout side effect (settingWrites.ts:12-14, CommandInput.tsx:46-53)
    alt matches a dial-owned or physics command (spread, springs, damping, reset, labels)
        CI->>CI: spread and label-budget wording reuse the dock dial transfer functions MACROS_BY_ID.density and .focus,<br/>reset reads DEFAULT_PHYSICS_SETTINGS instead of restating numbers CommandInput.tsx:201-205, CommandInput.tsx:392, CommandInput.tsx:400, CommandInput.tsx:334-340
    else edge transparency wording
        CI->>CI: edgeOpacityFromCommand(lower) - a stated value is a fraction or percentage of the Edge Opacity slider range 0 to 0.3,<br/>transparency inverts it, bare fainter or stronger words scale the default CommandInput.tsx:148, CommandInput.tsx:158-174
        Note over CI: INVARIANT: chat edge values stay inside the slider range - round clamps to EDGE_OPACITY_MAX (CommandInput.tsx:163),<br/>pinned by edgeOpacity.test.ts:29-37. The edge rule runs before the node rule, which used to swallow it (CommandInput.tsx:298-301)
    else any other opacity, transparent or visible wording
        CI->>CI: writes visualisation.graphs.knowledge.nodes.opacity CommandInput.tsx:313
        Note over CI: DEBT: no renderer reads knowledge nodes.opacity - the only client touch is the server-wins merge (coreSlice.ts:140).<br/>The chat reports Setting node opacity and nothing on screen changes
    else matches save/load/list view
        CI->>Pod: dynamic import SolidPodService then saveGraphView/loadGraphView/listGraphViews CommandInput.tsx:466, CommandInput.tsx:481, CommandInput.tsx:509
        CI->>CI: a loaded view's defined physics keys and nodeTypeVisibility become one writes list CommandInput.tsx:483-494
    else matches inject/show agent or swarm (and not node/only)
        CI->>UAC: POST /bots/mock-agents {agents: [...]} CommandInput.tsx:530
        UAC->>SRV: HTTP POST
        CI->>CI: window.dispatchEvent visionclaw:status with injected count CommandInput.tsx:539
    else no deterministic keyword match
        opt Jev command flow enabled (server-driven, jevClient.ts:122-124, on under Compose via docker-compose.unified.yml:21, server-side route is the shared agentbox proxy :7-11)
            CI->>CI: planJevCommand(cmd, registry fields) -- a patch or intent suggestion held for the user's decision, null on flag off, error, timeout or low confidence<br/>client/src/features/visualisation/components/CommandInput.tsx:682, client/src/services/jev/commandFlow.ts:43
            CI->>U: Apply, Run, Ask the settings assistant instead, or Dismiss -- Apply also goes through applySettingWrites, Jev itself writes nothing<br/>client/src/features/visualisation/components/CommandInput.tsx:728-732
        end
        CI->>CI: dispatchToSettingsLLM(command) when no review is held CommandInput.tsx:683-688
        CI->>UAC: POST /bots/settings-command {command, settingsContext} CommandInput.tsx:119, CommandInput.tsx:129
        Note right of CI: agent applies changes back through /api/settings/* -<br/>see VC-34.1 for that write path
    end
    CI->>SW: executeAction(action) - writes first, then localAction CommandInput.tsx:561-566
    CI->>CI: endpoint actions (non-setting jobs only) strip a leading /api and use<br/>unifiedApiClient.request(method, path, body) CommandInput.tsx:572-573
    Note over CI,SRV: raw fetch would send no auth and get 401 -<br/>unifiedApiClient's interceptor injects NIP-98/dev token (VC-33)
```
## VC-34.20 visualisation — EmbeddingCloudLayer and TransientBeamsLayer
```mermaid
sequenceDiagram
    autonumber
    participant ECL as EmbeddingCloudLayer.tsx<br/>client/src/features/visualisation/components/EmbeddingCloudLayer.tsx
    participant MCS as useMemoryCloudStore<br/>client/src/features/visualisation/memoryCloud/memoryCloudInstance.ts
    participant WSStore as useWebSocketStore
    participant Store as transientBeamStore<br/>client/src/store/transientBeamStore.ts
    participant TBH as useTransientBeams<br/>client/src/features/visualisation/hooks/useTransientBeams.ts:24
    participant TBL as TransientBeamsLayer<br/>client/src/features/visualisation/components/TransientBeamsLayer.tsx:69

    ECL->>MCS: reads snapshot, status, query run and response (selectors)<br/>EmbeddingCloudLayer.tsx:133
    Note right of MCS: points are a stratified PCA-projected sample from GET /api/memory-cloud, loaded by the memory explorer store.<br/>The static /embedding-cloud.json is retired.<br/>EmbeddingCloudLayer.tsx:4
    ECL->>ECL: enabled flag check, frame the cloud on the graph (cloudFrame.ts) and slice to maxPoints
    WSStore->>ECL: on('memoryFlash', event)<br/>EmbeddingCloudLayer.tsx:333
    ECL->>ECL: resolveFlashTargets -- the flashed key, else a few rows of its namespace, else counted as unmatched for the explorer HUD
    ECL->>MCS: recordFlash(match)<br/>EmbeddingCloudLayer.tsx:337
    ECL->>ECL: spawnBurst with semanticBurstColor at each matched point
    Note over ECL,MCS: while a query route is shown the cloud stops rotating and off-route points dim.<br/>The explorer panel is now a Memory tab of the right drawer, mounted only while the cloud is on<br/>(client/src/features/control-center/drawer/RightDrawer.tsx:91, :291, :301, see VC-57)
    Note over Store,TBH: 0x23 AGENT_ACTION frame decoded in binaryProtocol.ts<br/>pushes into transientBeamStore, state frames dropped (transientBeamStore.ts:70-72) - see VC-32.11 for the dispatch
    TBH->>Store: read beams, pruneExpired() useTransientBeams.ts:25-30
    TBL->>TBH: beams, prune() called every R3F frame TransientBeamsLayer.tsx:78-80
    TBL->>TBL: beamRadius prop default 0.35 - matches AgentVisualSettings.beamRadius default (VC-34.1 schema) TransientBeamsLayer.tsx:72
    Note over ECL,TBL: node/edge render layers themselves are VC-31 -<br/>this diagram covers only these two features' own data intake
```

## VC-34.21 voice — push-to-talk agent binding (graph selection to governed dispatch)
```mermaid
sequenceDiagram
    autonumber
    participant Graph as visionclaw:node-selected event<br/>window CustomEvent
    participant Hook as usePushToTalkAgentBinding<br/>client/src/features/voice/usePushToTalkAgentBinding.ts:41
    participant Pure as pttAgentBinding.ts pure helpers<br/>client/src/features/voice/pttAgentBinding.ts
    participant PTT as PushToTalkService<br/>client/src/services/PushToTalkService.ts
    participant VWS as VoiceWebSocketService<br/>client/src/services/VoiceWebSocketService.ts

    Hook->>PTT: activate(userId) on mount usePushToTalkAgentBinding.ts:49
    Graph-->>Hook: node-selected {nodeId, did_nostr, metadata}
    Hook->>Pure: resolveSelectedAgentDid(detail)
    Pure->>Pure: isCanonicalDid - did:nostr:64-hex regex (ADR-125 I1) pttAgentBinding.ts:11-12
    Hook->>PTT: setSelectedAgentDid(did or null) usePushToTalkAgentBinding.ts:56-58
    Note right of Pure: a non-agent node metadata carries no DID, resolves null,<br/>so PTT unbinds and never targets a non-agent
    PTT->>Hook: onServerNotify(active, did) - every PTT edge
    Hook->>VWS: setPtt(active, did) - server session carries target usePushToTalkAgentBinding.ts:62-63, VoiceWebSocketService.ts:272
    Hook->>Hook: handleTranscription(text, isFinal) fed from useVoiceInteraction
    Hook->>Pure: shouldDispatchGoverned(ptt.getState(), did), final transcripts only usePushToTalkAgentBinding.ts:73-82, pttAgentBinding.ts:64
    alt state commanding and did canonical
        Hook->>VWS: sendVoiceCommand(text, did), the SERVER then takes the governed path, signed 31402 to /v1/voice-intent<br/>usePushToTalkAgentBinding.ts:85, VoiceWebSocketService.ts:288-295
    else not commanding or unbound
        Hook->>Hook: inert - this hook sends nothing, the transcript is left to the settings assistant path (VC-34.19)
    end
    Note over Graph,VWS: see VC-35 for the STT capture / audio round-trip itself - not redrawn here
```
