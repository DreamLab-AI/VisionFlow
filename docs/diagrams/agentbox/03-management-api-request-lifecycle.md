---
id: AB-03
title: Management API request lifecycle and route table
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/docs/INGRESS-identity.md
adrs: [ADR-2005, ADR-2013, ADR-2003]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/management-api/lib/authz.js
  - ../project/agentbox/management-api/middleware/auth.js
  - ../project/agentbox/management-api/middleware/privacy-filter.js
  - ../project/agentbox/management-api/routes/comfyui.js
  - ../project/agentbox/management-api/routes/system.js
  - ../project/agentbox/management-api/routes/well-known.js
  - ../project/agentbox/management-api/routes/uri-resolver.js
  - ../project/agentbox/management-api/routes/llm-marketplace.js
  - ../project/agentbox/management-api/routes/voice-intent.js
  - ../project/agentbox/management-api/lib/uris.js
  - ../project/agentbox/management-api/utils/agent-event-publisher.js
  - ../project/agentbox/management-api/lib/audit-chain.js
  - ../project/agentbox/management-api/lib/failure-taxonomy.js
  - ../project/agentbox/management-api/lib/execution-journal.js
  - ../project/agentbox/management-api/observability/metrics.js
  - ../project/agentbox/tests/contract/execution-journal.contract.spec.js
  - ../project/agentbox/docker-compose.yml
  - ../project/agentbox/management-api/adapters/events/local-jsonl.js
  - ../project/agentbox/management-api/lib/elevation-stage.js
  - ../project/agentbox/management-api/middleware/consumer-payer.js
  - ../project/agentbox/management-api/middleware/spend-policy.js
  - ../project/agentbox/management-api/routes/admin-users.js
  - ../project/agentbox/management-api/routes/agent-events.js
  - ../project/agentbox/management-api/routes/approvals.js
  - ../project/agentbox/management-api/routes/beads.js
  - ../project/agentbox/management-api/routes/broker-bridge.js
  - ../project/agentbox/management-api/routes/dream.js
  - ../project/agentbox/management-api/routes/exec-record.js
  - ../project/agentbox/management-api/lib/action-plane.js
  - ../project/agentbox/management-api/routes/git-bridge.js
  - ../project/agentbox/management-api/routes/kg-elevation.js
  - ../project/agentbox/management-api/routes/linked-objects.js
  - ../project/agentbox/management-api/routes/mandate.js
  - ../project/agentbox/management-api/routes/memory.js
  - ../project/agentbox/management-api/routes/payments.js
  - ../project/agentbox/management-api/routes/pod-git.js
  - ../project/agentbox/management-api/routes/projects.js
  - ../project/agentbox/management-api/routes/sessions-boundary.js
  - ../project/agentbox/management-api/routes/tasks.js
  - ../project/agentbox/scripts/ci/check-ports-loopback.mjs
  - ../project/agentbox/management-api/routes/chain-payments.js
  - ../project/agentbox/config/nip98-proxy/proxy.mjs
  - ../project/agentbox/mcp/servers/nostr-bridge.js
  - ../project/agentbox/management-api/lib/agent-event-auth.js
  - ../project/agentbox/flake.nix
  - ../project/agentbox/scripts/agentbox-config-validate.js
verified_commit: c57d9c0508f4044915f9aecea4b9aa5629cc6a35
---

## AB-03.1 server.js boot part 1 — Fastify construction, hooks, static route registers

```mermaid
sequenceDiagram
    autonumber
    participant Boot as server.js top-level
    participant App as fastify()<br/>server.js:84
    participant RawBody as registerRawBody<br/>middleware/auth.js:262
    participant CORS as fastify-cors<br/>server.js:189
    participant WS as fastify-websocket<br/>server.js:199
    participant RL as fastify-rate-limit<br/>server.js:202
    participant Swag as fastify-swagger(-ui)<br/>server.js:279-338
    participant Routes as route modules<br/>server.js:341-434

    Boot->>App: fastify({loggerInstance, trustProxy true, routerOptions.maxParamLength 512})<br/>server.js:84-98
    Note over App: INVARIANT maxParamLength 512 (server.js:96) so urn-agentbox-bead content-addressed ids over 100 chars do not 404 in find-my-way.<br/>Fastify 5 (2026-10-01) moved it under routerOptions and the pino instance to loggerInstance (server.js:87)
    Boot->>RawBody: registerRawBody(app)<br/>server.js:107
    RawBody->>App: addContentTypeParser application-json parseAs buffer<br/>middleware/auth.js:263-279
    Note over RawBody: sets req.rawBody Buffer, still delivers parsed JSON, must run before route parsers (server.js:100-107)
    Boot->>App: app.register(cors, allowedOrigins credentials true, methods GET HEAD PUT PATCH POST DELETE)<br/>server.js:189-196
    Note over App: @fastify/cors 11 narrowed the default methods to GET HEAD POST, so the pre-11 list<br/>is pinned explicitly to keep PUT PATCH DELETE preflights working (server.js:192-195)
    Boot->>App: app.register(websocket)<br/>server.js:199
    Boot->>App: app.register(rateLimit max 100 per 1 minute allowList 127.0.0.1)<br/>server.js:202-209
    Boot->>App: addHook onResponse metrics.recordHttpRequest<br/>server.js:214-222
    Note over App: see AB-03.3 for the http_request_duration_seconds histogram this hook feeds
    Boot->>App: addHook preValidation runs authMiddleware unless url on public allowlist<br/>server.js:235-276
    Note over App: public allowlist livez health ready metrics v1-meta lo-star docs-star well-known-did well-known-x402 (server.js:237-273)
    Boot->>App: app.register(fastify-swagger openapi 3.0.0)<br/>server.js:279-328
    Boot->>App: app.register(fastify-swagger-ui routePrefix /docs)<br/>server.js:330-338
    Boot->>Routes: app.register(routes/tasks) prefix empty<br/>server.js:341-346
    Boot->>Routes: app.register(routes/status)<br/>server.js:348-354
    Boot->>Routes: app.register(routes/comfyui)<br/>server.js:356-361
    Boot->>Routes: app.register(routes/agent-events)<br/>server.js:363-367
    Boot->>App: addHook onReady starts agent-events WS forwarder to host<br/>server.js:375-395
    Boot->>Routes: app.register(routes/memory) prefix empty<br/>server.js:399
    Boot->>Routes: app.register(routes/broker-bridge)<br/>server.js:404
    Boot->>Routes: app.register(routes/git-bridge)<br/>server.js:410
    Boot->>Routes: app.register(routes/pod-git)<br/>server.js:417
    Boot->>Routes: app.register(routes/payments) with metrics<br/>server.js:421
    Boot->>Routes: app.register(routes/llm-marketplace)<br/>server.js:425
    Boot->>Routes: app.register(routes/dream)<br/>server.js:429
    Boot->>Routes: app.register(routes/exec-record), POST /v1/exec/record, ADR-2071 Phase 1<br/>server.js:434, called operator-gated at server.js:433 but carries no requireOperator, see Tension below
    Boot->>App: app.get /livez /ready /health /health/pods /v1/meta /metrics /<br/>server.js:438-792
    Note over App: ADR-2003 management-api (server.js included) is a flake.nix npm derivation built into the image (flake.nix:786-789),<br/>not a runtime install, and not generated from agentbox.toml (see AB-02.1 for the surrounding boot phases)
```

**Tension:** `POST /v1/exec/record` is described as "operator-gated" (`../project/agentbox/management-api/server.js:433`, `../project/agentbox/management-api/routes/exec-record.js:6-7`), but the only gate is the global `preValidation` auth hook (`../project/agentbox/management-api/server.js:235-276`). The route attaches no `requireOperator` preHandler, unlike `routes/mandate.js` (`../project/agentbox/management-api/routes/mandate.js:311`). In `hybrid` or `nip98` mode any NIP-98-verified signer passes `authMiddleware` (`../project/agentbox/management-api/middleware/auth.js:194-199`), so any authenticated principal can append to the journal, not only the operator.

## AB-03.2 server.js boot part 2 — async start() manifest-gated route mounts

```mermaid
sequenceDiagram
    autonumber
    participant Start as start()<br/>server.js:917
    participant Manifest as loadManifest<br/>adapters/manifest-loader.js
    participant App as fastify app
    participant Adapters as resolveAdapters<br/>adapters/index.js

    Start->>Manifest: loadManifest()<br/>server.js:922
    alt manifest file missing
        Manifest-->>Start: ManifestNotFound<br/>server.js:925-927
        Start->>Start: manifest = {} all-off defaults
    end
    Start->>Adapters: resolveAdapters(manifest)<br/>server.js:948
    Start->>App: app.decorate adapters resolvedAdapters<br/>server.js:949
    Note over Start,Adapters: see AB-04.1 for slot to implementation resolution detail
    Start->>App: middleware/linked-data createEncoder ALWAYS runs and decorates linkedData,<br/>linked_data.enabled only changes the log level on success or failure<br/>server.js:955-977
    Start->>App: register routes/uri-resolver always mounted<br/>server.js:986
    Start->>App: register routes/chain (self-gates 503 when sidechain off)<br/>server.js:999
    Start->>App: register routes/chain-payments, POST /v1/chain/pay with spend-policy in rail mode<br/>server.js:1012
    Start->>App: register routes/system logger manifest adapters execution snapshot<br/>server.js:1029-1040
    Start->>App: register routes/voice-intent with dispatchActionRequest<br/>server.js:1057-1058
    Start->>App: register routes/kg-elevation self-gates on sovereign_mesh.kg_elevation<br/>server.js:1072
    Start->>App: new ProjectTracker + register routes/projects<br/>server.js:1094-1102
    opt project_tracking.enabled true
        Start->>Start: tracker.scan() then tracker.startScheduler()<br/>server.js:1104-1111
    end
    alt sovereign_mesh.multi_user.enabled true
        Start->>App: register routes/admin-users<br/>server.js:1133-1135
    else disabled default
        Start->>Start: /admin/users/* not mounted<br/>server.js:1145
    end
    Start->>App: resolveViewerImpl + register routes/linked-objects<br/>server.js:1155-1158
    Start->>App: register routes/well-known x402 discovery<br/>server.js:1178
    Start->>App: buildAuthorityConsumer + buildAuthorityJournal + buildAuthorityGate, decorate authorityDenyJournal authorityConsumer authorityGate<br/>server.js:1197-1229
    Start->>App: bootReceiptPublisher over the same deny journal (boot flush, then one per minute), decorate governanceReceiptPublisher, stop on close<br/>server.js:1223-1227
    Note over Start: fallback to governance-decision-waiter.awaitDecision when the authority consumer is null (server.js:1211-1213)
    Start->>App: register routes/beads<br/>server.js:1243
    Start->>App: register routes/mandate<br/>server.js:1254
    Start->>App: register routes/sessions-boundary<br/>server.js:1264
    Start->>App: register routes/approvals<br/>server.js:1274
    Start->>App: register routes/briefing with processManager, ADR-2072<br/>server.js:1289
    Start->>Start: log SecurityProfileApplied resolved posture<br/>server.js:1349-1355
    Start->>Adapters: connectAdapters per-slot deadline<br/>server.js:1365-1381
    Note over Start,Adapters: see AB-04.6 connectAdapters per-slot deadline detail, not re-drawn here
    opt AGENTBOX_RELAY_ENABLED and AGENTBOX_RELAY_POD_BRIDGE true
        Start->>Start: new RelayConsumer(npubs...).start()<br/>server.js:1388-1389, :1432-1450
    end
    opt junkiejarvisEnabled(manifest)
        Start->>Start: startJunkieJarvis bridge logger<br/>server.js:1471-1490
    end
    Start->>Start: headroom.init(logger) decorate headroom or null<br/>server.js:1507-1515
    Start->>Start: initTracing, setBuildInfo, startMetricsServer<br/>server.js:1519-1521
    Start->>App: app.listen port PORT host HOST<br/>server.js:1524
    Note over Start: MANAGEMENT_API_PORT default 9090 (server.js:45) HOST default 0.0.0.0 (server.js:50)<br/>compose publishes 127.0.0.1 9090 9090 only
    Note over Start: agentbox main b9e0ea9a8 (S-3) leaves the 0.0.0.0 bind alone and instead makes the two readers of X-Agentbox-Pubkey (agent-event-auth and action-plane resolveAgentDid)<br/>honour it only from a loopback TCP peer (lib/agent-event-auth.js:98-106, :116-117, AGENTBOX_PROXY_PEERS overrides the peer list), so a cross-container<br/>MANAGEMENT_API_KEY holder can no longer assert an identity through it. The auth gate itself never reads the header, see AB-10.14
```

**Drift:** the route tables AB-03.6 to AB-03.11 do not draw four mounted surfaces: `routes/status` (`../project/agentbox/management-api/server.js:348`), `routes/chain` (`../project/agentbox/management-api/server.js:999`), `routes/chain-payments` (`../project/agentbox/management-api/server.js:1012`) and `routes/briefing` (`../project/agentbox/management-api/server.js:1289`). The topic is a partial route census, not the complete one its title implies.

## AB-03.3 generic request lifecycle — Fastify hook phases, auth and validation failure

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant PreVal as preValidation hook<br/>server.js:235
    participant Auth as authMiddleware<br/>middleware/auth.js:188
    participant Schema as Fastify schema validation
    participant PreH as route preHandler<br/>e.g. costGate/paymentGate/authz
    participant H as route handler
    participant OnResp as onResponse hook<br/>server.js:214

    C->>PreVal: HTTP request, Fastify starts its own reply timer at request start
    Note over C,PreVal: no app-level onRequest hook since the Fastify 5 upgrade - duration comes from<br/>reply.elapsedTime (server.js:211-215), which a cors preflight answered in onRequest still has.<br/>Body parsing (registerRawBody) runs before preValidation
    alt url on public allowlist
        PreVal->>H: skip authMiddleware entirely<br/>server.js:237-273
    else url requires auth
        PreVal->>Auth: authMiddleware(request, reply)<br/>server.js:275
        Auth->>Auth: verifyNip98Header + verifyBearerHeader<br/>middleware/auth.js:190-191
        Auth->>Auth: authMode gates which result counts — strict-nip98 and nip98 drop bearer<br/>middleware/auth.js:194-199
        opt neither accepted — task action token (lib/task-action-tokens.js)
            Auth->>Auth: verifyTaskTokenHeader, admitted only on TASK_TOKEN_ROUTES<br/>emit, batch, trajectory — else 403 (middleware/auth.js:139-144, :207-219)
        end
        alt no credential accepted for configured authMode
            Auth-->>C: 401 Unauthorized typed message<br/>middleware/auth.js:222-237
        else authenticated
            Auth->>Auth: request.auth = {mode nip98 or bearer or task-token, pubkey?}<br/>middleware/auth.js:217, :240
            Auth->>Schema: continue to schema validation
        end
    end
    alt request body fails JSON schema
        Schema-->>C: 400 Bad Request fastify default error shape
    else schema ok
        Schema->>PreH: continue
    end
    opt route declares a preHandler
        PreH->>PreH: e.g. costGate routes/tasks.js:43 or paymentGate routes/comfyui.js:74 or authz.requireOperator routes/mandate.js:311
        alt preHandler rejects
            PreH-->>C: 402 or 403 typed response, handler never runs
        end
    end
    PreH->>H: route handler executes
    H-->>C: reply.send(...)
    H->>OnResp: response phase
    OnResp->>OnResp: metrics.recordHttpRequest(method, routeOptions.url or url, statusCode, duration)<br/>server.js:214-221
    Note over OnResp: duration = reply.elapsedTime divided by 1000 (server.js:215), feeds<br/>http_request_duration_seconds (observability/metrics.js:58)
```

## AB-03.4 lib/authz.js — identity resolution predicates

```mermaid
sequenceDiagram
    autonumber
    participant Req as request.auth<br/>middleware/auth.js:240
    participant AP as authenticatedPubkey<br/>lib/authz.js:114
    participant OP as isOperator<br/>lib/authz.js:90
    participant AL as approvalAllowlist<br/>lib/authz.js:71
    participant SA as isSessionAgent<br/>lib/authz.js:160

    Note over Req: identity sources - NIP-98 Authorization re-verified here (the proxy passes it through verbatim)<br/>OR Bearer MANAGEMENT_API_KEY OR session-bound did in sessions-boundary registry
    Req->>AP: authenticatedPubkey(request)<br/>lib/authz.js:114-125
    alt auth.mode is nip98
        AP-->>AP: return normalised auth.pubkey if 64-hex<br/>lib/authz.js:116-119
    else auth.mode is bearer
        AP-->>AP: return operatorPubkey() — bearer IS the operator key<br/>lib/authz.js:120-123
    else no auth, or mode task-token
        AP-->>AP: return null — a task action token never resolves to a pubkey<br/>lib/authz.js:124
    end
    AP->>OP: isOperator(pubkey)<br/>lib/authz.js:90-94
    OP->>OP: pubkey equals operatorPubkey() from AGENTBOX_X_ONLY_PUBKEY_HEX or AGENTBOX_PUBKEY<br/>lib/authz.js:53-56
    AP->>AL: approvalAllowlist(manifest)<br/>lib/authz.js:71-87
    AL->>AL: operator plus sovereign_mesh.relay.allowed_pubkeys<br/>plus AGENTBOX_RELAY_ALLOWED_PUBKEYS or AGENTBOX_APPROVAL_ALLOWLIST
    Note over AL: isApprover(pk) is allowlist.has(pk) — lib/authz.js:97-101
    AP->>SA: isSessionAgent(pk, sessionId)<br/>lib/authz.js:160-170
    SA->>SA: reads AGENTBOX_STATE_DIR sessions JSON<br/>matches record.pubkey or hex inside record.did (lib/authz.js:136-153)
    Note over SA: default state dir var-lib-agentbox (lib/authz.js:137), independent of the on-disk filename hashing scheme
```

## AB-03.5 lib/authz.js — requireOperator and requireApprover preHandler gates

```mermaid
sequenceDiagram
    autonumber
    participant C as caller
    participant RO as requireOperator<br/>lib/authz.js:184
    participant RA as requireApprover<br/>lib/authz.js:207
    participant H as route handler

    rect rgb(232,244,255)
    Note over RO: consumed by routes/mandate.js create revoke list<br/>routes/mandate.js:311,389,437
    C->>RO: preHandler(request, reply)
    alt auth.mode is bearer
        RO->>H: allowed unconditionally — bearer IS the operator credential<br/>lib/authz.js:188
    else auth.mode is nip98 and isOperator(auth.pubkey)
        RO->>H: allowed<br/>lib/authz.js:189
    else neither condition holds
        RO-->>C: 403 forbidden_not_operator<br/>lib/authz.js:190-193
    end
    end

    rect rgb(255,240,230)
    Note over RA: preHandler form, plus an in-handler equivalent in routes/approvals.js:139-156
    C->>RA: preHandler(request, reply)
    alt auth.mode is bearer or pubkey missing
        RA-->>C: 403 forbidden_not_approver — bearer can never approve<br/>lib/authz.js:213-217
    else nip98 present
        RA->>RA: isApprover(auth.pubkey) checks approvalAllowlist<br/>lib/authz.js:218
        alt pubkey not on allowlist
            RA-->>C: 403 forbidden_not_approver<br/>lib/authz.js:219-224
        else on allowlist
            RA->>H: allowed
        end
    end
    end
    Note over RO,RA: ADR-2013 — 9096 is the ONE LAN-published ingress door<br/>(docker-compose.yml:47, sanctioned check-ports-loopback.mjs:94) — 9090 mgmt-api and<br/>9095 aoe serve are loopback-only (docker-compose.yml:48)
```

## AB-03.6 route table (a) — system, status, well-known, uri-resolver, meta probes

```mermaid
flowchart TD
    subgraph probes["public probes — auth-skip allowlist server.js:237-273"]
        A1["GET /livez<br/>server.js:438"] --> H1["event-loop-alive only<br/>server.js:453"]
        A2["GET /ready<br/>server.js:469"] --> H2["bootstrap+adapters+paths check, 503 with missing list<br/>server.js:494-577"]
        A3["GET /health<br/>server.js:582"] --> H3["adapterHealth snapshot plus adapter_failures<br/>server.js:608-619"]
        A4["GET /health/pods<br/>server.js:625"] --> H4["probePodHealth<br/>server.js:121-185"]
        A5["GET /v1/meta<br/>server.js:646"] --> H5["image_hash + adapter contract versions<br/>server.js:673-688"]
        A6["GET /metrics<br/>server.js:692"] --> H6["metrics.register.metrics<br/>server.js:705"]
        A7["GET /<br/>server.js:709"] --> H7["endpoint index<br/>server.js:727-790"]
    end
    subgraph sysroutes["routes/system.js — authed, always mounted"]
        B1["GET /v1/system<br/>routes/system.js:44"] --> HB1["buildSystemView + buildExecutionCoverage<br/>routes/system.js:58-61"]
        B2["GET /v1/system/audit-chain<br/>routes/system.js:65"] --> HB2["auditChain.verifyFiles(?days=N)<br/>routes/system.js:85-93 see AB-03.12"]
    end
    subgraph wellknown["public discovery"]
        C1["GET /.well-known/x402.json<br/>routes/well-known.js:64"] --> HC1["cached manifest or 404<br/>routes/well-known.js:34-52,90-95"]
        C2["GET /.well-known/did.json<br/>auth-skip only server.js:265"] -.->|"not a fastify route here"| HC2["served by solid-pod-rs<br/>lib/uris.js:278"]
    end
    subgraph uri["routes/uri-resolver.js — authed, always mounted"]
        D1["GET /v1/uri/:urn<br/>routes/uri-resolver.js:49"] --> HD1["400 malformed / 200+307 resolvable / 404 unknown —<br/>see AB-11.15 for the full alt-chain; 410 Gone is documented<br/>but WITHDRAWN, never emitted (ADR-2049) routes/uri-resolver.js:12-25"]
        D2["GET /v1/uri<br/>routes/uri-resolver.js:167"] --> HD2["resolver capability listing"]
    end
    Note1["DOC-DRIFT: the governing doc carries an EMPTY verified_commit<br/>(BASELINE-container.md:6), so nothing states the revision its route<br/>citations were read at. Every route line above was re-checked<br/>against the symbol at c57d9c0"]
```

## AB-03.7 route table (b) — tasks, beads, projects

```mermaid
flowchart TD
    subgraph tasks["routes/tasks.js"]
        T1["POST /v1/tasks<br/>tasks.js:16<br/>guard costGate tasks.js:43"] --> TH1["dispatchTaskSpawn via action plane<br/>tasks.js:8, no unjournalled spawn :76"]
        T2["GET /v1/tasks/:taskId<br/>tasks.js:107"] --> TH2["getTaskStatus<br/>tasks.js:148"]
        T3["GET /v1/tasks<br/>tasks.js:163"] --> TH3["getActiveTasks<br/>tasks.js:192"]
        T4["DELETE /v1/tasks/:taskId<br/>tasks.js:203"] --> TH4["stopTask<br/>tasks.js:244"]
        T5["GET /v1/tasks/:taskId/logs/stream<br/>tasks.js:273"] --> TH5["log tail<br/>tasks.js:316"]
    end
    subgraph beads["routes/beads.js — self-gates 503 when beads adapter off"]
        B1["GET /v1/beads<br/>beads.js:93"] --> BH1["list beads via adapter"]
        B2["GET /v1/beads/:id<br/>beads.js:121"] --> BH2["get bead"]
        B3["POST /v1/beads/epics<br/>beads.js:140"] --> BH3["create epic — ADR-043 D4.3"]
        B4["POST /v1/beads/:id/children<br/>beads.js:168"] --> BH4["add child bead"]
        B5["POST /v1/beads/:id/deps<br/>beads.js:197"] --> BH5["addDependency — see AB-04.10"]
        B6["POST /v1/beads/:id/claim<br/>beads.js:234"] --> BH6["claim bead"]
        B7["POST /v1/beads/:id/close<br/>beads.js:258"] --> BH7["close bead"]
    end
    subgraph projects["routes/projects.js — 503 unless project_tracking.enabled"]
        P1["GET /v1/projects<br/>projects.js:106"] --> PH1["tracker.list"]
        P2["GET /v1/projects/:id<br/>projects.js:146"] --> PH2["tracker.get"]
        P3["GET /v1/projects/:id/activity<br/>projects.js:185"] --> PH3["git commit activity"]
        P4["POST /v1/projects/scan<br/>projects.js:233"] --> PH4["tracker.scan<br/>boot scan server.js:1106-1107"]
        P5["POST /v1/projects/:id/primer<br/>projects.js:274"] --> PH5["PrimerGenerator — server.js:1089"]
        P6["POST /v1/projects/:id/publish<br/>projects.js:327"] --> PH6["kind-30841 nostr digest publish"]
    end
```

## AB-03.8 route table (c) — memory, kg-elevation, linked-objects

```mermaid
flowchart TD
    subgraph memory["routes/memory.js — memory slot first, pods fallback, 503 only when BOTH are off (memory.js:209)"]
        M1["POST /v1/memory<br/>memory.js:126<br/>imports assertPrivacyFilterApplied and _markPrivacyApplied memory.js:25-28"] --> MH1["memory.store when the memory slot is live (memory.js:151-154)<br/>else a JSON-LD MemoryEntry written to the pod (memory.js:158-206)"]
        M2["GET /v1/memory/:key<br/>memory.js:213"] --> MH2["read memory entry"]
        M3["POST /v1/memory/search<br/>memory.js:246"] --> MH3["search memory<br/>over 2 results: optional Jev ranking before headroom, off by default,<br/>_rankMemoryResults memory.js:274, falls back to the same array"]
        M4["GET /v1/memory<br/>memory.js:288"] --> MH4["list memory keys"]
    end
    subgraph kg["routes/kg-elevation.js — self-gates on sovereign_mesh.kg_elevation"]
        K1["POST /v1/kg-elevation/scan<br/>kg-elevation.js:77<br/>guard verifyAgentEventRequest kg-elevation.js:114"] --> KH1["scan personal KG via memory adapter<br/>ADR-2116 gateElevation before federation — see AB-03.18<br/>emit LINK beams + ontology-propose descriptor"]
    end
    subgraph lo["routes/linked-objects.js — viewer.mountPath, auth-skip for /lo/*"]
        L1["GET /lo/*<br/>linked-objects.js:133"] --> LH1["static viewer bundle — no auth (server.js:251-253)"]
        L2["GET viewer/manifest.json<br/>linked-objects.js:154"] --> LH2["pane manifest"]
        L3["GET viewer/panes/:file<br/>linked-objects.js:168"] --> LH3["pane asset"]
        L4["GET viewer/proxy<br/>linked-objects.js:206"] --> LH4["proxy fetch of a resource for the viewer"]
        L5["GET viewer.mountPath and viewer.mountPath/*<br/>linked-objects.js:258,261,271,275"] --> LH5["viewer shell — enabled/disabled variants"]
    end
    Note1["INVARIANT: /v1/* data endpoints stay authed even when the /lo static bundle is public (server.js:247-253)"]
```

**Drift:** the panel previously labelled `POST /v1/memory` as guarded by `wrapWithPrivacyFilter` at `middleware/privacy-filter.js:649`. The route never calls that wrapper. It imports only the marker and assertion helpers (`../project/agentbox/management-api/routes/memory.js:25-28`). Redaction happens one layer down, where `wrapDispatch` wraps every adapter method (`../project/agentbox/management-api/observability/metrics.js:127`), see AB-04.4.

## AB-03.9 route table (d) — sessions-boundary, agent-events, admin-users, approvals, mandate

```mermaid
flowchart TD
    subgraph sb["routes/sessions-boundary.js — ADR-043 D4.1-D4.5"]
        S1["POST /v1/sessions/boundary<br/>sessions-boundary.js:167"] --> SH1["bind did:nostr + URN + beads epic + memory namespace"]
    end
    subgraph ae["routes/agent-events.js"]
        E1["GET /v1/agent-events/stream<br/>agent-events.js:58<br/>websocket true"] --> EH1["WS push of agent action events"]
        E2["GET /v1/agent-events<br/>agent-events.js:129"] --> EH2["poll recent events"]
        E3["POST /v1/agent-events/emit<br/>agent-events.js:320<br/>guard verifyAgentEventRequest + reconcileTaskScope + reconcileSourceUrn<br/>agent-events.js:377-398"] --> EH3["emitAgentAction — failure classified via AB-03.13"]
        E4["POST /v1/agent-events/batch<br/>agent-events.js:455<br/>same guard per event"] --> EH4["batch emit"]
        E9["POST /v1/agent-events/trajectory<br/>agent-events.js:571<br/>ADR-2164, task token confined to its own task (agent-events.js:635)"] --> EH9["persistTrajectory, 503 without the external-pg memory adapter<br/>agent-events.js:626-636"]
        E10["POST /v1/agent-events/routing-label<br/>agent-events.js:661"] --> EH10["persistRoutingLabels over the same pool<br/>agent-events.js:699"]
        E5["GET /v1/agent-events/types<br/>agent-events.js:721"] --> EH5["AgentActionType enum listing"]
        E6["POST /v1/agent-events/hook<br/>agent-events.js:755"] --> EH6["Claude Code hook ingest"]
        E7["GET /v1/agent-events/registry<br/>agent-events.js:805"] --> EH7["registry snapshot"]
        E8["GET /v1/agent-events/status<br/>agent-events.js:828"] --> EH8["forwarder status"]
    end
    subgraph au["routes/admin-users.js — mounted only if sovereign_mesh.multi_user.enabled"]
        AU1["POST /admin/users/provision<br/>admin-users.js:138"] --> AUH1["501 stub — see PRD-007 (server.js:1140)"]
        AU2["POST /admin/users/:pubkey/git-init<br/>admin-users.js:196"] --> AUH2["501 stub"]
        AU3["POST /admin/users/:pubkey/suspend<br/>admin-users.js:230"] --> AUH3["501 stub"]
        AU4["POST /admin/users/:pubkey/archive<br/>admin-users.js:246"] --> AUH4["501 stub"]
    end
    subgraph ap["routes/approvals.js — ADR-043 D4.7"]
        AP1["GET /v1/approvals<br/>approvals.js:55"] --> APH1["authorityConsumer.listPending or wired false<br/>then annotateApprovals, advisory review_aid, Jev option off by default<br/>approvals.js:88-95"]
        AP2["POST /v1/approvals/:id/decide<br/>approvals.js:99<br/>guard nip98-only + isApprover approvals.js:139-156"] --> APH2["sign+publish kind-31403 — see AB-03.5"]
    end
    subgraph md["routes/mandate.js — ADR-043 D4.5, guard requireOperator"]
        MD1["POST /v1/mandate<br/>mandate.js:310<br/>guard requireOperator mandate.js:311"] --> MDH1["create scoped WAC mandate"]
        MD2["POST /v1/mandate/revoke<br/>mandate.js:388<br/>guard requireOperator mandate.js:389"] --> MDH2["revoke mandate"]
        MD3["GET /v1/mandate<br/>mandate.js:436<br/>guard requireOperator mandate.js:437"] --> MDH3["list mandates"]
    end
    sb ~~~ ae
    ae ~~~ au
    au ~~~ ap
    ap ~~~ md
```

**Drift:** two agent-events write routes joined since the previous stamp, `POST /v1/agent-events/trajectory` and `POST /v1/agent-events/routing-label` (`../project/agentbox/management-api/routes/agent-events.js:571`, `../project/agentbox/management-api/routes/agent-events.js:661`). Trajectory joins emit and batch as the only three routes a per-task action token may reach (`../project/agentbox/management-api/middleware/auth.js:139-144`). Routing-label is not on that list, so a task token gets 403 there.

## AB-03.10 route table (e1) — payments, llm-marketplace, dream, comfyui

```mermaid
flowchart TD
    subgraph pay["routes/payments.js — x402 web ledger"]
        PY1["GET /v1/pay/info<br/>payments.js:147"] --> PYH1["ledger info"]
        PY2["GET /v1/pay/balance<br/>payments.js:205"] --> PYH2["balance via solid-pod-rs"]
        PY3["POST /v1/pay/deposit<br/>payments.js:286"] --> PYH3["deposit — consumed by middleware/consumer-payer.js"]
        PY4["POST /v1/pay/estimate<br/>payments.js:315"] --> PYH4["cost estimate"]
        PY5["POST /v1/pay/buy<br/>payments.js:402"] --> PYH5["buy compute/LLM resource"]
        PY6["POST /v1/pay/withdraw<br/>payments.js:525"] --> PYH6["withdraw"]
    end
    subgraph llm["routes/llm-marketplace.js — Nostr kinds 38300-38305"]
        LM1["POST /v1/llm/advertise<br/>llm-marketplace.js:100"] --> LMH1["advertise LLM compute"]
        LM2["DELETE /v1/llm/advertise<br/>llm-marketplace.js:161"] --> LMH2["withdraw advert"]
        LM3["GET /v1/llm/discover<br/>llm-marketplace.js:180"] --> LMH3["discover offers"]
        LM4["POST /v1/llm/request<br/>llm-marketplace.js:221"] --> LMH4["request a grant"]
        LM5["POST /v1/llm/grant<br/>llm-marketplace.js:275"] --> LMH5["grant access"]
        LM6["POST /v1/llm/deny<br/>llm-marketplace.js:341"] --> LMH6["deny request"]
        LM7["POST /v1/llm/receipt<br/>llm-marketplace.js:378"] --> LMH7["settle receipt"]
        LM8["POST /v1/llm/revoke<br/>llm-marketplace.js:438"] --> LMH8["revoke grant"]
        LM9["GET /v1/llm/grants<br/>llm-marketplace.js:534"] --> LMH9["list grants"]
        LM10["GET /v1/llm/stats<br/>llm-marketplace.js:556"] --> LMH10["marketplace stats"]
    end
    subgraph dream["routes/dream.js — operator-gated, ADR-055"]
        DR1["GET /dream/status<br/>dream.js:24"] --> DRH1["read-only per-repo dream ledgers"]
    end
    subgraph comfy["routes/comfyui.js — guard paymentGate GPU-metered"]
        CF1["POST /v1/comfyui/workflow<br/>comfyui.js:23<br/>guard paymentGate costSats 100 tier gpu comfyui.js:74"] --> CFH1["submit ComfyUI workflow"]
        CF2["GET /v1/comfyui/workflow/:workflowId<br/>comfyui.js:112"] --> CFH2["workflow status"]
        CF3["GET /v1/comfyui/models<br/>comfyui.js:163"] --> CFH3["list models"]
        CF4["GET /v1/comfyui/outputs<br/>comfyui.js:207"] --> CFH4["list outputs"]
        CF5["DELETE /v1/comfyui/workflow/:workflowId<br/>comfyui.js:251"] --> CFH5["cancel workflow"]
        CF6["GET /v1/comfyui/stream<br/>comfyui.js:319<br/>websocket true"] --> CFH6["WS progress stream"]
    end
    pay ~~~ llm
    llm ~~~ dream
    dream ~~~ comfy
```

## AB-03.11 route table (e2) — voice-intent, broker-bridge, git-bridge, pod-git

```mermaid
flowchart TD
    subgraph vi["routes/voice-intent.js — WS7, mandate-gated"]
        VI1["POST /v1/voice-intent<br/>voice-intent.js:85"] --> VIH1["transcript to agent intent<br/>optional Jev interpret after the speaker, mandate and target checks, off by default,<br/>422 clarification-needed holds a doubted intent (voice-intent.js:227-229)<br/>dispatch signed 31402, or 503 dispatch-unavailable when no dispatcher is wired (voice-intent.js:237-241;<br/>dispatcher built by buildVoiceIntentDispatcher server.js:878, :1057)"]
    end
    subgraph bb["routes/broker-bridge.js — G6"]
        BB1["GET /api/broker/bridge/inbox<br/>broker-bridge.js:316"] --> BBH1["enrichment review inbox"]
        BB2["GET /api/broker/bridge/cases/:id<br/>broker-bridge.js:394"] --> BBH2["case detail"]
        BB3["POST /api/broker/bridge/cases/:id/decide<br/>broker-bridge.js:435<br/>guard authorityGate().guard, signed-31403 broker-bridge.js:509<br/>optional effective_tier forwarded in the payload broker-bridge.js:461,497"] --> BBH3["operation-bound signed approval<br/>durable received claim before mutation<br/>applied receipt requires committed upstream acknowledgement"]
        BBH3 --> BBH3B["receipt ladder mirrored to the forum<br/>mirrorReceipt broker-bridge.js:274<br/>consumer-received before the mutation broker-bridge.js:556<br/>then applied or not-applied broker-bridge.js:615<br/>INVARIANT: an unknown outcome posts NOTHING, broker-bridge.js:591"]
        BB4["GET /api/broker/bridge/events<br/>broker-bridge.js:842"] --> BBH4["event feed"]
        BB5["GET /api/broker/bridge/cases/:id/history<br/>broker-bridge.js:915"] --> BBH5["case history"]
    end
    subgraph gb["routes/git-bridge.js — G5, BC20"]
        GB1["POST /v1/git/clone<br/>git-bridge.js:315"] --> GBH1["clone remote"]
        GB2["POST /v1/git/submit-enrichment<br/>git-bridge.js:436<br/>guard did match git-bridge.js:356,486"] --> GBH2["submit enrichment to judgment broker"]
        GB3["GET /v1/git/case-status/:caseId<br/>git-bridge.js:620"] --> GBH3["poll broker case status"]
        GB4["POST /v1/git/approve-callback<br/>git-bridge.js:676<br/>guard webhook HMAC sig, fail-closed git-bridge.js:720-730"] --> GBH4["broker decision callback"]
    end
    subgraph pg["routes/pod-git.js — JSS smart HTTP protocol, app.get/app.post"]
        PG1["GET /pods/:npub/.git/info/refs<br/>pod-git.js:166"] --> PGH1["git smart discovery"]
        PG2["POST /pods/:npub/.git/git-upload-pack<br/>pod-git.js:197"] --> PGH2["fetch/clone"]
        PG3["POST /pods/:npub/.git/git-receive-pack<br/>pod-git.js:221"] --> PGH3["push"]
        PG4["GET /pods/:npub/.git/HEAD<br/>pod-git.js:257"] --> PGH4["HEAD ref"]
        PG5["GET /pods/:npub/clone-url<br/>pod-git.js:279"] --> PGH5["resolve clone URL"]
    end
```

## AB-03.12 lib/audit-chain.js — hash-chained events append and verify

```mermaid
sequenceDiagram
    autonumber
    participant Adapter as LocalJsonlEventsAdapter.dispatch<br/>adapters/events/local-jsonl.js:61
    participant Chain as audit-chain.js<br/>lib/audit-chain.js
    participant Disk as YYYY-MM-DD.jsonl<br/>adapters/events/local-jsonl.js:114
    participant Sys as GET /v1/system/audit-chain<br/>routes/system.js:65

    Note over Adapter,Chain: hash = SHA256(prev_hash || canonical_json(record minus prev_hash,hash))<br/>lib/audit-chain.js:6 and :68-72
    Adapter->>Chain: _initChain — readTail(dir) on first dispatch<br/>local-jsonl.js:122-127, audit-chain.js:195-229
    Chain->>Disk: scan files newest first, walk lines backward, skip torn lines<br/>audit-chain.js:198-226
    alt a chained tail record found
        Disk-->>Chain: {prevHash record.hash, seq record.seq+1}<br/>audit-chain.js:216-219
    else legacy or empty tail
        Disk-->>Chain: {prevHash GENESIS_HASH, seq 0}<br/>audit-chain.js:200,222,228
    end
    Adapter->>Adapter: build record {ts,session_id,execution_id,kind,payload,seq}<br/>local-jsonl.js:66-73
    Adapter->>Chain: hashRecord(prevHash, record)<br/>local-jsonl.js:75, audit-chain.js:68-72
    Adapter->>Disk: append JSON line, only advance chain state on success<br/>local-jsonl.js:76-78,134-140
    Note over Adapter: session_urn threads execution-journal envelopes into this same chain — see AB-03.14

    Sys->>Chain: verifyFiles(files sorted by name)<br/>routes/system.js:78, :85, audit-chain.js:148-186
    loop per file in order
        Chain->>Chain: verifyLines — recompute hashRecord per line, compare prev_hash and hash<br/>audit-chain.js:89-138
        alt unchained record before any chain started
            Chain-->>Chain: legacy_prefix++ tolerated<br/>audit-chain.js:113-117
        else prev_hash mismatch
            Chain-->>Sys: ok false reason splice — broken_at index<br/>audit-chain.js:120-122
        else hash mismatch
            Chain-->>Sys: ok false reason edit — content altered<br/>audit-chain.js:123-126
        end
    end
    Chain-->>Sys: {ok, checked, legacy_prefix, tail_hash, broken_at}<br/>audit-chain.js:148-186
    opt result.ok is false
        Sys->>Sys: logger.warn audit-chain.broken<br/>routes/system.js:87
    end
    Note over Chain: INVARIANT deletion at the tail is the one tamper mode a bare hash chain cannot see (lib/audit-chain.js:14-16)
```

## AB-03.13 lib/failure-taxonomy.js — MAST classifier on agent-events failure returns

```mermaid
sequenceDiagram
    autonumber
    participant R as POST /v1/agent-events/emit or /batch<br/>routes/agent-events.js:320,455
    participant V as verifyAgentEventRequest / reconcileTaskScope / reconcileSourceUrn<br/>agent-events.js:377,385,391
    participant T as taxonomy.tagFailure<br/>lib/failure-taxonomy.js:171
    participant C as taxonomy.classify<br/>lib/failure-taxonomy.js:143
    participant Pub as agentEventPublisher.emitAgentAction<br/>utils/agent-event-publisher.js:50

    R->>V: verifyAgentEventRequest(request)<br/>agent-events.js:377
    alt auth.ok is false
        V-->>R: !auth.ok check<br/>agent-events.js:378
        R->>T: tagFailure({error auth.error})<br/>agent-events.js:382
        T->>C: classify(ctx) — no mode, no reason, no stderr match<br/>lib/failure-taxonomy.js:146-159
        C-->>T: UNMAPPED sentinel<br/>lib/failure-taxonomy.js:30,159
        T-->>R: {failure_mode unmapped, failure_detail auth.error}<br/>lib/failure-taxonomy.js:171-183
        R-->>R: reply.code(auth.status).send success false + tag<br/>agent-events.js:383
    else auth ok, a task token's scope disagrees with the body
        R->>V: reconcileTaskScope(auth.taskId, body)<br/>agent-events.js:385
        V-->>R: !scope.ok<br/>agent-events.js:386
        R->>T: tagFailure({reason REASON.IDENTITY_MISMATCH, error})<br/>agent-events.js:387
        R-->>R: reply.code(scope.status) success false + tag<br/>agent-events.js:388
    else auth ok, reconcile source_urn vs verified did
        R->>V: reconcileSourceUrn(claimed, auth.did)<br/>agent-events.js:391
        alt reconciliation fails — caller-claimed identity mismatch
            V-->>R: !rec.ok check<br/>agent-events.js:392
            R->>T: tagFailure({reason REASON.IDENTITY_MISMATCH, error})<br/>agent-events.js:396
            T->>C: classify — reason matches REASON_TO_MODE table<br/>lib/failure-taxonomy.js:148-150
            C-->>T: FM-1.2 Disobey Role Specification<br/>lib/failure-taxonomy.js:82,100
            T-->>R: {failure_mode FM-1.2, failure_detail preserved text}
            R-->>R: reply.code(rec.status).send success false + tag<br/>agent-events.js:397
        else reconciled
            R->>Pub: emitAgentAction(emitPayload)<br/>agent-events.js:441
            opt event is a failure (outcome, failure or failure_mode present)
                Pub->>C: a valid top-level failure_mode wins, else metadata.failure_mode is promoted,<br/>else classify(ctx) (utils/agent-event-publisher.js:114-129)
            end
        end
    end
    Note over C: 14 MAST modes across 3 categories (spec, inter-agent, verification) — Cemri et al. 2025<br/>lib/failure-taxonomy.js:33-57 — stderr heuristics are deliberately tiny, ambiguous<br/>text stays unmapped (lib/failure-taxonomy.js:119-124)
```

## AB-03.14 lib/execution-journal.js — ExecutionJournal.append envelope + D2 provenance gate

```mermaid
sequenceDiagram
    autonumber
    participant Caller as harness caller<br/>tests/contract/execution-journal.contract.spec.js
    participant J as ExecutionJournal.append<br/>lib/execution-journal.js:133
    participant Ev as injected events adapter<br/>opts.eventsAdapter (ADR-005 slot)
    participant M as assertModelRequestTraceable<br/>lib/execution-journal.js:208

    Caller->>J: append({session_urn, type, payload, ...})<br/>lib/execution-journal.js:133
    alt session_urn missing
        J-->>Caller: throw JournalError bad_event<br/>lib/execution-journal.js:135
    else type not in VOCABULARY
        J-->>Caller: throw JournalError bad_type — 15-member canonical vocabulary<br/>lib/execution-journal.js:40-56,136-138
    end
    opt event_id already committed for this session_urn
        J-->>Caller: return {envelope prior, duplicate true} — idempotent replay<br/>lib/execution-journal.js:146-149
    end
    J->>J: seq = _peekSeq(session_urn) — per-session monotonic, D1 unique key<br/>lib/execution-journal.js:108-109,152
    J->>J: build envelope {schema exec-event/1, event_id, seq, occurred_at, harness, agent_did, turn, type, payload, privacy_class}<br/>lib/execution-journal.js:155-167
    J->>Ev: dispatch({kind exec.<type>, session_id session_urn, execution_id eventId, payload envelope})<br/>lib/execution-journal.js:174-179
    Note over Ev: rides the ADR-005 events slot — inherits ADR-039 hash chain (see AB-03.12), ADR-008 privacy filter, ADR-012 JSON-LD
    Ev-->>J: dispatch resolves
    J->>J: commit _nextSeq and _seenIds only after successful append<br/>lib/execution-journal.js:182-185
    opt D2 strict mode — model-visible means journalled
        Caller->>M: assertModelRequestTraceable(request)<br/>lib/execution-journal.js:208
        alt any message/context item cites no journal seq
            M-->>Caller: throw UntraceableModelRequest (strict) or degraded model.requested (compatibility)<br/>lib/execution-journal.js:75-84
        end
    end
    Note over J: CORRECTED 2026-10-02 — the journal is live: lib/action-plane.js:169 constructs ExecutionJournal<br/>(tasks.js dispatchTaskSpawn rides that plane), and since ADR-2071 Phase 1 routes/exec-record.js:128<br/>appends through the same singleton, so the management API stays the only chain writer
```

## AB-03.15 end-to-end authenticated write — nip98-proxy through the mandate registry

```mermaid
sequenceDiagram
    autonumber
    participant Cli as external caller
    participant Proxy as nip98-proxy port 9096<br/>config/nip98-proxy/proxy.mjs
    participant App as fastify app port 9090<br/>server.js:84
    participant PreVal as preValidation hook<br/>server.js:235
    participant Auth as authMiddleware<br/>middleware/auth.js:188
    participant Guard as authz.requireOperator<br/>lib/authz.js:184
    participant H as routes/mandate.js POST /v1/mandate<br/>mandate.js:310

    rect rgb(232,244,255)
    Note over Cli,Proxy: LAN-published trust boundary — the ONE ingress door (ADR-2013, docker-compose.yml:47)
    Cli->>Proxy: HTTPS request to /mgmt/v1/mandate, NIP-98 kind-27235 Authorization header
    Proxy->>Proxy: verify the NIP-98 header with NostrBridge.verifyNip98<br/>proxy.mjs:577
    Proxy->>Proxy: /mgmt/ route strips its prefix (proxy.mjs:304-310) and re-attaches the<br/>NIP-98 Authorization verbatim for the upstream to re-verify (proxy.mjs:886-892)
    end
    rect rgb(240,255,235)
    Note over App,H: loopback-only surface — 127.0.0.1 9090 9090 (docker-compose.yml:48)
    Proxy->>App: POST /v1/mandate, Authorization Nostr base64-event, x-agentbox-pubkey, body JSON
    App->>App: addContentTypeParser buffers body into request.rawBody<br/>middleware/auth.js:263-279
    App->>PreVal: preValidation phase<br/>server.js:235
    PreVal->>Auth: authMiddleware(request, reply) — url not on public allowlist<br/>server.js:275
    Auth->>Auth: verifyNip98Header — verifyNip98(header, method, url, request.rawBody)<br/>middleware/auth.js:68-98, :83
    Note over Auth: the signed URL is rebuilt from request.host, which keeps a non-default port<br/>(Fastify 5 request.hostname drops it), middleware/auth.js:73
    alt signature invalid or bridge absent
        Auth-->>Cli: 401 Unauthorized<br/>middleware/auth.js:234-237
    else verified
        Auth->>Auth: request.auth = {mode nip98, pubkey, event}<br/>middleware/auth.js:93-97, :240
        Auth->>Guard: preHandler requireOperator({manifest})<br/>mandate.js:311, lib/authz.js:184
        alt auth.pubkey is not the operator key
            Guard-->>Cli: 403 forbidden_not_operator<br/>lib/authz.js:190-193
        else operator verified
            Guard->>H: handler runs, refuses a caller-supplied issuer that is not the operator (mandate.js:349-364)
            H->>H: createSignedMandate — mint, sign kind-30078 with the operator key or keep unsigned (mandate.js:191-219)
            H->>H: record in the local mandate registry (mandate.js:221-228)
            H-->>Cli: 201 mandate envelope<br/>mandate.js:377
        end
    end
    end
    App->>App: onResponse records http_request_duration_seconds<br/>server.js:214-221, observability/metrics.js:58
```

**Drift:** the earlier panel drew this write ending in an events-adapter dispatch (`mandate.created`, hash-chained into the JSONL log). `createSignedMandate` dispatches nothing to the events slot. It signs a kind-30078, or warns and keeps the mandate unsigned, and saves a local registry entry (`../project/agentbox/management-api/routes/mandate.js:191-231`). The panel also had the proxy forwarding `/mgmt/v1/mandate`, but the `/mgmt/` route strips its prefix (`../project/agentbox/config/nip98-proxy/proxy.mjs:304-310`).

**Open:** source does not show that the forwarded NIP-98 header verifies a second time at the management API. The proxy forwards the caller's header verbatim after stripping `/mgmt/` (`../project/agentbox/config/nip98-proxy/proxy.mjs:886-892`). The API rebuilds the URL from the stripped path (`../project/agentbox/management-api/middleware/auth.js:73`). `verifyNip98` accepts only an exact or suffix match of the signed `u` tag against that URL (`../project/agentbox/mcp/servers/nostr-bridge.js:545-546`). A tag that keeps `/mgmt/` is not a suffix match of the stripped URL, so this path may answer 401. Only a runtime request through `:9096/mgmt/` can settle it.

## AB-03.16 middleware/*.js inventory — which routes and hooks actually consume each file

```mermaid
flowchart LR
    A["middleware/auth.js<br/>createAuthMiddleware + registerRawBody"] --> A1["global preValidation hook<br/>server.js:225,235,275"]
    B["middleware/cost-gate.js<br/>costGate"] --> B1["preHandler on POST /v1/tasks<br/>routes/tasks.js:7,43"]
    C["middleware/payment-gate.js<br/>paymentGate"] --> C1["preHandler on POST /v1/comfyui/workflow<br/>routes/comfyui.js:10,74"]
    D["middleware/privacy-filter.js<br/>ADR-008 layer 2"] --> D1["marker and assertion helpers imported by routes/memory.js<br/>memory.js:25-28"]
    D --> D2["wrapWithPrivacyFilter wraps every adapter dispatch<br/>see AB-04.4/AB-04.5, not re-drawn here"]
    E["middleware/linked-data/*<br/>createEncoder, viewer, surfaces"] --> E1["createEncoder always runs in start(); surfaces gated by linked_data.enabled<br/>server.js:955-977"]
    E --> E2["routes/linked-objects.js viewer mount<br/>server.js:1155-1158"]
    F["middleware/spend-policy.js<br/>spendPolicy factory"] --> F1["required by routes/chain-payments.js:34,<br/>rail-mode hook built at chain-payments.js:146"]
    G["middleware/consumer-payer.js<br/>resolveSidestr"] --> G1["required by routes/chain-payments.js:35"]
    F1 --> R["POST /v1/chain/pay, mounted in start()<br/>server.js:1012"]
    G1 --> R
    Note1["no other routes/*.js or server.js require them; scripts/agentbox-config-validate.js<br/>only checks payments.consumer coherence for spend-policy, it does not import it"]
```

**Drift:** this panel previously drew `spend-policy.js` and `consumer-payer.js` as "consumed only by scripts/agentbox-config-validate.js", with no `require()` from any route. Its own note contradicted that, and it misattributed the `chain-payments.js` line numbers to the two middleware files. At the pin, `routes/chain-payments.js` requires both (`../project/agentbox/management-api/routes/chain-payments.js:34-35`). The validator mentions spend-policy only in its rule text (`../project/agentbox/scripts/agentbox-config-validate.js:1730-1755`).

## AB-03.18 lib/elevation-stage.js — gateElevation, ADR-2116 vault-propose gate before federation

```mermaid
sequenceDiagram
    autonumber
    participant K as POST /v1/kg-elevation/scan<br/>routes/kg-elevation.js:77
    participant G as gateElevation<br/>lib/elevation-stage.js:226
    participant Val as validateStandalone<br/>lib/elevation-stage.js:187
    participant Vault as runVaultPropose<br/>lib/ontology-propose.js
    participant Pub as elevationPublisher<br/>lib/elevation-publisher.js

    K->>K: extractProposals(entries) builds candidates<br/>kg-elevation.js:157-170
    loop per candidate p
        K->>G: gateElevation(p, {env})<br/>kg-elevation.js:182
        G->>G: resolveRelationTargets, then renderStagedPage — status draft, sources id origin<br/>lib/elevation-stage.js:238, :244, :131-177
        alt a relation target unresolved, render fails, or two pages collide
            G-->>K: {proposable false, blocked true, stage_issues}<br/>lib/elevation-stage.js:239-253, :286-298
        else staged
            G->>Val: vault validate --strict --json in scratch repo<br/>lib/elevation-stage.js:200-211
            alt validation fails — not DANGLING_LINK
                Val-->>G: disqualifying issues<br/>lib/elevation-stage.js:207-211
                G-->>K: {proposable false, blocked true, stage_issues}<br/>lib/elevation-stage.js:263
            else standalone ok
                G->>Vault: vault propose --diff staged page(s)<br/>lib/elevation-stage.js:265-279
                Vault-->>G: {proposal, blockers, blocked}
                G-->>K: outcome — proposable equals not blocked<br/>lib/elevation-stage.js:280
            end
        end
        K->>K: p.patch_proposal, p.blockers, p.stage_issues, p.propose_error<br/>kg-elevation.js:183-186
    end
    opt outcome.blocked
        K->>K: logger.debug kg-elevation.blocked — not federated<br/>kg-elevation.js:188-194
    end
    K->>Pub: every candidate still emits its LINK beam, only unblocked ones are published<br/>kg-elevation.js:217, :229-233
    Note over G: DANGLING_LINK is the one issue code the gate tolerates standalone<br/>every relation target is dangling in a one-page scratch repo by construction (lib/elevation-stage.js:23-28, :47)
```
