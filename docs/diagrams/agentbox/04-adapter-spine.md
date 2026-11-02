---
id: AB-04
title: Five-slot adapter spine, dispatch middleware and connect lifecycle
area: agentbox
governing:
  - ../project/agentbox/docs/BASELINE-container.md
adrs: [ADR-2004, ADR-2005, ADR-2035, ADR-2036, ADR-2037, ADR-2064]
sources:
  - ../project/agentbox/docs/BASELINE-container.md
  - ../project/agentbox/management-api/adapters/index.js
  - ../project/agentbox/management-api/adapters/base.js
  - ../project/agentbox/management-api/adapters/lifecycle.js
  - ../project/agentbox/management-api/adapters/contract-versions.js
  - ../project/agentbox/management-api/adapters/errors.js
  - ../project/agentbox/management-api/adapters/manifest-loader.js
  - ../project/agentbox/management-api/adapters/memory/embedded-ruvector.js
  - ../project/agentbox/management-api/adapters/memory/off.js
  - ../project/agentbox/management-api/adapters/beads/local-sqlite.js
  - ../project/agentbox/management-api/adapters/pods/_solid-http-base.js
  - ../project/agentbox/management-api/adapters/pods/local-solid-rs.js
  - ../project/agentbox/management-api/adapters/events/local-jsonl.js
  - ../project/agentbox/management-api/adapters/events/agent-execution-event.schema.json
  - ../project/agentbox/management-api/adapters/orchestrator/local-process-manager.js
  - ../project/agentbox/management-api/lib/ontology-apply.js
  - ../project/agentbox/management-api/observability/metrics.js
  - ../project/agentbox/management-api/middleware/privacy-filter.js
  - ../project/agentbox/management-api/lib/pod-signer.js
  - ../project/agentbox/management-api/server.js
  - ../project/agentbox/tests/contract/adapter-lifecycle.contract.spec.js
  - ../project/agentbox/tests/contract/memory.contract.spec.js
  - ../project/agentbox/agentbox.sh
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/scripts/agentbox-config-validate.js
  - ../project/agentbox/tests/contract/beads.contract.spec.js
  - ../project/agentbox/tests/contract/events.contract.spec.js
  - ../project/agentbox/tests/contract/memory-encoder-bypass.contract.spec.js
  - ../project/agentbox/tests/contract/orchestrator.contract.spec.js
  - ../project/agentbox/tests/contract/pods.contract.spec.js
  - ../project/agentbox/tests/contract/privacy-filter.contract.spec.js
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/management-api/routes/linked-objects.js
  - ../project/agentbox/management-api/routes/sessions-boundary.js
  - ../project/agentbox/management-api/lib/pg-config.js
  - ../project/agentbox/management-api/adapters/memory/external-pg.js
verified_commit: c57d9c0508f4044915f9aecea4b9aa5629cc6a35
---

## AB-04.1 resolveAdapters — slot to implementation resolution
```mermaid
flowchart TD
    M["agentbox.toml [adapters]<br/>manifest-loader.js loadManifest()"] --> RA["resolveAdapters(manifest)<br/>adapters/index.js:196"]
    RA --> LOOP["for slot of SLOTS<br/>index.js:200"]
    LOOP --> S1["SLOTS = beads, pods, memory, events, orchestrator<br/>index.js:20"]
    S1 --> IMPL["impl = adapterDecls[slot] || 'off'<br/>index.js:201"]
    IMPL --> REQ["requireImpl(slot, impl)<br/>index.js:126"]
    REQ --> P1{"require adapters/&lt;slot&gt;/&lt;impl&gt;.js<br/>index.js:127-129"}
    P1 -->|found| CFG
    P1 -->|"MODULE_NOT_FOUND and impl==='off'"| PH["fallback adapters/&lt;slot&gt;/placeholder.js<br/>index.js:135-139"]
    PH -->|found| CFG
    PH -->|missing| UAI["throw UnknownAdapterImpl(slot, impl)<br/>index.js:22 / :143"]
    P1 -->|"MODULE_NOT_FOUND, impl!=='off'"| UAI
    CFG["slotConfig(slot, impl, manifest)<br/>index.js:38"] --> CM["memory + external-pg<br/>pool config from integrations.ruvector_external conninfo,<br/>password from the one file via lib/pg-config index.js:44-51"]
    CFG --> CB["beads / events + external<br/>externalUrl from federation.external_url<br/>index.js:55-59"]
    CFG --> CP["pods: buildPodNip98(manifest)<br/>index.js:75 → lib/pod-signer.js"]
    CP --> CP2["local-solid-rs → baseUrl<br/>sp.base_url or http://sp.bind:sp.port default 127.0.0.1:8484<br/>index.js:93-97"]
    CFG --> CO["orchestrator + stdio-bridge<br/>externalUrl + protocol default 'stdio'<br/>index.js:103-108"]
    CM --> NEW
    CB --> NEW
    CP2 --> NEW
    CO --> NEW
    NEW["new AdapterClass(cfg)<br/>index.js:213-215"] --> INST["instrumentAdapter(adapter, slot, impl, manifest)<br/>index.js:227"]
    NEW -->|"slotConfig or constructor throws"| CFA["ConfigFailedAdapter whose connect() rethrows the cause<br/>index.js:179-188, :216-225"]
    CFA --> INST
    INST --> META["adapter._implName = impl<br/>adapter._slot = slot<br/>index.js:230-231"]
    META --> OUT["returns beads, pods, memory, events, orchestrator<br/>index.js:234"]
    UAI --> FAIL["startup aborts — no silent 'off' substitution<br/>INVARIANT ADR-2004"]
    CFA -.-> LC["lifecycle withdraws the slot as a connect failure, failure class config-failed<br/>lifecycle.js:277-279, see AB-04.7"]
```

**Drift:** since the previous stamp a slot whose configuration cannot be built no longer aborts boot. `resolveAdapters` substitutes a `ConfigFailedAdapter` whose `connect()` rethrows the cause (`../project/agentbox/management-api/adapters/index.js:179-188`, `../project/agentbox/management-api/adapters/index.js:216-225`). The lifecycle then quarantines it and records `config-failed` with the offending file path, never its value (`../project/agentbox/management-api/adapters/lifecycle.js:277-279`). An unknown implementation name still throws `UnknownAdapterImpl` before construction (`../project/agentbox/management-api/adapters/index.js:143`).

## AB-04.2 Adapter class hierarchy across the three implementation classes
```mermaid
classDiagram
    class BaseAdapter {
        +String slot
        +String impl
        +String CONTRACT_VERSION
        +Boolean enabled
        +constructor(slot, impl, contractVersion) base.js:20
    }
    class EmbeddedRuvectorMemoryAdapter {
        +store(key, value, namespace) embedded-ruvector.js:76
        +search(query, opts) embedded-ruvector.js:98
        +retrieve(key, namespace) embedded-ruvector.js:119
        +del(key, namespace) embedded-ruvector.js:137
        +list(namespace) embedded-ruvector.js:149
        -_ns(namespace) embedded-ruvector.js:63
    }
    class OffMemoryAdapter {
        +store() throws AdapterDisabled off.js:19
        +search() throws AdapterDisabled off.js:20
        +retrieve() throws AdapterDisabled off.js:21
        +del() throws AdapterDisabled off.js:22
        +list() throws AdapterDisabled off.js:23
    }
    class LocalSqliteBeadsAdapter {
        +createEpic(opts) local-sqlite.js:78
        +createChild(opts) local-sqlite.js:113
        +claim(id, actor) local-sqlite.js:145
        +close(id, outcome) local-sqlite.js:168
        +addDependency(childId, blockerId, type) local-sqlite.js:195
        +getReady(filter) local-sqlite.js:236
        +show(id) local-sqlite.js:259
    }
    class SolidHttpPodsAdapter {
        -_nip98 signer or null _solid-http-base.js:54
        -_requireSigned ADR-2064 fail-closed _solid-http-base.js:55
        -_fetch signed when signer OR requireSigned _solid-http-base.js:62-64
        +write(uri, body, contentType) _solid-http-base.js:119
        +read(uri) _solid-http-base.js:135
        +patch(uri, patch) _solid-http-base.js:150
        +del(uri) _solid-http-base.js:165
        +list(container, opts) _solid-http-base.js:178
    }
    class LocalSolidRsPodsAdapter {
        +probeCapabilities() local-solid-rs.js:51
        +list(container, opts) local-solid-rs.js:72
        +patch(uri, patch, opts) local-solid-rs.js:126
    }
    class LocalJsonlEventsAdapter {
        +dispatch(event) local-jsonl.js:61
        +subscribe(filter, handler) local-jsonl.js:94
        +unsubscribe(subscriptionId) local-jsonl.js:106
        -_initChain() local-jsonl.js:122
    }
    class LocalProcessManagerOrchestratorAdapter {
        +spawnAgent(spec) local-process-manager.js:45
        +streamEvent(agentId, handler) local-process-manager.js:106
        +listAgents() local-process-manager.js:119
        +handleGovernanceDecision(event, opts) local-process-manager.js:142
        +terminateAgent(agentId) local-process-manager.js:549
    }
    BaseAdapter <|-- EmbeddedRuvectorMemoryAdapter
    BaseAdapter <|-- OffMemoryAdapter
    BaseAdapter <|-- LocalSqliteBeadsAdapter
    BaseAdapter <|-- SolidHttpPodsAdapter
    SolidHttpPodsAdapter <|-- LocalSolidRsPodsAdapter
    BaseAdapter <|-- LocalJsonlEventsAdapter
    BaseAdapter <|-- LocalProcessManagerOrchestratorAdapter
```

## AB-04.3 instrumentAdapter — prototype-chain walk that installs the wrappers
```mermaid
sequenceDiagram
    autonumber
    participant RA as resolveAdapters<br/>adapters/index.js:196
    participant IA as instrumentAdapter<br/>adapters/index.js:158
    participant PR as Object.getPrototypeOf chain<br/>index.js:160-170
    participant WD as wrapDispatch<br/>observability/metrics.js:125
    participant A as adapter instance

    RA->>IA: instrumentAdapter(new AdapterClass(cfg), slot, impl, manifest)
    IA->>PR: proto = getPrototypeOf(adapter)
    loop while proto and proto !== Object.prototype (index.js:161)
        PR-->>IA: getOwnPropertyNames(proto)
        loop for each name
            alt name already seen, in NON_DISPATCH, or startsWith underscore
                IA->>IA: skip — index.js:163, the most-derived definition wins
            else descriptor value is not a function
                IA->>IA: skip — index.js:165
            else
                IA->>WD: wrapDispatch(slot, impl, name, desc.value.bind(adapter), manifest)
                WD-->>IA: instrumentedDispatch
                IA->>A: adapter[name] = instrumentedDispatch (own property) — index.js:167
            end
        end
        IA->>PR: proto = getPrototypeOf(proto)
    end
    IA-->>RA: adapter (own props now shadow the prototype methods)
    Note over IA,PR: NON_DISPATCH = constructor, connect, disconnect — index.js:156
    Note over IA,PR: connect() stays unwrapped so lifecycle.js owns its failure semantics (index.js:153-154)
    Note over PR,A: pods walks LocalSolidRsPodsAdapter then SolidHttpPodsAdapter then BaseAdapter
    Note over A: INVARIANT ADR-2004 — every durable-state call rides one of the five slots
```

## AB-04.4 wrapDispatch — the middleware order actually wired
```mermaid
sequenceDiagram
    autonumber
    participant C as Route handler<br/>management-api/routes/*.js
    participant L1 as Layer 1 observability<br/>metrics.js:125 wrapDispatch
    participant L2 as Layer 2 privacy filter<br/>privacy-filter.js:649
    participant FN as raw adapter method<br/>bound at index.js:167
    participant PM as prom-client registry<br/>metrics.js:18 and :26
    participant LOG as stdout JSON log<br/>metrics.js:151-164

    C->>L1: adapter.store(key, value, namespace)
    rect rgb(232, 244, 255)
        L1->>L1: startHrTime = process.hrtime.bigint() (metrics.js:131)
        L1->>L1: executionId = uris.mint(kind event) (metrics.js:134)
        L1->>L2: await privacyWrapped(...args) (metrics.js:140)
        rect rgb(255, 242, 230)
            L2->>L2: isMutationMethod(methodName) (privacy-filter.js:652)
            L2->>FN: fn(...args) after redaction or pass-through
            FN-->>L2: result
        end
        L2-->>L1: result
        L1->>PM: adapterDispatchTotal.labels(slot, method, impl, 'success').inc()
        L1->>PM: adapterDurationSeconds.labels(slot, method, impl).observe(seconds)
        L1->>LOG: msg adapter_dispatch with slot, method, impl, duration_ms, execution_id, outcome
    end
    L1-->>C: result
    Note over L1,L2: privacyWrapped is built ONCE at wrap time — metrics.js:127
    Note over L1,FN: Layer 1 timing encloses Layer 2 so redaction latency is inside the span (metrics.js:138-139)
    Note over C,FN: DIVERGENCE — BASELINE-container "Adapter spine" states dispatch is wrapped<br/>observability then privacy then JSON-LD. Only Layers 1 and 2 are in the chain.<br/>metrics.js:110-112 states Layer 3 JSON-LD is ADR-012 and the CALLER must invoke<br/>encoder.dispatch after this wrapper returns.
    Note over C,FN: RESOLVED ADR-2036: the two-layer wrap is the decided design,<br/>not a gap. JSON-LD encoding is a per-surface gated stage invoked<br/>by the owning route — ordering is enforced by the privacy marker<br/>via assertPrivacyFilterApplied (privacy-filter.js:595). ADR-2005 superseded.
    Note over C,FN: This matches BASELINE "Adapter acceptance qualification 2026-09-04" — encoding is a separate caller action
    Note over L1,PM: On throw the same counters record outcome error — metrics.js:167-174
```

## AB-04.5 Layer 2 privacy filter — write-op gating and fail policy
```mermaid
sequenceDiagram
    autonumber
    participant L1 as wrapDispatch<br/>metrics.js:140
    participant PF as privacyFilteredDispatch<br/>privacy-filter.js:654
    participant POL as _slotPolicy(slot, manifest)<br/>privacy-filter.js:655
    participant OPF as OPF redactor<br/>_callOpfFields privacy-filter.js:700
    participant FN as raw adapter method

    L1->>PF: privacyWrapped(...args)
    PF->>POL: resolve policy for slot
    alt not a mutation method OR policy === 'off' (privacy-filter.js:661)
        PF->>PF: _markValueArg(args) — stamp traversal (privacy-filter.js:662)
        PF->>FN: fn(...args)
        FN-->>PF: result
    else mutation under an active policy
        PF->>PF: collectLeaves(args, methodName) (privacy-filter.js:685)
        alt truncated — cycle or depth limit (privacy-filter.js:687)
            PF->>PF: fail('payload could not be fully traversed')
        end
        alt leaves.length === 0 (privacy-filter.js:693)
            PF->>FN: fn(...args) after _markValueArg
        else
            PF->>OPF: _callOpfFields(texts, slot, methodName)
            OPF-->>PF: redacted[]
            alt a non-content field changed (privacy-filter.js:705-709)
                PF->>PF: opfIdentifierPii.inc(dirty.length) then fail(IdentifierPiiDetected)
            else
                PF->>PF: finalArgs = applyRedactions(args, leaves, redacted) (privacy-filter.js:716)
                PF->>FN: fn(...finalArgs)
            end
        end
    end
    FN-->>PF: result
    PF-->>L1: result
    Note over PF: fail() branches on policy — strict throws AdapterWriteRejected<br/>(privacy-filter.js:669-675), otherwise fail-open counter and caller continues<br/>(privacy-filter.js:677-682)
    Note over PF,OPF: Identifier and unclassified fields are SCREENED never rewritten — only role 'content' leaves are redacted in place
    Note over PF: assertPrivacyFilterApplied(payload, slot) at privacy-filter.js:595 verifies the layer was traversed (DDD-004 L08)
```


**Tension:** write-op gating is by method name. The exact set and prefix rule (`../project/agentbox/management-api/middleware/privacy-filter.js:283-313`) do not classify `dispatch`, `claim`, `close`, `del` or `spawnAgent` as mutations. The events slot's only write, `dispatch`, therefore reaches the hash-chained JSONL log without redaction, whatever the slot policy says. So do beads `claim` and `close`. The rule redacts what the vocabulary names, not everything that writes.

## AB-04.6 connectAdapters — per-slot deadline, not one aggregate budget
```mermaid
sequenceDiagram
    autonumber
    participant SV as server.js boot<br/>server.js:1364-1382
    participant LC as connectAdapters<br/>adapters/lifecycle.js:234
    participant CO as connectOneSlot<br/>adapters/lifecycle.js:194
    participant AD as adapter.connect()
    participant T as deadline timer<br/>lifecycle.js:210-212

    SV->>LC: connectAdapters(slots, adapters, manifest, logger, resolveOff) — server.js:1366
    par all five slots concurrently (lifecycle.js:249)
        LC->>LC: timeoutMs = connectTimeoutFor(slot, manifest) (lifecycle.js:252 and :119)
        alt adapter missing or no connect() hook (lifecycle.js:255)
            LC->>LC: state = 'off' when enabled === false else 'ready' (lifecycle.js:257)
        else
            LC->>CO: connectOneSlot(adapter, slot, timeoutMs, timers)
            CO->>AD: Promise.resolve().then(() => adapter.connect()) (lifecycle.js:199)
            CO->>T: setTimeout(timeoutMs) — deliberately NOT unref'd (lifecycle.js:206-212)
            CO->>CO: await Promise.race([observed, timeout]) (lifecycle.js:214)
            alt resolved first
                CO-->>LC: settled 'resolved'
                LC->>LC: state 'ready', log Adapter connected (lifecycle.js:267-269)
            else rejected first
                CO-->>LC: settled 'rejected'
            else deadline first
                CO-->>LC: settled 'timeout' plus latePromise (lifecycle.js:216)
            end
        end
    end
    LC-->>SV: readiness map and healthy flag (lifecycle.js:338-339)
    SV->>SV: app.adapters[slot] = resolvedAdapters[slot] (server.js:1379)
    SV->>SV: Object.assign(adapterHealth, toLegacyHealth(readiness)) (server.js:1380)
    SV->>SV: app.decorate('adapterReadiness', readiness) (server.js:1381)
    Note over LC,T: DEFAULT_CONNECT_TIMEOUT_MS = 10000 — lifecycle.js:72
    Note over LC: Manifest override [adapters] connect_timeout_ms scalar or per-slot map — lifecycle.js:107-109, :123,<br/>non-positive values ignored (lifecycle.js:126). The Stage 1 schema rejects the key, see AB-04.15
    Note over SV,LC: INVARIANT: ONE deadline PER SLOT, never one aggregate budget —<br/>adapters/lifecycle.js:234, wired from server.js:1365-1366. Aggregate wall-clock is<br/>bounded by the slowest single slot, not by a race that abandons work in flight<br/>(lifecycle.js:31-35).
    Note over SV,LC: RESOLVED ADR-2035: BASELINE-container.md:117 documents the<br/>per-slot deadline. Its line references there (server.js:1241-1242) are stale at this<br/>commit, see the Drift entry below this panel.
    Note over CO,T: Timeout is a CONNECT FAILURE identical in consequence to an explicit rejection — lifecycle.js:31-33
```

**Drift:** `docs/BASELINE-container.md` still cites `connectAdapters` as wired from `server.js:1241-1242` and `app.adapterReadiness` at `server.js:1257` (`../project/agentbox/docs/BASELINE-container.md:117`). At the pin those calls sit at `../project/agentbox/management-api/server.js:1365-1381`. It also places `instrumentAdapter` at `index.js:131` (`../project/agentbox/docs/BASELINE-container.md:112`), where the function is now at `../project/agentbox/management-api/adapters/index.js:158`. The governing document carries line numbers that no longer resolve.

## AB-04.7 Failure path — quarantine before replacement, fail-closed slots abort
```mermaid
sequenceDiagram
    autonumber
    participant LC as connectAdapters<br/>lifecycle.js:234
    participant Q as quarantineAdapter<br/>lifecycle.js:142
    participant RO as resolveOff callback<br/>server.js:1371-1374
    participant IDX as resolveAdapters<br/>index.js:196
    participant PX as process.exit(1)<br/>lifecycle.js:333

    LC->>LC: failureMode = timeout or rejected (lifecycle.js:273)
    LC->>LC: failure class config-failed, connect-timeout or connect-failed, plus the config file path (lifecycle.js:277-279)
    LC->>LC: record state 'unavailable' with reason and durationMs (lifecycle.js:280-281)
    opt outcome.latePromise present (lifecycle.js:286)
        LC->>LC: attach handler recording lateSettle and warn "settled AFTER its deadline — the slot stays withdrawn"
    end
    rect rgb(255, 232, 232)
        LC->>Q: quarantineAdapter(adapter, slot, failureMode plus reason) — lifecycle.js:296
        Q->>Q: collect own props and whole prototype chain (lifecycle.js:144-150)
        Q->>Q: replace every callable with async thrower AdapterQuarantined (lifecycle.js:161)
        Q->>Q: skip QUARANTINE_EXEMPT constructor and disconnect, and underscore-private (lifecycle.js:152)
        Q-->>LC: adapter._quarantined = true (lifecycle.js:166)
    end
    alt slot in FAIL_CLOSED_SLOTS — orchestrator (lifecycle.js:298 and :75)
        LC->>LC: log error "Fail-closed adapter slot failed to connect — FATAL"
        LC->>PX: fatals drained after Promise.all (lifecycle.js:329-336)
        PX-->>LC: startup aborts
    else degrade path
        LC->>RO: resolveOff(slot)
        RO->>IDX: resolveAdapters({adapters: {slot: 'off'}})[slot]
        alt replacement built
            IDX-->>LC: off adapter
            LC->>LC: refuseDispatch — the stand-in's dispatch() throws AdapterDisabled (lifecycle.js:178-184, :312)
            LC->>LC: adapters[slot] = offSlot, record.state = 'disabled' (lifecycle.js:315-317)
        else replacement threw (lifecycle.js:318)
            LC->>LC: record.state = 'unavailable', failureMode 'replacement-failed' (lifecycle.js:320-321)
            LC->>LC: slot stays QUARANTINED — dispatch throws AdapterQuarantined not AdapterDisabled
        end
    end
    Note over Q: Quarantine happens BEFORE any replacement so no window exists in which a half-connected adapter is reachable (lifecycle.js:294-296)
    Note over Q: AdapterQuarantined carries statusCode 503 — lifecycle.js:91
    Note over LC,RO: refuseDispatch exists because the events off adapter no-ops dispatch: as a stand-in for a FAILED<br/>slot it would accept a write and drop it while the authority journal reports it persisted (lifecycle.js:172-177)
    Note over LC,RO: RESOLVED ADR-2035: recorded as the decided policy — quarantine<br/>before replace (lifecycle.js:294-296), 'unavailable' on replacement<br/>failure (lifecycle.js:318-321).
    Note over LC: disconnect() stays callable through quarantine so shutdown can release what the adapter opened (lifecycle.js:42-43 and :78)
```

## AB-04.8 Per-slot readiness lifecycle and the legacy health collapse
```mermaid
stateDiagram-v2
    [*] --> Unresolved
    Unresolved --> Constructed: resolveAdapters index.js:200-226
    Unresolved --> ConfigFailed: slotConfig or constructor threw index.js:216-225
    ConfigFailed --> Instrumented: instrumentAdapter index.js:227
    Constructed --> Instrumented: instrumentAdapter index.js:227
    Instrumented --> NoConnectHook: adapter.connect not a function lifecycle.js:255
    Instrumented --> Connecting: connectOneSlot lifecycle.js:264

    NoConnectHook --> Off: enabled === false lifecycle.js:257
    NoConnectHook --> Ready: otherwise lifecycle.js:257

    Connecting --> Ready: settled resolved lifecycle.js:267
    Connecting --> Rejected: settled rejected lifecycle.js:273
    Connecting --> TimedOut: deadline won lifecycle.js:273

    Rejected --> Quarantined: quarantineAdapter lifecycle.js:296
    TimedOut --> Quarantined: quarantineAdapter lifecycle.js:296

    Quarantined --> Fatal: slot in FAIL_CLOSED_SLOTS lifecycle.js:298
    Quarantined --> Disabled: off replacement wired lifecycle.js:315-317
    Quarantined --> Unavailable: replacement-failed lifecycle.js:320-321

    Fatal --> [*]: process.exit(1) lifecycle.js:333

    note right of Ready
        toLegacyHealth lifecycle.js:350
        ready maps to healthy
    end note
    note right of Off
        off maps to off
        /health adapters map only
    end note
    note right of Disabled
        collapses to degraded
        dispatch throws AdapterDisabled errors.js
    end note
    note right of Unavailable
        also collapses to degraded
        dispatch throws AdapterQuarantined 503
    end note
    note left of TimedOut
        late settle is recorded never re-armed
        lifecycle.js:286-292
    end note
```

**Drift:** this panel used to map the legacy states onto the `agentbox_adapter_health` gauge as off 0, degraded 1, healthy 2. `setAdapterHealth` encodes healthy as 1, degraded as 2 and anything else as 0 (`../project/agentbox/management-api/observability/metrics.js:202-204`). Nothing in the management API calls it, so the gauge is never set at this revision. The collapse feeds only the `/health` JSON (`../project/agentbox/management-api/server.js:1380`, `../project/agentbox/management-api/server.js:608-619`).

## AB-04.9 memory.store — the configured external-pg dispatch through both layers
```mermaid
sequenceDiagram
    autonumber
    participant R as routes/memory.js
    participant L1 as Layer 1 wrapDispatch<br/>metrics.js:125
    participant L2 as Layer 2 privacy<br/>privacy-filter.js:654
    participant MA as ExternalPgMemoryAdapter<br/>memory/external-pg.js:41
    participant PG as ruvector-postgres sidecar<br/>db ruvector 5432

    Note over R,PG: the shipped manifest selects memory = external-pg (agentbox.toml:20)
    R->>L1: app.adapters.memory.store(key, value, namespace)
    L1->>L2: privacyWrapped(key, value, namespace)
    L2->>L2: isMutationMethod('store') is true — redaction path runs
    L2->>MA: store(key, redactedValue, namespace) — external-pg.js:80
    MA->>PG: check memory_entries exists, never create it (external-pg.js:62-71)
    MA->>PG: INSERT INTO memory_entries ... ON CONFLICT (id) DO UPDATE value, source_type agentbox<br/>no embedding column is written (external-pg.js:89-96)
    PG-->>MA: ok
    MA-->>L2: key, namespace, stored_at, urn (external-pg.js:97)
    L2-->>L1: result
    L1->>L1: adapterDispatchTotal.labels('memory','store',impl,'success').inc()
    L1-->>R: result
    alt impl resolved to embedded-ruvector
        Note over MA: in-process Map per namespace with 64-dim term-frequency vectors, for dev and tests,<br/>no Postgres at all (memory/embedded-ruvector.js:4-9, :31-47, :76-88)
    end
    alt impl resolved to off (adapters/memory/off.js:14)
        R->>L1: store(...)
        L1->>L2: privacyWrapped(...)
        L2->>MA: OffMemoryAdapter.store()
        MA-->>R: throw AdapterDisabled('memory') — off.js:19
    end
    alt slot quarantined by lifecycle.js:296
        MA-->>R: throw AdapterQuarantined statusCode 503 — lifecycle.js:80-93
    end
    Note over MA,PG: contract version memory 1.0.0 — contract-versions.js:10
    Note over R,PG: slotConfig supplies the pool config from integrations.ruvector_external — index.js:44-51
    Note over L1: Every method is wrapped identically by instrumentAdapter index.js:158-171
```

**Drift:** the previous panel drew `EmbeddedRuvectorMemoryAdapter` upserting into `memory_entries` with a 384-dimension embedding. The embedded adapter is an in-process `Map` with 64-dimension naive vectors (`../project/agentbox/management-api/adapters/memory/embedded-ruvector.js:4-9`, `../project/agentbox/management-api/adapters/memory/embedded-ruvector.js:31-47`). The Postgres writer is `external-pg`, which the manifest selects (`../project/agentbox/agentbox.toml:20`).

**Open:** `ExternalPgMemoryAdapter.store` writes `id, namespace, key, value, source_type, metadata` and no embedding (`../project/agentbox/management-api/adapters/memory/external-pg.js:89-96`). Agentbox's own `search` is an `ILIKE` text match over agentbox-owned rows (`../project/agentbox/management-api/adapters/memory/external-pg.js:103-115`), so `/v1/memory/search` is unaffected. Source cannot show whether a database-side trigger embeds these rows. If none does, estate-side semantic search over the shared table (HNSW) cannot reach agentbox-written entries.

## AB-04.10 beads.addDependency and beads.getReady — the 1.1.0 work-DAG pair
```mermaid
sequenceDiagram
    autonumber
    participant R as routes/beads.js
    participant L1 as Layer 1 wrapDispatch<br/>metrics.js:125
    participant L2 as Layer 2 privacy<br/>privacy-filter.js:654
    participant BA as LocalSqliteBeadsAdapter<br/>beads/local-sqlite.js:56
    participant DB as local sqlite bead store

    R->>L1: addDependency(childId, blockerId, type)
    L1->>L2: privacyWrapped(childId, blockerId, type)
    L2->>BA: addDependency(childId, blockerId, type default 'blocks') — local-sqlite.js:195
    BA->>DB: insert into bead_deps
    DB-->>BA: ok
    BA-->>L1: result
    L1->>L1: adapterDurationSeconds.labels('beads','addDependency',impl).observe(s)
    L1-->>R: result

    R->>L1: getReady(filter)
    L1->>L2: privacyWrapped(filter)
    L2->>L2: isMutationMethod('getReady') is false — pass-through, _markValueArg only (privacy-filter.js:661-663)
    L2->>BA: getReady(filter) — local-sqlite.js:236
    BA->>DB: select beads with no unmet blocker in bead_deps
    DB-->>BA: ready rows
    BA->>BA: _hydrate(row) — local-sqlite.js:274
    BA-->>L1: ready beads
    L1-->>R: ready beads
    Note over BA: beads contract 1.1.0 is the only non-1.0.0 pin — contract-versions.js:7-8 records it as additive for addDependency plus dependency-aware getReady
    Note over BA,DB: other slot methods createEpic :78, createChild :113, claim :145, close :168, show :259
    Note over L2: read methods traverse the privacy layer but are never redacted — only mutations are
```

## AB-04.11 pods.write — NIP-98 origination through the signed fetch
```mermaid
sequenceDiagram
    autonumber
    participant R as routes/pod-git.js or linked-objects.js
    participant L1 as Layer 1 wrapDispatch<br/>metrics.js:125
    participant L2 as Layer 2 privacy<br/>privacy-filter.js:654
    participant PA as LocalSolidRsPodsAdapter<br/>pods/local-solid-rs.js:27
    participant SF as _signedFetch<br/>_solid-http-base.js:78
    participant SG as nip98 signer<br/>lib/pod-signer.js:124
    participant SP as solid-pod-rs<br/>127.0.0.1:8484

    Note over PA: cfg.baseUrl resolved at index.js:93-97 from integrations.solid_pod_rs base_url, or bind and port.<br/>withSigner threads nip98 AND requireSigned onto it - index.js:74, :85-88
    R->>L1: pods.write(uri, body, contentType)
    L1->>L2: privacyWrapped(uri, body, contentType)
    L2->>PA: write(uri, body, contentType default application/ld+json) — _solid-http-base.js:119
    PA->>SF: this._fetch(base + uri) with method PUT
    alt signer wired — buildPodNip98 returned a function (index.js:75, pod-signer.js:70)
        SF->>SF: hasAuth check on existing headers — caller header is trusted, never overwritten (_solid-http-base.js:81-82)
        SF->>SG: nip98(method, url, body) — returns null when no signer resolves (pod-signer.js:124-128)
        SG-->>SF: Authorization header value or null
        alt header is falsy
            alt requireSigned — sign_requests on
                SF--xPA: throw SigningUnavailable, no bytes leave the process (_solid-http-base.js:104-106)
            else
                SF->>SP: _rawFetch without Authorization (_solid-http-base.js:107)
            end
        else
            SF->>SP: _rawFetch with Authorization header (_solid-http-base.js:109)
        end
    else signer null but requireSigned — signer could NOT be built and signing was demanded
        SF--xPA: throw SigningUnavailable per request (_solid-http-base.js:84-91)
    else signer null and gate off — unsigned, byte-identical to the pre-signing baseline (index.js:63-64)
        SF->>SP: _rawFetch unsigned — this._fetch is _rawFetch (_solid-http-base.js:62-64)
    end
    SP-->>PA: HTTP response
    PA->>PA: _assert(res, [200, 201]) — _solid-http-base.js:125, defined :195
    PA-->>L1: uri, status, created_at
    L1-->>R: result
    Note over R,SG: INVARIANT: ADR-2064 - sign_requests is a fail-closed switch, not a best-effort hint.<br/>buildPodNip98 failure only warns (index.js:75-84), but requireSigned rides with the config<br/>regardless, so the pods slot degrades VISIBLY instead of going out anonymous at a default-deny pod
    Note over PA: local-solid-rs overrides probeCapabilities :51, list :72, patch :126 over the shared base
    Note over PA,SP: pods contract 1.0.0 — contract-versions.js:9
```

## AB-04.13 orchestrator.spawnAgent and handleGovernanceDecision — the ADR-2109 ontology write
```mermaid
sequenceDiagram
    autonumber
    participant R as routes/tasks.js or sessions-boundary.js
    participant L1 as Layer 1 wrapDispatch<br/>metrics.js:125
    participant L2 as Layer 2 privacy<br/>privacy-filter.js:654
    participant OA as LocalProcessManagerOrchestratorAdapter<br/>orchestrator/local-process-manager.js:25
    participant OD as applyOntologyDecision<br/>lib/ontology-apply.js:372
    participant VC as vault CLI<br/>runVaultCommand ontology-apply.js:256
    participant P as spawned agent process

    R->>L1: orchestrator.spawnAgent(spec)
    L1->>L2: privacyWrapped(spec) — not a mutation name, pass-through (privacy-filter.js:661-663)
    L2->>OA: spawnAgent(spec) — local-process-manager.js:45
    OA->>P: launch process
    P-->>OA: agentId
    OA-->>L1: agentId
    L1-->>R: agentId

    R->>L1: orchestrator.handleGovernanceDecision(event, opts)
    L1->>L2: privacyWrapped(event)
    L2->>OA: handleGovernanceDecision(event, opts) — local-process-manager.js:142
    OA->>OA: outcome = parsed.action or parsed.outcome — local-process-manager.js:157
    alt outcome is approve on an ontology amend (local-process-manager.js:217-225)
        OA->>OA: _ontologyApprove applies it as a promote of the digest the 31403 signed (local-process-manager.js:379-409)
    else outcome is promote or demote — ADR-2109 (local-process-manager.js:226)
        OA->>OA: _ontologySignerRefusal — refuse a signer the operator did not name (local-process-manager.js:231-233)
        opt eTag present
            OA->>OA: stored request's d tag and iri must match the decision, else refused (local-process-manager.js:442-452)
            OA->>OD: proposal = proposalFromRequest(request) — ontology-apply.js:536
        end
        OA->>OD: applyOntologyDecision(proposal, outcome, iri, page, signerNpub, at, caseId, digest, eventId, kind) — local-process-manager.js:461-480
        alt proposal.kind is create
            OD->>OD: applyCreateProposal(decision, proposal, deps) — ontology-apply.js:385, :560
        else promote of an amend proposal
            OD->>OD: applyAmendProposal, ontomark apply — ontology-apply.js:391-392, :692
        else existing page
            OD->>OD: resolveIriToPage(iri, opts) — ontology-apply.js:401, :215
        end
        OD->>VC: vault edit page --set ... --expect N --json (ontology-apply.js:427-429)
        VC-->>OD: edited or refused
        OD->>OD: attest(ledger body) — ontology-apply.js:438, :627, :742
        OD-->>OA: applied, page, outcome, attested, attestError
        opt applyOntologyDecision throws
            OA->>OA: ontology = applied false, error message — local-process-manager.js:481-487
        end
    end
    OA->>OA: decision record with activity_urn, receipt_urn, ontology — local-process-manager.js:256-267
    OA->>P: matchedEntry.proc.stdin.write(decision) when an agent matches refId — local-process-manager.js:269-275
    OA-->>L1: dispatched, target, event_id, ontology — local-process-manager.js:327,350
    L1-->>R: result

    opt stream and control
        R->>OA: streamEvent(agentId, handler) — local-process-manager.js:106
        R->>OA: listAgents() — local-process-manager.js:119
        R->>OA: terminateAgent(agentId) — local-process-manager.js:549
    end
    Note over OA: INVARIANT ADR-2004 — orchestrator is the sole member of FAIL_CLOSED_SLOTS<br/>(lifecycle.js:75). Connect rejection, deadline expiry and quarantine are all equally<br/>fatal (lifecycle.js:50-53).
    Note over OA,OD: INVARIANT ADR-2109 — the ontology write runs BEFORE agent dispatch: the write<br/>IS the decision's effect and dispatch is only a notification about it. A write<br/>failure is recorded as ontology.applied=false and swallowed, never thrown — the<br/>signed 31403 stays durable whatever the corpus write managed to do (local-process-manager.js:204-215).
    Note over R,OA: DOC-DRIFT — the slot method is spawnAgent(spec) at local-process-manager.js:45, not spawn()
    Note over OA,P: orchestrator contract 1.0.0 — contract-versions.js:12
```

## AB-04.14 Contract-version pins per slot
```mermaid
flowchart LR
    CV["contract-versions.js:6<br/>module.exports"] --> B["beads 1.1.0<br/>contract-versions.js:8"]
    CV --> P["pods 1.0.0<br/>contract-versions.js:9"]
    CV --> M["memory 1.0.0<br/>contract-versions.js:10"]
    CV --> E["events 1.0.0<br/>contract-versions.js:11"]
    CV --> O["orchestrator 1.0.0<br/>contract-versions.js:12"]
    B --> SUP["super(slot, impl, CONTRACT_VERSIONS[slot])<br/>base.js:20-26"]
    P --> SUP
    M --> SUP
    E --> SUP
    O --> SUP
    SUP --> INST["this.CONTRACT_VERSION on every instance<br/>base.js:26"]
    INST --> MAN["system-manifest emits impl plus contract_version<br/>per core-layer slot entry"]
    B -.-> NOTE1["1.1.0 additive — addDependency plus<br/>dependency-aware getReady bead_deps work-DAG"]
    P -.-> NOTE2["DIVERGENCE BASELINE Known divergences —<br/>stale placeholders despite live churn<br/>a breaking change would need a MAJOR bump that has not happened"]
    M -.-> NOTE2
    E -.-> NOTE2
    O -.-> NOTE2
    SUP -.-> NOTE3["base.js:21-23 throws when slot, impl or<br/>contractVersion is missing — no unpinned adapter can construct"]
    E -.-> NOTE4["events payload shape is pinned by the standalone JSON Schema<br/>adapters/events/agent-execution-event.schema.json — ADR-057 D1:<br/>session_urn + seq unique and strictly increasing, append-only,<br/>corrections append a compensating event rather than mutating"]
```

## AB-04.15 The four validation stages — different files, different lifecycle points
```mermaid
flowchart TD
    subgraph S1["Stage 1 static schema — edit and build time"]
        A1["schema/agentbox.toml.schema.json"] --> A2["agentbox-config-validate.js:125-128<br/>Ajv"]
        A2 --> A3{"any E-code?<br/>schema or cross-field"}
        A3 -->|yes| A4["reject, exit 1<br/>:1920"]
        A3 -->|"no"| A5["W0xx warning only<br/>exit 0, :1912-1915"]
    end
    subgraph S2["Stage 2 boot probe — once per boot"]
        B1["server.js:1365-1366 require adapters/lifecycle"] --> B2["connectAdapters lifecycle.js:234"]
        B2 --> B3["per-slot deadline lifecycle.js:119 and :252"]
        B3 --> B4["ready | disabled | unavailable | off<br/>lifecycle.js:55-66"]
        B4 --> B5["toLegacyHealth lifecycle.js:350<br/>adapterHealth server.js:1380"]
    end
    subgraph S3["Stage 3 conformance — CI only, never at boot"]
        C1["tests/contract/memory.contract.spec.js"] --> C4["all three impl classes must behave identically"]
        C2["tests/contract/beads.contract.spec.js"] --> C4
        C3["tests/contract/adapter-lifecycle.contract.spec.js"] --> C4
        C5["tests/contract/pods.contract.spec.js"] --> C4
        C6["tests/contract/events.contract.spec.js"] --> C4
        C7["tests/contract/orchestrator.contract.spec.js"] --> C4
        C8["tests/contract/privacy-filter.contract.spec.js"] --> C4
        C9["tests/contract/memory-encoder-bypass.contract.spec.js"] --> C4
    end
    subgraph S4["Stage 4 SLO — continuous"]
        D1["agentbox_adapter_dispatch_total metrics.js:18"] --> D4["GET /metrics"]
        D2["agentbox_adapter_duration_seconds metrics.js:26"] --> D4
        D3["agentbox_adapter_health metrics.js:35<br/>never set"] --> D4
    end
    A4 --> B1
    A5 --> B1
    S3 -.-> NOTE["DIVERGENCE — legacy ADR-005 conflates<br/>all four into 'contract tests'"]
```

**Tension:** the schema's `[adapters]` table sets `additionalProperties: false` and does not declare `connect_timeout_ms` (`../project/agentbox/schema/agentbox.toml.schema.json:50-52`). Stage 1 compiles that schema with Ajv (`../project/agentbox/scripts/agentbox-config-validate.js:125-128`). The per-slot override that Stage 2 reads (`../project/agentbox/management-api/adapters/lifecycle.js:107-109`) is therefore rejected by Stage 1 as an unknown key, so the documented knob cannot pass validation.

**Drift:** BASELINE stage 4 says the slot SLO is "surfaced as the `agentbox_adapter_health` gauge ∈ {0,1,2}" (`../project/agentbox/docs/BASELINE-container.md:119`). The gauge is declared (`../project/agentbox/management-api/observability/metrics.js:35-40`), but its only setter has no caller in the repository. The help text and the setter agree on the encoding: 0 unhealthy (so `off` would read as 0), 1 healthy, 2 degraded (`../project/agentbox/management-api/observability/metrics.js:37`, `../project/agentbox/management-api/observability/metrics.js:202-204`). Earlier versions of this topic gave healthy as 2, which contradicts both. Stage 1's rejection is also wider than "structural only": cross-field E-rules such as E-PAY1 reject as well (`../project/agentbox/scripts/agentbox-config-validate.js:1755`).

## AB-04.16 SLO surfacing and the agentbox.sh health exit contract
```mermaid
sequenceDiagram
    autonumber
    participant OP as operator shell
    participant SH as cmd_health<br/>agentbox.sh:1131
    participant H as GET /health<br/>server.js:582-620
    participant MT as GET /v1/meta<br/>agentbox.sh:1185
    participant PM as prom-client registry<br/>metrics.js:18 :26 :35
    participant SP as speech sidecars<br/>localhost port 8897 and port 8898

    OP->>SH: ./agentbox.sh health
    SH->>H: curl HEALTH_URL http://localhost:$MGMT_PORT/health (agentbox.sh:631 and :1142)
    alt curl fails
        H-->>SH: no response
        SH-->>OP: ERROR could not reach — exit 1 (agentbox.sh:1142-1146)
    else --json
        SH-->>OP: print the raw payload, exit 0 whatever it says (agentbox.sh:1148-1151)
    else response received, jq present
        H-->>SH: status, uptime, image_hash, manifest_checksum, adapters, adapter_failures, adapter_failure_files, degraded_count, note
        SH->>SH: degraded = jq '.adapters // {} | to_entries[] | select(.value != "healthy" and .value != "off") | .key' (agentbox.sh:1157-1162)
        SH->>SH: degraded_count = jq '.degraded_count // 0' (agentbox.sh:1163-1164)
        SH->>SH: print "adapter/<slot>: <value>" from .adapters (agentbox.sh:1167-1176)
        SH->>MT: curl http://localhost:9090/v1/meta for observability.metrics_endpoint
        MT-->>SH: metrics_endpoint
        SH->>PM: curl metrics_endpoint, print first 5 non-comment lines (agentbox.sh:1193-1199)
        SH->>SP: curl -sf http://localhost:8897/health — nemotron-asr (agentbox.sh:1204-1209)
        SP-->>SH: healthy or speech_failed=1
        SH->>SP: curl -sf http://localhost:8898/health — pocket-tts (agentbox.sh:1210-1215)
        SP-->>SH: healthy or speech_failed=1
        alt degraded non-empty OR degraded_count > 0 OR speech_failed > 0
            SH-->>OP: exit 1 (agentbox.sh:1217-1218)
        else
            SH-->>OP: exit 0
        end
    else jq absent
        SH-->>OP: print the raw payload, falls off the end with status 0 (agentbox.sh:1220-1223)
    end
    Note over H: /health computes degradedCount from adapterHealth (server.js:608) and emits keys status, uptime, image_hash,<br/>manifest_checksum, adapters, adapter_failures, adapter_failure_files, degraded_count, note — NO services key (server.js:609-619)
    Note over SH,H: RESOLVED ADR-2037 — cmd_health derives failure from .adapters (a slot fails when its value<br/>is neither "healthy" nor "off", agentbox.sh:1157-1162) plus .degraded_count (agentbox.sh:1163),<br/>so exit 1 at agentbox.sh:1217-1218 is reachable. Same fix as AB-05.7.
    Note over SH,SP: the exit condition carries a third term, speech_failed (agentbox.sh:1203, :1217):<br/>nemotron-asr and pocket-tts are checked unconditionally by port, independent of the<br/>[adapters] manifest and never surfaced in /health's own adapters map.
    Note over SH: /health itself warns it is for human inspection only and points orchestrators at /ready (server.js:618)
    Note over PM: the agentbox_adapter_health gauge is declared but never set, so the sample lines carry no slot health, see AB-04.15
```

**Tension:** the non-zero exit is reachable only on the pretty-print path. `--json` prints the payload and exits 0 (`../project/agentbox/agentbox.sh:1148-1151`). A host without `jq` prints the raw payload and returns 0 even when slots are degraded (`../project/agentbox/agentbox.sh:1220-1223`). A script that pipes `agentbox.sh health --json` gets no failure signal.
