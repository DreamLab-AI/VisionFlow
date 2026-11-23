---
id: OO-04
title: Engine bridge — envelopes, generation guard, timeouts, restarts and stepped classify
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
adrs: []
sources:
  - ../ontomark-obsidian/src/protocol.ts
  - ../ontomark-obsidian/src/client.ts
  - ../ontomark-obsidian/src/guard.ts
  - ../ontomark-obsidian/src/worker.ts
  - ../ontomark-obsidian/src/worker-core.ts
  - ../ontomark-obsidian/src/api.ts
  - ../ontomark-obsidian/src/main.ts
verified_commit: {ontomark-obsidian: 64bdc5bf6309fca94bf0437b52c1484618470ed0}
---

## OO-04.1 Two protocol layers and what counts as a mutation

```mermaid
flowchart TD
    REQ["EngineRequest: JSON mirror of the Rust engine schema<br/>protocol.ts:4,5,50"]
    MUT["MUTATIONS: load, upsert, remove, rename, set_profile<br/>protocol.ts:86,87,88,89,90,91"]
    READ["reads: index, describe, frontmatter, query, propose_patch, ingest_plan, ingest_accept<br/>protocol.ts:56,57,58,63,70,76,78"]
    H2W["HostToWorker envelope v, epoch, id: init with wasm, engine, classify, cancel<br/>protocol.ts:343,344,345,346,347"]
    W2H["WorkerToHost: reply, progress, fatal<br/>protocol.ts:349,350,351,352"]
    SEND["send: a request is a read unless it is a mutation or version<br/>client.ts:391,393"]
    TO["timeouts: 30 s default, 15 s query, 120 s classify<br/>client.ts:118,394"]
    CH["only reads carry a consumer channel<br/>client.ts:395"]
    LOOP["worker passes the JSON string to WasmEngine.handle<br/>worker-core.ts:35<br/>worker.ts:19"]
    REQ --> MUT
    REQ --> READ
    MUT --> SEND
    READ --> SEND
    SEND --> TO
    SEND --> CH
    SEND --> H2W --> LOOP --> W2H
    INV["INVARIANT: the host owns both counters, a fresh generation per mutation and a fresh epoch per worker<br/>protocol.ts:9,10,11,12,13,14"]
    H2W --- INV
    NOTE["see OM-09.1 for how the engine bytes are pinned"]
    LOOP --- NOTE
```

**What it shows.** Rust is the schema's source of truth and these types only mirror it; `version` is neither read nor mutation, so it carries no channel and is never judged. The worker adds only `init`, `classify` and `cancel` around engine traffic.

## OO-04.2 One request, start to verdict

```mermaid
sequenceDiagram
    autonumber
    participant C as OntomarkClient<br/>client.ts:391
    participant G as GenerationGuard<br/>guard.ts:61
    participant W as worker message loop<br/>worker-core.ts:73
    participant E as WasmEngine<br/>worker.ts:19
    C->>C: mutation bumps the generation and cancels classify first (client.ts:358,359,360)
    C->>C: upsert records the text in the mirror before sending (client.ts:157,158)
    C->>C: track refuses when disposed or failed, starts the timer (client.ts:428,429,432)
    C->>G: issued, this id is now latest on its channel (client.ts:430)
    C->>W: post the engine envelope (client.ts:417,474)
    W->>W: drop if the epoch is not the live one (worker-core.ts:87)
    W->>E: handle the JSON request (worker-core.ts:91,35)
    E-->>W: response with the generation it was computed at
    W-->>C: reply (worker-core.ts:38,39)
    C->>C: dead epoch or unknown id never resolves (client.ts:323,324)
    C->>C: mutation result resolves, a refusal triggers a restart (client.ts:333,335,337,339)
    C->>G: judge epoch, channel, id, generation (client.ts:347)
    G-->>C: dead_epoch, stale_generation, superseded or accept (guard.ts:62,63,66,68)
    C-->>C: accept resolves, otherwise reject with stale or the reason (client.ts:348,349)
```

**What it shows.** A read only reaches its caller if it came from the live worker, was computed at the current generation and is still its channel's latest request. A refused mutation means the engine and the mirror disagree, so the worker is rebuilt from the mirror (`client.ts:338`).

## OO-04.3 Restart, replay and the failed state

```mermaid
flowchart TD
    TOUT["request timeout<br/>client.ts:432,435,436"]
    FAT["fatal reply after a wasm trap<br/>client.ts:325,327,328<br/>worker-core.ts:43,45"]
    REF["mutation refused by the engine<br/>client.ts:339"]
    ERR["worker onerror from the live worker<br/>client.ts:381,382,384"]
    RS["restart: kill worker, reject pending as worker_restarted<br/>client.ts:296,298,299"]
    CAP{"more than 3 restarts in 60 s?<br/>client.ts:300,301,302,303,304"}
    FAILD["failed: onFailed once, every later request rejects engine_failed<br/>client.ts:305,306,429"]
    SPAWN["spawn: new epoch, transfer a fresh copy of the wasm bytes<br/>client.ts:375,377,387,388"]
    LOADM["load replays the mirror at the current generation, posted back to back with init<br/>client.ts:365,366,367,370,371"]
    NOTI["onRestart Notice: engine restarted<br/>client.ts:310<br/>main.ts:288,289"]
    TOUT --> RS
    FAT --> RS
    REF --> RS
    ERR --> RS
    RS --> CAP
    CAP -->|yes| FAILD
    CAP -->|no| SPAWN --> LOADM
    SPAWN --> NOTI
    DEBT["DEBT: createApi passes no onFailed, so entering the failed state shows nothing until the next request's engine_failed Notice<br/>main.ts:284,285,286,287,288<br/>client.ts:72"]
    FAILD --- DEBT
    INV["INVARIANT: a page that traps the engine every time cannot cause an endless respawn loop<br/>client.ts:292,293,294"]
    CAP --- INV
```

**What it shows.** Four triggers share one restart path, and every restart replays the host's mirror rather than re-reading the vault. Requests swept up by a restart or dispose reject quietly (`client.ts:48`); the request that timed out or trapped, and every request after the cap, is reported (`client.ts:435,327,429`).

## OO-04.4 Stepped classify, joining and cancellation

```mermaid
sequenceDiagram
    autonumber
    participant A as OntomarkApi<br/>api.ts:209
    participant C as OntomarkClient<br/>client.ts:252
    participant W as worker message loop<br/>worker-core.ts:73
    A->>C: classify at the current generation (api.ts:210,213)
    C->>C: same generation already running, join that promise (client.ts:253,254)
    C->>C: else cancel the older run and dispatch with budget 20000 on the classify channel (client.ts:256,261,267)
    C->>W: classify envelope (client.ts:261)
    W->>W: classify_start (worker-core.ts:99,51)
    loop until done or error
        W-->>C: progress with steps and total (worker-core.ts:58)
        C->>A: onProgress only for the live epoch, generation and latest classify (client.ts:318,319)
        W->>W: yield to the event loop so queued mutations and cancels run (worker-core.ts:59)
        W->>W: cancelled id replies cancelled (worker-core.ts:61,63)
        W->>W: classify_step (worker-core.ts:66)
    end
    W-->>C: reply done, judged like any read (worker-core.ts:54,55 and client.ts:347)
    C-->>A: ClassifyDone or stale or cancelled
    A->>A: classified or classify_failed event (api.ts:216,220)
```

**What it shows.** Classification never blocks the worker for its whole run: each step yields, so an edit queued behind it advances the generation and the next step answers stale. No SharedArrayBuffer is needed (`worker-core.ts:3,4,5,6,7`).
