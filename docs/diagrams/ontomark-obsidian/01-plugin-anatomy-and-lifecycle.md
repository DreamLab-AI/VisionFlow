---
id: OO-01
title: Plugin anatomy — module map, load, change relay and unload
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-001-shared-graph.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-005-excluded-folders.md
adrs: [ADR-OBS-001, ADR-OBS-005]
sources:
  - ../ontomark-obsidian/src/main.ts
  - ../ontomark-obsidian/src/api.ts
  - ../ontomark-obsidian/src/client.ts
  - ../ontomark-obsidian/src/worker.ts
  - ../ontomark-obsidian/src/worker-core.ts
  - ../ontomark-obsidian/src/settings.ts
  - ../ontomark-obsidian/src/sparql-block.ts
  - ../ontomark-obsidian/src/ingest-transport.ts
  - ../ontomark-obsidian/src/jev.ts
  - ../ontomark-obsidian/src/views/entity-view.ts
  - ../ontomark-obsidian/src/views/graph-view.ts
  - ../ontomark-obsidian/src/graph-layout.ts
  - ../ontomark-obsidian/src/layout-engine.ts
  - ../ontomark-obsidian/manifest.json
verified_commit: {ontomark-obsidian: 64bdc5bf6309fca94bf0437b52c1484618470ed0}
---

## OO-01.1 Module map: one host class, one API, one worker, two embedded WebAssembly modules

```mermaid
flowchart TD
    subgraph HOST["Obsidian host adapters"]
        MAIN["OntomarkPlugin, desktop-only manifest<br/>main.ts:93<br/>manifest.json:9"]
        SET["OntomarkSettingTab<br/>settings.ts:92"]
        VIEWS["EntityView, QueryView, GraphView<br/>main.ts:128,129,130"]
        BLOCK["ontomark-sparql code block<br/>sparql-block.ts:145"]
    end
    subgraph CORE["Obsidian-free core"]
        API["OntomarkApi over a VaultPort<br/>api.ts:32,84"]
        CLI["OntomarkClient: generation, epoch, mirror<br/>client.ts:100"]
    end
    subgraph WORK["Engine worker, built from an inlined source string"]
        WK["worker entry, initSync on transferred bytes<br/>worker.ts:18,19"]
        WC["message loop and stepped classify<br/>worker-core.ts:26"]
    end
    subgraph NET["Opt-in network edge"]
        TR["postJson, the one requestUrl call<br/>ingest-transport.ts:90,97"]
        JEV["jevAsk typed decisions<br/>jev.ts:82"]
    end
    subgraph GR["Graph view layout"]
        GLC["GraphLayoutController<br/>graph-layout.ts:30"]
        LE["vowl-wasm engine, compiled on first use<br/>layout-engine.ts:19,21"]
    end
    MAIN --> SET
    MAIN --> VIEWS
    MAIN --> BLOCK
    MAIN -->|"createApi, main.ts:292"| API
    API --> CLI
    CLI -->|"spawn a Worker from the blob URL, main.ts:285"| WK
    WK --> WC
    MAIN --> TR
    JEV -->|"post injected, main.ts:217"| TR
    VIEWS --> GLC
    GLC -->|"dynamic import, graph-view.ts:49"| LE
    INV["INVARIANT: the only vault writer is VaultPort.process inside applyPatch<br/>api.ts:5,8,266"]
    API --- INV
    NOTE["see OM-09.1 for the engine pin seen from the engine side"]
    WK --- NOTE
```

**What it shows.** Everything Obsidian-specific sits in `main.ts`, the settings tab, views and modals; the API, client and worker loop take injected ports and run under Node tests. Engine wire detail is OO-04, the network edge OO-05.

## OO-01.2 onload: synchronous wiring, deferred first index

```mermaid
sequenceDiagram
    autonumber
    participant OB as Obsidian workspace
    participant P as OntomarkPlugin<br/>main.ts:119
    participant A as OntomarkApi<br/>api.ts:101
    participant C as OntomarkClient<br/>client.ts:144
    OB->>P: onload (main.ts:119)
    P->>P: loadData then normaliseSettings (main.ts:120,121)
    P->>P: worker source becomes a blob URL (main.ts:123)
    P->>A: createApi wires client spawn, wasm bytes, progress, restart Notice (main.ts:124,284,285,286,287,289)
    P->>OB: startApi defers the first index to layout-ready (main.ts:125,319)
    P->>OB: settings tab, three views, code block, ribbon (main.ts:127,128,129,130,131,132)
    P->>OB: vault listeners only after layout-ready (main.ts:138)
    P->>OB: eleven commands (main.ts:139)
    OB-->>P: layout ready
    P->>P: loadIgnoreFilters from configDir app.json (main.ts:311,336)
    P->>A: start with the profile (main.ts:312)
    A->>A: listMarkdown, excluded paths filtered (main.ts:267)
    A->>C: start with every file (api.ts:102,103)
    C->>C: bump generation, spawn and load back to back (client.ts:147,148,369,370,371)
    A-->>P: indexed event (api.ts:104)
    P->>P: autoClassify debounced 750 ms when the setting is on (main.ts:296,115)
```

**What it shows.** Load returns before any note is read: the index and the vault listeners both wait for layout-ready, so the startup `create` storm is never indexed twice (`main.ts:136,137`). `ready` is the promise every later step awaits.

## OO-01.3 The change relay that outlives a reindex

```mermaid
flowchart TD
    CE["OntomarkApi.emit: indexed, classified, classify_failed<br/>api.ts:63,64,65,66,299"]
    PR["client onProgress becomes a progress event<br/>main.ts:287"]
    SUB["api.onChange re-subscribed on every createApi<br/>main.ts:293,294"]
    EMIT["OntomarkPlugin.emit, a throwing listener is swallowed<br/>main.ts:349,350,353"]
    AUTO["indexed plus autoClassify setting runs classify<br/>main.ts:296"]
    LS["plugin.subscribe listener set<br/>main.ts:192,193"]
    EV["EntityView re-renders, progress text on progress<br/>entity-view.ts:48,49,55"]
    GV["GraphView refreshes on indexed or classified<br/>graph-view.ts:57"]
    SB["each SPARQL block debounces a rerun<br/>sparql-block.ts:54,55,57"]
    CE --> SUB --> EMIT
    PR --> EMIT
    SUB --> AUTO
    EMIT --> LS
    LS --> EV
    LS --> GV
    LS --> SB
    INV["INVARIANT: views subscribe to the plugin, not to an api, because a reindex replaces the api<br/>main.ts:191<br/>sparql-block.ts:52,53"]
    LS --- INV
    NOTE["see OM-09.2 and OM-09.3 for the block lifetime"]
    SB --- NOTE
```

**What it shows.** One listener set lives on the plugin and survives `reindex`; each new API is wired into it in `createApi`. The query view is not a subscriber: it runs only when the user presses Run.

## OO-01.4 onunload and reindex teardown

```mermaid
flowchart TD
    UN["onunload<br/>main.ts:142"]
    T1["cancel pending autoClassify<br/>main.ts:143"]
    T2["abort the extraction at its next chunk boundary<br/>main.ts:146,147"]
    T3["forget both session keys, mark unloaded<br/>main.ts:148,149,150"]
    T4["unsubscribe from the api, clear listeners<br/>main.ts:151,152,153"]
    T5["dispose the api: terminate worker, reject pending as disposed<br/>main.ts:154<br/>client.ts:283,286,287"]
    T6["revoke the worker blob URL<br/>main.ts:155,156"]
    UN --> T1 --> T2 --> T3 --> T4 --> T5 --> T6
    RX["reindex: dispose, createApi, startApi again<br/>main.ts:179,180,181,182"]
    RX --> T5
    QUIET["disposed is a quiet code, so in-flight work ends without a Notice<br/>client.ts:48,51,52"]
    T5 --- QUIET
    INV["INVARIANT: unloaded makes jevConfig report disabled, so no Jev request starts after unload<br/>main.ts:205"]
    T3 --- INV
    DEBT["DEBT: a change to Obsidian's own Excluded files is seen only at the next settings save, reindex or reload, no app.json event is used<br/>main.ts:328,329,330,331"]
    RX --- DEBT
```

**What it shows.** Unload and reindex share the dispose step; only unload clears keys and listeners. Work still holding the old API rejects quietly with `disposed` rather than touching the new engine (`main.ts:480,481`).
