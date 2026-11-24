---
id: OO-02
title: Commands, views, the graph layout controller and guarded edits
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-001-shared-graph.md
  - ../ontomark-obsidian/docs/desktop-acceptance.md
adrs: [ADR-OBS-001]
sources:
  - ../ontomark-obsidian/src/main.ts
  - ../ontomark-obsidian/src/api.ts
  - ../ontomark-obsidian/src/client.ts
  - ../ontomark-obsidian/src/views/diff-modal.ts
  - ../ontomark-obsidian/src/views/entity-view.ts
  - ../ontomark-obsidian/src/views/query-view.ts
  - ../ontomark-obsidian/src/views/graph-view.ts
  - ../ontomark-obsidian/src/graph-layout.ts
  - ../ontomark-obsidian/src/layout-engine.ts
  - ../ontomark-obsidian/package.json
  - ../ontomark-obsidian/docs/desktop-acceptance.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-001-shared-graph.md
verified_commit: {ontomark-obsidian: e1f2e4c85fccb4587f96c07e4fca79014d5a7103}
---

## OO-02.1 Command palette and ribbon: eleven commands, one gated

```mermaid
flowchart TD
    REG["registerCommands<br/>main.ts:555"]
    JEV["jev-propose-edit, listed only while Jev is enabled<br/>main.ts:558,560,561"]
    EXT["extract-facts, always listed, checks endpoint and key on run<br/>main.ts:566,462,471"]
    OPEN["open-graph, open-panel, open-query<br/>main.ts:567,568,569"]
    CLS["classify-now, reindex<br/>main.ts:570,571"]
    KIND["make-class, make-individual, make-property<br/>main.ts:572,574,579"]
    PAR["add-parent through AddParentModal, brackets stripped<br/>main.ts:584,592,48"]
    RIB["ribbon icon opens the graph view<br/>main.ts:132,133"]
    REG --> JEV
    REG --> EXT
    REG --> OPEN
    REG --> CLS
    REG --> KIND
    REG --> PAR
    OV["openView reuses an existing leaf, else the right sidebar<br/>main.ts:260,262,264,266"]
    OPEN --> OV
    RIB --> OV
    SK["setActiveKind proposes a type change<br/>main.ts:425,432"]
    KIND --> SK
    NOTE1["see OO-05.3 for the Jev edit path and OM-09.4 for extract-facts"]
    JEV --- NOTE1
    EXT --- NOTE1
```

**What it shows.** Every command but one is registered unconditionally; the Jev command uses `checkCallback`, so with Jev off the palette is unchanged. The three edit commands and add-parent all end in the same guarded write (OO-02.2).

## OO-02.2 A guarded frontmatter edit, from proposal to announcement

```mermaid
sequenceDiagram
    autonumber
    participant P as OntomarkPlugin<br/>main.ts:425
    participant A as OntomarkApi<br/>api.ts:250
    participant C as OntomarkClient<br/>client.ts:195
    participant M as DiffModal<br/>diff-modal.ts:68
    participant V as Obsidian vault
    P->>A: proposeSetType or proposeAddParent (main.ts:432,594)
    A->>C: frontmatter digest and index keys, in parallel (api.ts:251,237)
    A->>A: add-parent picks the instance or subclass key and dedupes the link (api.ts:240,245)
    A->>C: proposePatch at the frontmatter digest (api.ts:252,246)
    C-->>A: PatchProposal with diff and base text
    A->>A: no_op returns before any dialog (api.ts:263)
    A->>M: confirm shows the unified diff (api.ts:264)
    M-->>A: true only on Apply, closing counts as cancel (diff-modal.ts:51,52,61,62)
    A->>V: process compares the current text with base_text (api.ts:266,267,268)
    V-->>A: source_changed or write error becomes refused (api.ts:272,273,275)
    A->>C: upsert the new text, failure still applied (api.ts:286,287,288)
    A-->>P: applied, no_op, declined or refused
    P->>P: announce each outcome in a Notice (main.ts:437,439,442,445,448)
```

**What it shows.** The engine plans the patch; the host alone writes, and only if the file still holds the planned text. A persisted write is `applied` even if the engine update after it fails (`api.ts:277,278,279`).

## OO-02.3 Views, their guard channels and how each keeps stale data off screen

```mermaid
flowchart TD
    EV["EntityView: index on channel entity-view, token per render<br/>entity-view.ts:66,69,76"]
    EVK["stale and superseded keep the last render, other errors show only before a first render<br/>entity-view.ts:72,82"]
    EVW["incomplete-classification warning and plain-note count<br/>entity-view.ts:107,112,162,165"]
    QV["QueryView: run sequence, query on channel query-view, row limit from settings<br/>query-view.ts:127,130,136"]
    QVC["not_classified triggers classify then one retry<br/>query-view.ts:138,142"]
    QVQ["quiet or overtaken errors keep what is shown<br/>query-view.ts:150"]
    GV["GraphView: request counter, index on channel graph-view<br/>graph-view.ts:68,71,72"]
    GVH["hierarchyGraph with the include-inferred toggle<br/>graph-view.ts:73,44"]
    GVO["open note looks the IRI up on channel graph-open-note<br/>graph-view.ts:92"]
    GVL["entity list capped at 100 buttons<br/>graph-view.ts:94,95"]
    EV --> EVK
    EV --> EVW
    QV --> QVC
    QV --> QVQ
    GV --> GVH
    GV --> GVO
    GV --> GVL
    GUARD["one channel per consumer, latest request wins<br/>client.ts:179,191"]
    EV --> GUARD
    QV --> GUARD
    GV --> GUARD
    OPEN["OPEN: the Jev ask row is built only in onOpen, so toggling Jev needs the panel reopened<br/>query-view.ts:72"]
    QV --- OPEN
    NOTE["see OM-09.3 for the per-block channels of in-note queries"]
    GUARD --- NOTE
```

**What it shows.** Each view names its own channel and adds a local counter, so neither a superseded engine reply nor an older local run can repaint it. The query view does not subscribe to index events; the other two do (OO-01.3).

## OO-02.4 The graph layout controller: renderer first, layout optional

```mermaid
flowchart TD
    MOUNT["first refresh mounts the shared renderer then binds the controller<br/>graph-view.ts:79,81"]
    PARAMS["CHANGED at e1f2e4c, 0.6.0: the controller is built with the Graph layout settings as params<br/>graph-view.ts:49,54<br/>graph-layout.ts:43,44"]
    BIND["bind starts loading the engine once<br/>graph-layout.ts:48,53,54,57"]
    LOAD["loadVowlEngine compiles the embedded bytes, a failed compile is not cached<br/>layout-engine.ts:26,28,29,33,34"]
    READY["engineReady creates the layout driver with params and onIgnoredParams<br/>graph-layout.ts:118,126,128,129"]
    DROP["view closed while loading: dispose the engine<br/>graph-layout.ts:121"]
    POS["onPositions sets renderer positions and fits until the reader interacts<br/>graph-layout.ts:130,134,78"]
    SHOW["later refreshes call show: renderer setGraph, then layout setGraph<br/>graph-view.ts:77<br/>graph-layout.ts:69,72,151"]
    DRAG["drag pins and unpins nodes in the layout<br/>graph-layout.ts:82,84,90"]
    LIVE["CHANGED at e1f2e4c: a settings save reaches setLayoutParams, merged and passed to the running layout<br/>graph-view.ts:65<br/>graph-layout.ts:94,96,98,100"]
    IGN["ignored keys only warn, the engine keeps its default<br/>graph-layout.ts:101,169,170"]
    FAIL["fail once: destroy the layout, warn, status says layout unavailable<br/>graph-layout.ts:157,158,163,164,30"]
    MOUNT --> PARAMS --> BIND --> LOAD
    LOAD -->|"resolved, graph-layout.ts:118"| READY
    LOAD -->|"rejected, graph-layout.ts:157"| FAIL
    READY --> POS
    READY --> DROP
    SHOW --> POS
    DRAG --> POS
    LIVE --> POS
    LIVE --> IGN
    SHOW -->|"setGraph throws, graph-layout.ts:151"| FAIL
    LIVE -->|"setParams throws, graph-layout.ts:102,103"| FAIL
    INV["INVARIANT: every graph change goes through show, so renderer and layout hold the same graph and node order<br/>graph-layout.ts:69,72"]
    SHOW --- INV
    INV2["INVARIANT: params set before the engine arrives are kept and start the first layout<br/>graph-layout.ts:96,98,128"]
    LIVE --- INV2
    DRIFT["DOC-DRIFT: the 30 October acceptance note says the plugin runs no layout engine, superseded by the 14 November section<br/>and the 23 November 0.4.0 update, neither yet run in installed Obsidian<br/>desktop-acceptance.md:25,78,84"]
    LOAD --- DRIFT
    EXT["EXTERNAL: renderer and layout driver come from ontomark-graph 0.4.0, the simulation from vowl-wasm 0.4.0,<br/>CHANGED at e1f2e4c from 0.3.0 and 0.2.3<br/>package.json:25,26"]
    MOUNT --- EXT
```

**What it shows.** The graph always draws; the force layout is an optional layer whose failure only changes the status line. ADR-OBS-001's ledger still reads implementation `none` (`ADR-OBS-001-shared-graph.md:6`) although this view consumes the shared graph package (`graph-view.ts:3`).
