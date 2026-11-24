---
id: OM-09
title: Obsidian host — release, indexing, query lifecycle and guarded edits
area: ontomark
governing:
  - ../ontomark-obsidian/docs/adr/ADR-OBS-001-shared-graph.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-002-assisted-ingest-boundary-and-estate-notes.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-003-ingest-action-and-offline-contract.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-004-in-note-sparql-after-dot-triples.md
  - ../ontomark-obsidian/docs/adr/ADR-OBS-005-excluded-folders.md
adrs: [ADR-OBS-001, ADR-OBS-002, ADR-OBS-003, ADR-OBS-004, ADR-OBS-005]
sources:
  - ../ontomark-obsidian/engine.json
  - ../ontomark-obsidian/package.json
  - ../ontomark-obsidian/scripts/build-wasm.mjs
  - ../ontomark-obsidian/esbuild.config.mjs
  - ../ontomark-obsidian/.github/workflows/release.yml
  - ../ontomark-obsidian/src/main.ts
  - ../ontomark-obsidian/src/api.ts
  - ../ontomark-obsidian/src/excludes.ts
  - ../ontomark-obsidian/src/sparql-block.ts
  - ../ontomark-obsidian/src/sparql-rewrite.ts
  - ../ontomark-obsidian/src/guard.ts
  - ../ontomark-obsidian/src/ingest-transport.ts
  - ../ontomark-obsidian/src/ingest-run.ts
  - ../ontomark-obsidian/src/endpoint.ts
  - ../ontomark-obsidian/src/jev-ingest.ts
verified_commit: {ontomark-obsidian: e1f2e4c85fccb4587f96c07e4fca79014d5a7103}
---

## For developers

The Obsidian plugin is a separate engine consumer with its own repository revision. These panels describe plugin 0.6.0 (e1f2e4c; it adds Graph layout settings over vowl-wasm 0.4.0, see OO-03.3 and OO-04.5) and engine v0.9.0, adopted at 64bdc5b from ontomark source f9ba200 (`engine.json:3,6`); its new `review_binding` export is not called by the plugin. The 14 November source refresh reconciles per-block query channels, stable subscriptions, ingest cancellation and persisted-write reporting. Historical probe receipts remain bound to their original revisions. The [refresh audit](../../estate-review/2026-11-01-obsidian-refresh.md) binds five whole-ADR assessments, current engine execution and bounded lifecycle probes. The ADR ledger's metadata migration is not a revalidation of implementation or activation.

## For the business

A user can query typed notes locally and review a proposed note edit before applying it. Saved queries now retain independent channels and stop rendering after unload. A successful note write is reported as applied even when the following engine update fails. Optional model features send requests outside the vault; a local graph does not make those actions offline. The build and shim tests establish bounded behaviour, not installed-desktop acceptance.

## OM-09.1 Verified inputs and three release files

```mermaid
flowchart TD
    P["Engine archive and source pins<br/>engine.json:3,5,6"] --> H["Download and check SHA-256<br/>build-wasm.mjs:17,21"]
    H --> R["Check source receipt on release path<br/>build-wasm.mjs:44,45"]
    R --> W["Inline worker and WASM into main.js<br/>esbuild.config.mjs:11,32,46"]
    G["Versioned shared graph dependency, ontomark-graph 0.4.0 since e1f2e4c<br/>package.json:25"] --> W
    W --> T["Build, unit, WASM, bundle and offline gates<br/>release.yml:14"]
    T --> A["Attest three runtime files<br/>release.yml:15,18"]
    A --> F["Publish main.js, manifest.json, styles.css<br/>release.yml:24"]
```

**What it shows.** The release workflow publishes loose runtime files, not the removed ZIP. The linked 1 November audit verified the then-configured engine archive and source receipt. This source refresh checks the new pins, not a new release build. Workflow presence alone does not prove a published asset's attestation was verified.

**Why it is this way.** ADR-OBS-001 keeps reusable semantics and graph rendering upstream, while the plugin owns installation and host adapters. Licences and notices now live in the repository; this diagram makes no distribution-compliance determination.

## OM-09.2 Host filters and replacement of the API

```mermaid
flowchart TD
    C["Top-level folders plus native ignore filters<br/>excludes.ts:57,65"] --> L["Filter initial Markdown listing<br/>main.ts:275"]
    C --> E["Create and modify wait, then filter<br/>main.ts:389,393,394"]
    E --> I["Update current captured API<br/>main.ts:391,395"]
    C --> N{"Renamed target excluded?<br/>main.ts:411,417"}
    N -->|yes| DEL["Delete old path<br/>main.ts:417"]
    N -->|no| O["Rename or index target<br/>main.ts:419; api.ts:139"]
    S["Settings save reloads native filters<br/>main.ts:167,169"] --> RE["Dispose API and create a new one<br/>main.ts:180,181,182"]
    RE --> L
    RE --> Q["Blocks subscribe through the stable plugin relay<br/>sparql-block.ts:54"]
    Q --> FIX["Later index and classification events schedule a rerun<br/>sparql-block.ts:55-60"]
```

**What it shows.** Filtering belongs to the host's listing and event adapters. A settings-triggered reindex replaces the API. Existing blocks now subscribe through the stable plugin relay, so a replacement API does not strand their listeners.

**Why it is this way.** ADR-OBS-005 separates vault policy from engine semantics. Malformed native configuration can remove the extra filters by design; these convenience rules are not an authorisation boundary inside every public API method.

## OM-09.3 Saved queries have a lifetime as well as a generation

```mermaid
flowchart TD
    B["One renderer child per fence<br/>sparql-block.ts:144-146"] --> SUB["Stable subscription and 400 ms debounce<br/>sparql-block.ts:19,54-60"]
    SUB --> RW["Resolve note references and rewrite<br/>sparql-block.ts:79-81"]
    RW --> Q["Query channel includes a unique block counter<br/>sparql-block.ts:47,90"]
    Q --> GU["Latest request per channel wins<br/>guard.ts:46,68"]
    Q --> CL["not_classified triggers classify and retry<br/>sparql-block.ts:92-96"]
    GU --> CUR{"Still the latest block run?<br/>sparql-block.ts:76-80,98"}
    CL --> CUR
    CUR -->|yes| R["Render result with classification envelope<br/>sparql-block.ts:99-101"]
    U["Unload unsubscribes, clears timer and increments run sequence<br/>sparql-block.ts:65-70"] --> CUR
    RW -.-> SC["DRIFT: placeholder lacks token boundary<br/>directive accepted beyond leading comment<br/>sparql-rewrite.ts:48,51,89"]
```

**What it shows.** Each block has its own request channel, and each run checks its sequence before changing the display. Unloading invalidates an in-flight render. Concurrent entity lookups share one index read per generation (`api.ts:164-179`). These are source-level closures of the earlier lifecycle findings; the linked historical probes are not new installed-desktop acceptance.

**Why it is this way.** ADR-OBS-004 promises a reactive in-note result. That needs both per-block request identity and component lifetime checks. Inferred queries can still classify automatically, qualifying ADR-OBS-001's phrase “classification remains explicit”. The scanner mismatches remain separate from these lifecycle repairs.

## OM-09.4 Model waiting, review, persistence and indexing

```mermaid
flowchart TD
    CMD["Always-registered command checks endpoint and key<br/>main.ts:462-474,566"] --> PLAN["Capture one API and plan the request<br/>main.ts:490-495; ingest-run.ts:29"]
    PLAN --> NET["Chunk loop: abort check and ten-minute deadline<br/>ingest-transport.ts:154-168"]
    NET --> WAIT["Timeout races host request; no abort handle<br/>ingest-transport.ts:90-105,180"]
    WAIT --> ACCEPT["Accept and optional Jev judgement<br/>abort checked after each await<br/>ingest-run.ts:31-39"]
    ACCEPT --> CONF["Preview needs confirmation and final abort check<br/>ingest-run.ts:45-46; api.ts:264"]
    CONF --> WRITE["Atomic full-text comparison and write<br/>api.ts:266-271"]
    WRITE --> INDEX["Then update engine index<br/>api.ts:285-286"]
    INDEX --> RECEIPT["Return applied, including when index update fails<br/>api.ts:287-291"]
```

**What it shows.** Proposal, confirmation, persistence and indexing remain separate events. The ingest lifecycle checks cancellation after planning, model calls, acceptance and optional Jev, and again after confirmation. A persisted write stays an applied outcome when the engine update fails. The ten-minute deadline covers the model chunk loop; it is not a timer around planning, Jev or the user's review. A timeout stops waiting but does not recall the host request.

**Why it is this way.** ADR-OBS-003 places credentials and network calls in the host and keeps the engine pure. Plain HTTP outside loopback now requires explicit configuration (`endpoint.ts:18-21,45-49`). The original one-destination wording still exceeds enforcement: the host transport does not expose redirect control (`endpoint.ts:27-31`).

**Invariant (verification policy):** with the Jev withholding threshold above zero, unverified facts are withheld; at zero, the result is retained with an explicit `not_verified` state (`jev-ingest.ts:127-144`). This source review does not claim a live model or installed-desktop test.
