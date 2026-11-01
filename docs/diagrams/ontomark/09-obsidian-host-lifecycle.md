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
verified_commit: {ontomark-obsidian: 2a6c44e024121ec055329d37915e0ce5a05f75f2}
---

## For developers

The Obsidian plugin is a separate engine consumer with its own repository revision. These panels describe host behaviour at plugin 0.4.2, including reproduced failures. The [refresh audit](../../estate-review/2026-11-01-obsidian-refresh.md) binds five whole-ADR assessments, current engine execution and bounded lifecycle probes. The ADR ledger's metadata migration is not a revalidation of implementation or activation.

## For the business

A user can query typed notes locally and review a proposed note edit before applying it. Reliable saved queries and clear recovery after an index failure are still product gaps. Optional model features send requests outside the vault; a local graph does not make those actions offline. The build and shim tests establish bounded behaviour, not installed-desktop acceptance.

## OM-09.1 Verified inputs and three release files

```mermaid
flowchart TD
    P["Engine archive and source pins<br/>engine.json:3,5,6"] --> H["Download and check SHA-256<br/>build-wasm.mjs:17,21"]
    H --> R["Check source receipt on release path<br/>build-wasm.mjs:44,45"]
    R --> W["Inline worker and WASM into main.js<br/>esbuild.config.mjs:11,32,46"]
    G["Versioned shared graph dependency<br/>package.json:25"] --> W
    W --> T["Build, unit, WASM, bundle and offline gates<br/>release.yml:14"]
    T --> A["Attest three runtime files<br/>release.yml:15,18"]
    A --> F["Publish main.js, manifest.json, styles.css<br/>release.yml:24"]
```

**What it shows.** The release workflow publishes loose runtime files, not the removed ZIP. A fresh audit build verified the configured engine archive and source receipt. Workflow presence alone does not prove a published asset's attestation was verified.

**Why it is this way.** ADR-OBS-001 keeps reusable semantics and graph rendering upstream, while the plugin owns installation and host adapters. Licences and notices now live in the repository; this diagram makes no distribution-compliance determination.

## OM-09.2 Host filters and replacement of the API

```mermaid
flowchart TD
    C["Top-level folders plus native ignore filters<br/>excludes.ts:57,65"] --> L["Filter initial Markdown listing<br/>main.ts:267"]
    C --> E["Create and modify wait, then filter<br/>main.ts:381,385,386"]
    E --> I["Update current captured API<br/>main.ts:383,387"]
    C --> N{"Renamed target excluded?<br/>main.ts:403,409"}
    N -->|yes| DEL["Delete old path<br/>main.ts:409"]
    N -->|no| O["Rename or index target<br/>main.ts:411; api.ts:127"]
    S["Settings save reloads native filters<br/>main.ts:167,169"] --> RE["Dispose API and create a new one<br/>main.ts:180,181,182"]
    RE --> L
    RE -.-> Q["TENSION: blocks subscribe to old API<br/>sparql-block.ts:40"]
    Q --> FIX["Stable plugin subscription exists<br/>main.ts:193,294"]
```

**What it shows.** Filtering belongs to the host's listing and event adapters. A settings-triggered reindex replaces the API. Existing blocks subscribe directly to the old instance, unlike consumers using the stable plugin subscription.

**Why it is this way.** ADR-OBS-005 separates vault policy from engine semantics. Malformed native configuration can remove the extra filters by design; these convenience rules are not an authorisation boundary inside every public API method.

## OM-09.3 Saved queries have a lifetime as well as a generation

```mermaid
flowchart TD
    B["One renderer child per fence<br/>sparql-block.ts:120,121"] --> SUB["Subscribe and debounce 400 ms<br/>sparql-block.ts:19,40,43"]
    SUB --> RW["Resolve note references and rewrite<br/>sparql-block.ts:62,63,105"]
    RW --> Q["Query channel uses note path<br/>sparql-block.ts:71"]
    Q --> GU["Latest request per channel wins<br/>guard.ts:46,68"]
    GU -.-> COLL["TENSION: two blocks in one note collide<br/>sparql-block.ts:71"]
    Q -->|success| R
    Q --> CL["not_classified triggers classify and retry<br/>sparql-block.ts:73,74,75"]
    CL --> R["Render after await<br/>sparql-block.ts:77,79"]
    U["Unload only unsubscribes and clears timer<br/>sparql-block.ts:51,54"] -.-> R
    RW -.-> SC["DRIFT: placeholder lacks token boundary<br/>directive accepted beyond leading comment<br/>sparql-rewrite.ts:48,51,89"]
```

**What it shows.** The guard prevents obsolete reads on one channel. Reusing a channel across independent blocks defeats that independence, and unsubscribing does not invalidate an already running render. All three lifecycle cases and both scanner mismatches were reproduced with current host modules.

**Why it is this way.** ADR-OBS-004 promises a reactive in-note result. That needs both per-block request identity and component lifetime checks. Inferred queries can also classify automatically, qualifying ADR-OBS-001's phrase “classification remains explicit”.

## OM-09.4 Model waiting, review, persistence and indexing

```mermaid
flowchart TD
    CMD["Always-registered command checks endpoint and key<br/>main.ts:454,463,559"] --> PLAN["Shared engine plans request<br/>main.ts:478,479"]
    PLAN --> NET["Send chunks; abort checked between chunks<br/>ingest-transport.ts:149,152"]
    NET --> WAIT["Timeout races host request; no abort handle<br/>ingest-transport.ts:85,91,92"]
    WAIT --> ACCEPT["Check abort, accept, optional Jev judgement<br/>main.ts:488,489,494"]
    ACCEPT --> CONF["Preview requires confirmation<br/>main.ts:501; api.ts:240"]
    CONF --> WRITE["Atomic full-text comparison and write<br/>api.ts:242,243,246"]
    WRITE --> INDEX["Then update engine index<br/>api.ts:253"]
    INDEX --> RECEIPT["Return applied outcome<br/>api.ts:255"]
    INDEX -.-> FAIL["TENSION: rejection can follow persisted write<br/>upsert is outside write-error catch<br/>api.ts:248,253"]
```

**What it shows.** Proposal, confirmation, persistence and indexing are separate events. The timeout stops waiting; it does not recall a request. The current host has no second abort check after optional Jev judging and before confirmation. The write/index failure was reproduced; the Jev unload race remains a source-derived test target.

**Why it is this way.** ADR-OBS-003 places credentials and network calls in the host and keeps the engine pure. Its original absent-command and one-destination wording exceeds current enforcement: the command is always registered, optional Jev broadens the surface, and the host transport does not expose redirect control.
