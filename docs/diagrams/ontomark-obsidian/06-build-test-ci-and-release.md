---
id: OO-06
title: Build, bundle gates, tests, CI, release and the ADR ledger
area: ontomark-obsidian
governing:
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/README.md
  - ../ontomark-obsidian/docs/desktop-acceptance.md
adrs: [ADR-OBS-001, ADR-OBS-002, ADR-OBS-003]
sources:
  - ../ontomark-obsidian/engine.json
  - ../ontomark-obsidian/package.json
  - ../ontomark-obsidian/manifest.json
  - ../ontomark-obsidian/scripts/build-wasm.mjs
  - ../ontomark-obsidian/esbuild.config.mjs
  - ../ontomark-obsidian/scripts/check-bundle.mjs
  - ../ontomark-obsidian/scripts/offline-test.mjs
  - ../ontomark-obsidian/scripts/adr-ledger.sh
  - ../ontomark-obsidian/vitest.config.ts
  - ../ontomark-obsidian/test/wasm-harness.test.mjs
  - ../ontomark-obsidian/test/e2e/offline.e2e.cjs
  - ../ontomark-obsidian/.github/workflows/ci.yml
  - ../ontomark-obsidian/.github/workflows/release.yml
  - ../ontomark-obsidian/README.md
  - ../ontomark-obsidian/docs/adr/README.md
  - ../ontomark-obsidian/docs/evidence/provenance-engine-0.9.0.md
verified_commit: {ontomark-obsidian: e1f2e4c85fccb4587f96c07e4fca79014d5a7103}
---

## OO-06.1 npm run build: pinned engine in, three runtime files out

```mermaid
flowchart TD
    PIN["engine.json: DreamLab-AI/ontomark v0.9.0, asset, SHA-256, source commit<br/>engine.json:2,3,4,5,6"]
    BUILD["build script: build-wasm, tsc --noEmit, esbuild production<br/>package.json:10"]
    SHA["a 64-hex SHA-256 is required, the download must match it<br/>build-wasm.mjs:13,17,21"]
    LOCAL["ONTOMARK_ENGINE_BUNDLE substitutes a local archive and its own digest<br/>build-wasm.mjs:11,12,15"]
    SAFE["refuse absolute or parent paths, extract into src/generated<br/>build-wasm.mjs:25,26,27,33"]
    DTS{"tracked glue .d.ts differs from the fetched one?<br/>build-wasm.mjs:30,31,39"}
    CIF["under CI: fail, commit the new declaration<br/>build-wasm.mjs:40,41"]
    WARN["locally: warning only<br/>build-wasm.mjs:42"]
    RCPT["engine-manifest receipt must equal the pinned source, skipped for a local archive<br/>build-wasm.mjs:44,45"]
    P1["esbuild pass 1: worker as an IIFE string<br/>esbuild.config.mjs:11,15,25"]
    P2["pass 2: main.js as CJS, worker source virtual module, .wasm as binary<br/>esbuild.config.mjs:30,38,42,46"]
    EXT["obsidian, electron, codemirror and lezer stay external<br/>esbuild.config.mjs:45"]
    OUT["dist: main.js, manifest.json, styles.css<br/>esbuild.config.mjs:41,53,54"]
    PIN --> BUILD --> SHA --> SAFE --> DTS
    LOCAL --> SHA
    DTS -->|yes| CIF
    DTS -->|yes| WARN
    DTS -->|no| RCPT
    WARN --> RCPT
    RCPT --> P1 --> P2 --> OUT
    P2 --- EXT
    PROV["engine 0.9.0 adoption adds the read-only review_binding export, no certification UI<br/>provenance-engine-0.9.0.md:3,4,5,6"]
    PIN --- PROV
    NOTE["see OM-09.1 for the same pin from the engine corpus"]
    PIN --- NOTE
```

**What it shows.** No Rust toolchain is needed: the engine arrives as a digest-checked release archive, and the only tracked generated file is the glue's type declaration. The installed plugin is the three files in `dist/`, with the worker and both WebAssembly modules inlined (`esbuild.config.mjs:1,2,3,4`).

## OO-06.2 check-bundle: the static offline proof on the shipped main.js

```mermaid
flowchart TD
    IN["check-bundle on dist/main.js<br/>package.json:15<br/>check-bundle.mjs:8,9"]
    FORB["forbidden: fetch, XMLHttpRequest, WebSocket, EventSource, sendBeacon, importScripts, import.meta, node network modules, electron net, Obsidian request<br/>check-bundle.mjs:10,11,12,13,14,15,16,17,18,19"]
    ONE["requestUrl appears exactly once in the shipped file<br/>check-bundle.mjs:36,37,38"]
    MOD["and, in an unminified rebuild, only inside the ingest-transport module<br/>check-bundle.mjs:47,64,70,71"]
    URL["URL literals: only W3C, Dublin Core and FOAF namespace IRIs<br/>check-bundle.mjs:74,75,76"]
    WASM["engine embedded: base64 wasm magic present<br/>check-bundle.mjs:78,79,80"]
    TWO["at least two embedded wasm payloads, engine and layout<br/>check-bundle.mjs:83,84,85"]
    WRK["worker source embedded, found by classify_start<br/>check-bundle.mjs:86,87"]
    EXIT["exit 1 on any finding<br/>check-bundle.mjs:90"]
    IN --> FORB --> ONE --> MOD --> URL --> WASM --> TWO --> WRK --> EXIT
    NV["harness proves the check is not vacuous: it flags the raw web glue<br/>wasm-harness.test.mjs:98"]
    FORB --- NV
    INV["INVARIANT: the network primitive is confined to one module, the rule ADR-OBS-003 sets<br/>check-bundle.mjs:32,33,34,35"]
    MOD --- INV
```

**What it shows.** The check runs on the minified artefact users install, then on an unminified rebuild of the same sources to prove module locality. A single literal `requestUrl` is the whole permitted network surface.

## OO-06.3 Three test layers and what each one proves

```mermaid
flowchart TD
    UT["vitest: test/*.test.ts under Node with an Obsidian globals shim<br/>package.json:12<br/>vitest.config.ts:5,6,7"]
    WH["wasm harness: built node and web targets replay the script against the golden transcript<br/>package.json:13<br/>wasm-harness.test.mjs:71,78,85,90"]
    WR["wasm-only paths: entropy and clock execute<br/>wasm-harness.test.mjs:112"]
    OFF["offline test: stage a vault with the built plugin installed<br/>package.json:14<br/>offline-test.mjs:19,21,39"]
    CHN["3000 extra classes in an 8-ary taxonomy so classify spans many steps<br/>offline-test.mjs:32,33"]
    DOCK["docker run with network none, node 24 slim image, files streamed over stdin<br/>offline-test.mjs:17,49"]
    LOC["--local runs the same driver on the host, tripwire only, not an offline proof<br/>offline-test.mjs:7,16"]
    E2E["driver evaluates main.js as Obsidian does, checks three installed files and zero network attempts<br/>offline.e2e.cjs:6,7,89,374"]
    WH --> WR
    OFF --> CHN
    OFF --> DOCK --> E2E
    OFF --> LOC
    TENS["TENSION: test names say the wasm matches native, but this golden is blessed from the wasm replay itself, the vault has no native writer<br/>wasm-harness.test.mjs:66,67,68,69,70,72,74"]
    WH --- TENS
    GAP["OPEN: none of the three is installed-Obsidian acceptance, kept as a separate record<br/>ontomark-obsidian/README.md:150"]
    E2E --- GAP
```

**What it shows.** Unit tests cover the pure modules, the harness pins engine output byte for byte, and the offline test exercises the built plugin with networking removed. Native parity rests on Ontomark's own CI for its fixtures (`wasm-harness.test.mjs:68,69`).

## OO-06.4 CI, tag release and the ADR ledger gate

```mermaid
flowchart TD
    subgraph CI["Plugin checks, every push and pull request, read-only token"]
        C1["node 22, npm ci with dev dependencies<br/>ci.yml:2,3,10,11"]
        C2["build, unit, wasm, bundle check<br/>ci.yml:12"]
        C3["offline test<br/>ci.yml:13"]
        L1["adr-ledger job: fetch the agentbox checker at a pinned commit, verify its SHA-256, then check-index<br/>ci.yml:14,25<br/>adr-ledger.sh:19,20,29,30,34"]
        L2["CHANGED at e1f2e4c: checker pin moved from agentbox 8bd4314 to 1ea5494, with its SHA-256<br/>adr-ledger.sh:18,19,20"]
    end
    subgraph REL["Plugin release on a tag that starts with a digit"]
        R0["write contents, OIDC and attestations<br/>release.yml:2,3"]
        R1["tag must equal manifest.json version<br/>release.yml:12<br/>manifest.json:4"]
        R2["same build and test chain, offline included<br/>release.yml:14"]
        R3["attest build provenance of the three files<br/>release.yml:16,19,20,21"]
        R4["gh release create with the three files and generated notes<br/>release.yml:24"]
    end
    C1 --> C2 --> C3
    L1 --- L2
    R0 --> R1 --> R2 --> R3 --> R4
    OPEN1["OPEN: only manifest.json is compared with the tag, package.json and versions.json are not checked<br/>release.yml:12"]
    R1 --- OPEN1
    DRIFT1["CHANGED at e1f2e4c: README working status now names manifest 0.6.0 and the pinned engine v0.9.0 source f9ba200b,<br/>so the earlier v0.8.1 doc drift is gone<br/>ontomark-obsidian/README.md:148<br/>engine.json:3,6<br/>manifest.json:4"]
    R4 --- DRIFT1
    DRIFT2["DOC-DRIFT: ledger lists ADR-OBS-001 implementation none and 002, 003 proposed, none, inactive, while the graph view and ingest ship<br/>adr/README.md:9,10,11"]
    L1 --- DRIFT2
    DRIFT3["DOC-DRIFT: README local decisions list stops at ADR-OBS-004, ADR-OBS-005 is in the ledger<br/>ontomark-obsidian/README.md:206<br/>adr/README.md:13"]
    L1 --- DRIFT3
```

**What it shows.** CI and release run the same gates; only the release adds the tag check, attestation and publication. The ledger gate checks record syntax and index freshness, not whether a record's implementation status matches the code.
