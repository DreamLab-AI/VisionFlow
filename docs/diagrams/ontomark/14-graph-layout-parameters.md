---
id: OM-14
title: ontomark-graph 0.4.0 — layout parameters, the vowl-wasm version gate and its consumers
area: ontomark
governing:
  - ../ontomark/docs/adr/ADR-OM-002-shared-graph-and-obsidian-boundary.md
  - ../ontomark/packages/graph/README.md
adrs: [ADR-OM-002]
sources:
  - ../ontomark/packages/graph/src/layout.js
  - ../ontomark/packages/graph/src/layout.d.ts
  - ../ontomark/packages/graph/test/layout.test.mjs
  - ../ontomark/packages/graph/package.json
  - ../ontomark/packages/graph/CHANGELOG.md
  - ../ontomark/packages/graph/README.md
  - ../ontomark/.github/workflows/ci.yml
  - ../ontomark/docs/adr/ADR-OM-002-shared-graph-and-obsidian-boundary.md
  - ../ontomark-obsidian/src/settings.ts
  - ../ontomark-obsidian/package.json
verified_commit: {ontomark: 8222170c5243f58f1f9a93ac800660dfe0197ac4, ontomark-obsidian: e1f2e4c85fccb4587f96c07e4fca79014d5a7103}
---

## For developers

`@dreamlab-ai/ontomark-graph` 0.4.0 (e639efe) lets a host pass engine parameters through the layout driver; every addition is opt-in, so a host that passes none gets the 0.3.0 layout.

## OM-14.1 Parameters through the driver: at creation and live

```mermaid
flowchart TD
    OPT["createLayout options: params, onIgnoredParams<br/>layout.js:172,174<br/>layout.d.ts:75,77"]
    CHK["checkParams: an object of numbers, else TypeError<br/>layout.js:105,106,175"]
    TUNE["tune: engine.setParams, or every key ignored when the engine has none<br/>layout.js:211,212"]
    IGN["ignored keys reach onIgnoredParams, never an error<br/>layout.js:213"]
    FIRST["params from the options are applied before any graph<br/>layout.js:216"]
    LIVE["layout.setParams: check, tune, merged over earlier values<br/>layout.js:259,261,263<br/>layout.d.ts:95,97"]
    WARM["a loaded layout with an applied key reheats at warmAlpha and resumes<br/>layout.js:264,265,266"]
    FAIL["a throwing engine fails the layout and reports every key ignored<br/>layout.js:263"]
    OPT --> CHK --> FIRST --> TUNE --> IGN
    LIVE --> TUNE
    LIVE --> WARM
    LIVE --> FAIL
    INV["INVARIANT: without params no setParams call is made, so the 0.3.0 layout is unchanged<br/>layout.js:175,216<br/>layout.test.mjs:286,290"]
    FIRST --- INV
    TST["test: an engine without setParams ignores keys, a live change reheats a settled layout<br/>layout.test.mjs:237,240,241,252,253"]
    WARM --- TST
```

**What it shows.** Parameters are an engine concern the driver only relays; unknown keys are reported to the host, and a live change wakes a settled layout.

## OM-14.2 vowlEngine: which setParam keys a bundle has, read from its version

```mermaid
flowchart TD
    TAB["VOWL_PARAMS: keys by the vowl-wasm release that added them<br/>layout.js:89"]
    K020["0.2.0: charge, linkDistance, linkStrength, damping, centerGravity, alphaMin<br/>layout.js:90"]
    K021["0.2.1: theta<br/>layout.js:91"]
    K040["0.4.0: collideRadius, collideStrength, collideIterations, componentGravity<br/>layout.js:92"]
    TAB --> K020 --> K021 --> K040
    VER["vowlParamKeys: no setParam, no keys; read mod.version, a missing or unparsable one gets the 0.2.0 keys<br/>layout.js:95,96,98,99,101,102"]
    SUP["supportedParams frozen per engine<br/>layout.js:123"]
    SET["setParams: unsupported, non-number or non-finite keys are ignored, the rest set and remembered<br/>layout.js:132,135,136,138"]
    LOAD["after every load the remembered values are set again<br/>layout.js:125,142,144,153"]
    K040 --> VER --> SUP --> SET --> LOAD
    REC["RECOMMENDED_LAYOUT_PARAMS frozen: collideRadius 21, collideStrength 0.7,<br/>collideIterations 3, componentGravity 0.1<br/>layout.js:84<br/>layout.d.ts:43,106"]
    REC --> SET
    INV["INVARIANT: reheat is an action, never applied as a parameter<br/>layout.js:87,88<br/>layout.test.mjs:269,271"]
    SET --- INV
    TST["test: 0.3.0 and 0.2.0 bundles are not told a key applied that they would drop<br/>layout.test.mjs:277,279,281,282"]
    VER --- TST
```

**What it shows.** `NggExplorer.setParam` silently drops unknown keys and has no getter, so the bundle's own `version()` is the only way the adapter can report truthfully what took effect.

## OM-14.3 Release, test evidence and the first consumer

```mermaid
flowchart TD
    PKG["package 0.4.0, CHANGELOG entry dated 2026-11-23, GitHub release asset not npm<br/>graph/package.json:3<br/>CHANGELOG.md:3,5,7"]
    DEV["vowl-wasm 0.4.0 pinned as a dev dependency only<br/>graph/package.json:33,34"]
    CI["CI installs it and sets ONTOMARK_REQUIRE_VOWL, so the real-engine tests cannot skip<br/>ci.yml:14,15,16,17,18<br/>layout.test.mjs:299,302"]
    SAME["without params the real engine reproduces positions recorded from the 0.3.0 driver<br/>layout.test.mjs:351,353,354,355"]
    SEP["recommended params keep nodes two radii apart and draw loose components in<br/>layout.test.mjs:360,367,370"]
    PKG --> DEV --> CI
    CI --> SAME
    CI --> SEP
    OBS["EXTERNAL: ontomark-obsidian 0.6.0 pins graph 0.4.0 and defaults its Graph layout settings<br/>to RECOMMENDED_LAYOUT_PARAMS, see OO-03.1 and OO-04.5<br/>ontomark-obsidian/package.json:25<br/>settings.ts:59,60,61,62"]
    SEP --> OBS
    DRIFT["DOC-DRIFT: ADR-OM-002's graph addendum stops at package 0.3.0 and vowl-wasm 0.2.3<br/>ADR-OM-002-shared-graph-and-obsidian-boundary.md:60,66"]
    PKG --- DRIFT
```

**What it shows.** The "exactly 0.3.0 without params" claim is a recorded-position test against the real bundle, not an assertion in prose, and CI cannot skip it.
