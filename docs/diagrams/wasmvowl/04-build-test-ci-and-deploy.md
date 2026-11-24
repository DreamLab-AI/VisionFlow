---
id: WV-04
title: Build, test, CI and deploy surface
area: wasmvowl
governing:
  - ../WasmVOWL/README.md
  - ../WasmVOWL/tests/README.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
adrs: [WasmVOWL:ADR-001]
sources:
  - ../WasmVOWL/README.md
  - ../WasmVOWL/package.json
  - ../WasmVOWL/.github/workflows/ci.yml
  - ../WasmVOWL/.gitignore
  - ../WasmVOWL/checks/root-surface.test.mjs
  - ../WasmVOWL/tests/README.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
  - ../WasmVOWL/modern/package.json
  - ../WasmVOWL/modern/vitest.config.ts
  - ../WasmVOWL/modern/tests/setup.ts
  - ../WasmVOWL/modern/tests/unit/useWasmSimulation.test.ts
  - ../WasmVOWL/modern/tests/integration/wasm-integration.test.ts
  - ../WasmVOWL/modern/tests/integration/engine.real.test.ts
  - ../WasmVOWL/modern/tests/TEST-DOCUMENTATION.md
  - ../WasmVOWL/legacy/Gruntfile.js
  - ../WasmVOWL/legacy/test/karma.conf.js
verified_commit: {wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab}
---

## WV-04.1 Every script, by where it runs

```mermaid
flowchart TB
    subgraph ROOT["root package.json scripts"]
        RT["test: npm --prefix modern test, then node --test checks/root-surface.test.mjs<br/>WasmVOWL/package.json:45"]
        RL["package, release, test:legacy, webserver: grunt --gruntfile legacy/Gruntfile.js --base legacy<br/>WasmVOWL/package.json:43-44,46-47"]
    end
    subgraph MOD["modern/package.json scripts"]
        MB["build: tsc -b then vite build, dev: vite<br/>modern/package.json:8-9"]
        MT["test: vitest run, plus watch, ui and coverage variants<br/>modern/package.json:13-16"]
        ML["lint: eslint, type-check: tsc -b<br/>modern/package.json:10,12"]
    end
    subgraph LEG["legacy/Gruntfile.js tasks"]
        GT["package, release, zip, webserver, test, test-ci<br/>legacy/Gruntfile.js:170-178"]
    end
    RT --> MT
    RL --> GT
    MB --> ML
    N1["INVARIANT: install never triggers a legacy build, no postinstall<br/>checks/root-surface.test.mjs:23-25"]
    RT --- N1
```

## WV-04.2 What each test layer actually exercises

```mermaid
flowchart TB
    VC["vitest: jsdom, NODE_ENV test, setup file<br/>vitest.config.ts:8-15<br/>coverage thresholds 80/80/75/80, vitest.config.ts:27-32"] --> SU["setup shims canvas, ResizeObserver, IntersectionObserver<br/>only where a DOM exists, tests/setup.ts:16-74"]
    SU --> HT["hook test: the real driver and adapter, only the WASM module doubled<br/>useWasmSimulation.test.ts:1-9,35-40"]
    SU --> IT["store integration test imports only the graph store and types<br/>wasm-integration.test.ts:5-7"]
    RE["engine.real.test.ts, node environment, the pinned bundle from its own bytes<br/>engine.real.test.ts:1-11"] --> R1["bundle reports 0.4.0, app defaults equal the recommended set with radius 23<br/>engine.real.test.ts:173-185"]
    RE --> R2["each non-empty legacy file: one load, settles under 2000 ticks,<br/>every pair at least 2 x 23 - 1 apart<br/>engine.real.test.ts:240-258"]
    R2 --> C1["CHANGED at 2bb54b3: the real engine now runs in the suite,<br/>the old no-real-engine gap and the unused wasmMock.ts are gone"]
    HT --> C1
    C1 --> DD["DOC-DRIFT: TEST-DOCUMENTATION.md:19 still lists tests/__mocks__/wasmMock.ts"]
    RS["root checks F4, F5, F6 and CI: grunt scripts name a real Gruntfile, Docker stays retired,<br/>tests/README.md lists only real scripts, Travis stays out<br/>checks/root-surface.test.mjs:13-62"] --> LK
    LK["legacy karma: jasmine on PhantomJS, basePath legacy/<br/>legacy/test/karma.conf.js:3-13"]
    IT --> C1
```

The decision record's account of the evidence is its NggExplorer section (`ADR-001-explorer-consumer-contract-closeout.md:218-235`); the test guide describes the hook and real-engine layers the same way (`TEST-DOCUMENTATION.md:78,88`).

## WV-04.3 CI and deploy: what exists and what does not

```mermaid
flowchart TB
    CI["GitHub Actions ci.yml on push and pull request to master<br/>.github/workflows/ci.yml:8-13"] --> J1["modern job, Node 22: npm ci, lint, type-check, test, build<br/>.github/workflows/ci.yml:23-45"]
    CI --> J2["root-surface job: node --test checks/root-surface.test.mjs<br/>.github/workflows/ci.yml:47-56"]
    J2 --> G1["INVARIANT: .travis.yml stays deleted, ci.yml runs every modern gate,<br/>actions pinned by 40-hex sha, no Pages or deploy step<br/>checks/root-surface.test.mjs:48-62"]
    CI --> C1["CHANGED at 2bb54b3: replaces the Travis config that ran Node 4-6<br/>and grunt jshint with no root Gruntfile, .github/workflows/ci.yml:3-6"]
    MB["modern build to Vite output, no hosting or deploy config in the repo<br/>modern/package.json:9, .github/workflows/ci.yml:6"] --> D2["EXTERNAL: the estate roster exempts this repo as ci_by_design, see VF-03.1,<br/>which predates this CI. The deployed explorer is the visionGraph copy, see VG-06.1"]
    LR["legacy release and zip write deploy/ and webvowl-*.zip<br/>legacy/Gruntfile.js:9,174-175, ignored by .gitignore:13-15"] --> O1["OPEN: legacy has no package.json, and with --base legacy the Gruntfile copies d3 from<br/>node_modules/d3 under legacy/, legacy/Gruntfile.js:68, as karma loads it, legacy/test/karma.conf.js:6<br/>CI does not run the legacy build"]
    DK["Docker image retired 2026-11-21, it served upstream WebVOWL on Tomcat<br/>checks/root-surface.test.mjs:31-36"] --> D2
    O1 --> D2
```

The README's setup is local, `npm ci` then `npm run dev` on port 5173, and its CI paragraph matches the workflow (`WasmVOWL/README.md:41-54,63-65`).
