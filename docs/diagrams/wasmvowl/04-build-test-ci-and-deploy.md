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
  - ../WasmVOWL/.travis.yml
  - ../WasmVOWL/.gitignore
  - ../WasmVOWL/checks/root-surface.test.mjs
  - ../WasmVOWL/tests/README.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
  - ../WasmVOWL/modern/package.json
  - ../WasmVOWL/modern/vitest.config.ts
  - ../WasmVOWL/modern/tests/setup.ts
  - ../WasmVOWL/modern/tests/unit/useWasmSimulation.test.ts
  - ../WasmVOWL/modern/tests/integration/wasm-integration.test.ts
  - ../WasmVOWL/modern/tests/__mocks__/wasmMock.ts
  - ../WasmVOWL/modern/tests/TEST-DOCUMENTATION.md
  - ../WasmVOWL/legacy/Gruntfile.js
  - ../WasmVOWL/legacy/test/karma.conf.js
verified_commit: {wasmvowl: af25086a605482c8234580ccd1062cf506e0b181}
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
        ML["lint: eslint, type-check: tsc --noEmit<br/>modern/package.json:10,12"]
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
    VC["vitest: jsdom, NODE_ENV test, setup file<br/>vitest.config.ts:8-15<br/>coverage thresholds 80/80/75/80, vitest.config.ts:27-32"] --> SU["setup mocks canvas 2d context, ResizeObserver, IntersectionObserver<br/>tests/setup.ts:15-54"]
    SU --> HT["hook test replaces @dreamlab-ai/vowl-wasm wholesale with vi.mock<br/>useWasmSimulation.test.ts:16-41"]
    SU --> IT["integration test imports only the graph store and types, no WASM<br/>wasm-integration.test.ts:5-7"]
    SU --> MK["wasmMock.ts accepts data.nodes and data.edges<br/>wasmMock.ts:14-17<br/>documented, TEST-DOCUMENTATION.md:239, but imported by no test"]
    HT --> G1["TENSION: no test runs the real engine, so the WV-03.2 schema<br/>mismatch cannot fail the suite"]
    IT --> G1
    MK --> G1
    G1 --> CL["ADR-001 note of 2026-11-21: modern passes 79 of 79, hook/parser mismatch still open<br/>ADR-001-explorer-consumer-contract-closeout.md:25"]
    RS["root checks F4, F5, F6: grunt scripts name a real Gruntfile, Docker stays retired,<br/>tests/README.md lists only real scripts<br/>checks/root-surface.test.mjs:13-46"] --> CL
    LK["legacy karma: jasmine on PhantomJS, basePath legacy/<br/>legacy/test/karma.conf.js:3-13"] --> CL
```

The hook test's own comment records that its mock previously named a removed `rust-wasm/pkg` path and never applied (`useWasmSimulation.test.ts:12-15`).

## WV-04.3 CI and deploy: what exists and what does not

```mermaid
flowchart TB
    TR["only CI file is Travis: Node 4, 5, 6 and current, global grunt-cli,<br/>npm test then grunt jshint<br/>.travis.yml:1-10"] --> D1["DEBT: grunt jshint needs a root Gruntfile and the root has none,<br/>checks/root-surface.test.mjs:14. No .github workflows in the tree"]
    MB["modern build to Vite output, no hosting or deploy config in the repo<br/>modern/package.json:9"] --> D2["EXTERNAL: the estate roster exempts this repo as ci_by_design, see VF-03.1,<br/>which predates this area. The deployed explorer is the visionGraph copy, see VG-06.1"]
    LR["legacy release and zip write deploy/ and webvowl-*.zip<br/>legacy/Gruntfile.js:9,174-175, ignored by .gitignore:13-15"] --> O1["OPEN: legacy has no package.json, and with --base legacy the Gruntfile copies d3 from<br/>node_modules/d3 under legacy/, legacy/Gruntfile.js:68, as karma loads it, legacy/test/karma.conf.js:6"]
    DK["Docker image retired 2026-11-21, it served upstream WebVOWL on Tomcat<br/>checks/root-surface.test.mjs:31-36"] --> D2
    D1 --> D2
    O1 --> D2
```

The README's setup is local only, `npm ci` then `npm run dev` on port 5173 (`WasmVOWL/README.md:41-54`).
