---
id: WV-01
title: Repository anatomy, entry points and the status record
area: wasmvowl
governing:
  - ../WasmVOWL/README.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
adrs: [WasmVOWL:ADR-001]
sources:
  - ../WasmVOWL/README.md
  - ../WasmVOWL/CLAUDE.md
  - ../WasmVOWL/STATUS.md
  - ../WasmVOWL/TEST-RESULTS-SUMMARY.md
  - ../WasmVOWL/package.json
  - ../WasmVOWL/tests/README.md
  - ../WasmVOWL/docs/adr/ADR-001-explorer-consumer-contract-closeout.md
  - ../WasmVOWL/modern/package.json
  - ../WasmVOWL/modern/index.html
  - ../WasmVOWL/modern/README.md
  - ../WasmVOWL/modern/src/main.tsx
  - ../WasmVOWL/modern/src/App.tsx
  - ../WasmVOWL/legacy/src/index.html
  - ../WasmVOWL/legacy/webpack.config.js
verified_commit: {wasmvowl: 2bb54b3ee2dc3a7eda34b60d9f08fdb0fb535aab}
---

## WV-01.1 Three code bodies under one root, and the engine outside it

```mermaid
flowchart TB
    ROOT["root package.json, name webvowl 1.1.7, private<br/>WasmVOWL/package.json:2-3,50"]
    subgraph MOD["modern/ - the React explorer"]
        MPKG["package wasmvowl 2.0.0, private<br/>modern/package.json:2-4"]
        MDEP["React 19, R3F 9, drei 10, three, zustand, immer<br/>modern/package.json:21-27"]
    end
    subgraph LEG["legacy/ - upstream WebVOWL on d3 v3, kept for reference"]
        LWP["two webpack entries, webvowl and webvowl.app<br/>legacy/webpack.config.js:8-11"]
    end
    subgraph ORPH["tests/ - orphaned legacy suite"]
        TR["not runnable, it targets a root src/ that no longer exists<br/>tests/README.md:3-7"]
    end
    ENG["@dreamlab-ai/vowl-wasm 0.4.0 exact from npm, driven through<br/>@dreamlab-ai/ontomark-graph 0.4.0, modern/package.json:19-20<br/>CHANGED at 2bb54b3: was the v0.1.1 tarball. EXTERNAL: the engine, see VW-01.1"]
    ROOT -->|"npm test"| MPKG
    ROOT -->|"package, release, test:legacy, webserver"| LWP
    MPKG --> MDEP
    MPKG --> ENG
    MOD --> LEG
    LEG --> ORPH
    N1["The Rust crate is no longer vendored, it lives in DreamLab-AI/vowl-wasm<br/>WasmVOWL/README.md:19-21, no Rust toolchain needed WasmVOWL/README.md:29-30"]
    ENG --- N1
```

Root scripts and their targets are WV-04.1; the legacy body is WV-05.

## WV-01.2 Two entry points, one per code body

```mermaid
flowchart TB
    subgraph M["modern - Vite SPA"]
        MH["modern/index.html:10-11<br/>div root, module script /src/main.tsx"] --> MM["createRoot(root).render(App) in StrictMode<br/>main.tsx:6-10"]
        MM --> MA["App gate: hasData and nodes.size above 0<br/>App.tsx:15-23"]
        MA -->|"nothing loaded"| FDZ["FileDropZone, see WV-02.1<br/>App.tsx:35-36"]
        MA -->|"graph loaded"| GC["TopMenuBar, GraphCanvas, Sidebar, see WV-02.3<br/>App.tsx:32,38-41"]
    end
    subgraph L["legacy - webpack bundle into deploy/"]
        LH["legacy/src/index.html:539-541<br/>d3.min.js, js/webvowl.js, js/webvowl.app.js"] --> LO["window.onload = webvowl.app().initialize<br/>legacy/src/index.html:542"]
        LO --> LG["libraryTarget assign, globals named after each entry<br/>legacy/webpack.config.js:17-18, see WV-05.1"]
    end
    M --> L
    N2["DOC-DRIFT: the page title is still the Vite scaffold, modern/index.html:7,<br/>and the app header says WebVOWL Modern, App.tsx:28"]
    MH --- N2
```

## WV-01.3 What the status documents claim, against the held decision record

```mermaid
flowchart TB
    S1["STATUS.md:4 PRODUCTION READY, 100 percent complete<br/>STATUS.md:12 4-10x faster than legacy D3"]
    S2["STATUS.md:160-162 build the WASM in ../rust-wasm<br/>STATUS.md:303 rust-wasm/ in the directory tree"]
    S3["TEST-RESULTS-SUMMARY.md:13 Rust/WASM tests 47/47 passed<br/>TEST-RESULTS-SUMMARY.md:121 location rust-wasm/"]
    S4["modern/README.md:1-3 Vite React template text"]
    A1["ADR-001 axes: decision proposed, implementation partial, activation inactive<br/>ADR-001-explorer-consumer-contract-closeout.md:5-7<br/>owner role unassigned :8"]
    A2["Prior native test counts describe the removed local engine, not the installed archive<br/>ADR-001-explorer-consumer-contract-closeout.md:39"]
    A3["Standalone demo held, readiness and performance not re-certified<br/>WasmVOWL/README.md:83, CLAUDE.md:107"]
    S1 --> D1["DOC-DRIFT: no production claim survives the ADR, the README or CLAUDE.md"]
    A1 --> D1
    A3 --> D1
    S2 --> D2["DOC-DRIFT: rust-wasm/ is gone since the extraction, WasmVOWL/README.md:19-21"]
    S3 --> D2
    A2 --> D2
    S4 --> D3["DEBT: modern/ has no README of its own beyond the scaffold"]
    D1 --> D3
    D2 --> D3
```

STATUS.md and TEST-RESULTS-SUMMARY.md are dated 2025-11-10 (`STATUS.md:3`, `TEST-RESULTS-SUMMARY.md:3`) and were not updated by the extraction.
