---
id: VG-06
title: publishing-tools/WasmVOWL — the tree that actually ships, and how it has drifted from its sibling
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: []
sources:
  - ../visionGraph/publishing-tools/WasmVOWL/CLAUDE.md
  - ../visionGraph/publishing-tools/WasmVOWL/CAPABILITIES.md
  - ../visionGraph/publishing-tools/WasmVOWL/Dockerfile
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/mesh.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/router.tsx
  - ../visionGraph/.github/workflows/publish.yml
  - ../knowledgeGraph/explorer/CLAUDE.md
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/explorer/modern/src/site/mesh.ts
verified_commit: 9e308164c
---

## VG-06.1 Two copies of one codebase — 149 of 159 files byte-identical

```mermaid
flowchart TB
    subgraph HERE["visionGraph/publishing-tools/WasmVOWL — 180 files<br/>THE tree publish.yml builds, smokes and deploys"]
        MODERN["modern/ — React 19 + Vite 6 SPA<br/>same router.tsx, same GraphPage/NGG1 stack"]
    end
    subgraph SIBLING["knowledgeGraph/explorer — the OTHER copy<br/>built by nothing in that repo — see KG-04.6"]
        MODERN2["modern/ — same file tree, same structure"]
    end
    HERE -.->|"149/159 non-generated files BYTE-IDENTICAL<br/>#40;filecmp, this commit#41;"| SIBLING
    DIFF["10 files differ: package.json #40;dependency pin#41;,<br/>package-lock.json, CLAUDE.md, llms.txt #40;x2#41;,<br/>AboutPage/DataPage/HomePage.tsx, site/mesh.ts, site.css"]
    HERE --> DIFF
    SIBLING --> DIFF
    note1["Neither copy is strictly ahead: VG-06.2 below shows THIS #40;shipping#41; tree's<br/>CLAUDE.md is MORE stale than knowledgeGraph's copy of the same file, while<br/>VG-06.4 shows THIS tree's site/mesh.ts is missing an export knowledgeGraph's<br/>copy already has. They have diverged in BOTH directions, independently edited"]
```

## VG-06.2 DOC-DRIFT — this tree's own CLAUDE.md still documents a Rust crate that isn't here

```mermaid
flowchart TB
    CLAUDEMD["publishing-tools/WasmVOWL/CLAUDE.md — 12 references<br/>to rust-wasm/, incl. lines 25, 39, 96"]
    L25["line 25: 'rust-wasm/  # Physics engine' —<br/>directory tree diagram showing src/ontology, src/graph,<br/>src/layout #40;Barnes-Hut#41;, src/bindings"]
    L39["line 39-49: 'cd rust-wasm ... wasm-pack build<br/>--target web --release ... cargo test ... cargo bench'"]
    L96["line 96: 'cd rust-wasm && wasm-pack build<br/>--target web --release' — repeated as the WASM<br/>rebuild recipe"]
    REALITY["REALITY: no rust-wasm/ directory exists anywhere<br/>in publishing-tools/WasmVOWL at this commit —<br/>WasmVOWL/modern/package.json:20 pins '@dreamlab-ai/vowl-wasm': '0.1.2'<br/>from the npm registry"]
    CLAUDEMD --> L25 & L39 & L96
    L25 & L39 & L96 -.->|"none of these commands can succeed"| REALITY
    note1["DOC-DRIFT #40;worse than the sibling copy#41;: knowledgeGraph/explorer/CLAUDE.md<br/>has ALREADY been corrected to say 'The physics engine is NOT in this repo...<br/>consumed as a pinned, integrity-locked dependency' — the fix landed in the<br/>NON-shipping copy and was never propagated to the tree that actually ships"]
```

## VG-06.3 CAPABILITIES.md — evidence from before the vault split and before externalisation

```mermaid
flowchart LR
    TOOLCHAIN["CAPABILITIES.md:6 — 'measured with rustc 1.97.0,<br/>wasm-pack 0.15.0, node 22.23.1, vite 6.4.3, python 3.12'"]
    NOTOOL["no Rust toolchain is invoked anywhere in publish.yml —<br/>npm ci pulls the PUBLISHED @dreamlab-ai/vowl-wasm package<br/>EXTERNAL: see VG-03.3"]
    PIPEPATH["CAPABILITIES.md:24 — 'python -m pipeline.build<br/>mainKnowledgeGraph/pages www': 7,444 pages"]
    REALPATH["actual pipeline invocation today: python -m pipeline.build<br/>knowledge/pages www — publish.yml:95-97"]
    TOOLCHAIN -.->|"stale — pre-externalisation local Rust build"| NOTOOL
    PIPEPATH -.->|"stale — pre-split path;<br/>mainKnowledgeGraph/ was the Logseq-era vault root"| REALPATH
    note1["Both pieces of evidence in CAPABILITIES.md's 'Working' table predate two<br/>structural changes #40;the vowl-wasm externalisation and the Logseq→Obsidian<br/>vault split#41; that this repo's own docs #40;VG-01, VG-06.2#41; record elsewhere —<br/>the capability claims were never re-measured against the current tree"]
```

## VG-06.4 site/mesh.ts — the seven-repo ecosystem chrome, itself out of sync with its sibling

```mermaid
flowchart TB
    MESH["site/mesh.ts — PRD-NG-001 §8, DDD §5<br/>'Ecosystem context — shared kernel of identity'<br/>MESH_REPOS: VisionFlow, VisionClaw, agentbox,<br/>solid-pod-rs, nostr-rust-forum, + 2 more"]
    ROUTER["router.tsx mounts App #40;SiteChrome#41;<br/>6 lazy routes — IDENTICAL to knowledgeGraph's copy,<br/>byte-for-byte #40;not in the 10-file diff list#41;"]
    NOLOOM["THIS #40;shipping#41; copy — mesh.ts ends at line 27,<br/>no LOOM export"]
    LOOM["knowledgeGraph/explorer copy — explorer/modern/src/site/mesh.ts:38<br/>adds 'export const LOOM = #123; repo: .../loom #125;'<br/>the grounding-node link, kept separate from<br/>MESH_REPOS 'so the canonical seven-repository<br/>mesh copy stays intact'"]
    MESH --> ROUTER
    MESH -.->|"missing an edit present in the sibling"| NOLOOM
    note1["DIVERGENCE: the direction of drift is the OPPOSITE of VG-06.2 — there the<br/>SHIPPING tree #40;this one#41; has the stale content and the sibling has the fix;<br/>here the sibling #40;knowledgeGraph/explorer#41; has an edit #40;the Loom link#41; that<br/>was never ported INTO the tree that actually deploys to narrativegoldmine.com"]
```

## VG-06.5 Dockerfile — a vestigial WebVOWL 1.1.7/Tomcat image nothing in the deploy path uses

```mermaid
flowchart LR
    DOCKER["publishing-tools/WasmVOWL/Dockerfile — 15 lines<br/>FROM tomcat:9-jre8-alpine<br/>wget .../webvowl_1.1.7.war into ROOT.war"]
    DEPLOY["actual deploy path: publish.yml npm run build<br/>→ peaceiris/actions-gh-pages@v3 → gh-pages branch<br/>EXTERNAL: VG-03.1 — no Docker build, no container runtime"]
    DOCKER -.->|"not referenced by publish.yml, docker-compose.yml,<br/>or any script in this tree"| DEPLOY
    note1["Inherited unchanged from upstream WebVOWL — a legacy artefact from the<br/>pre-React, Java-Tomcat-served WebVOWL 1.1.7 lineage #40;see also KG-04.5's<br/>'WebVOWL lineage, and what was rewritten'#41;. Harmless, but describes a<br/>deployment mechanism this repo does not use"]
```
