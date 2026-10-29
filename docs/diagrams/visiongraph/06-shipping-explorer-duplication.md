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
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/types/scope.ts
  - ../visionGraph/publishing-tools/WasmVOWL/modern/src/site/domains.ts
  - ../visionGraph/.github/workflows/publish.yml
  - ../knowledgeGraph/explorer/CLAUDE.md
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/explorer/modern/src/site/mesh.ts
  - ../knowledgeGraph/explorer/modern/src/types/scope.ts
  - ../knowledgeGraph/explorer/modern/src/site/domains.ts
verified_commit: {visiongraph: 59e7dcc8d05ca447d03c74c819a740b390b93074, knowledgegraph: 1edcfc93de90ccff564073ef82cb198c422f75e3}
---

## VG-06.1 Two copies of one codebase — 39 of 108 shared files now differ

```mermaid
flowchart TB
    subgraph HERE["visionGraph/publishing-tools/WasmVOWL<br/>THE tree publish.yml builds, smokes and deploys"]
        MODERN["modern/src/router.tsx — 135 lines<br/>adds readBaseUrl#40;#41; + basename for sub-path deploys<br/>WasmVOWL/modern/src/router.tsx:64-75 and WasmVOWL/modern/src/router.tsx:135"]
    end
    subgraph SIBLING["knowledgeGraph/explorer — the OTHER copy<br/>built by nothing in that repo — see KG-04.6"]
        MODERN2["explorer/modern/src/router.tsx — 111 lines<br/>no readBaseUrl#40;#41;, createBrowserRouter takes no basename"]
    end
    HERE -.->|"router.tsx diverges: the shipping tree gained sub-path routing"| SIBLING
    DIFF["108 files exist in both trees and 39 of them differ — package.json, package-lock.json,<br/>README.md, llms.txt #40;x2#41;, About/Data/Home/GraphPage/PageView, site/mesh.ts, site.css,<br/>router.tsx, the nine-domain set: scope.ts, domains.ts, icons.tsx, palette.ts, EdgeLegend, theme.css,<br/>and the smoke and deployment scripts — see VG-06.5"]
    HERE --> DIFF
    SIBLING --> DIFF
    note1["Neither copy is strictly ahead: the shipping tree gained sub-path routing and nine domains,<br/>while VG-06.4 shows the sibling's site/mesh.ts has an export the shipping tree never received,<br/>and the sibling still carries Three.js canvas files that no longer exist in the shipping tree<br/>#40;EdgesSegments, LabelsTopN, NodesInstanced, lib/ngg1.ts#41;"]
    DOCKERFILE["Dockerfile — 15 lines, FROM tomcat:9-jre8-alpine,<br/>wget webvowl_1.1.7.war into ROOT.war — Dockerfile:6 and Dockerfile:13"]
    HERE --> DOCKERFILE
    note2["DOC-DRIFT: Dockerfile is vestigial — no 'docker' token<br/>anywhere in publish.yml; the actual deploy path is npm<br/>build #40;VG-03.1#41;. Inherited unchanged from the pre-React,<br/>Java-Tomcat WebVOWL 1.1.7 lineage #40;see also KG-04.6#41;"]
```

## VG-06.2 RESOLVED — this tree's CLAUDE.md no longer documents a Rust crate that is not here, but the two copies pin different vowl-wasm releases

```mermaid
flowchart TB
    SHIPCM["publishing-tools/WasmVOWL/CLAUDE.md:27-28 —<br/>'The physics engine is NOT in this repo... consumed here as a pinned,<br/>integrity-locked dependency'"]
    SIBCM["knowledgeGraph/explorer/CLAUDE.md:27-28 — the same sentence"]
    SHIPPKG["shipping WasmVOWL/modern/package.json:25 — @dreamlab-ai/vowl-wasm from the<br/>v0.2.0 release tarball, plus @dreamlab-ai/ontomark-graph at WasmVOWL/modern/package.json:24"]
    SIBPKG["sibling explorer/modern/package.json:20 — the v0.1.1 release tarball,<br/>no ontomark-graph dependency"]
    SHIPCM --> SHIPPKG
    SIBCM --> SIBPKG
    note1["RESOLVED: the earlier stamp recorded 12 references to a rust-wasm/ directory<br/>in the shipping tree's CLAUDE.md and a fix that had landed only in the sibling.<br/>Both files now carry the corrected paragraph at the same lines, and the shipping<br/>copy has no rust-wasm reference left"]
    note2["DRIFT: the two copies consume different releases of the engine, 0.2.0 shipping<br/>against 0.1.1 sibling, so the sibling cannot be dropped in without the<br/>owned decode and encode bindings that 0.2.0 adds #40;see VW-04.8#41;"]
```

## VG-06.3 CAPABILITIES.md — still measures a Python pipeline publish.yml no longer runs at all

```mermaid
flowchart LR
    TOOLCHAIN["CAPABILITIES.md:6 — 'measured with rustc 1.97.0,<br/>wasm-pack 0.15.0, node 22.23.1, vite 6.4.3, python 3.12'"]
    NOTOOL["no Rust toolchain builds the engine in publish.yml for this tree —<br/>npm ci pulls the PUBLISHED @dreamlab-ai/vowl-wasm release<br/>EXTERNAL: see VG-03.3"]
    PIPEPATH["CAPABILITIES.md:24 — 'python -m pipeline.build<br/>mainKnowledgeGraph/pages www': 7,444 pages"]
    REALPATH["publish.yml now downloads the released ontomark CLI by tag and digest<br/>and runs ontomark validate + ontomark build --vault all --publish-out<br/>publish.yml:98-117 and publish.yml:127-171;<br/>python -m pipeline.build does not appear in the workflow"]
    TOOLCHAIN -.->|"stale — pre-externalisation local Rust build"| NOTOOL
    PIPEPATH -.->|"doubly stale — the Python pipeline this line<br/>describes has been REPLACED, not just relocated"| REALPATH
    note1["DOC-DRIFT: CAPABILITIES.md's 'Working' table is dated 2026-07-23 #40;CAPABILITIES.md:5-6#41;<br/>and was already stale before the vowl-wasm externalisation and the Logseq to Obsidian<br/>split; the workflow it partly described has since moved from a Python pipeline to a<br/>VisionClaw vault binary and now to the released ontomark CLI — see VG-03 for that migration"]
```

## VG-06.4 site/mesh.ts and router.tsx — two files, drifting in OPPOSITE directions from the same sibling

```mermaid
flowchart TB
    MESH["site/mesh.ts — PRD-NG-001 §8, DDD §5<br/>'Ecosystem context — shared kernel of identity'<br/>MESH_REPOS: VisionFlow, VisionClaw, agentbox,<br/>solid-pod-rs, nostr-rust-forum, + 2 more<br/>WasmVOWL/modern/src/site/mesh.ts:15"]
    ROUTER["THIS #40;shipping#41; router.tsx — adds readBaseUrl#40;#41;<br/>+ basename for sub-path deploys, 135 lines<br/>WasmVOWL/modern/src/router.tsx:64-75 and WasmVOWL/modern/src/router.tsx:135 — see VG-06.1"]
    ROUTERSIB["knowledgeGraph/explorer's router.tsx — 111 lines,<br/>no readBaseUrl#40;#41;, fixed root-only routing"]
    NOLOOM["THIS #40;shipping#41; mesh.ts — 31 lines,<br/>no LOOM export"]
    LOOM["knowledgeGraph/explorer copy — explorer/modern/src/site/mesh.ts:38<br/>adds 'export const LOOM = #123; repo: .../loom #125;'<br/>the grounding-node link, kept separate from<br/>MESH_REPOS 'so the canonical seven-repository<br/>mesh copy stays intact'"]
    MESH -.->|"missing an edit present in the sibling"| NOLOOM
    ROUTER -.->|"gained a capability the sibling lacks"| ROUTERSIB
    note1["DIVERGENCE: mesh.ts and router.tsx drift in OPPOSITE directions from the<br/>same sibling repo. mesh.ts — the sibling #40;knowledgeGraph#47;explorer#41; has an<br/>edit #40;Loom link#41; never ported here. router.tsx — THIS #40;shipping#41; tree has<br/>gained sub-path basename support the sibling never received. Same-repo<br/>pair, no shared sync discipline either way"]
```

## VG-06.5 Nine domains in the shipping tree, six in the sibling

```mermaid
flowchart TB
    VGSCOPE["shipping tree — DOMAIN_SLUGS, index = NGG1 domain id,<br/>nine entries ending space-science-and-systems,<br/>earth-observation-and-geospatial-sensing and supply-chains-and-logistics<br/>WasmVOWL/modern/src/types/scope.ts:75-86"]
    VGMETA["shipping domains.ts — 'the nine-Domain descriptor table'<br/>WasmVOWL/modern/src/site/domains.ts:2"]
    KGSCOPE["knowledgeGraph/explorer — DOMAIN_SLUGS stops at<br/>infrastructure, six entries<br/>explorer/modern/src/types/scope.ts:72-80"]
    KGMETA["sibling domains.ts — 'the six-Domain descriptor table'<br/>explorer/modern/src/site/domains.ts:2"]
    VGSCOPE --> VGMETA
    KGSCOPE --> KGMETA
    VGSCOPE -.->|"ids 0-5 identical, 6, 7 and 8 only here"| KGSCOPE
    note1["INVARIANT: the slug order IS the NGG1 u16 domain id and is frozen,<br/>so new domains append — WasmVOWL/modern/src/types/scope.ts:75"]
```

**What it shows:** the space, Earth-observation and supply-chain domain changes landed only in the tree that ships. Its scope contract lists nine domain slugs, with the three new ones appended at ids 6, 7 and 8. The knowledgeGraph copy still lists six, and its descriptor table still calls itself "six-Domain".
**Why it is this way:** the domain work was done in visionGraph, where `publish.yml` builds the explorer, and no sync discipline carries edits to the sibling (VG-06.4). Because the sibling is built by nothing (KG-04.6), the gap does not reach the live site today. It does mean the sibling can no longer be dropped in as a replacement for the shipping tree.
**Debt:** the two explorer copies now differ in 39 of 108 shared files (counted at the two stamps by byte comparison, node_modules excluded; the earlier text counted 33 by a method this refresh did not repeat); the domain contract (`scope.ts`, `domains.ts`, icons, palette, legend) is the largest block.
