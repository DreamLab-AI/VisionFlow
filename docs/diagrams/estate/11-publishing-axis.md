---
id: ES-11
title: Publishing axis — corpus, bundle and kit edges the other estate topics do not own
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - docs/architecture/repository-map.md
adrs: []
sources:
  - ../visionGraph/.github/workflows/publish.yml
  - ../knowledgeGraph/explorer/modern/package.json
  - ../knowledgeGraph/CNAME
  - ../visionGraph/publishing-tools/WasmVOWL/modern/package.json
  - ../vowl-wasm/Cargo.toml
  - ../dreamlab-ai-website/.github/workflows/rust-ci.yml
  - ../dreamlab-ai-website/.github/workflows/kit-pin-guard.yml
  - scripts/estate-health/roster.json
  - scripts/estate-health.mjs
  - .github/workflows/drift-counter.yml
  - ../project/.github/workflows/ontology-publish.yml
  - ../project/src/services/ontology_pull.rs
verified_commit: {visionclaw: 36bb64e1e, visiongraph: 9e308164c, knowledgegraph: 2791111fc, vowl-wasm: 65e2d1e78, dreamlab-ai-website: 9a3dd8830, visionflow: bec06dc3a}
---
## ES-11.1 The publishing axis — five cross-repo edges, each previously drawn from one side only
```mermaid
flowchart TB
    subgraph AUTHOR["Authoring"]
        VG["EXTERNAL: visionGraph — the vault<br/>private, jjohare/visionGraph. see VG-01"]
    end
    subgraph PUB["Publication"]
        PUBYML["visionGraph publish.yml<br/>publish.yml:280 peaceiris/actions-gh-pages@v3"]
        KG["EXTERNAL: knowledgeGraph — Pages host<br/>CNAME:1 narrativegoldmine.com. see KG-01"]
        REL["VisionClaw ontology-latest release<br/>ontology-publish.yml:181. see ES-09.13"]
    end
    subgraph CONSUME["Consumption"]
        POD["VisionClaw embedded pod<br/>ontology_pull.rs:290. see ES-08.11"]
        EXPL["EXTERNAL: two VOWL explorers<br/>see KG-04, VG-03"]
        LOOM["Ontology Loom grounding<br/>see ES-06.1"]
    end
    subgraph PINS["Version pins that can drift"]
        VW["EXTERNAL: vowl-wasm 0.1.2<br/>Cargo.toml:5. see VW-01"]
        NF["EXTERNAL: nostr-rust-forum kit<br/>see NF-01"]
        DW["EXTERNAL: dreamlab-ai-website<br/>KIT_REF rust-ci.yml:21. see DW-01"]
    end

    VG -->|"edge 11 — cross-ORG deploy"| PUBYML
    PUBYML -->|"external_repository DreamLab-AI/knowledgeGraph<br/>publish_branch gh-pages, publish.yml:283-284"| KG
    VG -->|"edge 9 — the SAME corpus, a second path"| REL
    REL --> POD
    KG -->|"edge 16 — /ns/v2.jsonld is the contract<br/>consumers dereference, roster.json:32-36"| LOOM
    KG -->|"edge 12"| EXPL
    VW --> EXPL
    NF -->|"edge 13 — build-time clone at a pinned SHA"| DW

    INV["INVARIANT — ONE authored corpus, TWO distribution paths.<br/>The gh-pages deploy publishes the browsable site to<br/>narrativegoldmine.com; the release feeds the embedded pod<br/>by boot pull. Neither is the other's fallback: a change to<br/>the vault reaches consumers by BOTH routes, on different<br/>clocks. see ES-09.13 and ES-08.11"]
    GAP["Why this topic exists — before it, estate/ named none of<br/>vowl-wasm, KIT_REF, narrativegoldmine, roster.json or<br/>gh-pages. Each of these five edges was drawn only from the<br/>repo that owns one END of it, so a contributor changing a<br/>pin had no diagram showing who breaks."]

    PUBYML --> INV
    PINS --> GAP
```

## ES-11.2 Edge 11 — visionGraph publishes into another org's repo
```mermaid
sequenceDiagram
    autonumber
    participant VG as "EXTERNAL: visionGraph publish.yml"<br/>see VG-03
    participant EX as "existing gh-pages checkout"
    participant WWW as "www/ build output"
    participant KG as "EXTERNAL: DreamLab-AI/knowledgeGraph"<br/>see KG-05

    VG->>WWW: write the CNAME into the artefact, publish.yml:203
    VG->>KG: clone --branch gh-pages --depth 1, publish.yml:269-270
    KG-->>EX: the currently published tree
    VG->>WWW: preserve existing /notes SPA if present, publish.yml:272-275
    VG->>KG: actions-gh-pages@v3 publish.yml:280<br/>external_repository :283, publish_branch gh-pages :284
    Note over VG,KG: INVARIANT — this is a CROSS-ORG, CROSS-REPO deploy:<br/>the workflow lives in jjohare/visionGraph and writes into<br/>DreamLab-AI/knowledgeGraph, authenticated by a personal_token<br/>secret (publish.yml:282). knowledgeGraph does not build the<br/>site it serves — it is a publication TARGET.
    Note over EX: The /notes preservation step is why the deploy is not a<br/>clean replace — a directory the publisher never builds is<br/>carried across from the previous publication, so gh-pages<br/>content has two authors.
    Note over KG: knowledgeGraph/CNAME:1 is narrativegoldmine.com — the same<br/>domain publish.yml:203 writes, so the CNAME exists on both<br/>sides of the deploy. see KG-01
```

## ES-11.3 Edge 12 — two VOWL explorers, two divergent bundle pins
```mermaid
flowchart LR
    CRATE["EXTERNAL: vowl-wasm<br/>version 0.1.2, Cargo.toml:5<br/>see VW-01"]

    subgraph KGE["EXTERNAL: knowledgeGraph explorer — see KG-04"]
        KGP["@dreamlab-ai/vowl-wasm pinned to a<br/>RELEASE TARBALL URL for v0.1.1<br/>explorer/modern/package.json:20"]
    end
    subgraph VGE["EXTERNAL: visionGraph WasmVOWL — see VG-03"]
        VGP["@dreamlab-ai/vowl-wasm pinned to<br/>the registry range 0.1.2<br/>WasmVOWL/modern/package.json:20"]
    end

    CRATE -->|"published as a GitHub release asset"| KGP
    CRATE -->|"published to npm as @dreamlab-ai/vowl-wasm"| VGP

    D1["DOC-DRIFT — the two consumers are a MINOR version apart AND<br/>resolve by different mechanisms. knowledgeGraph fetches a<br/>frozen v0.1.1 tarball over HTTPS; visionGraph resolves 0.1.2<br/>from the registry. A vowl-wasm release therefore reaches one<br/>explorer and not the other, and no gate compares them."]
    D2["DIVERGENCE — the roster carries BOTH vowl-wasm and WasmVOWL<br/>as separate first-party repositories (roster.json:11,13), and<br/>registers @dreamlab-ai/vowl-wasm on npm alongside the<br/>crates.io crate (roster.json:41,46). The npm bundle is the<br/>artefact both explorers consume; the crate is what estate<br/>health actually measures. see VF-03.1"]

    KGP --> D1
    VGP --> D1
    CRATE --> D2
```

## ES-11.4 Edge 13 — the forum kit pin, and the guard that keeps four files in lockstep
```mermaid
sequenceDiagram
    autonumber
    participant NF as "EXTERNAL: nostr-rust-forum (the kit)"<br/>see NF-01
    participant CI as "EXTERNAL: dreamlab-ai-website rust-ci.yml"<br/>see DW-02
    participant GUARD as "EXTERNAL: kit-pin-guard.yml"<br/>see DW-02

    CI->>NF: git clone the kit, then checkout --detach $KIT_REF<br/>rust-ci.yml:34
    Note over CI: KIT_REF is a literal 40-char SHA pinned in the workflow<br/>env block, rust-ci.yml:21, with the comment at :18<br/>requiring lockstep with workers-deploy.yml and deploy.yml
    NF-->>CI: the kit tree at exactly that commit
    GUARD->>GUARD: compare the pin across deploy.yml, workers-deploy.yml,<br/>rust-ci.yml, forum-config/Cargo.toml crate versions and the<br/>CANONICAL_ entries in the kit-compatibility record
    alt any of the five disagrees
        GUARD--xCI: "::error::Kit pin drift detected" kit-pin-guard.yml:29
    end
    Note over NF,GUARD: INVARIANT — the website consumes the forum kit at BUILD time<br/>by source clone, not as a published package. The pin is<br/>therefore a SHA, and correctness depends on five files<br/>agreeing — kit-pin-guard.yml is the only thing that checks it.
    Note over NF: EXTERNAL — this diagram asserts only what the WEBSITE's<br/>workflows state about the kit. The kit's own contract is<br/>NF's to describe. see NF-01 and DW-02
```

## ES-11.5 Edges 14 and 15 — the roster is the estate's only enumeration, and what reads it
```mermaid
flowchart TB
    ROSTER["scripts/estate-health/roster.json<br/>14 repos, plus surfaces and registries<br/>the _doc at roster.json:3 calls it the ONLY<br/>place the estate is enumerated"]

    subgraph READERS["What walks it"]
        EH["scripts/estate-health.mjs collect<br/>reads the roster and interrogates the GitHub API,<br/>the public surfaces and the package registries<br/>estate-health.mjs:18-20. see VF-03"]
        DC[".github/workflows/drift-counter.yml<br/>checks out ONE sibling at a pinned ref<br/>drift-counter.yml:57-62. see VF-05"]
    end

    subgraph TARGETS["What it reaches"]
        R1["EXTERNAL: VisionClaw · agentbox · solid-pod-rs<br/>nostr-rust-forum · dreamlab-ai-website<br/>see VC-01, AB-01, SP-01, NF-01, DW-01"]
        R2["EXTERNAL: knowledgeGraph · visionGraph · vowl-wasm<br/>see KG-01, VG-01, VW-01"]
        R3["EXTERNAL: loom · WasmVOWL · prose-sanitiser<br/>diagram-ir · dream-engine — rostered with NO<br/>diagram area in this tree"]
        SURF["public surfaces incl. narrativegoldmine.com<br/>and /ns/v2.jsonld, roster.json:20-36"]
        REG["registries — crates.io and npm,<br/>incl. @dreamlab-ai/vowl-wasm, roster.json:40-46"]
    end

    ROSTER --> EH
    ROSTER --> DC
    EH --> R1
    EH --> R2
    EH --> R3
    EH --> SURF
    EH --> REG

    INV["INVARIANT roster.json:3 — adding a repository to the estate means<br/>adding a ROW here: the collector holds no repository names, and<br/>the page, the snapshot order and the dream evaluator all follow<br/>roster order. This is the enumeration the tree treats as canonical."]
    D1["DIVERGENCE — drift-counter does NOT walk the roster. It pins ONE<br/>sibling, agentbox, at a literal SHA (drift-counter.yml:60-61) so<br/>that moving the count source is a reviewed two-line diff rather<br/>than a silent change (drift-counter.yml:55-56). The two gates<br/>therefore disagree about what the estate is."]
    D2["DIVERGENCE — ES-01.1 draws SIX repositories, the count<br/>agentbox's ecosystem doc uses. That is the VisionClaw checkout's<br/>own neighbourhood, not the estate: it omits knowledgeGraph,<br/>visionGraph, vowl-wasm and five more the roster carries.<br/>ES-01.1 now states its scope explicitly. see VF-08.3"]

    ROSTER --> INV
    DC --> D1
    ROSTER --> D2
```
