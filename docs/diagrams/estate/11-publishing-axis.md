---
id: ES-11
title: Publishing axis — corpus, bundle and kit edges the other estate topics do not own
area: estate
governing:
  - ../project/docs/BASELINE-architecture.md
  - docs/architecture/repository-map.md
adrs: [visionclaw:ADR-2147]
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
  - ../project/docs/adr/ADR-2147-inferences-carry-a-re-checkable-certificate-and-proposals-carry-a-consequence-delta.md
  - ../dreamlab-ai-website/forum-config/Cargo.toml
verified_commit: {dreamlab-ai-website: a67cb2a8d474f4ba2e79bdbf55fad2e87bfd7e98, knowledgegraph: 1edcfc93de90ccff564073ef82cb198c422f75e3, visionclaw: 8692b12ba987ecb87165566beb7c320a2446e4b4, visionflow: 4d361b58ce2b3ee17e1570361eaf78270b630012, visiongraph: 7b0875f2c26b9e69dc5c97e9c88166acc14c47dc, vowl-wasm: 28af4365a68d9aacfd7c04f63299cc96ad717ce3}
---
## ES-11.1 The publishing axis — six cross-repo edges, each previously drawn from one side only
```mermaid
flowchart TB
    subgraph AUTHOR["Authoring"]
        VG["EXTERNAL: visionGraph — the vault<br/>private, jjohare/visionGraph. see VG-01"]
        OM["EXTERNAL: DreamLab-AI/ontomark — the released CLI<br/>v0.6.0 pinned by tag AND sha256, no build-from-source fallback<br/>publish.yml:48-51. see OM-03"]
    end
    subgraph PUB["Publication"]
        PUBYML["visionGraph publish.yml<br/>publish.yml:515-516 peaceiris/actions-gh-pages pinned SHA"]
        KG["EXTERNAL: knowledgeGraph — Pages host<br/>CNAME:1 narrativegoldmine.com. see KG-01"]
        REL["VisionClaw ontology-latest release<br/>ontology-publish.yml:287,332. see ES-09.13"]
    end
    subgraph CONSUME["Consumption"]
        POD["VisionClaw embedded pod<br/>ontology_pull.rs:295. see ES-08.11"]
        EXPL["EXTERNAL: two VOWL explorers<br/>see KG-04, VG-03"]
        LOOM["Ontology Loom grounding<br/>see ES-06.1"]
    end
    subgraph PINS["Version pins that can drift"]
        VW["EXTERNAL: vowl-wasm 0.2.2<br/>vowl-wasm/Cargo.toml:5. see VW-01"]
        NF["EXTERNAL: nostr-rust-forum kit<br/>see NF-01"]
        DW["EXTERNAL: dreamlab-ai-website<br/>KIT_REF rust-ci.yml:20. see DW-01"]
    end

    OM -->|"edge 17 - CHANGED 2026-10: publish.yml fetches the released ontomark CLI<br/>by tag and digest, publish.yml:98-115; the VisionClaw vault<br/>binary is no longer its producer"| PUBYML
    VG -->|"edge 11 - cross-ORG deploy"| PUBYML
    PUBYML -->|"external_repository DreamLab-AI/knowledgeGraph<br/>publish_branch gh-pages, publish.yml:519-520"| KG
    PUBYML -->|"edge 9 - since 2026-10 the SAME publish run dispatches<br/>corpus-sync with source_repo and source_sha, publish.yml:525-531,<br/>received at ontology-publish.yml:15-16, built at that sha :93"| REL
    REL --> POD
    KG -->|"edge 16 - /ns/v2.jsonld is the contract<br/>consumers dereference, roster.json:35-40"| LOOM
    KG -->|"edge 12"| EXPL
    VW --> EXPL
    NF -->|"edge 13 - build-time clone at a pinned SHA"| DW

    INV["INVARIANT - ONE authored corpus, TWO distribution paths.<br/>The gh-pages deploy publishes the browsable site to<br/>narrativegoldmine.com; the release feeds the embedded pod<br/>by boot pull. Neither is the other's fallback: a change to<br/>the vault reaches consumers by BOTH routes. The release is now<br/>triggered by the publish run itself, after the gh-pages deploy,<br/>and builds the corpus commit that run published, so the two<br/>routes start from one commit, while the pod still picks the<br/>release up on its own pull clock. Each route also carries a<br/>re-checkable derivation certificate (publish.yml:210-227,<br/>ontology-publish.yml:180-191). see ES-09.13 and ES-08.11"]
    GAP["Why this topic exists - before it, estate/ named none of<br/>vowl-wasm, KIT_REF, narrativegoldmine, roster.json, gh-pages<br/>or the CLI provisioning edge (then a vault binary, now the pinned ontomark release). Each was drawn only<br/>from the repo that owns one END of it, so a contributor<br/>changing a pin had no diagram showing who breaks."]

    PUBYML --> INV
    PINS --> GAP
    OM --> GAP
```

## ES-11.2 Edge 11 - visionGraph publishes into another org's repo, now via a pinned ontomark release, a certificate re-check and Quartz notes
```mermaid
sequenceDiagram
    autonumber
    participant VG as "EXTERNAL: visionGraph publish.yml"<br/>see VG-03
    participant OM as "EXTERNAL: DreamLab-AI/ontomark release"<br/>see OM-05
    participant OO as "EXTERNAL: oo-cert release<br/>fabio-rovai/open-ontologies"
    participant WWW as "www/ build output"
    participant KG as "EXTERNAL: DreamLab-AI/knowledgeGraph"<br/>see KG-05
    participant VC as "EXTERNAL: DreamLab-AI/VisionClaw"<br/>see VC-21

    VG->>OM: gh release download ONTOMARK_RELEASE_TAG, then sha256sum --check --strict, publish.yml:98-115
    Note over VG,OM: no fallback - a run that cannot obtain the pinned release<br/>fails, publish.yml:100-106 and the comment at publish.yml:41-47
    VG->>VG: ontomark validate --vault all, the source gate, publish.yml:127-135
    VG->>WWW: ontomark build --vault all --out site-data<br/>--publish-out quartz/content --certificate-out, publish.yml:165-170
    VG->>OO: download the pinned oo-cert, check its sha256, publish.yml:218-221
    OO-->>VG: re-derives every certificate line from its premises or exits 1<br/>site-data/certificate/ ships with the site, publish.yml:222-227
    VG->>VG: verify the space and Earth domain contract, then the supply chains<br/>and logistics domain contract, publish.yml:309-313
    VG->>WWW: Quartz v4 renders /notes/ from staged content<br/>(replaces the old Logseq SPA), publish.yml:318-328
    VG->>WWW: place machine artefacts at site root and check each unaltered<br/>publish.yml:330-345
    VG->>WWW: gzip the per-page artefacts at rest, write CNAME<br/>publish.yml:347-367
    VG->>WWW: build WasmVOWL explorer SPA at site root, publish.yml:410-451
    VG->>KG: actions-gh-pages pinned SHA publish.yml:516<br/>external_repository :519, publish_branch :520
    Note over VG,OM: INVARIANT - ontomark is the ONLY producer of both the machine<br/>artefacts and the Quartz content staging tree, pinned twice,<br/>by release tag and by digest, publish.yml:41-51. The pin moved from<br/>a VisionClaw vault binary to the ontomark release in 2026-10<br/>(publish.yml:17-18), so VisionClaw is no longer a build-time<br/>dependency of this pipeline.
    Note over VG,KG: INVARIANT - this is still a CROSS-ORG, CROSS-REPO deploy:<br/>the workflow lives in jjohare/visionGraph and writes into<br/>DreamLab-AI/knowledgeGraph, authenticated by a personal_token<br/>secret (publish.yml:518). knowledgeGraph does not build the<br/>site it serves - it is a publication TARGET.
    VG->>VC: repository_dispatch corpus-sync with source_sha, publish.yml:525-531.<br/>see ES-11.1 edge 9 and ES-09.13
    Note over WWW: DOC-DRIFT resolved - the old gh-pages clone-and-preserve<br/>/notes step is gone. Quartz v4 builds /notes/ fresh every<br/>run from staged content, so there is no cross-deploy<br/>carry-over left to document.
    Note over KG: knowledgeGraph/CNAME:1 is narrativegoldmine.com - the same<br/>domain publish.yml:367 writes, so the CNAME exists on both<br/>sides of the deploy. see KG-01
```

## ES-11.3 Edge 12 - two VOWL explorers, two divergent bundle pins
```mermaid
flowchart LR
    CRATE["EXTERNAL: vowl-wasm<br/>version 0.2.2, vowl-wasm/Cargo.toml:5<br/>see VW-01"]

    subgraph KGE["EXTERNAL: knowledgeGraph explorer - see KG-04"]
        KGP["@dreamlab-ai/vowl-wasm pinned to a<br/>RELEASE TARBALL URL for v0.1.1<br/>explorer/modern/package.json:20"]
    end
    subgraph VGE["EXTERNAL: visionGraph WasmVOWL - see VG-03"]
        VGP["@dreamlab-ai/vowl-wasm pinned to a<br/>RELEASE TARBALL URL for v0.2.2<br/>WasmVOWL/modern/package.json:25"]
    end

    CRATE -->|"published as a GitHub release asset"| KGP
    CRATE -->|"published as a GitHub release asset"| VGP

    D1["DOC-DRIFT - the two consumers are a MINOR version apart, though<br/>both now fetch a release tarball over HTTPS. knowledgeGraph<br/>stays on the frozen v0.1.1 tarball; visionGraph moved to v0.2.2.<br/>A vowl-wasm release therefore reaches one explorer and not the<br/>other. The only version-named check is visionGraph's own<br/>0.2.2 notice file, publish.yml:470, not a comparison of the two."]
    D2["DIVERGENCE - the roster carries BOTH vowl-wasm and WasmVOWL<br/>as separate first-party repositories (roster.json:16,18), and<br/>registers @dreamlab-ai/vowl-wasm on npm alongside the<br/>crates.io crate (roster.json:60,81). Neither explorer resolves<br/>from the npm copy at this revision, both pin release tarballs<br/>of the same bundle; the crate is what estate health<br/>actually measures. see VF-03.1"]

    KGP --> D1
    VGP --> D1
    CRATE --> D2
```

## ES-11.4 Edge 13 - the forum kit pin, and the guard that keeps four files in lockstep
```mermaid
sequenceDiagram
    autonumber
    participant NF as "EXTERNAL: nostr-rust-forum (the kit)"<br/>see NF-01
    participant CI as "EXTERNAL: dreamlab-ai-website rust-ci.yml"<br/>see DW-02
    participant GUARD as "EXTERNAL: kit-pin-guard.yml"<br/>see DW-02

    CI->>NF: git clone the kit, then checkout --detach $KIT_REF<br/>rust-ci.yml:33
    Note over CI: KIT_REF is a literal 40-char SHA pinned in the workflow<br/>env block, rust-ci.yml:20, with the comment at :18<br/>requiring lockstep with workers-deploy.yml and deploy.yml.<br/>On 2 Oct it moved three times, to forum 49904f4 (6ae4576),<br/>341c5d2 (8ab4ab4) and 13cbe6c (81ec18c), and it has kept<br/>moving with the kit since — this revision pins 9c36b60 at website a67cb2a. Forum main (1b27f79) has moved past that kit, and forum production still runs kit 9c36b60.<br/>The kit crates are pinned exactly at 1.0.0-beta.16, forum-config/Cargo.toml:50-53
    NF-->>CI: the kit tree at exactly that commit
    GUARD->>GUARD: compare the pin across deploy.yml, workers-deploy.yml,<br/>rust-ci.yml, forum-config/Cargo.toml crate versions and the<br/>CANONICAL_ entries in the kit-compatibility record
    alt any of the five disagrees
        GUARD--xCI: "::error::Kit pin drift detected" kit-pin-guard.yml:29
    end
    Note over NF,GUARD: INVARIANT - the website consumes the forum kit at BUILD time<br/>by source clone, not as a published package. The pin is<br/>therefore a SHA, and correctness depends on five files<br/>agreeing - kit-pin-guard.yml is the only thing that checks it.
    Note over NF: EXTERNAL - this diagram asserts only what the WEBSITE's<br/>workflows state about the kit. The kit's own contract is<br/>NF's to describe. see NF-01 and DW-02
```

## ES-11.5 Edges 14 and 15 - the roster is the estate's only enumeration, and what reads it
```mermaid
flowchart TB
    ROSTER["scripts/estate-health/roster.json<br/>18 repos, plus surfaces and registries<br/>the _doc at roster.json:3 calls it the ONLY<br/>place the estate is enumerated"]

    subgraph READERS["What walks it"]
        EH["scripts/estate-health.mjs collect<br/>reads the roster and interrogates the GitHub API,<br/>the public surfaces and the package registries<br/>estate-health.mjs:18-20. see VF-03"]
        DC[".github/workflows/drift-counter.yml<br/>checks out ONE sibling at a pinned ref<br/>drift-counter.yml:57-62. see VF-05"]
    end

    subgraph TARGETS["What it reaches"]
        R1["EXTERNAL: VisionClaw · agentbox · solid-pod-rs<br/>nostr-rust-forum · dreamlab-ai-website · sidestr-rs<br/>see VC-01, AB-01, SP-01, NF-01, DW-01, SR-01<br/>sidestr-rs rostered since c001415, roster.json:13"]
        R2["EXTERNAL: knowledgeGraph · visionGraph · vowl-wasm · ontomark<br/>see KG-01, VG-01, VW-01, OM-00"]
        R3["EXTERNAL: loom · WasmVOWL · prose-sanitiser · diagram-ir<br/>ontomark-obsidian - rostered with NO diagram<br/>area in this tree (dream-engine has DE-01, ontomark has OM-00,<br/>visiontouch-rs has an area of its own)"]
        SURF["public surfaces incl. narrativegoldmine.com<br/>and /ns/v2.jsonld, roster.json:30-40"]
        REG["registries - crates.io and npm,<br/>incl. @dreamlab-ai/vowl-wasm and, since c001415,<br/>the eight sidestr-rs crates, roster.json:59-81"]
    end

    ROSTER --> EH
    ROSTER --> DC
    EH --> R1
    EH --> R2
    EH --> R3
    EH --> SURF
    EH --> REG

    INV["INVARIANT roster.json:3 - adding a repository to the estate means<br/>adding a ROW here: the collector holds no repository names, and<br/>the page, the snapshot order and the dream evaluator all follow<br/>roster order. This is the enumeration the tree treats as canonical."]
    D1["DIVERGENCE - drift-counter does NOT walk the roster. It pins ONE<br/>sibling, agentbox, at a literal SHA (drift-counter.yml:61) so<br/>that moving the count source is a reviewed two-line diff rather<br/>than a silent change. The two gates therefore disagree about<br/>what the estate is."]
    D2["DIVERGENCE - ES-01.1 draws SIX repositories, the count<br/>agentbox's ecosystem doc uses. That is the VisionClaw checkout's<br/>own neighbourhood, not the estate: it omits knowledgeGraph,<br/>visionGraph, vowl-wasm and the other roster rows.<br/>ES-01.1 now states its scope explicitly. see VF-08"]

    EXEMPT["2026-10 (ea85a3c) - a repository may declare ci_by_design, a<br/>reason string, and then reads exempt rather than none when it<br/>has no verdict on HEAD: knowledgeGraph is a publish target and<br/>WasmVOWL a frozen shell, roster.json:12,16. A red or amber verdict<br/>on HEAD is never exempted, estate-health.mjs:107-112 and :558-562."]
    EH --> EXEMPT
    ROSTER --> INV
    DC --> D1
    ROSTER --> D2
```
