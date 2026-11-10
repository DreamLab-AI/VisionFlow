---
id: VG-04
title: Downstream publication, pod loading and distinct generation identities
area: visiongraph
governing:
  - ../visionGraph/docs/PUBLICATION-contract.md
adrs: [ADR-VG-002, visionclaw:ADR-2106]
sources:
  - ../visionGraph/docs/PUBLICATION-contract.md
  - ../visionGraph/.github/workflows/publish.yml
  - ../project/.github/workflows/ontology-publish.yml
  - ../project/scripts/ontology/pack-pod-resources.py
  - ../project/src/services/ontology_pull.rs
  - ../project/src/services/ontology_generation.rs
  - ../project/src/main.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
verified_commit: {visiongraph: d8277151a1404b456b5c7206187f7e4b50f25be1, visionclaw: 75b72416bac176ca8fde715cac3bd7c77657b3e7, ontomark: 6159bb6089753105ef839fd4aac46c1598a64bd2}
---

## For developers

Distinguish the configured pod-release workflow from its server-side loader and the separate Markdown/Loom consumers in KG-06; the [identity review](../../estate-review/2026-11-01-visiongraph-publication-and-identity.md) reproduces a source-refresh failure in the pod loader.

## For the business

Publishing the website and delivering a refresh request do not establish that the pod has activated the same corpus; source identity, built bytes and loaded generation need separate evidence.

## VG-04.1 Configured pod release path and actual server pull

```mermaid
flowchart TB
    EVENT["After site deploy: corpus-sync<br/>payload source_repo and source_sha"]
    SOURCE["VisionClaw resolves source checkout<br/>passes resolved SHA to build job"]
    BUILD["Build local vault crate<br/>knowledge-only build, certificate and oo-cert"]
    PACK["Pack Turtle, JSON-LD, context and index<br/>record source SHA and VisionClaw build SHA"]
    RELEASE["Rolling ontology-latest release<br/>five content/manifest hashes in SHA256SUMS"]
    PULL["Server boot and periodic pull<br/>verify downloads before generation publication"]
    PUSH["Optional direct PUT job<br/>only when SOLID_POD_URL variable is configured"]
    EVENT --> SOURCE --> BUILD --> PACK --> RELEASE --> PULL
    PACK --> PUSH
    %% Source: visionGraph/.github/workflows/publish.yml:545
    %% Source: project/.github/workflows/ontology-publish.yml:89
    %% Source: project/.github/workflows/ontology-publish.yml:150
    %% Source: project/.github/workflows/ontology-publish.yml:164
    %% Source: project/.github/workflows/ontology-publish.yml:177
    %% Source: scripts/ontology/pack-pod-resources.py:116
    %% Source: project/.github/workflows/ontology-publish.yml:305
    %% Source: project/.github/workflows/ontology-publish.yml:360
    %% Source: src/main.rs:917
    %% Source: src/services/ontology_pull.rs:295
    %% Source: src/services/ontology_pull.rs:389
```

**What it shows:** the workflow still builds VisionClaw's `vault`, while the site uses the released `ontomark`; the pod pull is wired into server startup, but this source map does not prove a running deployment's configuration (`ontology-publish.yml:164`, `publish.yml:48`, `src/main.rs:917`).
**Why it is this way:** the release route supports pull-based delivery; its default rolling URL and the optional push job are distinct delivery paths, and a successful dispatch is not an activation acknowledgement (`ontology_pull.rs:40`, `ontology-publish.yml:360`).

## VG-04.3 Source, bundle, deployment and loaded identity differ

```mermaid
flowchart TB
    SOURCE["Authored source revision"]
    SITE["Site compiler marker<br/>source commit, content and ontology digests<br/>output hashes, scope and vocabulary version"]
    PUBLIC["Distribution commit and served site bytes<br/>post-build transformations require their own receipt"]
    POD["Pod index records sourceSha separately<br/>buildSha comes from VisionClaw GITHUB_SHA"]
    CHECK["Puller compares stored buildSha only<br/>and requires an existing atomic generation"]
    SKIP["DEBT: same build SHA skips new source SHA<br/>before downloading checksum manifest"]
    ACTIVATE["Changed build SHA: verify five files<br/>publish an immutable local generation"]
    SOURCE --> SITE --> PUBLIC
    SOURCE --> POD --> CHECK
    CHECK -->|"same build"| SKIP
    CHECK -->|"different build"| ACTIVATE
    %% Source: docs/PUBLICATION-contract.md:14
    %% Source: crates/ontomark-compile/src/build/generation.rs:90
    %% Source: visionGraph/.github/workflows/publish.yml:359
    %% Source: scripts/ontology/pack-pod-resources.py:116
    %% Source: src/services/ontology_pull.rs:205
    %% Source: src/services/ontology_pull.rs:312
    %% Source: src/services/ontology_pull.rs:320
    %% Source: src/services/ontology_pull.rs:337
    %% Source: src/services/ontology_pull.rs:378
    %% Source: src/services/ontology_generation.rs:127
```

**What it shows:** the source and build axes are both recorded but only one controls freshness; the current-source tests keep source A active when source B arrives with the same build SHA (`pack-pod-resources.py:117`, `ontology_pull.rs:312`).
**Why it is this way:** the shortcut avoids redundant downloads, but an unchanged consumer-code commit does not establish unchanged corpus bytes; generation publication and this freshness decision are separate mechanisms (`ontology_pull.rs:312`, `ontology_pull.rs:378`).
