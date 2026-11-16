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
verified_commit: {visiongraph: 1df8eb49051b6e55f35ea493b3c42e79ffd77bb8, visionclaw: d2780e26f870ce21fa92bccb29fa47f001015b16, ontomark: fbe30244c6842879297883b571965f5bc1ce0441}
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
    NOTE["DEBT: pull_request trigger and PR-preview job removed 2026-11-15<br/>conditions testing for a pull request remain<br/>ontology-publish.yml:12-13,33,299,370"]
    BUILD --- NOTE
    %% Source: project/.github/workflows/ontology-publish.yml:12
    %% Source: project/.github/workflows/ontology-publish.yml:33
    %% Source: project/.github/workflows/ontology-publish.yml:299
    %% Source: visionGraph/.github/workflows/publish.yml:554
    %% Source: project/.github/workflows/ontology-publish.yml:96
    %% Source: project/.github/workflows/ontology-publish.yml:158
    %% Source: project/.github/workflows/ontology-publish.yml:172
    %% Source: project/.github/workflows/ontology-publish.yml:185
    %% Source: scripts/ontology/pack-pod-resources.py:116
    %% Source: project/.github/workflows/ontology-publish.yml:314
    %% Source: project/.github/workflows/ontology-publish.yml:370
    %% Source: src/main.rs:927
    %% Source: src/services/ontology_pull.rs:303
    %% Source: src/services/ontology_pull.rs:403
```

**What it shows:** the workflow still builds VisionClaw's `vault`, while the site uses the released `ontomark`; the pod pull is wired into server startup, but this source map does not prove a running deployment's configuration (`ontology-publish.yml:172`, `publish.yml:48`, `src/main.rs:927`).
**Why it is this way:** the release route supports pull-based delivery; its default rolling URL and the optional push job are distinct delivery paths, and a successful dispatch is not an activation acknowledgement (`ontology_pull.rs:41`, `ontology-publish.yml:370`).

## VG-04.3 Source, bundle, deployment and loaded identity differ

```mermaid
flowchart TB
    SOURCE["Authored source revision"]
    SITE["Site compiler marker<br/>source commit, content and ontology digests<br/>output hashes, scope, vocabulary version<br/>and compiler identity (built_by, absent in legacy bundles)"]
    PUBLIC["Distribution commit and served site bytes<br/>post-build transformations require their own receipt"]
    POD["Pod index records sourceSha separately<br/>buildSha comes from VisionClaw GITHUB_SHA"]
    CHECK["Puller compares stored buildSha and sourceSha<br/>and requires an existing atomic generation"]
    SKIP["DEBT: RESOLVED at visionclaw 46edc196 — a same build SHA with a new source SHA<br/>no longer skips, both must match before the checksum manifest is skipped"]
    ACTIVATE["Changed build SHA: verify five files<br/>publish an immutable local generation"]
    SOURCE --> SITE --> PUBLIC
    SOURCE --> POD --> CHECK
    CHECK -->|"same build and source"| SKIP
    CHECK -->|"different build or source"| ACTIVATE
    %% Source: docs/PUBLICATION-contract.md:14
    %% Source: crates/ontomark-compile/src/build/generation.rs:92
    %% Source: crates/ontomark-compile/src/build/generation.rs:95
    %% Source: visionGraph/.github/workflows/publish.yml:370
    %% Source: scripts/ontology/pack-pod-resources.py:116
    %% Source: src/services/ontology_pull.rs:207
    %% Source: src/services/ontology_pull.rs:325-327
    %% Source: src/services/ontology_pull.rs:752
    %% Source: src/services/ontology_pull.rs:334
    %% Source: src/services/ontology_pull.rs:351
    %% Source: src/services/ontology_pull.rs:392
    %% Source: src/services/ontology_generation.rs:127
```

**What it shows:** the source and build axes are both recorded and the puller compares both; a manifest with the same build SHA but a different source SHA is not treated as current (`pack-pod-resources.py:117`, `ontology_pull.rs:325-327`, test `ontology_pull.rs:752`).
**Why it is this way:** the shortcut now skips a download only when build and source SHAs both match, so an unchanged consumer-code commit no longer hides new corpus bytes; generation publication and this freshness decision remain separate mechanisms (`ontology_pull.rs:325-327`, `ontology_pull.rs:392`).
