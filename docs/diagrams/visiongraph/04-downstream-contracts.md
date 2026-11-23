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
  - ../project/scripts/ontology/build-local-release.sh
  - ../project/docker-compose.unified.yml
  - ../project/src/services/ontology_pull.rs
  - ../project/src/services/ontology_generation.rs
  - ../project/src/main.rs
  - ../ontomark/crates/ontomark-compile/src/build/generation.rs
verified_commit: {visiongraph: 08390b1fc7f7c509eae477b933731ee33d152480, visionclaw: 4900ec60bc1ba9075343fe848c62e64434126c7e, ontomark: f5cca7d16e56fca95c38fab5e46a3ef0fac065a2}
---

## For developers

Distinguish the configured pod-release workflow from its server-side loader and the separate Markdown/Loom consumers in KG-06; the [identity review](../../estate-review/2026-11-01-visiongraph-publication-and-identity.md) reproduces a source-refresh failure in the pod loader.

## For the business

Publishing the website and delivering a refresh request do not establish that the pod has activated the same corpus; source identity, built bytes and loaded generation need separate evidence.

## VG-04.1 Configured pod release paths and actual server pull

```mermaid
flowchart TB
    EVENT["After site deploy: visionGraph still dispatches corpus-sync<br/>payload source_repo and source_sha"]
    PARK["VisionClaw ontology-publish.yml parked 2026-11-16<br/>workflow_dispatch is the only trigger"]
    LOCAL["Default route: build-local-release.sh in agentbox<br/>committed visionGraph HEAD in a detached worktree"]
    BUILD["Build local vault crate<br/>knowledge-only build, certificate and oo-cert"]
    PACK["Pack Turtle, JSON-LD, context and index<br/>record source SHA and a build identity"]
    RELEASE["Rolling ontology-latest release, dispatch only<br/>five content/manifest hashes in SHA256SUMS"]
    GEN["Local release directory: gen-identity plus current symlink<br/>same five files and SHA256SUMS"]
    PULL["Server boot and periodic pull<br/>file:// or https fetcher by URL scheme<br/>verify downloads before generation publication"]
    PUSH["Optional direct PUT job<br/>only when SOLID_POD_URL variable is configured"]
    EVENT -.->|"no repository_dispatch listener"| PARK
    PARK --> BUILD --> PACK --> RELEASE --> PULL
    LOCAL --> BUILD
    PACK --> GEN --> PULL
    PACK --> PUSH
    TENSION["TENSION: visionGraph publish.yml:554-560 sends corpus-sync<br/>VisionClaw ontology-publish.yml:3-9 no longer accepts it"]
    EVENT --- TENSION
    NOTE["DEBT: triggers reduced to workflow_dispatch<br/>pull_request, corpus-sync and client_payload references remain<br/>ontology-publish.yml:21-26,29,93,292,363"]
    PARK --- NOTE
    DEF["DRIFT: DEFAULT_RELEASE_URL doc says ontology-publish.yml keeps it current on main<br/>that workflow is now dispatch only, and compose supplies file://<br/>ontology_pull.rs:48-50 vs ontology-publish.yml:3-9"]
    PULL --- DEF
    %% Source: project/.github/workflows/ontology-publish.yml:3-9
    %% Source: project/.github/workflows/ontology-publish.yml:26
    %% Source: project/.github/workflows/ontology-publish.yml:292
    %% Source: visionGraph/.github/workflows/publish.yml:554
    %% Source: project/.github/workflows/ontology-publish.yml:89
    %% Source: project/.github/workflows/ontology-publish.yml:165
    %% Source: project/.github/workflows/ontology-publish.yml:178
    %% Source: scripts/ontology/pack-pod-resources.py:116
    %% Source: project/.github/workflows/ontology-publish.yml:307
    %% Source: project/.github/workflows/ontology-publish.yml:363
    %% Source: scripts/ontology/build-local-release.sh:60
    %% Source: scripts/ontology/build-local-release.sh:62
    %% Source: scripts/ontology/build-local-release.sh:66
    %% Source: scripts/ontology/build-local-release.sh:79
    %% Source: scripts/ontology/build-local-release.sh:111
    %% Source: scripts/ontology/build-local-release.sh:116-118
    %% Source: docker-compose.unified.yml:141
    %% Source: src/main.rs:911
    %% Source: src/services/ontology_pull.rs:184-189
    %% Source: src/services/ontology_pull.rs:346
    %% Source: src/services/ontology_pull.rs:446
```

**What it shows:** both routes still build VisionClaw's `vault`, while the site uses the released `ontomark`; the compose default points the boot pull at the local release directory, and this source map does not prove a running deployment's configuration (`build-local-release.sh:62`, `ontology-publish.yml:165`, `publish.yml:48`, `docker-compose.unified.yml:141`, `src/main.rs:911`).
**Why it is this way:** the GitHub release route was parked as network-expensive and replaced by a local twin with the same gates; the rolling URL, the local directory and the optional push job are distinct delivery paths, and a successful dispatch is not an activation acknowledgement (`build-local-release.sh:4-9`, `ontology_pull.rs:48`, `ontology-publish.yml:363`).

## VG-04.3 Source, bundle, deployment and loaded identity differ

```mermaid
flowchart TB
    SOURCE["Authored source revision"]
    SITE["Site compiler marker<br/>source commit, content and ontology digests<br/>output hashes, scope, vocabulary version<br/>and compiler identity (built_by, absent in legacy bundles)"]
    PUBLIC["Distribution commit and served site bytes<br/>post-build transformations require their own receipt"]
    POD["Pod index records sourceSha separately<br/>buildSha comes from GITHUB_SHA: the workflow commit,<br/>or the local identity of source, vault version and binary hash"]
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
    %% Source: scripts/ontology/build-local-release.sh:84
    %% Source: src/services/ontology_pull.rs:250
    %% Source: src/services/ontology_pull.rs:368-370
    %% Source: src/services/ontology_pull.rs:869
    %% Source: src/services/ontology_pull.rs:377
    %% Source: src/services/ontology_pull.rs:394
    %% Source: src/services/ontology_pull.rs:435
    %% Source: src/services/ontology_generation.rs:127
```

**What it shows:** the source and build axes are both recorded and the puller compares both; a manifest with the same build SHA but a different source SHA is not treated as current (`pack-pod-resources.py:117`, `ontology_pull.rs:368-370`, test `ontology_pull.rs:869`).
**Why it is this way:** the shortcut now skips a download only when build and source SHAs both match, so an unchanged consumer-code commit no longer hides new corpus bytes; generation publication and this freshness decision remain separate mechanisms (`ontology_pull.rs:368-370`, `ontology_pull.rs:435`).
