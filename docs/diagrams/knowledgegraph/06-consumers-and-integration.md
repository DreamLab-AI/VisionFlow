---
id: KG-06
title: Current corpus consumers and the archived RDF import boundary
area: knowledgegraph
governing:
  - ../knowledgeGraph/docs/ecosystem.md
adrs: []
sources:
  - ../knowledgeGraph/docs/ecosystem.md
  - ../knowledgeGraph/docs/integrations/ontocast.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/README.md
  - ../knowledgeGraph/archive/logseq-era-2026-09-22/pipeline/ontocast_import.py
  - ../visionGraph/.github/workflows/publish.yml
  - ../project/src/services/corpus_source/mod.rs
  - ../project/src/services/corpus_source/github.rs
  - ../project/src/services/github_sync_service.rs
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/scripts/hp-loom-reconcile.sh
  - ../loom/crates/loom-facade/src/lib.rs
  - ../loom/crates/loom-facade/src/bundle.rs
verified_commit: {knowledgegraph: 5294e88b6602a63b5afe96c79f2779d00274a05e, visiongraph: 556a0156b5c326fb61141ed2ab1760dd8d9c4dd6, visionclaw: 46edc196b58a46c64fffa5c10e3dd72c0a36238e, agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93, loom: 741c5dffd3253038bbe4fadea7fada7c46474a25}
---

## For developers

Trace the public export, Markdown consumer and Loom generation independently; [the source review and four archive probes](../../estate-review/2026-11-01-knowledgegraph-consumers-and-archive.md) replace the historical ecosystem transcription with bounded implementation evidence.

## For the business

A successful website publication does not prove that a graph consumer has refreshed or that an agent serves the same generation; those outcomes need separate receipts, and the old extraction adapter remains historical code.

## KG-06.1 Publication and generation distribution

```mermaid
flowchart TB
    %% visionGraph/.github/workflows/publish.yml:175
    VG["visionGraph vault<br/>ontomark build in the publisher"]
    %% visionGraph/.github/workflows/publish.yml:526
    PUB["Public website publication<br/>knowledgeGraph gh-pages"]
    %% visionGraph/.github/workflows/publish.yml:536
    DISPATCH["After deployment: corpus-sync event<br/>sent to VisionClaw with source SHA<br/>Request delivery, not ingestion completion"]
    %% agentbox/scripts/hp-loom-reconcile.sh:133
    SNAP["Agentbox reconciliation route<br/>Clean worktree of committed vault HEAD"]
    %% agentbox/scripts/hp-loom-reconcile.sh:149
    BUILD["ontomark build --with-rvdb<br/>Promote and verify generation"]
    %% agentbox/scripts/hp-loom-reconcile.sh:198
    LOOM["Transfer and verifying HP reload<br/>Compare served generation with source SHA"]
    VG --> PUB --> DISPATCH
    VG -.->|"Separate distribution route"| SNAP --> BUILD --> LOOM
    NOTE["DRIFT: the old ontology-bridge route is retired<br/>Manifest describes CLI and Loom HTTP access<br/>agentbox/agentbox.toml:1173"]
    LOOM --- NOTE
```

**What it shows:** Public publication and agent grounding have distinct build, delivery and completion boundaries.
**Why it is this way:** The publisher emits a refresh request, while the agentbox reconciler separately builds a committed snapshot and checks Loom's served generation; neither route establishes the other's success.

## KG-06.2 Two consumers with different input contracts

```mermaid
flowchart TB
    %% src/services/corpus_source/mod.rs:200
    SELECT["VisionClaw CorpusSource selection<br/>Explicit local or github wins<br/>Otherwise nonempty VAULT_ROOT selects local"]
    %% src/services/corpus_source/github.rs:60
    REMOTE["GitHubSource lists Markdown<br/>Trees API, Contents API fallback"]
    %% src/services/github_sync_service.rs:315
    INGEST["GitHubSyncService filters page identities<br/>under configured base paths<br/>Full listing builds the vault index"]
    %% loom/crates/loom-facade/src/lib.rs:72
    BUNDLE["Loom activates local generation data<br/>LoadedBundle before lexical and graph loading"]
    %% loom/crates/loom-facade/src/bundle.rs:117
    VERIFY["Commit marker present: verify artefacts<br/>Absent marker: unattested degraded identity<br/>In-flight promotion: reject"]
    %% loom/crates/loom-facade/src/lib.rs:84
    LOAD["Load lexical index and local Oxigraph data<br/>Unavailable adapters degrade separately"]
    SELECT -->|"github selected"| REMOTE --> INGEST
    SELECT -->|"local selected"| INGEST
    BUNDLE --> VERIFY --> LOAD
    NOTE["DRIFT: historical ecosystem narrative is not<br/>evidence of a current static-Turtle fetch route<br/>ecosystem.md:102"]
    INGEST --- NOTE
```

**What it shows:** VisionClaw ingests pages through its source port; Loom loads its own generation and adapters, with explicit degradation cases.
**Why it is this way:** Current entry points supersede the old documentation-only claim that agent grounding always reads VisionClaw's reasoned graph; this panel does not claim to enumerate every optional RDF importer.

## KG-06.4 Archived OntoCast adapter and its evidence limits

```mermaid
flowchart TB
    %% knowledgeGraph/archive/logseq-era-2026-09-22/README.md:1
    ARCH["Historical Logseq pipeline<br/>No current publication input"]
    INPUT["Turtle input, preview by default<br/>ontocast_import.py:271"]
    SELECT["URI class and individual subjects<br/>ontocast_import.py:126"]
    CAND["Private schema-v3 candidates<br/>Pending review metadata<br/>ontocast_import.py:220,229"]
    EVID["Evidence copies selected subject statements<br/>ontocast_import.py:210<br/>DEBT: blank-node descendants are omitted"]
    IDS["DEBT: fallback label-derived references can<br/>alias a rejected duplicate to a retained entity<br/>ontocast_import.py:119,151"]
    WRITE["--write creates files sequentially<br/>Existing target check then write_text<br/>ontocast_import.py:247"]
    GAP["DEBT: dangling symlink can escape staging<br/>Dual type can cause partial output and failure<br/>ontocast_import.py:142,156,256"]
    ARCH --> INPUT --> SELECT --> CAND --> EVID --> WRITE
    SELECT --- IDS
    WRITE --- GAP
```

**What it shows:** Private defaults and review metadata survive, but evidence completeness, collision identity and filesystem confinement have reproduced limits.
**Why it is this way:** The archived adapter selects subjects and serialises their immediate statements; its checks are narrower than the integration document's losslessness and never-overwrite wording, as the retained synthetic probes demonstrate.
