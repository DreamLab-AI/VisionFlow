---
id: LM-05
title: Generation lifecycle — nightly build, promotion, the reload tick, start-up fallback and the semantic artefact
area: loom
governing:
  - ../loom/docs/published-generation-reload.md
  - ../loom/docs/design/ADR-141-loom-consumes-the-vault-build.md
  - ../loom/docs/design/ADR-142-consume-versioned-shared-ontology-contracts.md
adrs: [loom:ADR-135, loom:ADR-136, loom:ADR-137, loom:ADR-141, loom:ADR-142]
sources:
  - ../loom/deploy/loom-nightly-promotion.sh
  - ../loom/deploy/loom-nightly-promotion.timer
  - ../loom/deploy/loom-nightly-promotion.service
  - ../loom/deploy/loom-generation-reload.timer
  - ../loom/deploy/loom-generation-reload.service
  - ../loom/deploy/ontomark-release.env
  - ../loom/crates/loom-facade/src/promote.rs
  - ../loom/crates/loom-facade/src/reload.rs
  - ../loom/crates/loom-facade/src/fallback.rs
  - ../loom/crates/loom-facade/src/bundle.rs
  - ../loom/crates/loom-facade/src/mirror.rs
  - ../loom/crates/loom-facade/src/bin/reload_published_generation.rs
  - ../loom/crates/loom-vector-ruvector/src/artifact.rs
  - ../loom/crates/loom-vector-ruvector/src/hnsw.rs
  - ../loom/crates/loom-vector-ruvector/src/lib.rs
  - ../loom/crates/loom-vector-ruvector/src/bin/export_corpus.rs
  - ../loom/crates/loom-vector-ruvector/src/bin/stage_corpus.rs
  - ../loom/crates/loom-scaffold/src/bin/build_concept_records.rs
  - ../loom/crates/loom-embed-xinference/src/lib.rs
  - ../loom/app/mirror.sh
  - ../loom/README.md
verified_commit: {loom: 355946dd2deacfdef21171d53eafd025887e447e}
---

## LM-05.1 Two timers, one direction: build and install nightly, restart on the reload tick

```mermaid
sequenceDiagram
    autonumber
    participant NT as nightly timer<br/>loom-nightly-promotion.timer:5
    participant NS as loom-nightly-promotion.sh<br/>loom-nightly-promotion.sh:32
    participant VG as visionGraph checkout
    participant OM as pinned ontomark binary
    participant RT as reload timer<br/>loom-generation-reload.timer:5
    participant RB as reload_published_generation<br/>reload_published_generation.rs:185
    NT->>NS: daily 03:15 with up to 10 min jitter, loom-nightly-promotion.timer:5,6
    NS->>NS: single flock, loom-nightly-promotion.sh:60,61
    NS->>VG: fetch, resolve origin/HEAD to a sha, id visionGraph at sha, loom-nightly-promotion.sh:64,65,67
    NS->>RB: verify --data-dir, exit 0 as current if it already serves that id, loom-nightly-promotion.sh:70,71,72,73
    NS->>OM: install release by tag and check the pinned sha256, loom-nightly-promotion.sh:55,80,93
    NS->>OM: build all vaults from a detached clean clone with --with-rvdb, loom-nightly-promotion.sh:103,104,131,132,138
    NS->>NS: promote_vault_build on the build output, loom-nightly-promotion.sh:140
    NS->>NS: rename to bundles by sha only when complete, loom-nightly-promotion.sh:143,145,148
    NS->>NS: promote_generation into the data dir with a promotions.jsonl line, loom-nightly-promotion.sh:151,152,153,155
    NS->>NS: keep the newest LOOM_KEEP_BUNDLES, default 5, loom-nightly-promotion.sh:39,160,161
    Note over NS: INVARIANT: the nightly never restarts the service, loom-nightly-promotion.sh:7,8
    RT->>RB: two minutes after boot, then every five minutes, loom-generation-reload.timer:5,6
    RB->>RB: reconcile the data dir against the served identity, see LM-05.3
    Note over RB: the restart command is compose up --force-recreate for Profile A only, loom-generation-reload.service:6
```

**What it shows.** Installation and activation are decoupled: the nightly can only put a validated bundle on disk, and only the reload tick restarts the container. The release pin is a tag plus digest (`../loom/deploy/ontomark-release.env:3,5`).

## LM-05.2 promote_generation: lock, validate, compare, install, log

```mermaid
flowchart TB
    LOCK["try_lock beside the promotion log, a second promotion is refused<br/>loom-facade/src/promote.rs:149,156,157"]
    CAND["Candidate read: strict six-artefact check, ontomark marker naming a hex commit,<br/>commit present in the source repo, indexes and both Turtle files parse<br/>loom-facade/src/promote.rs:285,286,299,302,308,315"]
    PARSE["parse_check: scaffold index has classes, prose parses, two TTL files load<br/>loom-facade/src/promote.rs:373,378,383,386"]
    SAME{"same generation installed?<br/>loom-facade/src/promote.rs:164"}
    CUR["still verifies: Current, no-op<br/>loom-facade/src/promote.rs:165,166"]
    OLDER{"candidate a strict ancestor, or<br/>ancestry unknown?<br/>loom-facade/src/promote.rs:171,177,183"}
    REFUSE["refuse unless --allow-rollback<br/>loom-facade/src/promote.rs:178,184"]
    INSTALL["BundlePromoter: sentinel, swap artefacts, marker last<br/>with promoted_at, promoted_by, previous_generation<br/>loom-facade/src/promote.rs:197,205,206"]
    LOG["append and fsync one PromotionRecord line<br/>loom-facade/src/promote.rs:209,224"]
    LOCK --> CAND
    CAND --> PARSE
    PARSE --> SAME
    SAME -->|yes, still verifies| CUR
    SAME -->|yes, no longer verifies| INSTALL
    SAME -->|no| OLDER
    OLDER -->|yes, without allow-rollback| REFUSE
    OLDER -->|newer, or allow-rollback set| INSTALL
    INSTALL --> LOG
    REPAIR["a same-id copy that fails verification is reinstalled without the ancestry check<br/>loom-facade/src/promote.rs:164,165,168,169,170"]
    SAME --- REPAIR
    SWAP["sentinel first, marker rename is the commit, sentinel cleared after<br/>loom-facade/src/bundle.rs:361,364,410,419,425"]
    INSTALL --- SWAP
    INV["INVARIANT: any failure before install leaves the installed generation untouched<br/>loom-facade/src/promote.rs:22"]
    REFUSE --- INV
```

**What it shows.** A generation id does not order, so "older" means git ancestry in the source repository; when ancestry cannot be shown the step refuses rather than guesses (`loom-facade/src/promote.rs:28,29,30`).

## LM-05.3 The reload tick as a state machine

```mermaid
stateDiagram-v2
    [*] --> Verify
    Verify --> Refused: in-flight sentinel, symlinked marker, missing artefact, hash or size mismatch, sidecar mismatch
    Verify --> Probe: bundle verified, reload.rs line 417
    Probe --> Served: report proves the bundle and the disk is unchanged, reload.rs lines 418 to 426
    Probe --> Pending: no match, receipt reload-pending, reload.rs line 428
    Pending --> Polling: restart command ran, reload.rs line 429
    Pending --> Failed: restart command failed or timed out, reload.rs line 429
    Polling --> Served: proven before the deadline, reload.rs lines 437 to 442
    Polling --> Refused: publication changed during reload, reload.rs lines 433 to 435
    Polling --> Failed: deadline passed, Timeout or Fallback, reload.rs lines 446 to 447
    Served --> [*]
    Refused --> [*]
    Failed --> [*]
    note right of Probe
        served_matches requires generation, digest, atomicity,
        disk equals loaded, no drift and a bge-small 384 cosine
        qualification, loom-facade/src/reload.rs:286,287,288,289
        A start-up fallback never matches, reload.rs:297
    end note
    note right of Failed
        Failed attempts are never recorded as served,
        so the next tick retries, loom-facade/src/reload.rs:406
    end note
```

**What it shows.** The receipt says `served` only after the restarted process reports the exact bundle, and the disk is re-verified around every probe. The Rust binary replaces the Python reloader byte-for-byte on the receipt (`loom-facade/src/reload.rs:7,8`).

## LM-05.4 Start-up fallback: walk the promotion log newest first

```mermaid
flowchart TB
    SEL["fallback::select<br/>loom-facade/src/fallback.rs:63"]
    NOROOT["LOOM_RETAINED_BUNDLES_DIR unset or no promotions.jsonl: no fallback<br/>loom-facade/src/fallback.rs:64,65"]
    WALK["read records newest first, skip the failed id and repeats<br/>loom-facade/src/fallback.rs:69,72"]
    HEX["commit must be a hex sha to name a directory<br/>loom-facade/src/fallback.rs:77"]
    VAL["validate_bundle on retained commit dir, without the repo check<br/>loom-facade/src/fallback.rs:81,82"]
    PICK["serve it only if it is the logged generation<br/>loom-facade/src/fallback.rs:83,84"]
    SEL --> NOROOT
    SEL --> WALK --> HEX --> VAL --> PICK
    REB["rebased config: index and prose paths into the retained dir, the rvdb copied<br/>to a writable fallback dir or left absent<br/>loom-facade/src/fallback.rs:117,125,126,127"]
    PICK --> REB
    INV["INVARIANT: the fallback never writes the failed bundle or the retained one<br/>loom-facade/src/fallback.rs:13,14"]
    VAL --- INV
    GAP["OPEN: the source-repository check cannot run in the service, the log line<br/>written after it passed is its stand-in<br/>loom-facade/src/fallback.rs:20,21,22"]
    VAL --- GAP
    SEE["see LM-02.2 for where select is called"]
    SEL --- SEE
```

**What it shows.** Fallback trusts only bundles the promotion log names and that still validate, and it reports itself on every serving surface. Opening the semantic artefact mutates it, so the retained copy is never opened in place (`loom-facade/src/fallback.rs:108,109,110`).

## LM-05.5 What counts as a complete bundle: two marker shapes, six artefacts

```mermaid
flowchart TB
    subgraph SHAPES["two .generation.json shapes, one reader, loom-facade/src/mirror.rs:21,23"]
        C3["ontomark build: id visionGraph at sha, commit, content_digest, artifacts list<br/>loom-facade/src/mirror.rs:27"]
        LEG["legacy mirror: ISO generation stamp, artifacts map<br/>loom-facade/src/mirror.rs:28"]
    end
    STRICT["parse_marker_strict, the reload and promotion path<br/>loom-facade/src/mirror.rs:411"]
    REQ["REQUIRED six: two indexes, two Turtle files, the rvdb and its sidecar<br/>loom-facade/src/reload.rs:36,37,38,39,40,41,42"]
    SIDE["sidecar must name the generation, bge-small-en-v1.5 and 384 dims<br/>loom-facade/src/reload.rs:267,274,275,276"]
    SHAPES --> STRICT --> REQ --> SIDE
    MIR["app/mirror.sh: four artefacts fetched over HTTPS into app/data<br/>../loom/app/mirror.sh:28,29,35"]
    LEG --- MIR
    TENS["TENSION: the retained mirror writes four files to a directory neither profile<br/>mounts, the strict path refuses a four-file set<br/>../loom/README.md:38<br/>loom-facade/src/reload.rs:33,34"]
    MIR --- TENS
    READ["the reader keeps the legacy shape so a node still on a mirror marker stays up<br/>loom-facade/src/mirror.rs:30,31,32,33,34"]
    LEG --- READ
```

**What it shows.** The tolerant reader serves either shape; anything that promotes or reloads uses the strict reader and the six-artefact list. The README records the first served ontomark bundle, on 12 October 2026 (`../loom/README.md:158`).

## LM-05.6 The semantic artefact: two producers, one writer, a contract at open

```mermaid
flowchart TB
    subgraph PROD["producers"]
        PVB["promote_vault_build: ontomark records JSONL into the rvdb, in place<br/>loom-vector-ruvector/src/artifact.rs:9,10,165"]
        EXP["export_corpus, pg-write: read verified vectors from ruvector-postgres<br/>loom-vector-ruvector/src/artifact.rs:8<br/>loom-vector-ruvector/src/bin/export_corpus.rs:3,4,12"]
    end
    W["write_serving_artifact: index law m 16, ef_construction 128, 384 dims<br/>loom-vector-ruvector/src/artifact.rs:67,95,96,64"]
    PROD --> W
    subgraph STAGE["off-turn staging, pg-write"]
        BCR["build-concept-records: one JSONL record per class, pure<br/>loom-scaffold/src/bin/build_concept_records.rs:1,4,9"]
        STG["stage_corpus: embed and emit idempotent upsert SQL<br/>loom-vector-ruvector/src/bin/stage_corpus.rs:1,2,12,24"]
        BCR --> STG
    end
    STAGE --> EXP
    OPEN["HnswIndex open_with_contract reads stored options back<br/>loom-vector-ruvector/src/hnsw.rs:11,12,13,134"]
    W --> OPEN
    REJ["wrong metric or width rejected, model id from the sidecar required<br/>unless LOOM_SEMANTIC_REQUIRE_MODEL_ID=0<br/>loom-vector-ruvector/src/hnsw.rs:16,18,19,21,22"]
    OPEN --> REJ
    LOCK["INVARIANT: embedder locked to bge-small-en-v1.5 and 384, a wrong length is an error<br/>loom-embed-xinference/src/lib.rs:4,5,6,7,32,35"]
    W --- LOCK
    PROC["INVARIANT: promote in its own process, ruvector-core pools open databases<br/>for the life of the process<br/>loom-vector-ruvector/src/artifact.rs:20,21,22,23"]
    PVB --- PROC
    MUT["the open mutates the redb file, the entrypoint copies it first, see LM-06.2<br/>loom-facade/src/fallback.rs:108,109"]
    OPEN --- MUT
```

**What it shows.** Readiness means the artefact passed its contract, not merely that it opened; the fallback stays off in both profiles until the recall bench clears. See ES-12.1 for what that bench has measured.
