---
id: CS-02
title: Ingest to model to layout pipeline
area: codescape
governing:
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/docs/adr/ADR-0001-extraction-from-visionclaw.md
adrs: [codescape:ADR-0001, visionclaw:ADR-2143]
sources:
  - ../codescape/docs/BASELINE-codescape.md
  - ../codescape/crates/codescape-ingest/src/lib.rs
  - ../codescape/crates/codescape-ingest/src/watch.rs
  - ../codescape/crates/codescape-model/src/ids.rs
  - ../codescape/crates/codescape-model/src/delta.rs
  - ../codescape/crates/codescape-model/src/lod.rs
  - ../codescape/crates/codescape-layout/src/lib.rs
  - ../codescape/crates/codescape-layout/src/abi.rs
  - ../codescape/crates/codescape-layout/src/ptx_source.rs
  - ../codescape/crates/codescape-layout/src/cuda.rs
  - ../codescape/crates/codescape-layout/src/cpu.rs
  - ../codescape/crates/codescape-server/src/engine.rs
  - ../codescape/crates/codescape-server/src/persist.rs
  - ../codescape/crates/codescape-server/src/ingest_sink.rs
verified_commit: d66bcab80fc0ce1f57deb6a0b6fa74e56ee8e298
---

## CS-02.1 Cold open of one repository

```mermaid
sequenceDiagram
    autonumber
    participant ENG as RepoEngine::open<br/>codescape-server/src/engine.rs:298
    participant ING as Ingestor::open<br/>codescape-ingest/src/lib.rs:231
    participant GR as graphicle extraction
    participant MAP as map::assemble and analyse
    ENG->>ENG: load interner, bump epoch only when re-extract is asked (:272,:282)
    ENG->>ING: IngestConfig, leiden_resolution 1.0, centrality_sample 512 (engine.rs:317,318)
    ING->>ING: refuse an out_dir inside the repository (codescape-ingest/src/lib.rs:254,:269)
    ING->>GR: load cached graph.json if the manifest matches (codescape-ingest/src/lib.rs:286)
    GR-->>ING: otherwise extract_full then store_cached (codescape-ingest/src/lib.rs:291,:293)
    ING->>MAP: assemble snapshot, revision 1, epoch of the interner (codescape-ingest/src/lib.rs:320,:322,:323)
    MAP->>MAP: Leiden communities and betweenness (codescape-ingest/src/lib.rs:379,:383)
    ING->>ING: snapshot.validate before it is kept (codescape-ingest/src/lib.rs:326)
    ING-->>ENG: Ingestor and IngestReport with from_cache (codescape-ingest/src/lib.rs:340)
    Note over ING: INVARIANT: the repository root is read, never written, caches live under out_dir (codescape-ingest/src/lib.rs:13,:14)
    Note over ENG: see VC-39.2 for the host that mounts the roots read-only
```

**What it shows.** One cold start produces a validated revision-1 snapshot, from graphicle's cache when the file manifest is unchanged. Nothing graphicle-typed leaves the ingest crate (`codescape-ingest/src/lib.rs:3,:4` of its crate docs).

## CS-02.2 A file change becomes a delta and a layout command

```mermaid
sequenceDiagram
    autonumber
    participant SRC as Watcher or hook paths<br/>codescape-ingest/src/watch.rs:37
    participant ILP as ingest_loop<br/>codescape-server/src/engine.rs:783
    participant ING as Ingestor::apply_changed<br/>codescape-ingest/src/lib.rs:423
    participant AFT as after_change<br/>codescape-server/src/engine.rs:538
    participant LAY as layout thread
    SRC->>ILP: debounced 250 ms, at most 1 s after the first path (watch.rs:37,:40)
    ILP->>ILP: coalesce paths arriving within 50 ms (engine.rs:796,:798)
    ILP->>ING: apply_paths takes the write lock (engine.rs:518,:520)
    ING->>ING: re-extract only these files, patch resident graph (codescape-ingest/src/lib.rs:451)
    ING->>ING: remap touched nodes, revision plus one (codescape-ingest/src/lib.rs:474,:487)
    ING-->>ILP: Some delta, or None when nothing changed (codescape-ingest/src/lib.rs:489)
    ILP->>AFT: publish snapshot, persist ids, push timeline marker (engine.rs:540,:541,:545)
    AFT->>LAY: LayoutCmd::Delta, and mark analytics dirty (engine.rs:549,:552)
    Note over ING: INVARIANT: communities and importance of existing nodes are kept until recompute (codescape-ingest/src/lib.rs:421)
    Note over SRC: hook paths are filtered to files only, a directory would re-extract a subtree (ingest_sink.rs:112)
```

**What it shows.** Two path sources feed one coalescing loop, so the watcher and the activity hook cannot apply the same file twice in one batch. The delta, not the snapshot, is what reaches the layout thread.

## CS-02.3 Analytics recompute and the settle correction

```mermaid
flowchart TD
    DIRTY["analytics_dirty set by every topology delta<br/>codescape-server/src/engine.rs:549"]
    TICK["recompute at most every 30 s<br/>codescape-server/src/engine.rs:33,810,811"]
    REC["Ingestor::recompute_analytics<br/>codescape-ingest/src/lib.rs:511"]
    UNS{"files live-patched since the last exact build?<br/>codescape-ingest/src/lib.rs:512"}
    SET["settle: re-extract through the per-file cache, store graph.json, remap every node<br/>codescape-ingest/src/lib.rs:542,550,553,565"]
    ANA["Leiden stable-matched against current labels, betweenness<br/>codescape-ingest/src/lib.rs:379,383,387"]
    COMMIT["commit as one delta, revision plus one even if empty<br/>codescape-ingest/src/lib.rs:520,522,531"]
    RELOAD["after_change with topology false sends LayoutCmd::Reload<br/>codescape-server/src/engine.rs:553,554"]
    DIRTY --> TICK --> REC --> UNS
    UNS -->|yes| SET --> ANA
    UNS -->|no| ANA
    ANA --> COMMIT --> RELOAD
    NOTE["a use of another crate module that resolved to an external stub in a single-file patch is corrected only here<br/>codescape-ingest/src/lib.rs:506,507,508,509"]
    SET --- NOTE
    TENS["TENSION: a live patch is approximate for up to 30 s, so cross-file edges can lag the edit<br/>codescape-server/src/engine.rs:33"]
    UNS --- TENS
```

**What it shows.** Live patches are cheap and approximate; the periodic settle makes the graph exact again. Community labels survive because the new labels are matched to the old ones (`codescape-ingest/src/lib.rs:376,377`).

## CS-02.4 Engine selection and the PTX acceptance gate

```mermaid
flowchart TD
    SEL["select_engine params, prefer_cuda<br/>codescape-layout/src/lib.rs:367"]
    NOREQ["cpu: CUDA not requested<br/>codescape-layout/src/lib.rs:368,371"]
    NOFEAT["cpu: built without the cuda feature<br/>codescape-layout/src/lib.rs:387,391"]
    CUDA["CudaEngine::new reads the PTX pair first, then cuInit<br/>codescape-layout/src/cuda.rs:467,472,473"]
    NODEV["no CUDA device visible is an error<br/>codescape-layout/src/cuda.rs:475,476"]
    FALL["any error becomes the CPU engine with the reason<br/>codescape-layout/src/lib.rs:381,383"]
    P1["manifest provenance must be compiled<br/>codescape-layout/src/abi.rs:187,194"]
    P2["content tag of the PTX must equal the manifest rewritten tag<br/>codescape-layout/src/ptx_source.rs:107,108"]
    P3["c_params 212 bytes and kernel arities 30, 13, 24<br/>codescape-layout/src/abi.rs:34,41,42,43,103"]
    OK["engine is cuda, reason names device and tag<br/>codescape-layout/src/lib.rs:378,379"]
    SEL -->|"prefer_cuda false"| NOREQ
    SEL -->|"cuda feature off"| NOFEAT
    SEL -->|"cuda feature on"| CUDA
    CUDA --> P1 --> P2 --> P3
    P3 -->|pass| OK
    CUDA --> NODEV
    P1 -->|fail| FALL
    P2 -->|fail| FALL
    P3 -->|fail| FALL
    NODEV --> FALL
    DEBT["DEBT: the content tag is FNV-1a, a binding to the manifest and not an integrity check, so a hand-written manifest passes<br/>codescape-layout/src/ptx_source.rs:129,130,131,132,133"]
    P2 --- DEBT
    NOTE["see VC-42.4 for a retained pair surviving a failed refresh"]
    FALL --- NOTE
```

**What it shows.** The CPU engine is both the default and the universal fallback, so a missing device, feature or artefact never stops the server. The reason string reaches `/api/health` (`codescape-layout/src/lib.rs:18,19` crate docs).

## CS-02.5 The layout loop between deltas

```mermaid
flowchart TD
    START["select engine, load snapshot with warm positions<br/>codescape-server/src/engine.rs:844,852,853"]
    WARM["warm positions accepted only for the current epoch<br/>codescape-server/src/persist.rs:93,99,100"]
    CMD["drain LayoutCmd, block 250 ms only when settled<br/>codescape-server/src/engine.rs:880,881,882"]
    DEL["Delta: apply_delta, keep surviving positions<br/>codescape-server/src/engine.rs:890,891"]
    RETRY["apply_delta failed: reload with current positions as warm<br/>codescape-server/src/engine.rs:895,896,897"]
    FOCUS["focus mask on dirty ids for 240 steps, others frozen<br/>codescape-server/src/engine.rs:906,908,909,41"]
    RELOAD["Reload: engine.load with current positions<br/>codescape-server/src/engine.rs:913,914,915"]
    STEP["step 1, publish every 16 ms, persist every 30 s<br/>codescape-server/src/engine.rs:937,953,957"]
    SETTLED["settled after 5 consecutive calm steps<br/>codescape-layout/src/lib.rs:353<br/>codescape-layout/src/cpu.rs:151,152"]
    START --> CMD
    WARM --> START
    CMD --> DEL
    DEL -->|error| RETRY
    CMD --> RELOAD
    DEL --> FOCUS
    FOCUS --> STEP
    RELOAD --> STEP
    RETRY --> STEP
    STEP --> SETTLED
    SETTLED --> CMD
    DRIFT["DIVERGENCE: 100k-node targets of 5 ms step and 10 s settle were missed, measured 7.8 ms and 13.7 s<br/>BASELINE-codescape.md:119"]
    STEP --- DRIFT
```

**What it shows.** Edits move only the dirty bodies for a bounded burst, so an agent typing in one file does not re-energise the whole scene. A failed incremental update degrades to a warm reload rather than an error (`codescape-server/src/engine.rs:895`).

## CS-02.6 Ids, epochs and what the wire record carries

```mermaid
flowchart LR
    KEY["graphicle node key<br/>codescape-model/src/ids.rs:110"]
    INT["IdInterner hands out ids in order, remembers them<br/>codescape-model/src/ids.rs:3,6"]
    MASK["wire id keeps the low 26 bits<br/>codescape-model/src/ids.rs:16"]
    EPOCH["re-extract bumps the epoch and starts over<br/>codescape-model/src/ids.rs:7<br/>codescape-model/src/ids.rs:140"]
    TIERS["LOD tiers L0 directory, L1 file, L2 symbol, aggregates get their own ids<br/>codescape-model/src/lod.rs:63,66,105"]
    DELTA["GraphDelta diff and apply keep node order for consumers<br/>codescape-model/src/delta.rs:61,99"]
    PERSIST["ids.bin written atomically after every change<br/>codescape-server/src/engine.rs:512"]
    KEY --> INT --> MASK
    INT --> EPOCH
    EPOCH -->|"epoch mismatch discards warm positions"| PERSIST
    TIERS --> DELTA
    MASK --> PERSIST
    TENS["TENSION: ids are never reused inside an epoch, so a long-lived repository only reclaims id space by a full re-extract<br/>codescape-model/src/ids.rs:21,22"]
    INT --- TENS
    NOTE["see VC-39.4 for how each V3 field is filled"]
    MASK --- NOTE
```

**What it shows.** Interned ids replace hashing because a 200k-symbol graph would collide in 26 bits (`codescape-model/src/ids.rs:5`). The epoch is the only reset, and it also invalidates persisted positions.
