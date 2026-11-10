---
id: DE-03
title: Night recovery and requested-source accounting
area: dream-engine
governing:
  - ../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md
  - ../dream-engine/docs/adr/ADR-DE-008-requested-sources-are-honoured-first.md
adrs: [dream-engine:ADR-DE-007, dream-engine:ADR-DE-008]
sources:
  - ../dream-engine/crates/dream-core/src/runstate.rs
  - ../dream-engine/crates/dream-core/src/source.rs
  - ../dream-engine/crates/dream-gate/src/manifest.rs
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---

## For developers

Night recovery and source requests both carry state across attempts. The journal distinguishes a finished experiment from one that may run again; the source record describes what the next model call was shown. The [dated audit](../../estate-review/2026-11-01-dream-recovery-and-source-supply.md) tests process-crash recovery and counterexamples to source completion and budget claims.

## For the business

A stopped job can resume without automatically repeating publication after the persistence boundary. That trades possible missing drafts for fewer duplicate attempts, subject to retained storage and the host following the lock contract. A source request marked met does not yet prove that its requested lines reached the model, which can create avoidable operator alerts and unsuccessful nights.

## DE-03.1 Recovery depends on retained state

```mermaid
flowchart TB
    LOCK["Caller holds night lock<br/>runstate.rs:261"] --> LOAD["Read and validate journal schema<br/>runstate.rs:337"]
    LOAD --> BEGIN["Check night identity and attempt budget<br/>runstate.rs:523<br/>runstate.rs:542"]
    BEGIN --> PERSIST["Record verdict at Persisted before forge and ledger<br/>runstate.rs:615"]
    PERSIST --> SKIP["Restart finalises matching Persisted record<br/>runstate.rs:415"]
    PERSIST --> SAVE["Temporary file and rename<br/>no file or directory sync<br/>manifest.rs:328"]
    SKIP --> RESET["Explicit reset takes lock and links archives<br/>runstate.rs:441<br/>runstate.rs:466"]
    RESET --> UNLINK["Remove source names separately<br/>rollback errors are ignored<br/>runstate.rs:477"]
```

**What it shows.** A real fixture process was killed after ModelCalled and after Persisted. The lock released in both cases; the former resumed at attempt two and the latter returned AlreadyComplete despite a different run ID. This is process-death evidence, not a power-loss test.

**Why it is this way.** ADR-DE-007 makes completion and attempt limits belong to the night. Reset preserves archives on successful completion, but its multi-file sequence and ignored rollback failures do not make it a crash-atomic transaction. The separate manifest-freeze archive issue remains documented in DE-02. **Open (storage fault model):** temporary-file replacement is not evidence of durable storage after power loss; at-most-once publication requires the persisted journal to survive.

## DE-03.2 Requested paths pass through several different filters

```mermaid
flowchart TB
    TREE["Read tracked paths using NUL-separated ls-tree output<br/>source.rs:117<br/>source.rs:133"] --> EXPAND["Expand requested directories<br/>source.rs:899"]
    EXPAND --> CAP["Reserve half the slots<br/>count entries before validation<br/>source.rs:1099"]
    CAP --> PLAN["Tell planner which paths are selected<br/>source.rs:1121"]
    PLAN --> MERGE["Merge requested, planned, mentioned paths<br/>deduplicate by path<br/>source.rs:494"]
    MERGE --> READ["Read baseline blobs<br/>source.rs:1162"]
    READ --> RENDER["Render and label file outcomes<br/>source.rs:803"]
```

**What it shows.** Invalid or duplicate requests consume positions in the reservation before usable files are selected. Git's quoted non-ASCII paths used to be retained as quoted text by the line parser, so a tracked file could be missed; RESOLVED at dream-engine b40d507, the tracked listing is now NUL-separated and returns such paths unescaped (`../dream-engine/crates/dream-core/src/source.rs:117`, `../dream-engine/crates/dream-core/src/source.rs:124`).

**Why it is this way.** ADR-DE-008 prioritises unmet requests while retaining room for planner choices. Its current implementation reserves input positions rather than a count of successfully supplied files. Directory expansion and path deduplication also lose the identity of individual range requests.

## DE-03.3 A completion marker can outstate the supplied content

```mermaid
flowchart TB
    RANGE["Slice requested lines<br/>past EOF returns empty text<br/>source.rs:650"] --> CLIP["Clip payload to allowance<br/>then append elision marker<br/>source.rs:661"]
    CLIP --> REDACT["Apply caller redactor after clipping<br/>source.rs:859"]
    REDACT --> COUNT["Count final body bytes and mark inlined<br/>source.rs:873"]
    COUNT --> HONOUR["Inlined means requested_honoured<br/>source.rs:1173"]
    HONOUR --> RESULT["No returned record when nothing is inlined<br/>source.rs:1192"]
```

**What it shows.** A request beyond EOF is honoured with zero bytes and a reversed displayed line interval. Clipping markers make body usage exceed both the requested half-budget and the overall budget. At default settings the probe records 30,134 requested bytes and 60,134 total body bytes, before section headers.

**Why it is this way.** File status is based on completing the rendering branch rather than validating meaningful range coverage. The caller needs refusal records even when every file fails; the current Option return loses those details. AB-23.21 shows how the adapter settles requests from these markers.

**Drift (ADR-DE-008 delivery):** the half-byte allocation and complete honoured/refused accounting are weaker than the decision describes. Empty ranges, duplicate ranges and all-refused reads are reproduced in the audit.
