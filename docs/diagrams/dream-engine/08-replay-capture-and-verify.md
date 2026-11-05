---
id: DE-08
title: Replay capture and verify
area: dream-engine
governing: [../dream-engine/docs/adr/ADR-DE-003-protected-path-guard.md]
adrs: [ADR-DE-003]
sources:
  - ../dream-engine/crates/dream-core/src/replay.rs
  - ../dream-engine/crates/dream-gate/src/gate.rs
  - ../dream-engine/crates/dream-gate/src/receipts.rs
  - ../dream-engine/crates/dream-cli/src/lib.rs
  - ../dream-engine/docs/adr/ADR-DE-003-protected-path-guard.md
verified_commit: 815f963b1cbd6999e78144ce5475005ae1dd1d8e
---
## For developers

Because the gate is pure, a night can be recorded as its inputs and judged
again later. `capture` writes the inputs and the decision under the run
directory, and `verify` re-judges from the stored inputs and diffs the result
against the recorded decision, which it never trusts
(`crates/dream-core/src/replay.rs:6`). `capture` and `verify` share one
function, `decide`, so they cannot disagree about method
(`crates/dream-core/src/replay.rs:305`). The command-line entry is
`replay_verify` (`crates/dream-cli/src/lib.rs:301`).

## For the business

A complete captured bundle can be re-judged and compared with its recorded decision. Missing or inconsistent files may make verification fail; capture itself can leave a partial bundle. It is not proof
against someone deliberately rewriting the record: the module says so itself,
and an auditor who needs that has to sign the run folder some other way.

## DE-08.1 What is captured

```mermaid
flowchart TB
    IN["NightInputs<br/>replay.rs:189"] --> DEC["decide: policy and strict verdict<br/>then gate::decide<br/>replay.rs:305"]
    DEC --> CAP["capture<br/>replay.rs:332"]
    CAP --> MAN["manifest.json frozen<br/>replay.rs:336"]
    MAN --> RC["receipts/baseline and receipts/candidate, cleared first<br/>replay.rs:348"]
    RC --> RP["report.md and config.json<br/>replay.rs:354"]
    RP --> PA["candidate.patch, or removed when none<br/>replay.rs:361"]
    PA --> RJ["replay.json: schema 2<br/>report digest and policy extras<br/>replay.rs:367"]
    RJ --> DJ["decision.json<br/>replay.rs:379"]
```

**What it shows.** The file groups and receipt directories written by `capture`, in sequence rather than as one transaction.
Nothing is written if judging fails (`crates/dream-core/src/replay.rs:333`).
**Why it is this way.** A restarted night may have fewer receipts than the
attempt before, and loading reads every file in a phase directory, so each
phase directory is cleared first to avoid judging a stale receipt
(`crates/dream-core/src/replay.rs:340`). Schema 1 records are refused because
they cannot say which protected-path policy the night used and guessing would
silently weaken it (`crates/dream-core/src/replay.rs:77`).
**Invariant:** `night_policy` reconstructs the same policy from the same manifest, config and extras; agreement with a live run depends on the caller supplying those same inputs
(`crates/dream-core/src/replay.rs:173`). That function now also applies
`PolicyExtras::protected_paths`, the evaluator-named files the host resolved
against the repository and recorded in `replay.json`, so a replay protects
exactly what the live night did; a record from before that field existed
reads it as empty (`crates/dream-core/src/replay.rs:111`).

## DE-08.2 What verify re-checks

```mermaid
sequenceDiagram
    participant V as verify
    participant D as run directory
    participant G as decide
    V->>D: load replay.json and check the schema, replay.rs line 399
    V->>D: report.md digest against replay.json, replay.rs line 406
    V->>D: decision.json as recorded, replay.rs line 413
    V->>G: re-judge from stored inputs, replay.rs line 487
    G-->>V: recomputed decision
    V->>V: compare all decision fields
    V->>V: reclassify receipts, replay.rs line 525
    V->>G: guard check stored patch, replay.rs line 528
    V-->>V: matches only if no diffs and the guard agrees
```

**What it shows.** Three independent checks: the recomputed decision against
the recorded one, each receipt's typed outcome against its own raw
exit status and streams, and the stored patch against the recorded candidate
state (`crates/dream-core/src/replay.rs:485`).
**Why it is this way.** The gate reads the typed outcome, so a receipt whose
exit status was edited after the fact would otherwise pass unnoticed
(`crates/dream-core/src/replay.rs:467`).
**Tension:** replay is not tamper evidence. Only the report and the config
are digest-bound, so editing a receipt's exit status, outcome and streams
together with `decision.json` defeats `verify`
(`crates/dream-core/src/replay.rs:10`). A witness over the run directory is
the stated remedy and lives elsewhere (`crates/dream-core/src/replay.rs:18`).
**Debt:** receipt directories are deleted before replacement and the remaining files are written individually. A failed write can destroy an earlier complete bundle or leave mixed generations (`crates/dream-core/src/replay.rs:340`). Receipt filenames can collide after slugification; capture does not check that every supplied receipt survives (`crates/dream-gate/src/receipts.rs:264`).
**Debt:** replay checks recorded evidence, not whether commands really ran on the named tree. It neither re-executes tests nor authenticates receipt producers (`crates/dream-core/src/replay.rs:485`).
**Open:** nothing in this repository calls `capture` during a night. The host
driver has to, so whether any real night is replayable depends on the host.
