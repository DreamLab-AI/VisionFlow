---
id: DE-07
title: The nightly loop and the run-state machine
area: dream-engine
governing: [../dream-engine/docs/concepts.md]
adrs: [ADR-DE-007]
sources:
  - ../dream-engine/crates/dream-core/src/runstate.rs
  - ../dream-engine/crates/dream-core/src/lib.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-gate/src/manifest.rs
  - ../dream-engine/docs/concepts.md
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---
## For developers

One night is one experiment against one repository. This topic is the
restart-safety half of that: the `Phase` journal that `dream-core` writes beside
the frozen manifest, the lock that keeps two engines off one night, and the
rules that decide, on every start, whether a night is fresh, resumed, already
finished, abandoned or untrustworthy. It describes process-restart behaviour when the host follows the lock and persistence ordering; it does not establish power-loss durability.

The phases are an ordered enum so `>=` comparisons are meaningful
(`crates/dream-core/src/runstate.rs:56`). The ladder the concepts document
describes (`docs/concepts.md:20`) is `ManifestFrozen`, `BaselineEvaluated`,
`ModelCalled`, `CandidateEvaluated`, `Gated`; the code adds `Initialised`
before it and `Persisted`, `Complete` and `Abandoned` after it
(`crates/dream-core/src/runstate.rs:58`). The run id is computed from selected experiment inputs (`crates/dream-gate/src/manifest.rs:258`), so a restart finds
its own journal. But the run id folds in the baseline HEAD, which the engine's
own ledger commit moves, so the terminal rule is night-level, not run-level
(`crates/dream-core/src/runstate.rs:401`, ADR-DE-007). The attempt budget is
night-level for the same reason: a changed run id carries the count forward
(`crates/dream-core/src/runstate.rs:567`).

Since the audit hardening, three things changed shape. A journal is read with
`load_checked`, which separates absent (`Ok(None)`) from unusable, and every
unusable journal is a typed `JournalError` rather than a silent fresh start
(`crates/dream-core/src/runstate.rs:337`). The host is required to hold a `NightLock` across the
whole night (`crates/dream-core/src/runstate.rs:248`). And `reset_night` is a
two-step archive that cannot collide with an earlier archive
(`crates/dream-core/src/runstate.rs:441`).

This crate ships the journal, not the driver that sequences the phases: the
crate docs list `runstate` among the pieces a host drives
(`crates/dream-core/src/lib.rs:21`).

## For the business

The journal limits repeated attempts and recognises finished nights even when the baseline revision changes. Its default budget is two attempts. A cooperating host holds a lock and records the verdict before external delivery; a crash can then lose a publish. Damaged journals return errors for the host to report. Reset archives the old files before removing them, but its rollback is best effort. These controls do not guarantee that every night finishes or every verdict reaches a pull request.

## DE-07.1 Phases of a night

```mermaid
stateDiagram-v2
    [*] --> Initialised
    Initialised --> ManifestFrozen
    ManifestFrozen --> BaselineEvaluated
    BaselineEvaluated --> ModelCalled
    ModelCalled --> CandidateEvaluated
    CandidateEvaluated --> Gated
    Gated --> Persisted
    Persisted --> Complete : normal close or finished_night on restart
    Initialised --> Abandoned : budget spent
    ManifestFrozen --> Abandoned : budget spent
    BaselineEvaluated --> Abandoned : budget spent
    ModelCalled --> Abandoned : budget spent
    CandidateEvaluated --> Abandoned : budget spent
    Gated --> Abandoned : budget spent
    Complete --> [*]
    Abandoned --> [*]
```

**What it shows.** The nine `Phase` values in the intended host sequence. The enum permits jumps: `advance` enforces monotonic order, not adjacent transitions. The host must enter `Persisted` before external delivery, and
`Abandoned` is reachable from any non-terminal phase when `begin` finds the
night's attempt budget spent.
**Why it is this way.** `advance` only moves forward, so a resumed run
re-walking earlier steps cannot rewrite history
(`crates/dream-core/src/runstate.rs:604`). `Persisted` is deliberately entered
first, so a crash can lose a publish but never repeat one
(`crates/dream-core/src/runstate.rs:72`).
**Invariant:** `advance` never lowers the phase (`crates/dream-core/src/runstate.rs:605`).
**Tension:** the at-most-once ordering is a documented caller contract on
`persist` (`crates/dream-core/src/runstate.rs:615`); no code in this repository
sequences it, so nothing here enforces it. Nothing records the forge push or
the ledger commit either, so a crash after `Persisted` loses that night's
publish and the restart finalises the night to `Complete` without
reconciling against the forge; recovery is `reset_night`
(`crates/dream-core/src/runstate.rs:401`).

## DE-07.2 What begin decides on every start

```mermaid
flowchart TB
    S["begin<br/>runstate.rs:511"] --> LC{"load_checked<br/>runstate.rs:523"}
    LC -->|"Io, Corrupt or UnknownSchema"| JE["Err JournalError<br/>file left untouched<br/>runstate.rs:166"]
    LC -->|"journal for another repo, date or night"| WN["Err WrongNight<br/>runstate.rs:526"]
    JE --> BE["host reports BLOCKED-ENV<br/>and points at reset-night"]
    WN --> BE
    LC -->|"Ok, absent or matching"| F{"finished_night<br/>runstate.rs:534"}
    F -->|"Complete or Persisted"| AC["AlreadyComplete<br/>do not re-run<br/>runstate.rs:535"]
    F -->|no| A{"no journal?<br/>runstate.rs:537"}
    A -->|yes| FR["Fresh<br/>phase Initialised, attempts 1<br/>runstate.rs:540"]
    A -->|no| B{"Abandoned, or attempts at the budget?<br/>runstate.rs:542"}
    B -->|yes| AB["Abandoned<br/>record and move on<br/>runstate.rs:547"]
    B -->|no| R{"same run id?<br/>runstate.rs:549"}
    R -->|yes| RS["Resumed<br/>attempts plus 1, resumed_from set<br/>runstate.rs:557"]
    R -->|no| NF["Fresh at Initialised<br/>attempts carried plus 1<br/>runstate.rs:572"]
```

**What it shows.** What `begin` returns or refuses. `Fresh` and `Resumed` say
proceed (`crates/dream-core/src/runstate.rs:154`); `AlreadyComplete` and
`Abandoned` say stop; a `JournalError` says the night cannot be judged at all.
The enum has no corrupt variant: damage is the `Err` side of the `Result`.
**Why it is this way.** A damaged journal must never read as a fresh night,
because that reopens a finished one and resets its attempt budget
(`crates/dream-core/src/runstate.rs:333`). Absence starts attempt one; a changed run id also returns `Fresh` while carrying the count (`crates/dream-core/src/runstate.rs:537`, `crates/dream-core/src/runstate.rs:561`). The budget check
comes before the run-id comparison, so an exhausted night stays abandoned
whatever the run id is, and a different run id (the baseline or config
moved) starts a fresh experiment but still counts the attempt, so a run id
that changes on every failure cannot loop forever
(`crates/dream-core/src/runstate.rs:567`). The updated record is saved
before `begin` returns, so a crash straight after still counts the attempt
(`crates/dream-core/src/runstate.rs:556`). The default budget is two: one
clean try plus one recovery (`crates/dream-core/src/runstate.rs:51`). For a
moved baseline the manifest freeze has already archived the old document
(`crates/dream-core/src/runstate.rs:559`).
**Debt:** the absent-journal branch starts attempt one even if `max_attempts` is zero (`crates/dream-core/src/runstate.rs:537`). Atomic writes rename a temporary file but do not fsync the file or parent directory (`crates/dream-gate/src/manifest.rs:328`).
**Invariant:** a night already `Persisted` or `Complete` is `AlreadyComplete`
whatever the run id says (`crates/dream-core/src/runstate.rs:534`).
**Invariant:** only a missing file is absent; an unreadable, garbled or
unknown-schema journal is an error, never `Ok(None)`
(`crates/dream-core/src/runstate.rs:341`).

## DE-07.3 The night-level terminal rule

```mermaid
sequenceDiagram
    participant D as Driver
    participant J as runstate
    participant G as ledger commit
    D->>J: host persists before delivery
    J->>J: phase Persisted, verdict recorded
    D->>G: ledger commit moves baseline HEAD
    Note over J,G: baseline changed - run id differs
    D->>J: restart calls begin with the new run id
    J->>J: finished_night checks night identity
    J-->>D: AlreadyComplete, best-effort Complete
    D->>J: operator requests reset_night
    J->>J: take the night lock, else WouldBlock
    J->>J: hard link each file to a new archive
    J->>J: then remove the source files
    Note over J: best-effort rollback on handled errors
```

**What it shows.** Why the rule is keyed on the night, not the run id: the
engine's own commit changes the id a restart computes. And how the operator
reopens a night: `reset_night` attempts a two-file archive under the cooperative lock.
**Why it is this way.** ADR-DE-007 records that a night is identified by its
directory (`crates/dream-core/src/runstate.rs:7`). Finalising `Persisted` to
`Complete` is best effort and the answer does not depend on the write landing
(`crates/dream-core/src/runstate.rs:419`). The archive stamp is UTC
nanoseconds plus the process id and a counter (`crates/dream-core/src/runstate.rs:452`),
and each archive is made by `hard_link`, which fails rather than replacing an
existing name (`crates/dream-core/src/runstate.rs:466`). That removes the old
failure where two resets in one clock tick could overwrite an archive. If the
second archive cannot be made deletion of the first is attempted, and if a source cannot be
removed the code attempts to link back the already removed ones
(`crates/dream-core/src/runstate.rs:468`). `reset_night` takes the night lock
and waits at most 500 ms, so it refuses while another cooperating holder owns that lock
(`crates/dream-core/src/runstate.rs:442`).
**Debt:** reset is not a transaction across both files. A crash can interrupt its stages and restoration errors are ignored (`crates/dream-core/src/runstate.rs:468`, `crates/dream-core/src/runstate.rs:478`).
**Open:** the manifest archive on a moved baseline is named by run id
(`crates/dream-gate/src/manifest.rs:297`), the reset archive by timestamp
(`crates/dream-core/src/runstate.rs:452`); two naming schemes for superseded
documents.

## DE-07.4 The night lock and what a crash costs

```mermaid
sequenceDiagram
    participant H as Host driver
    participant K as NightLock
    participant J as runstate journal
    participant X as Forge and ledger
    H->>K: acquire_within before the first journal read
    K-->>H: Held error if another engine owns the night
    H->>J: begin
    H->>J: advance through freeze, baseline, model, candidate, gate
    H->>J: persist verdict, phase Persisted
    H->>X: draft pull request and ledger row
    H->>J: complete
    Note over H,X: crash before persist - retry or abandon at budget
    Note over H,X: crash after persist, restart is AlreadyComplete, publish may be lost
    H->>K: drop releases the lock
```

**What it shows.** The required host sequence holds the lock across check, execution and persistence. Release occurs when all owning descriptors close; an inherited child descriptor can delay release after the parent dies. These process-crash windows assume the journal writes remain readable.
**Why it is this way.** The lock is `flock`-style on `<night_dir>/.lock` and
is per open file description, so a second acquire in the same process
conflicts like one from another process (`crates/dream-core/src/runstate.rs:274`).
`acquire_within` retries for a short while because a descriptor briefly
duplicated into a forked child can make a just released lock look held
(`crates/dream-core/src/runstate.rs:295`). A crash before `persist` costs one
attempt out of two; a crash after it costs the publish
(`crates/dream-core/src/runstate.rs:615`).
**Open:** holding the lock is, like the persist ordering, a contract on the
host. `begin` does not take it and nothing else in this repository calls
`NightLock::acquire`, so an engine that skips it is not stopped here
(`crates/dream-core/src/runstate.rs:261`).
