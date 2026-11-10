---
id: DE-04
title: Host seam contracts and publication policy
area: dream-engine
governing:
  - ../dream-engine/docs/adr/ADR-DE-005-seams.md
  - ../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md
adrs: [dream-engine:ADR-DE-005, dream-engine:ADR-DE-006]
sources:
  - ../dream-engine/crates/dream-core/src/seams.rs
  - ../dream-engine/crates/dream-core/src/runner.rs
  - ../dream-engine/crates/dream-core/src/automerge.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-gate/src/receipts.rs
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---

## For developers

Five interfaces let a host supply models, execution, memory, notifications and publication. Their failure contracts differ: three use futures, two are synchronous, and a publication outcome is not an error-bearing Result. The [dated audit](../../estate-review/2026-11-01-dream-forge-and-seam-contracts.md) distinguishes these APIs from the consumer's orchestration and verifies the GhForge command boundary offline.

## For the business

An accepted experiment, a pushed branch, an opened draft and an enabled merge are distinct outcomes. Default behaviour requests a draft for human review. The optional merge path can leave a ready-for-review PR when enabling the merge fails; its result does not prove that a merge happened or that the remotely published head matches the evaluated change.

## DE-04.1 Five seams, different error contracts

```mermaid
flowchart TB
    HOST["Host supplies and drives implementations"] --> LLM["Llm: future of Result<br/>seams.rs:96"]
    HOST --> MEM["MemorySink: future of Result<br/>caller chooses best-effort handling<br/>seams.rs:148"]
    HOST --> NOTE["Notifier: future of unit<br/>implementation absorbs errors<br/>seams.rs:320"]
    HOST --> RUN["EvaluatorRunner: synchronous ExecOutcome<br/>runner.rs:30"]
    HOST --> FORGE["Forge: synchronous partial outcome<br/>query and merge methods return Result<br/>seams.rs:367"]
```

**What it shows.** Llm, MemorySink and Notifier return boxed futures. Forge and EvaluatorRunner are synchronous. LocalRunner uses a shell timeout but always sets timed_out to false; the receipt classifier separately recognises exit 124 and 137 (`runner.rs:69`, `receipts.rs:239`).

**Why it is this way.** The host owns sequencing and error policy. Returning unit from Notifier excludes a returned error; it does not enforce a deadline or prevent an implementation from panicking. No seam automatically records errors in a run journal.

## DE-04.2 Publication precedes the optional merge decision

```mermaid
flowchart TB
    PUBLISH["Publish draft first<br/>persist.rs:298"] --> FACTS["If enabled and pushed with URL,<br/>query actual base and required checks<br/>persist.rs:304"]
    FACTS --> DECIDE["Decide from supplied facts<br/>automerge.rs:110"]
    DECIDE -->|"disabled or refused"| KEEP["Return publication outcome<br/>persist.rs:330"]
    DECIDE -->|"allowed"| READY["Mark PR ready<br/>seams.rs:471"]
    READY -->|"success"| MERGE["Request auto squash merge<br/>seams.rs:480"]
    READY -->|"failure"| ERROR["Record enable error<br/>persist.rs:327"]
    MERGE -->|"failure"| ERROR
    MERGE -->|"success"| ENABLED["Enable command succeeded<br/>persist.rs:326"]
    ERROR --> STATE["No draft-restoration or outcome query<br/>seams.rs:480"]
```

**What it shows.** A rejected verdict or self-referential target prevents merge enablement but does not prevent this helper from first publishing a draft. The adapter separately gates publication before calling the lower-level forge. With the option off, only draft publication is attempted.

**Why it is this way.** ADR-DE-006 governs merging, while callers own publication eligibility. A successful ready command followed by a failed merge command leaves a partial transition; the offline command fixture reproduces that sequence and absence of an undo command.

## DE-04.3 Identity checks are narrower than a verified remote change

```mermaid
flowchart TB
    REQUEST["DraftPr names checkout, slug and branch<br/>no expected commit field<br/>seams.rs:341"] --> PUSH["Push branch to origin<br/>seams.rs:426"]
    PUSH --> CREATE["Create draft in supplied repo slug<br/>success stdout becomes Some URL<br/>seams.rs:512"]
    CREATE --> CLEAN["Clean means pushed and Some URL<br/>persist.rs:299"]
    CLEAN --> SELF["Self-reference: known slug,<br/>protected paths or three fixed directories<br/>automerge.rs:114<br/>automerge.rs:144"]
    SELF --> HEAD["Merge command has no expected-head guard<br/>seams.rs:481"]
```

**What it shows.** Empty successful stdout qualifies as Some URL. A tracked crate named dream-gate under a different directory is not recognised by the fixed-path check. Neither the request nor merge command carries the evaluated commit hash.

**Why it is this way.** The API assumes the caller supplies the correct checkout and branch and that those identities stay suitable. The [audit](../../estate-review/2026-11-01-dream-forge-and-seam-contracts.md) records these limits; it does not claim that a live unverified commit was merged.

**Drift (draft after enable failure):** Published's error-field documentation says the PR stays a draft (`persist.rs:278`), but successful ready followed by failed merge does not restore draft status.
