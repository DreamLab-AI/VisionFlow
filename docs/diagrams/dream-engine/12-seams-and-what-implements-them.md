---
id: DE-12
title: Seams and what implements them
area: dream-engine
governing: [../dream-engine/docs/adr/ADR-DE-005-seams.md]
adrs: [ADR-DE-005]
sources:
  - ../dream-engine/crates/dream-core/src/seams.rs
  - ../dream-engine/crates/dream-core/src/runner.rs
  - ../dream-engine/crates/dream-core/src/candidate.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-core/src/lib.rs
  - ../dream-engine/crates/dream-core/src/source.rs
  - ../dream-engine/docs/adr/ADR-DE-005-seams.md
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---
## For developers

Everything the engine must not know (which model, where memory lives, how a
test is run, where notices go, which forge hosts the pull request) is a trait.
Four traits live in `seams.rs` and the fifth, `EvaluatorRunner`, in
`runner.rs`, matching the five listed in ADR-DE-005
(`docs/adr/ADR-DE-005-seams.md:18`). The async ones return a plain boxed future
so the crate links no async runtime (`crates/dream-core/src/seams.rs:39`).

## For the business

A product plugs in its own model, memory, test machines, chat channel and code
host without forking the engine. Scripted evaluators support local tests. A full night still needs a host driver and model implementation; the local runner needs its command dependencies and the GitHub forge needs git, gh and authentication.

## DE-12.1 The five seams and the implementations that ship

```mermaid
classDiagram
    class Llm {
        +complete request
        +identity
    }
    class MemorySink {
        +record
    }
    class Notifier {
        +notify
    }
    class Forge {
        +publish_draft
        +required_status_checks
        +pr_base_branch
        +enable_automerge
    }
    class EvaluatorRunner {
        +run
        +describe
    }
    class NullSink
    class JsonlSink
    class StderrNotifier
    class GhForge
    class LocalRunner
    class ScriptedRunner
    MemorySink <|.. NullSink
    MemorySink <|.. JsonlSink
    Notifier <|.. StderrNotifier
    Forge <|.. GhForge
    EvaluatorRunner <|.. LocalRunner
    EvaluatorRunner <|.. ScriptedRunner
    note for Llm "host supplies Llm<br/>seams.rs line 96"
    note for GhForge "git push and gh CLI<br/>seams.rs line 424"
```

**What it shows.** Each trait with the implementations in this crate. `Llm` has
none: the host supplies the model (`crates/dream-core/src/seams.rs:96`).
`NullSink` and `JsonlSink` are at `seams.rs:157` and `seams.rs:202`,
`StderrNotifier` at `seams.rs:329`, `GhForge` at `seams.rs:424`, `LocalRunner` at
`runner.rs:48` and `ScriptedRunner` at `runner.rs:85`, all under
`crates/dream-core/src/`.
**Why it is this way.** The traits are dyn-compatible so a consumer holds a
`Box<dyn ...>` chosen at run time (`docs/adr/ADR-DE-005-seams.md:30`).
**Invariant:** `Notifier::notify` returns no `Result` (`crates/dream-core/src/seams.rs:320`).
**Debt:** the signature does not contain panics or impose a deadline. A host implementation can interrupt or stall the caller; verdict persistence must not depend on an uncontained notification.
**Debt:** `GhForge` has one test in this tree and it only proves the trait is
dyn-safe (`crates/dream-core/src/seams.rs:647`). The push, PR creation and
branch-protection paths have not been exercised against a live GitHub.

## DE-12.2 Where each seam is crossed in a night

```mermaid
sequenceDiagram
    participant H as Host driver
    participant L as Llm
    participant R as EvaluatorRunner
    participant M as MemorySink
    participant F as Forge
    participant N as Notifier
    H->>R: baseline evaluators (candidate.rs line 343 evaluate)
    H->>L: complete request with the prompt
    L-->>H: report with a patch block
    H->>R: build the candidate and re-run (candidate.rs line 547 and 552)
    H->>F: publish_draft and optional automerge (persist.rs line 291)
    H->>M: record the night, best effort
    H->>N: notify with no Result - host contains faults
```

**What it shows.** The order a host drives the seams in. The engine's own
code crosses `EvaluatorRunner` in `candidate::run` and `Forge` in
`persist::publish`; source gathering and repair also call the supplied `Llm` (`crates/dream-core/src/source.rs:1048`, `crates/dream-core/src/source.rs:1308`). The host orchestrates these operations and its memory and notification calls.
**Why it is this way.** Memory writes are best effort and callers log and
continue (`crates/dream-core/src/seams.rs:147`). The `publish_draft` signature returns `PrOutcome`; persistence of the night is a separate host responsibility and a trait implementation can still panic
(`crates/dream-core/src/seams.rs:365`).
**Open:** the sequence above is the intended host order from the crate docs
(`crates/dream-core/src/lib.rs:3`). No driver in this repository enforces it,
so the order is not machine-checked here.
