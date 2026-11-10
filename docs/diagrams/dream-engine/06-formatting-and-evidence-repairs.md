---
id: DE-06
title: Confined formatting and post-evaluation evidence repairs
area: dream-engine
governing:
  - ../dream-engine/docs/adr/ADR-DE-003-protected-path-guard.md
  - ../dream-engine/docs/adr/ADR-DE-008-requested-sources-are-honoured-first.md
  - ../dream-engine/docs/adr/ADR-DE-009-candidates-are-formatted-before-the-gate.md
adrs: [dream-engine:ADR-DE-003, dream-engine:ADR-DE-008, dream-engine:ADR-DE-009]
sources:
  - ../dream-engine/crates/dream-core/src/candidate.rs
  - ../dream-engine/crates/dream-core/src/source.rs
  - ../dream-engine/crates/dream-core/Cargo.toml
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---

## For developers

These scoped repairs supersede the formatter and final-state observations in DE-02.2 and the repair-excerpt supply gap. They do not turn the evaluator host into a sandbox or reverify the other gate, replay and persistence panels.

## For the business

Formatting no longer follows a candidate's module declarations to other files. A changed final candidate cannot receive the earlier tree's successful identity receipt. The reliability of tests and isolation of their execution remain separate.

## DE-06.1 Explicit-file formatting and final tree identity

```mermaid
flowchart TD
    %% Source: ../dream-engine/crates/dream-core/src/candidate.rs:113
    A["Selected candidate files"] --> B["Require regular files with canonical paths inside worktree"]
    %% Source: ../dream-engine/crates/dream-core/src/candidate.rs:243
    B --> C["rustfmt reads stdin; ignores candidate config and child modules"]
    C --> D["Write stdout to explicit file only after success"]
    D --> E["Record formatted candidate; build and evaluate"]
    %% Source: ../dream-engine/crates/dream-core/src/candidate.rs:567
    E --> F["Check clean status and HEAD tree; rehash tracked files using fresh index"]
    F --> G{"Local and relocated trees match?"}
    G -->|Yes| H["Return Applied with prepared tree hash"]
    G -->|No or unverifiable| I["Return integrity error and non-applied state"]
```

**What it shows:** A spaced path attribute cannot make the formatter traverse an external module. A fresh index prevents assume-unchanged or skip-worktree flags from hiding final tracked-file changes. Archive-based hosts must establish a clean Git snapshot with the expected tree before evaluation.

**Why it is this way:** ADR-DE-003/009 require the tested candidate to match the published one. Tests reproduce the former external-file write and evaluator mutations. This detects final-state drift; shared Git metadata, transient writes restored before checking and arbitrary evaluator commands require host isolation.

## DE-06.2 Repair supply uses the same withholding rules

```mermaid
flowchart TD
    %% Source: ../dream-engine/crates/dream-core/src/source.rs:1343
    A["Failed hunk path and range"] --> B["Validate tracked relative path and host denylist"]
    %% Source: ../dream-engine/crates/dream-core/src/source.rs:1355
    B --> C["Read committed whole blob and check credential content"]
    C -->|Denied or credential match| D["Withhold excerpt"]
    C -->|Allowed| E["Select bounded line excerpt"]
    %% Source: ../dream-engine/crates/dream-core/src/source.rs:757
    F["Shared Ontomark scanner plus existing key rules"] --> C
    %% Source: ../dream-engine/crates/dream-core/src/source.rs:1120
    G["Planner deep, scans, commands and evidence"] --> H["Apply caller redactor before prompt construction"]
```

**What it shows:** A harmless-looking requested range cannot hide a credential elsewhere in the committed file. Publication and prompt supply share known token shapes through the Ontomark compiler crate with optional reasoning disabled.

**Why it is this way:** ADR-DE-008 bounds source supply consistently across the initial and repair stages. Tests cover denied names, traversal, out-of-range key material, provider tokens and a valid public excerpt. Token detection does not classify all confidential information.
