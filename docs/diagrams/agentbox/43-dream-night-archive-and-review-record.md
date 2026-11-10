---
id: AB-43
title: Dream night archive and the human review record — durable artefacts, review import, locked ledger fills
area: agentbox
governing:
  - ../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md
  - ../project/agentbox/docs/developer/dream-engine.md
adrs: [ADR-2190, ADR-2087, ADR-2141, ADR-2071, ADR-2115]
sources:
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../project/agentbox/services/dream-engine/src/main.rs
  - ../project/agentbox/services/dream-engine/src/review.rs
  - ../project/agentbox/services/dream-engine/src/ledger_io.rs
  - ../project/agentbox/services/dream-engine/src/archive.rs
  - ../project/agentbox/services/dream-engine/src/estate.rs
  - ../project/agentbox/services/dream-engine/src/records.rs
  - ../project/agentbox/services/dream-engine/src/config.rs
  - ../project/agentbox/schema/agentbox.toml.schema.json
  - ../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md
verified_commit: 720e0d89656a4360998b3a716d0f3ee8cac66f93
---

## For developers

ADR-2190 (merged in agentbox PR 43, 795fd6107) amends ADR-2087 point 5: the ledger's `Reviewer` and `Review-minutes` cells now come only from a night's `review.json`, every ledger writer takes one lock per ledger, night artefacts live in a durable directory that is committed and pushed each night to a private archive repository, a configured git identity is a fallback only, and a start-up guard refuses to dream over an un-migrated artefact directory. Read with AB-23 (the nightly loop) and AB-41.

## For the business

A nightly improvement is now reviewed on the record: a person's review is written down against that night's evidence and only then appears in the ledger, so the ledger can no longer claim a human looked at work an agent merged. The evidence for every night is kept and backed up off the machine.

## AB-43.1 One night: start-up guard, backfill, row, archive

```mermaid
sequenceDiagram
    autonumber
    participant MAIN as main loop<br/>main.rs:516
    participant GUARD as migration_guard_clear<br/>main.rs:471
    participant ENG as run_night<br/>engine.rs:156
    participant REC as append_and_commit_ledger<br/>records.rs:44
    participant ARC as archive_night<br/>archive.rs:381
    MAIN->>GUARD: every tick, before a night (main.rs:518)
    GUARD->>GUARD: unmigrated_artefacts, legacy dir has nights and the durable one has none (estate.rs:86-88)
    GUARD-->>MAIN: false, inbox alert artefact-migration, sleep 600 s and retry (main.rs:477, :519-520)
    Note over MAIN: --once exits 1 instead (main.rs:430-431)
    MAIN->>MAIN: hold_night, .night-running.lock, fail-open (main.rs:486-487)
    MAIN->>ENG: run the night
    ENG->>ENG: backfill_reviews first, journalled review.backfill, fail-open (engine.rs:190, :510-511)
    ENG->>REC: ledger row with empty Reviewer and Review-minutes (engine.rs:1944-1953)
    REC->>REC: LedgerLock, append_row, commit, lock dropped after the commit (records.rs:65-66, :96-97)
    ENG->>ARC: end of night, journalled archive.push (engine.rs:396, :471)
    Note over ARC: add, commit, push to the upstream or the only remote, never forced<br/>a failure is fail-open and raised as an inbox alert (archive.rs:470, :486, :433-438)
```

**What it shows:** the order of a night after ADR-2190: the migration guard, the review backfill, the row written with empty human cells, and the archive commit and push. **Why it is this way:** the human cells could not be filled honestly from a merge event, because agents merge under the operator's account (comment at `engine.rs:1949-1951`).

**Invariant:** the night's own row never carries a reviewer: both cells are written empty (`../project/agentbox/services/dream-engine/src/engine.rs:1952-1953`) and are filled only by a later backfill from a validated record.

## AB-43.2 Recording a review: import, validate against the night, fill under the lock

```mermaid
sequenceDiagram
    autonumber
    participant CLI as review import<br/>main.rs:209
    participant RF as refuse_during_night<br/>main.rs:156
    participant IMP as import_jsonl<br/>review.rs:1040
    participant VAL as validate_record<br/>review.rs:543
    participant FILL as fill_ledger<br/>review.rs:803
    CLI->>RF: unless --dry-run or --force-during-night (main.rs:214)
    RF-->>CLI: exit 1 while a night holds .night-running.lock (main.rs:157, :165)
    CLI->>IMP: one review per JSON line (main.rs:228)
    IMP->>VAL: schema 1, deny_unknown_fields, witness recomputed from the night folder (review.rs:116-119, :459, :544)
    VAL-->>IMP: mismatch refused
    IMP->>FILL: write review.json, then fill the row
    FILL->>FILL: LedgerLock.acquire, crash-safe replace, commit (review.rs:815, :847, :850)
    Note over FILL: refuses a row whose Reviewer or Review-minutes cell is not empty (review.rs:745-746)
    FILL-->>CLI: fill failed after the record was written, review recorded, ledger fill pending (review.rs:1014)
    Note over CLI: exit 3 for a pending fill, the next night's backfill completes it (main.rs:243)
```

**What it shows:** `dream-engine review import` validates each record against the night it names, writes `review.json`, and fills the ledger row under the same lock the night uses. **Why it is this way:** an import runs outside the loop's singleton lock, so without a ledger lock an import and a night could each rewrite the ledger and lose the other's change (ADR-2190 context).

**Open:** the record is explicit but not authenticated: `reviewer` is free text from the importing tool and the witness binds a review to a night, not to a person (`../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md:46`).

## AB-43.3 Locks and identity: who serialises on what

```mermaid
flowchart TB
    LOCKP["lock_path: dream-ledger.lock in the git dir of the ledger's work tree<br/>ledger_io.rs:56, ledger_io.rs:124"] --> LL["LedgerLock.acquire, wait bounded at 300 s<br/>ledger_io.rs:176, ledger_io.rs:189, ledger_io.rs:47"]
    LL --> W1["night row append<br/>records.rs:65"]
    LL --> W2["review import and backfill<br/>review.rs:815"]
    NR["NightRunning, .night-running.lock in the artefact dir<br/>ledger_io.rs:59, ledger_io.rs:220"] --> REF["import and backfill refuse to run during a night<br/>main.rs:157"]
    CFG["dream_machine.git_author_name and git_author_email<br/>config.rs:100-103, schema agentbox.toml.schema.json:2981-2988"] --> GI["git_identity<br/>config.rs:140"]
    GI --> LAYER["fallback config layer, includes the account's real config<br/>GIT_CONFIG_GLOBAL set at start-up<br/>archive.rs:161, main.rs:343-345"]
    GI --> ARGS["archive commit: -c user.name and user.email<br/>only when the checkout has no identity<br/>archive.rs:87, archive.rs:102"]
    LAYER -.-> INV["INVARIANT: a checkout's own identity outranks the fallback"]
    ARGS -.-> INV
```

**What it shows:** one advisory lock per ledger shared by every writer, a separate night-in-progress lock that keeps manual imports out of a running night, and a configured identity that only fills a gap. **Why it is this way:** the estate gives checkouts their own identities on purpose, so the engine's identity may not override them (ADR-2190 decision, identity paragraph).

**Debt:** the ledger lock binds only processes that take it; the legacy `scripts/dream-machine-nightly.mjs` fallback and a loop built before this change rewrite the ledger without it (`../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md:52`).

## AB-43.4 The archive step: ignore list, SSH mode, push target

```mermaid
flowchart TB
    START["archive_night<br/>archive.rs:381"] --> OWN{"artefact dir is its own work tree<br/>archive.rs:284"}
    OWN -->|"no"| SKIP["nothing to archive"]
    OWN -->|"yes"| IGN["ensure_gitignore: .lock, *.lock, *.tmp*, loop.log<br/>archive.rs:305, archive.rs:51"]
    IGN --> ADD["git add -A<br/>archive.rs:470"]
    ADD --> COMMIT["commit, identity args only if the checkout has none<br/>archive.rs:486"]
    COMMIT --> SSH{"batch_ssh_command: BatchMode only when neither the<br/>environment nor core.sshCommand chooses one<br/>archive.rs:342-347, archive.rs:47"}
    SSH --> TARGET{"upstream, or the only remote<br/>archive.rs:433-435"}
    TARGET -->|"several remotes, no upstream"| FAIL["failed push, inbox alert, never force<br/>archive.rs:436-438"]
    TARGET --> PUSH["push to a private archive repository"]
```

**What it shows:** the end-of-night archive commits whatever the ignore list admits and pushes it to the checkout's own remote. **Why it is this way:** night artefacts were in a temporary directory and nights before 6 October 2026 were lost; the archive repository's name and remote are deliberately kept out of the public repositories (ADR-2190 consequences).

**Debt:** the artefact directory grows without bound; pruning it would orphan every row's witness and is now a decision-level change (`../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md:48`).

**Open:** ADR-2190 is `activation_status: staged` (`../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md:7`); it goes live after a rebuilt image pushes the archive, one real `review import` fills a row and the legacy directory is migrated (`../project/agentbox/docs/adr/ADR-2190-dream-ledger-human-cells-come-only-from-recorded-reviews-and-night-artefacts-are-a-durable-archived-record.md:75`). None of the three is evidenced at 720e0d896.
