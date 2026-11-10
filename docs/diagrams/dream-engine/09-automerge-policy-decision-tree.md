---
id: DE-09
title: The automerge policy decision tree
area: dream-engine
governing: [../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md]
adrs: [ADR-DE-006]
sources:
  - ../dream-engine/crates/dream-core/src/automerge.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-core/src/runstate.rs
  - ../dream-engine/crates/dream-core/src/seams.rs
  - ../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md
verified_commit: b40d50724bb6344dc00d57270f7166b138714406
---
## For developers

An ACCEPT says the repository's evaluators did not object, not that the
change is wanted, so the default is a draft pull request a human merges
(`crates/dream-core/src/automerge.rs:3`). `automerge::decide` is a pure
function that turns facts the caller gathered into `Disabled`, `Refused` or
`Allowed` (`crates/dream-core/src/automerge.rs:110`). `persist::publish`
gathers the facts, calls it, and only then asks the forge to enable
automerge (`crates/dream-core/src/persist.rs:291`).

## For the business

Automerge defaults off. A host may publish a draft and opt in to requesting automerge when the policy accepts its supplied facts. The policy refuses unknown or empty required-check lists and recognised engine repositories. It does not establish that checks passed, the PR head matches the evaluated tree, or every renamed engine fork is recognised.

## DE-09.1 The decision tree

```mermaid
flowchart TB
    S["decide<br/>automerge.rs:110"] --> A{"cfg.auto_merge on?<br/>automerge.rs:111"}
    A -->|no, the default| DIS["Disabled"]
    A -->|yes| B{"engine repo, protects gate crates, or contains them?<br/>automerge.rs:114"}
    B -->|yes| R1["Refused SelfReferential"]
    B -->|no| C{"verdict is ACCEPT?<br/>automerge.rs:120"}
    C -->|no| R2["Refused NotAccepted"]
    C -->|yes| D{"clean draft: pushed and PR url?<br/>automerge.rs:123"}
    D -->|no| R3["Refused PrNotClean"]
    D -->|yes| E{"required status checks<br/>automerge.rs:126"}
    E -->|"error, 404 or unsupported"| R4["Refused ChecksUnknown"]
    E -->|"list empty or blank"| R5["Refused NoRequiredChecks"]
    E -->|"at least one name"| OK["Allowed"]
```

**What it shows.** Five refusal reasons in a fixed order, ahead of the one
`Allowed` leaf. The order matters for what is reported, not for the outcome.
**Why it is this way.** `gh pr merge --auto` merges at once when nothing is
required, so an empty list, a 404 and any API error are refusals
(`crates/dream-core/src/automerge.rs:13`). The self-referential test has
three heuristics for recognising engine targets: the slug, a config
whose protected set covers the gate crates, and a checkout that contains them
(`crates/dream-core/src/automerge.rs:136`, `crates/dream-core/src/automerge.rs:144`).
The flag lives in the protected config, so a candidate patch cannot switch it
on (`docs/adr/ADR-DE-006-automerge-is-an-option.md:30`).
**Invariant:** `Disabled` is returned before any other fact is consulted
(`crates/dream-core/src/automerge.rs:111`).

## DE-09.2 Gathering the facts in publish

```mermaid
sequenceDiagram
    participant P as publish
    participant F as Forge
    participant D as automerge decide
    P->>F: publish_draft, persist.rs line 298
    F-->>P: pushed and PR url
    P->>P: draft clean means pushed with a url, persist.rs line 299
    alt auto_merge on and draft clean
        P->>F: pr_base_branch of the PR, persist.rs line 307
        F-->>P: actual base branch
        P->>F: required_status_checks of that branch only if it matches, persist.rs line 309
    else flag off or draft not clean
        P->>P: required checks left unsupported, branch protection never queried
    end
    P->>D: decide with the gathered facts, persist.rs line 322
    D-->>P: Disabled, Refused or Allowed
    alt Allowed with PR URL
        P->>F: enable_automerge, persist.rs line 325
        F-->>P: ok or error kept in automerge_error
    else Disabled or Refused
        P->>P: no enable request
    end
```

**What it shows.** With the flag off no protection query is made at all
(`crates/dream-core/src/persist.rs:304`). With it on, protection is read for
the branch the PR really targets, and a mismatch with the expected base is a
refusal (`crates/dream-core/src/persist.rs:311`).
**Why it is this way.** A forge that cannot say what protects a branch never
gets automerge: the trait defaults return `Unsupported`, which the policy
treats as a refusal (`crates/dream-core/src/seams.rs:377`).
**Tension:** the forge step is untested against live GitHub. The only test
touching `GhForge` is a dyn-safety check (`crates/dream-core/src/seams.rs:647`),
so the protection parsing, the draft-to-ready transition and the squash
request have only been exercised against fakes.
**Debt:** the publish outcome is not reconciled. The host contract requires journalling `Persisted` before the forge request and nothing records the push, the pull request URL or
the ledger commit afterwards, so the delivery guarantee is at-most-once: a
crash after `Persisted` can lose a publish, whether before the pull request
exists or after it exists but before its URL or ledger row is recorded, and the
restart finalises the night to `Complete` without asking the forge
(`crates/dream-core/src/runstate.rs:615`, `crates/dream-core/src/persist.rs:298`).
Recovery is the operator's `reset_night`, which archives the journal so the
night runs again; the night key `(repo, date, night_id)` is the idempotency key
for any future reconciliation (`crates/dream-core/src/runstate.rs:441`).
**Debt:** `GhForge` runs `gh pr ready` before `gh pr merge --auto --squash`. If the second command fails, the PR can remain ready while `automerge_error` reports failure; no rollback or retry restores draft state (`crates/dream-core/src/seams.rs:469`).
**Debt:** `draft_clean` means only `pushed && pr_url.is_some()`. It does not validate an empty URL, remote head or PR identity; required-check names do not prove check success (`crates/dream-core/src/persist.rs:299`).
