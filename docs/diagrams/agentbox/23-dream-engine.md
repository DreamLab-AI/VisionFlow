---
id: AB-23
title: Dream machine — the adapter, the extracted core and the nightly path
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2024, ADR-2053, ADR-2071, ADR-2081, ADR-2084, ADR-2087, ADR-2114, ADR-2115, ADR-2122, ADR-2141, ADR-2143, ADR-DE-008, ADR-DE-009]
sources:
  - ../project/agentbox/services/dream-engine/src/zone_crypto.rs
  - ../project/agentbox/docs/adr/ADR-2115-dream-decisions-surface-on-the-forum-governance-panel.md
  - ../project/agentbox/scripts/activation/adr-2087-check.sh
  - ../project/agentbox/management-api/lib/action-plane.js
  - ../project/agentbox/management-api/lib/execution-journal.js
  - ../project/agentbox/management-api/routes/exec-record.js
  - ../project/agentbox/services/dream-engine/src/baseline.rs
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../project/agentbox/services/dream-engine/src/config.rs
  - ../project/agentbox/services/dream-engine/src/dispatch.rs
  - ../project/agentbox/services/dream-engine/src/llm.rs
  - ../project/agentbox/services/dream-engine/src/runner.rs
  - ../project/agentbox/services/dream-engine/src/main.rs
  - ../project/agentbox/services/dream-engine/src/journal.rs
  - ../project/agentbox/services/dream-engine/src/governance.rs
  - ../project/agentbox/services/dream-engine/src/relay.rs
  - ../project/agentbox/services/dream-engine/src/inbox.rs
  - ../project/agentbox/services/dream-engine/src/digest.rs
  - ../project/agentbox/services/dream-engine/src/judge.rs
  - ../project/agentbox/services/dream-engine/src/policy.rs
  - ../project/agentbox/services/dream-engine/src/candidate_flow.rs
  - ../project/agentbox/services/dream-engine/src/records.rs
  - ../project/agentbox/services/dream-engine/src/notify.rs
  - ../project/agentbox/services/dream-engine/src/prompt.rs
  - ../project/agentbox/services/dream-engine/src/estate.rs
  - ../project/agentbox/services/dream-engine/src/wanted.rs
  - ../project/agentbox/services/dream-engine/src/jev_triage.rs
  - ../project/agentbox/services/agentbox-manifest/src/system_one.rs
  - ../project/agentbox/docs/adr/ADR-2053-dream-default-provider-is-zai.md
  - ../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md
  - ../project/agentbox/services/dream-engine/Cargo.toml
  - ../project/agentbox/services/dream-engine/src/lib.rs
  - ../project/agentbox/dream.config.json
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/hooks/dream-inbox-surface.cjs
  - ../project/agentbox/management-api/routes/dream.js
  - ../project/agentbox/management-api/lib/dream-ledger.js
  - ../project/agentbox/skills/dream-machine/commands/dream.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/scripts/dream-forum-suggestions.mjs
  - ../project/agentbox/scripts/dream-harvest.mjs
  - ../project/agentbox/scripts/dream-hooks-syntax.sh
  - ../project/agentbox/scripts/dream-inbox.mjs
  - ../project/agentbox/scripts/dream-machine-nightly.mjs
  - ../project/agentbox/management-api/lib/role-secret.js
  - ../project/agentbox/config/custody/identity-port-acl.json
  - ../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs
  - ../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
  - ../project/agentbox/scripts/activation/adr-2071-api-down-night.sh
  - ../project/agentbox/skills/podcast-knowledge-ingest/crontab
  - ../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md
  - ../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md
  - ../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md
  - ../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md
  - ../dream-engine/crates/dream-core/src/screen.rs
  - ../dream-engine/docs/adr/ADR-DE-008-requested-sources-are-honoured-first.md
  - ../dream-engine/docs/adr/ADR-DE-009-candidates-are-formatted-before-the-gate.md
  - ../dream-engine/crates/dream-core/src/runstate.rs
  - ../dream-engine/crates/dream-core/src/readiness.rs
  - ../dream-engine/crates/dream-core/src/roster.rs
  - ../dream-engine/crates/dream-core/src/candidate.rs
  - ../dream-engine/crates/dream-core/src/persist.rs
  - ../dream-engine/crates/dream-core/src/source.rs
  - ../dream-engine/crates/dream-core/src/ledger.rs
  - ../dream-engine/crates/dream-core/src/seams.rs
  - ../dream-engine/crates/dream-core/src/replay.rs
  - ../dream-engine/crates/dream-core/src/streak.rs
  - ../dream-engine/crates/dream-core/src/runner.rs
  - ../dream-engine/crates/dream-core/src/automerge.rs
  - ../dream-engine/crates/dream-core/src/lib.rs
  - ../dream-engine/crates/dream-core/Cargo.toml
  - ../dream-engine/crates/dream-gate/Cargo.toml
  - ../dream-engine/crates/dream-gate/src/receipts.rs
  - ../dream-engine/crates/dream-gate/src/gate.rs
  - ../dream-engine/crates/dream-gate/src/guard.rs
  - ../dream-engine/crates/dream-gate/src/manifest.rs
  - ../dream-engine/crates/dream-gate/src/verdict.rs
  - ../dream-engine/crates/dream-gate/src/patch.rs
  - ../dream-engine/crates/dream-gate/src/config.rs
  - ../dream-engine/crates/dream-gate/src/lib.rs
  - ../dream-engine/crates/pairwise-gate/src/lib.rs
  - ../dream-engine/crates/dream-cli/src/lib.rs
  - ../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md
  - ../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md
verified_commit: {agentbox: 720e0d89656a4360998b3a716d0f3ee8cac66f93, dream-engine: b40d50724bb6344dc00d57270f7166b138714406}
---
Repair follow-up, 2026-11-03: [AB-41.3](41-process-governance-and-dream-repairs.md#ab-413-supply-the-reusable-core-with-verifiable-host-inputs) and [DE-06](../dream-engine/06-formatting-and-evidence-repairs.md) record the adapter's registry adoption and core formatting/source/integrity repairs. This topic retains its historical source stamp.

This topic draws the adapter and the boundary after the 2026-10-15 extraction (agentbox ADR-2141). The judgement and the loop mechanism moved to the permissive repository DreamLab-AI/dream-engine, mapped in DE-01 and in its own corpus at `../dream-engine/docs/diagrams/README.md`, which is not duplicated here. AB-23.9 onward are adapter-owned and re-cited at the new revision.

State at this stamp: the adapter pins `dream-core =0.4.1` and `dream-gate =0.3.0` from the registry (`../project/agentbox/services/dream-engine/Cargo.toml:26-27`), and the extraction is live: ADR-2141 is `accepted`, `complete`, `live`, with the operator's rebuild activated on 2026-10-16 and later releases adopted since (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:6-8`). Three adapter-owned changes since the first stamp are drawn here: source supply as plumbing (AB-23.21, ADR-2143), the opt-in Jev triage before the candidate build (AB-23.22, AB-23-S1-01) and the formatter step inside the core's candidate run (AB-23.4, ADR-DE-009). The crates.io versions are recorded in DE-01.

## AB-23.1 Repo-night phases and early exits

```mermaid
flowchart TB
    CFG["Load config and select slot<br/>engine.rs:644"] --> LOCK{"Dry run or per-night lock acquired?<br/>engine.rs:655"}
    LOCK -->|held or IO error| STOP["Return error; no repo-night execution<br/>engine.rs:662"]
    LOCK -->|yes| READY{"Readiness admitted?<br/>engine.rs:671"}
    READY -->|no| HAND["HANDOFF; writes report and ledger unless dry run<br/>engine.rs:691"]
    READY -->|yes| PROMPT["Compile prompt and carry-over<br/>engine.rs:704"]
    PROMPT --> DRY{"Dry run?<br/>engine.rs:718"}
    DRY -->|yes| END["Return without dispatch<br/>engine.rs:720"]
    DRY -->|no| FIN{"finished_night by repo, date and night directory<br/>engine.rs:736"}
    FIN -->|Persisted or Complete| SKIP["Return stored outcome; no publication reconciliation<br/>engine.rs:750"]
    FIN -->|unreadable| BLOCK["BLOCKED-ENV result<br/>engine.rs:739"]
    FIN -->|unfinished| MAN["Resolve baseline, policy and frozen manifest<br/>engine.rs:795,838,858"]
    MAN --> HEALTH{"Sweep then annexe health<br/>engine.rs:901,918"}
    HEALTH -->|unhealthy| NOATT["BLOCKED-ENV before attempt count<br/>engine.rs:945"]
    HEALTH -->|healthy| BEGIN{"begin: fresh, resumed, terminal or error<br/>engine.rs:962"}
    BEGIN -->|terminal| SKIP
    BEGIN -->|journal error| BLOCK
    BEGIN -->|abandoned| ABANDON["Alert and BLOCKED-ENV record<br/>engine.rs:985"]
    BEGIN -->|fresh or resumed| BASE["Dispatch, checkout probe and baseline evaluation<br/>engine.rs:1049,1074,1192"]
    BASE -->|harness fault| EARLY["Persist blocked result and best-effort Complete<br/>engine.rs:1230"]
    BASE -->|usable| MODEL["Source, model, repair and optional triage<br/>engine.rs:1370,1412,1505,1534"]
    MODEL --> CAND["Optional candidate run; mark CandidateEvaluated even if skipped<br/>engine.rs:1560,1617"]
    CAND --> GATE["Judge then best-effort Gated<br/>engine.rs:1696"]
    GATE --> PERSIST{"Persisted journal write succeeds?<br/>engine.rs:1805"}
    PERSIST -->|no| PFAIL["BLOCKED-ENV persistence error<br/>engine.rs:1807"]
    PERSIST -->|yes| EFFECT["Draft branch handling and ledger tail<br/>engine.rs:1828"]
    EFFECT --> COMPLETE["Best-effort Complete<br/>engine.rs:2031"]
```

This is the control-flow overview; phase names record progress, not proof that every named action ran. Config loading precedes the lock, admission precedes the finished-night check, and several intermediate phase writes ignore errors (`../project/agentbox/services/dream-engine/src/engine.rs:1029,1192,1473,1617,1696`). AB-23.7 covers recovery in detail.

The [controls probe](../../estate-review/2026-11-01-dream-adapter-controls-and-egress.md) reproduced a completed night gaining a HANDOFF row and an overwritten report after its evaluator config became unusable. `Persisted` suppresses a repeat after a process crash; it neither confirms a draft was delivered nor supplies a power-loss transaction. Those limits are assessed in the [recovery](../../estate-review/2026-11-01-dream-recovery-and-source-supply.md) and [forge](../../estate-review/2026-11-01-dream-forge-and-seam-contracts.md) reviews.

## AB-23.2 Scheduling, admission and finished-night boundaries

```mermaid
flowchart TB
    LOOP["UTC loop: start <= hour < end; tick every 600 seconds<br/>migration guard first, AB-43<br/>main.rs:496, main.rs:518"]
    TARGET{"target supplied?<br/>main.rs:523"}
    LOOP --> TARGET
    TARGET -->|yes| SINGLE["run_cycle: selected repo; bypass pause, standby and roster cap<br/>engine.rs:128"]
    TARGET -->|no| PAUSE{"pause flag exists?<br/>engine.rs:158"}
    PAUSE -->|yes| RETRY["Return None: date not consumed; retry next tick<br/>main.rs:538"]
    PAUSE -->|no| NIGHT["Governance ingest; discover; optional branch sweep<br/>engine.rs:178"]
    NIGHT --> FILTER["Exclude manual standby and dry streak >= configured threshold<br/>engine.rs:213-223"]
    FILTER --> ROS["Load roster, select oldest date then fewest runs then name; cap<br/>dream-core/src/roster.rs:67"]
    ROS --> LOCK["Selected repos run serially; acquire per-night lock<br/>engine.rs:295,658"]
    SINGLE --> LOCK
    LOCK --> READY{"readiness admitted?<br/>engine.rs:671"}
    READY -->|no| HAND["Persist HANDOFF before checking finished state<br/>engine.rs:694"]
    READY -->|yes| DONE{"finished_night?<br/>engine.rs:736"}
    DONE -->|yes| OLD["Return stored result<br/>engine.rs:750"]
    DONE -->|no| CYCLE["Run experiment and delivery; see AB-23.4 to AB-23.7<br/>engine.rs:769"]
    HAND --> RECORD["Roster mode records returned results or failures; locked repos skipped<br/>engine.rs:302,315"]
    OLD --> RECORD
    CYCLE --> RECORD
    RECORD --> HEALTH["Health, governance publish, digest and optional forum suggestions<br/>engine.rs:343,408,416,426"]
    HEALTH --> DATE["Date consumed after a cycle, even a failed target cycle;<br/>a night that did not run retries in 600 s; marker is process-local<br/>main.rs:545, main.rs:538"]
    ROS -.-> LOSS["Unreadable roster resets history; failed save only warns<br/>dream-core/src/roster.rs:111<br/>engine.rs:316"]
```

**Limits.** The window controls start time, not a night deadline. Pause is checked only at roster entry, not during a night or targeted execution. Readiness can append HANDOFF to an already finished night; see the 1 November controls audit.

## AB-23.3 Module map and the licence boundary

```mermaid
flowchart TB
    AD["Agentbox adapter, AGPL-3.0-only<br/>services/dream-engine/Cargo.toml:8"] --> MODULES["Host orchestration, providers, transport,<br/>memory and forum modules<br/>services/dream-engine/src/lib.rs:25"]
    MODULES --> ENVELOPE["zone_crypto remains an adapter module<br/>services/dream-engine/src/lib.rs:62"]
    AD --> PIN["Exact registry dependencies<br/>core 0.4.1 and gate 0.3.0<br/>services/dream-engine/Cargo.toml:26"]
    PIN --> CORE["dream-core: host-driven loop mechanisms<br/>dream-core/src/lib.rs:3"]
    PIN --> GATE["dream-gate: judgement and evidence types<br/>dream-gate/src/lib.rs:3"]
    CORE --> GATE
    GATE --> PAIR["pairwise-gate<br/>crates/dream-gate/Cargo.toml:23"]
    GATE --> FP["failure-fingerprint<br/>crates/dream-gate/Cargo.toml:24<br/>crates/dream-core/Cargo.toml:18"]
    CORE --> FP
    CLI["dream-cli: offline command checks<br/>dream-cli/src/lib.rs:3"] --> CORE
    CLI --> GATE
    AD --> POLICY["Written no-publication policy<br/>No publish=false field in adapter manifest<br/>services/dream-engine/Cargo.toml:4"]
```

**What it shows.** The adapter imports the reusable engine and retains product-specific modules. Its exact pins resolve to one dream-gate version. The [migration/provenance audit](../../estate-review/2026-11-01-dream-migration-and-provenance.md) compares all 36 Rust source files in the four consumed crate releases and checks their cached archives against the consumer lockfile.

**Why it is this way.** ADR-2141 separates the engine mechanism from host orchestration. That dependency boundary does not itself enforce the adapter's no-publication policy: offline Cargo metadata leaves publication unrestricted and the selected package files include zone_crypto. No package was uploaded.

**Open (publication control):** the adapter's manifest declares its licence and version but no publication restriction (`../project/agentbox/services/dream-engine/Cargo.toml:4`, `../project/agentbox/services/dream-engine/Cargo.toml:8`). Source placement and stated policy must not be presented as a registry guard.

**Drift (activation evidence):** ADR-2141 records activation on 16 October and a later 0.3.0 re-verification (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:84`, `../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:92`). Current source pins core 0.4.1 (`../project/agentbox/services/dream-engine/Cargo.toml:26`); those historical records do not establish its current running image. The local tests in the dated audit use fixture services.

## AB-23.4 Guard ordering and the gate's evidence boundary

```mermaid
flowchart TB
    FLOW["attempt_judged enters candidate::run<br/>candidate_flow.rs:140"] --> BIN{"Binary deletion?<br/>candidate.rs:494"}
    BIN -->|yes| REF["Refused; no apply<br/>candidate.rs:496"]
    BIN -->|no| GUARD{"Protected-path and diff guard<br/>candidate.rs:502"}
    GUARD -->|refused| PRO["Protected or Refused state<br/>candidate.rs:503"]
    GUARD -->|allowed| PRE["Prepare worktree; recheck normalised patch<br/>candidate.rs:507,516"]
    PRE --> FMT["Configured fixer on control plane; record formatting outcome<br/>candidate.rs:540"]
    FMT --> RUN["Optional relocation and build; required evaluators<br/>candidate.rs:541,559,564"]
    RUN --> STATE["Applied state, actual patch, tree and receipts<br/>candidate.rs:593"]
    REF --> JUDGE
    PRO --> JUDGE
    STATE --> JUDGE["Adapter replay capture with baseline policy<br/>judge.rs:36"]
    JUDGE --> G["Gate: verify receipt identity; recheck applied patch on ACCEPT<br/>gate.rs:197,420"]
    G --> CTX["If supplied: config digest and per-test vetoes<br/>gate.rs:471,486"]
    CTX --> RESULT["ACCEPT claim plus no vetoes accepts<br/>gate.rs:489"]
```

Both the pre-apply guard and the pure gate recheck are present. The gate trusts caller-supplied receipt outcomes and candidate state; it is not an execution attestation. The adapter obtains a candidate config digest through an Option-returning helper, so a missing digest skips that comparison (`../project/agentbox/services/dream-engine/src/candidate_flow.rs:330`; `../dream-engine/crates/dream-gate/src/gate.rs:471`). Required evaluator failures veto an otherwise eligible ACCEPT; unreadable or absent evidence has its own classification.

Formatting is not a confinement boundary: a valid spaced Rust path attribute caused rustfmt to modify an external temporary file in the [candidate probe](../../estate-review/2026-11-01-dream-candidate-and-evidence.md). The same review reproduced an evaluator changing tracked files after the candidate tree was recorded. The repository's own config names no fixer at this stamp, but another nominated repository can enable one (`../project/agentbox/dream.config.json:43`; `../dream-engine/crates/dream-core/src/candidate.rs:540`).

## AB-23.5 Candidate worktree, relocation and apply repair

```mermaid
sequenceDiagram
    participant CF as candidate_flow<br/>candidate_flow.rs:189
    participant RUN as candidate::run<br/>candidate.rs:482
    participant PRE as prepare and format<br/>candidate.rs:507,540
    participant EV as runner and relocation<br/>candidate.rs:541
    participant SRC as apply_repair<br/>candidate_flow.rs:229
    CF->>RUN: guarded attempt at the dispatched base
    RUN->>PRE: apply strategies and normalisation, then configured fixer
    alt prepared successfully
        PRE-->>RUN: worktree, applied patch and tree hash
        RUN->>EV: relocate if configured, optional build, required evaluators
        EV-->>RUN: results, or provisioning failure
        RUN-->>CF: Attempt with prepared tree and receipts when available
    else PatchDidNotApply with no prepared tree
        RUN-->>CF: typed apply error
        CF->>CF: load apply-repair.json only if failed-patch digest matches
        opt no matching receipt
            CF->>SRC: report, git error, failed patch and baseline excerpt
            Note over CF,SRC: RESOLVED: excerpt now passes the source denylist and key-material check.<br/>source.rs:1353-1356
            SRC-->>CF: correction, decline, absent diff or error
            CF->>CF: best-effort receipt write
        end
        opt usable correction without binary deletion
            CF->>RUN: second guarded attempt
            RUN-->>CF: correction's own state and receipts
        end
    end
    CF-->>CF: return Attempt and the original or corrected patch text
    Note over PRE,EV: Worktree separation does not confine<br/>formatter or evaluator writes.<br/>Build results stay separate<br/>from evaluator receipts.
```

The baseline revision is passed into the candidate worktree, which protects against ordinary HEAD movement. The failing-hunk excerpt bypass the execution review reproduced is RESOLVED at dream-engine b40d507: `failing_excerpt_with` validates the path against the host denylist and withholds a blob with key material (`../dream-engine/crates/dream-core/src/source.rs:1343-1356`), and the adapter passes the source denylist (`../project/agentbox/services/dream-engine/src/candidate_flow.rs:208-214`). Apply repair is keyed to the failed patch, not to a complete baseline/prompt identity, and its receipt write is best effort (`../project/agentbox/services/dream-engine/src/candidate_flow.rs:198-238`). A missing receipt after a crash can therefore permit another repair request; the code does not establish exactly one external call across all failures.

The formatter runs locally before relocation. A successful worktree build does not establish the identity of a remote extracted tree or protect the operator's filesystem from subprocess effects. The [candidate](../../estate-review/2026-11-01-dream-candidate-and-evidence.md) and [annexe](../../estate-review/2026-11-01-dream-annexe-and-receipts.md) probes qualify those guarantees. Full ADR-2114 dispositions are in the [execution review](../../estate-review/2026-11-01-dream-execution-and-triage.md).

## AB-23.6 Persistence and the human merge gate

```mermaid
flowchart TB
    PERSIST["Persist verdict before forge and ledger<br/>engine.rs:1805"] --> CHOOSE["Prepared candidate and gate acceptance<br/>engine.rs:1828"]
    CHOOSE -->|"accepted and persistence enabled"| PUSH["Direct draft publication<br/>engine.rs:1869"]
    CHOOSE -->|"accepted, persistence disabled"| LOCAL["Keep branch locally<br/>engine.rs:1911"]
    CHOOSE -->|"candidate vetoed"| DISCARD["Discard candidate<br/>engine.rs:1916"]
    CHOOSE -->|"no prepared candidate"| NONE["No publication<br/>engine.rs:1928"]
    PUSH --> RESULT["URL present selects Accepted notice<br/>otherwise LostPr<br/>engine.rs:1883"]
    RESULT --> LED["Append ledger for every continuing branch<br/>engine.rs:1956"]
    LOCAL --> LED
    DISCARD --> LED
    NONE --> LED
    LED --> COMMIT["Append under the ledger lock; append failure stops,<br/>commit failure is logged<br/>records.rs:66, records.rs:80<br/>records.rs:110"]
    COMMIT --> MEMORY["Memory write failure is logged and tolerated<br/>engine.rs:1990<br/>engine.rs:2000"]
    MEMORY --> COMPLETE["Summary, remote cleanup, then complete<br/>engine.rs:2004<br/>engine.rs:2031"]
```

**What it shows.** The adapter checks publication eligibility before calling the forge, and calls `Forge::publish_draft` directly. The common ledger path follows all continuing branches, including vetoed and locally retained candidates. A publication failure does not erase the verdict; it changes the PR reference and notice. Earlier inbox and request-store effects precede the persistence boundary described here.

**Why it is this way.** A human still decides the merge. The core's `autoMerge` policy is only used by `persist::publish`, which this adapter does not call. The [1 November forge/seam audit](../../estate-review/2026-11-01-dream-forge-and-seam-contracts.md) exercises the real GhForge subprocess boundary with local Git remotes and an offline replacement for gh; it does not establish live GitHub behaviour or current deployment.

**Open (publication identity):** the PR request carries a branch name, not the verified commit hash (`../dream-engine/crates/dream-core/src/seams.rs:341`). A successful create command returns Some even for empty stdout (`../dream-engine/crates/dream-core/src/seams.rs:531`); the adapter then selects the acceptance notice by Option presence. The tested command boundary does not validate the URL or bind the pushed head to the evaluated tree.

**Tension (error ownership):** draft creation exposes only pushed and an optional URL, losing the command error detail (`../dream-engine/crates/dream-core/src/seams.rs:425`, `../dream-engine/crates/dream-core/src/seams.rs:512`). Memory errors are journalled and logged but tolerated (`../project/agentbox/services/dream-engine/src/engine.rs:1994`); they are not automatically written into the core run-state last_error field. ADR-DE-005's universal error-propagation wording is broader than this implementation.

## AB-23.7 Run journal — restart, resume, abandonment and the finished-night rule

```mermaid
flowchart TB
    LOCK["Acquire night lock<br/>engine.rs:658"] --> CHECK["Check finished night<br/>engine.rs:736"]
    CHECK -->|"matching terminal journal"| SKIP["Return recorded result<br/>engine.rs:744"]
    CHECK -->|"damaged journal"| BLOCK["Report journal fault<br/>engine.rs:739"]
    CHECK -->|"absent, unfinished or wrong night"| FREEZE["Build and freeze manifest<br/>engine.rs:795<br/>engine.rs:858"]
    FREEZE --> HEALTH["Annexe health before begin<br/>engine.rs:918"]
    HEALTH --> BEGIN["Begin checks night identity and attempts<br/>runstate.rs:523<br/>runstate.rs:542"]
    BEGIN -->|"same run, budget remains"| RESUME["Resumed with attempt increment<br/>runstate.rs:549"]
    BEGIN -->|"new run, budget remains"| FRESH["Fresh with carried attempt count<br/>runstate.rs:560"]
    BEGIN -->|"wrong night or exhausted"| STOP["Refuse or abandon<br/>runstate.rs:526<br/>runstate.rs:542"]
    RESUME --> PERSIST["Persist verdict before forge and ledger<br/>engine.rs:1805"]
    FRESH --> PERSIST
    PERSIST --> IO["Journal write uses temporary file and rename<br/>manifest.rs:328"]
```


**What it shows.** The early terminal check precedes manifest freezing, but it returns no result for a journal belonging to another night. `begin` rejects that identity later, after freezing and annexe health. Same-run retries increment the attempt; changed-run retries start at Initialised while carrying the night's count. A persisted verdict stops a restart even if HEAD moved.

**Why it is this way.** ADR-DE-007 repairs the earlier run-id-only rule. The [1 November recovery audit](../../estate-review/2026-11-01-dream-recovery-and-source-supply.md) verifies lock release and restart decisions after killing a fixture process before and after persistence. These tests cover process death, not power loss or successful publication. The writer does not synchronise file or directory contents (`../dream-engine/crates/dream-gate/src/manifest.rs:328`); the reset helper uses several link/unlink operations with best-effort rollback (`../dream-engine/crates/dream-core/src/runstate.rs:466`).

**Invariant:** the adapter checks the result of `persist` before attempting accepted-candidate publication (`../project/agentbox/services/dream-engine/src/engine.rs:1805`). At-most-once claims depend on the journal surviving, the lock being honoured and no explicit reset; the library does not reconcile forge outcomes.

## AB-23.8 Core types

```mermaid
flowchart TB
    RS["RunState: identity, phase, attempts, budget,<br/>timestamps, prior phase, last error and verdict<br/>runstate.rs:99"]
    PH["Phase<br/>Initialised, ManifestFrozen, BaselineEvaluated,<br/>ModelCalled, CandidateEvaluated, Gated,<br/>Persisted, Complete, Abandoned<br/>runstate.rs:56"]
    RES["Resume carries RunState<br/>Fresh, Resumed, AlreadyComplete, Abandoned<br/>runstate.rs:131"]
    LOCK["NightLock<br/>acquire and acquire_within<br/>runstate.rs:261,295"]
    CS["CandidateState: non-exhaustive<br/>Applied carries tree_hash and patch<br/>Protected carries paths<br/>Refused and DidNotApply carry detail<br/>NoPatch and NotAttempted have no payload<br/>gate.rs:255"]
    GD["GateDecision<br/>accepted, verdict, model_verdict,<br/>vetoes, required_outcomes, summary<br/>gate.rs:90"]
    VC["VetoClass<br/>Harness, Evidence, Unproven<br/>gate.rs:42"]
    V["Verdict<br/>Accept, Reject, Inconclusive,<br/>BlockedEnv, Handoff<br/>verdict.rs:16"]
    NH["Adapter NightHealth<br/>date, outcomes, nominated, standby,<br/>deferred, optional digest, journal, sweep<br/>digest.rs:74"]
    RES --> RS
    RS --> PH
    CS -->|input| GD
    GD --> VC
    GD -->|verdict_enum| V
    LOCK -.->|held around repo-night| RS
    NH -.->|reporting only| V
```


This graph shows selected record fields, phase/resume variants, lock methods, gate result and night-health fields. `CandidateState` is non-exhaustive and its payloads are abbreviated. `NotAttempted` also represents optional triage deferral, and `CandidateEvaluated` can be written when no evaluator ran (`../project/agentbox/services/dream-engine/src/engine.rs:1560,1617`). A type name does not establish execution or delivery.

`NightHealth.digest` is an optional status string. Health records are adapter reporting inputs; they do not attest that ledger rows, publication or all journal calls succeeded. AB-23.23 and AB-23.24 show those evidence boundaries (`../project/agentbox/services/dream-engine/src/digest.rs:86`).

## AB-23.9 Reminder counts local open items, not confirmed forum cases

```mermaid
flowchart TB
    BOOT["Boot prefers baked hook, then checkout; dedupes command substring<br/>entrypoint-unified.sh:2210-2223"] --> INPUT["UserPromptSubmit stdin<br/>dream-inbox-surface.cjs:35-39"]
    INPUT --> MACHINE{"Prompt starts with a known machine-turn tag?<br/>dream-inbox-surface.cjs:31-41"}
    MACHINE -->|yes| EXIT["No context injection<br/>dream-inbox-surface.cjs:68-70"]
    MACHINE -->|no| READ["Read JSON array; count status=open<br/>No published_event_id or relay check<br/>dream-inbox-surface.cjs:42-45"]
    READ -->|"missing, malformed, empty"| EXIT
    READ -->|"open items"| TIME{"More than 4 hours since shared stamp?<br/>dream-inbox-surface.cjs:48-51"}
    TIME -->|no| EXIT
    TIME -->|yes| OUT["Emit additionalContext under<br/>hookSpecificOutput<br/>One count and panel URL, no item bodies<br/>dream-inbox-surface.cjs:51-56"]
    OUT --> STAMP["Write stamp in stdout callback; inbox untouched<br/>Stamp write failure is ignored<br/>dream-inbox-surface.cjs:59-61"]
```

**Meaning.** The pointer says decisions await on the panel even when all open items lack a publication receipt. It measures local backlog; publication success and panel visibility need separate evidence. Existing fixture tests use unpublished items and still receive that wording.
**Limits.** The timestamp is shared by sessions but read and written without locking, so four hours is a sequential suppression policy, not a concurrent delivery guarantee. Failed stamp writes permit another reminder. See the [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md).
**Source pin.** Boot citations use the existing topic revision. The selected registration block is unchanged at the current checkout but moved to lines 2171–2196; the audit manifest records that comparison.

## AB-23.10 Control and reporting surfaces

```mermaid
flowchart LR
    SKILL["Operator /dream commands<br/>commands/dream.md:7"]
    SKILL --> STATUS["status: supervisor, pause flag, roster, ledgers and last-night health<br/>commands/dream.md:10"]
    SKILL --> RUN["run: stop loop, invoke baked Rust binary --once, restart loop<br/>commands/dream.md:46"]
    RUN --> ALL["No target: run_night checks pause and standby<br/>main.rs:438"]
    RUN --> ONE["--target: run_cycle bypasses pause and standby<br/>main.rs:446"]
    SKILL --> PAUSE["off/on: create/remove dream-paused flag<br/>commands/dream.md:41"]
    SKILL --> STAND["standby/revive: create/remove repo marker<br/>commands/dream.md:61"]
    SKILL --> FORUM["questions/answer/dismiss: forum decisions; local inbox is break-glass<br/>commands/dream.md:17"]
    SKILL --> DIG["digest and governance use Rust subcommands<br/>main.rs:183"]
    SKILL --> NOM["nominate: author repo config and verify evaluators<br/>commands/dream.md:74"]
    SKILL --> HAR["harvest: dream-harvest.mjs<br/>commands/dream.md:31"]
    API["Separate HTTP surface: GET /dream/status only; no run control<br/>routes/dream.js:24"]
    API --> JS["aggregateDreamStatus reads nominated repo ledgers<br/>dream-ledger.js:368"]
    JS --> PATH["Ledger path and realpath containment; 5 MiB limit<br/>dream-ledger.js:317"]
    JS -.-> DIFF["Status discovery ignores symlink dirs; Rust scheduler follows them<br/>dream-ledger.js:219<br/>dream-engine/src/config.rs:246"]
    RESET["reset-night joins caller date/repo into artefact path<br/>main.rs:273"]
    RESET --> ARCH["Acquire night lock then archive state and manifest; see AB-23.7<br/>dream-core/src/runstate.rs:441"]
```

**Scope.** This corrects the stale route from `/dream run` to the legacy JavaScript orchestrator. The skill's fixed window/cap summary is guidance, not effective-config reporting; HTTP status and scheduler discovery can disagree on symlinked nominations.

## AB-23.11 Annexe depth preserves layout, not immutable input identity

```mermaid
flowchart TB
    BASE["Capture target HEAD commit and tree<br/>baseline.rs:37; engine.rs:776"]
    BASE --> SUB["Canonical repo path relative to canonical workspace<br/>dispatch.rs:295"]
    SUB -->|inside workspace| DEPTH["Preserve nested depth, including symlink resolution<br/>dispatch.rs:307"]
    SUB -->|outside or unresolved| LEAF["Fall back to leaf name<br/>dispatch.rs:296-305"]
    DEPTH --> SHIP["clone_repo_and_siblings ships target first<br/>dispatch.rs:272"]
    LEAF --> SHIP
    SHIP --> ARCH["git archive resolves HEAD again at dispatch time<br/>dispatch.rs:183-194"]
    ARCH --> TAR["Copy archive, then tar into remote_dir/subpath<br/>dispatch.rs:203-214"]
    TAR --> SIB["For each configured sibling: resolve path, derive subpath, archive<br/>dispatch.rs:275-285"]
    SIB --> MISSING["Missing directory: warn and skip<br/>dispatch.rs:277"]
    SIB --> OVERLAY["Same destination: extract over existing target files<br/>No destination uniqueness check<br/>dispatch.rs:283-285"]
    BASE -.-> DRIFT["Frozen A, dispatch after HEAD moves to B: archive ships B<br/>baseline.rs:38; dispatch.rs:191"]
    OVERLAY -.-> MIX["Outside sibling with target's leaf can replace target source<br/>while retaining target-only files<br/>dispatch.rs:203-214"]
```

**Boundary.** Canonical depth fixes ordinary relative path dependencies. It does not freeze target/sibling revisions or prevent leaf-name overlays; the receipt audit reproduces both using local transport substitutes. Missing siblings are skipped, while archive/transfer errors propagate.

## AB-23.12 Pipefail preserves pipeline failure, with classification limits

```mermaid
flowchart TB
    CMD["Evaluator command and timeout budget"]
    CMD --> LOC["LocalRunner: cwd + timeout + bash -o pipefail -c<br/>dream-core/src/runner.rs:46"]
    CMD --> SSH["SshRunner: remote cd + timeout + bash -o pipefail -c<br/>dream-engine/src/runner.rs:24"]
    LOC --> PIPE["Pipeline returns rightmost nonzero status<br/>A failing producer survives a successful tail<br/>dream-core/src/runner.rs:53"]
    SSH --> PIPE
    PIPE --> RAW["ExecOutcome: status, stdout, stderr, duration<br/>dream-core/src/runner.rs:66; dispatch.rs:108"]
    RAW --> CLASS["classify prioritises transport, then timeout codes, then status/output<br/>dream-gate/src/receipts.rs:231"]
    CLASS --> RESERVED["124 or 137 => TimedOut even if child returned it immediately<br/>dream-gate/src/receipts.rs:239"]
    CLASS --> FAIL["Other nonzero status => Failed unless transport set<br/>dream-gate/src/receipts.rs:259"]
    CLASS --> ZERO["Zero: silent => Silent; recognised marker => ExplicitFail; otherwise Passed<br/>dream-gate/src/receipts.rs:250"]
    SSH -.-> DIFFER["Missing cwd => exit 1 / Failed; remote exit 255 => Blocked<br/>Local missing cwd => Blocked; local exit 255 => Failed<br/>dispatch.rs:102; dream-core/src/runner.rs:72"]
    PIPE -.-> MASK["No errexit guarantee: false followed by echo can pass<br/>Nested shells need their own failure policy<br/>dream-engine/src/runner.rs:26"]
```

**Scope.** Pipefail preserves the status of this shell's pipeline; it does not prove evaluator sensitivity or identify which process failed. The local and SSH paths can classify the same environment fault differently. GNU Bash specifies the rightmost failing stage, correcting the former first-stage claim.

## AB-23.13 No-patch ACCEPT is unproven, but the adapter still alerts

```mermaid
flowchart TB
    ACCEPT["Strict model verdict is ACCEPT<br/>dream-gate/src/gate.rs:406"]
    ACCEPT --> STATE{"candidate state?<br/>dream-gate/src/gate.rs:419"}
    STATE -->|NoPatch / Refused / NotAttempted| UNPROVEN["Unproven veto; no trusted candidate tree<br/>dream-gate/src/gate.rs:432-451"]
    STATE -->|DidNotApply| HARNESS["Harness veto<br/>dream-gate/src/gate.rs:441"]
    STATE -->|Applied| GUARD["Check protected paths, then candidate receipts if tree trusted<br/>dream-gate/src/gate.rs:421,461"]
    UNPROVEN --> VERDICT["Only unproven vetoes => INCONCLUSIVE<br/>dream-gate/src/gate.rs:506"]
    HARNESS --> BLOCK["Harness veto => BLOCKED-ENV<br/>dream-gate/src/gate.rs:500"]
    VERDICT --> STREAK["INCONCLUSIVE advances dry streak<br/>engine.rs:1697; dream-core/src/streak.rs:151"]
    VERDICT --> ALERT["Any nonempty veto list adds operator inbox alert<br/>engine.rs:1784-1800"]
    BLOCK --> ALERT
    GUARD --> RESULT["Gate result and typed vetoes<br/>dream-gate/src/gate.rs:489"]
    RESULT -->|nonempty veto list| ALERT
    ALERT -.-> GAP["ADR-2081 no-operator-alert clause is not implemented here<br/>engine.rs:1784"]
```

**Verified distinction.** The no-patch fixture yields INCONCLUSIVE and no Harness veto, but does add an alert. Candidate receipt grading and notification policy are separate; a verdict class alone does not suppress operator messages.

## AB-23.14 Finding selection, validation and append-time repair

```mermaid
flowchart TB
    REPORT["Model report text"] --> TABLE["First pipe row with enough cells, matching date and valid finding<br/>Quoted/backticked rows included; no Step-19 or deep check<br/>dream-gate/src/verdict.rs:488"]
    TABLE -->|found| SELECTED["Selected finding<br/>dream-gate/src/verdict.rs:362"]
    TABLE -->|none| LINE["First Finding: line passing cell contract<br/>dream-gate/src/verdict.rs:367"]
    LINE -->|found| SELECTED
    LINE -->|none| FALL["select_finding then take 80 Rust chars<br/>May retain Given prefix or exceed 80 UTF-16 units<br/>dream-gate/src/verdict.rs:374"]
    FALL --> SELECTED
    SELECTED --> BACK["Adapter checks entire row and repairs violations before append<br/>records.rs:53; dream-core/src/ledger.rs:254"]
    BACK --> LEDGER["Append repaired or original row; commit is fail-open<br/>records.rs:66-111"]
    RULES["Cell rules: nonempty, <=80 UTF-16 units, no hypothesis/pointer prefix<br/>Checks through repeated VETOED prefixes<br/>dream-gate/src/verdict.rs:394"] -.-> TABLE
    RULES -.-> LINE
    REPORT --> FULL["Full finding uses older hypothesis-first selection with a separate bound<br/>dream-gate/src/verdict.rs:514,523"]
    FULL --> OTHER["Memory/PR presentation path; not the ledger cell selector<br/>dream-gate/src/verdict.rs:510"]
```

**Limits.** Selection establishes syntax and date, not provenance from a particular report section or evaluated deep. The fallback needs the adapter's repair backstop; UTF-16 validation does not make its character truncation equivalent. Unicode parity differences remain recorded in the migration audit.

## AB-23.15 Provider selection, retries and cross-provider fallback

```mermaid
flowchart TB
    CFG["DREAM_LLM_PROVIDER overrides runtime config; unknown name becomes Zai<br/>estate.rs:165<br/>llm.rs:35"]
    CFG --> PRIMARY["Primary LlmAdapter; request token budget honoured<br/>llm.rs:278"]
    PRIMARY --> ZAI["Zai: POST /v1/messages; one transient retry after 20s<br/>llm.rs:49,120"]
    PRIMARY --> LOOM["Loom: published client, 600s timeout, 20s backoff<br/>declining_verbatim; pinned sampling<br/>llm.rs:205"]
    ZAI --> RESULT{"primary verdict request succeeds?<br/>engine.rs:1421"}
    LOOM --> RESULT
    RESULT -->|yes| REPORT["Use report text<br/>engine.rs:1422"]
    RESULT -->|no| FB{"fallback configured?<br/>estate.rs:205"}
    FB -->|no| DEG["Record degraded INCONCLUSIVE report<br/>engine.rs:1467"]
    FB -->|yes| SEND["Send the same prompt to fallback provider<br/>engine.rs:1439"]
    SEND --> OUT["Use fallback report, or degraded INCONCLUSIVE if both fail<br/>engine.rs:1450"]
    CFG -.-> POLICY["Zai primary: Loom fallback; Loom primary: Zai if key exists<br/>DREAM_LLM_FALLBACK=off disables fallback<br/>estate.rs:205-266"]
    POLICY -.-> FB
    POLICY -.-> EGRESS["Selecting Loom alone does not establish LAN confinement<br/>ADR-2053's one-setting claim exceeds implementation<br/>estate.rs:205"]
```

**Egress.** Retaining Z.AI as the default is the recorded operator decision. LAN confinement requires the effective primary, fallback and endpoint policy to agree; a provider label alone does not enforce it. Client-level guarantees remain owned by AB-28.11.

## AB-23.16 Placeholder and dispatch guards have limited coverage

```mermaid
flowchart TB
    INPUT["Parsed RuntimeConfig<br/>main.rs:293"]
    INPUT --> WHOLE["hp_host, loom_url, zai_url: whole-value placeholder resolution<br/>dream-engine/src/config.rs:150"]
    INPUT --> INFIX["hp_annexe_dir: expand embedded placeholders<br/>dream-engine/src/config.rs:169"]
    WHOLE --> DEFAULT["Unset/empty whole placeholder uses field default<br/>dream-engine/src/config.rs:151"]
    INFIX --> DEFAULT2["Missing/empty referenced variable uses annexe default<br/>dream-engine/src/config.rs:158"]
    WHOLE -.-> REMAIN["Embedded host/URL placeholders and nested replacement values survive<br/>dream-engine/src/config.rs:127"]
    INFIX -.-> NESTED["Replacement values are not expanded recursively<br/>dream-engine/src/config.rs:181"]
    DEFAULT --> GUARD["ssh, ssh_capture and scp_to reject trimmed-empty host<br/>dispatch.rs:65,92,124"]
    DEFAULT2 --> GUARD
    GUARD --> EXEC["Nonempty host: BatchMode and 10s connection timeout<br/>remote command completion has no overall deadline<br/>dispatch.rs:38"]
    FROM["scp_from: no empty-host guard; invokes scp with host:path<br/>dispatch.rs:147"]
    BAD["Explicit runtime file unreadable or invalid?<br/>main.rs:293"]
    BAD --> FALL["Log then use defaults, including enabled=true and provider=Zai<br/>main.rs:305<br/>dream-engine/src/config.rs:52,206"]
```

**Limit.** Existing tests cover the original simple-placeholder failures, not a universal ban on unresolved placeholders. The isolated controls probe confirms nested values survive and empty-host scp_from launches a fake executable.

## AB-23.17 Forum decisions — local application and delivery boundaries

```mermaid
flowchart TB
    ING["Start of night: ingest published open items<br/>governance.rs:1060-1071"] --> FETCH["Fetch request times and responses<br/>Query errors become empty results<br/>governance.rs:1088-1116"]
    FETCH --> VALID["Keep kind 31403, valid signature, signer not this agent<br/>No local admin or delegated-role lookup<br/>governance.rs:859-864"]
    VALID --> NEW["Choose newest timestamp and event ID per d tag FIRST<br/>governance.rs:866-877"]
    NEW --> BIND{"Request e tag absent, or equals stored request?<br/>governance.rs:881-885"}
    BIND -->|yes| MAP["Approve, reject or nonempty amend maps to resolution<br/>Delegate and unsupported responses stay open<br/>governance.rs:785-815"]
    BIND -->|no| NONE["No older valid decision is reconsidered"]
    NONE -->|"alert only"| ACK["Alert fallback: panel action plus publication time<br/>No outcome or local authority check<br/>governance.rs:897-916"]
    MAP -->|"supported resolution"| SAVE["Resolve OPEN local item; later decisions do not revise it<br/>inbox.rs:157-170"]
    MAP -->|"delegate or unsupported"| NONE
    ACK --> SAVE
    SAVE --> WITHDRAW["Sign kind-5 request; ACK increments withdrawn<br/>Local receipt write can fail, allowing resend<br/>governance.rs:663-709"]
    PUB["End of night: publish panel if content or tags differ<br/>Comparison does not verify returned event signature<br/>governance.rs:991-1014"] --> SWEEP["Sweep settled items, then publish unmarked open items<br/>governance.rs:1017-1046"]
    SWEEP --> WITHDRAW
    SWEEP --> CASE["Sign case 31402; relay ACK triggers local mark<br/>No request read-back here<br/>governance.rs:1023-1037"]
    CASE --> REOPEN["mark_published selects FIRST item with matching ID<br/>A reopened duplicate can update its closed predecessor<br/>inbox.rs:114-130"]
    REOPEN --> RISK["Old record can then target the new request for deletion<br/>Coordinate suppression alone does not protect its e tag<br/>governance.rs:589-603; governance.rs:622-635"]
```

**Authority.** Signature validity and rejection of this agent's key do not establish a human decision. The configured relay supplies admission policy; current relay source allows admins and case-delegated reviewers. The adapter also accepts missing request binding and ignores supersession after local resolution; these boundaries are tested in the [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md).
**Delivery.** An accepted deletion is a request acknowledgement, not proof of remote removal, local persistence or live card removal. The relay deletes rows but ignores database errors; the current panel subscription excludes kind 5 and its registry ignores it. Those cross-repository source findings are recorded separately in the audit, without a live UI claim.
**Compatibility.** Shared wire types reduce duplication but do not prevent version drift: the adapter consumes core beta.16 while the inspected forum checkout is beta.19. Missing keys, transport failures and local write errors remain fail-open; a zero-count report is not proof that no work was pending.

## AB-23.18 Annexe health and retention limits

```mermaid
flowchart TB
    MAN["Manifest frozen; attempt not yet counted<br/>engine.rs:858"] --> SWEEP["SSH retention: find date-like dirs by parent mtime, then rm<br/>engine.rs:901-905"]
    SWEEP --> MASK["Shell ends with true; find and deletion errors hidden<br/>engine.rs:904"]
    MASK --> SSH["annexe_health via synchronous SSH<br/>engine.rs:918; dispatch.rs:347"]
    SSH --> WRITE{"mkdir, write probe, remove probe succeed?<br/>dispatch.rs:327-332"}
    WRITE -->|no| FAIL["Health error<br/>dispatch.rs:348"]
    WRITE -->|yes| DF["df -Pk emits AVAIL-KB<br/>dispatch.rs:331"]
    DF --> PARSE{"Parse KiB and floor to GiB<br/>dispatch.rs:336,351"}
    PARSE -->|missing or below floor| FAIL
    PARSE -->|sufficient| BEGIN["begin counts attempt<br/>engine.rs:962"]
    FAIL --> REC["Alert plus BLOCKED-ENV record; no attempt count<br/>engine.rs:931,850"]
    BEGIN --> CLONE["Clone then check directory exists and is non-empty<br/>engine.rs:1049,1074"]
    CLONE --> RETRY["One reprovision attempt if checkout probe fails<br/>engine.rs:1088"]
```

The write test catches failures that a free-space figure alone misses; local fixtures exercise the actual public health function through a substitute SSH executable. The minimum defaults to 10 GiB; an explicit zero disables the capacity floor, not the write test (`../project/agentbox/services/dream-engine/src/engine.rs:2392`). Health checks are observations at that moment, not reservations of space or proof of checkout identity.

The exact retention command deleted a disposable old parent directory containing a freshly written child. It checks neither active jobs nor night locks, and a failing find command still returned success with no stderr. This is fixture evidence, not a claim that a live run was deleted. SSH has a 10-second connection timeout but no overall command deadline (`../project/agentbox/services/dream-engine/src/dispatch.rs:39-60`); a connected hung sweep or probe can therefore hold up later serial work. See the [execution review](../../estate-review/2026-11-01-dream-execution-and-triage.md).

## AB-23.19 ADR-2071 clause (c) — historical acceptance and retained tick

```mermaid
flowchart TB
    %% ADR-2071-journal-the-nightly-dream-cycle.md:204-212
    %% adr-2071-api-down-night.sh:29-31
    %% adr-2071-api-down-night.sh:83-95
    %% adr-2071-api-down-night.sh:94
    %% adr-2071-api-down-night.sh:100-105
    %% adr-2071-api-down-night.sh:102-103
    %% adr-2071-api-down-night.sh:106-108
    %% adr-2071-api-down-night.sh:108
    %% adr-2071-api-down-night.sh:109-111
    %% adr-2071-api-down-night.sh:113-118
    %% adr-2071-api-down-night.sh:129
    %% adr-2071-api-down-night.sh:137-142
    %% adr-2071-api-down-night.sh:132-138
    %% adr-2071-api-down-night.sh:139-140
    %% adr-2071-api-down-night.sh:141-142
    %% adr-2071-api-down-night.sh:145-148
    %% adr-2087-check.sh:397-419
    %% adr-2087-check.sh:410-417
    %% adr-2071-api-down-night.sh:121-124
    HIST["Historical acceptance: manual wrapper on 4 October<br/>Cron experiment was NOT used; marker and cron block retired<br/>ADR-2071 lines 195-203"]
    HIST -.-> TICK["Retained script for future reuse<br/>Requires external scheduling plus marker and state<br/>tick lines 29-31"]
    TICK --> PH{"Marker present and syntactically dated;<br/>read existing phase<br/>tick lines 83-95"}
    PH -->|"done or missed, even for older night"| RET["Retire new marker without stopping<br/>tick lines 94"]
    PH -->|"no phase, matching date 00:30-00:59"| PAUSE{"Dream paused?<br/>tick lines 100-105"}
    PAUSE -->|"yes"| MISS["Mark missed and retire<br/>tick lines 102-103"]
    PAUSE -->|"no"| TRY["Arm EXIT trap; attempt stop<br/>tick lines 106-108"]
    TRY --> OK{"Stop succeeded or status STOPPED?<br/>tick lines 108"}
    OK -->|"yes"| STOP["Write stopped state, then disarm trap<br/>tick lines 109-111"]
    OK -->|"no"| STOPFAIL["Attempt restart; mark missed and retire<br/>tick lines 113-118"]
    PH -->|"no phase, matching date at or after 01:00"| MISS
    STOP --> LATER["Later tick with phase stopped<br/>No independent watchdog in this script<br/>tick lines 129"]
    PH -->|"stopped"| LATER
    LATER --> WHY{"Ordered checks<br/>tick lines 137-142"}
    WHY -->|"first: matching date text in health file"| REC["night-record; no JSON or run-identity validation<br/>tick lines 132-138"]
    WHY -->|"else: RUNNING or STARTING"| INT["interrupted<br/>tick lines 139-140"]
    WHY -->|"else: deadline passed"| DEAD["deadline<br/>tick lines 141-142"]
    REC --> START["Attempt start; ignore failure<br/>Write phase done and started_at; retire marker<br/>tick lines 145-148"]
    INT --> START
    DEAD --> START
    START --> C3["C3: dated health, at least one verdict and failed/skipped post;<br/>matching done state, nonempty stopped_at, allowed reason<br/>checker lines 397-419"]
    C3 --> LIMIT["No restart-success or engine-exit validation;<br/>no proof of continuous outage or ledger append<br/>checker lines 410-417"]
```

**History.** ADR-2071 records a manual 4 October acceptance run, three newly dreamed repositories and one previously completed skip. This is historical evidence, not a fresh deployment check (`../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:204-212`).
**Current boundary.** The retained tick can consume a fresh marker using old done state, restart before the window from matching date text, mask an interruption and retire after a failed start. C3 trusts aggregate files rather than binding an outage to a specific completed run; see the [journal audit](../../estate-review/2026-11-01-dream-journal-and-api-down.md).
**Recovery limit.** The EXIT trap is removed after writing stopped state; later recovery depends on another tick and a successful supervisor start. It is not an unconditional guarantee that the API returns (`../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:106-111`, `../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:145-148`).

## AB-23.20 Signing keys — the identity-port capability and the local fallback

```mermaid
flowchart TB
    GOV["governance load_key, purpose agent_governance<br/>governance.rs:922-923"] --> ISO{"AGENTBOX_ROLE_ISOLATION equals 1?<br/>relay.rs:155"}
    ISO -->|"no"| LOCAL["Local key: FILE variable, bare variable or repo env line<br/>load_signing_key relay.rs:242"]
    ISO -->|"yes"| PUB["port pubkey for the purpose, no key file opened<br/>relay.rs:158"]
    PUB --> PORTK["PublishingKey.Port, public identity and alias only<br/>relay.rs:49-53"]
    LOCAL --> SIGNL["sign in engine memory<br/>relay.rs:104-106"]
    PORTK --> REQ["port_sign_request: kind 22242 goes to nip42_auth<br/>agent_governance goes to governance_event<br/>relay.rs:125, relay.rs:134, relay.rs:138-139"]
    REQ --> SOCK["one JSON line on the identity socket, 2 s timeouts<br/>port_request relay.rs:56"]
    SOCK --> ACL["shipped ACL: devuser may governance_event kinds 31400, 31402, 5<br/>with the optional agent_governance key<br/>identity-port-acl.json:39-42, identity-port-acl.json:54"]
    ACL --> GUARD["agent key: only AGENT_GOVERNANCE_KINDS, else refused at load<br/>identity_port/acl.rs:102, identity_port/acl.rs:342-343"]
    GUARD --> HUMAN["Human-decision kinds 31401, 31403-31405 stay never-grantable<br/>identity_port/acl.rs:89-91, identity_port/acl.rs:348-351"]
    PUB -->|"error"| SKIP["Governance skipped fail-open<br/>governance.rs:924-928"]
    LOCAL -->|"error"| SKIP
```

**Correction.** RESOLVED at agentbox 720e0d8 — the dream engine no longer reads its governance key under isolation: it takes a purpose-bound port capability (`../project/agentbox/services/dream-engine/src/relay.rs:150-158`) and the shipped ACL grants `governance_event` for kinds 31400, 31402 and 5 to the optional `agent_governance` key (`../project/agentbox/config/custody/identity-port-acl.json:54`); the key is `required: false` and must be provisioned by the operator before use (`../project/agentbox/config/custody/identity-port-acl.json:41`).
**Activation.** Flag-on behaviour depends on effective boot configuration, ownership and running identity. This pass reads those contracts but does not enable isolation, inspect keys or certify a running image. Key-file precedence alone does not imply the flag is on.
**Scope.** ADR-2122's ER-6 decision is assessed here, not its entire custody programme. Complete the signing, authentication and withdrawal design together, including old-key cases; see the [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md).

## AB-23.21 ADR-2143 — source supply is plumbing, not a decision

```mermaid
flowchart TB
    REPORT["Extract report requests and classify human actions<br/>engine.rs:1743<br/>wanted.rs:191"] --> STORE["Record pending keys or report a repeated inactive key<br/>wanted.rs:231"]
    STORE --> ALERT["Loop alert per record call<br/>engine.rs:1764"]
    STORE --> TAKE["Take active requests and increment each offer count<br/>wanted.rs:276"]
    TAKE --> SCAN["Append path-shaped scan entries<br/>engine.rs:1361"]
    SCAN --> CORE["Validate and select requested entries first<br/>source.rs:1099"]
    CORE --> RENDER["Clip then add markers and redact<br/>source.rs:858"]
    RENDER --> RECEIPT["Return receipt only if a file is marked inlined<br/>source.rs:1192"]
    RECEIPT --> SETTLE["Settle by path or directory prefix<br/>ranges are not compared<br/>wanted.rs:328"]
```


**What it shows.** Requests are offered before planner picks, with scan paths appended afterwards. Reservation counts candidate entries before validation, clipping markers can exceed the nominal byte allocation, and an empty line window can count as honoured. `take_for` counts invocations, not distinct night identifiers; `settle` treats any range from the same path as satisfying all its ranges (`../project/agentbox/services/dream-engine/src/wanted.rs:284`, `../project/agentbox/services/dream-engine/src/wanted.rs:328`). **Why it is this way.** ADR-2143 moves repetitive file requests out of the operator panel, but the [source-supply audit](../../estate-review/2026-11-01-dream-recovery-and-source-supply.md) finds that its completion markers do not prove the requested content was supplied. A requested directory is satisfied by any file under it. An all-refused read returns no source record (`../dream-engine/crates/dream-core/src/source.rs:1192`).

**Drift (question shape):** four fields are the prompt's desired format, not an enforced invariant. Parsing requires only Decision and If approved; unstructured questions pass through unchanged (`../project/agentbox/services/dream-engine/src/inbox.rs:278`, `../project/agentbox/services/dream-engine/src/inbox.rs:289`).

**Tension (expiry and evidence):** source receipt writes and settlement writes are best effort, and the shared request store has no transaction around read-modify-write (`../project/agentbox/services/dream-engine/src/engine.rs:1390`, `../project/agentbox/services/dream-engine/src/wanted.rs:91`). Current source does not establish three distinct nights, exact range delivery or one lifetime loop alert per request.

## AB-23.22 Optional triage can defer without a durable receipt

```mermaid
flowchart TB
    CLAIM{"Report claims ACCEPT and triage enabled?<br/>engine.rs:1534-1536"}
    CLAIM -->|no| BUILD["Proceed to ordinary candidate handling<br/>engine.rs:1560"]
    CLAIM -->|yes| ROUTE{"Endpoint available under configured locality?<br/>jev_triage.rs:147-188"}
    ROUTE -->|no| BUILD
    ROUTE -->|yes| ASK["First request: report head, path hints and evaluator names<br/>jev_triage.rs:320"]
    ASK --> SIGNAL{"Valid typed usefulness and support values?<br/>jev_triage.rs:347"}
    SIGNAL -->|fallback| BUILD
    SIGNAL -->|yes| DEFER["Compute defer from support and explicit threshold<br/>jev_triage.rs:491"]
    DEFER --> SCREEN["Second request: library discipline screen, capped inputs<br/>jev_triage.rs:522; screen.rs:222"]
    SCREEN --> RECORD["Write annotation or screen-fallback record<br/>jev_triage.rs:522-550"]
    RECORD --> CHECK{"Defer computed true?<br/>jev_triage.rs:558"}
    CHECK -->|no| BUILD
    CHECK -->|yes| SKIP["Skip build even if annotation write failed<br/>engine.rs:1560"]
    SKIP --> GATE["NotAttempted: ACCEPT stays unproven<br/>gate.rs:445"]
    RECORD -.-> AUTH["Each HTTP call has 15-second timeout and no redirects<br/>jev_triage.rs:275-279"]
```

Default threshold is unset, so valid signals annotate without deferral. A configured threshold must lie in 0 to 0.5. Neither usefulness nor the discipline score can approve a candidate. The two requests are sequential; the first failing means the second is not sent. An initial fallback writes nothing and leaves any prior annotation in place (`../project/agentbox/services/dream-engine/src/jev_triage.rs:506-512`).

The first request clips the report to 4,000 characters but does not cap the complete request. Its path hints omit deletions and pure renames. The second request retains at most 16 disciplines, while the saved annotation lists the full input set (`../project/agentbox/services/dream-engine/src/jev_triage.rs:306,320,528`; `../dream-engine/crates/dream-core/src/screen.rs:230`). These signals cannot replace evaluator evidence. The [execution review](../../estate-review/2026-11-01-dream-execution-and-triage.md) records transport-free probes, current configuration and the limits of ADR-2155's staged activation record. A staged record does not prove that no other running image has the feature.

## Audit qualification — 2026-10-29

1 November continuation: AB-23.7 and AB-23.21 were reassessed against matching current source and updated for recovery and source-supply limits. Later forge/seam and migration passes also reassessed AB-23.6 and AB-23.3. The controls pass reassessed AB-23.2, AB-23.10, AB-23.15 and AB-23.16. The receipt/depth pass reassessed AB-23.11 to AB-23.14. The execution/triage pass subsequently assessed the six remaining original panels (1, 4, 5, 8, 18 and 22). All 22 original panels and additions 23–24 now have a source assessment across the linked passes; open findings, the separate nested engine corpus and the wider estate audit remain outstanding.

This topic describes the extraction as source at the stamped revisions. ADR-2141 records `implementation_status: complete` and `activation_status: live`: the operator rebuilt and activated the registry-pinned image on 2026-10-16 and the supervised daemon was RUNNING with no degradation (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:84`). The later pins (0.2.0, 0.3.0, 0.4.0) are source in `Cargo.toml`; the record carries activation evidence for 0.2.0 only (`../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md:57`).

**Resolved:** the earlier follow-on that `ontomark/dream.config.json` (`"adrConvention": "ADR-OM-NNN"`) would not load under the stricter `dream-gate` validation is closed: the legacy free-form template is accepted and rendered with its prefix from `dream-gate` 0.1.1 (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:72`).

**Tension (activation records):** ADR-2141 says the image running the adapter was rebuilt and activated (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:8`), while ADR-2081 still records `activation_status: staged` for the annexe-depth and receipt corrections that the same engine source carries (`../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md:7`); its review_trigger names the first nightly run on the rebuilt image as the receipt that moves it to live (`../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md:13`); AB-23.11 to AB-23.14 describe source behaviour and claim no live receipt.


## AB-23.23 Journal delivery — local counters and remote evidence

```mermaid
flowchart TB
    SESSION["Night ID plus whole seconds, normalised and truncated to 160<br/>Not unique across all attempts or names<br/>journal.rs:300-315"] --> CALL["called: post tool.called, then increment open_calls<br/>journal.rs:165-187"]
    CALL --> EFFECT["Caller performs effect independently of delivery success<br/>Phase 1 is fail-open, not an approval gate<br/>journal.rs:11-17"]
    EFFECT --> DONE["completed: decrement open_calls BEFORE posting<br/>causation present only if called returned an event ID<br/>journal.rs:199-215"]
    CALL --> HTTP
    DONE --> HTTP
    HTTP{"Client enabled and fewer than 3 consecutive failures?<br/>journal.rs:236-244"}
    HTTP -->|"no client"| OFF["No delivery counters; open_calls still changes<br/>journal.rs:236"]
    HTTP -->|"breaker open"| SKIP["skipped++; no recovery probe this session<br/>journal.rs:237-243"]
    HTTP -->|"yes"| POST["POST /v1/exec/record<br/>same API journal singleton<br/>exec-record.js:59-62"]
    POST --> KEY["Caller key becomes deterministic event ID<br/>Duplicate ID returns prior envelope, without payload comparison<br/>exec-record.js:52-54; execution-journal.js:144-151"]
    KEY --> APPEND["Assign seq, await dispatch, then update in-memory state<br/>Concurrent appends can share seq; restart needs hydration<br/>execution-journal.js:152-185"]
    APPEND --> ACK{"HTTP success and string event_id?<br/>journal.rs:257-270"}
    ACK -->|"yes, including duplicate or empty ID"| REC["recorded++; reset failure streak<br/>journal.rs:267-270"]
    ACK -->|"no"| FAIL["failed++; increment streak; no retry or local outbox<br/>journal.rs:272-276"]
    DONE --> STATS["unpaired = current open_calls, not remote pair reconciliation<br/>journal.rs:224-229"]
    STATS --> HEALTH["Normal path records repo stats, then night-level stats<br/>Early returns and crashes have narrower coverage<br/>engine.rs:359-368; engine.rs:461-462; records.rs:411-428"]
```

**Evidence.** Loopback probes reproduce unmatched remote records with unpaired=0; route probes reproduce changed-payload deduplication, restart sequence reuse and concurrent duplicate sequence numbers while the on-disk hash chain verifies. These are separate integrity properties, detailed in the [journal audit](../../estate-review/2026-11-01-dream-journal-and-api-down.md).
**Coverage.** Several inbox writes have no tool pair, including shared-remote warnings, annexe-health failure and damaged-run alerts; inbox::add does not journal internally (`../project/agentbox/services/dream-engine/src/engine.rs:251`, `../project/agentbox/services/dream-engine/src/engine.rs:933`, `../project/agentbox/services/dream-engine/src/engine.rs:2160`, `../project/agentbox/services/dream-engine/src/inbox.rs:174-215`).
**Skip semantics.** The repo wrapper posts turn.started before calling the body and turn.completed afterwards, including locked or already-finished returns. No tool effect is required for those lifecycle events (`../project/agentbox/services/dream-engine/src/engine.rs:603-624`).


## AB-23.24 Digest — evidence limits and encryption configuration

```mermaid
flowchart TB
    DISK["Read health, configured ledgers and open-item count<br/>digest.rs:386-399"] --> DATE{"Matching health date?<br/>digest.rs:186"}
    DATE -->|no| ABSENT["Text says nothing was tested<br/>Missing evidence does not establish that conclusion<br/>digest.rs:187-191"]
    DATE -->|yes| ANY{"Any scheduled outcomes?<br/>digest.rs:193-217"}
    ANY -->|no| ZERO["No-eligible text and available standby reasons<br/>digest.rs:193-215"]
    ZERO --> LOCAL
    ANY -->|yes| WORDS["Compose from outcome labels and last ledger row<br/>ACCEPT says confirmed; REJECT says disproved<br/>digest.rs:242-275"]
    WORDS --> LOCAL["Append local open count as panel backlog<br/>digest.rs:220-227"]
    ABSENT --> LOCAL
    LOCAL --> ZONE["Resolve section and zone metadata; select encryption plan<br/>digest.rs:421-430"]
    ZONE --> PLAIN["Malformed or absent zones become empty;<br/>unknown section can select first configured zone<br/>zone_crypto.rs:86-105"]
    ZONE --> SECRET["Known encrypted zone with gate on: latest held epoch or refuse<br/>zone_crypto.rs:109-115; zone_crypto.rs:221-233"]
    PLAIN --> FALL["No encrypted match means plaintext plan<br/>zone_crypto.rs:222-224"]
    SECRET -->|"no held key"| STOP["Withhold digest<br/>digest.rs:431-432"]
    SECRET -->|"held key"| POST["Sign and publish; retry once after missing read-back<br/>digest.rs:453-479"]
    FALL --> POST
    POST --> VERIFY["Read-back match checks ID only<br/>Relay query does not verify signature or filter membership<br/>digest.rs:467-476; relay.rs:483-489"]
```

**Presentation.** A missing health file, REJECT label or unpublished inbox count is insufficient for the digest's stronger claims. Standby repositories disappear from the ordinary outcomes branch. Case context now reads the configured ledgerPath, RESOLVED at agentbox 720e0d8 (`../project/agentbox/services/dream-engine/src/governance.rs:279`).
**Privacy.** The no-key refusal applies only after zone policy identifies an encrypted zone. Malformed metadata produces plaintext selection even with the gate on; the probe does not publish plaintext or claim the live relay would admit it. No current zone-key epoch is fetched, so latest means latest locally held.
**Evidence.** The [governance audit](../../estate-review/2026-11-01-dream-governance-and-delivery.md) records isolated probes and the source boundary. Neither a publish ACK nor an ID-only read-back establishes delivery to the operator or a verified current deployment.
