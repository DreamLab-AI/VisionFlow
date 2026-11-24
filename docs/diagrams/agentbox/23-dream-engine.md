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
verified_commit: {agentbox: 1ea5494a1a9eca5e8da62a3f7db7354970b1c0f2, dream-engine: f438e031fbb5792274fb5f0dd73075ff771c0123}
---
Split 2026-11-23: AB-23.13 to AB-23.24 moved to [AB-44](44-dream-adapter-selection-decisions-and-delivery-limits.md) as AB-44.1 to AB-44.12. The old IDs resolve through the aliases declared there.

Repair follow-up, 2026-11-03: [AB-41.3](41-process-governance-and-dream-repairs.md#ab-413-supply-the-reusable-core-with-verifiable-host-inputs) and [DE-06](../dream-engine/06-formatting-and-evidence-repairs.md) record the adapter's registry adoption and core formatting/source/integrity repairs. This topic retains its historical source stamp.

This topic draws the adapter and the boundary after the 2026-10-15 extraction (agentbox ADR-2141). The judgement and the loop mechanism moved to the permissive repository DreamLab-AI/dream-engine, mapped in DE-01 and in its own corpus at `../dream-engine/docs/diagrams/README.md`, which is not duplicated here. AB-23.9 onward are adapter-owned and re-cited at the new revision.

State at this stamp: the adapter pins `dream-core =0.5.0` (adopted in agentbox e31fe2f, PR #53) and `dream-gate =0.3.0` from the registry (`../project/agentbox/services/dream-engine/Cargo.toml:26-27`), and the extraction is live: ADR-2141 is `accepted`, `complete`, `live`, with the operator's rebuild activated on 2026-10-16 and later releases adopted since (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:6-8`). Three adapter-owned changes since the first stamp are drawn here: source supply as plumbing (AB-44.9, ADR-2143), the opt-in Jev triage before the candidate build (AB-44.10, AB-23-S1-01) and the formatter step inside the core's candidate run (AB-23.4, ADR-DE-009). The crates.io versions are recorded in DE-01.

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
    EFFECT --> COMPLETE["Best-effort Complete<br/>engine.rs:2057"]
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
    READY -->|no| HAND["Persist HANDOFF before checking finished state<br/>engine.rs:690-694"]
    READY -->|yes| DONE{"finished_night?<br/>engine.rs:736"}
    DONE -->|yes| OLD["Return stored result<br/>engine.rs:750"]
    DONE -->|no| CYCLE["Run experiment and delivery; see AB-23.4 to AB-23.7<br/>engine.rs:1179<br/>engine.rs:1876"]
    HAND --> RECORD["Roster mode records returned results or failures; locked repos skipped<br/>engine.rs:302,315"]
    OLD --> RECORD
    CYCLE --> RECORD
    RECORD --> HEALTH["Health, governance publish, digest and optional forum suggestions<br/>engine.rs:343,408,416,426"]
    HEALTH --> DATE["Date consumed after a cycle, even a failed target cycle;<br/>a night that did not run retries in 600 s; marker is process-local<br/>main.rs:545, main.rs:538"]
    ROS -.-> LOSS["Unreadable roster resets history; failed save only warns<br/>dream-core/src/roster.rs:111<br/>engine.rs:316"]
```

**Limits.** The window controls start time, not a night deadline. Pause is checked only at roster entry, not during a night or targeted execution. Readiness can append HANDOFF to an already finished night; see the 1 November controls audit.

## AB-23.3 Module map and the licence boundary

> **Superseded** by AB-23.25 at 1ea5494 (frozen at 59b1aca).

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

## AB-23.6 Persistence and the external merge decision

> **Superseded** by AB-23.26 at 1ea5494 (frozen at 59b1aca).

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

**Why it is this way.** The engine does not merge. The command policy also permits an operator-delegated `/dream sweep` on non-excluded repositories, with green head checks, protected paths respected and a recorded sweep comment; it must not invent human review evidence (`skills/dream-machine/commands/dream.md:80-105`). The core's `autoMerge` policy is only used by `persist::publish`, which this adapter does not call. The [1 November forge/seam audit](../../estate-review/2026-11-01-dream-forge-and-seam-contracts.md) exercises the real GhForge subprocess boundary with local Git remotes and an offline replacement for gh; it does not establish live GitHub behaviour or current deployment.

**Open (publication identity):** at the dream-engine head the core `DraftPr` carries `head_sha`, the commit at the branch tip that was reviewed and pushed (`../dream-engine/crates/dream-core/src/seams.rs:351-355`), but it is used only to pin automerge with `--match-head-commit` (`../dream-engine/crates/dream-core/src/seams.rs:506-521`); `publish_draft` never reads it (`../dream-engine/crates/dream-core/src/seams.rs:433`). The agentbox adapter's request still carries only a branch name, not the verified commit hash: it builds `DraftPr` without `head_sha` (`../project/agentbox/services/dream-engine/src/engine.rs:1869-1875`) against the registry pin `dream-core =0.4.1` (`../project/agentbox/services/dream-engine/Cargo.toml:26`). A successful create command returns Some even for empty stdout (`../dream-engine/crates/dream-core/src/seams.rs:566`); the adapter then selects the acceptance notice by Option presence. The tested command boundary does not validate the URL or bind the pushed head to the evaluated tree.

**Tension (error ownership):** draft creation exposes only pushed and an optional URL, losing the command error detail (`../dream-engine/crates/dream-core/src/seams.rs:433`, `../dream-engine/crates/dream-core/src/seams.rs:547`). Memory errors are journalled and logged but tolerated (`../project/agentbox/services/dream-engine/src/engine.rs:1994`); they are not automatically written into the core run-state last_error field. ADR-DE-005's universal error-propagation wording is broader than this implementation.

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

`NightHealth.digest` is an optional status string. Health records are adapter reporting inputs; they do not attest that ledger rows, publication or all journal calls succeeded. AB-44.11 and AB-44.12 show those evidence boundaries (`../project/agentbox/services/dream-engine/src/digest.rs:86`).

## AB-23.9 Reminder counts local open items, not confirmed forum cases

```mermaid
flowchart TB
    BOOT["Boot prefers baked hook, then checkout; dedupes command substring<br/>entrypoint-unified.sh:2211-2224"] --> INPUT["UserPromptSubmit stdin<br/>dream-inbox-surface.cjs:35-39"]
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
**Source pin.** The boot registration is read at the topic revision (`config/entrypoint-unified.sh:2211-2224`). The earlier audit remains a dated receipt.

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
    SKILL --> SWEEP["sweep: delegated agent review, exclusions first<br/>green head checks and protected-path limits<br/>no automatic merge or human-review receipt<br/>commands/dream.md:80-105"]
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

## Audit qualification — 2026-10-29

1 November continuation: AB-23.7 and AB-44.9 were reassessed against matching current source and updated for recovery and source-supply limits. Later forge/seam and migration passes also reassessed AB-23.6 and AB-23.3. The controls pass reassessed AB-23.2, AB-23.10, AB-44.3 and AB-44.4. The receipt/depth pass reassessed AB-23.11 to AB-44.2. The execution/triage pass subsequently assessed the six remaining original panels (1, 4, 5, 8, 18 and 22). All 22 original panels and additions 23–24 now have a source assessment across the linked passes; open findings, the separate nested engine corpus and the wider estate audit remain outstanding.

This topic describes the extraction as source at the stamped revisions. ADR-2141 records `implementation_status: complete` and `activation_status: live`: the operator rebuilt and activated the registry-pinned image on 2026-10-16 and the supervised daemon was RUNNING with no degradation (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:84`). The later pins (0.2.0, 0.3.0, 0.4.0) are source in `Cargo.toml`; the record carries activation evidence for 0.2.0 only (`../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md:57`).

**Resolved:** the earlier follow-on that `ontomark/dream.config.json` (`"adrConvention": "ADR-OM-NNN"`) would not load under the stricter `dream-gate` validation is closed: the legacy free-form template is accepted and rendered with its prefix from `dream-gate` 0.1.1 (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:72`).

**Tension (activation records):** ADR-2141 says the image running the adapter was rebuilt and activated (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:8`), while ADR-2081 still records `activation_status: staged` for the annexe-depth and receipt corrections that the same engine source carries (`../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md:7`); its review_trigger names the first nightly run on the rebuilt image as the receipt that moves it to live (`../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md:13`); AB-23.11 to AB-44.2 describe source behaviour and claim no live receipt.

## AB-23.25 The registry pins after the dream-core 0.5.0 adoption

```mermaid
flowchart TB
    AD["Agentbox adapter, AGPL-3.0-only<br/>services/dream-engine/Cargo.toml:8"]
    AD --> PIN["Exact registry dependencies: core 0.5.0 and gate 0.3.0, one dream-gate copy<br/>services/dream-engine/Cargo.toml:26-27"]
    PIN --> CORE["dream-core 0.5.0: DraftPr gains head_sha, automerge pinned to it<br/>../dream-engine/crates/dream-core/src/seams.rs:351-355"]
    AD --> POLICY["Written no-publication policy, no publish=false field<br/>services/dream-engine/Cargo.toml:4"]
    CORE -.-> DR["DRIFT: ADR-2141's activation and 0.3.0 re-verification records are historical;<br/>they do not establish that the running image carries 0.5.0<br/>ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:84"]
```

**What it shows.** The frozen AB-23.3 drew the adapter on `dream-core =0.4.1`. Agentbox e31fe2f (PR #53, 2026-11-23) moved it to `=0.5.0`, the release that made the automerge head pin a breaking change (DE-01), and kept `dream-gate =0.3.0`, the gate release 0.5.0 needs. The module map, licence boundary and missing publication guard are otherwise as AB-23.3 draws them. **Why it is this way.** The two pins must name the same dream-gate release, or two copies link and their types do not unify (`../project/agentbox/services/dream-engine/Cargo.toml:24-25`).

## AB-23.26 Draft publication now carries the reviewed branch tip

```mermaid
flowchart TB
    CHOOSE["accepted candidate, persistence enabled<br/>engine.rs:1828"]
    TIP["branch_tip: rev-parse the verified branch, a lowercase hex commit<br/>engine.rs:1873, engine.rs:2232"]
    PUB["publish_draft with DraftPr head_sha set to that tip<br/>engine.rs:1876, engine.rs:1882"]
    NOPUB["tip unreadable: nothing is pushed, PrOutcome without URL<br/>engine.rs:1900"]
    LOST["no URL: the lost-PR alert fires<br/>engine.rs:1912"]
    GH["GhForge.publish_draft pushes and runs gh pr create; it does not read head_sha<br/>../dream-engine/crates/dream-core/src/seams.rs:433"]
    AM["automerge_args refuses an empty or non-hex head_sha, pins --match-head-commit<br/>../dream-engine/crates/dream-core/src/seams.rs:506-521"]
    CHOOSE --> TIP
    TIP -->|ok| PUB --> GH
    TIP -->|error| NOPUB --> LOST
    PUB -.->|"only on the automerge path, which this adapter does not call"| AM
```

**What it shows.** The frozen AB-23.6 recorded that the adapter built `DraftPr` without `head_sha`. Since e31fe2f it reads the tip of the branch being pushed and passes it; a tip it cannot read publishes nothing and raises the lost-PR alert. **Open (publication identity, narrowed):** the hash now travels with the request, but the adapter still calls only `publish_draft`, which ignores it, so the pinned automerge is never exercised in the estate; a successful create still returns Some even for empty stdout (`../dream-engine/crates/dream-core/src/seams.rs:566`).
