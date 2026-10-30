---
id: AB-23
title: Dream machine — the adapter, the extracted core and the nightly path
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2024, ADR-2053, ADR-2071, ADR-2081, ADR-2084, ADR-2087, ADR-2114, ADR-2115, ADR-2122, ADR-2141, ADR-2143, ADR-DE-008, ADR-DE-009]
sources:
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
  - ../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md
  - ../project/agentbox/services/dream-engine/Cargo.toml
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
verified_commit: {agentbox: 7208f7e0d749b44efcfe24bd503c567c9ff8f9bc, dream-engine: 59a3e215e826aa1e9ebe157d49c1ac504f7f39c9}
---
This topic draws the adapter and the boundary after the 2026-10-15 extraction (agentbox ADR-2141). The judgement and the loop mechanism moved to the permissive repository DreamLab-AI/dream-engine, mapped in DE-01 and in its own corpus at `../dream-engine/docs/diagrams/README.md`, which is not duplicated here. AB-23.9 onward are adapter-owned and re-cited at the new revision.

State at this stamp: the adapter pins `dream-core =0.4.0` and `dream-gate =0.3.0` from the registry (`../project/agentbox/services/dream-engine/Cargo.toml:26-27`), and the extraction is live: ADR-2141 is `accepted`, `complete`, `live`, with the operator's rebuild activated on 2026-10-16 and later releases adopted since (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:6-8`). Three adapter-owned changes since the first stamp are drawn here: source supply as plumbing (AB-23.21, ADR-2143), the opt-in Jev triage before the candidate build (AB-23.22, AB-23-S1-01) and the formatter step inside the core's candidate run (AB-23.4, ADR-DE-009). The crates.io versions are recorded in DE-01.

## AB-23.1 One repo-night — run phases

```mermaid
stateDiagram-v2
    [*] --> Locked
    note right of Locked
        NightLock::acquire_within 250 ms (agentbox/services/dream-engine/src/engine.rs:563),
        a dry run takes no lock. The lock is held until the night returns.
        A held lock skips the night with EngineError::NightLocked (engine.rs:567).
    end note
    Locked --> Skipped : lock held by another engine
    Locked --> Admission : lock taken
    Admission --> Handoff : readiness refuses
    note right of Admission
        readiness::assess (dream-core/src/readiness.rs:187) runs BEFORE any clone, build or
        model call. Refusals are the Unusable variants (readiness.rs:23), the HANDOFF row
        is written by persist_handoff (agentbox/services/dream-engine/src/records.rs:282).
    end note
    Admission --> FinishedCheck : admitted
    FinishedCheck --> Skipped : night already Persisted or Complete
    note right of FinishedCheck
        finished_night (dream-core/src/runstate.rs:401) is asked by night directory, not by run id,
        BEFORE any manifest, model call, commit or forge request (engine.rs:641).
        An unreadable journal is BLOCKED-ENV, never a fresh start (engine.rs:642-647).
    end note
    FinishedCheck --> Initialised : begin (engine.rs:867)
    note right of Initialised
        Before begin, the engine builds the manifest (engine.rs:700), the night's one protected
        policy (engine.rs:743), freezes the manifest (engine.rs:763) and probes the annexe
        (engine.rs:823). An unhealthy annexe is BLOCKED-ENV and costs no attempt.
    end note
    Initialised --> Abandoned : attempt budget exhausted
    Initialised --> ManifestFrozen
    ManifestFrozen --> BaselineEvaluated
    BaselineEvaluated --> ModelCalled
    ModelCalled --> CandidateEvaluated
    note right of CandidateEvaluated
        candidate_flow runs every patch through dream_core::candidate::run, which refuses,
        then guards, then builds the worktree, then evaluates (dream-core/src/candidate.rs:469).
    end note
    CandidateEvaluated --> Gated
    Gated --> Persisted
    note right of Persisted
        runstate::persist (dream-core/src/runstate.rs:615) is entered BEFORE the forge
        request and the ledger commit (agentbox/services/dream-engine/src/engine.rs:1710),
        accepted or not. A crash after this point can lose a publish, never repeat one.
    end note
    Persisted --> Complete
    Complete --> [*]
    Abandoned --> [*]
    Handoff --> [*]
    Skipped --> [*]
    note right of Abandoned
        Phase::Abandoned (dream-core/src/runstate.rs:76) is terminal and never silently retried,
        the operator is alerted (agentbox/services/dream-engine/src/engine.rs:890).
    end note
```

**What it shows.** The states one repo-night can pass through, now including the per-night lock that comes first and the `Persisted` phase that sits between the gate and `Complete`. Every night that reaches the gate passes through `Persisted`, whatever the verdict, because the journal write at `engine.rs:1710` precedes the accepted-or-not match.

**Why it is this way.** ADR-2141 and the core repository's ADR-DE-007 (`../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md:21`) make the night, not the run id, the unit of "finished": the run id folds in the baseline commit, which the engine's own ledger commit moves.

**Open:** admission (`../project/agentbox/services/dream-engine/src/engine.rs:576`) runs before the finished-night check (`../project/agentbox/services/dream-engine/src/engine.rs:641`), and a refusal writes a HANDOFF ledger row (`../project/agentbox/services/dream-engine/src/records.rs:282`). A night that already finished, whose evaluator configuration was then edited into an unusable state, would therefore be re-judged at admission rather than skipped. The record does not say whether that ordering is intended.

## AB-23.2 Nightly cycle — the adapter's night, in order

```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord<br/>agentbox/flake.nix:2896
    participant ENG as Engine::run_night<br/>agentbox/services/dream-engine/src/engine.rs:154
    participant GOV as governance<br/>agentbox/services/dream-engine/src/governance.rs:1058
    participant ROS as roster::select<br/>dream-core/src/roster.rs:67
    participant LK as NightLock<br/>dream-core/src/runstate.rs:295
    participant MAN as manifest::freeze<br/>dream-gate/src/manifest.rs:297
    participant HP as connected-node annexe<br/>agentbox/agentbox.toml:2570
    participant LLM as call<br/>agentbox/services/dream-engine/src/llm.rs:49
    participant JUD as judge::judge<br/>agentbox/services/dream-engine/src/judge.rs:36
    participant FG as GhForge::publish_draft<br/>dream-core/src/seams.rs:425
    participant LED as ledger::append_row<br/>dream-core/src/ledger.rs:350

    SUP->>ENG: dream-engine --loop --agentbox-toml /etc/agentbox.toml
    Note over SUP,ENG: autostart=true autorestart=true priority=230 user=devuser (agentbox/flake.nix:2896-2903)
    alt [dream_machine] enabled = false (agentbox/agentbox.toml:2569)
        ENG-->>SUP: byte-identical-when-off, no supervisor block is generated at all
    else enabled
        loop nightly window, at most one cycle per UTC day (agentbox/services/dream-engine/src/main.rs:284-298)
            ENG->>ENG: dream-paused flag check, the night is NOT consumed if paused (engine.rs:156-161)
            ENG->>GOV: ingest(inbox_path, false), carry forum decisions into tonight (engine.rs:180)
            opt DREAM_SWEEP is not 0
                ENG->>ENG: sweep_branches, the seven-day rule for dream/* branches (engine.rs:200)
            end
            loop each nominated repo (engine.rs:203-232)
                alt .dream-standby marker present
                    ENG->>ENG: standby{repo, reason marker, streak 0}
                else dry streak at or above prune_dry_streak 5 (agentbox.toml:2616)
                    ENG->>ENG: standby{repo, reason dry-streak} from streak::dry_streak_text (engine.rs:473-478)
                    Note over ENG: REJECT and INCONCLUSIVE advance the streak, ACCEPT resets it, BLOCKED-ENV and<br/>HANDOFF leave it (dream-core/src/streak.rs:151-157). RESOLVED 2026-10-15: the old engine reset on REJECT
                else eligible
                    ENG->>ENG: push to the eligible list
                end
            end
            ENG->>ROS: roster::load then select(eligible, max_repos_per_night), least recently dreamed first (engine.rs:254-263)
            loop each selected repo (engine.rs:288-297)
                ENG->>LK: NightLock::acquire_within(night_dir, 250 ms) (engine.rs:563)
                alt lock held elsewhere
                    LK-->>ENG: NightLocked, skipped, no outcome recorded (engine.rs:295)
                else lock taken
                    ENG->>ENG: readiness::assess then prompt::compile then finished_night (engine.rs:576, 609, 641)
                    ENG->>MAN: manifest::build (engine.rs:700), policy::resolved_extras and for_night (engine.rs:724, 743), then freeze (engine.rs:763)
                    Note over MAN: INVARIANT: the manifest is frozen before any model call, and it carries the config digest of the exact<br/>bytes the protected-path policy was built from (engine.rs:700-708, dream-gate/src/manifest.rs:341)
                    ENG->>HP: annexe_health (engine.rs:823), then runstate::begin (engine.rs:867), see AB-23.18
                    ENG->>HP: clone via git archive, run the REQUIRED evaluators on the BASELINE tree
                    ENG->>ENG: tree-read honours the previous night's source requests first, then the slot's path-shaped scan entries (engine.rs:1254, 1266), see AB-23.21
                    ENG->>LLM: call the model with the report prompt
                    Note over ENG,LLM: RESOLVED ADR-2053: the default reasoning provider is Z.AI (agentbox.toml:2585), a deliberate operator<br/>choice, and nightly repository content leaves the LAN unless llm_provider = loom
                    LLM-->>ENG: report text with a VERDICT line and an optional dream-patch
                    opt an ACCEPT is claimed and AGENTBOX_JEV_DREAM_TRIAGE=1 (off by default)
                        ENG->>ENG: jev_triage::consider annotates the hypothesis, and defers the candidate build only below an explicit threshold (engine.rs:1439-1463), see AB-23.22
                    end
                    ENG->>ENG: candidate_flow::attempt_judged (engine.rs:1499), see AB-23.4 and AB-23.5
                    ENG->>JUD: judge(night_dir, NightInputs) (engine.rs:1561)
                    Note over JUD: every night that reaches the gate is judged AND recorded by replay::capture<br/>(dream-core/src/replay.rs:332), so replay verify can recompute the verdict. There is no second<br/>way to judge in the adapter (agentbox/services/dream-engine/src/judge.rs:11-12)
                    JUD-->>ENG: GateDecision {accepted, verdict, vetoes}
                    ENG->>ENG: record_streak (engine.rs:1602), inbox alerts for questions and vetoes, source-supply asks routed to the next night's tree-read (engine.rs:1648-1704)
                    ENG->>LK: runstate::persist, Phase::Persisted (engine.rs:1710)
                    Note over ENG,LK: ORDER: persist, then the forge request, then the ledger commit. A crash from here on<br/>can lose a publish but never repeat one (engine.rs:1706-1709)
                    ENG->>FG: publish_draft, only when the GATE accepted and persist_accepts (engine.rs:1734, 1774)
                    ENG->>LED: append_and_commit_ledger (engine.rs:1857), the append is fatal, the commit is fail-open
                    ENG->>ENG: RuVectorSink record, fail-open (engine.rs:1891)
                    ENG->>HP: remove the night's remote directory (fail-open)
                    ENG->>LK: runstate::complete (engine.rs:1932)
                end
                ENG->>ROS: roster.record(name, date, verdict_label) (engine.rs:308)
            end
            ENG->>ENG: build NightHealth and write dream-last-night.json (engine.rs:336-361, see AB-23.8)
            ENG->>GOV: publish(inbox_path, CasePaths, false), see AB-23.17 (engine.rs:395-399)
            ENG->>ENG: digest::run when DREAM_DIGEST is not 0 (engine.rs:403-405)
        end
    end
    Note over ENG: ADR-2071 Phase 1 LANDED, partial. The night is journalled through POST /v1/exec/record, not policed,<br/>and the policing half stays open. Every journalled side effect fails open (agentbox/services/dream-engine/src/journal.rs:30)
```

**What it shows.** The adapter's night, with the core's calls named where they happen. The order inside a repo-night is lock, admission, prompt, finished-night check, manifest, policy, freeze, annexe probe, journal begin, baseline, model, candidate, judge, persist, forge, ledger, memory, complete.

**Why it is this way.** Each step that moved into the core is still driven from `engine.rs`, because the composition root is the only place that knows the estate's seams. The judge call replaces the pre-extraction direct call of `gate::decide`, so that capture and verdict cannot diverge. The terminal, persist-before-effect ordering is ADR-2141's restart-safety consequence (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:39`).

**Drift:** `../project/agentbox/docs/GOVERNANCE-capabilities.md:513` (was :520 at the previous stamp) still says ADR-2071 is proposed, not landed, while the journal ships Phase 1 (`../project/agentbox/services/dream-engine/src/journal.rs:1-3`).

**Tension:** the persist-before-effect order trades a lost publish for never publishing twice. Nothing records the forge or ledger outcome after `Persisted`, so a crash between the journal write and the pull request leaves a night marked finished with no draft (`../dream-engine/crates/dream-core/src/runstate.rs:22-28`, `../dream-engine/docs/adr/ADR-DE-007-a-night-is-identified-by-its-directory.md:34`). Recovery is the operator's `dream-engine reset-night <repo> <date>` (`../project/agentbox/services/dream-engine/src/main.rs:142-144`).

## AB-23.3 Module map and the licence boundary

```mermaid
flowchart TB
    subgraph adapter["agentbox/services/dream-engine, AGPL-3.0-only (the adapter)"]
        direction TB
        E1["engine.rs, the composition root<br/>agentbox/services/dream-engine/src/engine.rs:95"]
        E2["estate.rs, every estate default as an explicit parameter<br/>agentbox/services/dream-engine/src/estate.rs:21"]
        E3["candidate_flow.rs, the ONLY caller of candidate::run<br/>agentbox/services/dream-engine/src/candidate_flow.rs:100"]
        E4["policy.rs, the night's single protected-path policy<br/>agentbox/services/dream-engine/src/policy.rs:107"]
        E5["judge.rs, the one way to judge a night<br/>agentbox/services/dream-engine/src/judge.rs:36"]
        E6["llm.rs, the Z.AI and Loom clients as the Llm seam<br/>agentbox/services/dream-engine/src/llm.rs:49"]
        E7["runner.rs SshRunner, dispatch.rs, ruvector.rs, notify.rs<br/>agentbox/services/dream-engine/src/runner.rs:34"]
        E8["governance.rs, relay.rs, inbox.rs, digest.rs, zone_crypto.rs<br/>the forum drivers, linked to nostr-bbs-core"]
        E9["journal.rs, prompt.rs, records.rs, present.rs, baseline.rs, config.rs"]
        E10["wanted.rs, the source-request store, and jev_triage.rs, the opt-in hypothesis annotation<br/>agentbox/services/dream-engine/src/wanted.rs:1, agentbox/services/dream-engine/src/jev_triage.rs:1"]
    end
    subgraph core["DreamLab-AI/dream-engine, MIT OR Apache-2.0 (public, five crates, dream-core 0.4.0 and dream-gate 0.3.0)"]
        direction TB
        C1["dream-core: runner, ledger, readiness, runstate and NightLock, roster, sweep, compile,<br/>source, context, candidate, persist, seams, testfmt, streak, replay, automerge, screen<br/>dream-core/src/lib.rs:1"]
        C2["dream-gate: verdict, manifest, receipts, witness, gate, guard, patch, outcome, config schema<br/>dream-gate/src/lib.rs:1"]
        C3["pairwise-gate and failure-fingerprint, leaf crates<br/>pairwise-gate/src/lib.rs:1"]
        C4["dream-cli, a thin shell over the core<br/>dream-cli/src/lib.rs:1"]
    end
    E1 -->|"registry dependency, exact pin"| C1
    E1 -->|"registry dependency, exact pin"| C2
    C1 --> C2
    C2 --> C3
    C4 --> C1
    E6 -.->|"implements the Llm seam"| C1
    E7 -.->|"implements EvaluatorRunner, MemorySink and Notifier"| C1
    BARRIER["INVARIANT: the adapter depends on the core and never the reverse. zone_crypto is never published<br/>and the adapter links nostr-bbs-core, which is why it stays AGPL-3.0-only<br/>agentbox/services/dream-engine/Cargo.toml:8"]
    E8 --- BARRIER
    STAMP["RESOLVED: the extraction is live (ADR-2141 activation live), agentbox stamped at 1da8d2171,<br/>dream-engine at 61581d3 whose workspace version is 0.4.0"]
    BARRIER --- STAMP
```

**What it shows.** Which modules stayed in the adapter and which moved. The line between them follows the licence: anything that links `nostr-bbs-core` or holds the private zone envelope stays under AGPL-3.0-only, and everything else is permissive and reusable by any repository.

**Why it is this way.** ADR-2141 chose the boundary so that the gate and the loop can be reviewed and versioned without the estate. The seams (`Llm`, `EvaluatorRunner`, `MemorySink`, `Notifier`, `Forge`) are implemented in the adapter and defined in `dream-core` (CRT-02 in the core corpus).

**Resolved (pin):** the two core crates are registry dependencies pinned `dream-core =0.4.0` and `dream-gate =0.3.0` (`../project/agentbox/services/dream-engine/Cargo.toml:26-27`), the lockfile carries their checksums, and ADR-2141 records that `lib/dream-engine.nix` needs no `outputHashes` because every dependency is a registry crate (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:31`). The image rebuild that bakes them in was done on 2026-10-16 (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:47`).

**Invariant:** both pins must stay on one `dream-gate` release, because a mismatch links two copies of `dream-gate` whose types would not unify (`../project/agentbox/services/dream-engine/Cargo.toml:22-27`).

**Drift:** the record's Pinning bullet still names `=0.2.0` for both crates (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:31`) and its last re-verification stops at `=0.3.0` (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:55`), while the manifest pins `dream-core =0.4.0` (`../project/agentbox/services/dream-engine/Cargo.toml:26`); no activation receipt for the 0.4.0 image is in the record.

**Resolved (CI):** the agentbox GitHub Actions run again, with CI, ShellCheck and Secret Scan green on 2026-10-29 (`gh run list`); the earlier gap since 2026-10-08 is closed.

## AB-23.4 The guard ordering and the deterministic gate

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:1499
    participant CF as candidate_flow<br/>agentbox/services/dream-engine/src/candidate_flow.rs:100
    participant RUN as candidate::run<br/>dream-core/src/candidate.rs:469
    participant GRD as guard::check<br/>dream-gate/src/guard.rs:1168
    participant J as judge::judge<br/>agentbox/services/dream-engine/src/judge.rs:36
    participant G as gate::decide<br/>dream-gate/src/gate.rs:381

    ENG->>CF: attempt_judged(flow, patch, journal)
    CF->>RUN: run(input, runner, relocate), the only route to git
    RUN->>RUN: deletes_binary(patch) refused before any git command (candidate.rs:481)
    RUN->>GRD: guard::check(patch, policy), BEFORE any worktree, build or evaluator (candidate.rs:489)
    alt protected path or an unparseable diff
        GRD-->>RUN: GuardViolation
        RUN-->>CF: CandidateState::Protected (REJECT) or Refused (INCONCLUSIVE), nothing ran
    else clean
        RUN->>RUN: prepare, the worktree is built and the patch applied (candidate.rs:494)
        Note over RUN: prepare re-checks the NORMALISED text it actually applies (dream-core/src/candidate.rs:503)
        opt a required evaluator names a fixer (ADR-DE-009)
            RUN->>RUN: format_candidate re-guards the patch, runs the fixer on the candidate's own touched files and commits the result as a second commit before the tree is hashed (candidate.rs:527)
        end
        RUN->>RUN: required evaluators re-run on the candidate tree
        RUN-->>CF: Attempt with state Applied, receipts and the tree
    end
    CF-->>ENG: the Attempt, and the patch text that was judged
    ENG->>J: judge(night_dir, NightInputs incl. candidate config digest)
    J->>G: replay::capture then gate::decide(manifest, strict, candidate, receipts, ctx)
    G->>GRD: guard::check AGAIN on an applied candidate's patch (dream-gate/src/gate.rs:420)
    alt the second check fails
        G->>G: tree not trusted, receipts not graded, Veto patch-guard (gate.rs:421-422)
    end
    alt tree trusted and ACCEPT claimed
        G->>G: candidate config digest must equal the manifest's, else Veto config-digest, harness class (gate.rs:471-481)
    end
    G-->>J: accepted only if ACCEPT was claimed, the patch applied and EVERY required evaluator Passed
    Note over G: harness-class vetoes give BLOCKED-ENV, evidence-class REJECT, unproven-only INCONCLUSIVE.<br/>A refused patch skips the receipt step entirely, so a guard veto cannot be diluted into BLOCKED-ENV (gate.rs:496-501)
```

**What it shows.** Where the protected-path guard sits: once inside `candidate::run` before anything is applied, and once more inside `gate::decide`, which re-derives the touched paths from the patch text rather than trusting the caller. The config-digest check binds the environment the candidate was evaluated in to the one the manifest froze.

**Why it is this way.** The old engine had no guard before apply (ADR-2141, Context). Making `candidate_flow` the only caller of `candidate::run` means the apply-repair pass is guarded too, because its corrected diff re-enters `run` (`../project/agentbox/services/dream-engine/src/candidate_flow.rs:5-9`). The second check is defence in depth, owned by the core (GAT-01).

**Open:** the formatter step is inert at this stamp, because it runs only when a required evaluator names a `fixer` and no evaluator in the repository's own `dream.config.json` does (`../dream-engine/crates/dream-core/src/candidate.rs:527`, `../project/agentbox/dream.config.json:43`); `judge.rs` asserts the adapter itself holds no `rustfmt` or `cargo fmt`, so the mechanism stays in the core (`../project/agentbox/services/dream-engine/src/judge.rs:303`).

**Invariant:** no path in the adapter builds a gate context without the night's one policy, and no path calls `gate::decide` directly (`../project/agentbox/services/dream-engine/src/judge.rs:11-12`, `../project/agentbox/services/dream-engine/src/policy.rs:5-9`).

**Invariant:** evaluator failure vetoes acceptance, and an ACCEPT with no applied candidate is unproven, not a harness fault (`../dream-engine/crates/dream-gate/src/gate.rs:461`, see AB-23.13).

## AB-23.5 Candidate re-evaluation on an isolated worktree

```mermaid
sequenceDiagram
    autonumber
    participant CF as candidate_flow::run_once<br/>agentbox/services/dream-engine/src/candidate_flow.rs:100
    participant RUN as candidate::run<br/>dream-core/src/candidate.rs:469
    participant PRE as candidate::prepare<br/>dream-core/src/candidate.rs:301
    participant BW as persist::build_branch_worktree<br/>dream-core/src/persist.rs:131
    participant SRC as source::apply_repair<br/>dream-core/src/source.rs:1371
    participant EV as candidate::evaluate<br/>dream-core/src/candidate.rs:343

    CF->>RUN: run(input, runner, relocate)
    RUN->>PRE: prepare(repo, branch, patch, msg, base, policy)
    Note over PRE,BW: ADR-2114: the base is the DISPATCHED baseline revision, not always HEAD, so the diff lands on exactly<br/>the tree the model was shown even if the operator commits while the night runs
    PRE->>BW: build_branch_worktree, as-written strategies first (persist.rs:86), then a normalised retry (dream-gate/src/patch.rs:73)
    alt the diff does not land
        BW-->>RUN: PatchDidNotApply with git's stderr
        RUN-->>CF: Attempt{state DidNotApply, error PatchDidNotApply}
        CF->>SRC: ONE apply-repair call per night, keyed on the same failed patch digest (candidate_flow.rs:186-206)
        SRC-->>CF: a corrected patch, or nothing
        CF->>RUN: run again with the corrected diff, so it is GUARDED AGAIN (candidate_flow.rs:5-9)
    else lands
        BW-->>PRE: worktree, applied_via
        PRE-->>RUN: PreparedCandidate with the tree hash
        RUN->>EV: evaluate(runner, work_dir, required)
        EV-->>RUN: one receipt per required evaluator
    end
    RUN-->>CF: Attempt
    Note over RUN: INVARIANT: the operator's working tree is NEVER touched, the worktree is a separate checkout at the base<br/>revision (dream-core/src/candidate.rs:7)
    Note over CF: the receipt apply-repair.json is reused on restart ONLY for the same failed patch (candidate_flow.rs:186-190)
```

**What it shows.** The adapter side of the candidate step and the core calls beneath it. The apply strategies, the normalisation rule that never edits a plus or minus line, and the typed `AttemptError` are all in the core and are drawn in NIT-01 and GAT-01 there.

**Why it is this way.** ADR-2114 (candidate diffs are written against engine-read source) fixed the base revision, and ADR-2141 collapsed the engine's four separate call sites into `candidate::run` so that the guard cannot be skipped.

## AB-23.6 Persistence and the human merge gate

```mermaid
sequenceDiagram
    autonumber
    participant G as judge::judge<br/>agentbox/services/dream-engine/src/judge.rs:36
    participant ENG as Engine persist arm<br/>agentbox/services/dream-engine/src/engine.rs:1710
    participant RS as runstate::persist<br/>dream-core/src/runstate.rs:615
    participant FG as Forge::publish_draft<br/>dream-core/src/seams.rs:425
    participant LED as append_and_commit_ledger<br/>agentbox/services/dream-engine/src/engine.rs:1857
    participant NT as notify::lost_pr<br/>agentbox/services/dream-engine/src/notify.rs:28
    participant H as Human reviewer
    participant CFG as dream.config.json<br/>agentbox/dream.config.json

    G-->>ENG: GateDecision
    ENG->>RS: persist(night_dir, run, verdict), Phase::Persisted (engine.rs:1710)
    Note over ENG,RS: BEFORE any externally visible effect. If the journal cannot be written the engine refuses to publish<br/>and the night is BLOCKED-ENV (engine.rs:1710-1725)
    alt not accepted
        ENG->>ENG: candidate::discard deletes the branch and worktree, the ledger row says VETOED (engine.rs:1821-1831)
    else accepted but persist_accepts is false
        ENG->>ENG: branch kept locally only, no push (engine.rs:1816-1819)
    else accepted and persist_accepts (engine.rs:1734)
        ENG->>FG: publish_draft(DraftPr), git push then gh pr create --draft (engine.rs:1774, dream-core/src/seams.rs:425)
        FG-->>H: a PULL REQUEST, never a merge
        opt pr_url is None
            ENG->>NT: lost_pr alert naming the kept local branch (engine.rs:1791)
        end
        ENG->>LED: append the row then commit THAT file only (engine.rs:1857, dream-core/src/ledger.rs:411)
        Note over LED: git commit --only on the default branch, local only, never pushed (dream-core/src/ledger.rs:408-411).<br/>The append stays fatal, the commit is fail-open
        Note over CFG: autoMerge is false (agentbox/dream.config.json:80), labels at :76-79
        H->>H: review and merge, or not
    end
    Note over CFG: extraDisciplines also fence the self-referential case: a change to services/dream-engine changes the dreamer,<br/>so it is never let past the human-merge gate (dream.config.json:67), the image build is HANDOFF only (dream.config.json:68),<br/>secrets are never quoted (dream.config.json:70), and only ADR ids that exist may be cited (dream.config.json:71)
```

**What it shows.** The persistence order after the gate has ruled: journal, then the forge request, then the ledger, with the human still deciding the merge.

**Why it is this way.** Persist-before-effect is the at-most-once guarantee (ADR-DE-007). The merge gate stays a process, not a code control, because `autoMerge` is false and, as below, is not consulted by the adapter at all.

**Open:** `autoMerge` is read from the repository's `dream.config.json` and defaults to off (`../dream-engine/crates/dream-gate/src/config.rs:437-442`), and the core ships a pure decision tree plus a `persist::publish` that applies it (`../dream-engine/crates/dream-core/src/persist.rs:291`, `../dream-engine/crates/dream-core/src/automerge.rs:110`). The adapter calls `Forge::publish_draft` directly (`../project/agentbox/services/dream-engine/src/engine.rs:1774`) and never `persist::publish`, so setting `autoMerge: true` would have no effect here: the option is inert in the adapter. The core now says so itself, that only `persist::publish` applies the policy and a host publishing through `Forge::publish_draft` never merges whatever `autoMerge` says (`../dream-engine/README.md:130-135`, `../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md:44-48`), and that the first consuming host is exactly that case. The forge calls behind the option have not been exercised against real GitHub (`../dream-engine/docs/adr/ADR-DE-006-automerge-is-an-option.md:41-42`). See ES-05.

**Tension (dream.config.json vs the prompt):** RESOLVED. The previous stamp recorded that every night's prompt cited "agentbox ADR-2024" while the discipline `cite-existing-adrs` listed ADR-2024 as non-existent (`../project/agentbox/dream.config.json:71`). The estate tag is now stripped from the shared compiler's output: the prompt test removes the string from the pre-extraction golden before comparing (`../project/agentbox/services/dream-engine/src/prompt.rs:82`).

**Drift:** the pull request body gains a note when a fixer reformatted the candidate, so a reviewer reads the logic as the model's and the layout as the formatter's (`../project/agentbox/services/dream-engine/src/engine.rs:1753-1760`); the option is inert here for the reason recorded in AB-23.4.

**Debt:** RETIRED. `persist::persist_accept` and `push_and_open_pr`, recorded as dead code at the previous stamp, no longer exist in either repository.

**Tension (forge step):** the forge request is the one effect of a night the engine cannot undo and, per the core, cannot reconcile after a crash. Its only test double is the `Forge` trait object the engine accepts (`../project/agentbox/services/dream-engine/src/engine.rs:112`); the real `GhForge` shells out to `git push` and `gh` (`../dream-engine/crates/dream-core/src/seams.rs:424-440`) and no record at these revisions shows a full night publishing a pull request through the extracted path.

## AB-23.7 Run journal — restart, resume, abandonment and the finished-night rule

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:532
    participant LK as NightLock<br/>dream-core/src/runstate.rs:295
    participant RS as runstate<br/>dream-core/src/runstate.rs:511
    participant F as run-state.json<br/>night directory
    participant MAN as manifest<br/>dream-gate/src/manifest.rs:258

    ENG->>LK: acquire_within(night_dir, 250 ms), before the first journal read (engine.rs:563)
    alt held by another engine
        LK-->>ENG: LockError::Held, the night is skipped (engine.rs:567)
    else taken
        ENG->>RS: finished_night(night_dir, repo, date, night_id) (engine.rs:641)
        RS->>F: load_checked (runstate.rs:337)
        alt journal unreadable, corrupt or of an unknown schema
            F-->>RS: JournalError
            RS-->>ENG: Err, the engine reports BLOCKED-ENV and points at reset-night (engine.rs:642-647)
        else phase Persisted or Complete for this night
            RS-->>ENG: Some(state), a Persisted record is finalised to Complete (runstate.rs:415-421)
            ENG->>ENG: finished_result, no model call, no manifest, no commit, no forge request (engine.rs:649-661)
        else not finished
            RS-->>ENG: None
            ENG->>RS: begin(night_dir, run_id, ...) (engine.rs:867)
            alt a prior run for this night
                RS-->>ENG: Resume::Resumed, the attempt is counted
            else attempt budget exhausted
                RS-->>ENG: Resume::Abandoned, an operator alert and BLOCKED-ENV (engine.rs:888-912)
            else first attempt
                RS-->>ENG: Resume::Fresh
            end
        end
    end
    ENG->>MAN: run_id(repo, date, deep, baseline, config_digest) (manifest.rs:258)
    Note over MAN: the run id folds in the baseline HEAD, which the engine's own ledger commit moves, so it cannot recognise<br/>a finished night. The attempt budget belongs to the NIGHT, not the run id (dream-core/src/runstate.rs:497-500)
    ENG->>RS: advance after each phase, then persist before effects, then complete (engine.rs:934, 1710, 1932)
```

**What it shows.** The restart rules after the extraction. A night is identified by its directory, so a restart after its own ledger commit cannot start a second experiment, and a damaged journal is a fault, never a fresh start. An operator reopens a night only with `dream-engine reset-night`.

**Why it is this way.** Before 2026-10-15 the run id was the only identity, and the engine's ledger commit moved HEAD and therefore the run id, so a restart could repeat a publish. ADR-DE-007 is the repair. The adapter changes are the lock and the early skip at `engine.rs:563-661`.

**Invariant:** `Persisted` is journalled before the forge request and the ledger commit (`../dream-engine/crates/dream-core/src/runstate.rs:22-23`, `../project/agentbox/services/dream-engine/src/engine.rs:1710`).

## AB-23.8 Core types

```mermaid
classDiagram
    class RunState {
        +String run_id
        +String night_id
        +Phase phase
        +u32 attempts
        +u32 max_attempts
        +Option~String~ verdict
    }
    class Phase {
        <<enum>>
        Initialised
        ManifestFrozen
        BaselineEvaluated
        ModelCalled
        CandidateEvaluated
        Gated
        Persisted
        Complete
        Abandoned
    }
    class Resume {
        <<enum>>
        Fresh
        Resumed
        AlreadyComplete
        Abandoned
        +should_run() bool
    }
    class NightLock {
        +acquire(dir)
        +acquire_within(dir, wait)
    }
    class CandidateState {
        <<enum>>
        NoPatch
        NotAttempted
        Applied
        Protected
        Refused
        DidNotApply
    }
    class GateDecision {
        +bool accepted
        +String verdict
        +String model_verdict
        +Vec~Veto~ vetoes
        +verdict_enum() Verdict
    }
    class VetoClass {
        <<enum>>
        Harness
        Evidence
        Unproven
    }
    class Verdict {
        <<enum>>
        Accept
        Reject
        Inconclusive
        BlockedEnv
        Handoff
    }
    class NightHealth {
        +String date
        +Vec~Outcome~ outcomes
        +Vec~String~ nominated
        +Vec~Standby~ standby
        +Vec~String~ deferred
        +Vec~JournalStats~ journal
        +Vec~SweepRecord~ sweep
    }
    RunState --> Phase
    Resume --> RunState
    GateDecision --> VetoClass
    GateDecision ..> Verdict : verdict_enum
    note for Phase "dream-core/src/runstate.rs:56, Persisted is entered before the forge request"
    note for CandidateState "dream-gate/src/gate.rs:255, Protected and Refused are the guard outcomes"
    note for VetoClass "dream-gate/src/gate.rs:42, Harness = could not gather evidence,<br/>Evidence = gathered and against, Unproven = nothing to test"
    note for NightHealth "agentbox/services/dream-engine/src/digest.rs:74, the ADAPTER's own record<br/>written to dream-last-night.json (engine.rs:336)"
```

**What it shows.** The types that cross the boundary. `Phase`, `Resume` and `NightLock` live in `dream-core`, the gate types in `dream-gate`, and `NightHealth` is the adapter's.

**Why it is this way.** The `Unusable` readiness variants and `ExperimentManifest` are omitted here: they are drawn in the core corpus. The `Persisted` phase and `NightLock` are new since the previous stamp, and `CandidateState` gained `Protected` for a guard refusal.

## AB-23.9 Dream-inbox surfacing hook

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator turn<br/>any Claude session
    participant CC as Claude Code UserPromptSubmit
    participant HK as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:1
    participant INBOX as dream-inbox.json<br/>/home/devuser/workspace/.agentbox/dream-inbox.json
    participant STAMP as dream-inbox.json.surfaced
    participant EP as entrypoint registration<br/>agentbox/config/entrypoint-unified.sh:2208

    Note over EP: the entrypoint prefers /opt/agentbox/config/hooks/dream-inbox-surface.cjs and falls back<br/>to the repo path (:2208-2209), then dedupes on the command substring (:2221)
    U->>CC: submits a prompt
    CC->>HK: UserPromptSubmit with stdin JSON
    alt prompt matches a harness-generated turn (task-notification, agent-message, system-reminder)
        HK-->>CC: exit — no injection, waits for a turn the operator actually typed
    else
        HK->>INBOX: readFileSync
        alt file missing, unparseable, not an array, or zero status=="open" items
            HK-->>CC: exit — fail-open, no injection
        else at least one open item
            HK->>STAMP: read last stamp, skip unless now - last > RESURFACE_HOURS 4 * 3600
            alt too soon
                HK-->>CC: exit — no injection
            else due
                HK-->>CC: inject ONE pointer line — count of open decisions and the forum governance panel URL<br/>(no item bodies, no /dream answer instruction)
                HK->>STAMP: write now, only after the pointer was actually written out
            end
        end
    end
    Note over HK: ADR-2115: the engine no longer relays item bodies into sessions — every open question and<br/>alert is a case on the forum governance panel, which is where the operator now decides —<br/>this hook only says how many are waiting and where
    Note over HK: rate limiting is now GLOBAL (one stamp file, not per item) and fail-open on any error
```

## AB-23.10 Control and reporting surfaces

```mermaid
flowchart TB
    subgraph skill["/dream control skill — agentbox/skills/dream-machine/commands/dream.md"]
        S1["/dream status (or no argument) — skills/dream-machine/commands/dream.md:7"]
        S2["/dream questions · /dream answer id text · /dream dismiss id — skills/dream-machine/commands/dream.md:17"]
        S3["/dream harvest [--days N] — skills/dream-machine/commands/dream.md:31"]
        S4["/dream off · /dream on — skills/dream-machine/commands/dream.md:41"]
        S5["/dream run [repo] — skills/dream-machine/commands/dream.md:46"]
        S6["/dream standby repo · /dream revive repo — skills/dream-machine/commands/dream.md:61"]
        S7["/dream digest [date] — skills/dream-machine/commands/dream.md:66"]
        S8["/dream nominate repo — skills/dream-machine/commands/dream.md:74"]
    end
    subgraph scripts["Scripts — agentbox/scripts/"]
        N1["dream-machine-nightly.mjs"]
        N2["dream-inbox.mjs — BREAK-GLASS ONLY, the forum panel is canonical"]
        N3["dream-harvest.mjs"]
        N4["dream-forum-suggestions.mjs — separate tenant, forum-suggestions triage"]
        N6["dream-hooks-syntax.sh — an evaluatorEntrypoint, not a control surface"]
    end
    subgraph cli["dream-engine subcommands — main.rs"]
        C1["dream-engine digest #91;--date D#93; #91;--dry-run#93;<br/>Cmd::Digest main.rs:72 -&gt; digest::run main.rs:155-157"]
        C2["dream-engine governance publish, ingest or withdraw #91;--dry-run#93;<br/>Cmd::Governance main.rs:67 -&gt; governance::publish / ingest / withdraw<br/>main.rs:117-131 — publish builds CasePaths from the CLI workspace and artefact dir, main.rs:120-123, see AB-23.17"]
        C3["dream-engine reset-night repo date — archives a finished night's run-state and manifest<br/>Cmd::ResetNight main.rs:83 -&gt; dream_core::runstate::reset_night main.rs:142-144, see AB-23.7"]
    end
    subgraph api["management-api"]
        R1["GET /dream/status (fastify)<br/>agentbox/management-api/routes/dream.js:24"]
        L1["dream-ledger.js parseLedger management-api/lib/dream-ledger.js:102 · verdictStats management-api/lib/dream-ledger.js:135 · latestNights management-api/lib/dream-ledger.js:146<br/>discoverNominatedRepos management-api/lib/dream-ledger.js:248 · pendingMerges management-api/lib/dream-ledger.js:333<br/>readRepoDreamStatus management-api/lib/dream-ledger.js:346 · aggregateDreamStatus management-api/lib/dream-ledger.js:397"]
    end
    subgraph out["Outputs"]
        O1["docs/dream-cycle/LEDGER.md<br/>ledgerPath, agentbox/dream.config.json:74"]
        O2["docs/dream-cycle/FORUM-SUGGESTIONS.md"]
        O3["dream-inbox.json — see AB-23.9"]
        O4["voice/console/site/dream.html"]
        O5["forum zone4-chat-with-agents digest post<br/>+ Dream machine decisions panel, see AB-23.17"]
    end
    S1 --> R1
    R1 --> L1
    L1 --> O1
    S3 --> N3
    S2 --> N2
    S2 --> C2
    N2 --> O3
    S7 --> C1
    C1 --> O5
    C2 --> O5
    S5 --> N1
    N4 --> O2
    L1 --> O4
    subgraph notes["Invariants and drift"]
        direction TB
        ND1["DIVERGENCE: routes/dream.js exposes exactly ONE endpoint, GET /dream/status. There is no<br/>HTTP control surface for run/nominate/answer — those are skill-plus-script paths only"]
        ND2["RESOLVED ADR-2115: dream-night-digest.mjs is DELETED — /dream digest now calls the<br/>dream-engine binary's own digest subcommand. The 056/058/061-072 legacy governance band<br/>is superseded by the forum panel; dream.html cockpit status is unverified"]
        ND3["Both repo roots carry a dream.config.json — agentbox/dream.config.json (the agentbox<br/>nomination) and the VisionClaw root dream.config.json. Each nominated repo declares its<br/>own evaluatorEntrypoints and extraDisciplines"]
        ND1 ~~~ ND2 ~~~ ND3
    end
```

**Invariant (2026-10-02, owner decision Q10):** the forum-suggestions tenant asks a member to clarify only when JunkieJarvis may speak; it reads the same `junkiejarvisEnabled(manifest)` that management-api uses, and with it off an unclear post is held — no DM, no ledger row, not parked — so it is asked once JunkieJarvis is on (`../project/agentbox/scripts/dream-forum-suggestions.mjs:246-252`, `../project/agentbox/scripts/dream-forum-suggestions.mjs:421-424`).

## AB-23.11 ADR-2081 — the annexe mirrors real workspace depth

```mermaid
flowchart TB
    subgraph before["Before 2026-09-07: one level too shallow"]
        B1["remote_dir/agentbox (flat repo_name)"]
        B2["cargo path-dep ../../../../nostr-rust-forum<br/>climbs to ONE ABOVE remote_dir"]
        B3["manifest load fails, sovereign-mesh-bridge<br/>REQUIRED gate red every night"]
        B1 --> B2 --> B3
    end
    subgraph after["annexe_subpath and clone_repo_and_siblings, now in dispatch.rs"]
        A1["annexe_subpath(repo_path, workspace_root)<br/>canonicalizes both, strip_prefix, joins components<br/>agentbox/services/dream-engine/src/dispatch.rs:295"]
        A2["target ships at remote_dir/project/agentbox<br/>(its REAL path under the workspace, not the leaf name)<br/>called at agentbox/services/dream-engine/src/engine.rs:947"]
        A3["each annexe_include sibling ships at remote_dir/its own subpath<br/>e.g. remote_dir/nostr-rust-forum, dispatch.rs:276-285"]
        A4["cargo ../../../../nostr-rust-forum now climbs to<br/>remote_dir exactly as it climbs to the workspace root locally"]
        A1 --> A2
        A1 --> A3
        A2 --> A4
        A3 --> A4
    end
    subgraph dispatch["dispatch::clone_to_hp"]
        D1["archive_name = dream-{repo_name with slashes flattened}.tar.gz<br/>agentbox/services/dream-engine/src/dispatch.rs:179"]
        D2["a nested subpath like project/agentbox is FLATTENED to a single<br/>archive filename dream-project-agentbox.tar.gz, the archive is<br/>always a flat file in remote_dir even though its CONTENTS unpack<br/>to the mirrored depth"]
        D1 --> D2
    end
    A2 -.->|"repo_subpath passed as repo_name"| D1
    subgraph fallback["Fallback: repo outside the workspace, or either path uncanonicalisable"]
        F1["annexe_subpath returns the leaf, the final path component only<br/>agentbox/services/dream-engine/src/dispatch.rs:303-305, the pre-2026-09-07 behaviour"]
    end
    subgraph notes["ADR-2081"]
        direction TB
        N1["INVARIANT ADR-2081: symlinked nominations resolve to their REAL depth,<br/>workspace/agentbox to workspace/project/agentbox reports project/agentbox,<br/>proven by annexe_subpath_mirrors_real_depth_under_the_workspace<br/>agentbox/services/dream-engine/src/dispatch.rs:371"]
        N2["RESOLVED: sibling-path-deps is a REAL required gate again (dream.config.json:69),<br/>the FALLBACK rule from PR 4 that skipped sovereign-mesh-bridge is withdrawn"]
        N1 ~~~ N2
    end
```

**What it shows.** The depth law for the build machine: the target and each sibling are unpacked at their real position under the workspace so relative cargo path dependencies resolve there exactly as they do locally. The functions moved from `engine.rs` into `dispatch.rs` in the extraction, with the same behaviour.

**Why it is this way.** ADR-060 (archived) and the 2026-09-07 night that found the sovereign-mesh gate vacuous. The mechanism is adapter-owned because the annexe is an estate machine.

## AB-23.12 ADR-2081 — pipefail closes the tail-masking false positive

```mermaid
sequenceDiagram
    autonumber
    participant CFG as evaluatorEntrypoints<br/>agentbox/dream.config.json
    participant SSH as SshRunner wrap<br/>agentbox/services/dream-engine/src/runner.rs:24
    participant LOC as LocalRunner::run<br/>dream-core/src/runner.rs:46
    participant SH as remote or local bash

    Note over CFG: a declared entrypoint commonly ends in output piped to tail -12, the receipt must carry the<br/>PRODUCER's exit code, not the pipe's last stage
    rect rgb(250,235,235)
        Note over SSH,SH: BEFORE 2026-09-06: bash -c with no pipefail
        SSH->>SH: cd into the checkout, then run under timeout: bash -c cargo build, stderr redirected to stdout, piped to tail -12
        SH-->>SSH: cargo aborts non-zero, but tail exits 0, so the pipeline status is tail's 0
        Note over SSH: the receipt recorded outcome=PASSED exit=0, a pipe-masked false positive that upheld<br/>an ACCEPT on 09-06 (PR 4)
    end
    rect rgb(235,245,235)
        Note over SSH,SH: AFTER: both wrappers use bash -o pipefail -c
        SSH->>SH: cd into the checkout, then run under timeout: bash -o pipefail -c cargo build, piped to tail -12
        SH-->>SSH: pipeline status is the FIRST failing stage's exit code (cargo's, not tail's)
        Note over SSH: tailing output is still allowed, masking status is not (ADR-2081 Decision).<br/>Repos need no set -o pipefail of their own in dream.config.json
    end
    Note over LOC: the local wrapper is at dream-core/src/runner.rs:53, pinned by local_runner_does_not_let_a_tail_pipe_mask_a_failure<br/>at dream-core/src/runner.rs:157, which pipes a failing producer through tail and asserts the receipt keeps its exit code
    Note over SSH: the SSH wrapper stays in the adapter and is pinned by timeout_wrapper_quotes_the_command_and_the_directory<br/>(agentbox/services/dream-engine/src/runner.rs:57), it asserts the string bash -o pipefail -c at agentbox/services/dream-engine/src/runner.rs:64
    Note over SSH,LOC: with pipefail, piping to tail -N discards the HEAD of a failing run, repos should tail<br/>generously (website bench raised to 60, ADR-2081 Consequences)
```

**What it shows.** The two evaluator runners after the split. `SshRunner` (estate-specific) and `LocalRunner` (a core seam implementation) both wrap the command in `bash -o pipefail`, so a producer's failure is never masked by `tail`.

**Why it is this way.** The `EvaluatorRunner` seam is defined in `dream-core`, so the receipts, the gate and the verdict consume one `ExecOutcome` type and never know which runner produced it (`../project/agentbox/services/dream-engine/src/runner.rs:1-6`).

## AB-23.13 ADR-2081 — a no-patch ACCEPT is unproven, not a harness fault

```mermaid
sequenceDiagram
    autonumber
    participant G as gate::decide<br/>dream-gate/src/gate.rs:381
    participant C as CandidateState<br/>dream-gate/src/gate.rs:255

    Note over G: BEFORE 2026-09-07: step 3 graded candidate-phase receipts whenever<br/>the candidate was Applied OR an ACCEPT was claimed, so an ACCEPT with NO patch had<br/>nothing to run and every required evaluator's receipt was Missing
    rect rgb(250,235,235)
        Note over G,C: the old condition, measurement night, model says ACCEPT, no dream-patch emitted
        G->>C: candidate = NoPatch, claimed_accept = true
        G->>G: step 3 fires because claimed_accept is true
        loop each required evaluator
            G->>G: receipt Missing gives Veto{class: Harness}
        end
        G-->>G: three false Harness vetoes give verdict BLOCKED-ENV, operator alerted<br/>about a harness that had in fact passed every baseline evaluator<br/>(2026-09-07 dreamlab-ai-website)
    end
    rect rgb(235,245,235)
        Note over G,C: dream-gate/src/gate.rs:461, step 3 is gated on tree_trusted, which needs an Applied candidate
        G->>C: candidate = NoPatch, claimed_accept = true
        G->>G: tree_trusted is false, so step 3 does not fire
        Note over G: step 2 already pushed Veto::unproven for candidate, dream-gate/src/gate.rs:432.<br/>The Refused variant is the same unproven class (gate.rs:437) and DidNotApply is the Harness one (gate.rs:441), see AB-23.4 and AB-23.5
        G-->>G: verdict = INCONCLUSIVE (counts toward the dry streak, raises no operator alert)<br/>NOT BLOCKED-ENV
    end
    Note over G: pinned by accept_without_a_candidate_patch_is_unproven_not_a_harness_fault<br/>dream-gate/src/gate.rs:775, it asserts verdict INCONCLUSIVE and every veto class is not Harness
    Note over G: INVARIANT ADR-2081: Unproven is not Harness. Harness means the evidence could not be<br/>gathered (an operational fault), Unproven means there was nothing to test, see the VetoClass note in AB-23.8
```

**What it shows.** The one rule that stops a measurement night from being reported as a broken harness. The code moved to `dream-gate` with the same behaviour, and the gate gained the `Protected` candidate state in the same file (GAT-01 in the core corpus).

**Why it is this way.** The two receipt fixes (this and AB-23.12) landed together: pipefail with the siblings still unresolved would have vetoed every annexe ACCEPT.

## AB-23.14 ADR-2081 — Step-19 ledger cell provenance

```mermaid
flowchart TB
    R["nightly report text"] --> SF["sanitise_finding(report, verdict, night_date)<br/>dream-gate/src/verdict.rs:354"]
    SF --> Q0{"0. report_ledger_row_finding(report, night_date)<br/>dream-gate/src/verdict.rs:488, the report's own Step-19<br/>ledger table row, a pipe-delimited line,<br/>tonight's row only"}
    Q0 -->|"the cell satisfies ledger_cell_ok<br/>dream-gate/src/verdict.rs:421"| CELL["return that cell verbatim"]
    Q0 -->|"no ledger row, or its cell fails the contract"| Q1{"1. a Finding: line<br/>whose text satisfies ledger_cell_ok"}
    Q1 -->|"ok"| CELL
    Q1 -->|"none"| Q2["2. select_finding(report, verdict)<br/>the older heuristics, truncated to 80 chars<br/>dream-gate/src/verdict.rs:374"]
    Q2 --> CELL
    subgraph contract["finding_violations(cell), dream-gate/src/verdict.rs:393"]
        direction TB
        C1["non-empty, finding-empty"]
        C2["at most 80 UTF-16 code units, finding-too-long"]
        C3["does not start with given, see or gist, finding-hypothesis-leak and finding-pointer"]
        C4["does not contain see report or see gist"]
        C1 --- C2 --- C3 --- C4
    end
    Q0 -.-> contract
    Q1 -.-> contract
    CELL --> LEDGER["docs/dream-cycle/LEDGER.md finding cell<br/>see AB-23.10"]
    FULL["sanitise_finding_full(report, verdict)<br/>dream-gate/src/verdict.rs:514, NO 80-char cap, used for<br/>RuVector memory rows and PR bodies"] -.->|"still carries the WHOLE hypothesis"| MEM["memory and PR body"]
    subgraph notes["ADR-2081"]
        direction TB
        N1["RESOLVED: before this, the engine discarded the model's own Step-19 row for the<br/>truncated hypothesis, violating the downstream ledger row contract<br/>(finding-hypothesis-leak, PR 10). Pinned by<br/>sanitise_prefers_the_reports_own_ledger_row_cell, dream-gate/src/verdict.rs:680"]
        N2["a Step-19 cell that itself leaks the hypothesis, points elsewhere or<br/>overruns 80 units is ignored and the OLDER heuristics apply instead, pinned by<br/>sanitise_ignores_a_ledger_row_cell_that_breaks_the_contract, dream-gate/src/verdict.rs:702"]
        N1 ~~~ N2
    end
```

**What it shows.** Where the ledger's finding cell comes from. The functions moved unchanged into `dream-gate`, and the one difference is that the length is counted in UTF-16 code units so that it agrees with the TypeScript ledger contract the previous fork shipped (`../dream-engine/crates/dream-gate/src/verdict.rs:379`).

**Why it is this way.** The extracted crate has exactly one implementation of the finding rules, which the adapter's ledger-row contract (`../dream-engine/crates/dream-core/src/ledger.rs:254`) builds on.

## AB-23.15 ADR-2084 - the Loom call goes through the published loom-client

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:95
    participant CALL as call<br/>agentbox/services/dream-engine/src/llm.rs:49
    participant LOOM as call_loom<br/>agentbox/services/dream-engine/src/llm.rs:205
    participant ZAI as call_zai<br/>agentbox/services/dream-engine/src/llm.rs:120
    participant CRATE as loom-client crate<br/>published, see AB-28.11
    participant F as Loom facade<br/>agentbox/agentbox.toml:2572

    ENG->>CALL: call(cfg, prompt)
    alt llm_provider = zai - the DEFAULT (agentbox.toml:2585)
        CALL->>ZAI: POST the Anthropic Messages body with x-api-key
        ZAI-->>CALL: text parts joined, or EmptyResponse (llm.rs:168-186)
        Note over CALL,ZAI: exactly ONE retry, 20s apart, and only on a transient fault -<br/>transport, empty body, or an HTTP 5xx including Cloudflare 52x.<br/>is_transient refuses to retry a 4xx (llm.rs:69)
    else llm_provider = loom
        CALL->>LOOM: call_loom(cfg, prompt)
        LOOM->>CRATE: LoomClient::builder(url), timeout 600s, retry_backoff 20s (llm.rs:206-208)
        LOOM->>CRATE: ChatRequest temperature 1.0, top_p 0.95, top_k 20, max_tokens from cfg (llm.rs:213-219)
        LOOM->>CRATE: LoomOptions::declining_verbatim() (llm.rs:220)
        Note over LOOM,CRATE: INVARIANT ADR-2084: a dream prompt is generative and its subject IS in<br/>the ontology, so the scaffold stays and a retrieval-only serve is REFUSED.<br/>The crate raises Error::ScaffoldOnly, surfaced here as LlmError::Loom (llm.rs:22)
        CRATE->>F: POST the chat completion
        F-->>CRATE: answer, or ontology prose with no model call
        CRATE-->>LOOM: Answer with content, reasoning, served_mode and attempts (llm.rs:222-231)
        LOOM-->>CALL: answer.content (llm.rs:234)
        Note over CALL: the client already retried to its own ceiling, so the wrapper adds<br/>NO retry of its own - LlmError::Loom is non-transient here (llm.rs:78)
    end
    Note over CRATE: the hand-rolled client that lived in llm.rs knew ONE of the three facade<br/>traps. Two nights of verdicts in 2026-09 were derived from ontology prose that<br/>never reached a model, and truncation on a reasoning model returns EMPTY content<br/>rather than a short answer - the crate owns a token floor and a doubling retry<br/>the wrapper could not do, because it cannot see finish_reason (llm.rs:191-204)
```

**Debt:** `agentbox/agentbox.toml:2585` keeps `llm_provider = "zai"` as the default, so nightly repository content leaves the LAN on every unattended night; the LAN-only Loom path at `../project/agentbox/services/dream-engine/src/llm.rs:205` is opt-in.

## AB-23.16 Placeholder resolution and the refusal to dispatch to nowhere

```mermaid
flowchart TB
    TOML["agentbox.toml ships PLACEHOLDERS, not addresses<br/>hp_host CONNECTED_NODE_SSH agentbox.toml:2570<br/>hp_annexe_dir composite agentbox.toml:2571<br/>loom_url LOOM_BASE_URL agentbox.toml:2572"]
    TOML --> RP["RuntimeConfig.resolve_placeholders<br/>agentbox/services/dream-engine/src/config.rs:127"]
    RP -->|"whole-value form"| W["resolve_env_placeholder<br/>agentbox/services/dream-engine/src/config.rs:112 - a value that IS a placeholder"]
    RP -->|"composite form"| I["resolve_env_placeholders_infix<br/>agentbox/services/dream-engine/src/config.rs:146 - a value that CONTAINS one"]
    W -->|"unset or empty"| D1["the struct default applies<br/>default_loom_url agentbox/services/dream-engine/src/config.rs:168"]
    I -->|"any referenced var unset or empty"| D2["POISON the WHOLE value to None<br/>agentbox/services/dream-engine/src/config.rs:146-165 - half a path is worse<br/>than the caller's default"]
    D2 --> D3["default_hp_annexe_dir agentbox/services/dream-engine/src/config.rs:97"]
    I -->|"all set"| OK["the expanded path"]
    OK --> DISP["dispatch ssh, ssh_capture, scp_to"]
    D3 --> DISP
    DISP --> GUARD{"hp_host trimmed is empty<br/>agentbox/services/dream-engine/src/dispatch.rs:65"}
    GUARD -->|"yes"| NAH["DispatchError::NoAnnexeHost<br/>dispatch.rs:22 - the compose contract says<br/>an empty host means NO annexe"]
    GUARD -->|"no"| SSH["ssh under BatchMode, the script piped on stdin to a fixed bash -lc eval command<br/>dispatch.rs:35, dispatch.rs:38"]
    subgraph notes["What this closed"]
        direction TB
        N1["INVARIANT: no placeholder survives into the running config -<br/>pinned by an_unresolvable_manifest_placeholder_falls_back_to_the_default<br/>agentbox/services/dream-engine/src/config.rs:412 and the_manifest_placeholder_no_longer_reaches_the_loom_client<br/>agentbox/services/dream-engine/src/config.rs:398"]
        N2["INVARIANT: an unresolved composite annexe dir can no longer reach a live<br/>ssh command - pinned by<br/>the_annexe_dir_literal_no_longer_reaches_a_live_ssh_command agentbox/services/dream-engine/src/config.rs:376"]
        N3["scp_to and ssh_capture carry the same guard, so an empty host fails at the FIRST<br/>dispatch rather than part way through a night - dispatch.rs:124, dispatch.rs:92"]
        N1 ~~~ N2 ~~~ N3
    end
```

**Invariant:** every dispatch entry point refuses an empty annexe host with a typed error rather than shelling out (`../project/agentbox/services/dream-engine/src/dispatch.rs:65`, `../project/agentbox/services/dream-engine/src/dispatch.rs:124`).

**What changed (fish-safe transport):** the remote login shell is fish, which reads a backslash before a quote as an escape even inside single quotes, so a script embedded in the ssh command line broke on nested quoting; the script now travels over stdin and the only text fish parses is a fixed command with no backslash and two single quotes (`../project/agentbox/services/dream-engine/src/dispatch.rs:35`, `../project/agentbox/services/dream-engine/src/dispatch.rs:38`).

**Drift (diagram corpus vs repo):** `2899b3b7e` generalised estate addressing out of this public repository, so `../project/agentbox/dream.config.json:68` now says "the connected node annexe" where the corpus previously named a host and a rail address.

## AB-23.17 ADR-2115 — the forum governance round trip

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine::run_night<br/>agentbox/services/dream-engine/src/engine.rs:154
    participant GOV as governance<br/>agentbox/services/dream-engine/src/governance.rs:1
    participant INBOX as inbox::InboxItem<br/>agentbox/services/dream-engine/src/inbox.rs:1
    participant REL as RelaySession<br/>agentbox/services/dream-engine/src/relay.rs
    participant PANEL as forum panel #40;31400, d=dream-machine#41;<br/>governance.rs:108-139
    participant ADMIN as forum admin decision #40;31403#41;

    Note over GOV: three wire kinds, all nostr-bbs-core::governance types so the engine cannot drift from<br/>the relay/forum's own wire format — 31400 panel #40;JunkieJarvis#41;, 31402 case per open item<br/>#40;JunkieJarvis#41;, 31403 reply #40;forum admin#41;
    rect rgb(235,245,235)
        Note over ENG,GOV: start of night — ingest(inbox_path, false) governance.rs:1058
        ENG->>GOV: ingest(inbox_path, dry_run)
        GOV->>INBOX: load_from(inbox_path) — filter status=="open" AND published_event_id non-empty
        alt no published open items
            GOV-->>ENG: Report::default#40;#41; — nothing to do
        else published items exist
            GOV->>REL: RelaySession::connect(relay_url, key)
            alt relay unreachable or key missing
                GOV-->>ENG: fail-open, warn, return Report::default#40;#41; (governance.rs:1068-1076)
            else connected
                REL->>ADMIN: query KIND_ACTION_RESPONSE #35;d in #91;case ids, PANEL_D#93;
                ADMIN-->>REL: 31403 events, newest per case wins
                GOV->>GOV: resolution_for(item_kind, content) — DecisionOutcome::from_response_content<br/>(governance.rs:783)
                Note over GOV: approve on a question becomes answered "approve: reason", approve on an alert<br/>becomes dismissed "acknowledged: reason", reject becomes answered "reject: reason", amend<br/>#123;diff#125; becomes answered "amend: diff — reason" #40;empty diff refused#41;, delegate or other<br/>stays open
                GOV->>INBOX: resolve the matched inbox item — write status/answer back
                opt any item resolved tonight
                    GOV->>REL: send_withdrawals for the just-resolved cases only (governance.rs:1140-1146)
                    Note over GOV,REL: each is a NIP-09 kind-5 signed by the SAME agent key — e = the case's request id,<br/>a = its kind:pubkey:d coordinate, k = the request kind (withdrawal_event governance.rs:579-606)
                    GOV->>INBOX: mark_withdrawn_in records the deletion id as withdrawn_event_id (inbox.rs:134)
                end
            end
        end
    end
    rect rgb(235,245,235)
        Note over ENG,GOV: end of night — publish(inbox_path, paths, false) governance.rs:945
        ENG->>GOV: publish(inbox_path, CasePaths, dry_run) — CasePaths{workspace, artefact_dir} is built from the Engine (governance.rs:220, engine.rs:395-398)
        GOV->>INBOX: load_from(inbox_path) — filter status=="open"
        GOV->>REL: query current 31400 for #40;pubkey, d=dream-machine, governance.rs:989-995#41;
        alt panel content or tags differ from the target definition
            GOV->>PANEL: sign and publish the 31400 panel_event (governance.rs:996,1001-1013)
            alt relay rejects the panel
                GOV-->>ENG: stop — a case with no panel would not render (governance.rs:1004-1008)
            end
        end
        GOV->>REL: the sweep — send_withdrawals over plan_withdrawals(all items), BEFORE any new case<br/>(governance.rs:1015-1017). It catches items settled outside ingest, by dream-inbox.mjs or by hand
        Note over GOV: plan_withdrawals takes every item that is not open, was published and has no withdrawn_event_id.<br/>It withholds the a coordinate when an OPEN item reuses the same id, because the relay deletes every<br/>version at a coordinate and would take the live case too (governance.rs:620-633, governance.rs:575-578)
        loop each open item with no published_event_id
            GOV->>GOV: case_context(paths, item) (governance.rs:253) — reads the night's manifest.json (deep) and<br/>gate.json (verdict, modelVerdict, first veto) under artefact_dir/night_id and the newest matching LEDGER.md row<br/>under workspace/repo (finding, PR url), every field optional, a missing file leaves it empty (governance.rs:241,253)
            Note over GOV: the 31402 now says in words what the night produced — outcome_line fills the new panel columns<br/>"outcome" and "pr" (panel version 1.1.0, governance.rs:108-139), reasoning_for writes What happened /<br/>What you are being asked / How to answer, the title carries date and PR number, and a PR url becomes<br/>context_url (request_event governance.rs:494, outcome_line :327, reasoning_for :419, title_for :398).<br/>A gate veto is explained in plain English (explain_veto), and a vetoed night says there is nothing to merge.<br/>A question in the four-field shape renders as labelled lines (Decision, Why, If you approve, If you do not) and titles the case by its Decision, with the closing line that approving merges nothing (title_for governance.rs:398, reasoning_for governance.rs:419, ADR-2143)
            GOV->>PANEL: sign and publish a 31402 case, d=`dream-#123;id#125;` #40;case_d, governance.rs:87,<br/>publish loop :1019-1044#41;
            alt accepted
                PANEL-->>GOV: event id
                GOV->>INBOX: mark_published_in#40;inbox_path, item.id, event.id#41;
            else rejected or publish failed
                GOV-->>GOV: warn and continue #40;or stop the run on a hard publish error#41;
            end
        end
    end
    Note over GOV: MAX_PENDING_HOURS 168 escalates an undecided case #40;governance.rs:66, wired into<br/>PanelPolicy inside panel_event#41; — the single panel-level action ACK_ALERTS_ACTION<br/>"acknowledge-alerts" #40;governance.rs:62#41; dismisses every open alert published before it<br/>was pressed
    Note over GOV: INVARIANT: a withdrawal is sent at most once — an accepted kind-5 is recorded as withdrawn_event_id<br/>and the planner skips any item that carries one (governance.rs:626, inbox.rs:142). A relay rejection<br/>is left unrecorded so the next sweep retries it (governance.rs:697-700). `dream-engine governance withdraw`<br/>runs the sweep on its own (main.rs:129-130)
    Note over GOV,REL: OPEN: the card leaving the panel rests on the relay hard-deleting an author's own events on a<br/>kind-5 and the forum reading cases by subscription (governance.rs:29-32). The relay and forum code are<br/>outside this topic's sources, so that half of the round trip is asserted, not verified here
    Note over ENG,GOV: both calls are FAIL-OPEN and switch-gated separately — DREAM_GOVERNANCE=0<br/>#40;governance::enabled, governance.rs:82#41; disables the whole round trip — forum trouble<br/>never blocks or taints the night's evaluation, see AB-23.2
```

## AB-23.18 Annexe preflight — the write-then-measure health probe

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine::cycle_repo_body<br/>agentbox/services/dream-engine/src/engine.rs:532
    participant SWEEP as retention sweep<br/>agentbox/services/dream-engine/src/engine.rs:806-818, agentbox/services/dream-engine/src/dispatch.rs:64
    participant AH as dispatch::annexe_health<br/>agentbox/services/dream-engine/src/dispatch.rs:347
    participant HP as connected-node annexe<br/>agentbox/agentbox.toml:2570
    participant RS as runstate::begin<br/>dream-core/src/runstate.rs:511
    participant INBOX as inbox::add<br/>agentbox/services/dream-engine/src/inbox.rs:1

    Note over ENG: runs AFTER manifest::freeze but BEFORE the run journal counts an attempt,<br/>an unhealthy node costs the night, never the experiment's retry budget (engine.rs:793-797)
    ENG->>SWEEP: ssh, find night dirs older than 3 days under hp_annexe_dir, rm -rf (engine.rs:806-818)
    alt sweep fails
        SWEEP-->>ENG: warn only, fail-open (engine.rs:819-821)
    end
    ENG->>AH: annexe_health(hp_host, hp_annexe_dir, min_free_gib_from DREAM_ANNEXE_MIN_FREE_GIB, default ANNEXE_MIN_FREE_GIB 10) (engine.rs:823-827, engine.rs:2293, dispatch.rs:319)
    AH->>HP: ssh, annexe_probe_cmd: mkdir -p, printf ok to a probe file, rm -f the probe file,<br/>then df -Pk (dispatch.rs:327-332)
    Note over AH,HP: the WRITE is the real test. On a fully allocated btrfs volume df still reports<br/>tens of GiB free while every file create fails ENOSPC (metadata chunks exhausted,<br/>2026-09-26), free space alone would have passed a broken annexe
    HP-->>AH: AVAIL-KB=N line
    AH->>AH: parse_avail_kb(out) (dispatch.rs:336-340)
    alt no AVAIL-KB figure printed
        AH-->>ENG: Err, probe printed no free-space figure (dispatch.rs:348-349)
    else parsed
        AH->>AH: gib = kb / (1024*1024)
        alt gib below min_free_gib
            AH-->>ENG: Err, annexe dir has gib GiB free, below the floor (dispatch.rs:351-355)
        else healthy
            AH-->>ENG: Ok(gib)
        end
    end
    alt annexe unhealthy
        ENG->>INBOX: add("alert", repo, night_id, date, "did not start: annexe unhealthy") (engine.rs:836-848)
        ENG->>ENG: persist_blocked_env(...), BLOCKED-ENV, no attempt counted (engine.rs:850-864)
    else healthy
        ENG->>RS: begin(dir, ...), the journal now counts this attempt, see AB-23.7
    end
```

**Invariant:** the annexe probe writes and deletes a file before it trusts `df`'s free-space figure, so a metadata-exhausted btrfs volume that still reports free capacity is still caught (`../project/agentbox/services/dream-engine/src/dispatch.rs:327-332`).

## AB-23.19 ADR-2071 clause (c) — the one-shot night with management-api stopped

```mermaid
flowchart TB
    CRON["RAN VIA crontab 5-7 Oct 2026 — the crontab block is now REMOVED<br/>#40;ADR-2071:203 retired it once clause #40;c#41; passed#41;; the script stays<br/>scripts/activation/adr-2071-api-down-night.sh for reuse after an image change"]
    CRON --> TICK["adr-2071-api-down-night.sh tick<br/>stateless: marker, state file and the UTC clock only<br/>adr-2071-api-down-night.sh:9-15"]
    TICK --> MK{"marker names a YYYY-MM-DD?<br/>adr-2071-api-down-night.sh:83-88"}
    MK -->|"no marker"| NOOP["exit 0, nothing happens"]
    MK -->|"yes"| PH{"state phase<br/>adr-2071-api-down-night.sh:92-95"}
    PH -->|"done or missed"| RET["retire: marker renamed .consumed<br/>adr-2071-api-down-night.sh:90"]
    PH -->|"none, 00:30-00:59 UTC on the night"| PAUSE{"dream-paused flag?<br/>adr-2071-api-down-night.sh:100-105"}
    PAUSE -->|"yes"| MISS["phase missed, the API stays up<br/>adr-2071-api-down-night.sh:102"]
    PAUSE -->|"no"| STOP["EXIT trap armed, supervisorctl stop management-api<br/>phase stopped, stopped_at written<br/>adr-2071-api-down-night.sh:106-111"]
    PH -->|"none, first tick at or after 01:00"| MISS2["phase missed, reason no-tick-before-window<br/>adr-2071-api-down-night.sh:121-124"]
    STOP --> NIGHT["the dream window runs with no API:<br/>window_start 1, window_end 5 UTC<br/>agentbox.toml:2600-2601"]
    NIGHT --> RS{"phase stopped: which comes first?<br/>adr-2071-api-down-night.sh:129-144"}
    RS -->|"dream-last-night.json dated the night"| R1["reason night-record<br/>adr-2071-api-down-night.sh:137-138"]
    RS -->|"API RUNNING again"| R2["reason interrupted, a container restart<br/>adr-2071-api-down-night.sh:139-140"]
    RS -->|"07:30 UTC passed"| R3["reason deadline<br/>adr-2071-api-down-night.sh:141-142"]
    R1 --> START["supervisorctl start, phase done, marker retired<br/>adr-2071-api-down-night.sh:144-148"]
    R2 --> START
    R3 --> START
    START --> CHK["the morning after: adr-2087-check.sh --api-down-night DATE<br/>C3 passes only on night-record or deadline, see AB-14.15"]
```

**What it shows.** How the estate will run its first dream night with the management API deliberately down: a marker file names the night, a ten-minute cron tick stops the API in the half-hour before the window, and the first tick that sees the night record, the 07:30 UTC deadline or an API that came back on its own restarts it and retires the marker.
**Why it is this way.** ADR-2071 can only be accepted once clause (c) shows the night still completes, and journals its failures, with nothing to journal to. The owner fixed the night of Monday 5 October (owner decision 2026-10-02, Q9; `../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:191`), then asked for it early: clause (c) ran on 4 October in place of the 6 October night, all three Phase 1 clauses passed on the live image, and the record is **accepted, implementation complete, activation live** (`../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:195`). The schedule rode the checkout crontab rather than the image so that a container restart neither lost it nor needed a rebuild; that crontab block was removed once clause (c) passed and the 6 October marker was retired (`../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:203`). The tick is pure bash plus coreutils because that cron PATH has no `sed` or `awk` (`../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:59-60`).

**Invariant:** the API is never left down by a dying tick: an EXIT trap restarts management-api from the moment the stop is attempted until the state write lands (`../project/agentbox/scripts/activation/adr-2071-api-down-night.sh:106-110`).

**Debt:** the checkout crontab that `[program:podcast-cron]` reads still ticks the EXP-B8 label-log experiment every 30 minutes until its own PR merges (`../project/agentbox/skills/podcast-knowledge-ingest/crontab:34-44`), so a supervised program named for podcast ingest carries a schedule that has nothing to do with podcasts. The retired api-down-night script and its tests are kept in `scripts/activation/` for reuse after a future image change (`../project/agentbox/docs/adr/ADR-2071-journal-the-nightly-dream-cycle.md:203`).

## AB-23.20 Which key signs the forum round trip — role isolation off and on

```mermaid
flowchart TB
    GOVN["governance load_key, before any panel, case or withdrawal<br/>services/dream-engine/src/governance.rs:920-928"]
    LSK["relay::load_signing_key on JUNKIEJARVIS_PRIVKEY_HEX<br/>services/dream-engine/src/relay.rs:76-78, governance.rs:64"]
    FV{"JUNKIEJARVIS_PRIVKEY_HEX_FILE set?<br/>relay.rs:95"}
    GOVN --> LSK --> FV
    subgraph off["[security].role_isolation off - how it ships"]
        O1["the bare variable from the environment, inherited from PID 1<br/>relay.rs:102-103"]
        O2["else the KEY= line of the repo .env on the workspace bind<br/>relay.rs:104-118, default path governance.rs:75-79"]
        O3["dream-engine signs 31400, 31402 and kind 5 with the key in its<br/>own memory - AB-23.17"]
        O1 --> O2 --> O3
    end
    subgraph on["[security].role_isolation on"]
        E1["boot writes the key to /run/secrets/ab-identity/JUNKIEJARVIS_PRIVKEY_HEX,<br/>0400, owned by ab-identity uid 960, dir 0500<br/>config/entrypoint-unified.sh:460-461, config/entrypoint-unified.sh:470"]
        E2["exports the _FILE path to every program and unsets the variable<br/>config/entrypoint-unified.sh:462-463"]
        E3["dream-engine stays devuser - ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:80-81"]
        E4["the _FILE wins and an unreadable file is an error, never a fallback<br/>relay.rs:70-71, relay.rs:96"]
        E5["the bare variable and the repo .env are refused, a set variable<br/>is reported as ROLE-ISOLATION-LEAK - relay.rs:97-101, relay.rs:89-94"]
        E6["devuser cannot open an ab-identity 0400 file, so load_key warns<br/>agent key unavailable and the round trip is SKIPPED, fail-open<br/>governance.rs:922-926"]
        E1 --> E2 --> E3 --> E4 --> E6
        E4 --> E5
    end
    FV -->|"no, flag off"| off
    FV -->|"yes, flag on"| on
    subgraph port["The identity port, the intended signer"]
        P1["devuser may ask forum_event for the junkiejarvis key,<br/>kinds 1, 42 and 31923 only - config/custody/identity-port-acl.json:40"]
        P2["31400 to 31405 may NEVER be granted: governance kinds record a<br/>human decision and must not carry a container key<br/>services/nostr-pod-bridge/src/identity_port/acl.rs:65-73"]
        P3["the dream-engine cutover to the port is owed, W3b<br/>ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:237-238"]
        P1 --> P2
        P3 --> P2
    end
    E6 -.->|"no client for it at this revision"| port
    P2 -.-> T["TENSION: the 31400 panel and 31402 cases this engine signs are<br/>exactly the kinds the port refuses to grant - governance.rs:9-10"]
```

**Invariant (under role_isolation):** dream-engine never reads the JunkieJarvis key from its environment or from the repo `.env` on the workspace bind: with the flag on only `JUNKIEJARVIS_PRIVKEY_HEX_FILE` is consulted, and its absence is an error (`../project/agentbox/services/dream-engine/src/relay.rs:95-101`). With the flag off, which is how it ships, the pre-W2 order stands: the bare variable, then the `.env` line (`../project/agentbox/services/dream-engine/src/relay.rs:102-118`), the same loader contract as `scripts/dream-forum-suggestions.mjs`, which reads it through `role-secret.js` (`../project/agentbox/scripts/dream-forum-suggestions.mjs:215-221`, `../project/agentbox/management-api/lib/role-secret.js:120`).

**Tension (identity port ACL vs the forum governance round trip):** the port's loader refuses to grant kinds 31400-31405 to anyone, because a governance event records a human decision and must not carry a container key (`../project/agentbox/services/nostr-pod-bridge/src/identity_port/acl.rs:65-73`). Yet the dream engine publishes the 31400 panel and 31402 cases signed with the JunkieJarvis key (`../project/agentbox/services/dream-engine/src/governance.rs:9-10`), and ADR-2122 still plans to move dream-engine onto the port in W3b (`../project/agentbox/docs/adr/ADR-2122-role-service-accounts-run-secrets-and-the-identity-port.md:237-238`). As written, the port can never sign what AB-23.17 sends.

**Open:** under `[security].role_isolation` the forum governance round trip stops without an error. The delivered key file belongs to `ab-identity`, dream-engine runs as devuser, and `load_key` treats the refusal as fail-open and skips (`../project/agentbox/config/entrypoint-unified.sh:460-462`, `../project/agentbox/services/dream-engine/src/governance.rs:920-928`). Nothing at this revision says whether that is the intended state until W3b, or a gap the boot rehearsal must catch.

## AB-23.21 ADR-2143 — source supply is plumbing, not a decision

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine, step 11b<br/>agentbox/services/dream-engine/src/engine.rs:1648
    participant WNT as wanted.rs<br/>agentbox/services/dream-engine/src/wanted.rs:231
    participant STORE as dream-source-requests.json<br/>agentbox/services/dream-engine/src/wanted.rs:62
    participant INB as dream inbox and forum panel
    participant NEXT as Engine tree-read, next night<br/>agentbox/services/dream-engine/src/engine.rs:1254
    participant CORE as dream_core::source::gather<br/>dream-core/src/source.rs:1040

    Note over ENG: the night's report lists files under Sources wanted next night and may carry Human action lines
    ENG->>WNT: extract_wanted(report), one key per list item, path or path:a-b (engine.rs:1648, wanted.rs:105)
    loop each Human action line (engine.rs:1649)
        alt a supply, inline or bundle verb naming a path, and not in the four-field shape
            ENG->>WNT: is_source_supply then path_tokens, the paths join the wanted keys (engine.rs:1650-1657, wanted.rs:191)
            Note over ENG,INB: the line is NOT queued as a question, the operator is never asked to supply source
        else a real decision
            ENG->>INB: plain_decision rewrites an If approved that promises a merge, then inbox::add (engine.rs:1659-1662)
        end
    end
    ENG->>WNT: record(repo, deep, keys, night_id, report) (engine.rs:1668)
    WNT->>STORE: a new key is stored active with nights_seen 0 (wanted.rs:231)
    alt a key asked again after it was read, or after MAX_NIGHTS 3 nights unmet
        WNT-->>ENG: returned as looped and NOT re-armed (wanted.rs:26)
        ENG->>INB: ONE alert from loop_alert, the window was wrong or something else blocks the night (engine.rs:1669-1683, wanted.rs:343)
    end
    NEXT->>WNT: take_for(repo, deep), active requests oldest first, each counted as offered (engine.rs:1254, wanted.rs:276)
    NEXT->>NEXT: a slot's path-shaped scan entries are appended after the explicit requests (engine.rs:1266-1273)
    NEXT->>CORE: GatherInput::new(...).with_requested(requested) (engine.rs:1275)
    Note over CORE: requested paths are read FIRST, capped at half the file slots and half the byte budget,<br/>and pass the same checks as a planner pick (dream-core/src/source.rs:474, dream-core/src/source.rs:795)
    CORE-->>NEXT: SourceRecord with requested_honoured (dream-core/src/source.rs:716)
    NEXT->>WNT: settle marks every matching key met (engine.rs:1299, wanted.rs:295)
```

**What it shows.** How a file a night could not read goes back to the next night's tree-read instead of to a person. The store is keyed on repository, deep slot and path, and a request is active until it has been inlined once or offered for three nights. **Why it is this way.** On 2026-10-17 nine of fifty-seven open governance cases asked the operator to supply named files, several for the third time, because the planner never saw the previous night's ask; ADR-2143 makes source supply plumbing so the panel shows decisions only (`../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md:22-24`). The core half is ADR-DE-008 (`../dream-engine/docs/adr/ADR-DE-008-requested-sources-are-honoured-first.md:6-8`).

**Invariant:** every operator question is one line in four labelled fields, Decision, Why, If approved, If not, parsed by `decision_fields`, and an If approved that promises a merge, promotion or release is rewritten because approval never merges anything (`../project/agentbox/services/dream-engine/src/inbox.rs:237`, `../project/agentbox/services/dream-engine/src/inbox.rs:288`).

**Tension (ADR-2143 vs the scan offer):** a slot whose `scan` entries are paths offers them after the explicit requests, so on a quiet night they take the reserved half of the read, and ADR-2143 calls that intended (`../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md:23`) while the engine appends them for every slot without checking that a request is pending (`../project/agentbox/services/dream-engine/src/engine.rs:1266-1273`).

## AB-23.22 AB-23-S1-01 — the opt-in Jev triage before the candidate build

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine, step 8b<br/>agentbox/services/dream-engine/src/engine.rs:1439
    participant JT as jev_triage::consider<br/>agentbox/services/dream-engine/src/jev_triage.rs:497
    participant S1 as System One endpoint, the proxy on 127.0.0.1:8094 or the vendor
    participant SCR as dream_core::screen<br/>dream-core/src/screen.rs:362
    participant REC as jev-triage.json in the night directory
    participant GATE as candidate build and the deterministic gate

    ENG->>ENG: only when the report claims ACCEPT (engine.rs:1439)
    ENG->>JT: Config::from_env, report, patch, evaluator names, declared disciplines (engine.rs:1440-1447)
    alt AGENTBOX_JEV_DREAM_TRIAGE not set, no endpoint, or a local-only refusal
        JT-->>ENG: false, nothing built and nothing sent (jev_triage.rs:147, jev_triage.rs:383)
    else enabled
        JT->>S1: one request, two typed questions: usefulness as a four-level Score, problem_supported as a Noul (jev_triage.rs:319, jev_triage.rs:383)<br/>to the keyless LAN local_backend only when AGENTBOX_JEV_DREAM_TRIAGE_LOCAL_ONLY is set (jev_triage.rs:151-158),<br/>else the proxy and its selected backend, refused only when local-only is set and the backend is not declared local (jev_triage.rs:165-167)
        S1-->>JT: signals, or a timeout of 15 seconds, a non-2xx or an unparsable body, each a fallback (jev_triage.rs:72, jev_triage.rs:346)
        JT->>SCR: a second call, the library screen of the hypothesis against declared_disciplines (jev_triage.rs:431, dream-core/src/screen.rs:91)
        JT->>REC: annotation only, authority none, written as jev-triage.json (jev_triage.rs:70, jev_triage.rs:497)
        alt AGENTBOX_JEV_DREAM_TRIAGE_DEFER_BELOW is set in 0 to 0.5 and problem_supported is below it
            JT-->>ENG: true, defer
            ENG->>GATE: the candidate build is skipped, the state stays NotAttempted and the gate vetoes the ACCEPT claim as unproven (engine.rs:1461-1465)
        else the default, no threshold
            JT-->>ENG: false
            ENG->>GATE: the build and the gate run exactly as before (engine.rs:1465)
        end
    end
    Note over GATE: INVARIANT: a high usefulness never means ACCEPT. The signals never reach the gate,<br/>the deferral is the only action and it is inert until a threshold is configured (jev_triage.rs:490)
```

**What it shows.** An optional annotation step between the model's report and the costly candidate build. **Why it is this way.** The candidate evaluation is the expensive part of a night, and a hypothesis that the evidence does not support is cheaper to hold back than to build; but the verdict is advice about where to spend measurement, never about the change, so the deterministic gate, the protected-path guard and the human merge gate decide everything that does run (`../project/agentbox/services/dream-engine/src/jev_triage.rs:1-30`). The screening mechanism itself lives in `dream-core` 0.4.0 (`../dream-engine/crates/dream-core/src/screen.rs:1-30`).

**Resolved (configured, not live):** the manifest sets `jev_triage = true` with `[system_one] enabled = true` (`../project/agentbox/agentbox.toml:2612`, `../project/agentbox/agentbox.toml:890`), and the projector now exports `AGENTBOX_JEV_DREAM_TRIAGE_LOCAL_ONLY=0` unless `dream_triage` is named in `[system_one].local_only`, which is empty (`../project/agentbox/services/agentbox-manifest/src/system_one.rs:535`, `../project/agentbox/services/agentbox-manifest/src/system_one.rs:564-577`, `../project/agentbox/agentbox.toml:895`), so triage takes the proxy and the selected `typesafe` backend, with the private LAN route still projected for the opt-in (`../project/agentbox/services/agentbox-manifest/src/system_one.rs:402`, `../project/agentbox/config/entrypoint-unified.sh:1259-1263`); the flake bakes `AGENTBOX_JEV_DREAM_TRIAGE=1` into the dream-engine program only when the System One proxy is on (`../project/agentbox/flake.nix:280-281`). All of this is configured, not live: it is rebuild-class and ADR-2155 records the consumer activation as `staged` (`../project/agentbox/docs/adr/ADR-2155-system-one-consumer-activation.md:7`), so no running night is annotated or deferred until an image carrying that manifest is activated.

**Debt:** the module re-implements the System One request body that `crates/system-one/system-one-client` speaks, because the service builds from `services/dream-engine` alone and the client is a workspace path dependency (`../project/agentbox/services/dream-engine/src/jev_triage.rs:44-48`).

## Audit qualification — 2026-10-29

This topic describes the extraction as source at the stamped revisions. ADR-2141 records `implementation_status: complete` and `activation_status: live`: the operator rebuilt and activated the registry-pinned image on 2026-10-16 and the supervised daemon was RUNNING with no degradation (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:47`). The later pins (0.2.0, 0.3.0, 0.4.0) are source in `Cargo.toml`; the record carries activation evidence for 0.2.0 only (`../project/agentbox/docs/adr/ADR-2143-source-supply-is-plumbing-and-decisions-are-plain-english.md:35`).

**Resolved:** the earlier follow-on that `ontomark/dream.config.json` (`"adrConvention": "ADR-OM-NNN"`) would not load under the stricter `dream-gate` validation is closed: the legacy free-form template is accepted and rendered with its prefix from `dream-gate` 0.1.1 (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:35`).

**Tension (activation records):** ADR-2141 says the image running the adapter was rebuilt and activated (`../project/agentbox/docs/adr/ADR-2141-extract-the-dream-engine-core-into-a-standalone-permissive-repo.md:8`), while ADR-2081 still records `activation_status: staged` for the annexe-depth and receipt corrections that the same engine source carries (`../project/agentbox/docs/adr/ADR-2081-annexe-depth-law-and-honest-evaluator-receipts.md:7`); AB-23.11 to AB-23.14 describe source behaviour and claim no live receipt.
