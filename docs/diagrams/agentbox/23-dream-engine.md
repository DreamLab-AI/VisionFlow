---
id: AB-23
title: Dream machine — nightly cycle, gates and acceptance path
area: agentbox
governing:
  - ../project/agentbox/docs/GOVERNANCE-capabilities.md
adrs: [ADR-2024, ADR-2053, ADR-2081]
sources:
  - ../project/agentbox/services/dream-engine/src/engine.rs
  - ../project/agentbox/services/dream-engine/src/gate.rs
  - ../project/agentbox/services/dream-engine/src/verdict.rs
  - ../project/agentbox/services/dream-engine/src/runner.rs
  - ../project/agentbox/services/dream-engine/src/dispatch.rs
  - ../project/agentbox/services/dream-engine/src/runstate.rs
  - ../project/agentbox/services/dream-engine/src/readiness.rs
  - ../project/agentbox/services/dream-engine/src/manifest.rs
  - ../project/agentbox/services/dream-engine/src/receipts.rs
  - ../project/agentbox/services/dream-engine/src/candidate.rs
  - ../project/agentbox/services/dream-engine/src/persist.rs
  - ../project/agentbox/services/dream-engine/src/roster.rs
  - ../project/agentbox/dream.config.json
  - ../project/agentbox/agentbox.toml
  - ../project/agentbox/flake.nix
  - ../project/agentbox/config/hooks/dream-inbox-surface.cjs
  - ../project/agentbox/management-api/routes/dream.js
  - ../project/agentbox/management-api/lib/dream-ledger.js
  - ../project/agentbox/skills/dream-machine/commands/dream.md
  - ../project/agentbox/config/entrypoint-unified.sh
  - ../project/agentbox/services/dream-engine/src/ledger.rs
  - ../project/agentbox/services/dream-engine/src/llm.rs
  - ../project/agentbox/management-api/lib/action-plane.js
  - ../project/agentbox/scripts/dream-forum-suggestions.mjs
  - ../project/agentbox/scripts/dream-harvest.mjs
  - ../project/agentbox/scripts/dream-hooks-syntax.sh
  - ../project/agentbox/scripts/dream-inbox.mjs
  - ../project/agentbox/scripts/dream-machine-nightly.mjs
  - ../project/agentbox/scripts/dream-night-digest.mjs
verified_commit: 771d96ed5ac6f5daa1e78a60d109c130b9ef9b99
---

## AB-23.1 One repo-night — run phases

```mermaid
stateDiagram-v2
    [*] --> Admission
    Admission --> Handoff : readiness refuses
    note right of Admission
        readiness::assess (readiness.rs:162) runs BEFORE scheduling —
        no clone, no build, no model call. Unusable variants (readiness.rs:25-45):
        NoEvaluators, NoEvaluatorForDeep, NoRequiredEvaluatorForDeep,
        EmptyCommand, MissingScript, NonProbativeCommand, DarwinSandboxMissing.
    end note
    Admission --> Initialised : admitted
    Initialised --> ManifestFrozen
    note right of ManifestFrozen
        manifest::freeze (manifest.rs:196) writes ATOMICALLY and BEFORE any
        model call — baseline revision AND tree hash, evaluator identities with
        sha256(command), the dream.config.json digest, the intended model
        identity, and a deterministic run_id (manifest.rs:166).
    end note
    ManifestFrozen --> BaselineEvaluated
    BaselineEvaluated --> ModelCalled
    ModelCalled --> CandidateEvaluated
    note right of CandidateEvaluated
        candidate::evaluate (candidate.rs:77) applies the emitted dream-patch on
        an ISOLATED git worktree at HEAD and re-runs the required evaluators
        against THAT tree. The operator's working tree is never touched.
    end note
    CandidateEvaluated --> Gated
    Gated --> Persisted : accepted
    Gated --> Complete : REJECT or INCONCLUSIVE or BLOCKED-ENV
    Persisted --> Complete
    Complete --> [*]
    Initialised --> Abandoned : attempt budget exhausted
    note right of Abandoned
        Phase::Abandoned (runstate.rs:31-42) is TERMINAL and never silently
        retried — the operator is alerted. Resume::AlreadyComplete means the
        caller must NOT re-run (runstate.rs:83-95).
    end note
    Abandoned --> [*]
    Handoff --> [*]
```

## AB-23.2 Nightly cycle — every gate as a branch

```mermaid
sequenceDiagram
    autonumber
    participant SUP as supervisord<br/>agentbox/flake.nix:2235
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:59
    participant ROS as roster<br/>agentbox/services/dream-engine/src/roster.rs
    participant RS as runstate::begin<br/>agentbox/services/dream-engine/src/runstate.rs:132
    participant MAN as manifest::freeze<br/>agentbox/services/dream-engine/src/manifest.rs:196
    participant HP as HP annexe<br/>john@10.10.10.1
    participant LLM as llm provider<br/>agentbox/services/dream-engine/src/llm.rs
    participant GATE as gate::decide<br/>agentbox/services/dream-engine/src/gate.rs:187
    participant LED as ledger<br/>agentbox/services/dream-engine/src/ledger.rs

    SUP->>ENG: dream-engine --loop --agentbox-toml /etc/agentbox.toml
    Note over SUP,ENG: autostart=true autorestart=true priority=230 user=devuser (flake.nix:2235-2245)
    alt [dream_machine] enabled = false (agentbox/agentbox.toml:1731)
        ENG-->>SUP: byte-identical-when-off — no supervisor block is generated at all
    else enabled
        loop nightly window
            ENG->>ENG: UTC hour within window_start 1 .. window_end 5 (agentbox.toml:1755-1756)
            ENG->>ROS: least-recently-dreamed ordering, durable file
            Note over ROS: replaces alphabetical-sort-plus-truncate so max_repos_per_night 5 rotates the whole<br/>roster and survives a restart (agentbox.toml:1763)
            alt dry streak — last prune_dry_streak 5 ledger rows ALL INCONCLUSIVE (agentbox.toml:1680)
                ENG->>ENG: skip repo in nightly mode
                Note over ENG: REJECT counts as learning and RESETS the streak — revive via --target or a harness fix
            end
            ENG->>ENG: readiness::assess(cfg, repo, deep, repo_root) — readiness.rs:162, see AB-23.3
            alt not admitted
                ENG->>ENG: ReadinessReport refusal — HANDOFF disposition
                ENG->>LED: record HANDOFF — no clone, no build, no model call
            else admitted
                ENG->>RS: begin(dir, ...)
                alt Resume::AlreadyComplete
                    RS-->>ENG: do NOT re-run
                else Resume::Abandoned
                    RS-->>ENG: attempts exhausted, alert operator
                else Fresh or Resumed
                    ENG->>MAN: freeze(dir, manifest)
                    Note over MAN: INVARIANT: frozen BEFORE any model call. A restart recomputes the same run_id and<br/>resumes the same document — a MOVED baseline ARCHIVES the superseded manifest rather<br/>than overwriting (manifest.rs:176-220)
                    ENG->>HP: clone annexe via git archive HEAD
                    ENG->>HP: run required evaluators on the BASELINE tree
                    ENG->>LLM: call the model with the report prompt
                    Note over ENG,LLM: RESOLVED ADR-2053: the dream engine's DEFAULT reasoning provider is Z.AI<br/>(llm_provider = "zai", zai_model = "glm-5.3"), a deliberate operator choice for<br/>reasoning-token headroom. GOVERNANCE-capabilities now says so, and names the<br/>egress posture — nightly repo content leaves the LAN unless llm_provider = loom.<br/>The Loom is the opt-in LAN-only path.
                    LLM-->>ENG: report text with a VERDICT line and an optional dream-patch
                    ENG->>ENG: candidate::prepare then evaluate on an ISOLATED git worktree at HEAD<br/>(candidate.rs:41,:77) — see AB-23.5
                    ENG->>GATE: decide(manifest, strict, candidate, candidate_receipts)
                    GATE-->>ENG: GateDecision {accepted, verdict, model_verdict, vetoes, required_outcomes, summary}
                    ENG->>LED: append the row
                end
            end
        end
    end
    Note over ENG: PROPOSED ADR-2071 #40;2026-09-05#41;: still a bypass. ADR-2041 wired the execution journal<br/>onto POST /v1/tasks only, and a faithful routing would DENY the night — its SSH, external LLM,<br/>git push and forum side effects classify as egress or mutate, and action-plane.js wires no approver.<br/>ADR-2071 stages journalling ahead of policing, HTTP-only because local-jsonl caches the hash-chain<br/>head in memory and appends unlocked. GOVERNANCE-capabilities divergences 1 and 6 stay OPEN
```

## AB-23.3 Evaluator-readiness admission — refusal before scheduling

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:59
    participant RDY as readiness::assess<br/>agentbox/services/dream-engine/src/readiness.rs:162
    participant CFG as dream.config.json evaluatorEntrypoints<br/>agentbox/dream.config.json:50-54
    participant TREE as checked-out annexe tree

    ENG->>RDY: assess(cfg, repo, deep, repo_root)
    RDY->>CFG: read evaluatorEntrypoints
    Note over CFG: values are bare command strings or {cmd, required, deeps, timeoutSecs}. A BARE STRING<br/>reads FAIL-CLOSED as required=true, all deeps, 1800s
    alt no evaluators declared
        RDY-->>ENG: Unusable::NoEvaluators
    else none covers tonight's deep
        RDY-->>ENG: Unusable::NoEvaluatorForDeep
    else all covering evaluators are advisory
        RDY-->>ENG: Unusable::NoRequiredEvaluatorForDeep
        Note over RDY: nothing could ever veto, so acceptance would be UNFALSIFIABLE (readiness.rs:30-32)
    else empty command
        RDY-->>ENG: Unusable::EmptyCommand
    else script absent from the tree
        RDY->>TREE: resolve the script path
        TREE-->>RDY: not present
        RDY-->>ENG: Unusable::MissingScript
        Note over RDY: the annexe clone is git archive HEAD, so an untracked script cannot run there<br/>(readiness.rs:36-38)
    else non-probative command
        RDY-->>ENG: Unusable::NonProbativeCommand
        Note over RDY: an echo, a true, a bare colon — green every night, informative never<br/>(readiness.rs:39-41)
    else darwin entrypoint without a sandbox flag
        RDY-->>ENG: Unusable::DarwinSandboxMissing
        Note over RDY: INVARIANT ADR-2024 — every @metaharness/darwin entrypoint MUST run --sandbox mock or<br/>--sandbox agent, never the no-op real default which is documented surface-INDEPENDENT<br/>and emits the same output regardless of the code under test (agentbox.toml:1663-1668)
    else usable
        RDY-->>ENG: admitted
    end
    Note over ENG,RDY: on any refusal the disposition is HANDOFF with NO clone, NO build and NO model call
    Note over RDY: config load-time validation ALSO rejects a darwin entrypoint without a sandbox flag —<br/>re-checked here so the admission report is complete on its own terms<br/>(readiness.rs:42-44)
```

## AB-23.4 The deterministic required-check gate

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:59
    participant VP as verdict::parse_verdict_strict<br/>agentbox/services/dream-engine/src/verdict.rs:243
    participant CR as receipts::complete_receipts<br/>agentbox/services/dream-engine/src/gate.rs:132
    participant EV as environment_vetoes<br/>agentbox/services/dream-engine/src/gate.rs:157
    participant G as gate::decide<br/>agentbox/services/dream-engine/src/gate.rs:187

    ENG->>VP: parse_verdict_strict(report)
    Note over VP: acceptance consults ONLY a bare unambiguous "VERDICT: TOKEN" line — missing, noisy,<br/>conflicting or unknown declarations are TYPED errors that can never reach ACCEPT<br/>(VerdictParseError, verdict.rs:204)
    alt parse error
        VP-->>G: Err(VerdictParseError)
        G->>G: push Veto{class: Unproven, subject: "verdict"}
    else Ok(Verdict)
        VP-->>G: Accept | Reject | Inconclusive | BlockedEnv | Handoff (verdict.rs:16-31)
    end
    alt claimed_accept AND candidate == NoPatch (gate.rs:210,217-224)
        G->>G: Veto{class: Unproven, subject: "candidate"} — "declared ACCEPT but emitted no dream-patch block"
        Note over G: RESOLVED ADR-2081 (2026-09-07 dreamlab-ai-website): this is now the ONLY consequence of an<br/>ACCEPT with no patch. Step 3 below is gated on CandidateState::Applied, so absent candidate<br/>receipts are never graded — before this fix they read as three "never ran" Harness vetoes and<br/>the night surfaced BLOCKED-ENV over a baseline that had passed every evaluator (gate.rs:236-243)
    else claimed_accept AND candidate == DidNotApply (gate.rs:225-228)
        G->>G: Veto{class: Harness, subject: "candidate"} — the patch failed to apply to the baseline tree
    end
    alt candidate is CandidateState::Applied (gate.rs:244)
        G->>CR: complete_receipts(required, candidate_receipts, Phase::Candidate)
        CR-->>G: one receipt per REQUIRED evaluator, Missing where absent
        G->>EV: environment_vetoes(...)
        loop each required evaluator outcome
            alt Passed
                G->>G: no veto (gate.rs:96-99)
            else Missing
                G->>G: Veto{class: Harness}
            else Blocked or TimedOut or Silent
                G->>G: Veto{class: Harness}
                Note over G: is_harness_fault (receipts.rs:88-98) — Blocked, TimedOut, Silent, Missing. Silent is<br/>exit 0 with NO output on either stream: surface-independent, therefore unfalsifiable,<br/>the ADR-065 no-op in its most literal form (receipts.rs:59-61)
            else Failed or ExplicitFail
                G->>G: Veto{class: Evidence}
                Note over G: ExplicitFail is exit 0 whose output declares failure in so many words — FAIL:, FAILED,<br/>"test result: FAILED" (receipts.rs:52-55)
            end
        end
    else candidate not Applied — nothing to grade (gate.rs:244)
        Note over G: no Harness/Evidence veto is raised here — the ONLY consequence of a no-patch ACCEPT is<br/>the Unproven veto above
    end
    alt any veto
        G-->>ENG: accepted=false
        Note over G: harness-class vetoes yield BLOCKED-ENV, evidence-class REJECT, unproven-only INCONCLUSIVE<br/>(gate.rs:255-273) — regardless of the model's report text
        Note over G: an ACCEPT claim with candidate==NoPatch now lands INCONCLUSIVE (unproven-only), not<br/>BLOCKED-ENV — test accept_without_a_candidate_patch_is_unproven_not_a_harness_fault<br/>(gate.rs). INCONCLUSIVE counts toward the dry streak (AB-23.2) but raises no operator alert,<br/>unlike BLOCKED-ENV
    else claimed ACCEPT AND candidate applied AND every required evaluator Passed
        G-->>ENG: accepted=true verdict=ACCEPT
    end
    Note over G: INVARIANT: evaluator failure VETOES acceptance. Before this landed, failure text could<br/>coexist with ACCEPT (ADR-2024 closeout 2026-09-04)
    Note over VP: BLOCKED-ENV never counts toward a repo's dry streak — a broken harness must not park a<br/>healthy repo — and is raised by the engine's pre-flight probe, not parsed from an LLM<br/>report (verdict.rs:21-26)
```

## AB-23.5 Candidate re-evaluation on an isolated worktree

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:59
    participant PP as persist::extract_patch<br/>agentbox/services/dream-engine/src/persist.rs:38
    participant PREP as candidate::prepare<br/>agentbox/services/dream-engine/src/candidate.rs:41
    participant WT as isolated git worktree at HEAD
    participant EVAL as candidate::evaluate<br/>agentbox/services/dream-engine/src/candidate.rs:77
    participant REC as receipts::persist<br/>agentbox/services/dream-engine/src/receipts.rs:263
    participant MW as manifest::write_candidate<br/>agentbox/services/dream-engine/src/manifest.rs:329
    participant CL as candidate::cleanup<br/>agentbox/services/dream-engine/src/candidate.rs:61

    ENG->>PP: extract_patch(report)
    alt no dream-patch in the report
        PP-->>ENG: None
        Note over ENG: CandidateState records the absence — a claim of ACCEPT with no applied candidate cannot<br/>be accepted (gate.rs:171)
    else patch present
        PP-->>ENG: Some(patch)
        ENG->>PREP: prepare(...)
        PREP->>WT: create worktree at HEAD and apply the patch
        Note over PREP,WT: INVARIANT: the operator's working tree is NEVER touched
        PREP-->>ENG: PreparedCandidate with the candidate tree hash
        ENG->>EVAL: evaluate(...) — re-run the REQUIRED evaluators against THAT tree
        loop each required evaluator
            EVAL->>EVAL: run with its timeoutSecs budget, wrapped bash -o pipefail -c '…' (runner.rs:65,99)
            Note over EVAL: RESOLVED ADR-2081: both the SSH and LocalRunner wrappers gained -o pipefail<br/>(2026-09-06 sovereign-mesh night). Every declared evaluator tails its output<br/>(cargo build piped to tail -12) — without pipefail the pipeline's exit status was<br/>tail's 0, so an aborting cargo build recorded outcome=PASSED exit=0 — a<br/>pipe-masked false positive that upheld an ACCEPT on 2026-09-06 (PR 4)
            EVAL->>EVAL: receipts::classify(exec, timeout_secs) (receipts.rs:206)
        end
        EVAL-->>ENG: Vec<EvaluatorReceipt>
        ENG->>REC: persist(dir, Phase::Candidate, receipts)
        Note over REC: raw receipts — exit code, BOTH streams verbatim, duration and a typed outcome per<br/>evaluator, per phase under the night directory
        ENG->>MW: write_candidate(dir, CandidateRecord)
        alt accepted downstream
            ENG->>CL: cleanup(repo, prepared)
        else rejected
            ENG->>CL: discard(repo, prepared) (candidate.rs:67)
        end
    end
```

## AB-23.6 Persistence and the human merge gate

```mermaid
sequenceDiagram
    autonumber
    participant G as gate::decide<br/>agentbox/services/dream-engine/src/gate.rs:187
    participant PA as persist::persist_accept<br/>agentbox/services/dream-engine/src/persist.rs:134
    participant BN as persist::branch_name<br/>agentbox/services/dream-engine/src/persist.rs:54
    participant BW as persist::build_branch_worktree<br/>agentbox/services/dream-engine/src/persist.rs:80
    participant PR as persist::push_and_open_pr<br/>agentbox/services/dream-engine/src/persist.rs:158
    participant H as Human reviewer
    participant CFG as dream.config.json<br/>agentbox/dream.config.json

    alt not accepted
        G-->>PA: skipped — verdict recorded in the ledger only
    else accepted
        G->>PA: persist_accept(...)
        PA->>BN: branch_name(deep, date)
        Note over BN: branchPrefix "dream/" (agentbox/dream.config.json:82)
        PA->>BW: build_branch_worktree(...)
        PA->>PR: push_and_open_pr(...)
        PR-->>H: PrOutcome — a PULL REQUEST, never a merge
        Note over CFG: autoMerge = false (agentbox/dream.config.json:87). labels dream-cycle, agentbox-self<br/>(:83-86)
        alt the change touches services/dream-engine — a SELF-REFERENTIAL hypothesis
            Note over PA,H: INVARIANT: agentbox contains the dream-engine that dreams it. A change to that crate<br/>changes the dreamer itself — apply EXTRA review scrutiny and NEVER let a self-modifying<br/>hypothesis bypass the human-merge gate. Still evidence-gated, still witnessed<br/>(dream.config.json:74 extraDisciplines self-referential)
        end
        H->>H: review and merge, or not
    end
    Note over CFG: extraDisciplines also fence: nix-build-out-of-annexe (the full flake.nix image build<br/>needs the host build shell, so record image/Nix findings as HANDOFF (rebuild) notes and<br/>never invent a nix-build evaluator that would silently no-op, dream.config.json:75) —<br/>sibling-path-deps (services/nostr-pod-bridge path-deps the sibling repos, dream.config.json:76 — RENAMED<br/>2026-09-07 from sibling-path-deps-fenced: since the annexe now mirrors real workspace<br/>depth #40;see AB-23.11#41; the siblings DO ship and sovereign-mesh-bridge is a REAL required<br/>gate, not a permanent skip) — secrets-never-in-report (never quote .env or any<br/>KEY/PRIVKEY value into a report, ledger or gist, dream.config.json:77) — cite-existing-adrs (NEW dream.config.json:78 —<br/>cite only ADR ids that exist under docs/adr/ — nights have cited ADR-056/ADR-0057/ADR-2024<br/>that do not exist)
    Note over H: DIVERGENCE: the human-merge boundary is a PROCESS, not a code control — ADR-2024<br/>implementation_status stays partial for that reason
```

## AB-23.7 Run journal — restart, resume and abandonment

```mermaid
sequenceDiagram
    autonumber
    participant ENG as Engine<br/>agentbox/services/dream-engine/src/engine.rs:59
    participant RS as runstate<br/>agentbox/services/dream-engine/src/runstate.rs
    participant F as run-state.json<br/>night directory
    participant MAN as manifest<br/>agentbox/services/dream-engine/src/manifest.rs

    ENG->>RS: begin(dir, ...) (runstate.rs:132)
    RS->>F: load(dir) (runstate.rs:123)
    alt no prior record
        F-->>RS: none
        RS-->>ENG: Resume::Fresh(RunState)
    else prior attempt died part-way
        F-->>RS: RunState with phase < Complete and attempts < max_attempts
        RS-->>ENG: Resume::Resumed — continue from resumed_from
        ENG->>MAN: run_id(repo, date, deep, baseline_revision, config_digest) (manifest.rs:166)
        Note over MAN: a restart recomputes the SAME id from the same inputs, so it resumes against the same<br/>frozen document
    else already finished
        RS-->>ENG: Resume::AlreadyComplete — caller must NOT re-run
    else attempts exhausted
        RS-->>ENG: Resume::Abandoned — record the abandonment and move on rather than looping
    end
    Note over RS: should_run() is true only for Fresh or Resumed (runstate.rs:103-105)
    loop each phase transition
        ENG->>RS: advance(dir, state, phase) (runstate.rs:188)
        RS->>F: durable write
    end
    alt success
        ENG->>RS: complete(dir, state, verdict) (runstate.rs:197)
    else failure
        ENG->>RS: fail(dir, state, error) (runstate.rs:205)
    end
    Note over MAN: manifest::freeze returns Freeze (manifest.rs:176) — a diverged baseline ARCHIVES the<br/>superseded manifest instead of overwriting it. A defect caught in test: the digest<br/>originally included its own timestamp, which would have made every restart read as a<br/>diverged experiment
```

## AB-23.8 Core types

```mermaid
classDiagram
    class RunState {
        +u32 schema
        +String run_id
        +String night_id
        +String repo
        +String date
        +Phase phase
        +u32 attempts
        +u32 max_attempts
        +Option~Phase~ resumed_from
        +String first_seen
        +String updated_at
        +Option~String~ last_error
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
        +state() RunState
        +should_run() bool
    }
    class Verdict {
        <<enum>>
        Accept
        Reject
        Inconclusive
        BlockedEnv
        Handoff
        +as_str() str
        +is_significant() bool
    }
    class GateDecision {
        +bool accepted
        +String verdict
        +String model_verdict
        +Vec~Veto~ vetoes
        +Vec~Tuple~ required_outcomes
        +String summary
        +verdict_enum() Verdict
    }
    class Veto {
        +VetoClass class
        +String subject
        +String reason
    }
    class VetoClass {
        <<enum>>
        Harness
        Evidence
        Unproven
    }
    class EvaluatorOutcome {
        <<enum>>
        Passed
        Failed
        ExplicitFail
        Blocked
        TimedOut
        Silent
        Missing
        +label() str
        +is_pass() bool
        +is_harness_fault() bool
    }
    class EvaluatorReceipt {
        +String name
        +String command
    }
    class ExperimentManifest {
        +String run_id
        +String baseline_revision
        +String baseline_tree
        +Vec~EvaluatorIdentity~ evaluators
        +String config_digest
        +ModelIdentity model
    }
    class EvaluatorIdentity {
        +String name
        +String command_sha256
        +bool required
        +u64 timeout_secs
    }
    class Unusable {
        <<enum>>
        NoEvaluators
        NoEvaluatorForDeep
        NoRequiredEvaluatorForDeep
        EmptyCommand
        MissingScript
        NonProbativeCommand
        DarwinSandboxMissing
        +describe() String
    }
    RunState --> Phase
    Resume --> RunState
    GateDecision --> Veto
    Veto --> VetoClass
    GateDecision ..> Verdict : verdict_enum
    EvaluatorReceipt --> EvaluatorOutcome
    ExperimentManifest --> EvaluatorIdentity
    note for VetoClass "Harness = the evidence could not be gathered, an operational fault. Evidence = the<br/>evidence was gathered and it is against the candidate. Unproven = there was nothing to<br/>test or nothing readable to act on (gate.rs:36-44)"
```

## AB-23.9 Dream-inbox surfacing hook

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator turn<br/>any Claude session
    participant CC as Claude Code UserPromptSubmit
    participant HK as dream-inbox-surface.cjs<br/>agentbox/config/hooks/dream-inbox-surface.cjs:1
    participant INBOX as dream-inbox.json<br/>/home/devuser/workspace/.agentbox/dream-inbox.json
    participant EP as entrypoint registration<br/>agentbox/config/entrypoint-unified.sh:1350

    Note over EP: the entrypoint prefers /opt/agentbox/config/hooks/dream-inbox-surface.cjs and falls back<br/>to the repo path, then dedupes on the command substring (:1279,:1291)
    U->>CC: submits a prompt
    CC->>HK: UserPromptSubmit with stdin JSON
    HK->>INBOX: readFileSync
    alt file missing or unparseable or not an array
        HK-->>CC: exit — fail-open, no injection
    else
        HK->>HK: filter status == open AND now - last_surfaced > RESURFACE_HOURS 4 * 3600
        HK->>HK: slice(0, MAX_PER_TURN 2)
        alt nothing due
            HK-->>CC: exit — no injection
        else items due
            HK->>INBOX: stamp last_surfaced = now and write back
            HK-->>CC: inject the open items as additional context with instructions to relay them and record<br/>answers via /dream answer
        end
    end
    Note over HK: the nightly engine has NO session with the operator — this hook is the bridge that makes<br/>the self-improvement loop part of working praxis instead of a log nobody reads (:4-10)
    Note over HK: rate limiting is PER ITEM via last_surfaced, and fail-open on any error (:12-15)
```

## AB-23.10 Control and reporting surfaces

```mermaid
flowchart TB
    subgraph skill["/dream control skill — agentbox/skills/dream-machine/commands/dream.md"]
        S1["/dream status (or no argument) — skills/dream-machine/commands/dream.md:7"]
        S2["/dream questions · /dream answer id text · /dream dismiss id — skills/dream-machine/commands/dream.md:17"]
        S3["/dream harvest [--days N] — skills/dream-machine/commands/dream.md:29"]
        S4["/dream off · /dream on — skills/dream-machine/commands/dream.md:39"]
        S5["/dream run [repo] — skills/dream-machine/commands/dream.md:44"]
        S6["/dream standby repo · /dream revive repo — skills/dream-machine/commands/dream.md:57"]
        S7["/dream digest [date] — skills/dream-machine/commands/dream.md:62"]
        S8["/dream nominate repo — skills/dream-machine/commands/dream.md:70"]
    end
    subgraph scripts["Scripts — agentbox/scripts/"]
        N1["dream-machine-nightly.mjs"]
        N2["dream-inbox.mjs"]
        N3["dream-harvest.mjs"]
        N4["dream-forum-suggestions.mjs"]
        N5["dream-night-digest.mjs"]
        N6["dream-hooks-syntax.sh — an evaluatorEntrypoint, not a control surface"]
    end
    subgraph api["management-api"]
        R1["GET /dream/status (fastify)<br/>agentbox/management-api/routes/dream.js:24"]
        L1["dream-ledger.js parseLedger management-api/lib/dream-ledger.js:52 · verdictStats management-api/lib/dream-ledger.js:80 · latestNights management-api/lib/dream-ledger.js:91<br/>discoverNominatedRepos management-api/lib/dream-ledger.js:117 · pendingMerges management-api/lib/dream-ledger.js:202<br/>readRepoDreamStatus management-api/lib/dream-ledger.js:215 · aggregateDreamStatus management-api/lib/dream-ledger.js:264"]
    end
    subgraph out["Outputs"]
        O1["docs/dream-cycle/LEDGER.md<br/>ledgerPath, agentbox/dream.config.json:81"]
        O2["docs/dream-cycle/FORUM-SUGGESTIONS.md"]
        O3["dream-inbox.json — see AB-23.9"]
        O4["voice/console/site/dream.html"]
    end
    S1 --> R1
    R1 --> L1
    L1 --> O1
    S3 --> N3
    S2 --> N2
    N2 --> O3
    S7 --> N5
    S5 --> N1
    N4 --> O2
    L1 --> O4
    subgraph notes["Invariants and drift"]
        direction TB
        ND1["DIVERGENCE: routes/dream.js exposes exactly ONE endpoint, GET /dream/status. There is no<br/>HTTP control surface for run/nominate/answer — those are skill-plus-script paths only"]
        ND2["DIVERGENCE: legacy ADR-055 dream cockpit is PARTIAL — dream.html exists, the full<br/>cockpit is unverified. The 056/058/061/062-072 governance band is PAPER: the engine runs<br/>ahead of its decision-surface, self-GC and telemetry-contract designs<br/>(GOVERNANCE-capabilities divergence 6)"]
        ND3["Both repo roots carry a dream.config.json — agentbox/dream.config.json (the agentbox<br/>nomination) and the VisionClaw root dream.config.json. Each nominated repo declares its<br/>own evaluatorEntrypoints and extraDisciplines"]
        ND1 ~~~ ND2 ~~~ ND3
    end
```

## AB-23.11 ADR-2081 — the annexe mirrors real workspace depth

```mermaid
flowchart TB
    subgraph before["Before 2026-09-07: one level too shallow"]
        B1["remote_dir/agentbox #40;flat repo_name#41;"]
        B2["cargo path-dep ../../../../nostr-rust-forum<br/>climbs to ONE ABOVE remote_dir"]
        B3["manifest load fails — sovereign-mesh-bridge<br/>REQUIRED gate red every night"]
        B1 --> B2 --> B3
    end
    subgraph after["clone_repo_and_siblings + annexe_subpath<br/>engine.rs:1420, engine.rs:1451"]
        A1["annexe_subpath#40;repo_path, workspace_root#41;<br/>canonicalizes both, strip_prefix, joins components<br/>engine.rs:1451-1470"]
        A2["target ships at remote_dir/project/agentbox<br/>#40;its REAL path under the workspace, not the leaf name#41;"]
        A3["each annexe_include sibling ships at remote_dir/&lt;its own subpath&gt;<br/>e.g. remote_dir/nostr-rust-forum — engine.rs:1432-1440"]
        A4["cargo ../../../../nostr-rust-forum now climbs to<br/>remote_dir/ exactly as it climbs to the workspace root locally"]
        A1 --> A2
        A1 --> A3
        A2 --> A4
        A3 --> A4
    end
    subgraph dispatch["dispatch::clone_to_hp — dispatch.rs:142"]
        D1["archive_name = format#40;dream-{}.tar.gz, repo_name.replace#40;'/','-'#41;#41;<br/>dispatch.rs:150"]
        D2["a nested subpath like project/agentbox is FLATTENED to a single<br/>archive filename dream-project-agentbox.tar.gz — the archive is<br/>always a flat file in remote_dir even though its CONTENTS unpack<br/>to the mirrored depth"]
        D1 --> D2
    end
    A2 -.->|"repo_subpath passed as repo_name"| D1
    subgraph fallback["Fallback — repo outside the workspace, or either path uncanonicalisable"]
        F1["annexe_subpath returns the leaf#40;#41; — final path component only<br/>#40;engine.rs:1459-1461, matches pre-2026-09-07 behaviour#41;"]
    end
    subgraph notes["ADR-2081"]
        direction TB
        N1["INVARIANT ADR-2081: symlinked nominations resolve to their REAL depth —<br/>workspace/agentbox -> workspace/project/agentbox reports project/agentbox,<br/>proven by annexe_subpath_mirrors_real_depth_under_the_workspace#40;#41;<br/>engine.rs:1554"]
        N2["RESOLVED: sibling-path-deps is a REAL required gate again (dream.config.json:76) —<br/>the FALLBACK rule from PR #4 that skipped sovereign-mesh-bridge is withdrawn"]
        N1 ~~~ N2
    end
```

## AB-23.12 ADR-2081 — pipefail closes the tail-masking false positive

```mermaid
sequenceDiagram
    autonumber
    participant CFG as evaluatorEntrypoints<br/>agentbox/dream.config.json
    participant SSH as SSH runner wrap<br/>agentbox/services/dream-engine/src/runner.rs:63
    participant LOC as LocalRunner::run<br/>agentbox/services/dream-engine/src/runner.rs:93
    participant SH as remote/local bash

    Note over CFG: a declared entrypoint commonly ends in output piped to tail -12 — the receipt must carry the<br/>PRODUCER's exit code, not the pipe's last stage
    rect rgb(250,235,235)
        Note over SSH,SH: BEFORE 2026-09-06: bash -c (no pipefail)
        SSH->>SH: cd into the checkout, then run under timeout: bash -c cargo build, stderr redirected to stdout, piped to tail -12
        SH-->>SSH: cargo aborts non-zero, but tail exits 0 — pipeline status = tail's 0
        Note over SSH: receipt recorded outcome=PASSED exit=0 — a pipe-masked false positive that upheld<br/>an ACCEPT on 09-06 (PR #4)
    end
    rect rgb(235,245,235)
        Note over SSH,SH: AFTER — runner.rs:63,:93 both wrap `bash -o pipefail -c`
        SSH->>SH: cd into the checkout, then run under timeout: bash -o pipefail -c cargo build, stderr redirected to stdout, piped to tail -12
        SH-->>SSH: pipeline status = the FIRST failing stage's exit code (cargo's, not tail's)
        Note over SSH: tailing output is still allowed — masking status is not (ADR-2081 Decision).<br/>Repos need no set -o pipefail of their own in dream.config.json
    end
    Note over LOC: local_runner_does_not_let_a_tail_pipe_mask_a_failure#40;#41; — runner.rs:215 — pipes a<br/>failing producer #40;exit 101#41; through tail -3 and asserts the receipt's exit_code is 101,<br/>not tail's 0
    Note over SSH,LOC: with pipefail, piping to tail -N discards the HEAD of a failing run — repos should tail<br/>generously #40;website bench raised to 60, ADR-2081 Consequences#41;
```

## AB-23.13 ADR-2081 — a no-patch ACCEPT is unproven, not a harness fault

```mermaid
sequenceDiagram
    autonumber
    participant G as gate::decide<br/>agentbox/services/dream-engine/src/gate.rs:187
    participant C as CandidateState

    Note over G: BEFORE 2026-09-07: step 3 graded candidate-phase receipts whenever<br/>`matches!(candidate, Applied) OR claimed_accept` — an ACCEPT with NO patch had<br/>nothing to run, so every required evaluator's receipt was Missing
    rect rgb(250,235,235)
        Note over G,C: the old condition — measurement night, model says ACCEPT, no dream-patch emitted
        G->>C: candidate = NoPatch, claimed_accept = true
        G->>G: step 3 fires because claimed_accept is true
        loop each required evaluator
            G->>G: receipt Missing -> Veto{class: Harness}
        end
        G-->>G: three false Harness vetoes -> verdict BLOCKED-ENV, operator alerted<br/>about a harness that had in fact passed every baseline evaluator<br/>#40;2026-09-07 dreamlab-ai-website#41;
    end
    rect rgb(235,245,235)
        Note over G,C: gate.rs:244 — the condition drops `OR claimed_accept`
        G->>C: candidate = NoPatch, claimed_accept = true
        G->>G: `if matches!#40;candidate, CandidateState::Applied {..}#41;` is FALSE — step 3 does not fire
        Note over G: step 2 already pushed Veto::unproven#40;"candidate", "report declared ACCEPT but emitted<br/>no dream-patch block, so no candidate tree could be built or re-evaluated"#41; — gate.rs:220-224.<br/>#40;NotAttempted is a distinct CandidateState with its own unproven veto at gate.rs:229-232#41;
        G-->>G: verdict = INCONCLUSIVE #40;counts toward the dry streak, raises no operator alert#41;<br/>NOT BLOCKED-ENV
    end
    Note over G: pinned by accept_without_a_candidate_patch_is_unproven_not_a_harness_fault<br/>gate.rs:483 — asserts verdict INCONCLUSIVE and every veto class != Harness
    Note over G: INVARIANT ADR-2081: Unproven is not Harness — Harness means the evidence could not be<br/>gathered #40;an operational fault#41; — Unproven means there was nothing to test #40;see AB-23.8<br/>VetoClass note, gate.rs:36-44#41;. The two receipt fixes #40;this and AB-23.12#41; land together —<br/>pipefail with the siblings still unresolved would have vetoed every annexe ACCEPT
```

## AB-23.14 ADR-2081 — Step-19 ledger cell provenance

```mermaid
flowchart TB
    R["nightly report text"] --> SF["sanitise_finding#40;report, verdict#41;<br/>agentbox/services/dream-engine/src/verdict.rs:347"]
    SF --> Q0{"0. report_ledger_row_finding#40;report#41;<br/>verdict.rs:387 — the report's own Step-19<br/>ledger table row, a #124;-delimited line,<br/>&ge;12 cells, cells#91;1#93; an ISO date"}
    Q0 -->|"cell#91;3#93; satisfies ledger_cell_ok#40;#41;<br/>verdict.rs:372"| CELL["return that cell verbatim"]
    Q0 -->|"no ledger row, or its cell fails the contract"| Q1{"1. a Finding: line<br/>whose text satisfies ledger_cell_ok#40;#41;"}
    Q1 -->|"ok"| CELL
    Q1 -->|"none"| Q2["2. select_finding#40;report, verdict#41;<br/>#40;the pre-2026-09-07 heuristics#41;<br/>.chars#40;#41;.take#40;80#41;"]
    Q2 --> CELL
    subgraph contract["ledger_cell_ok#40;cell#41; — verdict.rs:372-383"]
        direction TB
        C1["non-empty"]
        C2["&le; 80 chars"]
        C3["does not start with given / see  / gist"]
        C4["does not contain see report / see gist"]
        C1 --- C2 --- C3 --- C4
    end
    Q0 -.-> contract
    Q1 -.-> contract
    CELL --> LEDGER["docs/dream-cycle/LEDGER.md finding cell<br/>see AB-23.10"]
    FULL["sanitise_finding_full#40;report, verdict#41;<br/>verdict.rs:417 — NO 80-char cap, used for<br/>RuVector memory rows and PR bodies"] -.->|"still carries the WHOLE hypothesis"| MEM["memory / PR body"]
    subgraph notes["ADR-2081"]
        direction TB
        N1["RESOLVED: before this, the engine discarded the model's own Step-19 row for the<br/>truncated hypothesis — violating dream-engine's own ledger row contract<br/>#40;finding-hypothesis-leak, PR #10#41;. Pinned by<br/>sanitise_prefers_the_reports_own_ledger_row_cell #40;verdict.rs test#41;"]
        N2["a Step-19 cell that itself leaks the hypothesis, points elsewhere #40;see report#41; or<br/>overruns 80 chars is ignored and the OLDER heuristics apply instead — pinned by<br/>sanitise_ignores_a_ledger_row_cell_that_breaks_the_contract"]
        N1 ~~~ N2
    end
```

## Audit qualification — 2026-09-07

ADR-2081 source corrections above remain **staged** for the supervised nightly loop. Its loaded Nix-store binary requires an image rebuild and process-identity receipt before these source fixes can be called live. This audit did not rebuild or activate it.
