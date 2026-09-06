---
title: Self-improvement and the evidence boundary
status: source-and-local-test-verified
date: 2026-09-04
type: explanation
---

# Self-improvement and the evidence boundary

The dream system has a useful organisational purpose: reserve time for neglected questions, preserve failed hypotheses, and turn findings into reviewable proposals. It already implements more than a scheduled prompt. The agentbox service dispatches checkouts, captures evaluator output, assembles context, records reports and witnesses, maintains a ledger and can create draft PRs. Its main weakness is the meaning assigned to an accepted result: the current execution order does not test the patch subsequently proposed by the model.

This chapter distinguishes the toolkit repository from the service that consumes the estate's configurations. [Receipts](evidence/dream-snapshot.json) record source hashes, **78 passing Rust library tests**, **125 passing TypeScript tests**, and [synthetic parser probes](evidence/dream-probes.py). No nightly cycle, SSH dispatch, provider request, database write, forum publication or PR creation was executed for this review.

## Three paths share one vocabulary

The local [dream-machine repository](../../../dream-machine/FORK.md) is DreamLab's dream-engine fork. Its TypeScript packages provide configuration compilation, scheduling helpers, ledger handling, witnesses, memory and a CLI. Compiling instructions for an agent is distinct from implementing each instructed action in a runtime.

Its [GitHub Actions nightly script](../../../dream-machine/scripts/dream-nightly.mjs) makes that limit explicit: it performs research and hypothesis generation, records `INCONCLUSIVE`, and does not claim candidate evaluation. Its [CLI entrypoint checker](../../../dream-machine/packages/cli/src/entrypoint.ts) also distinguishes non-zero exits from suspicious silence. That checker is wired to the `verify-entrypoint` CLI command; it is not automatically invoked by the Rust service's evaluator loop.

The operational estate path reviewed here is [agentbox/services/dream-engine](../../../project/agentbox/services/dream-engine/src/engine.rs). It independently compiles a prompt and orchestrates remote evaluation and model calls. Tests for the toolkit therefore do not establish the service's composed behaviour, and the toolkit's research-only policy does not constrain the service's acceptance path.

## Real receipts, but no candidate evaluation loop

The service's `cycle_repo` order is consequential:

1. Load local configuration and compile the prompt.
2. Archive the repository's `HEAD`, copy it and configured sibling repositories to the annexe, and check that the extracted working directory exists and is non-empty.
3. Run the build and all configured evaluator entrypoints.
4. Add their output, recent ledger rows and context to the prompt; ask an LLM for a report.
5. Parse its verdict and bind the report to a commit with a witness.
6. On `ACCEPT`, optionally extract a `dream-patch`, apply it in an isolated worktree, commit it, push a branch and open a draft PR.

There is no intervening application and evaluation of the candidate before step 5, nor an evaluator rerun after applying it in step 6. The [prompt](../../../project/agentbox/services/dream-engine/src/compile.rs) nevertheless instructs the model to build a candidate, compare it with its parent and accept only when tests are green. The [model adapter](../../../project/agentbox/services/dream-engine/src/llm.rs) is a request/response interface; this orchestration does not give the model a shell/tool loop with which to carry out those instructions.

An `ACCEPT` can therefore represent a model's judgement about collected baseline evidence and its proposed change. It is not proof that the emitted patch passed the named evaluators. A finding that requires no patch may still be useful, but the record should distinguish that case from a validated change.

**Gap:** either label this path as evidence-informed proposal generation, or implement a bounded candidate loop that applies a frozen patch, runs the required checks, compares results and binds acceptance to that candidate tree. Human review should receive that distinction before being asked to merge.

## Evaluator failure does not deterministically veto acceptance

[Dispatch](../../../project/agentbox/services/dream-engine/src/dispatch.rs) handles a failed build as an error, which aborts the cycle. Evaluator failures are treated differently: each SSH error becomes a `BLOCKED: ...` output string and the loop continues. A zero-exit evaluator's stdout is accepted as evidence without a typed success/failure result. Successful commands' stderr is not preserved by this helper.

This resolves the open question from [the four VisionFlow probes](canon-and-verification.md): their `FAIL` text with exit zero reaches the model as text. It is neither an automatic pass nor a deterministic rejection. Even a non-zero evaluator does not independently force `INCONCLUSIVE`; its blocked status becomes material for the model to interpret. The preflight `BLOCKED-ENV` path checks checkout presence, not the validity or availability of every evaluator.

The [verdict parser](../../../project/agentbox/services/dream-engine/src/verdict.rs) prioritises an explicit final `VERDICT:` line, then a section, field or last standalone keyword. Existing regression tests correctly protect an explicit inconclusive verdict from an earlier acceptance keyword. Our probes also establish these limitations:

| Synthetic report | Actual parsed verdict |
|---|---|
| `Evaluator: FAIL` followed by `VERDICT: ACCEPT` | ACCEPT |
| All evaluators blocked, followed by `VERDICT: ACCEPT` | ACCEPT |
| `No evidence justifies ACCEPT` with no explicit verdict | ACCEPT |
| Earlier acceptance mention, then `VERDICT: INCONCLUSIVE` | INCONCLUSIVE |
| Unknown explicit verdict followed by an example containing ACCEPT | ACCEPT |

These are parser-level reproductions, not observed production misclassifications. Combined with the inspected caller, they show that acceptance is not conditioned on a machine-checked evaluator contract. Required checks need typed availability, exit status and semantic results; invalid or absent verdict fields should not borrow an acceptance word from explanatory prose.

## A witness binds bytes, not the whole experiment

The [witness implementation](../../../project/agentbox/services/dream-engine/src/witness.rs) hashes the report and commit deterministically. Its tests cover changed reports, changed commits and reference vectors. This is a useful integrity primitive, but its interpretation must stay narrow.

Dispatch archives `HEAD` first; `cycle_repo` reads `HEAD` again after evaluation, and [patch persistence](../../../project/agentbox/services/dream-engine/src/persist.rs) creates its worktree at the then-current `HEAD`. These are separate reads, not one frozen revision passed through the transaction. If another actor advances the branch, the archived tree, witnessed commit and patch base can differ. This is a source-established race possibility; it was not induced in a running night.

Configuration is read from the local worktree, while repository content is archived from committed `HEAD`. Sibling archives also lack a common revision manifest. The witness contains the report and commit, not a digest of every raw receipt, configuration, sibling revision, evaluator executable and candidate tree. A cryptographically valid witness consequently does not prove that all those inputs match its implied experiment.

The service preserves receipt sidecars and context-governance recovery pointers, which are useful foundations. Freeze a run manifest before dispatch and carry its identifiers through reports, candidate checks, persistence and later verification.

## Human promotion is preserved; recovery remains uneven

The persistence helper uses an isolated worktree and opens a **draft** PR. The existing isolation test passes and confirms that the main worktree's uncommitted file survives. This is a real control: the reviewed service does not automatically merge the patch.

Push or PR failure is fail-open and leaves a local branch for recovery. The ledger reports a PR URL, branch reference or `PERSIST-LOCAL`, but branch naming uses deep topic and date. Repeating the same topic on the same date can collide with an existing branch. Local worktree paths use that branch name too, so separate repositories with the same topic/date share the temporary name. The singleton process lock reduces concurrent service execution, but these names are not durable run identifiers.

Operational history also has two limits. [Discovery](../../../project/agentbox/services/dream-engine/src/config.rs) sorts nominated repositories alphabetically and the engine truncates the eligible list at the configured cap. There is no rotating offset: later repositories can starve while the earlier ones remain eligible. The [nightly loop](../../../project/agentbox/services/dream-engine/src/main.rs) remembers the last run date in process memory; restarting inside the window can repeat a night. Reports and memory keys use date plus repository, so repeated runs can overwrite artefacts even while ledger rows accumulate.

These are source-level recovery and fairness findings. A safe local restart/failure simulation remains needed before claiming deployed loss or starvation. Useful measures would include eligible-to-run delay, required-evaluator coverage, candidate rerun rate, review conversion and duplicate-run recovery, alongside counts of completed nights.

## Memory integration differs between toolkit and service

The toolkit's [memory package](../../../dream-machine/packages/memory/src/index.ts) can probe for RuVector WASM availability but still instantiates flat-file storage. Explicit selection can label that object `ruvector-rvf`; the code comments acknowledge that a real RVF/HNSW implementation is not present there.

The Rust service instead has a [concrete embedding and PostgreSQL adapter](../../../project/agentbox/services/dream-engine/src/ruvector.rs). It requests `bge-small-en-v1.5`, checks for 384 dimensions and upserts a finding with source metadata. That implementation is distinct from the mandatory shared-memory MCP used by this review. Its local unit tests do not prove live embedding availability or retrieval compatibility. Inconclusive findings are stored at lower importance; blocked-environment nights are excluded, which usefully separates operational faults from attempted learning.

The next acceptance criterion is an experiment record whose interpretation survives every hand-off: what was proposed, what ran, which tree was tested, which checks were unavailable, what a human approved and what ultimately changed. The current system already supplies many of those pieces. Their binding and enforcement require more work than the existing green unit suites establish.

## Decision closeout reconciliation — 2026-09-04

All three dream-machine decision candidates and agentbox ADR-2024 now carry scoped CP-01/07/08 acceptance conditions. Toolkit, operational inner loop and proposed outer loop remain distinct. The parser probes were rerun against unchanged source hashes and reproduce the recorded verdicts; no live nightly cycle ran. Outer-loop optimisation depends on first establishing trustworthy candidate evaluation and deterministic rejection in the inner service.

## Evaluator readiness before scheduling

Historical agentbox ADR-072 remains proposed. Current Rust DreamConfig validation checks repository/slot names and rejects Darwin commands missing an accepted sandbox substring. The [extracted unchanged validation method](evidence/dream-admission-probe.py) accepts an empty evaluator map, an inline echo command and a nonexistent script command; it rejects the Darwin negative control. [Four results](evidence/dream-admission-probe.json) establish admission scope only. No command was executed and no dream night was scheduled.

The engine loads that configuration, selects the slot by day integer modulo slot count and compiles its prompt. It later passes the configured evaluator map to the HP dispatch path. The selected deep is not matched to a required evaluator in this admission logic. Prompt compilation also permits an empty evaluator section. Presence of a command string is therefore distinct from checked-in ownership, target-environment executability and a decidable receipt.

This does not erase the implemented Darwin guard or the historical repair from quoted inline commands to script files. It means those narrower controls do not enforce ADR-072's evaluator-before-schedule policy for every deep. The dream-machine sibling's schedule package constructs a cloud routine body; its existence is not evidence of admission enforcement by this container-local Rust engine. Scheduling paths need separate acceptance evidence.

CP-01/07/08/09 requires an explicit per-deep evaluator association, versioned script and dependencies, target-environment readiness evidence, and a deterministic no-ready-evaluator disposition such as the proposed HANDOFF. Test empty, missing, inline, wrong-deep, unavailable dependency and ambiguous-output cases before admitting work. Bind readiness to the same source/toolchain/environment used for the night. Keep this gate distinct from evaluating the candidate after a patch and from deciding whether successful evaluation authorises promotion. Existing candidate-ordering/verdict requirements still apply. No SSH, evaluator, model or schedule operation ran.
