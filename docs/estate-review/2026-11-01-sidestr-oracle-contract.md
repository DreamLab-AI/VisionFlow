---
title: SideStr compatibility-oracle contract and evidence boundaries
date: 2026-11-01
status: in-progress
type: explanation
---

# SideStr compatibility-oracle contract

SideStr has a working reference-based CI design, but a Rust test reported as passed does not necessarily mean its reference comparison ran. Four selected tests return successfully when configured reference directories lack their required files. The latest successful CI run uses real pinned checkouts; its source matches the current implementation. This is an evidence-reporting gap, not a demonstrated failure of that CI run.

This pass assesses every section of local ADR-0001 and corrects panels SR-01.1 and SR-01.2. SideStr is read-only at `9c6de86c657a25ec2f2a189baf63db7dedad40eb`; [26 source fingerprints](evidence/2026-11-01/oracle/source-manifest.json) bind the inspected contract, manifests and test paths. Product code, VisionClaw springs, deployed services and release settings are unchanged.

## Findings and acceptance criteria

| Finding | Evidence and consequence | Acceptance criterion |
| --- | --- | --- |
| **O1: configured-but-missing references can report success** | `sidestr-round/tests/support/interop.rs:18-40` returns `None` when `bin/siding.mjs` is missing, even with all three reference variables set. Its three interop callers return before starting their scenarios. `sidestr-core/tests/pegtweak.rs:406-414,465-467` similarly returns when `lib/pegtweak.mjs` is absent. Fresh native executions report four passes, zero ignored, while printing skip messages. A consumer of aggregate pass counts cannot distinguish these from exercised comparisons. | Keep convenient optional local tests, but introduce a strict oracle mode in which absent configured files fail. Emit and assert reference identity plus suite/scenario participation. Missing checkout, missing module and zero-comparison cases must fail that mode. |
| **O2: decision text understates reference scope** | ADR-0001 decision 1 says four pinned repositories; `.github/workflows/ci.yml:17-30,105-145` pins six, adding Reef and teller. Its review trigger also omits those two. | Update the owner decision and trigger when next revising it; derive any displayed reference list from the workflow or a shared manifest. The workflow is the current executable source. |
| **O3: reference pinning is not a fully frozen build** | Actions and reference commits are pinned, and EVM uses `npm ci`. Rust remains `stable`, Node is major version 22, the runner is `ubuntu-latest`, and Cargo commands omit `--locked` (`ci.yml:35-97,147-158,196,208`). The checked-in lockfile helps resolution but does not make lockfile changes fail automatically. | Separate a moving compatibility lane from a reproducible release lane. The latter should enforce the lockfile and record exact toolchain, target and runner identity. No actual dependency drift is demonstrated here. |
| **O4: successful CI does not establish release enforcement** | Two independent jobs have no `needs` edge; the prior diagram wrongly put quality after oracle. The workflow runs tests, but this pass does not inspect branch rules, publication credentials or a release approval path. | Bind release claims to both successful jobs at the candidate implementation and its explicit oracle participation receipt. Verify required-status or publication enforcement before claiming it exists. |

EVM fixture generation copies required modules, checks their SHA-256 in the generators, runs `npm ci`, regenerates output, checks tracked differences and refuses untracked fixture changes (`ci.yml:186-200`). The overlay generator permits a driver-only local mode when its reference file is absent (`sidestr-evm/tests/oracle/oracle.mjs:24-32`); the CI copy step prevents that particular omission from silently succeeding. The RPC generator instead refuses missing or mismatched reference modules. These controls should be retained.

## ADR-0001 section assessment

| Section | Disposition |
| --- | --- |
| Metadata, status and review trigger | Accepted/complete/live are the owner's recorded axes for the oracle process. Historical CI exists and current implementation CI is successful; “live” is not evidence of an activated Rust chain. Reef/teller are missing from the review trigger (O2). |
| Context | Executable comparison is appropriate for a Rust port across serialisation, state, wire and channel behaviours. The source explicitly documents deliberate restrictions; similarity to the JavaScript source alone would not establish compatibility. The named reference set has since expanded. |
| Decision 1 | Six checkouts are pinned and supplied to the oracle job. Correct the four-repository wording. O1 limits what a Rust test result alone proves; it does not invalidate the actual checkouts observed in CI. |
| Decision 2 | The workflow runs upstream tests, EVM generation and cross-engine suites. Source examples include pegtweak values, mixed Rust/JS rounds and a peg-in metadata regression. This assesses the method and selected paths, not every wire format, opcode or state transition. Full consensus/channel/EVM review remains open. |
| Decision 3 | Both EVM generators run before the Rust suite and fixture drift fails. The practical scope is the EVM fixture corpus; not every other checked-in fixture is shown to regenerate in this workflow. |
| Decision 4 | `sidestr-core/src/lib.rs:170-220` documents stricter genesis validation, record framing, supported script templates and admission behaviour. Unsupported script leaves can be valid upstream and invalid here; “parity” must carry that domain restriction. The source links regression and channel-oracle tests. This pass does not certify the complete departure inventory or rerun all those paths. |
| Decision 5 | `sidestr-core/tests/audit_regressions_0_0_3.rs` retains the missing-address peg-in counterexample and compares both engines. The latest CI log reports that test passing. This establishes a concrete practice, not proof that every historical finding has a regression. |
| Consequences | Rust-only success is insufficient for a parity release. O1 makes explicit participation important; O3 qualifies reproducibility and O4 leaves release enforcement unverified. Node and reference checkouts remain real prerequisites. |
| Verification | The ADR's historical run `36831527612` at `bcbe30b209f8b5703c614bfa832683ad9094ab33` is completed/success, with the listed quality and oracle steps successful. The more recent run below refreshes implementation evidence without rewriting the historical record. |

## Executed checks and CI receipts

[Run 41128639012](https://github.com/DreamLab-AI/sidestr-rs/actions/runs/41128639012), created on 30 October, completed successfully at `d7e39ea2ad0ba46f46510a920912af4aa5516c78`. The [API receipt](evidence/2026-11-01/oracle/ci-receipt.json) retains both jobs and their steps. The [diff to current HEAD](evidence/2026-11-01/oracle/head-delta.txt) contains only `docs/dream-cycle/LEDGER.md`; implementation and workflow bytes are unchanged. No workflow run was returned for current HEAD itself. This is source-equivalent historical CI, not a fresh full-suite rerun.

The [numbered log excerpts](evidence/2026-11-01/oracle/ci-oracle-excerpt.txt) show the peg-out interop test taking 29.01 seconds and the two round scenarios taking 113.73 seconds, along with pegtweak and selected regression results. Those durations support actual interop execution, unlike the immediate returns reproduced locally. Successful test stdout is captured, so these logs do not enumerate every comparison or assertion; no exhaustive oracle-participation count is inferred.

Fresh local commands used Rust/Cargo 1.99.1, `--locked --offline`, an external Cargo target directory and an empty reference directory. [Round output](evidence/2026-11-01/oracle/missing-round.txt) records three reported-pass skips; [pegtweak output](evidence/2026-11-01/oracle/missing-pegtweak.txt) records one. The [reproduction script](evidence/2026-11-01/oracle/reproduce.sh) enables the round features needed to compile those tests. All four exit successfully without performing their oracle comparisons. They receive **zero parity acceptance credit**. No public relay, parent node, operational key or deployed chain was used.

## Comparison and useful next work

Ethereum's execution-spec testing documentation separates fixture generation from client consumption, and distinguishes direct module tests from Engine API integration. SideStr already follows part of that pattern with regenerated EVM fixtures and separate interoperability suites. A useful next step is a small receipt showing which reference, fixture set, comparison domain and consumer actually ran; fixture generation alone should never imply consumer acceptance. This comparison concerns testing structure, not equivalent protocol coverage or maturity. [Ethereum filling tests](https://steel.ethereum.foundation/docs/execution-specs/filling_tests/), [running methods](https://steel.ethereum.foundation/docs/execution-specs/running_tests/running/).

For an integrator, the practical deliverable is a compatibility matrix per crate: reference revision, supported domain, documented departures, exercised scenarios and exact CI evidence. That makes an upgrade decision reviewable without treating a green workspace badge as universal consensus compatibility. Strict participation checks belong with SideStr's existing oracle tooling; reusable estate receipt checks should follow the standing shared-tooling method rather than duplicating validators in consumers.

## Diagram disposition and remaining scope

[SR-01.1](../diagrams/sidestr-rs/01-workspace-oracles-and-live-boundary.md) now distinguishes normal, optional and dev-only dependencies, includes agent/round's missing edges and points Liquid towards neutral reserve. Eight crates are described as publishable from their manifests; this is not a registry census. The [dependency receipt](evidence/2026-11-01/oracle/workspace-edges.json) records every direct local edge.

SR-01.2 now shows the two independent CI jobs and separates reported results from demonstrated reference participation. Only these two panels receive assessment credit. SR-01.3/01.4 retain their earlier activation/runtime boundary and pins; they are not freshly accepted by rendering their sibling panels. ADR-0002, the remaining SideStr panels and full runtime/release enforcement need separate work. The wider estate audit remains active.
