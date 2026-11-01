---
title: Dream-engine nested diagram corpus and failure identities
date: 2026-11-01
status: assessed-with-open-findings
scope: eight dream-engine diagram topics, 24 panels, fingerprint and streak contracts
---

# Dream-engine nested diagram corpus and failure identities

All eight topics and 24 panels in dream-engine's own diagram corpus now have source assessments at `815f963b1cbd6999e78144ce5475005ae1dd1d8e`. The corrections include the formatter stage, gate control flow, automerge refusal, partial replay capture, best-effort reset rollback and caller obligations. Seven new probes reproduce fingerprint and streak limitations. Production code and ADR statuses remain unchanged; these are open findings, not fixes.

This closes the queued source review of this nested corpus. It does not establish complete engine module coverage, deployed-night acceptance or completion of the estate goal. The [AB-23 coverage table](2026-11-01-dream-execution-and-triage.md#panel-coverage-and-user-utility) and the nine local ADR assessments below remain separate evidence. Documentation changes are in dream-engine commit `bfe8c8885a58cc576a18ecb55d4e05e6f3bd37c7`.

## Evidence and limits

| Item | Result |
|---|---|
| Source identity | [Per-topic source manifest](evidence/2026-11-01/dream-corpus-source-manifest.json) records current source hashes and changes from the previous `0f6a669` pin. Selection sources were unchanged but still contained unsupported diagram claims. |
| Baseline gate | [Eight topics, 24 renders, zero citation warnings](evidence/2026-11-01/dream-nested-baseline.log) before correction. This demonstrates why a passing citation/grammar gate cannot replace semantic review. |
| New probes | [Seven pass](evidence/2026-11-01/dream-nested-probes.log); [Rust source](evidence/2026-11-01/dream-nested-probes.rs). They assert observed behaviour, including defects. Only disposable local files and harmless strings were used. |
| Existing selection tests | [64 pass](evidence/2026-11-01/dream-nested-selection-tests.log), including documentation tests for failure-fingerprint and pairwise-gate. The earlier 555-test engine run remains in the candidate assessment; it was not repeated for documentation changes. |
| Final nested gate | [24/24 rendered, zero strict citation warnings](evidence/2026-11-01/dream-nested-final.log), with [machine report](evidence/2026-11-01/dream-nested-final.json). A separate [working-tree citation check](evidence/2026-11-01/dream-nested-worktree.log) also passes. README, coverage and the 51-marker register were regenerated. |
| Visual inspection | All 24 SVGs were inspected using six contact sheets; long labels and the fingerprint lifecycle layout were corrected and re-inspected. SVGs, previews and scripts are in the retained artefact directory named by the receipt below. |
| Estate gate | [Regeneration](evidence/2026-11-01/dream-nested-estate-regeneration.log) and [verification](evidence/2026-11-01/dream-nested-estate-verification.log) cover the unchanged 180-topic, 1,598-panel root corpus, including its declared source revisions. This is structural verification, not completion of its semantic queue. |

The first compilation attempt used a target directory on the noexec `/tmp` mount and could not execute build scripts. Moving the target and fixture temporary directory under the owned VisionFlow worktree resolved this environment failure. The successful logs above are the subsequent runs. No live model, forge, notification, daemon or publication was exercised.

## New findings and acceptance work

Paths refer to dream-engine at the source revision above. The probe source is retained so findings can be converted into implementation acceptance tests.

| Finding | Evidence and consequence | Required acceptance |
|---|---|---|
| DE-C1 / medium: field separators do not make tuples unambiguous | `crates/failure-fingerprint/src/lib.rs:229,357` accepts arbitrary stage strings and hashes unescaped task/separator/stage/separator/cause. Task `alpha U+001F beta`, stage `gamma` aliases task `alpha`, stage `beta U+001F gamma`. The probe observes `Persisting`, not `New`, for the distinct second tuple. This is an encoding alias, not an attack on SHA-256 or proof of an exploited estate input. | Reject separator-bearing names or use an unambiguous, versioned tuple encoding. Preserve migration semantics for existing manifests and test distinct tuples at every field boundary. Increasing digest length alone cannot resolve it. |
| DE-C2 / medium: normalisation is not credential scrubbing | `crates/failure-fingerprint/src/lib.rs:89,118,190` replaces selected path, identifier and digit runs. Harmless `password=canary! token=abcd.efgh.ijkl` survives unchanged and is serialised. SEL-02's former statement that tokens never reach stored causes was false. No real secret was read or exposed. | Treat cause text as potentially sensitive. Define a separate pre-persistence redaction policy, its coverage and retention boundary; test punctuation-separated and short canaries. Do not advertise heuristic grouping as an egress/privacy control. |
| DE-C3 / medium: parsed manifests do not enforce record invariants | `crates/failure-fingerprint/src/lib.rs:467,514` validates JSON shape and kind, not fingerprint correspondence, normalised cause, uniqueness, ordering or step relations. A constructed record with changed task and raw cause round-trips unchanged under its old fingerprint. | Validate semantic invariants at the trust boundary or explicitly distinguish trusted in-memory records from checked persisted inputs. Reject or quarantine incompatible records rather than silently treating them as trustworthy history. |
| DE-C4 / medium: repeated calls can simulate successive nights | `crates/dream-core/src/streak.rs:68,117` increments a manifest step for each call and accepts no night identity. Two `record_night` calls for the same fixture immediately produce a stuck-harness signal. Generic `advance` also increments count when given the same step twice, as its API documents. | Make the host's once-per-night responsibility explicit, or deduplicate on a persisted night key. Test retries, duplicate delivery and cross-process writers. This probe demonstrates duplicate calls, not a confirmed duplicate call in the current deployed driver. |
| DE-C5 / low: extreme stored step panics in debug builds | `crates/dream-core/src/streak.rs:75` evaluates `prev.step + 1` without a checked operation. A manifest with `u64::MAX` is structurally permitted and panics under the tested debug build. No release overflow behaviour is asserted. | Reject impossible step values on load or use checked progression with a typed recovery path. Keep this separate from the saturating per-record count inside `advance`. |
| DE-C6 / semantic limit: different failure meanings deliberately merge | The probe shows `HTTP 401` and `HTTP 500` share an identity, and distinct suffixes after the 200-character bound disappear (`crates/failure-fingerprint/src/lib.rs:146,190`). This follows the normaliser's design, but limits claims about novel root causes. | Measure false merges and splits on a labelled failure corpus. Keep structured error codes/stages where meaningful and show the grouping explanation to reviewers. A matching fingerprint alone must not be presented as proof that a regression is already known. |

The C1 fixture exercises the general public API. The gate usually fixes the stage to one evaluator name, and streak detection uses the constant `harness` stage; those caller restrictions narrow practical collision paths. Their own name validation and cross-evaluator use must be checked before claiming an operational bypass. C2 similarly demonstrates persistence of a harmless shape, not a production secret incident.

Previously verified findings are now reflected in the owner corpus: [candidate confinement, duplicate identities, receipt collisions and caller-trusted evidence](2026-11-01-dream-candidate-and-evidence.md); [process recovery and source supply](2026-11-01-dream-recovery-and-source-supply.md); [notification/forge contracts and draft-to-ready partial failure](2026-11-01-dream-forge-and-seam-contracts.md); and [dependency/provenance limits](2026-11-01-dream-migration-and-provenance.md). The highest-priority formatter and repair-egress findings remain open despite clearer diagrams.

## Panel dispositions

Each row is a source assessment, including panels whose mechanism remained correct. The source pin is common to all rows. The files are under dream-engine `docs/diagrams/`.

| Panel | Disposition |
|---|---|
| CRT-01.1 | Five-member versioned dependency DAG confirmed. Corrected source anchors and removed the false four-library/five-member contradiction. |
| CRT-01.2 | Licence metadata, deny policy and CI confirmed. Removed absolute downstream/provenance guarantees; declared MSRV still lacks a CI floor job. |
| CRT-02.1 | Five interfaces and shipped implementations confirmed. No-Result notification does not contain panics or impose a deadline. |
| CRT-02.2 | Host orchestration remains a contract. Corrected evaluator anchors and identified source gathering/repair as core LLM callers; full-night defaults require a host. |
| GAT-01.1 | Outcome classes confirmed, scoped to receipt grading and the precedence described in panel 2. |
| GAT-01.2 | Added the non-ACCEPT path, untrusted-tree bypass and optional config comparison. A parsed non-ACCEPT verdict remains the model's verdict. |
| GAT-01.3 | Added current formatter and relocation stages, normalised fallback recheck and conditional gate recheck. Recorded control-host execution and stale prepared-hash risks. |
| GAT-01.4 | Manifest identity checks confirmed. Explicitly separated them from raw-outcome reclassification and execution attestation. |
| GAT-01.5 | Bounds and shell-word parsing confirmed. Qualified total-size versus per-line checks, indirect dependencies and equal-input policy reconstruction. |
| NIT-01.1 | Nine phases confirmed as intended order; `advance` permits jumps. Persistence-before-delivery remains a caller contract. |
| NIT-01.2 | Begin branches confirmed; absent and changed-run-id paths can both be Fresh. Zero-budget first-attempt and non-fsync persistence limits added. |
| NIT-01.3 | Night identity and hard-link reset confirmed. Removed all-or-nothing rollback claim; rollback errors and crash windows remain. |
| NIT-01.4 | Cooperative lock contract confirmed. Qualified process-death release for inherited descriptors and readable journal assumptions. |
| NIT-02.1 | Capture now displays sequential writes and receipt-directory replacement, rather than implying a bundle transaction. |
| NIT-02.2 | Decision diff, receipt reclassification and guard checks confirmed. Recorded evidence can be re-judged; execution and producer authenticity are not proved. |
| NIT-03.1 | Decision order confirmed. Self-reference detection is a set of heuristics; required-check names and a URL do not bind the evaluated head. |
| NIT-03.2 | Added Disabled/Refused outcome branch. Corrected the false claim that every failed enable request leaves the PR a draft. |
| SEL-01.1 | Fault exclusion and strict greater-than guard boundary confirmed. Labelled 0.5/one-comparable thresholds as defaults. |
| SEL-01.2 | Six policies confirmed. Corrected TrainDev to require both Select decisions; held-out Inconclusive remains Inconclusive. |
| SEL-01.3 | Additive veto mapping confirmed. Recorded last-duplicate-wins identity collapse and validation skipped for an empty test map. |
| SEL-02.1 | Sequential normalisation and hash construction confirmed. Removed separator and secret-filter guarantees; C1/C2/C6 reproduced. |
| SEL-02.2 | Replaced an ambiguous lifecycle layout with the actual map/classification flow. Normalisation applies to generated records, not every accepted manifest; C3/C4 reproduced. |
| SEL-02.3 | Baseline step 1/candidate step 2 comparison confirmed. Novel fingerprints add evidence vetoes; grouping limitations still apply. |
| SEL-02.4 | Any verdict can update history; only the matching BLOCKED-ENV classification is stuck. Any read error starts empty. Duplicate-call and overflow limits reproduced. |

## Decision coverage and ADR-DE-004 supplement

| Local decisions | Existing section assessments |
|---|---|
| ADR-DE-001, ADR-DE-002 | [Migration and provenance](2026-11-01-dream-migration-and-provenance.md#adr-de-001-section-dispositions). |
| ADR-DE-003, ADR-DE-004, ADR-DE-009 | [Candidate/evidence assessment](2026-11-01-dream-candidate-and-evidence.md#adr-section-dispositions). |
| ADR-DE-005, ADR-DE-006 | [Forge and seam contracts](2026-11-01-dream-forge-and-seam-contracts.md#adr-de-005-section-dispositions). |
| ADR-DE-007, ADR-DE-008 | [Recovery and source supply](2026-11-01-dream-recovery-and-source-supply.md#adr-de-007-section-dispositions). |

All nine have dated section dispositions; that does not mean their proposed guarantees are implemented or accepted. ADR-DE-004 was reread in full for this supplement:

| Section | Supplementary disposition |
|---|---|
| Metadata/review trigger | The record calls for review upon a fingerprint collision in practice. C1 supplies a deterministic local API counterexample. It is not a naturally observed production collision, but it is sufficient to revisit the encoding claim. |
| Context | Separating infrastructure faults from candidate losses remains useful. Collapsed test identities and grouped causes limit that inference. |
| Decision: policies | Six implementations exist. TrainDev's wording “does not reject” is too broad: held-out Inconclusive does not Select. Twenty cases is a configurable gate default; the pure statistical policies do not impose that floor themselves. |
| Decision: fingerprints | SHA-256 comes from the sha2 crate; the defect is field encoding, not a hand-written primitive. Normalisation and truncation deliberately lose distinctions, in addition to the 64-bit collision risk. |
| Decision: reference arithmetic | Existing policy/statistical tests pass. This pass did not independently recompute a wider numerical reference grid, so it adds no stronger accuracy claim. |
| Implementation status | In-progress remains appropriate: held-out policy use is absent from the gate builder and the new identity findings are unresolved. |
| Consequences | Fault handling and grouping are bounded by the evidence supplied. A 64-bit width argument does not address equal input byte strings or semantic merges. Below min_cases, the gate builder also omits Margin, despite the suggestion that small suites lean on it. |

## Comparison and useful next demonstration

Sentry is an adjacent failure-triage reference, not a patch-selection benchmark. Its [grouping documentation](https://docs.sentry.io/product/data-management-settings/event-grouping/) describes versioned grouping behaviour, visible grouping explanations and custom fingerprint rules. Its [data-scrubbing documentation](https://docs.sentry.io/security-legal-pii/scrubbing/) treats scrubbing as a separate concern and distinguishes server-side options from SDK handling. Those are useful design separations for dream-engine's smaller deterministic grouping component. This comparison makes no claim that Sentry or dream-engine prevents every leak or false merge.

A useful next demonstration would run a labelled set of recurring, genuinely novel and intentionally similar failures through versioned grouping. Report false merges, false splits, duplicate-night alerts, reviewer time and canary persistence separately. Preserve structured error categories and show why each cause grouped with its predecessor. In a nightly maintainer product, those explanations would make rejection and stall notices more useful than an unexplained short hash. This is a proposed utility evaluation, not measured user benefit or a market ranking.

The [retention receipt](evidence/2026-11-01/dream-nested-retention.json) inventories local artefacts and hashes. The wider estate audit remains active.
