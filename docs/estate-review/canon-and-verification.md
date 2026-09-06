---
title: Canon implementation and verification gaps
status: source-and-local-probe-verified
date: 2026-09-04
type: explanation
---

# Canon implementation and verification gaps

VisionFlow has a concrete implementation of its coordination role: a static publishing path, count reconciliation, fixture tooling, release inventory and nightly evaluators. Their guarantees are narrower than some of the surrounding prose implies. The most useful improvement would be to make each claim carry an explicit verification scope and to exercise rejection paths as carefully as successful ones.

This chapter assesses the checked-out files and isolated local commands. It does not report GitHub Actions status or production behaviour. The [snapshot](evidence/snapshot.json) records the committed bases, selected source hashes and executable receipts.

## Publishing is implemented; publication quality is a separate path

[website/build.sh](../../website/build.sh) copies static files, writes the custom domain and stages images. [.github/workflows/deploy.yml](../../.github/workflows/deploy.yml) builds and uploads `website/dist`, then deploys that artefact through GitHub Pages actions. This supports the [baseline's](../BASELINE-visionflow.md) account of a static HTML/CSS/JS site.

[package.json](../../package.json) defines a broader `verify` sequence: build, check browser sidecar, run Playwright. The deployment workflow inspected here does not invoke that sequence; its deploy job depends on its build job. Therefore the workflow definition alone does not establish that accessibility, browser behaviour or performance checks block publication. External branch rules or other controls could impose constraints, but were not inspected.

**Assessment:** a simple static build is well matched to a public explanation site. The outstanding coordination question is which quality failures actually prevent publishing and which merely produce a separate report. This matters because the site's content is part of the product's claim surface.

## Drift checking works within a limited scope

[scripts/drift-counter/drift-counter.mjs](../../scripts/drift-counter/drift-counter.mjs) queries an agentbox skill counter and counts the ontology bridge's tool registry. Its [allowlist](../../scripts/drift-counter/allowlist.json) identifies the prose locations to check. This is a concrete way to bind descriptive claims to an implementation source.

The captured `node scripts/drift-counter/drift-counter.mjs --json` invocation returned exit **1**:

- The local source reported **126 skills**, while checked prose sites stated **124** or **115**.
- The ontology bridge count was **12**, and the existing matched sites agreed.
- Two entries referenced the missing `docs/ADR-002-ecosystem-alignment-governance.md` path. The documentation index now points to its archived location.
- The ontology-class axis was unavailable and unenforced; the roster axis remained planned.

These figures describe one local observation, not a newly declared estate-wide canonical count. In particular, agentbox had tracked worktree changes. The receipt preserves that limitation.

The implementation also makes the scope explicit: it checks named files and patterns, not every claim in the corpus. The workflow checks out agentbox without a fixed `ref`, while the allowlist excludes the book from whole-file scanning. Consequently, a local or CI result depends on the source revision queried, and it cannot certify every number in the book or every sibling README.

**Assessment:** the failure is evidence that the mechanism detects some drift, not evidence that no mechanism exists. Its own allowlist must evolve with document moves. Long-lived snapshots and historical text also need dated values distinguished from live counts, so maintenance does not erase history merely to make a counter green.

## Fixture CI cannot establish cross-repository parity as configured

[.github/workflows/fixture-drift.yml](../../.github/workflows/fixture-drift.yml) checks out only VisionFlow. Its locate step looks for a local canonical corpus; its no-fixture branch succeeds and explicitly explains that this is a local convenience tool, not an enforcing cross-repository gate. The workflow header documents the canonical corpus's move into VisionClaw's `tests/fixtures`.

There is a further configuration detail: the manual `canonical_dir` override is applied in the drift step, but that step is conditional on the earlier locate step finding a canonical directory. An override by itself cannot rescue the no-canonical branch in this definition.

The [baseline](../BASELINE-visionflow.md) lists fixture tooling in its CI-gate account. Readers should interpret that against the workflow's narrower behaviour: a workflow file exists, but a green no-fixture run is not a parity check. Neither the workflow inspection nor this review asserts that current sibling fixtures differ; no cross-repository comparison was run here.

**Assessment:** the eventual gate should pin the canonical and consumer revisions, enumerate what it compared, and distinguish “no comparison” from “comparison passed”. The local script remains useful, but its existence cannot close the federation evidence gap.

## Release inventory trails the public estate

[scripts/generate-release-manifest.sh](../../scripts/generate-release-manifest.sh) emits a `local-draft` manifest covering VisionFlow, VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum and dreamlab-ai-website. It records local HEAD, branch and dirty state. Its use of Git rather than a `.git` directory check correctly handles submodule/worktree-style Git directories.

However, the current [README](../../README.md) adds Loom, knowledgeGraph and dream-engine to the architecture. These do not appear in the generator. Nor does the generator record ontology generation, model configuration or vector index compatibility. This is a source observation about the generator, not a claim that no other release artefact exists anywhere in the estate.

**Assessment:** coordinated reproducibility now needs more than the older code-repository list. Source code, ontology data, retrieval configuration and evaluation artefacts can evolve independently. An eventual release record should describe the combinations actually consumed and the evidence for their compatibility. Expanding the list alone would not prove integration.

## Dream evaluators have ambiguous failure contracts

The [dream configuration](../../dream.config.json) names four evaluators and sets `autoMerge` to false. The [ledger](../dream-cycle/LEDGER.md) records nightly outcomes and witness identifiers. Those records establish that an evaluation process is described and logged; they do not alone establish which mutations were promoted or whether each evaluator's failure was consumed correctly.

The [receipt collector](evidence/collect.py) copies the current evaluator scripts to a disposable directory and supplies broken fixtures. It does not alter the website or launch a nightly cycle.

| Evaluator | Deliberate defect | Recorded output | Exit code |
|---|---|---|---|
| `dream-link-check.sh` | Image refers to absent `missing.png` | One missing reference; `LINK-INTEGRITY-FAIL` | 0 |
| `dream-meta-tags-scan.sh` | Required metadata absent | Four required items missing; `META-SCAN-FAIL` | 0 |
| `dream-structured-data-scan.sh` | Malformed JSON-LD | One parse error; `SD-SCAN-FAIL` | 0 |
| `dream-build-check.sh` | `index.html` absent | `BUILD-FAIL` | 0 |

The source explains the behaviour. Link, metadata and build scripts finish with successful `echo` commands on their failure branches. The JSON-LD scanner logs a failure but does not set a nonzero Node exit status. Several missing-input branches also explicitly exit zero.

**Established result:** process status alone cannot distinguish these failures from successful evaluations. **Consumer follow-up:** the subsequent [self-improvement review](self-improvement.md) traces engine verdict handling and candidate identity. These script probes alone do not prove that the nightly system promoted broken work.

The configured build and link commands also pipe output through `tail`. That is another boundary to trace: which output survives, whether pipeline failure status propagates, and whether the engine retains the full underlying receipt. Use the later [consumer assessment](self-improvement.md) for verdict and candidate-boundary findings; the original script receipt remains a bounded local probe.

**Assessment:** use one explicit evaluator verdict contract across the scripts and the engine, and verify it with deliberate defects through the actual consumer. Human promotion remains a separate control; it works best when the evidence is mechanically unambiguous.

## What this says about the canon

The canon's strongest design choice is treating self-description as something that can fail a check. The practical gap is the distance between that idea and enforcement across the estate: some axes are unavailable, some comparisons never run in CI, the release boundary omits newer components, and evaluator process status can disagree with evaluator text.

These findings warrant follow-up implementation work, but this user-requested review records them rather than silently changing the systems being assessed. The [investigation ledger](investigation-ledger.md) defines the next evidence needed and preserves the broader estate scope.

## Decision register and reader navigation

Agentbox's consolidation is partly implemented: operative 2001-series records and archived predecessors coexist. Its docs entry page still called the old reference shelf authoritative and linked to pre-move paths. This pass repairs links when exact archive counterparts exist and gives readers an explicit operative-versus-historical route. This is navigation repair, not complete historical reconciliation.

The [validator receipt](evidence/adr-navigation-snapshot.json) reports six current stale records: 2012, 2013, 2019, 2020, 2023 and 2030. Regeneration stops, so the generated table still contains earlier implementation declarations. Individual amended records and the estate inventory carry the current declarations; the table must not be used as fresh acceptance evidence.

An isolated actual-generator fixture with an intentionally stale README passes `--check`. Source shows that this mode does not compare generated output. The optional verified_paths gate compares commit-to-HEAD changes; uncommitted changes and the meaning of the recorded verification remain outside that check. A full SHA by itself does not enable path staleness validation. The template comment is corrected accordingly.

CP-01/08 requires semantic re-verification before updating baselines, index generation/comparison as its own CI check, current-worktree evidence, and historical disposition/navigation coverage. ADR-2001 retains partial/staged status. A thin decision core can link to larger evidence and closeout material without concealing unresolved work.

## VisionClaw operative-pack coverage and baseline debt

The operative series now has closeout extensions through ADR-2042, including ADR-2031's terminal tombstone disposition. This is assessment coverage, not a declaration that every decision is implemented or accepted in deployment. ADR-2001 retains partial/staged status. The user-requested estate closeout extends records beyond the original one-page preference; historical verification remains distinct from dated current evidence.

The [actual current validator run](evidence/dev-docs-closeout.json) exits 1 with four stale records: ADR-2004 and ADR-2005 govern a changed Cargo.toml; ADR-2008 and ADR-2027 govern changed Compose input. README is not regenerated. Changing implementation status or adding an annex does not refresh these baselines. The generator checks commit-to-HEAD governed paths, not current uncommitted source content; records without verified_paths have weaker mechanical coverage. Its check mode validates records without comparing the generated README with existing contents.

CP-01/08/09 requires semantic re-verification of each governed path before accepting a new baseline, then generator and navigation checks. Reconcile generated status tables with reviewed records, explicit historical supersession and all surviving links. Preserve frozen history as history, with a disposition for each candidate; do not manufacture current verified commits merely to obtain a green index. Existing source receipts make this review reproducible but do not complete historical/upstream reconciliation.


## VisionFlow operative closeout scope

All seven operative VisionFlow records now carry dated closeout extensions linking their implemented mechanisms to CP-01/08/09 acceptance. Existing historical verification and decision axes are preserved. ADR-2005 remains partial and ADR-2007 proposed/partial/staged; the complete mechanism labels on other records do not declare publication quality, historical reconciliation or whole-estate acceptance complete.

The [current local receipt](evidence/canon-operative-closeout.json) records nine source hashes, a successful seven-record validator check and a successful browserless check of all ten committed diagram baselines. No website build, browser re-render, hosted CI, branch-policy inspection or deployment ran. In particular, checking committed diagrams does not compare them with their sources, and validating ADR fields does not establish semantic completeness.

The two engineering decisions remain a separate sequence requiring section-level assessment. CP-01/09 must reconcile their proposed/deferred obligations with actual executors and authority boundaries; moving the canon to 2xxx numbers did not retire engineering ADR-004/005.
