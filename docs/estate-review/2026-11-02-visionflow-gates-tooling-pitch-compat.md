---
title: VisionFlow gates, diagram tooling, pitch pipelines and compatibility
date: 2026-11-02
status: assessed
type: explanation
---

# VisionFlow gates, diagram tooling, pitch pipelines and compatibility

I assessed all 46 panels of four VisionFlow topics by source tracing at VisionFlow `5f8b1da`. The topics are 05 governance gates (13 panels), 06 diagram tooling (15), 07 pitch, presentation and reports (11), and 08 compatibility and dependency direction (7). Topic 08 was also traced at VisionClaw `86cdd76`, Agentbox `c57d9c0` and nostr-rust-forum `c39f41f`. Results: 19 panels are accurate, 20 were corrected, 7 are accurate with limits, and none is left inaccurate. All four topics were restamped and every citation re-resolved, and each passes the strict per-topic gate.

The most important finding is that two topics were already wrong when they were last stamped:
- **VF-05:** the topic says `tests/gates/run-all.sh` runs in no workflow. A fourth `harness-fitness-gates.yml` job, `gate-suites`, has run it since `5a3107d` (2026-10-14).
- **VF-07:** the topic describes LaTeX intermediates, arXiv tarballs and deck logs as committed. These were working-tree observations; those files have been git-ignored since July.

## Findings

**VFG-01 — `run-all.sh` is in CI, but only under a path filter.**
- **Trigger:** VF-05.7, VF-05.11 and VF-05.13 said `run-all.sh` was "wired to NO workflow" and that `harness-fitness-gates.yml` has three jobs.
- **Evidence:** job `gate-suites` (`.github/workflows/harness-fitness-gates.yml:178`, `:206`) checks out Agentbox at the drift-counter pin and runs the aggregate.
- **Consequence:**
  - The suites it aggregates run in CI only when the workflow's path filter fires (`harness-fitness-gates.yml:4-22`).
  - The augmentation-citations suite still skips its real-checkout assertion there, because the runner has no sibling checkouts (`tests/gates/augmentation-citations.test.cjs:77`).
  - The graded matrix is therefore checked against real code only by a local run.
- **Status:** corrected in VF-05.1, VF-05.7, VF-05.11 and VF-05.13. The remaining local-only gap is recorded as Debt.

**VFG-02 — Bare line references were checked against the wrong file.**
- **Trigger:** VF-05.13's suite node listed ten defect lines as bare `:35 … :72`.
- **Mechanism:** the generator binds a bare reference to the most recent path cited earlier in the diagram. That path was `scripts/check-augmentation-citations.cjs`.
- **Consequence:** the gate "verified" ten line numbers against the checker, where they mean nothing. The numbers are correct for `tests/gates/augmentation-citations.test.cjs:35-72`.
- **Status:** corrected by naming the suite on the first reference.
- **To close:** this is a generator hazard. A bare list following a different file passes silently, and only a reader catches it. A symbol- or path-scoped bare reference would close it in tooling.

**VFG-03 — The ADR ratchet's backlog rule has expired and is untested; its evidence rule is enforced and tested.**
- **Backlog rule:** `adr-ratchet.sh` defaults `ADR_RATCHET_UNTIL` to 2026-10-20 (`scripts/adr-ratchet.sh:44`). `adr-index.yml` sets no value (`.github/workflows/adr-index.yml:46-49`), so the "added undecided must close a proposed record" rule now only reports (`scripts/adr-ratchet.sh:118-121`).
- **Evidence rule:** the second, unexpiring rule is enforced. A record closed to `complete` or `live` without `evidence_class` and `acceptance_boundary` fails the step (`scripts/adr-ratchet.sh:80-111`). It has a suite, `tests/gates/adr-ratchet-evidence.test.sh`, listed in `run-all.sh:29`.
- **Status:** VF-05.14 and the VF-05.13 Debt are corrected.
- **To close:** either an owner decision to retire or renew the backlog rule, or a suite that drives an added-undecided record through it.

**VFG-04 — VF-07 recorded working-tree artefacts as committed.**
- **Claims:**
  - VF-07.8 said `main.aux`, `main.bbl`, `main.log` and `main.toc` are committed.
  - VF-07.8 said each pitch ships `.aux`, `.log` and `.out` beside its source.
  - VF-07.9 reported `ARXIV.md` as contradicted by "two committed tarballs".
  - VF-07.1 and VF-07.8 dated `texput.log` against `dist/arxiv-2026-07-08.tar.gz`.
  - VF-07.10 named `render*.log` files as the record of past runs.
- **Evidence:**
  - At `5f8b1da`, and at the earlier stamp `4d361b5`, only `presentation/report/main.pdf` and the two pitch PDFs are tracked.
  - The intermediates, `dist/` and the deck's logs, `slides/` and `jpg/` are ignored (`.gitignore:17-38`, `presentation/pitch-deck-2026-07/.gitignore:1-3`). The ignore rules date from 2026-07-08, and the last tracked intermediates were removed in `8d4990d` (2026-07-10).
- **Status:** corrected. `ARXIV.md:13` and `ARXIV.md:18` are accurate.

**VFG-05 — `texput.log` in `sources:` (Drift).**
- **Trigger:** VF-07 listed the generated, git-ignored `texput.log` (`.gitignore:48-50`) in `sources:`. It exists in no commit, so the source-path check failed in every clean checkout.
- **Change:** the entry was dropped. `.gitignore` is now listed as the committed evidence that the log is deliberately untracked; no committed file can carry its content.
- **Allowlist:** the five citations (VF-07.1 line 1; VF-07.8 lines 1, 5, 10 and 21) are byte-identical. They still resolve as allowlisted, and the `--only` run prints five `~ … allowlisted` lines.
- **For the allowlist owner:** the reason for those entries in `docs/diagrams/citation-allowlist.json` says "written by pdflatex", but the panels describe a XeTeX run. This worker does not own that file.

**VFG-06 — arXiv packaging depends on an uncommitted input.**
- **Evidence:** `build-arxiv-package.sh` exits unless a pre-built `main.bbl` exists (`presentation/report/scripts/build-arxiv-package.sh:106-107`), and `main.bbl` is git-ignored (`.gitignore:22`).
- **Consequence:** a clean checkout cannot package the book until the whole XeLaTeX and biber sequence is run by hand, and that sequence is a comment, not a script (`presentation/report/main.tex:6`).
- **Status:** recorded as Debt in VF-07.9.

**VFG-07 — The releases README.**
- **VF-07.5 was wrong at its stamp:**
  - It said the README still carries a v1 candidate row at `releases/README.md:28`. That line is the `## Required Fields` heading.
  - The README already reads "None committed" (`releases/README.md:26`).
  - It dated the deletion 2026-10-15. The deleting commit `5a3107d` is dated 2026-10-14.
- **Still true:** the Required Fields table (`releases/README.md:30-40`) omits `provenance` and `fixtures`, both of which the v2 schema requires (`docs/releases/ecosystem-release.schema.json:7`, `:18`).
- **Status:** VF-07.5, VF-07.6 and the VF-07 prose Drift are corrected. The prose had cited the deleted candidate file.

**VFG-08 — ADR-2020 contradicts itself.**
- **Evidence:** the frontmatter reads `implementation_status: complete` (`docs/adr/ADR-2020-pele-renders-every-estate-diagram.md:6`), but the Consequences bullet still says the follow-ons are "tracked by `implementation_status: partial`" (`:54-56`).
- **VF-06.1 NOTE4** had said "partial" and is corrected. The ADR's own boundary — local evidence only, no hosted workflow, no running-image inspection (`:87`) — is carried into the note.
- **To close:** the ADR owner amends the stale bullet.

**VFG-09 — Smaller corrections.**
- **VF-05.9:** `copyright-guard.yml` is not "the only gate with no path filter". `deploy.yml` has none either, but runs only on `main` (`.github/workflows/deploy.yml:34-37`).
- **VF-07.2:** the WasmVOWL divergence said the repository map names the visualiser `vowl-wasm`. The map now lists both repositories and records WasmVOWL as a retained standalone demo (`docs/architecture/repository-map.md:16`, `:32`).
- **Anchors repointed** because the cited line did not hold the claim, all now resolved from the symbol or sentence:
  - VF-06.1 NOTE2 (`docs/diagrams/README.md:78-82`)
  - VF-06.2 (`diagram-render.yml:34-39`)
  - VF-06.3 (`check-cdp-sidecar.mjs:14-16`)
  - VF-06.12 `writeIndexes` (`:724`) and the hosted-runner invariant (`diagram-index.yml:36-49`)
  - VF-07.10 (`render-batch.sh:20`)
  - VF-07.11 (`BASELINE-visionflow.md:73-81`)
  - VF-08.1 map invariant (`scripts/diagram-index-gen.cjs:206-216`)
  - VF-08.2 Multikey (`agent-identity.js:22-25`)

**VFG-10 — Jev advisories, checked by hand.**
- **True positives**, both fixed:
  - `docs/releases/README.md:28` (VF-07.5)
  - `render-batch.sh:19` (VF-07.10)
- **False positives:**
  - `docs/architecture/compatibility-matrix.md:20` (VF-08.11): the row holds all four claims.
  - `docs/protocol/mesh-smoke-test.md:84` (VF-05.1): it is the "Preflight Results" heading the label names.

## Panel dispositions

| Panel | Disposition | Evidence |
| --- | --- | --- |
| VF-05.1 | corrected | Added job `gate-suites` (`harness-fitness-gates.yml:178`, `:206`, `run-all.sh:47-50`); other routes hold (`drift-counter.yml:28`, `fixture-drift.yml:149`, `copyright-guard.yml:36`, `deploy.yml:128`, `:138`). |
| VF-05.2 | accurate | Pin check before counts (`drift-counter.mjs:104`, `:111`, `:253-254`); skill and tool truth queries (`:129`, `:141`, `:165`); exit (`:298`). |
| VF-05.3 | accurate | Scan and sites modes (`allowlist.json:22-28`, `:41-51`); findings (`drift-counter.mjs:210`, `:213`, `:228`, `:231`, `:234`). |
| VF-05.4 | accurate | Strict flips unavailability (`drift-counter.mjs:266-267`); planned roster skipped (`:258`); pin states (`:95-121`). |
| VF-05.5 | accurate-with-limits | Enforcement and waiver logic hold (`fixture-drift.yml:131-149`, `:163`, `:172`). That the hosted checkout fails is the workflow's own expectation, not observed. |
| VF-05.6 | accurate | Stale default probe (`check-fixture-drift.sh:82`). VisionClaw `86cdd76` has no `docs/specs/fixtures`. MISS does not set drift (`:215`, `:258`); exits (`:335`, `:337`). |
| VF-05.7 | corrected | Four jobs, not three (`harness-fitness-gates.yml:178`, `:188-192`, `:206`); NOTE4 now twelve suites (`run-all.sh:18-31`). |
| VF-05.8 | accurate | De-duplication, dangling edges, source states and verdict (`harness-audit.sh:257`, `:264`, `:276`, `:305`, `:396-398`, `:430-437`, `:464`). |
| VF-05.9 | corrected | Trigger claim qualified (`copyright-guard.yml:9`; `deploy.yml:34-37`); guard logic holds (`copyright-guard.yml:25`, `:34-48`). |
| VF-05.11 | corrected | Twelve suites (`run-all.sh:18-31`, `:28-30`), runner and verdict lines re-resolved (`:37`, `:44`), CI wiring (`harness-fitness-gates.yml:206`). |
| VF-05.12 | accurate | Read-only probe, totals, no exit contract (`mesh-smoke-preflight.sh:23`, `:321`, `:334`, `:337`). |
| VF-05.13 | corrected | Bare references bound to the suite (`augmentation-citations.test.cjs:35-72`); CI path via `gate-suites`; gate logic holds (`check-augmentation-citations.cjs:18-23`, `:31-80`, `:116`). |
| VF-05.14 | corrected | Ratchet verdict rewritten (`adr-ratchet.sh:44`, `:108-111`, `:118-121`); index workflow steps hold (`diagram-index.yml:56`, `:59`, `:65`, `:71`, `:77`; `adr-index.yml:44`, `:49`, `:55`). |
| VF-06.1 | corrected | NOTE4 now matches ADR-2020's frontmatter and boundary (`ADR-2020…md:6-7`, `:87`) and flags its stale bullet (`:54-56`); NOTE2 repointed; generator anchors re-resolved (+2 lines). |
| VF-06.2 | corrected | Anchor for the no-browser invariant (`diagram-render.yml:34-39`); stage lines hold (`:16-25`, `:42-43`, `:50-56`). |
| VF-06.3 | corrected | CDP anchor (`check-cdp-sidecar.mjs:14-16`); Pele pin and verification hold (`pele-static.cjs:21`, `:26-32`; `VERSION:1-6`). |
| VF-06.4 | accurate | Flatten, background and fill (`pele-static.cjs:40-50`, `:56-57`; `render.mjs:24-29`, `:40-60`). |
| VF-06.5 | accurate | Transform and label extraction (`preprocess.mjs:16-57`); ADR-2004's semantic limit (`ADR-2004…md:110-111`). |
| VF-06.6 | accurate | Visibility rules and exits (`check-diagram-text.js:46-69`, `:100`, `:160-208`). |
| VF-06.7 | accurate | Word-set diff (`check-diagram-text.js:118-154`, `:216-220`). |
| VF-06.8 | accurate | `walk`, `parseTopic`, `parseFrontmatter` and main (`diagram-index-gen.cjs:115`, `:144`, `:193`, `:838`). |
| VF-06.9 | accurate | Allowlist and stale-entry handling, strict mode and exit (`diagram-index-gen.cjs:109`, `:410`, `:495`, `:517`, `:618-626`, `:880`, `:909-917`). |
| VF-06.10 | accurate | Resolution order and fallback reasons (`diagram-index-gen.cjs:175-185`, `:287`, `:346-366`, `:528-534`, `:552`, `:608-610`, `:883`). |
| VF-06.11 | accurate-with-limits | Width ceiling (`diagram-index-gen.cjs:697`). The render counts are ADR-recorded runs (`ADR-2020…md:62-65`), not reproduced here. |
| VF-06.12 | corrected | `writeIndexes` anchor (`:724`) and the hosted-runner invariant (`diagram-index.yml:36-49`); the steps hold (`:55-77`). |
| VF-06.13 | accurate-with-limits | The scripts' contents hold (`validate-mermaid-diagrams.sh:7-8`, `:31`, `:60`, `:136`, `:154`; `generate-diagram-art.sh:3-6`, `:16`, `:83`). Nothing outside review records invokes them. "Silently returns no image" is runtime behaviour. |
| VF-06.14 | accurate | Byte parity, exits and method string (`check-diagram-render-sources.py:14-50`); `rendered/` is ignored (`docs/diagrams/.gitignore:2`); no workflow or npm wiring. |
| VF-06.15 | accurate-with-limits | Flag parsing, ordering and thresholds (`diagram-index-gen.cjs:104-106`, `:807-834`, `:885`; `jev.cjs:41-58`; `jev-questions.cjs:54-55`, `:83-96`). Environment projection by Agentbox is configuration, not observed. |
| VF-07.1 | corrected | `texput.log` node no longer dates against an uncommitted tarball; the pdf-reports copies are byte-identical (same blob ids); the README link holds (`./README.md:17`). The `texput.log:1` citation is allowlisted, not verified. |
| VF-07.2 | corrected | WasmVOWL divergence restated (`repository-map.md:16`, `:32`; `generate-release-manifest.sh:100`); sixteen-entry roster holds (`:91-108`). |
| VF-07.3 | accurate | Refusals and digest (`generate-release-manifest.sh:166-214`, `:223-244`); consumers empty (`:205`); comparator (`check-fixture-drift.sh:257`). |
| VF-07.4 | accurate | Schema constraints (`ecosystem-release.schema.json:7-32`, `:48`, `:67-80`, `:103`). |
| VF-07.5 | corrected | Rewritten to the README at this revision (`releases/README.md:8`, `:11`, `:13-26`, `:30-40`) and the deletion date (`5a3107d`, 2026-10-14). |
| VF-07.6 | corrected | DIVERGENCE note no longer cites the deleted candidate (`releases/README.md:26`); suite steps hold (`release-manifest.test.sh:27-166`, twelve first-party at `:66-67`). |
| VF-07.7 | accurate-with-limits | Export steps hold (`render-wardley.mjs:5-88`; `main.tex:154`; `14-implementation.tex:209`). The DPI and librsvg rendering outcome is not observed. |
| VF-07.8 | corrected | Committed-output claims fixed (`.gitignore:17-38`). The `texput.log` nodes rest on four allowlisted citations, readable in one operator working tree only: a limit. |
| VF-07.9 | corrected | ARXIV.md is now reported accurate (`ARXIV.md:13`, `:18`; `.gitignore:33`); the `main.bbl` Debt is added (`build-arxiv-package.sh:106-107`). |
| VF-07.10 | corrected | `BATCH DONE` anchor (`render-batch.sh:20`); run records are ignored (`pitch-deck-2026-07/.gitignore:1-3`); retry logic holds (`:10-16`). |
| VF-07.11 | corrected | Two-surface anchor (`BASELINE-visionflow.md:73-81`); drift-counter paths and the allowlist note hold. |
| VF-08.1 | corrected | Map-invariant anchor (`diagram-index-gen.cjs:206-216`); fallback lines re-resolved (`:550`, `:883`); the Agentbox dispatch holds at `c57d9c0` (`ontology-apply.js:153-160`, `flake.nix:942`, `:960`, `lib/ontomark.nix:44`). |
| VF-08.2 | corrected | Multikey anchor (`agent-identity.js:22-25`); NIP-98 cache and single-process invariant hold at `86cdd76` (`nip98.rs:164-166`, `:210-213`). |
| VF-08.4 | accurate-with-limits | Planner and gate (`mesh.rs:59`, `:185` at `c39f41f`); broker modules (`broker/mod.rs:31-33`). End-to-end application is explicitly open. |
| VF-08.7 | accurate-with-limits | No citations. The consumer-contract caution is consistent with `pod-tier-matrix.md`, which still labels the Agentbox tier "Embedded". It asserts nothing beyond source. |
| VF-08.8 | accurate | Matrix and reconciliation separate current from historical sections (`status-reconciliation.md` § "Current reconciliation — 2026-09-07" vs "Historical notes"; `compatibility-matrix.md` § "Current compatibility" vs "Historical …"). |
| VF-08.10 | accurate | The evidence rule (`ADR-2007…md:26`). The record is now `complete` and `live` (`:6-7`), noted in prose. |
| VF-08.11 | accurate | Supporting-repository row (`compatibility-matrix.md:20`); isolation rule (`../project/AGENTS.md:15`); map rows and branch edge (`repository-map.md:37-38`, `:83`). |

## Verification

**Evidence class:** source tracing at pinned revisions only. I ran no product test, workflow, render pipeline or script other than the diagram generator's gate. Nothing here is runtime, CI or deployment evidence.

**Pinned revisions:**
- VisionFlow `5f8b1da92ce5d9e6b32a0bd90ec3630c66619083`: the worktree HEAD, confirmed against `snapshot-wave2-start.json`.
- VisionClaw `86cdd76355823a51bb64d503c005fe277726e724`.
- Agentbox `c57d9c0508f4044915f9aecea4b9aa5629cc6a35`.
- nostr-rust-forum `c39f41f104b114d6a3fa29013daaf6fa58b45080`.

All four topics' `verified_commit` values were restamped to these revisions; topic 08 restamps all four map keys.

**How citations were re-resolved:**
- Every mermaid citation was dumped with its cited lines from the gate's own resolver at the new revision, then read against its claim.
- Moved lines were re-derived by matching the exact line content at the new revision. The only moves were in the generator (+1/+2), ADR-2020 (+2), ADR-2007 (+2), `run-all.sh` and `adr-ratchet.sh`. One range end that matched a bare brace was fixed by hand.
- Prose `path:line` citations, which the gate does not scan, were re-resolved and read by hand.

**Generator commands**, run from the worktree root after the last edit:

```sh
node scripts/diagram-index-gen.cjs docs/diagrams --only 05-governance-gates.md --check --cite-check --strict-citations --render                        # exit 0; 249 verified, 0 unverified, 0 unresolvable, 0 allowlisted; 13/13 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 06-diagram-tooling.md --check --cite-check --strict-citations --render                         # exit 0; 186 verified, 0 allowlisted; 15/15 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 07-pitch-presentation-reports.md --check --cite-check --strict-citations --render              # exit 0; 131 verified, 5 allowlisted (texput.log); 11/11 rendered
node scripts/diagram-index-gen.cjs docs/diagrams --only 08-compatibility-and-dependency-direction.md --check --cite-check --strict-citations --render  # exit 0; 22 verified, 0 allowlisted; 7/7 rendered
```

The pre-edit baseline run of topic 07 printed `source path does not exist: texput.log`. Its exit status was not captured correctly, because the pipe through `tail` hid it.

**Not verified:**
- **Hosted CI:** whether any workflow has run green or red at this revision.
- **`texput.log`:** its content. It exists in no commit. Its five citations are allowlisted, and the description of the aborted XeTeX run is not source-traced.
- **Recorded test results:** the ADR-recorded render and pin-check outcomes (ADR-2020, ADR-2007).
- **Configuration effects:** Agentbox's environment projection of `VISIONFLOW_JEV`.

**Whole-tree run:** these four restamps change the declared-revision set. The committed `VERIFICATION.md` anchor will be stale until the lead's whole-tree `--check --strict-citations` run regenerates it. A `--only` run skips the allowlist staleness check, so the five `texput.log` allowlist entries should be confirmed as still used in that whole-tree run.

**Files changed:** I edited only the four assigned topic files, this report and the claims proposal.
