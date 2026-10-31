---
title: Release manifest and fixture evidence audit
status: in-progress
date: 2026-10-31
type: explanation
---

# Release manifest and fixture evidence audit

Source at VisionFlow `de1fd11`: `scripts/generate-release-manifest.sh`, `scripts/check-fixture-drift.sh`, the release schema and README, `tests/gates/release-manifest.test.sh`, and ADR-2006. The two scripts are unchanged from VF-07's declared revision `4d361b58`. [Synthetic probe inputs/results and source hashes](evidence/2026-10-31/release-boundary-probes.json) support the findings. No release was published, no fixture in a real repository was changed, and no product code was repaired in this pass.

## Findings by section

| Section | Observation | Evidence and consequence |
|---|---|---|
| Roster and identity | Sixteen hardcoded entries lag behind the current linked estate; `dream-engine` still resolves to `dream-machine`. | Generator lines 91–108. It omits the separate current dream-engine, vowl-wasm, sidestr-rs, Codescape and other currently rostered supporting packages. The release and nightly-health inventories are distinct hardcoded authorities. The manifest cannot pin omitted products. |
| Checkout state | Missing checkouts are recorded with zero HEAD, dirty true and present false; dirty/present are not qualification gates. | `repo_json`, lines 111–140. This is useful disclosure for a draft. A synthetic `released` manifest still emitted with fifteen of sixteen repositories missing, so the status is a requested label rather than a verified release state. |
| Canonical revision | The commit is resolved, but its fixture tree is never read. | Lines 179–185 validate the object, then lines 185–202 hash the current filesystem directory. A named commit and corpus digest can describe different bytes. |
| Fixture digest | Only top-level `*.json` files affect the digest/count. | Generator line 194 uses `find -maxdepth 1`; the comparator recursively discovers JSON. Adding a nested fixture did not change the manifest digest/count, although the comparator saw two files. The stated corpus identity therefore does not cover its comparison scope. |
| Consumer parity | Missing and extra files are counted but only checksum differences set failure. | Comparator lines 251–258. A consumer missing every canonical fixture returns success, which the generator converts to `match`. This contradicts the schema's description that every consumer copy equals the corpus. |
| Consumer provenance | `consumers` remains an empty array; comparison output is discarded. | Generator lines 205–219. The manifest cannot identify which consumer revisions or paths were compared, nor distinguish an empty/missing set from full coverage. |
| Workspace selection | The child comparator receives a canonical path but no consumer arguments or explicit workspace assignment. | Generator line 207; comparator defaults depend on the `WORKSPACE` environment. If `WORKSPACE` was not exported on entry, the generator's local `--workspace` setting does not automatically become the child's environment. This is a source-level configuration gap; the probes set both explicitly to avoid touching unrelated paths. |
| Error classification | Every non-zero comparator result becomes `drift`. | Generator lines 207–211 fold usage/missing-input failures into mismatch. The unavailable-executable path instead says `not-run`; both can still be emitted with `fixtures.status=compared`. Consumers must not read that status as proof a comparison ran successfully. |
| Schema | Minimum length is sixteen, with no per-repository identity uniqueness/completeness constraint or conditional rule binding top-level `candidate/released` to a successful fixture verdict. | `ecosystem-release.schema.json` permits verdicts match/drift/not-run and statuses independently. It validates shape, not estate compatibility or source-to-artefact identity. |
| Compatibility prose | Identity, mesh, pod and governance assertions are fixed strings. | Generator lines 260–265. No release-specific negotiation, selected features, deployment identity or protocol execution is measured by generation. |

The release README correctly says that an entry is not maturity/compatibility qualification. Its “current” sixteen-repository inventory and early Ontomark/plugin release observations are now historical, however. An explicit dated qualifier is needed before those are used as current installation guidance.

## Adversarial receipts

A temporary `ontomark` Git repository contained one committed `fixtures/case.json`. A temporary forum consumer directory contained only unrelated JSON. Both the `--workspace` option and `WORKSPACE` environment pointed to that temporary estate.

1. `--status released --fixtures-canonical ontomark@COMMIT:fixtures` exited 0 with `verdict=match`, `consumers=[]`, and fifteen missing repositories.
2. Changing the canonical JSON from version 1 to version 2 without committing emitted the **same canonical revision** and a **different corpus digest**, still labelled released/match. This directly disproves revision binding, rather than merely noting a missing check.
3. Adding `fixtures/nested/extra.json` left the recorded digest and count unchanged. Running the comparator directly saw two canonical files, zero matched and two missing, yet exited 0.

These results do not imply a deployed release is corrupt. They establish that the evidence generator cannot prove the stronger interpretation its labels and diagram previously suggested.

## ADR-2006 section assessment

| Section | Disposition |
|---|---|
| Context | The separate canon/repository ownership motivation remains coherent. This pass does not repeat the dated predecessor review or certify every current cross-repo claim. |
| Decision | The canon owns the release-evidence surface, so its source identity and coverage are material obligations. A valid canonical commit alone does not meet evidence-bounded maturity when the bytes come from the mutable checkout. |
| Consequences | The explicit ceiling on maturity claims remains appropriate. Schema/fixture gates are load-bearing but do not currently enforce all the properties implied by “parity”. Product fixes still belong to their owners; the findings here concern VisionFlow's own evidence tooling. |
| Verification at the original commit | Historical presence checks establish that artefacts existed, not that the present implementation is complete or that its fixed strings are measured. No historical deployment result is promoted by this audit. |
| Closeout extension — 2026-09-04 | Its open obligation to reconcile roster, consumed artefacts and dated producer/consumer evidence is still relevant. Today's expanded estate reopens roster completeness independently of the earlier six-to-fourteen repair. |
| Acceptance progress — 2026-09-05 | Refusal without a canonical spec and commit-object validation are implemented. The stronger “real compared revision set” interpretation is contradicted by the working-tree-byte probe. Preserve the historical passing tests and recorded drift, but qualify what they establish. |
| Ontomark roster continuation — 2026-10-11 | The two added producer entries and optional URLs exist. That dated extension does not include the later separate engine/supporting projects. Its own text says local checks are not coordinated release qualification, which remains the correct boundary. |

ADR-2007's newer evidence-class/acceptance-boundary gate is relevant but is not fully audited here. A syntactically present `source+test` label cannot repair a test whose scope misses source-to-artefact binding. The release claim needs tests of the actual boundary, not a stronger label on the same assertions.

## Validation and repair acceptance

The existing release-manifest suite completed with **56 assertions passed, zero failed** ([command output](evidence/2026-10-31/release-manifest-suite.txt)). Its valid-canonical test explicitly accepts match, drift or not-run and checks digest format, not equality with the named Git tree. The new probes demonstrate why that green result does not refute the findings.

Proposed owner: VisionFlow release-tooling maintainers. Acceptance should require the following properties before representing a manifest as qualified:

- Read the canonical file set from the resolved revision, or refuse dirty/untracked inputs and independently prove byte equality with that revision. Record exactly the same recursive file set used by comparison.
- Name each compared consumer and revision. Define required versus optional fixture subsets explicitly; a missing required file must fail parity, while an optional subset needs a distinct verdict.
- Pass the selected workspace/consumer paths explicitly, preserving reproducibility regardless of inherited environment.
- Separate descriptive draft generation from release qualification. A qualified status needs declared policy for missing/dirty repositories, comparison errors and non-match verdicts. The CLI must enforce that policy or label output as unqualified.
- Derive or reconcile release scope with the current estate inventory and bind protocol claims to selected consumers, features and artefacts. Keep historical roster snapshots dated.

For users, this reduces the chance of receiving a nominally coordinated release whose recorded commit cannot reproduce its fixtures. It also makes upgrade failures diagnosable from the manifest instead of requiring access to the original operator's workspace.

## Diagram change

VF-07.3 now shows that the digest covers top-level working-tree JSON, that commit existence does not bind those bytes, and that missing consumer fixtures can still produce `match`. The unsupported “generator will not manufacture parity” invariant was replaced with the actual admission rule. All eleven VF-07 diagrams passed structural/render validation. The topic's broader pitch/report panels and governing-file drift remain on the semantic queue; no whole-topic restamp was made.
