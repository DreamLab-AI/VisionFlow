---
id: ADR-2006
title: VisionFlow is canon-only — it owns the cross-repo view and evidence-bounded maturity, never substrate implementation truth
date: 2026-08-31
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: cf535f8
owner: jjohare
review_trigger: any move to host substrate implementation in this repo, or to publish a maturity claim above its evidence tier
repo: visionflow
domain: BASELINE-visionflow.md
lineage: distils legacy docs/archive/adr/ADR-002-ecosystem-alignment-governance.md (canon role, maturity tiers) and ADR-004-gap-close-sprint-governance.md (evidenced-tier register); makes Baseline Invariant 4 a standing constraint.
---

# ADR-2006 — VisionFlow is canon-only — it owns the cross-repo view and evidence-bounded maturity, never substrate implementation truth

## Context

VisionFlow could have been built as an application, or as a repo that vendors and
restates substrate truth, or that asserts its own maturity claims. Legacy ADR-002 chose
instead to make it the ecosystem *canon* over the DreamLab repos, with a shared maturity
vocabulary; ADR-004 added an evidenced-tier register. The constraining question a reader
must be able to answer: is a given claim VisionFlow's to make, and is it backed?

## Decision

This repo holds **no substrate implementation** (no server, DB, or Rust code) and does
not assert implementation status about sibling repos — repo-local docs stay authoritative
for their own code. VisionFlow owns exactly the cross-repo surface: the compatibility
matrix, the release-evidence manifest, and the shared maturity vocabulary. Two rules
bind canon prose: a maturity/tier claim above the tier its evidence supports is a
governance defect (not a footnote); and every count VisionFlow asserts has one queryable
source (enforced by ADR-2005's drift gate). Maturity labels in the matrix are read from
the substrates' own template fields, not hand-typed here.

## Consequences

- Forecloses VisionFlow-as-application and VisionFlow-as-mirror: no substrate code lands
  here, and the canon may not overwrite a substrate's own status. Cross-repo claims that
  belong to a substrate must cite it, not restate it.
- Forecloses aspirational maturity: a claim is capped at its evidence tier, so promoting
  a tier requires new evidence (a closure SHA / canary), not new prose. This is a real
  ceiling on marketing language in the canon.
- Concentrates cross-repo authority in a few artefacts (matrix, release schema, drift
  gate); their integrity is the whole governance guarantee, so their CI gates are
  load-bearing, not optional.
- Cross-repo findings (bridge write-path, degraded discover) are tracked here but fixed
  upstream — this repo records the divergence and cannot close it.

## Verification

At `cf535f8`: `find` confirms no `Cargo.toml`/`*.rs`/`*.wasm` (no substrate code);
`docs/architecture/compatibility-matrix.md` carries the cross-repo view, an "Evidenced
tier" register keyed to closure SHAs/canary state, and maturity labels sourced from the
substrates' own `maturity` template fields (`compatibility-matrix.md:48,79-84`);
`scripts/generate-release-manifest.sh` + `docs/releases/ecosystem-release.schema.json`
provide the release-evidence surface; the count-integrity half is enforced by
`scripts/drift-counter/` (ADR-2005).

## Closeout extension — 2026-09-04

Retain accepted/complete/live for the canon ownership boundary. The complete label does not certify estate-wide claim integrity. The reviewed release generator covers six repositories while the assessment inventory contains fourteen identities, and fixture CI can succeed without comparing a canonical corpus. These are narrower implemented surfaces than complete-system reproducibility.

**Closeout (CP-01/08/09):** Reconcile the authoritative release roster and consumed artefacts, bind maturity claims to dated producer/consumer evidence, and require a real compared revision set for parity. Repository-local decision authority remains intact; cross-repository findings are evidence-qualified assessments requiring owner disposition.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.

## Acceptance progress — 2026-09-05

Retain accepted/complete/live. The ownership boundary is unchanged — the canon
still owns the cross-repo *view*, not substrate implementation. Two of the three
CP-01/08/09 obligations are discharged: the release roster is reconciled with
the inventory, and parity now requires a real compared revision set. Binding
maturity claims to dated producer/consumer evidence remains open.

**Roster reconciled: six repositories → fourteen.**
`scripts/generate-release-manifest.sh` covered VisionFlow, VisionClaw, agentbox,
solid-pod-rs, nostr-rust-forum and dreamlab-ai-website while
`docs/estate-review/evidence/adr-inventory.json` records fourteen. A
"coordinated release manifest" that silently omits eight repositories
coordinates nothing about them: loom, knowledgeGraph, WasmVOWL, visionGraph,
dream-engine, logseq, ruvector and RuView could each move under a release with
no record. All fourteen are now listed, and each carries an explicit
`provenance` flag so a reader can tell what the estate *authors* from what it
merely *carries* or *consumes*:

- **first-party** (10) — VisionFlow, VisionClaw, agentbox, solid-pod-rs,
  nostr-rust-forum, dreamlab-ai-website, loom, knowledgeGraph, WasmVOWL,
  visionGraph;
- **imported** (2) — dream-engine (forked from `ruvnet/dream-machine`) and
  logseq (forked from `logseq/logseq`), each naming its upstream;
- **upstream** (2) — ruvector and RuView, consumed from `ruvnet` as-is.

Release qualification applies to the first two classes; upstream repositories
are pinned, not qualified. Each entry also carries `present` and a `role`, so an
absent checkout is recorded with an all-zero head rather than omitted.

**Parity requires a real compared revision set.** The manifest gains a
`fixtures` block, and the generator **refuses** (exit 3) to emit a `candidate`
or `released` manifest without an explicit canonical revision set
(`--fixtures-canonical REPO@REV:DIR`). Previously the manifest asserted "run
npm run verify plus substrate-specific CI" in prose and carried no fixture
evidence at all, so a candidate could be cut with no corpus ever compared. When
supplied, `HEAD` is resolved to a concrete commit (via
`rev-parse --verify …^{commit}`, since plain `rev-parse` echoes any well-formed
40-hex string back unverified), the corpus is digested as a SHA-256 over its
sorted per-file hashes, and a `verdict` of match/drift/not-run is recorded. A
local draft may omit it and is stamped `fixtures.status: "not-compared"` with a
reason — the gap visible in the artefact rather than implied by its absence.

The companion hole is closed too: `.github/workflows/fixture-drift.yml` was a
green no-op, because VisionFlow carries no canonical corpus so the locate step
always set `has_canonical=false` and exited 0. It now **fails** without a
canonical corpus, with an explicit `waive_reason` dispatch input as the only
escape.

`docs/releases/ecosystem-release.schema.json` moves to `manifest_version: 2`:
`minItems: 14` on the roster, `provenance` required and enumerated, and a
`oneOf` on `fixtures` so a "compared" block cannot exist without a resolved
40-hex revision, a 64-hex corpus digest and a non-zero fixture count.

**Tests and results.** `tests/gates/release-manifest.test.sh`, 29 assertions,
all passing, wired into `harness-fitness-gates.yml`. The roster size is asserted
against `adr-inventory.json` itself rather than a hard-coded 14, so the two
cannot drift apart. A candidate without a canonical revision set exits 3
([4]); so do `--require-fixtures` on a draft ([4b]), a released manifest ([4c]),
an unknown revision, an unknown repository, a missing corpus directory and a
malformed spec ([5]). With a valid set ([6]) the generator resolved
VisionClaw@HEAD to `b00c28a0d766…`, digested 13 fixtures, and returned
verdict `drift` — an honest verdict from a real comparison, which is precisely
what the previous manifest could not produce.

Receipts: [local-draft manifest](../estate-closeout/2026-09-05/release-manifest.local-draft.json),
[gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json).

**Remaining.** Binding maturity claims to dated producer/consumer evidence is
not done: `compatibility-matrix.md` still sources maturity from the substrates'
own template fields without a dated receipt per claim. The recorded fixture
verdict is `drift` — the consumer copies genuinely differ from the canonical
corpus, and reconciling them is substrate work outside this repository's
authority. No hosted CI run; `fixture-drift.yml` will stay red on GitHub Actions
until sibling-repo credentials exist or a run is explicitly waived, which is
deliberate.

Governed paths changed: `scripts/generate-release-manifest.sh`,
`docs/releases/ecosystem-release.schema.json`,
`.github/workflows/fixture-drift.yml`,
`.github/workflows/harness-fitness-gates.yml`,
`tests/gates/release-manifest.test.sh` (new).
