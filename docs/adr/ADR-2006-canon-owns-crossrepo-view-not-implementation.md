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
