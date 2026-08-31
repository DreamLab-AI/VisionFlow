---
id: ADR-2001
title: Consolidate the ADR corpus into a living baseline plus a thin ledger
date: 2026-08-31
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: c205575
owner: DreamLab AI maintainers
review_trigger: any new website-build/deploy decision, or a new canon governance decision
repo: visionflow
---

# ADR-2001 — Consolidate the ADR corpus into a living baseline plus a thin ledger

## Context

Seven ADRs (ADR-001..007) sat loose in `docs/` and had drifted from the code. The
sharpest case: ADR-001 decides a Rust/WASM website built with `wasm-pack` and a
Cargo workspace of `mesh-hero`/`particle-field` crates, but at this commit the site
is pure static HTML/CSS/JS — `website/build.sh:6-7` says "No compile step, no
bundler, no WASM" and no `Cargo.toml`/`*.rs`/`*.wasm` exists in the repo. Legacy ADR
prose was being read as current build instructions. A further hazard: the
`docs/engineering/` sequence reuses numbers ADR-004/005, colliding with the canon.

## Decision

The living decision surface is **`docs/BASELINE-visionflow.md`** (what this repo is
and runs today, present tense, `file:line`-cited) plus a **thin ledger** at
`docs/adr/` for new decisions. The seven legacy ADRs are **archived**
(`git mv` to `docs/archive/adr/`) under a frozen, do-not-edit tombstone; they remain
citable as evidence and history, never as authority. New decisions are recorded from
`docs/adr/TEMPLATE.md` with three-axis status, and `docs/adr/README.md` is a
generated index (`scripts/adr-index-gen.cjs`, `repo` enum = `visionflow`). The
`docs/engineering/` ADR sequence is out of scope and left in place.

## Consequences

- One present-tense, cited baseline replaces seven drifted narratives; the dead
  Rust/WASM website premise is corrected at source.
- New decisions carry honest decision/implementation/activation status and a
  verified commit; the index generator fails CI on malformed frontmatter.
- `docs/README.md` ADR links now point into the archive and the new baseline.
- The engineering/canon number collision is not fully resolved: the canon side moves
  to a namespaced 2xxx ledger, but `docs/engineering/ADR-004/005` still exist under
  their own numbers (recorded as a known divergence in the baseline).

## Verification

Established at `c205575`: `git mv` of the seven `docs/ADR-00{1..7}-*.md` into
`docs/archive/adr/` (confirmed: no `docs/ADR-*.md` remain); `docs/BASELINE-visionflow.md`
written with citations verified against `website/build.sh`, `website/static/index.html`,
`.github/workflows/deploy.yml`, `scripts/diagram-render/` and `scripts/drift-counter/`;
index regenerated with `node scripts/adr-index-gen.cjs docs/adr` exiting 0.
