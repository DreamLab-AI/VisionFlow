---
id: ADR-2002
title: Ship the website as a copy-only static build — no compiler, bundler, or WASM
date: 2026-08-31
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: cf535f8
owner: jjohare
review_trigger: any proposal to add a build step, bundler, framework, or WASM crate to website/
repo: visionflow
domain: BASELINE-visionflow.md
lineage: distils legacy docs/archive/adr/ADR-001-website-technology.md Decisions 1 & 3 (Rust/WASM via wasm-pack; mesh-hero/particle-field Cargo workspace) — reversed, not amended.
---

# ADR-2002 — Ship the website as a copy-only static build — no compiler, bundler, or WASM

## Context

Legacy ADR-001 D1/D3 chose a Rust/WASM site built with `wasm-pack` over a Cargo
workspace (`mesh-hero`, `particle-field`) to "demonstrate Rust competence". No such
code was ever committed. The audience-signalling rationale is real but optional; the
toolchain it justified is a standing liability (a Rust build, WASM budgets, lazy-load
plumbing) for a zero-dynamic-data marketing page.

## Decision

`website/build.sh` is the whole build and stays a copy-only step: wipe `dist/`, copy
`static/*`, emit `CNAME`, stage repo images — nothing compiled. The hero/scroll
visuals are a hand-written WebGL2 ES module (`static/js/mesh-webgl.js`), not a WASM
crate. Reintroducing a compiler, bundler, framework, or WASM to `website/` requires a
superseding ADR and a Baseline update; it is not a silent change.

## Consequences

- Forecloses the "WASM proves competence" path and the whole `wasm-pack`/Cargo
  toolchain, plus per-module WASM size budgets and lazy-loading — deleted, not
  deferred. Reviewers may assume `dist/ == static/ + generated images`.
- The competence signal now rests on hand-written WebGL2, which must degrade
  gracefully (null without WebGL2; honour `prefers-reduced-motion`).
- Any interactivity beyond one ES module has no build affordance and pays the full
  cost of reintroducing one — an accepted ceiling for a marketing surface.
- The docs that still assert a Rust/WASM build (`README.md`, `docs/PRD-website.md`,
  `docs/site-verification.md`) are now provably wrong against source; correcting them
  is follow-on debt tracked in the Baseline.

## Verification

At `cf535f8`: `website/build.sh:6-7` reads "No compile step, no bundler, no WASM";
`build.sh:10-24` is `rm -rf dist` + `cp -r static/*` + CNAME + image staging, no
compiler invocation. `find` for `Cargo.toml`/`*.rs`/`*.wasm` (excluding
`node_modules`) returns nothing. `website/static/index.html` loads one
`type="module"` entry; `static/js/mesh-webgl.js` is the WebGL2 module.
