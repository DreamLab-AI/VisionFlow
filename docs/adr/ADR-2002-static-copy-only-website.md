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

## Closeout extension — 2026-09-04

Retain accepted/complete/live for the copy-only build decision. Current build.sh copies static assets, writes CNAME and stages optional image directories. Optional image-copy failures are suppressed; a completed copy step therefore does not certify all expected media. No website build or browser session ran in this pass.

**Closeout (CP-08/09):** Specify the required asset inventory and verify the generated output and browser behaviour, including WebGL fallback and reduced motion. Bind publication-quality evidence to the published revision. A copy-only build is an implemented mechanism, not proof that every public claim or interaction is correct.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.

## Acceptance progress — 2026-09-05

Retain accepted/complete/live. The decision is unchanged: the build is still
copy-only, still has no compiler, bundler, framework or WASM. What changed is
that a completed build now certifies its output, and the graceful-degradation
claim in *Consequences* is verified in a browser rather than asserted.

**Implemented.** `website/assets.manifest.json` declares the asset inventory:
13 `required` entries (the page, CNAME, the stylesheet, both JS modules, the
five referenced showcase images, both videos, the social card) and 4 explicit
`optional` groups (the repo diagram, generated, hero and screenshot
directories). `scripts/website-assets.mjs` stages and verifies against it, and
`website/build.sh` calls both. The old `cp -r ../assets/... 2>/dev/null || true`
is gone: a missing **required** asset now fails the build, a missing **optional**
group is recorded in the receipt as `source-absent` rather than swallowed, and
`static/.claude-flow/` is explicitly excluded from the artefact. `og:image`,
`og:image:alt` and the four `twitter:*` tags were added to `index.html`, which
is what makes `img/og-card.png` a genuine required asset rather than a
decorative one.

The build emits `website/build-receipt.json` — written outside `dist/` so it is
never part of the artefact it describes — recording file count, byte total, a
`tree_sha256` over the sorted per-file digests, a SHA-256 for every required
asset, the manifest hash enforced, the source revision and the published
revision.

**Tests and results.** `tests/gates/website-assets.test.sh`, 20 assertions, all
passing: a removed required asset and a zero-byte required asset each exit 1
and name the offending file; every optional group carries an explicit status;
the receipt carries counts, byte totals, the tree digest and a hash per
required asset. Build output: 50 files, 60,145,148 bytes, 13/13 required
present, `ASSET-INVENTORY-OK`.

**Browser receipts.** `scripts/website-browser-check.mjs` drives the built site
in Chrome 151 on the browsercontainer sidecar (ANGLE / NVIDIA RTX A6000,
Vulkan 1.4.341) over raw CDP, in three scenarios — baseline, forced
`prefers-reduced-motion: reduce`, and `getContext('webgl2')` forced to null
before any page script runs. 26/26 checks pass. The *Consequences* claims are
now evidenced: WebGL2 initialises and animates (rAF delta 20 over ~1.2 s); with
WebGL2 removed the page still renders all 14 sections and 38,377 characters of
text and throws nothing, and the canvas region differs byte-for-byte from the
baseline capture, proving the mesh layer really paints; under reduced motion
the rAF delta is 0 against a baseline of 20, i.e. the single settled frame the
module promises. Zero console errors, zero failed requests and zero broken
images in all three scenarios.

Receipts: [browser receipt and 9 screenshots](../estate-closeout/2026-09-05/website-browser-receipt.json),
[gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json).

**Remaining.** No hosted CI run and no GitHub Pages deployment: the receipt's
`revision.published` is null locally and is stamped only by the deploy workflow
(ADR-2003). The browser evidence covers one desktop viewport (1440×900) in one
Chrome build; no mobile-viewport, cross-browser or accessibility-audit pass was
run here.

Governed paths changed: `website/build.sh`, `website/assets.manifest.json`
(new), `website/static/index.html`, `scripts/website-assets.mjs` (new),
`scripts/website-browser-check.mjs` (new), `tests/gates/website-assets.test.sh`
(new), `package.json`.
