---
id: ADR-2004
title: Gate diagrams on a committed light-theme baseline rendered by a vendored Mermaid; re-render only to detect drift
date: 2026-08-31
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: cf535f8
owner: jjohare
review_trigger: any change to the diagram render engine, the text-fill probe, or the rendered/ baseline contract
repo: visionflow
domain: BASELINE-visionflow.md
lineage: implements PRD-gap-close-canon RES-b; diverges from legacy docs/archive/adr/ADR-005-gap-close-canon-decisions.md Decision 3, which specified scripts/render-diagrams.sh wrapping a pinned mmdc/puppeteer/chromium.
---

# ADR-2004 — Gate diagrams on a committed light-theme baseline rendered by a vendored Mermaid; re-render only to detect drift

## Context

Dark-theme Mermaid diagrams were exported onto the report's white page with near-white
text on transparent backgrounds — every label invisible. ADR-005 D3 prescribed a
`render-diagrams.sh` wrapping a pinned `mmdc`/puppeteer/chromium at CI time. Rendering
live at CI time makes the gate's authority depend on a browser toolchain reproducing
identical output, and pins the engine to a CDN/npm dependency.

## Decision

The authoritative guard is the **committed** `rendered/` SVG baseline: a browserless,
deterministic probe (`check-diagram-text.js`) asserts visible text fill and key labels
on the checked-in files, and runs first. Re-rendering the `.mmd` sources is a
**secondary drift check**, not the source of truth — it re-renders with a **vendored**
`scripts/diagram-render/vendor/mermaid.min.js` (11.16.0) driven over raw CDP, forces
real SVG `<text>` (`htmlLabels:false`) on the light `default` theme, then diffs visible
words against the baseline. The named ADR-005 artefact `render-diagrams.sh` does not
exist and is not required.

## Consequences

- Forecloses "CI render is truth": a green build never depends on a browser matching
  byte-for-byte (font metrics drift is informational). An edited `.mmd` with a stale
  baseline fails the word-diff instead.
- Forecloses a CDN/live-`mmdc` render dependency — the engine is vendored (a 3.5 MB
  blob in-tree, updated deliberately). The CI job still `npm install`s `mermaid-cli`
  + `puppeteer`, but only to provision a Chrome binary for `render.mjs` to attach to.
- The baseline must be regenerated and committed with any diagram edit; a forgotten
  regen is exactly what the drift check catches, but it adds a commit step authors must
  learn.
- Engine upgrades are a manual vendored-blob bump plus a `MERMAID_CLI_VERSION` bump
  kept in lockstep — two edits that can silently diverge if uncoordinated.

## Verification

At `cf535f8`: `diagram-render.yml:45-47` runs the baseline probe first as the
authoritative guard; `:51-55` installs pinned `mermaid-cli@11.16.0` + `puppeteer` only
to resolve a Chrome binary; `:74-83` re-renders via `render.mjs` and word-diffs against
a copied baseline. `scripts/diagram-render/vendor/mermaid.min.js` contains
`"11.16.0"`; `render.mjs` header confirms the vendored no-CDN bundle and `htmlLabels`
handling. No `scripts/render-diagrams.sh` exists.

## Closeout extension — 2026-09-04

Retain accepted/complete/live for the committed-baseline guard. The current browserless probe passes all ten checked-in diagrams for visible text and required labels. Workflow source runs that guard before rendering and word comparison. This pass did not render with Chrome or run hosted CI; the local result does not certify arbitrary diagram semantics or source-to-baseline parity.

**Closeout (CP-08/09):** For changed diagrams, retain the exact renderer/toolchain identity, source-to-baseline word-diff receipt and output inspection. Ensure the intended release entry point requires the gate. A passing checked-in baseline alone cannot close source drift.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.

## Acceptance progress — 2026-09-05

Retain accepted/complete/live. The decision is unchanged. The gap the previous
annex recorded — "This pass did not render with Chrome or run hosted CI" — is
now closed locally: the full three-stage gate has been executed against a real
Chrome, so source-to-baseline parity is measured rather than assumed.

**Implemented and executed.** `scripts/diagram-render/render.mjs` already
supported an explicit DevTools endpoint and this environment's sidecar; that
path was exercised. All three stages ran in sequence:

1. **Baseline guard** (browser-free): all 10 committed diagrams pass for visible
   text and required labels.
2. **Re-render from source** via Chrome 151 on the browsercontainer sidecar
   (`ws://…:9223`), using the vendored `mermaid.min.js` 11.16.0 light-theme
   engine — 10/10 diagrams rendered, every text node visible (e.g.
   `07-change-architecture`: 140 nodes, 140 visible).
3. **Drift comparison**: `check-diagram-text.js --diff` reports *all rendered
   diagrams match the committed baseline's visible words*, and `git diff --stat`
   over `rendered/` reports **no byte-level drift at all** — the committed
   baseline is reproducible from source in this environment, not merely
   word-equivalent.

The committed baseline was copied aside before rendering and restored
afterwards, so the working tree is unchanged (`git status` over `rendered/` is
empty). This is the same sequence `diagram-render.yml` runs, executed by hand
because that workflow has not run hosted.

**Release entry point.** The closeout asks that the intended release entry point
require the gate. The browser-free baseline guard is now a **blocking** step in
`.github/workflows/deploy.yml`, so no site publication can proceed past a
diagram with invisible text. The re-render half stays in `diagram-render.yml`,
because it needs a Chrome the deploy runner does not otherwise provision.

Receipts: [full render log, all three stages](../estate-closeout/2026-09-05/logs/diagram-render-chrome.log),
[gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json) (`gates.diagram_render_chrome`).

**Remaining.** Still no hosted CI run: `diagram-render.yml` provisions its own
pinned `@mermaid-js/mermaid-cli` and puppeteer chromium on the runner, which is
a different toolchain instance from the sidecar Chrome used here, so
cross-environment byte parity remains untested. The gate continues to certify
text visibility and label presence only — it does not, and is not intended to,
certify that a diagram is semantically correct.

Governed paths changed: `.github/workflows/deploy.yml` (adds the baseline guard
as a blocking publication gate). No change to `scripts/diagram-render/`,
`scripts/check-diagram-text.js` or the committed baseline.

## Estate audit — 2026-09-07

The topic-tree coverage index and report-image gate certify different things. `scripts/diagram-index-gen.cjs` validates structure and offers warning-only citation checks; it does not verify semantic claims or that `verified_commit` describes all working-tree bytes. The generator now calls these values declared source revisions and qualifies ADR lookup keys by repository. Unqualified estate references remain explicitly unresolved until their owners are identified. The regression test `node tests/gates/diagram-index.test.cjs` verifies that equal ADR numbers in different repositories remain distinct.

Keep render visibility, source-path existence, semantic source review, target execution and deployed acceptance as separate evidence. The [dated audit](../estate-review/2026-09-07-estate-audit.md) records source hashes, initial failures and final validation. Existing live activation labels are not a fresh deployment attestation.
