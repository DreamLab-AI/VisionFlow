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
