---
id: ADR-2020
title: Render every estate diagram with a pinned Pele; Mermaid stays the source syntax
date: 2026-10-29
decision_status: accepted
implementation_status: partial
activation_status: live
supersedes: [ADR-2004]
superseded_by: []
verified_commit: 1b0dd42
owner: jjohare
review_trigger: a Pele release we adopt, a diagram the Mermaid 11 renderer accepts and Pele rejects, or any surface (site, CI gate, agent tool) added that renders Mermaid
repo: visionflow
---

# ADR-2020 — Render every estate diagram with a pinned Pele; Mermaid stays the source syntax

## Context
ADR-2004 rendered the RES-b report diagrams with a vendored `mermaid.min.js` in Chrome over raw
CDP, and the diagrams-as-code corpus rendered through `mmdc` (Puppeteer and Chromium; in agentbox,
a browser sidecar, ADR-2152 there). Hosted CI never ran the corpus render gate because it had no
browser, and five corpus diagrams that Mermaid 11 itself rejects had shipped. Pele (obsidianmd/pele,
MIT, no dependencies) reads Mermaid 12.1 syntax and renders synchronously in Node. Measured on
2026-10-28: it rendered 1,490 of the corpus' 1,495 blocks in 1.6 s, and every one of the 34 estate
blocks it rejected (5 corpus, 29 elsewhere) is also rejected by Mermaid 11.

## Decision
- **Mermaid remains the authoring syntax; Pele is the only renderer** on every surface the estate
  controls: CI gates, the corpus generator, agent tools and the published sites.
- **One exact pinned version estate-wide** (1.12.0 at adoption), with an integrity check on every
  path: `scripts/vendor/pele/VERSION` carries the npm integrity and the sha256 that
  `scripts/lib/pele-static.cjs` verifies before import; a site lockfile carries the integrity, a
  CDN tag carries SRI. npm `pele` 1.0.0 is an unrelated 2020 package and is never accepted.
- **Files that leave the browser are static.** An SVG written to disk is flattened: every Pele CSS
  variable is replaced by the default Pele declares for it (`pele-static.cjs flatten`), and the
  RES-b baseline also bakes a white background and an explicit text fill.
- **Gates render with `responsive: false, autoDirection: false`**, so a width belongs to the diagram,
  not to a viewport. The 4500 px legibility ceiling stays, measured on Pele's layout.
- **The committed RES-b baseline stays authoritative** (unchanged from ADR-2004); re-rendering only
  detects drift.

## Consequences
- The corpus render gate now runs on the hosted runner (`diagram-index.yml`); a diagram that does not
  parse cannot reach `main`.
- No browser, Puppeteer or CDP client remains in VisionFlow's diagram path.
- Layout and style are Pele's own, so diagrams look different from GitHub's native Mermaid rendering
  of the same Markdown, and widths are not comparable with earlier `mmdc` numbers.
- **Limit:** a Pele-only gate cannot catch a Mermaid 12.1 construct that GitHub's older Mermaid
  rejects. GitHub's rendering, the Obsidian app and its plugins are outside this decision.
- Pele releases several times a week; adopting a new version is a deliberate bump that reruns the
  corpus and RES-b gates, never a range.
- Follow-on (tracked by `implementation_status: partial`): agentbox (the `mmdc` tool, browser sidecar,
  skills), dreamlab-ai-website lesson pages, the Quartz research-notes site, sealmap's CI and the
  dream-engine and sealmap corpus generators adopt the same pin.

## Verification
At `1b0dd42`: `node scripts/diagram-render/render.mjs` rendered all 10 report diagrams;
`node scripts/check-diagram-text.js presentation/report/diagrams/rendered` passed (visible text and
key labels), and `--diff` against the previous Mermaid baseline reported identical visible words for
all 10. `node scripts/diagram-index-gen.cjs docs/diagrams --check --no-source-paths --render`
rendered 1,473 of 1,494 corpus diagrams in about 2 s. The 21 failures are 4 blocks with a bare `;` in
sequence text, which Mermaid 11 rejects too, and 17 renders wider than 4500 px. Both sets are fixed
in the corpus refresh on the same branch; until then `diagram-index.yml` is red by design.
