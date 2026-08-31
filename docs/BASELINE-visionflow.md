---
title: VisionFlow Baseline — What This Repo Is and Runs Today
doc_id: VF-BASELINE
version: 0.1.0
status: draft-for-ratification
verified_commit: c205575
sources:
  - website/build.sh
  - website/static/index.html
  - website/static/js/mesh-webgl.js
  - website/static/css/styles.css
  - .github/workflows/deploy.yml
  - .github/workflows/diagram-render.yml
  - .github/workflows/drift-counter.yml
  - scripts/diagram-render/render.mjs
  - scripts/drift-counter/drift-counter.mjs
  - scripts/generate-release-manifest.sh
  - docs/architecture/compatibility-matrix.md
  - docs/README.md
  - package.json
date: 2026-08-31
---

# VisionFlow Baseline — What This Repo Is and Runs Today

## Purpose

Single source of truth for what the VisionFlow repository **is** and **runs** at
this commit. VisionFlow is not an application: it is the **ecosystem canon** — the
coordination and governance layer over the DreamLab repositories (VisionClaw,
agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website) — plus one shipped
artefact of its own, the `visionflow.info` **marketing website**. This document
fixes the two things that actually exist here in present tense, with `file:line`
citations, so that the legacy ADR-001..007 prose (now archived) is read as
evidence and history, never as the current build. Ground-truth order:
live code/config in this repo > legacy ADR prose.

## Current State

### The repo has two concrete surfaces

1. **A static marketing website** under `website/`, and
2. **Governance/coordination canon** under `docs/` — registers, compatibility
   matrix, protocol positions, PRDs/DDDs — plus CI gates that enforce claim
   integrity across the ecosystem.

Everything else in the tree (`pitch/`, `presentation/`, `website/`,
`the-bubble-is-the-architecture*.md`) is content that these two surfaces publish.
There is **no server runtime, no database, and no Rust code in this repository**
(`find` for `Cargo.toml`/`*.rs`/`*.wasm` returns nothing at `c205575`).

### The website is pure static HTML/CSS/JS — no WASM

`website/build.sh` is a pure copy step: it wipes `dist/`, copies `static/*`,
writes the `CNAME`, and stages repo images (`website/build.sh:11-27`). Its own
header states it outright: *"No compile step, no bundler, no WASM"*
(`website/build.sh:6-7`).

- The hero and scroll figures are a **hand-written WebGL2 ES module**,
  `website/static/js/mesh-webgl.js` (`initMesh(canvas)`, `getContext('webgl2')` —
  `mesh-webgl.js:13-15`), progressively enhanced (returns `null` without WebGL2;
  honours `prefers-reduced-motion` — `mesh-webgl.js:11,17`).
- The page loads exactly one script, `js/main.js` as `type="module"`
  (`website/static/index.html:826`), and one local stylesheet, `css/styles.css`
  (`index.html:17`). The only external fetch is Google Fonts (Inter + JetBrains
  Mono — `index.html:16`).
- CSS is **local static** (`website/static/css/styles.css`); there is no
  `cdn.tailwindcss` reference anywhere in `website/`.

### The website deploys to GitHub Pages via the artifact actions

`.github/workflows/deploy.yml` runs `website/build.sh`, uploads
`website/dist` with `actions/upload-pages-artifact@v3`, and publishes with
`actions/deploy-pages@v4` (`deploy.yml:24-49`). There is **no push to a
`gh-pages` branch**. The custom domain is `www.visionflow.info`, written into the
artefact by `build.sh` (`website/build.sh:20`).

### Browser verification runs through the external sidecar, not a local Chrome

`package.json` drives verification: `build` → `check:sidecar` → `test:site`
(`package.json:6-11`). Playwright connects over CDP to the shared
`browsercontainer` Chrome DevTools sidecar; the contract is
`http://browsercontainer:9223` in-network / `http://localhost:9222` from the host
(`docs/site-verification.md`). No local Chromium is installed in CI.

### The canon owns cross-repo governance and enforces it with CI gates

The governance role is documented, not merely asserted, and is backed by three
running CI gates:

- **Drift counter** — `scripts/drift-counter/drift-counter.mjs`, gated by
  `.github/workflows/drift-counter.yml` on changes under `scripts/drift-counter/**`
  (`drift-counter.yml:32,64`), anchored to `scripts/drift-counter/allowlist.json`.
  It fails CI when a prose count in the canon disagrees with its queried source.
- **Diagram-render gate (RES-b)** — `.github/workflows/diagram-render.yml` renders
  `presentation/report/diagrams/*.mmd` with a **vendored** Mermaid engine
  (`scripts/diagram-render/render.mjs`; `scripts/diagram-render/vendor/mermaid.min.js`
  11.16.0) and asserts rendered text carries a visible fill via
  `scripts/check-diagram-text.js` (`diagram-render.yml:19,32,47,74`).
- **Fixture / copyright / harness gates** — `fixture-drift.yml`,
  `copyright-guard.yml`, `harness-fitness-gates.yml`.

Release coordination uses `scripts/generate-release-manifest.sh` against
`docs/releases/ecosystem-release.schema.json`; the human-readable cross-repo posture
lives in `docs/architecture/compatibility-matrix.md`.

## Known divergences & open items

- **The website is not a Rust/WASM build — the docs still say it is.** Legacy
  ADR-001 D1/D3 decided "static HTML/CSS/JS with Rust WASM modules built via
  wasm-pack" and a Cargo workspace of `mesh-hero` + `particle-field` crates. **No
  such code exists** at `c205575`. Stale claims that must be corrected:
  `README.md:192` ("The site is a Rust/WASM build"), `README.md:196` ("wasm-pack
  builds both WASM crates"), `docs/site-verification.md:10,17` ("builds both WASM
  crates" / "wasm-pack release builds"), and `docs/PRD-website.md:75-101,153,167,197`
  (WASM module budgets, `mesh-hero`/`particle-field`). The truthful statement is
  the one in `website/build.sh:6-7`.
- **Tailwind Play CDN was never shipped.** Legacy ADR-001 D2 (Tailwind Play CDN)
  is superseded by local CSS — the ADR self-notes this and
  `docs/site-verification.md` records it. No follow-up ADR ever ratified local CSS
  as the permanent path; this baseline does so (see Invariants).
- **gh-pages branch push is superseded.** Legacy ADR-001 D4 (push a built `dist/`
  to the `gh-pages` branch) does not match `deploy.yml`, which uses the Pages
  artifact/deploy actions. ADR-001 self-amends; recorded here as settled.
- **Ontology-bridge tool count still shows two figures in the README.** ADR-005
  Decision 2 folded the MCP ontology-bridge count into the drift gate to close the
  "7 vs 12" conflict. Today `README.md:148` renders a diagram node "7 Ontology MCP
  Tools" **and** `README.md:155` "12 MCP Ontology Tools", while
  `compatibility-matrix.md:15` and `ecosystem-map.md:30` both read 12. The two
  numbers denote different things (VisionClaw-side vs bridge tools) but the
  adjacency in one README diagram is exactly the re-drift the gate exists to catch;
  the allowlist should disambiguate or annotate them.
- **The diagram-render gate is built under a different design than ADR-005 D3
  specified.** ADR-005 D3 called for `scripts/render-diagrams.sh` wrapping a pinned
  `mmdc`/puppeteer/chromium. What shipped is `scripts/diagram-render/render.mjs`,
  which itself depends on **neither** `mmdc` nor puppeteer — it renders a **vendored
  `mermaid.min.js`** and drives a Chrome binary over raw CDP (`scripts/diagram-render/lib/cdp.mjs`).
  The CI gate around it (`.github/workflows/diagram-render.yml`) still `npm install`s
  `@mermaid-js/mermaid-cli` and `puppeteer`, but only to provision that Chrome binary
  (puppeteer's resolved chromium) for `render.mjs` to attach to — not to render. Plus
  the text-fill probe `check-diagram-text.js`. The intent (reproducible render +
  invisible-text probe + baseline) is honoured; the named artefact in the ADR does not exist.
- **Dual ADR numbering hazard.** `docs/engineering/` carries its own
  `ADR-004-harness-engineering-framework.md` and `ADR-005-mandate-at-grant-governance.md`,
  colliding by number with the archived canon ADR-004/005. ADR-005's numbering note
  flagged this as unresolved. The new `docs/adr/` ledger (2xxx range, namespaced,
  index-validated) is the structural fix for the canon side; the `docs/engineering/`
  sequence is **out of scope of this cut** and remains where it is.
- **Governance-loop closure is proposed, not implemented.** Legacy ADR-007 (close
  the governance loop) is Proposed; its D1–D5 describe fixes to the ontology-bridge
  write path, decision-to-closure consumption, store backups, and key-registration
  canaries that are **specified, not shipped**. Legacy ADR-006's findings (bridge
  create-path bug; degraded discover) are open with named owners upstream. These are
  cross-repo (agentbox/VisionClaw) concerns tracked here, not code in this repo.
- **Engineering ADR-005 (mandate-at-grant) is speculative.** It self-labels
  "implementation is not committed" (kind 31406 absent from `agentbox.toml`); noted
  so a reader does not treat it as active.

## Invariants (must not silently change)

1. **The website has no build-time compilation.** `website/build.sh` stays a
   copy-only step; any reader or CI job may assume `dist/` is `static/` plus
   generated images. Reintroducing WASM/a bundler requires a new ADR and an update
   here — it is not a silent change.
2. **One script, one stylesheet, sidecar-verified.** The site loads a single
   `type="module"` entry and local CSS; browser verification runs through the
   external sidecar with **no local Chromium** in CI.
3. **Deploy is via the Pages artifact/deploy actions**, never a `gh-pages` branch
   push. The `CNAME` is emitted by `build.sh`, not committed as a branch file.
4. **The canon owns the cross-repo view; substrates own implementation.** Repo-local
   docs stay authoritative for their own code; VisionFlow owns compatibility,
   maturity vocabulary, and release evidence (legacy ADR-002). A maturity claim above
   the tier its evidence supports is a governance defect.
5. **Every count claimed in canon prose has one queryable source.** The drift gate
   must stay green; a second distinct figure for one axis anywhere in the tree is a
   failure, not a footnote.

## Change process

Any change to the website build, deploy mechanism, the canon's governance role, or
a CI gate requires: (1) updating the affected section here with the new `file:line`;
(2) confirming the relevant Invariant still holds (or amending it deliberately);
(3) bumping `version` and re-recording `verified_commit` from
`git rev-parse --short HEAD` in this repo; (4) recording the decision as a new
`docs/adr/ADR-NNNN-*.md` from `docs/adr/TEMPLATE.md` and regenerating the index
(`node scripts/adr-index-gen.cjs docs/adr`). Legacy ADR-001..007 prose in
`docs/archive/adr/` is citable evidence, never authority — cite it, do not defer to
it.
