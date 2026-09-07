---
id: ADR-2009
title: The webgl-mesh deep is sidecar-only and leaves the nightly rotation until a browser runner exists
date: 2026-09-07
decision_status: accepted
implementation_status: complete
activation_status: live
supersedes: []
superseded_by: []
verified_commit: 626636b6a9a7923add04e5c866734ff8564e2b53
owner: jjohare
review_trigger: a browser-sidecar-equipped dream runner becoming available, or the 02:30 UTC CI beginning to collect mesh artefacts (shader source, canvas-fallback markup, noscript content) into the evidence pack
repo: visionflow
domain: BASELINE-visionflow.md
lineage: applies the ADR-2008 pattern — a surface the annexe cannot observe is collected by CI and read offline — to the mesh, and declines the alternative until that collection exists; shares ADR-2008's premise that the HP annexe holds no credentials and runs no browser.
---

# ADR-2009 — The webgl-mesh deep is sidecar-only and leaves the nightly rotation until a browser runner exists

## Context

Four nights have been spent on the `webgl-mesh` deep and none produced a mesh
observation. 2026-08-30 was INCONCLUSIVE; 2026-09-02 reached ACCEPT only by
redefining the deep as a static source audit; 2026-09-06 and 2026-09-07 were both
INCONCLUSIVE with zero mesh observations, the second confirming the hypothesis that
**0 of 5 required entrypoints** touch shader source or canvas-fallback content. The
cause is structural, not editorial: the deep's subject is compiled and rendered
behaviour, the HP annexe has no browser sidecar, and no evaluator reads
`website/static/js/mesh-webgl.js`. A slot whose evidence cannot be collected cannot
produce a falsifiable hypothesis, and 2026-09-06 correctly refused to close the gap
with a new-file "shader audit doc" that would have passed all four gate evaluators
while being exercised by none — promotion without evaluation.

## Decision

The `webgl-mesh` deep is **sidecar-only** and is **removed from the nightly rotation**
in `dream.config.json`. The rotation is four slots: `content-integrity`,
`build-pipeline`, `seo-and-meta`, `estate-health`.

Mesh questions — shader compilation and program linkage on real GPU/ANGLE drivers,
`WEBGL_lose_context` resilience, the canvas fallback with WebGL2 disabled, canvas
role/label accessibility, `prefers-reduced-motion` — remain legitimate and remain
unanswered. They are recorded as `HANDOFF (browser):` notes for a sidecar-equipped
run, which is what the `browser-checks-out-of-annexe` discipline in
`dream.config.json` already requires of every browser-bound finding. Removing the
slot changes where those questions are asked, not whether they are asked.

The slot returns when its evidence does, by either route: a dream runner with the
browser sidecar attached, or — the better option, and the one ADR-2008 has already
proven for the estate snapshot — CI collecting mesh artefacts (shader source dump,
canvas-fallback markup, `<noscript>` content) into the evidence pack at 02:30 UTC so
the annexe can read them offline. Until one of those exists, a night that would have
drawn `webgl-mesh` draws the next slot in the rotation instead.

## Consequences

- **The rotation stops burning nights.** Four of the last nine nights drew a deep that
  could not be measured. Those nights now land on surfaces that have evaluators.
- **The mesh loses its standing slot, and that is a real loss.** The hero mesh is the
  site's most complex artefact and is now unexamined by the nightly cycle until the
  sidecar or the CI collector lands. This record makes the gap explicit rather than
  letting an INCONCLUSIVE verdict imply the mesh was looked at and found unremarkable.
- **INCONCLUSIVE keeps its meaning.** A verdict that a night could not observe its
  subject should be rare and informative. Repeating it on a schedule for a structural
  reason turns it into noise and hides the nights where it means something.
- **The refusal precedent is preserved.** 2026-09-06 declined to invent a document to
  satisfy a gate. Parking the slot is the honest form of that refusal; widening the
  deep's definition until the existing evaluators appear to cover it would be the
  dishonest one.
- **Reinstatement has a named trigger, not a hope.** `review_trigger` above binds the
  slot's return to a specific capability existing, so the deep does not quietly stay
  parked once the blocker is gone.
- **Cost.** `dream.config.json` and this repo's baseline must be edited again to restore
  the slot, and the `shader-source` / `canvas-fallback` scan surfaces leave the config
  with it.

## Verification

At the `verified_commit` above:

- `node -e "console.log(require('./dream.config.json').slots.map(s=>s.deep).join(', '))"`
  prints `content-integrity, build-pipeline, seo-and-meta, estate-health` — four slots,
  no `webgl-mesh`.
- `grep -rn "mesh-webgl" scripts/dream-*.sh scripts/estate-health.mjs` returns nothing:
  no evaluator entrypoint reads the mesh module, which is the fact that makes the deep
  unmeasurable in the annexe and is the premise of this record.
- The four remaining evaluators pass against the built site at this commit:
  `BUILD-OK` (`pages: 1  bytes: 60241098`), `LINK-INTEGRITY-OK`
  (`internal-refs-checked: 13  missing: 0`), `META-SCAN-OK` (`required-missing: 0`),
  `SD-SCAN-OK` (`json-ld-blocks: 1  parse-errors: 0`), plus `ASSET-SCAN-OK`.
- `node scripts/adr-index-gen.cjs docs/adr` validates this record's frontmatter and
  regenerates the index.

Not verified by the above: that the mesh itself is correct. That is precisely the
question this record declines to answer in the annexe, and it stays open as a browser
handoff — no evidence here should be read as a statement about the mesh's runtime
behaviour.
