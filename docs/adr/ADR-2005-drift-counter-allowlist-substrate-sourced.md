---
id: ADR-2005
title: Police canon self-description counts with an allowlist-anchored, substrate-sourced, fail-open drift counter
date: 2026-08-31
decision_status: accepted
implementation_status: partial
activation_status: live
supersedes: []
superseded_by: []
verified_commit: cf535f8
owner: jjohare
review_trigger: adding a counted axis, exposing a new substrate count source, or flipping an axis from reported to enforced
repo: visionflow
domain: BASELINE-visionflow.md
lineage: implements legacy docs/archive/adr/ADR-005-gap-close-canon-decisions.md Decision 2 (one canon counter, substrate-exposed sources, four axes), which itself folds in the "7 vs 12" MCP ontology-bridge tool-count drift finding it names in its own Decision 2 rationale.
---

# ADR-2005 — Police canon self-description counts with an allowlist-anchored, substrate-sourced, fail-open drift counter

## Context

Multiple self-description figures ran live in one tree with no single source: three
skill counts (90+/83+/106) and two ontology-bridge tool counts (7/10 vs the real 12).
A blind tree grep would false-positive on legitimately distinct figures elsewhere (a
case study's "350 skills", VisionClaw's native "7 MCP tools", agentbox's "180+"). The
count truths also live in a *sibling* repo (agentbox), not here.

## Decision

`scripts/drift-counter/drift-counter.mjs` reads each axis's truth from its
substrate-exposed source (agentbox `skill-count-check.js`; the `ontology-bridge.js`
`TOOLS` array length) and fails any PR whose canon figure disagrees, or that adds a
second distinct figure at a policed site. Two design choices are load-bearing: (1) it
is **allowlist-anchored** — only the sites in `allowlist.json` are policed, never a
whole-tree scan; (2) it is **fail-open per axis** — an axis whose source is not exposed
this wave is reported UNAVAILABLE and *not* enforced, so a down source blocks only that
axis; `--strict` flips unavailability into failure. Truth sources are fetched by
checking out agentbox beside the canon in CI (`DRIFT_AGENTBOX_DIR`), not vendored.

## Consequences

- Forecloses the simpler blind-grep gate: adding a genuinely new "N skills" figure
  about a different subject does not turn CI red, but every tracked figure now needs an
  explicit allowlist entry — the allowlist is maintenance debt that must track the prose.
- Forecloses hard-fail-on-missing-source: the `ontology-classes` (needs VisionClaw
  `ClassCountSource`) and `roster` axes ship reported-but-unenforced. `implementation_status`
  is therefore **partial** — the mechanism is live, two of four axes are not yet policed.
- Couples the canon's CI to a sibling repo checkout; an agentbox path/rename breaks the
  count source and shows as UNAVAILABLE, not a false failure — deliberate, but it means
  green does not always mean "all axes checked".
- The README ontology-tool count adjacency (VisionClaw-side 7 vs bridge 12) is the exact
  re-drift this gate exists to catch; the allowlist must disambiguate the two subjects.

## Verification

At `cf535f8`: `drift-counter.mjs` header + body implement the four-axis, source-of-truth,
allowlist model with `--strict`; `allowlist.json` enumerates the `skills` and
`mcp-ontology-tools` policed sites and names the excluded subjects; `drift-counter.yml:50-55`
checks out `DreamLab-AI/agentbox` into `_agentbox` and points `DRIFT_AGENTBOX_DIR` at it;
`drift-counter.yml:19-23` documents the UNAVAILABLE/PLANNED axes.
