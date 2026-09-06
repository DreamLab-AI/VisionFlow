---
id: ADR-2005
title: Police canon self-description counts with an allowlist-anchored, substrate-sourced, fail-open drift counter
date: 2026-08-31
decision_status: accepted
implementation_status: complete
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

## Closeout extension — 2026-09-04

Retain accepted/partial/live. Earlier captured drift results found divergent skill counts and stale pre-archive allowlist paths; unavailable/planned axes remain distinct from checked axes. The allowlist and source query implement a useful bounded check, while an unpinned sibling checkout and excluded prose limit repeatability and coverage. Earlier numerical observations remain dated evidence, not refreshed canonical counts.

**Closeout (CP-01/08/09):** Repair moved allowlist targets, define each denominator and pin the queried substrate revision. Explicitly accept or resolve unavailable axes. Demonstrate detection of an intentional mismatch at each enforced claim site and identify prose outside the gate.

[Canon assessment](../estate-review/canon-and-verification.md), [source hashes and local check receipt](../estate-review/evidence/canon-operative-closeout.json), [execution sequence](../estate-review/closeout/execution-sequence.md). Historical verification above is preserved; this annex assesses the current working tree.

## Acceptance progress — 2026-09-05

Retain accepted/live; **implementation_status advances `partial` → `complete`**
for the counter mechanism. The four CP-01/08/09 obligations — repair moved
targets, define each denominator, pin the queried substrate revision, and
explicitly accept or resolve unavailable axes — are all discharged, and
detection of an intentional mismatch is now a test rather than a claim. The two
axes that remain unenforced are unenforced because their *substrates* have not
published a source, which is the designed partial-source behaviour, not an
incomplete counter.

**Moved targets repaired.** Both `docs/ADR-002-ecosystem-alignment-governance.md`
sites reported `file-missing` on every run — the file had been archived to
`docs/archive/adr/`. A policed site that cannot be read is a hole in the gate
wearing the shape of coverage. Repointed; both now read and both agree at 12.

**Substrate revision pinned.** `allowlist.json` gains `source_pin`
(`DreamLab-AI/agentbox` @ `89301ec7c911eab270c00a0cf81596d0d4f15535`), and the
counter verifies the checkout's HEAD against it and fails on mismatch
(`--allow-pin-drift` downgrades to a warning). Unpinned, the gate compared canon
prose against whatever agentbox HEAD the runner happened to fetch, so the same
canon commit could pass on Monday and fail on Tuesday with no canon change: it
was measuring the sibling's motion, not the canon's drift. `drift-counter.yml`
now checks out that exact `ref`, and the counter fails if the two drift apart —
so moving the truth is a reviewed two-line diff.

**Denominators defined.** Every axis carries a `denominator` block stating what
is counted, the expression, the authority and — importantly — the exclusions,
since these are the estate's most confused figures. Skills counts
`skills/*/SKILL.md` in agentbox (not the 610 turbo-flow templates, not agent
specs); mcp-ontology-tools counts the `TOOLS` array of the ontology bridge (not
agentbox's ~180 total MCP tools, not VisionClaw's native ones). The denominator
is printed in the human report and carried in `--json`.

**Unavailable/planned axes explicitly resolved.** Both now carry a `resolution`
block naming the blocking condition, the owning substrate, the exact switch that
unblocks them, and `enforce_when_available: true`; the report prints
"resolves when: …" beside each. An axis marked unavailable with no stated exit
condition is indistinguishable from one nobody intends to finish.
`ontology-classes` waits on VisionClaw publishing a ClassCountSource; `roster`
waits on the forum exposing `agent_registry`. Neither is enforced, both are now
accountable.

**Drift found and repaired.** With the counter working, the skills axis was
genuinely red: seven policed sites stated 124 or 115 against a sourced truth of
126. All seven corrected (`README.md` ×2, `docs/ecosystem-map.md`,
`docs/PRD-website.md`, `website/static/index.html` ×3).
`docs/engineering/ADR-004` was *removed* from the scanned file list instead: its
`115 skills` sits in a dated §Context paragraph describing the situation when
the decision was taken, and rewriting an ADR's Context to track a live count
falsifies the record. This follows the exclusion the allowlist already applies
to `presentation/` narrative, and the reason is documented in `_files_note`.
That is the "prose outside the gate" the closeout asked to be identified.

**Tests and results.** `tests/gates/drift-counter.test.sh`, 28 assertions, all
passing, driven through a synthetic allowlist with a settable truth so it tests
the mechanism rather than today's figures. Case [2] is the canary the closeout
demanded: **an intentional mismatch at a policed site is detected** — the
counter reports `DRIFT … states 41, truth 42`, `RESULT: FAIL`, exit 1, and
`"ok": false` in `--json`. Case [3] proves a moved policed site now fails rather
than passing quietly; case [4] a vanished figure; cases [5]/[5b] the pin
mismatch and its explicit override. `drift-counter.yml` runs this suite *before*
the counter, so CI proves the gate can still fail before trusting it to pass.
Live run: **PASS**, both axes enforced, 20 policed sites read, zero
`file-missing`, pin OK.

Receipts: [drift-counter log](../estate-closeout/2026-09-05/logs/drift-counter.log),
[gate self-test log](../estate-closeout/2026-09-05/logs/gate-tests.log),
[gate closeout receipt](../estate-closeout/2026-09-05/gate-closeout-receipt.json).

**Remaining.** No hosted CI run. The two axes above stay unenforced pending
their substrates. The pin is a manual contract in two places (allowlist and
workflow `ref`) kept honest by the counter's own check rather than by
generation. Prose under `presentation/` and ADR Context paragraphs remain
outside the gate by design and are reconciled by hand.

Governed paths changed: `scripts/drift-counter/allowlist.json`,
`scripts/drift-counter/drift-counter.mjs`, `.github/workflows/drift-counter.yml`,
`tests/gates/drift-counter.test.sh` (new), and the seven corrected figures in
`README.md`, `docs/ecosystem-map.md`, `docs/PRD-website.md`,
`website/static/index.html`.
