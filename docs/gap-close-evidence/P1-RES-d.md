# P1 — RES-d: Self-description drift counter

| Field | Value |
|-------|-------|
| Register item | RES-d self-description drift counter (residual, P1) |
| Canon owner docs | `docs/PRD-gap-close-canon.md` §"Automated Self-Description Counter (RES-d)"; `docs/ADR-005-gap-close-canon-decisions.md` §Decision 2; `docs/DDD-gap-close-canon-context.md` §9 (`DriftCounter`) |
| Canary | `CANARY-CANON-DRIFT` |
| Branch | `gap-close/2026-07` |
| Base commit at work start | `f72d173cd57d48fcf66c1a8c42628e6e6d11511c` |
| Environment | node v22.22.3; agentbox count sources at `/home/devuser/workspace/project/agentbox` (`96736244`) |
| Maturity (honest) | `scaffolded` — mechanism complete, both canary halves proven **locally** (green tree ↔ red on injected drift); `CANARY-CANON-DRIFT` has **not** fired in a live GitHub Actions session (this box cannot push or run Actions). Not scored `integrated`, per the canon falsification clause (DDD Gap-Close Invariant 4). |

## What was built

| Path | Purpose |
|------|---------|
| `scripts/drift-counter/drift-counter.mjs` | The counter + gate. Pure node ESM (no runtime deps). Queries each axis from its substrate-exposed source, checks every policed canon figure against that truth, exits `1` on any drift on an enforced axis. |
| `scripts/drift-counter/allowlist.json` | The four counted axes + the policed assertion sites. Allowlist-anchored, not a blind tree grep. |
| `scripts/drift-counter/README.md` | Axis table, run instructions, the allowlist-vs-grep rationale, the partial-source failure mode. |
| `.github/workflows/drift-counter.yml` | CI gate. Checks out the canon + agentbox (count sources), runs the counter on any PR touching README/docs/the two book files/the counter. `CANARY-CANON-DRIFT` fires here. |

## Axes (ADR-005 §Decision 2)

| Axis | Source of truth | Exposed by | This wave |
|---|---|---|---|
| `skills` | `scripts/skill-count-check.js` (`.count`) — **invoked** | agentbox | **enforced**, truth = **115** |
| `mcp-ontology-tools` | `mcp/servers/ontology-bridge.js` `TOOLS` array length | agentbox | **enforced**, truth = **12** |
| `ontology-classes` | Oxigraph `owl:Class` count / committed count file | VisionClaw | **UNAVAILABLE** — `ClassCountSource` is `planned` (`ddd-gap-close-visionclaw-context.md:155`); partial-source mode (below) |
| `roster` | forum `agent_registry` | nostr-rust-forum | **PLANNED** — not yet single-sourced (DDD-gap-close-canon-context §12 open issue 3) |

**Partial-source failure mode (ADR-005 §Decision 2; the join's canon-side
decision).** An axis whose source is not exposed this wave is reported
`UNAVAILABLE` and is not enforced — a source being down blocks *that* axis, not the
whole gate. A figure that *disagrees* with an *available* source is a hard failure.
`--strict` turns unavailability into a failure. The ontology axis wiring is live and
merely awaiting VisionClaw's source: setting `DRIFT_VISIONCLAW_CLASS_COUNT=5975`
flips it to `ENFORCED` (proof below), so the counter is not hardcoded to three axes.

## Counted truth as of 2026-07-08

- **Skills = 115.** Emitted by agentbox `scripts/skill-count-check.js` (`.count`), source `skills/*/SKILL.md`.
- **MCP ontology-bridge tools = 12.** `const TOOLS = [ … ];` in `ontology-bridge.js` — 11 object-literal elements + 1 imported reference (`propose.ONTOLOGY_PROPOSE_TOOL`); the `{ name: 'ontology-bridge', version }` server-identity line is not a tool and is not counted.

## The six drifting prose assertions the recon enumerated (replaced)

All six live in `README.md`; all replaced with the counted value:

| # | Site | Was | Now | Axis |
|---|------|-----|-----|------|
| 1 | `README.md:130` | `7 MCP Ontology Tools` | `12 MCP Ontology Tools` | mcp-ontology-tools |
| 2 | `README.md:134` | `90+ Agent Skills` | `115 Agent Skills` | skills |
| 3 | `README.md:219` | `90+ skills` | `115 skills` | skills |
| 4 | `README.md:395` | `90+ skills` | `115 skills` | skills |
| 5 | `README.md:473` | `90+ skills` | `115 skills` | skills |
| 6 | `README.md:780` | `90+ skills` | `115 skills` | skills |

## Additional consistency reconciliations (required for a tree-wide-green gate)

The recon's reconciled-claims table under-enumerated the drift. The "10 MCP tools"
ontology-bridge figure — which the table claimed was "retired by `edad233`" —
survived in six further canon docs, and a third skill figure ("106", "83+", "101")
survived in the engineering framework and the book. All are the same tracked axes;
leaving them would let a second distinct count survive in the tree and falsify RES-d.
Reconciled to the counted values:

**Skills → 115:** `docs/ecosystem-map.md:30`, `docs/PRD-website.md:122`,
`docs/engineering/ADR-004-harness-engineering-framework.md:13` (`106`→`115`),
`presentation/the-coordination-collapse.md:470` (`101`→`115`), `:620` (`83+`→`115`),
`:794` (`83`→`115`), `presentation/google-analysis.md:561` (`83+`→`115`).

**MCP ontology-bridge → 12:** `docs/DDD-ecosystem-alignment-context.md:24,112`,
`docs/architecture/status-reconciliation.md:36`,
`docs/ADR-002-ecosystem-alignment-governance.md:46,83`,
`docs/engineering/DDD-harness-engineering-context.md:223`,
`docs/engineering/PRD-harness-engineering.md:52`,
`docs/protocol/mesh-smoke-test.md:28`, `docs/PRD-ecosystem-alignment.md:90` (all `10`→`12`).

## Deliberately NOT changed (distinct subjects / not tracked axes)

- `README.md:779` "**7 MCP tools**" — VisionClaw's *native* MCP tool count (7 per the
  `edad233` fix), a different axis from the agentbox ontology bridge (12). Left at 7.
- "**180+ MCP tools**" (agentbox total) — a different axis. Left.
- `presentation/the-coordination-collapse.md:667` "**10-15 agent skills**" — a
  pilot-selection *range*, not a substrate count. Left.
- The Ramp/Glass case study's "**350 skills**"
  (`presentation/2026-04-20-best-companies-ai/*`) — an external company, not agentbox.
  Left. (This is why the gate is allowlist-anchored, not a blind `\d+ skills` grep.)
- The reconciliation records themselves (`PRD-gap-close-*`, `ADR-005-gap-*`,
  `DDD-gap-close-*`, `registers/`, `gap-close-evidence/`) legitimately quote the old
  figures as history; excluded from the gate.

## Receipts

### 1. Counter RED on the drifting tree (before fix) — 2026-07-08T12:45:55Z, base `f72d173cd`
```
$ DRIFT_AGENTBOX_DIR=…/agentbox node scripts/drift-counter/drift-counter.mjs
  [ENFORCED]    skills = 115   (source: …/skill-count-check.js (.count))
      DRIFT README.md:134  states 90, truth 115
      DRIFT README.md:219  states 90, truth 115
      … (README:395,473,780; ecosystem-map:30; PRD-website:122; ADR-004-harness:13=106;
         coordination-collapse:470=101,:620=83,:794=83; google-analysis:561=83) …
  [ENFORCED]    mcp-ontology-tools = 12   (source: …/ontology-bridge.js (TOOLS.length))
      DRIFT README.md:130  states 7, truth 12
      DRIFT docs/DDD-ecosystem-alignment-context.md:24  states 10, truth 12
      … (DDD:112; status-reconciliation:36; ADR-002:46,83; DDD-harness:223;
         PRD-harness:52; mesh-smoke-test:28; PRD-ecosystem-alignment:90 — all 10) …
  [UNAVAILABLE] ontology-classes — VisionClaw ClassCountSource is `planned` …
  [PLANNED]     roster …
RESULT: FAIL — drift detected.                                          # exit 1
```
(The whole-file skills scan also surfaced `coordination-collapse.md:667` "10-15 agent
skills" — a pilot *range* — which is why the book is excluded from the scan and its
current-count references reconciled by hand; see "Deliberately NOT changed".)

### 2. Counter GREEN after the reconciliations — 2026-07-08T12:49:09Z
```
$ DRIFT_AGENTBOX_DIR=…/agentbox node scripts/drift-counter/drift-counter.mjs
  [ENFORCED]    skills = 115   … ok ×8
  [ENFORCED]    mcp-ontology-tools = 12   … ok ×13
  [UNAVAILABLE] ontology-classes …
  [PLANNED]     roster …
RESULT: PASS — no drift on any enforced axis.                           # exit 0
```

### 3. CANARY-CANON-DRIFT — injected second figure turns it RED — 2026-07-08T12:49:47Z
```
# probe: README:219 115→83 skills ; README:130 12→7 MCP Ontology Tools
$ DRIFT_AGENTBOX_DIR=…/agentbox node scripts/drift-counter/drift-counter.mjs
      DRIFT README.md:219  states 83, truth 115
      DRIFT README.md:130  states 7, truth 12
RESULT: FAIL — drift detected.                                          # exit 1
# probe reverted → GREEN restored 2026-07-08T12:50:12Z, exit 0
```

### 4. Ontology axis wiring is live (not hardcoded) — 2026-07-08T12:49:47Z
```
$ DRIFT_VISIONCLAW_CLASS_COUNT=5975 …/drift-counter.mjs
  [ENFORCED]    ontology-classes = 5975   (source: DRIFT_VISIONCLAW_CLASS_COUNT (env))
```

### 5. Final green + JSON summary — 2026-07-08T12:56:16Z
```
$ …/drift-counter.mjs --json | (summary)
ok: true
  skills: state=enforced truth=115 sites=8 drift=0
  mcp-ontology-tools: state=enforced truth=12 sites=13 drift=0
  ontology-classes: state=unavailable sites=0 drift=0
  roster: state=planned sites=0 drift=0
$ …/drift-counter.mjs --quiet ; echo EXIT=$?
EXIT=0
```

### 6. No residual tracked-axis drift outside reconciliation docs — 2026-07-08T12:50:12Z
```
$ grep -rnE '90\+ skills|83\+ skills|10 MCP tools|10-tool MCP|7 MCP Ontology' --include='*.md' . \
    | grep -vE 'node_modules|\.git/|PRD-gap-close|ADR-005-gap|DDD-gap-close|gap-close-evidence|registers/'
(no output)
```

## Acceptance mapping (PRD §RES-d line 74)

| Criterion | State |
|-----------|-------|
| The counter runs as a CI job | `.github/workflows/drift-counter.yml` authored; runs the counter against agentbox-exposed sources. Live-CI fire pending (box cannot push). |
| An injected second count on a probe branch turns the build red | Proven **locally** (receipt 3): injected `83`/`7` → exit 1; reverted → exit 0. |
| The drifting prose assertions replaced by the counted values | The six README sites + all further tracked-axis drift replaced with 115 / 12 (receipts 2, 6). |
| `CANARY-CANON-DRIFT` has fired | **Logic proven locally (both halves).** Has **not** fired in a live GitHub Actions session — item held at `scaffolded`, not `integrated`, per the falsification clause. |

## To reach `integrated` / `released`

- `integrated`: open a probe PR that injects a second skill/MCP figure and observe
  `drift-counter.yml` turn the build **red**, then a clean push turn it **green** — the
  first live `CANARY-CANON-DRIFT` fire. Cannot be done from this box (no push / no Actions).
- `released`: pin the counter in a release manifest under `docs/releases/`.
- Ontology axis: `integrated` on VisionClaw publishing its `ClassCountSource` (env or
  committed count file), at which point the axis enforces without further canon change.

## Reproduce
```
DRIFT_AGENTBOX_DIR=/path/to/agentbox node scripts/drift-counter/drift-counter.mjs
DRIFT_AGENTBOX_DIR=/path/to/agentbox node scripts/drift-counter/drift-counter.mjs --json
```
