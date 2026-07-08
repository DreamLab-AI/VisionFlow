# P0 — RES-b: Diagram-as-code render gate

| Field | Value |
|-------|-------|
| Register item | RES-b diagram-as-code render gate (residual, P0) |
| Canon owner docs | `docs/PRD-gap-close-canon.md` §"Diagram-as-Code Gate (RES-b)"; `docs/ADR-005-gap-close-canon-decisions.md` §Decision 1, §Decision 3 |
| Canary | `CANARY-CANON-DIAGRAM` |
| Branch | `gap-close/2026-07` |
| Base commit at work start | `86fb8284251101d9e7e8f557a6eb54a502aac46d` |
| Environment | node v22.22.3; Chrome sidecar `browsercontainer` (CDP `:9223`, health `:8931`) |
| Status | **Gate mechanism + light-theme baseline COMPLETE and proven locally (both canary halves). CI workflow authored. NOT yet `integrated` per canon: `CANARY-CANON-DIAGRAM` has not fired in a live GitHub Actions session (this box cannot push or run Actions).** |
| Maturity (honest) | `scaffolded` → blocks pending first live CI canary fire on a probe branch |

---

## The defect this gate exists to stop

The ten `presentation/report/diagrams/*.mmd` sources were authored for a **dark**
mermaid theme: each carries a leading `%%{init:{'theme':'dark',…}}%%` directive
and hardcoded near-white label colours (`color:#e0e0e0`) on a
`background-color: transparent`. Exported onto the white page of a PDF the theme
background dropped away and every label became invisible white-on-white text.

The legacy sibling SVGs shipped with that defect are still in the tree
(`presentation/report/diagrams/NN-*.svg`). Running the new checker against them
reproduces the failure concretely:

```
$ node scripts/check-diagram-text.js presentation/report/diagrams
  FAIL  07-change-architecture.svg: no visible text nodes; missing label words …
  FAIL  08-mesh-architecture-stack.svg: no visible text nodes; missing label words …
  …
8/10 diagram(s) failed the text-visibility gate.
```

(The two that pass are chart types — `quadrantChart`, `xychart` — that always
emit real `<text>`; the eight flowcharts used dark htmlLabels and have no visible
text at all.)

## What was built

| Path | Purpose |
|------|---------|
| `scripts/diagram-render/render.mjs` | Renders each `.mmd` to a **light** (`default` theme) SVG in a real headless Chrome via **raw Chrome DevTools Protocol** over node's built-in WebSocket (no npm runtime dependency). Strips the dark theme directive + dark palette (`lib/preprocess.mjs`), forces `htmlLabels:false` so labels are real `<text>`, and bakes the browser-computed text fill + a white background rect into every SVG. |
| `scripts/diagram-render/lib/cdp.mjs` | Minimal CDP client: connect to the sidecar (resolves the hostname to an IP because Chrome's `/json/version` rejects a non-IP Host header) or launch a local Chrome via `DIAGRAM_CHROME_BIN` for CI. |
| `scripts/diagram-render/lib/preprocess.mjs` | Dark-theme strip + `extractLabelWords()` (distinctive ≥4-char label words, stopwords removed). |
| `scripts/diagram-render/vendor/mermaid.min.js` | Vendored mermaid **11.16.0** UMD bundle — **no CDN dependency at render time**. `sha256 74d7c46dabca328c2294733910a8aa1ed0c37451776e8d5295da38a2b758fb9b`. |
| `scripts/check-diagram-text.js` | Standalone gate (pure node built-ins, no browser). Fails non-zero unless every rendered SVG has visible `<text>` (fill not none/transparent/white/near-white, opacity ≥ 0.1) **and** every distinctive label word from its `.mmd` appears in the **visible** text. A `--diff <baselineDir> <candidateDir>` mode compares by visible-word set (font-metric independent) as the drift guard. |
| `.github/workflows/diagram-render.yml` | On any PR touching `presentation/report/diagrams/**` (or the gate itself): validate the committed baseline (no browser), install pinned `@mermaid-js/mermaid-cli@11.16.0` + `puppeteer@23.11.1` for a chromium, re-render from source with the same vendored engine, re-check, and word-set-diff against the committed baseline. |
| `presentation/report/diagrams/rendered/NN-*.svg` | The ten light-theme baselines — the source of truth this gate protects. |

## Divergence from ADR-005 §Decision 3 (declared, justified)

ADR-005 §Decision 3 sketched `scripts/render-diagrams.sh` wrapping `mmdc` with a
pinned puppeteer/chromium plus a **pixel diff** baseline. The implemented
mechanism follows the gap-close task's refinement and diverges in three honest
ways, each an improvement for this specific defect:

1. **Render engine.** Local `mmdc` is broken (`ERR_MODULE_NOT_FOUND: puppeteer`)
   and there is no local chromium, so the baseline is rendered with a raw-CDP
   driver + vendored mermaid against the Chrome sidecar. CI still installs the
   pinned `@mermaid-js/mermaid-cli` the ADR named — used there as the chromium
   provider (via its `puppeteer` peer) for the same vendored-engine render, so
   the baseline and the CI re-render come from one engine.
2. **Regression check.** A **text-fill visibility assertion** replaces a bare
   pixel diff. ADR-005 §Decision 3 itself notes "a pixel diff alone can miss
   white-on-white text that matches a white baseline"; the fill assertion targets
   exactly that failure, and is what the two canary halves exercise.
3. **Drift diff.** Byte/pixel identity across a GPU sidecar and an ubuntu runner
   is font-metric fragile (sub-pixel text widths change `<tspan>` wrapping), so
   drift is enforced at the **visible-word-set** level (invariant to wrapping and
   element order, still catching an added/removed/changed label). The raw
   `git diff --stat` is printed informationally.

This does not edit ADR-005 in place; it is recorded here as the implemented
refinement of Decision 3.

## Scope boundary (honest)

- The gate protects the **SVG** baseline. The shipped report is LaTeX
  (`presentation/report/main.tex`), and **`main.tex` does not currently
  `\includegraphics` any of these ten diagrams** (verified: only `wardley-*.png`
  and `nb-*.jpg` are included). So the current `main.pdf` does **not** embed
  invisible mermaid text from these sources.
- The legacy defective `diagrams/NN-*.svg` and `images/NN-*.pdf` remain tracked
  and unreferenced by `main.tex`. Regenerating those PDFs (SVG→PDF) and wiring
  them into the report is a **follow-up** outside this residual's scope; deleting
  or overwriting tracked `presentation/` artifacts is deliberately not done here.
  When the diagrams are (re-)embedded, the correct source of truth is
  `presentation/report/diagrams/rendered/`.

## Receipts

### Sidecar health (2026-07-08 11:55:55Z)
```
$ curl -s http://browsercontainer:8931/health
{"status":"ok","transport":"sse","sessions":1,"chrome":true,"cdp":"127.0.0.1:9222","connector":"chrome-devtools-mcp"}
```

### Render run — all 10 (2026-07-08 11:55:55Z)
```
$ node scripts/diagram-render/render.mjs
browser: connected ws://172.20.0.3:9223/devtools/browser/…
  rendered 01-hierarchy-evolution.mmd -> rendered/01-hierarchy-evolution.svg  (text nodes: 124, visible: 124)
  rendered 02-three-models-comparison.mmd -> … (text nodes: 120, visible: 120)
  rendered 03-insight-ingestion-loop.mmd -> … (text nodes: 62, visible: 62)
  rendered 04-compound-flywheel.mmd -> … (text nodes: 56, visible: 56)
  rendered 05-kpi-dashboard.mmd -> … (text nodes: 13, visible: 13)
  rendered 06-collaboration-modes.mmd -> … (text nodes: 119, visible: 119)
  rendered 07-change-architecture.mmd -> … (text nodes: 140, visible: 140)
  rendered 08-mesh-architecture-stack.mmd -> … (text nodes: 120, visible: 120)
  rendered 09-resistance-patterns.mmd -> … (text nodes: 16, visible: 16)
  rendered 10-1-9-90-adoption.mmd -> … (text nodes: 78, visible: 78)
Rendered 10 diagram(s)
```
Every diagram: `visible == total` — no invisible text node.

### `ls rendered/` (2026-07-08 11:56:20Z)
```
01-hierarchy-evolution.svg      32040
02-three-models-comparison.svg  36566
03-insight-ingestion-loop.svg   24686
04-compound-flywheel.svg        23499
05-kpi-dashboard.svg             6819
06-collaboration-modes.svg      35024
07-change-architecture.svg      43697
08-mesh-architecture-stack.svg  26178
09-resistance-patterns.svg       8686
10-1-9-90-adoption.svg          23505
```

### Text check — PASS on baseline (2026-07-08 11:55:56Z)
```
$ node scripts/check-diagram-text.js presentation/report/diagrams/rendered
  ok    01-hierarchy-evolution.svg: 84 visible text nodes, 46/46 label words
  … (all 10 ok, N/N label words each) …
All 10 diagram(s) passed: visible text + key labels present.   # exit 0
```

### CANARY-CANON-DIAGRAM — both halves proven locally
```
# invisible-text regression (white-on-white injected into a copy) -> RED
$ node scripts/check-diagram-text.js <copy-with-white-text>
  FAIL  09-resistance-patterns.svg: no visible text nodes; 16 invisible text node(s) …
  1/1 diagram(s) failed the text-visibility gate.               # exit 1

# render error (broken .mmd) -> RED ; clean .mmd -> GREEN  (sidecar, 11:58:16Z)
$ node .tmp/probe-render-error.mjs
broken .mmd threw (render-error path): true  (Error: Parse error on line 3:)
clean  .mmd rendered (control): true                            # exit 0 (both correct)

# drift guard: renamed label -> RED ; identical -> GREEN ; different wrapping same words -> GREEN
$ node scripts/check-diagram-text.js --diff <renamed>  rendered   # exit 1 (removed "renamed", added "anxiety")
$ node scripts/check-diagram-text.js --diff <identical> rendered   # exit 0
$ node scripts/check-diagram-text.js --diff rendered   <rewrapped> # exit 0 (same words, different <tspan> split)
```

## Acceptance mapping (PRD §RES-b line 82)

| Criterion | State |
|-----------|-------|
| A broken `.mmd` fails the workflow | Proven **locally** (render-error probe → exit 1; renderer's try/catch → exit 1). Live-CI fire pending. |
| A clean `.mmd` renders and passes | Proven: all 10 render, text-check exit 0. |
| No published PDF ships with invisible/unrendered diagram text | Current `main.pdf` does not embed these diagrams (verified); gate + correct baseline prevent future regression; legacy PDF re-embedding is a documented follow-up. |
| `CANARY-CANON-DIAGRAM` has fired | **Logic proven locally (both halves).** Has **not** fired in a live GitHub Actions session — item is therefore **not** scored `integrated`, per the canon falsification clause. |

## To reach `integrated` / `released`

- `integrated`: open a probe PR touching a `.mmd` (or introducing an invisible-text
  regression) and observe `diagram-render.yml` turn the build **red**, then a clean
  push turn it **green** — the first live `CANARY-CANON-DIAGRAM` fire. Cannot be
  done from this box (no push / no Actions).
- `released`: pin the gate in a release manifest under `docs/releases/`.

## Reproduce
```
node scripts/diagram-render/render.mjs                              # needs the sidecar or DIAGRAM_CHROME_BIN
node scripts/check-diagram-text.js presentation/report/diagrams/rendered
```
