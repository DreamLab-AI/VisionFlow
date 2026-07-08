# P2 — RES-e: Wardley export quality (clean re-export, print resolution)

| Field | Value |
|-------|-------|
| Register item | RES-e Wardley export quality (residual, P2) |
| Canon owner docs | `docs/PRD-gap-close-canon.md` §"Wardley Export Quality (RES-e)"; `docs/DDD-gap-close-canon-context.md` §4 (`Residual RES-e`) |
| Canary | none — one-shot content correction, acceptance check not a canary (PRD §"Wardley Export Quality") |
| Branch | `gap-close/2026-07` |
| Base commit at work start | `b47c5fd652eaf5306118dbf9a511e31a74755b33` |
| Maturity (honest) | `integrated` — scripted clean export, 5/5 maps re-exported without chrome at ≥300 DPI (3000×2000), text-visibility verified, images the book includes replaced. |

## What was done

The five maps ship as self-contained HTML (`presentation/report/wardley/*.html`)
produced by the project's `WardleyMapGenerator` — the same generator the `.owm`
sources drive. Each HTML embeds a clean 1200×800 `<svg>` map wrapped in the tool's
UI chrome (a page card, a *"Wardley Map"* heading, and Export SVG / Export PNG /
Toggle Grid buttons). The shipped book figures were full-page screenshots that baked
that chrome and sat below print resolution: `wardley-01/04/05` at 1400×1094 with the
buttons visible.

`scripts/render-wardley.mjs` (committed) re-exports **without chrome at print
resolution**, deterministically and with no network or sidecar dependency:

1. extracts the embedded `<svg>` from each HTML — this is exactly what the tool's
   *Export SVG* action serialises (`exportSVG()` in the HTML), i.e. an export, **not
   a screenshot** — dropping the card, heading and buttons;
2. normalises in-`<text>` whitespace (component names carry literal newlines; a
   browser collapses each to one space, librsvg drops it — normalising matches the
   browser figure the book already shipped) and adds an explicit white background
   plus an Arial-metric-compatible sans font (`Liberation Sans`);
3. rasterises the clean SVG to a 3000×2000 PNG via ImageMagick's librsvg delegate
   (RSVG 2.62.1) at density 240 — ≈312 DPI for the book's ~9.6-inch `\textwidth`
   figure, ≥300 DPI and ≥2800 px wide;
4. writes clean `.svg` + `.png` to `presentation/report/wardley/rendered/` and
   **replaces the chrome-baked `images/wardley-0N-*.png` the `.tex` includes, keeping
   the filenames**.

The book includes only two of the five directly (grep receipt below); all five were
re-exported for a consistent, print-quality set. No `.tex` include path changed.

## Receipts

**Render command + output:**
```
$ node scripts/render-wardley.mjs
  01-coordination-value-chain.html -> …/wardley/rendered/01-….svg, …/images/wardley-01-….png (3000x2000)
  02-three-models-compared.html   -> …/wardley/rendered/02-….svg, …/images/wardley-02-….png (3000x2000)
  03-middle-manager-evolution.html-> …/wardley/rendered/03-….svg, …/images/wardley-03-….png (3000x2000)
  04-governance-landscape.html    -> …/wardley/rendered/04-….svg, …/images/wardley-04-….png (3000x2000)
  05-visionclaw-tech-stack.html   -> …/wardley/rendered/05-….svg, …/images/wardley-05-….png (3000x2000)
  done: 5 maps re-exported clean at print resolution.
```

**Image dimensions (identify) — all ≥2800 px wide:**
```
$ identify -format "%f  %wx%h\n" presentation/report/images/wardley-0*.png
wardley-01-coordination-value-chain.png  3000x2000
wardley-02-three-models-compared.png     3000x2000
wardley-03-middle-manager-evolution.png  3000x2000
wardley-04-governance-landscape.png      3000x2000
wardley-05-visionclaw-tech-stack.png     3000x2000
```
(before: wardley-01/04/05 = 1400×1094 chrome-baked; wardley-02/03 = 2254×1568.)

**grep of `.tex` includes (which maps the book uses):**
```
$ grep -rn "graphicspath\|includegraphics.*wardley" presentation/report/{main.tex,chapters,appendices}
presentation/report/main.tex:154:\graphicspath{{images/}{diagrams/}{wardley/}}
presentation/report/chapters/14-implementation.tex:209:\includegraphics[width=\textwidth]{wardley-03-middle-manager-evolution.png}
presentation/report/appendices/appendix-b-competitor-matrix.tex:117:\includegraphics[width=\textwidth]{wardley-02-three-models-compared.png}
```
Both book-used figures (02, 03) replaced at 3000×2000; filenames unchanged, so the
includes resolve unchanged.

**Text-visibility check (RES-b invariant, applicable part):** the RES-b checker's
default mode requires a paired `.mmd` source (it is the mermaid diagram gate) and so
reports *"no matching .mmd source"* for the `.owm`/`.html`-sourced wardley SVGs — a
source-pairing mismatch, not a visibility failure. Its core invariant — every
non-empty `<text>` resolves to a visible (non white/none/transparent, non
near-zero-opacity) fill — **does** apply and passes 5/5:
```
  PASS  01…svg: 21 visible text nodes, 0 invisible; fills={#666,(inherited),#333}
  PASS  02…svg: 19 visible text nodes, 0 invisible; fills={#666,(inherited),#333}
  PASS  03…svg: 20 visible text nodes, 0 invisible; fills={#666,(inherited),#333}
  PASS  04…svg: 20 visible text nodes, 0 invisible; fills={#666,(inherited),#333}
  PASS  05…svg: 19 visible text nodes, 0 invisible; fills={#666,(inherited),#333}
  All wardley SVGs pass the text-visibility invariant.
```

**Rendered-image read:** `wardley-01` and `wardley-03` were read back at 3000×2000
and confirmed: no page card, no *"Wardley Map"* heading, no Export buttons; white
background; sans-serif labels legible; axes (Genesis/Custom/Product/Commodity,
Visible/Invisible, Value Chain →, Evolution →), evolution zones, and component dots
all present; multi-word labels correctly spaced after the whitespace fix.

## Acceptance mapping (PRD §"Wardley Export Quality (RES-e)")

| Criterion | State |
|-----------|-------|
| Each of the five `.owm` re-exports via the export action, not a screenshot | Done — embedded-SVG extraction (the Export-SVG serialisation), rasterised; no screenshot |
| ≥300 DPI with app chrome cropped | Done — 3000×2000 (≈312 DPI at `\textwidth`), chrome removed |
| Scripted | Done — `scripts/render-wardley.mjs` |
| Filenames the `.tex` expects preserved | Done — `images/wardley-0N-*.png` replaced in place |
| ≥2800 px wide | Done — 3000 px wide, all five |

## Falsification exposure

Falsified if any replaced `images/wardley-0N-*.png` still shows tool chrome (card,
heading or Export buttons), is under 2800 px wide, carries invisible text, or if a
`.tex` include path was changed. None holds: all five are 3000×2000, chrome-free,
text-visible, at the original filenames.
