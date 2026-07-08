# VisionFlow Pitch Deck — 20 Slides (2026-07)

A marketing deck spanning the full argument of *After the Collapse* (third edition,
339pp): the coordination problem, the societal stakes, the mesh answer, the four
surfaces, the honesty engine, and the wager. Each slide is one Nano Banana
infographic; the prompts live in [prompts.md](prompts.md) and are self-contained.

## Narrative arc

| # | Slide | Book source |
|---|-------|-------------|
| 1 | Title — After the Collapse | title page, ch1 |
| 2 | Hierarchy was an information technology | ch2–3 |
| 3 | 1812 / 2026 — who bears the transition | ch6 (new) |
| 4 | The jagged frontier | ch4 |
| 5 | The vigilance problem | ch5 |
| 6 | Centaur or reverse-centaur | ch6, ch4 |
| 7 | Three models, one survivor | ch7 |
| 8 | The economics of coordination | ch7a |
| 9 | The mesh, in three layers | ch7, ch14 |
| 10 | Four surfaces, one loop | ch13 (12a) |
| 11 | Identity is the spine | canon, ADR-125 |
| 12 | The forum owns the decision | ch13b |
| 13 | The desktop owns observation | ch13 (substrate) |
| 14 | Copresence at room scale | ch13 (XR) |
| 15 | Voice is the narrowest ingress | ch13, V-items |
| 16 | The sixpence, built in | ch6 §6, pods/402 |
| 17 | The honesty engine | ch14b (register) |
| 18 | KPIs of a compounding org | ch8–9 |
| 19 | The wager | ch14c, 14a |
| 20 | Build the institution | ch16, close |

## House style (ADR-111)

- Background cream `#F7F4EA`; linework charcoal `#2D2D2D` (hand-sketched
  engineering-notebook feel); accents teal `#1A6B6B` and burnt orange `#C85A2A`.
- 16:9, 4K. Generous margins; one idea per slide; label-light (≤10 exact strings).
- Every prompt pins its text with "render exactly, spell every word exactly" —
  the known garbled-label failure mode needs it.

## Render (after prompt review — do not run yet)

```bash
cd presentation/pitch-deck-2026-07
# per slide N with prompt file extracted from prompts.md:
node /home/devuser/workspace/project/agentbox/skills/art/tools/nb-generate.cjs \
  --prompt "$(cat slide-NN.prompt)" \
  --out slides/slide-NN.png --model gemini-3-pro-image --size 4K --aspect 16:9
```

Each render gets a visual QA pass (exact-label check) before acceptance; garbled
text is regenerated with a simplified label set, per the figure-QA precedent.
