# Industrial show-and-tell

A show-and-tell script for a hypothetical industrial application of VisionFlow.

**Status:** first complete script and production plan; recordings, voiceover and generated scenes have not started. **Audience:** operations leaders and engineering teams considering a pilot. **Format:** a narrated walkthrough with presenter cues, provisionally 7 minutes 8 seconds at 16:9. Timings remain editorial estimates until the voiceover is recorded.

**Northbank Packaging** is the fictional setting. A technician reports a recurring problem on cooling-loop pump **P-204**. A maintenance engineer brings together the incident, the asset's relationships, document revisions and an earlier work record. The story ends with a reviewed investigation plan and a useful handover, while the cause of the fault remains unresolved.

Test one central promise: **can the next person see what was observed, what supports the decision, and what still needs checking?** This is an application concept built around existing estate components, not a claim that a complete industrial product has shipped.

## Read and use the package

| File | Purpose |
| --- | --- |
| [Script](script.md) | Exact narrator and character lines, timings, presenter cues and the intended audience takeaway |
| [Shot list](shot-list.csv) | One row per shot, including framing, actions, overlays, capture requirements and fallback |
| [Walkthrough data](walkthrough.json) | Structured scene record used to produce the script and CSV; keep all three views in sync when editing |
| [Production guide](production-guide.md) | Recording batches, three hero-scene briefs, voice direction, continuity and acceptance checks |
| [Research and claims](research.md) | Inspected source boundaries, application assumptions and evidence required before a shot can be called a working demonstration |
| [Scenario fixture](fixtures/scenario.json) | Fictional people, asset, documents, relations, questions and proposed decision; production data, not an application import format |
| [Planning review](review.json) | Timeline, continuity, source-hash and document checks; media remains unproduced |
| [Source pins](source-pins.json) | Exact checkout revisions, inspected ranges and whole-file hashes for the technical foundation |

Start with the script. Keep the research file beside the editor during capture. Use the shot IDs in filenames and review comments so that a changed line can be traced to the affected footage.

## What the film will show

Follow one incident through observation, connected evidence, an unanswered question, human review and handover. A short technical reveal near the end names Ontomark, Loom, Agentbox and VisionClaw. Other estate components can support later applications; the film does not imply that all of them are required for this pilot.

Use three explicit visual labels:

- **Software capture · synthetic data:** recorded behaviour from an identified running build. A source file alone does not earn this label.
- **Application concept · staged:** designed industrial screens, editorial diagrams, scripted answers and manual handoffs. Keep the label visible while they are on screen.
- **Illustrative GenAI scene:** cinematic plant imagery, with titles and asset identifiers added deterministically in the edit.

Opening disclosure: **“A hypothetical industrial workflow. Synthetic data. Staged integrations are labelled.”** Leave a small “Hypothetical workflow” identifier throughout. The proposed answer cards, document-supersession rules, maintenance workflow and work-order exchange need application work and capture evidence. They must not be passed off as current product UI.

## Scope of this side quest

This request covers the script and shot-by-shot direction, with recording, voiceover and hero generation later. This package completes that planning deliverable. It does not start model jobs, record operator sessions, publish a site, connect to a plant or change product code. The broader estate audit remains active and separate; this work adds no ADR or diagram-review completion credit.

Equipment maintenance, the fictional plant and the audience are first-draft choices. They can be changed without changing the underlying story: a team checks evidence, reviews a proposed change and leaves a traceable account for somebody else.
