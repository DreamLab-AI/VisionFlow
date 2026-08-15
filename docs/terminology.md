# Terminology playbook — ontology, knowledge graph, reasoning, and the layers between

**Status**: Canon. This page owns the words. Every public surface in the mesh — README files,
narrativegoldmine.com, visionflow.info, pitch material — uses these terms this way, and the
first use of each term on any surface carries its one-line gloss from the table below.

The problem this solves: the ecosystem genuinely contains an ontology, a knowledge graph, a
reasoner and a grounding layer, and our own copy has drifted into using the words
interchangeably. A reader who meets "living ontology" on one page and "knowledge graph" on the
next for the same artefact concludes we don't know the difference. We do; the copy should show it.

## The stack, bottom-up

Read as layers. Each one consumes the layer below it.

| # | Term | What it is here | The artefact |
|---|------|-----------------|--------------|
| 1 | **Taxonomy** | The is-a hierarchy: `rdfs:subClassOf` only. One part of the ontology, never a synonym for it. Ours is a **lattice, not a tree** — 1,396 classes declare more than one parent, and that overlap is published as data. | The subClassOf backbone inside `ontology.ttl` |
| 2 | **Ontology** | The formal vocabulary: OWL 2 classes, 15 typed object properties (`enables`, `requires`, `hasPart`, `bridgesTo`, …), and the axioms that constrain what a valid statement looks like. Defines what *can* be said. **narrativegoldmine.com is the ontology itself in readable form**: the corpus is pure TBox — every page declares a class, zero individuals by design — and the pipeline compiles it losslessly into the formal artefact. | The published corpus + `ontology.ttl` / `.owl` / JSON-LD compiled from it |
| 3 | **Knowledge graph** | The ontology populated with live instances — TBox *plus* ABox. This layer lives at **runtime**, not in the published corpus: VisionClaw's running graph, agents' working graphs, and the personal graphs written into Solid pods are where individuals exist and are asserted against the ontology's classes. Calling the published corpus a knowledge graph overclaims (it has no instance data); calling the runtime graph one is exact. | VisionClaw runtime graph; workingGraph; pod-resident personal graphs |
| 4 | **Reasoning** | What *follows*, machine-checked — at two points. At **build time** the pipeline's pure-Python EL-profile reasoner computes the inferred closure and gates the published ontology. At **runtime** the Whelk EL++ reasoner (whelk-rs, in VisionClaw) classifies the shared graph and rejects contradictions before they enter it. "Reasoned" on our surfaces always means machine-checked by one of these two, never "an LLM thought hard". | `pipeline/reason.py` (build); Whelk in VisionClaw (runtime) |
| 5 | **Grounding** (context assembly) | The serving layer: at query time the **Ontology Loom** retrieves the relevant slice of the reasoned ontology and injects it into an LLM's context as a structured scaffold, so the model restates checked facts instead of doing open-ended recall. This is the layer the 2026 industry calls a **context graph** — the top of the stack, consuming everything below it to assemble an agent's working set. The term is emerging, not settled: vendors also use it for decision-trace audit logs and temporal agent memory. We use only the assembly sense, and we present it as an industry label, not a standard. | Loom (`/loom/scaffold`, `/v1/chat/completions`); `ontology_ask` in agentbox |
| 6 | **Semantic layer** | *Not ours.* The parallel concept from the BI world — governed meaning for business metrics over warehouse data. We do not ship one; never use this term for any part of this stack. | — |

Two ecosystem-specific facts that explain why our copy drifted, and that the copy may state:

- **The published corpus is pure TBox, and that licenses the word "ontology".** Every
  narrativegoldmine page declares an OWL class (pages and classes are 1:1); every typed
  relation onto a declared class also emits an `owl:Restriction` as an extra
  `rdfs:subClassOf`; there are **zero individuals, by design**. This is not a gap — it is
  the defining structural fact. It means "ontology" is defensible down to the last axiom,
  while "knowledge graph" invites the instance-data challenge ("show me your entity
  resolution") that has no answer here. When a sentence needs one word for the published
  artefact, that word is **the ontology** (or **corpus** for the markdown source form,
  which compiles into it losslessly). Instance data genuinely exists in the mesh — in the
  runtime layer — and that is where the words "knowledge graph" now point.
- **Reasoning is a named, running component, not an aspiration.** The industry stack diagram
  leaves inference implicit inside "logical rules"; ours runs at two points, gates writes,
  and has names: the pipeline's EL-profile closure at build, Whelk at runtime.
  Say the names, and attribute each check to its owner.

## One-line glosses (use verbatim at first mention on a surface)

- **ontology** — "the formal vocabulary: the classes, typed properties and rules that define what can be said"
- **knowledge graph** — "the ontology populated with live instances at runtime: the graph VisionClaw renders and agents and pods write against it"
- **reasoning** — "the machine check: an EL-profile reasoner computes and gates the published closure at build; Whelk classifies the shared runtime graph and rejects contradictions before they enter it"
- **Ontology Loom** — "the grounding layer: it retrieves the relevant slice of the graph into an LLM's context at query time, so answers restate checked facts rather than guesses"
- **taxonomy** — "the is-a backbone of the ontology — here a lattice, not a tree"

## Rules

1. **The published artefact is the ontology; the runtime graph is the knowledge graph.**
   Class and axiom counts belong to the ontology; page and word counts belong to the corpus
   (its markdown source form, compiled in losslessly); "knowledge graph" is reserved for
   the runtime layer where individuals live. Never call the published corpus a knowledge
   graph — it has no instance data to back the claim.
2. **"Living ontology" is retired.** The canonical line for the whole published artefact is
   "a Logseq corpus that is also an OWL ontology".
3. **The `/ontology` explorer may keep its name** — it genuinely visualises the class-and-
   property structure. Call it "the 3D ontology explorer"; do not call the *page* graph "the
   3D ontology".
4. **"Reasoned" is earned, not decorative — and attributed.** Unqualified "reasoning" on
   these surfaces is symbolic: the pipeline's EL-profile closure at build time, Whelk at
   runtime. Never credit Whelk with the pipeline's closure or vice versa. LLM inference is
   always qualified: "LLM reasoning", "chain-of-thought".
5. **"Neurosymbolic" is positioning, not a brand.** Use it at most once per surface, as the
   industry's name for the architecture we already run — thin agents over a shared formal
   semantic layer — and pair it with the concrete pieces (OWL 2 EL + Whelk + Loom).
6. **The synthetic-corpus line is one sentence, once per surface.** The corpus is produced by
   an automated research process between researchers and agents, is validated (0 errors,
   0 warnings), and stands on its own merits. State the provenance honestly in one line
   (the existing `corpus.statement` pattern) and move on. Do not apologise for it, do not
   re-explain it in every section, and do not let it displace what the corpus *is*.
7. **The whole is the Dynamic Agentic Mesh.** Seven repositories, five running substrates,
   one identity spine. "Ecosystem" is acceptable in running prose; the proper noun is the mesh.
8. **Numbers come from the pipeline — cite, don't restate.** Stats quoted on any surface
   carry the `stats.json` `datasetDate` they came from and link to the live artefact.
   Prefer order-of-magnitude prose ("8,100+ classes") with a link over precise restated
   figures: builds ship often enough that restated precision rots in days.

## The canonical sentence

When one sentence has to carry the whole stack:

> Researchers and agents write a corpus; a pipeline compiles it losslessly into a formal
> OWL 2 ontology, machine-checking every statement as it builds; at runtime the mesh's
> knowledge graphs — VisionClaw's live graph, agents' working graphs, pod-resident personal
> graphs — populate that ontology with instances under Whelk's gate; and the Ontology Loom
> serves the checked ontology back into any model's context at query time.

## Where the terms live

| Surface | File | Owner |
|---------|------|-------|
| VisionFlow README | `README.md` (this repo) | canon |
| visionflow.info | `website/static/index.html` | canon |
| narrativegoldmine front page | `logseq:publishing-tools/WasmVOWL/modern/src/pages/HomePage.tsx` | knowledgeGraph |
| narrativegoldmine repo README | `logseq:README.md`, `knowledgeGraph:README.md` | knowledgeGraph |
| Loom README | `loom:README.md` | loom |
| Sibling READMEs | VisionClaw, agentbox, solid-pod-rs, nostr-rust-forum, dreamlab-ai-website | each repo |
| KG term pages | `Ontology.md`, `Knowledge Graph.md`, `Reasoning.md`, `Glossary Index.md` + this playbook mirrored as a page | knowledgeGraph |
