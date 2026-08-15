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
| 2 | **Ontology** | The formal vocabulary: OWL 2 classes, 15 typed object properties (`enables`, `requires`, `hasPart`, `bridgesTo`, …), and the axioms that constrain what a valid statement looks like. Defines what *can* be said. | `ontology.ttl` / `.owl` / JSON-LD, compiled by the pipeline |
| 3 | **Knowledge graph** | The vocabulary populated at scale: the corpus of pages compiled into triples and resolvable typed edges. What *is* said. narrativegoldmine.com is this graph in readable form; VisionClaw renders the same graph in immersive 3D. | The compiled graph (286,533 triples with the Whelk-inferred closure, dataset 2026-08-11) + the page corpus that sources it |
| 4 | **Reasoning** | What *follows*, machine-checked: the Whelk EL++ reasoner classifies the graph (derives entailments) and gates writes — a contradiction is rejected before it enters the shared graph. "Reasoned" on our surfaces always means *checked by Whelk*, never "an LLM thought hard". | Whelk (whelk-rs) in VisionClaw; the validation gate in the pipeline |
| 5 | **Grounding** (context assembly) | The serving layer: at query time the **Ontology Loom** retrieves the relevant slice of ontology + knowledge graph and injects it into an LLM's context as a structured scaffold, so the model restates checked facts instead of doing open-ended recall. This is the layer the 2026 industry calls a **context graph** — the top of the stack, consuming everything below it to assemble an agent's working set. The term is emerging, not settled: vendors also use it for decision-trace audit logs and temporal agent memory. We use only the assembly sense, and we present it as an industry label, not a standard. | Loom (`/loom/scaffold`, `/v1/chat/completions`); `ontology_ask` in agentbox |
| 6 | **Semantic layer** | *Not ours.* The parallel concept from the BI world — governed meaning for business metrics over warehouse data. We do not ship one; never use this term for any part of this stack. | — |

Two ecosystem-specific facts that explain why our copy drifted, and that the copy may state:

- **Each page is both a node and a class.** A narrativegoldmine page carries a JSON-LD block
  that *declares an OWL class*; the links between pages populate the graph. So the corpus is
  simultaneously ontology source (layer 2) and knowledge-graph content (layer 3), and it is
  TBox-heavy — most of what the graph knows is class-level, with individuals in the minority.
  When a sentence needs one word for the whole artefact, that word is **knowledge graph** (or
  **corpus** for the source form). "Ontology" is reserved for the schema the pipeline compiles out.
- **Reasoning is a named, running component, not an aspiration.** The industry stack diagram
  leaves inference implicit inside "logical rules"; ours runs, gates writes, and has a name.
  Say the name.

## One-line glosses (use verbatim at first mention on a surface)

- **ontology** — "the formal vocabulary: the classes, typed properties and rules that define what can be said"
- **knowledge graph** — "that vocabulary populated at scale: the pages and typed links you can traverse, query and cite"
- **reasoning** — "the machine check: a Whelk EL++ reasoner classifies the graph and rejects contradictions before they enter it"
- **Ontology Loom** — "the grounding layer: it retrieves the relevant slice of the graph into an LLM's context at query time, so answers restate checked facts rather than guesses"
- **taxonomy** — "the is-a backbone of the ontology — here a lattice, not a tree"

## Rules

1. **"The ontology" means the schema.** If you can count pages or words in it, it is the
   corpus or the knowledge graph, not the ontology. `8,142 OWL classes` belongs to the
   ontology; `8,138 public pages` and `6.8 million words` belong to the corpus.
2. **"Living ontology" is retired.** For the whole artefact say "a knowledge graph with a
   formal ontology compiled from it", or lead with corpus and follow with graph.
3. **The `/ontology` explorer may keep its name** — it genuinely visualises the class-and-
   property structure. Call it "the 3D ontology explorer"; do not call the *page* graph "the
   3D ontology".
4. **"Reasoned" is earned, not decorative.** Unqualified "reasoning" on these surfaces is
   symbolic (Whelk). LLM inference is always qualified: "LLM reasoning", "chain-of-thought".
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
8. **Numbers come from the pipeline.** Stats quoted on any surface match the published
   `stats.json` / eval reports, with their date. No hand-typed figures drifting apart across
   surfaces.

## The canonical sentence

When one sentence has to carry the whole stack:

> Researchers and agents write a corpus; a pipeline compiles it into a knowledge graph under
> a formal OWL 2 ontology; a Whelk reasoner checks every statement before it enters; and the
> Ontology Loom serves the checked graph back into any model's context at query time.

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
