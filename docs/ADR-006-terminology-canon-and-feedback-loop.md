# ADR-006: Terminology canon and the knowledge-stack feedback loop

**Status**: Accepted
**Date**: 2026-08-15
**Context**: Terminology cleanup across the mesh's public surfaces, run as a live end-to-end
test of the ontology/knowledge-graph/Loom write-and-serve loop.

## Decision 1 — the terminology canon

[`docs/terminology.md`](terminology.md) is the canonical playbook for knowledge-stack
terminology on every public surface in the mesh. The stack it fixes: taxonomy (the is-a
lattice, part of the ontology) → ontology (OWL 2 schema; what can be said) → knowledge graph
(the populated corpus; what is said) → reasoning (Whelk EL++ classification and write
gating; what follows) → grounding (the Ontology Loom assembling an agent's working set — the
layer the industry now calls a context graph). "Semantic layer" is the parallel BI concept
and is not used for any part of this stack. The playbook is mirrored as a published page
(`Terminology Playbook`) in the knowledge graph and stored in RuVector
(`project-state/terminology-playbook-canon`).

## Decision 2 — new classes enter through markdown, not through the bridge

Exercising the write path surfaced the actual state of each route:

| Route | Result |
|-------|--------|
| `ontology_propose` (amend) | **Works end-to-end.** Whelk consistency gate passed, conflict gate passed, proposal staged with cryptographic receipt, ACSP human approval pending. Proposal `7de21296` (knowledge-graph definition + hasPart sense fix) is the live example. |
| `ontology_propose` (create) | **Broken server-side.** The handler reads a `subject` field the MCP schema never defines (`subject 'undefined' not in local corpus`), and `target_iri` is required on create although the schema documents it as amend-only. Until fixed, class creation happens the way the corpus intends anyway: author the page in `mainKnowledgeGraph/pages/` with its JSON-LD blocks and let the pipeline compile it. |
| Raw markdown + pipeline | **The reliable path**, and the one that matches the architecture (pages are the source of truth). Used for `Context Graph`, `Ontology Loom` and `Terminology Playbook`. |

## Findings that need owners

1. **Bridge discover is degraded** (`ontology_search`, `kg_node_search`): uniform 0.55
   relevance scores, empty metadata, unrelated hits. The Loom façade's `/loom/scaffold`
   retrieval on the same corpus is precise. Owner: VisionClaw discover endpoint /
   agentbox ontology-bridge.
2. **`ontology_propose` create-path bug** as above. Owner: agentbox ontology-bridge
   (PRD-020/ADR-112 lineage).
3. **Feedback-loop latency is real and unmanaged.** The Loom's scaffold index was built
   2026-08-11 (8,143 classes); pages added today are invisible to grounding until a pipeline
   rebuild and a Loom reload. Next step: trigger a Loom index reload from the corpus CI
   build, so the serve tier tracks the corpus with bounded staleness.
4. **The pipeline needs an IRI-integrity gate.** A historic string-level rename
   (`tax` → `corporate-tax-compliance-framework`) corrupted the Taxonomy class IRI to
   `urn:ngm:class:corporate-tax-compliance-frameworkonomy` in 10 pages including the class's
   own declaration, and it survived every existing gate. Extend the ADR-NG-001 gate set
   (logseq repo) with a check that every referenced `urn:ngm:class:` IRI resolves to a
   declared class and that no IRI embeds another class's slug as a substring artefact.
5. **Sense collisions exist in the graph.** `Knowledge Graph hasPart Inference Engine`
   points at a class defined as an ML model-serving runtime, not a symbolic reasoner
   (that class exists separately as `reasoning-engine`). The staged amend fixes this
   instance; enrichment QA should check relation targets against the *sense* of the
   definition, not just the label.

## What the live test validated

Asked the terminology question through the Loom's grounded chat path, the deployed model
returned the playbook's own layering — ontology says what you *may* assert, the knowledge
graph records what you *have* asserted, reasoning works out what *follows* — in one
paragraph, 522 completion tokens, grounded on a 1,129-token scaffold. The graph taught the
words back. That is the feedback loop working: corpus → ontology → scaffold → model → prose
that matches the canon.

## Consequences

- Public surfaces are being aligned to the playbook (VisionFlow README + site,
  narrativegoldmine front page + repo READMEs, sibling repo READMEs).
- Repo-local notes: logseq repo gets the IRI-integrity gate proposal; agentbox gets the
  bridge bug report. Cross-references live here, detail lives with the owner.
- The staged knowledge-graph amend awaits human ACSP approval; the three new pages await
  the next pipeline build to enter the served ontology.
