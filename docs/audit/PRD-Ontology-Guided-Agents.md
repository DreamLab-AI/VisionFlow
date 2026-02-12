# PRD: Ontology-Guided Agent Intelligence

**Sprint**: Current
**Status**: Implementation-Ready
**Date**: 2026-02-11

---

## Problem

Agents spawned in the Control Center operate on tasks without awareness of the
living ontology corpus that defines the knowledge domain. The corpus is a set of
Logseq markdown notes with OWL ontology headers (term-id, owl:class,
is-subclass-of, has-part, enables, requires, bridges-to…). These notes ARE the
graph nodes. They are synced from GitHub, parsed by `OntologyParser`, persisted
in Neo4j as `OwlClass` nodes, and reasoned over by the Whelk EL++ engine.

Today agents cannot:
- **Discover** relevant notes via ontological semantics (class hierarchy,
  domain relationships, transitive closure)
- **Read** note content enriched with Whelk-inferred axioms
- **Write back** improvements, new notes, or corrections
- **Submit** changes for human review via GitHub pull requests

The ontology is a living corpus. It must grow through both human curation and
agent contribution, with every agent-originated change reviewed by humans before
entering the canonical graph.

## Evidence

| Technique | Source | Metric |
|-----------|--------|--------|
| OG-RAG hypergraph retrieval | EMNLP 2025, Microsoft | +55% fact recall, +40% correctness |
| Ontology-based query validation (OBQC) | ISWC 2024, data.world | 4.2x accuracy vs SQL baseline |
| Ontology semantic tagging | Sourcely 2025 | +40% review speed, +30% citation quality |
| Schema-guided multi-agent extraction | KARMA, NeurIPS 2025 | Production-scale KG enrichment |
| Agentic KG management | AGENTiGraph, CIKM 2025 | 95.12% classification, 90.45% execution |
| Neuro-symbolic HITL workflows | EmergentMind synthesis | 2 weeks → 24 min for KG creation |

## Users

1. **Agents** (researcher, coder, analyst, optimizer, coordinator) — spawned
   via Control Center, managed by voice, need ontology-aware context for tasks
2. **Human curators** — review agent-proposed changes via GitHub PRs, approve
   or reject ontology modifications
3. **System operators** — monitor ontology health, consistency, and agent
   contribution quality in the 3D visualization

## Requirements

### P0: Agent Reads Ontology (Discovery + Context)

**R1. Semantic Discovery**: Agents discover relevant notes by querying the
ontology graph using OWL class hierarchies and Whelk-inferred subsumptions.
An agent researching "transformer architectures" finds notes classified under
`ai:TransformerArchitecture` and all its superclasses/related concepts via
transitive closure.

**R2. Ontology-Enriched Context**: When an agent reads a note, it receives:
- The raw Logseq markdown content
- The parsed OntologyBlock metadata (term-id, domain, quality-score, maturity)
- Whelk-inferred axioms (parent classes, equivalent classes, disjoint classes)
- Related notes via `has-part`, `requires`, `enables`, `bridges-to` relationships
- The `SchemaService.to_llm_context()` schema summary for query grounding

**R3. Ontology-Based Query Validation**: Agent-generated Cypher queries against
Neo4j are validated against the OWL schema before execution:
- Node labels must match registered `OwlClass` IRIs
- Relationship types must match registered `OwlProperty` IRIs
- Domain/range constraints checked via `OntologyRepository`
- Invalid queries get error explanations and auto-repair (up to 3 iterations)

### P0: Agent Writes Back to Ontology

**R4. Note Creation**: Agents create new Logseq markdown notes with valid
OntologyBlock headers. The `OntologyParser` validates the structure; the note
enters Neo4j with `status: "agent_proposed"`.

**R5. Note Amendment**: Agents propose changes to existing notes (add
relationships, update definitions, correct metadata). Changes are tracked as
diffs against the current canonical content.

**R6. Whelk Consistency Check**: Before any agent-proposed change is accepted
into the staging area, Whelk runs a consistency check. If the proposed axioms
introduce an inconsistency (a non-Bottom class subsumes `owl:Nothing`), the
proposal is rejected with an explanation.

### P1: GitHub PR Feedback Loop

**R7. PR Creation**: Agent-proposed changes (new notes, amendments) are
serialized as Logseq markdown, committed to a feature branch, and submitted
as a GitHub pull request with:
- PR title: `[ontology] {agent_type}: {summary}`
- PR body: agent's reasoning, affected classes, Whelk consistency report
- Labels: `ontology`, `agent-proposed`, `{agent_type}`

**R8. Human Review**: PRs require human approval before merge. Reviewers see:
- Diff of OntologyBlock changes
- Whelk inference impact (new subsumptions added/removed)
- Quality score and authority score of affected concepts
- The agent's task context (why this change was proposed)

**R9. Merge → Sync Loop**: On PR merge, the GitHub sync pipeline
(`GitHubSyncService`) detects the changed files, re-parses them, updates
Neo4j, and triggers the Whelk reasoning pipeline. The ontology evolves.

### P1: Solid Pod Integration

**R10. Proposal Storage**: Agent proposals are stored in the user's Solid pod
at `ontology_proposals/` with NIP-98 authentication. This provides:
- Per-user proposal history
- Decentralized ownership of contributions
- ACL-controlled access (agents write, humans review)

### P2: Visualization

**R11. Agent-Ontology Interaction Rendering**: In the 3D graph, when an agent
reads a note, the corresponding node glows. When an agent proposes a change,
the node pulses with a "pending" color until the PR is merged.

**R12. Consistency Dashboard**: A panel showing Whelk consistency status,
pending proposals, recent merges, and ontology growth metrics.

## Success Metrics

| Metric | Target |
|--------|--------|
| Agent task accuracy with ontology context | +40% vs without (matches OBQC research) |
| Agent-proposed notes accepted by humans | >70% acceptance rate |
| Whelk consistency maintained | 100% (no inconsistent states reach prod) |
| Time from agent proposal to human review | <24 hours |
| Ontology growth rate | >10 new notes/week from agents |

## Out of Scope

- Automated merge without human review (always HITL)
- Multi-language ontology support
- Real-time collaborative ontology editing between multiple agents
