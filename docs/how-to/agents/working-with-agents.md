---
title: Working with AI Agents
description: Guide to interacting with VisionFlow AI agents via the 7 MCP ontology tools, knowledge graph interaction patterns, and per-user note ownership.
category: how-to
tags:
  - agents
  - mcp
  - ontology
  - tools
updated-date: 2026-02-12
difficulty-level: intermediate
---

# Working with AI Agents

## Overview

VisionFlow agents operate through the Model Context Protocol (MCP). Each agent
can discover, read, query, traverse, propose, validate, and check the status of
the ontology knowledge graph. This guide explains how to work alongside these
agents, understand their capabilities, and leverage per-user ownership to keep
your notes separate from other users' contributions.

## The 7 MCP Ontology Tools

Every agent has access to the following tool set. Tools are invoked as MCP tool
calls or equivalent REST requests.

| # | Tool                 | Purpose                                         | Mutates Data |
|---|----------------------|-------------------------------------------------|:------------:|
| 1 | `ontology_discover`  | Keyword search with Whelk inference expansion   | No           |
| 2 | `ontology_read`      | Fetch an enriched note by IRI                   | No           |
| 3 | `ontology_query`     | Execute validated Cypher against Neo4j          | No           |
| 4 | `ontology_traverse`  | BFS walk from a starting class                  | No           |
| 5 | `ontology_propose`   | Create or amend an ontology note                | Yes          |
| 6 | `ontology_validate`  | Check axiom consistency via Whelk               | No           |
| 7 | `ontology_status`    | Health check and statistics                     | No           |

### Typical Agent Workflow

1. **Discover** relevant classes with `ontology_discover`.
2. **Read** promising hits via `ontology_read` to inspect definitions and axioms.
3. **Traverse** the graph around the target concept to understand context.
4. **Query** with Cypher for precise relationship patterns.
5. **Validate** any new axioms before committing.
6. **Propose** a new note or amendment, triggering Whelk consistency checks.
7. **Status** can be polled at any time to confirm service health.

## How Agents Interact with the Knowledge Graph

The knowledge graph is composed of Logseq markdown notes annotated with
`OntologyBlock` headers. Each note maps to an OWL 2 EL++ class managed by the
Whelk inference engine. Agents never edit files directly; instead they issue
`ontology_propose` calls that:

1. Generate Logseq-compatible markdown.
2. Round-trip through the OntologyParser for validation.
3. Run a Whelk consistency check (SubClassOf cycles, disjointness violations).
4. Stage the result in the OntologyRepository.
5. Optionally open a GitHub pull request when `GITHUB_TOKEN` is configured.

```text
Agent ──MCP──▶ ontology_propose
                  │
                  ├─ OntologyParser round-trip
                  ├─ Whelk EL++ consistency
                  ├─ Quality score (0.0 – 1.0)
                  └─ Stage in repo / open PR
```

Proposals with a quality score above `auto_merge_threshold` (default 0.95)
can be merged automatically; lower-scoring proposals await human review.

## Per-User Note Ownership

Every note carries an `owner_user_id`. Agents inherit their user's identity
through the `AgentContext.user_id` field:

```json
{
  "agent_context": {
    "agent_id": "researcher-001",
    "agent_type": "researcher",
    "user_id": "user-456",
    "confidence": 0.9
  }
}
```

Ownership rules:

- An agent can **create** notes under its user's namespace only.
- An agent can **amend** only notes where `owner_user_id` matches its
  `user_id`.
- **Read** and **discover** operations are not restricted by ownership;
  the full graph is visible to all agents.

This prevents one user's agents from silently rewriting another user's
contributions while still enabling cross-user knowledge discovery.

## Coordinating Multiple Agents

When several agents run concurrently (e.g., a researcher and a reviewer),
coordinate through the knowledge graph itself:

- Use `ontology_discover` to check whether a concept already exists before
  proposing a duplicate.
- Use `ontology_validate` to pre-flight axioms without side-effects.
- Poll `ontology_status` to detect if another agent's proposal changed class
  counts unexpectedly.

For orchestration details (spawning, health checks, task assignment), see the
agent orchestration guide.

## Configuration Reference

| Setting                  | Default | Description                              |
|--------------------------|---------|------------------------------------------|
| `auto_merge_threshold`   | 0.95    | Quality score for automatic merge        |
| `min_confidence`         | 0.6     | Reject proposals below this confidence   |
| `max_discovery_results`  | 50      | Cap on `ontology_discover` output        |
| `require_consistency_check` | true | Enforce Whelk check before staging       |

## See Also

- [Ontology Agent Tools](ontology-agent-tools.md) -- full API reference for all 7 tools
- [Agent Orchestration](agent-orchestration.md) -- deploying and coordinating agents
- [Orchestrating Agents](orchestrating-agents.md) -- multi-agent patterns
- [Using Skills](using-skills.md) -- agent skill definitions
