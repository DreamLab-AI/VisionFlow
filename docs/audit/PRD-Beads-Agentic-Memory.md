# PRD: Beads-Inspired Agentic Memory for VisionFlow

**Status**: EVALUATION
**Author**: Claude
**Date**: 2026-02-12
**Source**: [steveyegge/beads](https://github.com/steveyegge/beads) — distributed git-backed graph issue tracker for AI agents

---

## 1. Context

The user requested evaluation of the Beads project (by Steve Yegge) for adaptation into VisionFlow's node-and-edge agentic memory system, with the constraint that **RuVector PostgreSQL must remain the backend** (not Dolt/SQLite as in Beads). This PRD analyses what Beads offers, what VisionFlow already has, and whether the delta justifies implementation effort.

---

## 2. Beads Core Concepts

Beads is a **dependency-aware graph issue tracker** for coding agents. Its key innovations:

### 2.1 Semantic Compaction (Memory Decay)

Three-tier summarisation of stale work items:

| Tier | Trigger | Action |
|------|---------|--------|
| Tier 1 | Closed 30+ days, no open dependents | AI summarises description+design+notes+acceptance |
| Tier 2 | Closed 90+ days, 100+ events, already tier 1 | Aggressive AI summarisation |
| TTL (Wisps) | Type-dependent (heartbeat=6h, patrol=24h, error=7d) | Auto-delete ephemeral items |

Safety: only compacts if summary is smaller. Pinned items are immune. Tracks `original_size` and `compacted_at_commit` for provenance.

### 2.2 Dependency-Aware Ready Work

The `bd ready` command returns tasks with **no open blocking dependents**:

- **Blocking types**: `blocks`, `parent-child`, `conditional-blocks`, `waits-for`
- **Non-blocking types**: `related`, `discovered-from`, `replies-to`, `duplicates`, `supersedes`
- **Sort policies**: hybrid (recent by priority, old by age), priority-first, oldest-first
- **Time scheduling**: `defer_until`, `due_at`, overdue detection

### 2.3 Graph Edge Taxonomy

17 well-known dependency types across 5 categories:

| Category | Types |
|----------|-------|
| Workflow | blocks, parent-child, conditional-blocks, waits-for |
| Association | related, discovered-from |
| Graph Links | replies-to, relates-to, duplicates, supersedes |
| Entity | authored-by, assigned-to, approved-by, attests |
| Cross-Reference | tracks, until, caused-by, validates |

All edges stored in a unified `dependencies` table with type field and JSON metadata blob.

### 2.4 Agent State Tracking

- `AgentState` enum: idle, spawning, running, working, stuck, done, stopped, dead
- `HookBead`: current work item on agent's hook (0..1)
- `RoleBead`: role definition reference
- `LastActivity`: timeout detection

### 2.5 Hash-Based IDs & Content Hashing

- Random UUID → short hash (`bd-a1b2`) prevents merge collisions
- SHA256 content hash enables idempotent merging (same ID + same hash = skip)

### 2.6 Federation

- Peer-to-peer sync via Dolt remotes
- `SourceSystem` + `SourceRepo` fields for provenance
- Data sovereignty tiers (T1-T4)

### 2.7 HOP Entity Tracking

- Creator EntityRef (name, platform, org, id)
- Validation chains (validator, outcome, timestamp, score)
- QualityScore (aggregate 0.0-1.0)
- Crystallizes flag (work that compounds vs evaporates)

### 2.8 Messaging & Threading

- `replies-to` edges form conversation threads
- ThreadID for efficient thread queries
- Ephemeral flag excludes from JSONL export

---

## 3. VisionFlow's Existing Memory Stack

VisionFlow already has a **5-tier memory architecture**:

```
┌──────────────────────────────────────────────────────────┐
│ TIER 1: SESSION (Ephemeral)                              │
│ ├─ MCP session tools (status, agents, metrics)           │
│ ├─ Process manager (agent lifecycle)                     │
│ └─ Performance metrics collection                        │
├──────────────────────────────────────────────────────────┤
│ TIER 2: AGENT RUNTIME (Medium-term)                      │
│ ├─ Unified Memory (SONA + ReasoningBank)                │
│ │   ├─ 5 modes: real-time/balanced/research/edge/batch  │
│ │   ├─ Confidence tiers: bronze→silver→gold→platinum     │
│ │   └─ Dream cycle consolidation (30s intervals)        │
│ ├─ @claude-flow/memory SDK (per-agent isolation)        │
│ ├─ Memory Graph (relationship tracking)                 │
│ └─ HNSW indexing (150x faster vector search)            │
├──────────────────────────────────────────────────────────┤
│ TIER 3: PERSISTENT (Long-term)                           │
│ ├─ RuVector PostgreSQL (19,659+ entries, MCP access)    │
│ │   ├─ HNSW: 384 dims, ef_construction=200, M=16       │
│ │   └─ Namespaces: qe, patterns, coverage, tests       │
│ ├─ Neo4j 5.13 (ontology + knowledge graph)              │
│ └─ Qdrant (semantic vector search)                      │
├──────────────────────────────────────────────────────────┤
│ TIER 4: KNOWLEDGE (Ontology)                             │
│ ├─ OntologyQueryService (discovery + reasoning)         │
│ ├─ OntologyMutationService (write + validation)         │
│ ├─ Whelk EL++ consistency engine                        │
│ └─ GPU semantic forces (ontology → physics)             │
├──────────────────────────────────────────────────────────┤
│ TIER 5: SOURCE OF TRUTH                                  │
│ ├─ GitHub markdown (canonical, human-readable)          │
│ ├─ Git history (full audit trail)                       │
│ └─ PR-based change proposals (agent → human review)     │
└──────────────────────────────────────────────────────────┘
```

### 3.1 Cross-Project Learning (Already Active)

- `sona-optimizer` agent: self-optimising neural architecture
- `reasoning-bank-manager`: adaptive learning management
- `cross-project-transfer` agent: pattern transfer across projects
- Post-deployment phases: `learning_consolidation`, `cross_project_transfer`

### 3.2 Agent Tracking (Already Active)

- `agent-tracker.js`: agent lifecycle tracking in MCP server
- `process-manager.js`: session lifecycle management
- `system-monitor.js`: resource monitoring
- Management API at port 9090 with health checks

---

## 4. Gap Analysis: Beads vs VisionFlow

| Beads Concept | VisionFlow Equivalent | Gap? | Notes |
|---------------|----------------------|------|-------|
| **Semantic compaction** | SONA dream cycle (30s consolidation) | **Partial** | SONA consolidates patterns, but doesn't do AI summarisation of stale entries with provenance tracking |
| **Ephemeral TTL cleanup** | Session-scoped memory (auto-cleared) | **Partial** | No explicit WispType + TTL classification |
| **Ready-work logic** | No equivalent | **Yes** | Agents don't have dependency-aware task scheduling |
| **Graph edge taxonomy** | OWL relationships (SubClassOf, DisjointWith, etc.) | **No** | VisionFlow's ontology edges are richer than Beads' 17 types |
| **Agent state enum** | agent-tracker.js + process-manager.js | **No** | Already tracked via MCP tools |
| **Hash-based IDs** | Neo4j UUIDs + OWL IRIs | **No** | Different ID strategy but same collision-free goal |
| **Content hashing** | SHA1 in GitHub sync (file-level) | **Partial** | File-level not entry-level |
| **Federation** | RuVector shared across Docker network | **No** | Already federated via shared PostgreSQL + MCP |
| **HOP entity tracking** | Git commit attribution | **Partial** | No formal validation chains or quality score aggregation |
| **Messaging/threading** | No explicit threading model | **Partial** | Agent communication is via MCP tools, not threaded messages |
| **Pinned context** | No equivalent | **Yes** | No way to mark memory entries as immune to cleanup |
| **Time scheduling** | No equivalent | **Yes** | No defer_until / due_at for agent tasks |
| **Crystallization flag** | No equivalent | **Yes** | No tracking of work that compounds vs evaporates |
| **Sort policies** | No equivalent | **Yes** | No hybrid/priority/oldest sorting for agent task queues |

---

## 5. Evaluation: Does This Add Sufficient Value?

### 5.1 What Beads Solves That VisionFlow Doesn't

**Novel concepts with genuine value:**

1. **Semantic compaction with provenance** — SONA consolidates patterns, but doesn't do explainable AI summarisation of individual memory entries with git-commit provenance. Useful for long-running agents that accumulate thousands of entries in RuVector.

2. **Ready-work scheduling** — Dependency-aware "what should I work on next?" is genuinely missing. Agents currently pick tasks without understanding blocker chains.

3. **Pinned context** — Protecting critical memory entries from any cleanup/consolidation is a simple but valuable primitive.

4. **Time-based deferral** — `defer_until` prevents agents from repeatedly re-discovering deferred work.

### 5.2 What Beads Solves That VisionFlow Already Handles Better

1. **Graph storage** — Beads uses Dolt/SQLite with a flat dependencies table. VisionFlow has Neo4j with OWL ontology reasoning, Whelk inference, and GPU semantic forces. Neo4j's graph model is orders of magnitude richer.

2. **Federation** — Beads uses Dolt remotes. VisionFlow has shared RuVector PostgreSQL accessible by all agents over Docker network. Already federated.

3. **Agent tracking** — Beads has AgentState enum. VisionFlow has agent-tracker.js, process-manager.js, system-monitor.js, MCP session tools. Already more comprehensive.

4. **Cross-project memory** — Beads has SourceSystem + ExternalRef. VisionFlow has `cross-project-transfer` agent, SONA namespaces, shared RuVector with project-scoped namespaces.

5. **Audit trail** — Beads has an events table. VisionFlow has Git history, GitHub PR reviews, Whelk consistency checking. Already more thorough.

6. **Content deduplication** — Beads uses SHA256 content hash. VisionFlow's GitHub sync already uses SHA1 for differential processing. SONA uses pattern matching for deduplication.

### 5.3 Effort Estimation

| Feature | Effort | Value | Verdict |
|---------|--------|-------|---------|
| Semantic compaction in RuVector | Medium (2-3 days) | Medium — useful for long-horizon agents | **Maybe** |
| Ready-work scheduling | Medium (2-3 days) | Low — agents use MCP discovery, not task queues | **No** |
| Pinned context flag | Trivial (2 hours) | Medium — simple protective primitive | **Yes** |
| Time-based deferral | Low (1 day) | Low — agents don't schedule deferred work | **No** |
| Crystallization tracking | Low (1 day) | Low — academic interest only | **No** |
| Sort policies | Low (1 day) | Low — not how VisionFlow agents pick tasks | **No** |
| WispType + TTL cleanup | Low (1 day) | Medium — RuVector could use TTL cleanup | **Maybe** |

### 5.4 The Core Question

> *Does Beads add sufficient new features for the effort, given that we already do most of this around GitHub?*

**Verdict: NO — Beads does not warrant a full implementation cycle.**

**Rationale:**

1. **Different problem domain.** Beads is a task tracker for coding agents. VisionFlow is a knowledge management platform with ontology reasoning. The impedance mismatch is fundamental. Beads' ready-work logic, dependency chains, and hierarchical epics assume a coding workflow (create issue → implement → close). VisionFlow's agents discover knowledge, reason about ontology, and propose graph mutations — a fundamentally different interaction pattern.

2. **VisionFlow's memory is already more sophisticated.** The 5-tier architecture (session → runtime → persistent → ontology → source-of-truth) with SONA consolidation, HNSW indexing, confidence tiers, and Whelk reasoning exceeds Beads' flat Dolt/SQLite + JSONL model in every dimension except explicit compaction provenance.

3. **RuVector vs Dolt is not a like-for-like swap.** Beads' design centres on Dolt's version-controlled SQL with cell-level merge. Adapting this to RuVector PostgreSQL would lose Dolt's core differentiator (branch/merge/history) while adding complexity to a system that already has Git for versioning.

4. **GitHub already provides the audit trail.** Beads' events table and content hashing exist because it needs its own audit trail separate from Git. VisionFlow already uses Git commits and PR reviews as the canonical audit trail, making Beads' approach redundant.

5. **The 80/20 is two small features.** The only genuinely useful Beads concepts for VisionFlow are (a) pinned context flags on RuVector entries and (b) TTL-based cleanup of ephemeral entries. Both are trivial to implement without importing Beads' entire architecture.

---

## 6. Recommendation

**Do NOT implement a Beads-inspired rewrite of VisionFlow's memory system.**

Instead, adopt two micro-features directly into RuVector:

### 6.1 Pinned Context (2 hours)

Add a `pinned` boolean to RuVector memory entries. Pinned entries are protected from SONA dream-cycle consolidation and any future compaction.

**Implementation**: Add column to RuVector schema, filter in consolidation queries.

### 6.2 TTL-Based Ephemeral Cleanup (4 hours)

Add optional `ttl_expires_at` timestamp to RuVector entries. A background task (cron or Actix scheduler) deletes expired entries.

**Categories**: agent heartbeats (6h), patrol reports (24h), error logs (7d), session artifacts (48h).

**Implementation**: Add column + background cleanup task in management API.

### 6.3 What NOT to Adopt

- **Ready-work scheduling**: VisionFlow agents use ontology discovery, not task queues
- **Hierarchical epic IDs**: VisionFlow uses OWL IRIs and Neo4j UUIDs
- **Dolt-style version control**: Git + GitHub already provides this
- **JSONL export**: RuVector + MCP tools are the interface, not file-based sync
- **HOP entity tracking**: Git attribution + PR reviews already cover this
- **Federation protocol**: Shared RuVector over Docker network is simpler
- **Crystallization tracking**: Academic interest, no practical agent workflow benefit

---

## 7. Decision

**STOP HERE.** The evaluation concludes that Beads does not add sufficient value to warrant AFD/DDD/implementation. The two micro-features (pinned context + TTL cleanup) can be implemented as a small enhancement to the existing RuVector configuration without architectural changes.

If the user agrees, these can be implemented in ~6 hours. If the user disagrees and wants the full Beads adaptation, an AFD and DDD would be required.
