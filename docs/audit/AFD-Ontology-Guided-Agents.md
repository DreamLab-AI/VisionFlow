# AFD: Ontology-Guided Agent Intelligence

**Status**: Implementation
**Date**: 2026-02-11
**Depends on**: PRD-Ontology-Guided-Agents.md

---

## Architecture Decision: Ontology as Agent Operating System

The ontology layer becomes the agent's "operating system" — every agent reads
from it, reasons through it, and proposes changes back to it. The Logseq
markdown notes ARE the knowledge atoms. Whelk EL++ reasoning provides the
inference backbone. GitHub PRs provide human oversight.

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CONTROL CENTER                                │
│  User spawns agents → Voice commands → Agent lifecycle management    │
└──────────────┬───────────────────────────────────────────────────────┘
               │ spawn / command
               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     AGENT LAYER (MCP)                                 │
│                                                                      │
│  ┌─────────────┐ ┌──────────┐ ┌─────────┐ ┌───────────┐            │
│  │  Researcher  │ │  Coder   │ │ Analyst │ │ Optimizer │  ...       │
│  └──────┬──────┘ └────┬─────┘ └────┬────┘ └─────┬─────┘            │
│         │              │            │             │                   │
│         └──────────────┴────────────┴─────────────┘                  │
│                        │                                             │
│              ┌─────────▼─────────┐                                   │
│              │  ONTOLOGY TOOLS   │  ← New MCP tool surface           │
│              │  (agent-callable) │                                    │
│              └─────────┬─────────┘                                   │
└────────────────────────┼─────────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
          ▼              ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
│ READ PATH    │ │ WRITE PATH   │ │ VALIDATE PATH    │
│              │ │              │ │                   │
│ discover()   │ │ propose()    │ │ validate_cypher() │
│ read_note()  │ │ amend()      │ │ check_consist()   │
│ query_kg()   │ │ create_pr()  │ │ explain()         │
│ traverse()   │ │              │ │                   │
└──────┬───────┘ └──────┬───────┘ └────────┬──────────┘
       │                │                   │
       ▼                ▼                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    ONTOLOGY SERVICES LAYER                           │
│                                                                      │
│  ┌────────────────────┐  ┌─────────────────────┐                    │
│  │ OntologyQuerySvc   │  │ OntologyMutationSvc │                    │
│  │ (semantic discovery │  │ (proposals, staging, │                    │
│  │  + context assembly)│  │  markdown generation)│                    │
│  └─────────┬──────────┘  └──────────┬──────────┘                    │
│            │                        │                                │
│  ┌─────────▼────────────────────────▼──────────┐                    │
│  │            WHELK EL++ REASONER               │                    │
│  │  infer() → subsumptions, consistency check   │                    │
│  │  is_entailed() → axiom verification          │                    │
│  │  classify_instance() → type assignment        │                    │
│  │  get_subclass_hierarchy() → transitive closure│                    │
│  └─────────┬────────────────────────┬──────────┘                    │
│            │                        │                                │
│  ┌─────────▼──────────┐  ┌─────────▼──────────┐                    │
│  │ Neo4j              │  │ GitHub PR Service   │                    │
│  │ (OwlClass nodes,   │  │ (branch, commit,    │                    │
│  │  axioms, graph)     │  │  pull request)      │                    │
│  └────────────────────┘  └─────────────────────┘                    │
└──────────────────────────────────────────────────────────────────────┘
               │                        │
               ▼                        ▼
┌──────────────────────┐  ┌──────────────────────────┐
│ Neo4j Database       │  │ GitHub Repository         │
│ OwlClass nodes       │  │ Logseq markdown files     │
│ Whelk inferences     │  │ with OntologyBlock headers │
│ Agent proposals      │  │                            │
└──────────────────────┘  └──────────────────────────┘
                                    │
                              human review
                                    │
                              merge → GitHubSyncService
                                    │
                              re-parse → Neo4j → Whelk
```

## Read Path: Agent Discovers and Reads Notes

### Step 1: Semantic Discovery

Agent calls `ontology_discover(query, scope)`:

```
Agent: "I need information about transformer architectures and their dependencies"
                │
                ▼
OntologyQueryService.discover(query, scope)
    │
    ├─ 1. Parse query → extract concept keywords
    │     ["transformer", "architecture", "dependency"]
    │
    ├─ 2. Map keywords to OWL classes via Neo4j
    │     MATCH (c:OwlClass)
    │     WHERE c.preferred_term CONTAINS $keyword
    │        OR c.label CONTAINS $keyword
    │     RETURN c.iri, c.preferred_term, c.quality_score
    │     → ai:TransformerArchitecture
    │
    ├─ 3. Expand via Whelk transitive closure
    │     whelk.get_subclass_hierarchy()
    │     → ai:TransformerArchitecture
    │       ├─ ai:AttentionMechanism
    │       ├─ ai:MultiHeadAttention
    │       ├─ ai:PositionalEncoding
    │       └─ ai:LayerNormalization
    │
    ├─ 4. Follow semantic relationships
    │     MATCH (c:OwlClass {iri: $class_iri})-[r]->(related)
    │     WHERE type(r) IN ['HAS_PART','REQUIRES','ENABLES','BRIDGES_TO']
    │     RETURN related.iri, type(r), related.preferred_term
    │     → requires: ai:MatrixMultiplication
    │     → enables: ai:LargeLanguageModel
    │     → bridges_to: tc:ParallelComputation
    │
    └─ 5. Return ranked discovery results
          [{iri, preferred_term, relevance_score, relationships, quality_score}]
```

### Step 2: Read Note with Enriched Context

Agent calls `ontology_read(iri)`:

```
OntologyQueryService.read_note(iri)
    │
    ├─ 1. Fetch OwlClass from Neo4j
    │     MATCH (c:OwlClass {iri: $iri})
    │     RETURN c   → full node with all properties
    │
    ├─ 2. Fetch raw markdown content
    │     c.markdown_content → the original Logseq note
    │
    ├─ 3. Fetch Whelk-inferred axioms
    │     whelk.get_class_axioms(iri)
    │     → SubClassOf(ai:Transformer, ai:NeuralNetwork)
    │     → SubClassOf(ai:Transformer, ai:SequenceModel)  [inferred]
    │
    ├─ 4. Fetch related notes via relationships
    │     MATCH (c {iri: $iri})-[r]-(related:OwlClass)
    │     RETURN related.iri, related.preferred_term,
    │            related.markdown_content, type(r)
    │
    ├─ 5. Assemble enriched context
    │     {
    │       note: { iri, term_id, preferred_term, markdown_content },
    │       ontology: { owl_class, physicality, role, domain },
    │       quality: { quality_score, authority_score, maturity, status },
    │       inferred: [ axioms from Whelk ],
    │       related: [ { iri, term, relationship_type, summary } ],
    │       schema_context: SchemaService.to_llm_context(),
    │     }
    │
    └─ 6. Return to agent as structured context
```

### Step 3: Ontology-Validated Queries

Agent calls `ontology_query(cypher)`:

```
OntologyQueryService.validate_and_execute(cypher)
    │
    ├─ 1. Parse Cypher AST (extract labels, rel types, properties)
    │
    ├─ 2. Validate against OWL schema:
    │     ├─ All node labels exist as OwlClass IRIs?
    │     ├─ All relationship types exist as OwlProperty IRIs?
    │     ├─ Domain/range constraints hold?
    │     └─ Properties exist on the declared classes?
    │
    ├─ 3. If invalid → generate error explanation + repair hint
    │     {valid: false, errors: [...], hints: ["Did you mean 'ai:NeuralNetwork'?"]}
    │     Agent auto-repairs (up to 3 iterations)
    │
    └─ 4. If valid → execute against Neo4j → return results
```

## Write Path: Agent Proposes Changes

### Step 1: Agent Creates or Amends a Note

```
Agent calls ontology_propose(action, content):
    │
    ├─ action = "create_note"
    │   Agent provides:
    │   - preferred_term: "Vision Transformer"
    │   - definition: "A transformer architecture adapted for image recognition..."
    │   - owl_class: "ai:VisionTransformer"
    │   - is_subclass_of: ["ai:TransformerArchitecture", "ai:ComputerVision"]
    │   - relationships: {has_part: ["ai:PatchEmbedding"], enables: ["ai:ImageClassification"]}
    │   - domain: "ai"
    │
    ├─ OR action = "amend_note"
    │   Agent provides:
    │   - iri: "ai:TransformerArchitecture"
    │   - changes: {add_relationship: {enables: "ai:ProteinFolding"}}
    │
    │   OntologyMutationService:
    │
    ├─ 1. Generate valid Logseq markdown from proposal
    │     Uses OntologyParser in REVERSE:
    │     - Build OntologyBlock with all tiers
    │     - Validate tier 1 required fields
    │     - Generate term-id from domain prefix + sequence
    │
    ├─ 2. Whelk consistency check
    │     Load proposed axioms into Whelk
    │     whelk.check_consistency()
    │     If inconsistent → REJECT with explanation:
    │       "Adding SubClassOf(ai:VisionTransformer, ai:TextModel) introduces
    │        inconsistency: ai:VisionTransformer infers owl:Nothing because
    │        ai:ImageModel and ai:TextModel are disjoint"
    │
    ├─ 3. Score the proposal
    │     quality_score = f(completeness, consistency, novelty)
    │     agent_confidence = from agent's task context
    │
    ├─ 4. Stage in Neo4j with status: "agent_proposed"
    │     CREATE (p:OntologyProposal {
    │       proposal_id, agent_id, agent_type, timestamp,
    │       action, target_iri, markdown_content,
    │       consistency_check: "passed",
    │       quality_score, agent_confidence,
    │       status: "staged"
    │     })
    │
    └─ 5. Submit to GitHub PR pipeline
```

### Step 2: GitHub PR Creation

```
GitHubPRService.create_ontology_pr(proposal):
    │
    ├─ 1. Create feature branch
    │     git checkout -b ontology/agent-{agent_type}-{proposal_id}
    │
    ├─ 2. Write markdown file
    │     data/markdown/{domain}/{term-id}.md
    │     Content: generated Logseq markdown with OntologyBlock
    │
    ├─ 3. Commit with metadata
    │     message: "[ontology] {agent_type}: Add {preferred_term}"
    │     co-authored-by: agent-{agent_id}@visionflow
    │
    ├─ 4. Push branch
    │
    ├─ 5. Create PR via GitHub API
    │     title: "[ontology] {agent_type}: {summary}"
    │     body:
    │       ## Proposed Change
    │       {agent's reasoning for the change}
    │
    │       ## Affected Classes
    │       - {iri}: {description of change}
    │
    │       ## Whelk Consistency Report
    │       ✅ Consistency check passed
    │       New subsumptions: {list}
    │       No disjoint violations
    │
    │       ## Quality Assessment
    │       - Quality Score: {score}
    │       - Agent Confidence: {confidence}
    │       - Completeness: {tier coverage}
    │
    │       ## Agent Context
    │       - Agent: {agent_type} ({agent_id})
    │       - Task: {original task description}
    │       - Session: {voice session if applicable}
    │
    │     labels: ["ontology", "agent-proposed", "{agent_type}"]
    │
    └─ 6. Update proposal status in Neo4j
          SET p.status = "pr_created", p.pr_url = {url}
```

### Step 3: Merge → Sync Loop

```
Human merges PR on GitHub
    │
    ▼
GitHubSyncService.sync_graphs() (existing, runs on interval)
    │
    ├─ filter_changed_files() → detects merged markdown
    │
    ├─ OntologyParser.parse_enhanced() → parses new/changed note
    │
    ├─ Neo4jOntologyRepository.add_owl_class() → persists
    │
    ├─ OntologyPipelineService.on_ontology_modified()
    │   │
    │   ├─ WhelkInferenceEngine.infer() → updated subsumptions
    │   │
    │   ├─ generate_constraints_from_axioms() → GPU physics
    │   │
    │   └─ SemanticTypeRegistry.build_dynamic_gpu_buffer()
    │
    └─ Update proposal: SET p.status = "merged"

The ontology has evolved. All agents now see the new knowledge.
```

## Agent Tool Surface (MCP Tools)

New tools exposed to agents via MCP:

| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `ontology_discover` | query: string, limit: int, domain?: string | [{iri, term, score, rels}] | Semantic discovery via class hierarchy + Whelk |
| `ontology_read` | iri: string | {note, ontology, quality, inferred, related} | Read note with full ontology context |
| `ontology_query` | cypher: string | query results or validation errors | Validated Cypher execution against KG |
| `ontology_traverse` | start_iri: string, depth: int, rel_types?: string[] | subgraph | Walk the ontology graph from a starting concept |
| `ontology_propose` | action: create\|amend, content: {...} | {proposal_id, consistency, quality} | Propose new note or amendment |
| `ontology_schema` | domain?: string | schema context string | Get LLM-friendly schema summary |
| `ontology_validate` | axioms: [...] | {consistent: bool, explanation?: string} | Check if axioms are Whelk-consistent |

## Key Design Decisions

### D1: Whelk as Gatekeeper

Every agent proposal passes through Whelk consistency checking before staging.
This prevents agents from introducing logical contradictions into the ontology.
The EL++ fragment is decidable and tractable — consistency checks run in
polynomial time even on large ontologies.

### D2: GitHub as Single Source of Truth

The Logseq markdown files on GitHub are canonical. Neo4j is a derived cache
(populated by GitHubSyncService). This means:
- Agents never write directly to Neo4j's OwlClass nodes
- All mutations go through the PR → merge → sync pipeline
- Git history provides full audit trail of ontology evolution
- Reverting a bad change = reverting a Git commit

### D3: Staged Proposals in Neo4j

Agent proposals are stored in Neo4j as `OntologyProposal` nodes (separate from
`OwlClass`) so agents can see pending proposals and avoid duplicates. Proposals
are linked to the affected `OwlClass` nodes via relationships.

### D4: Opus/Sonnet as Ontology-Aware Context Window

When an agent receives ontology context from `ontology_read`, the enriched
payload is structured for optimal LLM comprehension:
- Raw markdown for content understanding
- Parsed metadata for structured reasoning
- Whelk inferences for logical grounding
- Related notes for contextual breadth
- Schema summary for query formulation
