# DDD: Ontology-Guided Agent Intelligence

**Status**: Implementation
**Date**: 2026-02-11
**Implements**: PRD + AFD Ontology-Guided Agents

---

## Module Map

```
src/services/
  ontology_query_service.rs       ← NEW: Agent read path (discover, read, query, traverse)
  ontology_mutation_service.rs    ← NEW: Agent write path (propose, amend, generate markdown)
  github_pr_service.rs            ← NEW: GitHub branch/commit/PR creation

src/types/
  ontology_tools.rs               ← NEW: Tool input/output types for MCP surface

src/config/mod.rs
  OntologyAgentSettings           ← NEW: Config for agent-ontology integration
```

## 1. OntologyQueryService — Agent Read Path

### Struct Definition

```rust
// src/services/ontology_query_service.rs

pub struct OntologyQueryService {
    ontology_repo: Arc<dyn OntologyRepository>,
    graph_repo: Arc<dyn KnowledgeGraphRepository>,
    whelk: Arc<RwLock<WhelkInferenceEngine>>,
    schema_service: Arc<SchemaService>,
    reasoner: Arc<OntologyReasoner>,
    pathfinding: Arc<SemanticPathfindingService>,
}
```

### Method: discover

```rust
pub async fn discover(
    &self,
    query: &str,
    limit: usize,
    domain_filter: Option<&str>,
) -> Result<Vec<DiscoveryResult>, String>
```

**Algorithm:**

1. **Keyword extraction**: Split query into stems, map to known OWL classes via
   `preferred_term` and `alt_terms` matching in Neo4j
2. **Whelk expansion**: For each matched class, retrieve the transitive closure
   via `whelk.get_subclass_hierarchy()` — include all subclasses AND superclasses
   up to `depth=3`
3. **Relationship fan-out**: Follow `HAS_PART`, `REQUIRES`, `ENABLES`,
   `BRIDGES_TO`, `RELATES_TO` edges from each matched class
4. **Scoring**: Rank results by `(keyword_match × 0.4) + (quality_score × 0.3) +
   (authority_score × 0.2) + (recency × 0.1)`
5. **Domain filter**: If specified, restrict to classes where
   `source_domain = domain_filter`
6. **Dedup and limit**: Return top-N unique results

**Return type:**

```rust
pub struct DiscoveryResult {
    pub iri: String,
    pub preferred_term: String,
    pub definition_summary: String,    // First 200 chars of definition
    pub relevance_score: f32,
    pub quality_score: f32,
    pub domain: String,
    pub relationships: Vec<RelationshipSummary>,
    pub whelk_inferred: bool,          // true if found via inference
}

pub struct RelationshipSummary {
    pub rel_type: String,              // "has_part", "requires", etc.
    pub target_iri: String,
    pub target_term: String,
}
```

### Method: read_note

```rust
pub async fn read_note(
    &self,
    iri: &str,
) -> Result<EnrichedNote, String>
```

**Return type:**

```rust
pub struct EnrichedNote {
    pub iri: String,
    pub term_id: String,
    pub preferred_term: String,
    pub markdown_content: String,      // Full Logseq markdown
    pub ontology_metadata: OntologyMetadata,
    pub whelk_axioms: Vec<InferredAxiomSummary>,
    pub related_notes: Vec<RelatedNote>,
    pub schema_context: String,        // SchemaService.to_llm_context()
}

pub struct OntologyMetadata {
    pub owl_class: String,
    pub physicality: String,
    pub role: String,
    pub domain: String,
    pub quality_score: f32,
    pub authority_score: f32,
    pub maturity: String,
    pub status: String,
    pub parent_classes: Vec<String>,
}

pub struct InferredAxiomSummary {
    pub axiom_type: String,            // "SubClassOf", "EquivalentClass"
    pub subject: String,
    pub object: String,
    pub is_inferred: bool,             // true = Whelk inferred, false = asserted
}

pub struct RelatedNote {
    pub iri: String,
    pub preferred_term: String,
    pub relationship_type: String,
    pub direction: String,             // "outgoing" or "incoming"
    pub summary: String,               // First 150 chars of markdown
}
```

### Method: validate_and_execute_cypher

```rust
pub async fn validate_and_execute_cypher(
    &self,
    cypher: &str,
    max_repair_attempts: usize,
) -> Result<CypherResult, CypherValidationError>
```

**Algorithm:**

1. Parse Cypher to extract node labels, relationship types, property keys
2. For each label: check `ontology_repo.get_owl_class(label_as_iri)`
3. For each rel type: check against known `OwlProperty` IRIs
4. If invalid: generate error message with closest matching IRIs
   (Levenshtein distance on preferred_terms)
5. Return errors + hints for agent self-repair
6. If valid: execute against Neo4j via `graph_repo`

### Method: traverse

```rust
pub async fn traverse(
    &self,
    start_iri: &str,
    depth: usize,
    relationship_types: Option<Vec<String>>,
) -> Result<TraversalResult, String>
```

Uses `SemanticPathfindingService.query_traversal()` with ontology-aware
weighting. Returns subgraph of notes within `depth` hops, filtered by
relationship types if specified.

---

## 2. OntologyMutationService — Agent Write Path

### Struct Definition

```rust
// src/services/ontology_mutation_service.rs

pub struct OntologyMutationService {
    ontology_repo: Arc<dyn OntologyRepository>,
    whelk: Arc<RwLock<WhelkInferenceEngine>>,
    parser: OntologyParser,
    github_pr: Arc<GitHubPRService>,
}
```

### Method: propose_create

```rust
pub async fn propose_create(
    &self,
    proposal: NoteProposal,
    agent_context: AgentContext,
) -> Result<ProposalResult, ProposalError>
```

**NoteProposal:**

```rust
pub struct NoteProposal {
    pub preferred_term: String,
    pub definition: String,
    pub owl_class: String,             // e.g., "ai:VisionTransformer"
    pub physicality: String,           // VirtualEntity, PhysicalEntity, etc.
    pub role: String,                  // Process, Artifact, Concept, etc.
    pub domain: String,                // ai, bc, mv, etc.
    pub is_subclass_of: Vec<String>,   // parent class IRIs
    pub relationships: HashMap<String, Vec<String>>,  // rel_type → [target_iris]
    pub alt_terms: Vec<String>,
}

pub struct AgentContext {
    pub agent_id: String,
    pub agent_type: String,
    pub task_description: String,
    pub session_id: Option<String>,
    pub confidence: f32,
}
```

**Algorithm:**

1. **Generate term-id**: Look up next sequence number for domain prefix
   (e.g., `AI-0851` if highest existing is `AI-0850`)
2. **Generate markdown**: Build Logseq note with full OntologyBlock:

```rust
fn generate_logseq_markdown(proposal: &NoteProposal, term_id: &str) -> String {
    format!(r#"- {preferred_term}
  - ### OntologyBlock
    - ontology:: true
    - term-id:: {term_id}
    - preferred-term:: {preferred_term}
    - source-domain:: {domain}
    - status:: agent-proposed
    - public-access:: true
    - last-updated:: {today}
    - definition:: {definition}
    - owl:class:: {owl_class}
    - owl:physicality:: {physicality}
    - owl:role:: {role}
    - is-subclass-of:: {parents_as_wiki_links}
    - quality-score:: {computed_quality}
    - authority-score:: 0.5
    - maturity:: draft
    {relationships_section}
    {alt_terms_section}
"#)
}
```

3. **Parse and validate**: Run `OntologyParser.parse_enhanced()` on the
   generated markdown to verify it round-trips correctly
4. **Whelk consistency check**:

```rust
async fn check_consistency(
    &self,
    proposed_axioms: Vec<OwlAxiom>,
) -> Result<ConsistencyReport, String> {
    let mut whelk = self.whelk.write().await;

    // Load existing ontology + proposed axioms
    let existing = self.ontology_repo.get_classes().await?;
    let existing_axioms = self.ontology_repo.get_axioms().await?;

    let mut all_axioms = existing_axioms;
    all_axioms.extend(proposed_axioms.clone());

    whelk.load_ontology(existing, all_axioms).await?;
    let results = whelk.infer().await?;

    let consistent = whelk.check_consistency().await?;
    let new_subsumptions = results.inferred_axioms.len();

    Ok(ConsistencyReport {
        consistent,
        new_subsumptions,
        explanation: if !consistent {
            Some(whelk.explain_entailment(&problematic_axiom).await?)
        } else { None },
    })
}
```

5. **Stage proposal in Neo4j**:

```cypher
CREATE (p:OntologyProposal {
  proposal_id: $id,
  agent_id: $agent_id,
  agent_type: $agent_type,
  action: "create",
  target_iri: $iri,
  markdown_content: $markdown,
  consistency_check: $consistency_status,
  quality_score: $quality,
  agent_confidence: $confidence,
  task_description: $task,
  status: "staged",
  created_at: datetime()
})
WITH p
MATCH (target:OwlClass {iri: $parent_iri})
CREATE (p)-[:PROPOSES_SUBCLASS_OF]->(target)
```

6. **Submit to GitHub PR pipeline**

### Method: propose_amend

```rust
pub async fn propose_amend(
    &self,
    target_iri: &str,
    amendment: NoteAmendment,
    agent_context: AgentContext,
) -> Result<ProposalResult, ProposalError>
```

**NoteAmendment:**

```rust
pub struct NoteAmendment {
    pub add_relationships: HashMap<String, Vec<String>>,
    pub remove_relationships: HashMap<String, Vec<String>>,
    pub update_definition: Option<String>,
    pub update_quality_score: Option<f32>,
    pub add_alt_terms: Vec<String>,
    pub custom_fields: HashMap<String, String>,
}
```

Generates a diff against the existing `markdown_content`, applies changes,
validates, consistency-checks, and stages.

---

## 3. GitHubPRService — Feedback Loop

### Struct Definition

```rust
// src/services/github_pr_service.rs

pub struct GitHubPRService {
    github_token: String,
    repo_owner: String,
    repo_name: String,
    base_branch: String,
    http_client: reqwest::Client,
}
```

### Method: create_ontology_pr

```rust
pub async fn create_ontology_pr(
    &self,
    proposal: &ProposalResult,
    markdown: &str,
    file_path: &str,
    agent_context: &AgentContext,
    consistency_report: &ConsistencyReport,
) -> Result<PullRequestResult, GitHubError>
```

**Implementation via GitHub REST API:**

1. **Get base branch SHA**: `GET /repos/{owner}/{repo}/git/ref/heads/{base}`
2. **Create blob**: `POST /repos/{owner}/{repo}/git/blobs` with markdown content
3. **Create tree**: `POST /repos/{owner}/{repo}/git/trees` with the new/modified file
4. **Create commit**: `POST /repos/{owner}/{repo}/git/commits`
5. **Create branch ref**: `POST /repos/{owner}/{repo}/git/refs`
   Branch name: `ontology/{agent_type}-{proposal_id}`
6. **Create pull request**: `POST /repos/{owner}/{repo}/pulls`

**PR body template:**

```markdown
## Proposed Change

{agent's task_description}

**Action**: {create|amend}
**Agent**: {agent_type} ({agent_id})

## Affected Classes

| IRI | Change |
|-----|--------|
| {iri} | {description} |

## Whelk Consistency Report

{if consistent}
✅ **Consistent** — no logical contradictions introduced
- New subsumptions inferred: {count}
{endif}

{if not consistent}
❌ **Inconsistent** — proposal rejected before staging
- Explanation: {whelk explanation}
{endif}

## Quality Assessment

- Quality Score: {score}/1.0
- Agent Confidence: {confidence}/1.0
- Tier Coverage: {tier_1_complete}/{tier_2_fields}/{tier_3_fields}

## Diff

```diff
{unified diff of markdown changes}
`` `
```

---

## 4. Ontology Tool Types

```rust
// src/types/ontology_tools.rs

use serde::{Deserialize, Serialize};

/// Input for ontology_discover tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoverInput {
    pub query: String,
    pub limit: Option<usize>,
    pub domain: Option<String>,
}

/// Input for ontology_read tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadNoteInput {
    pub iri: String,
}

/// Input for ontology_query tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryInput {
    pub cypher: String,
}

/// Input for ontology_traverse tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraverseInput {
    pub start_iri: String,
    pub depth: Option<usize>,
    pub relationship_types: Option<Vec<String>>,
}

/// Input for ontology_propose tool
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action")]
pub enum ProposeInput {
    #[serde(rename = "create")]
    Create(NoteProposal),
    #[serde(rename = "amend")]
    Amend {
        target_iri: String,
        amendment: NoteAmendment,
    },
}

/// Input for ontology_validate tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateInput {
    pub axioms: Vec<AxiomInput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomInput {
    pub axiom_type: String,
    pub subject: String,
    pub object: String,
}
```

---

## 5. Configuration

```rust
// Added to AppFullSettings in src/config/mod.rs

pub struct OntologyAgentSettings {
    /// Max discovery results per query
    pub discovery_limit: usize,        // default: 20
    /// Max depth for traverse operations
    pub max_traverse_depth: usize,     // default: 5
    /// Max Cypher repair iterations
    pub max_query_repair: usize,       // default: 3
    /// GitHub repo for PR creation
    pub github_repo_owner: Option<String>,
    pub github_repo_name: Option<String>,
    pub github_base_branch: Option<String>,  // default: "main"
    /// Minimum quality score for proposals
    pub min_proposal_quality: f32,     // default: 0.5
    /// Auto-reject proposals that fail Whelk consistency
    pub reject_inconsistent: bool,     // default: true
}
```

---

## 6. Neo4j Schema Additions

```cypher
-- OntologyProposal node label and indexes
CREATE CONSTRAINT proposal_id IF NOT EXISTS
  FOR (p:OntologyProposal) REQUIRE p.proposal_id IS UNIQUE;

CREATE INDEX proposal_status IF NOT EXISTS
  FOR (p:OntologyProposal) ON (p.status);

CREATE INDEX proposal_agent IF NOT EXISTS
  FOR (p:OntologyProposal) ON (p.agent_id);

CREATE INDEX proposal_target IF NOT EXISTS
  FOR (p:OntologyProposal) ON (p.target_iri);

-- Relationship: proposal → target class
-- (p:OntologyProposal)-[:PROPOSES_CHANGE_TO]->(c:OwlClass)
-- (p:OntologyProposal)-[:PROPOSES_SUBCLASS_OF]->(c:OwlClass)
```

---

## 7. Data Flow Summary

```
Agent Task → ontology_discover("transformers")
         → [{ai:TransformerArchitecture, ai:AttentionMechanism, ...}]

Agent    → ontology_read("ai:TransformerArchitecture")
         → {markdown, metadata, whelk_axioms, related_notes, schema}

Agent    → ontology_query("MATCH (n:OwlClass)-[:HAS_PART]->(p) RETURN p")
         → validate → execute → results

Agent    → ontology_propose(create, {term: "Vision Transformer", ...})
         → Whelk check → stage in Neo4j → GitHub PR → human review

Human    → approves PR → merge → GitHubSyncService → Neo4j → Whelk re-infer

All agents → now see "Vision Transformer" in the ontology
```
