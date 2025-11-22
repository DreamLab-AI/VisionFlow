// ========================================================================
// Ontology Import Script for Neo4j
// Generated from Logseq Ontology Blocks
// Supports: AI, Blockchain, Robotics, Metaverse, Telecollaboration, Disruptive Technologies
// ========================================================================
// Usage: cypher-shell -u neo4j -p [password] < ontology.cypher
// ========================================================================

// Clear existing data (CAUTION: Deletes all nodes and relationships!)
// MATCH (n) DETACH DELETE n;

// Create constraints and indexes
CREATE CONSTRAINT concept_iri IF NOT EXISTS FOR (c:Concept) REQUIRE c.iri IS UNIQUE;
CREATE INDEX concept_term_id IF NOT EXISTS FOR (c:Concept) ON (c.term_id);
CREATE INDEX concept_preferred_term IF NOT EXISTS FOR (c:Concept) ON (c.preferred_term);
CREATE INDEX concept_domain IF NOT EXISTS FOR (c:Concept) ON (c.domain);
CREATE INDEX concept_status IF NOT EXISTS FOR (c:Concept) ON (c.status);

// ========================================================================
// Concept Nodes
// ========================================================================

CREATE (cAI-0348:Concept:UnknownDomain {
  iri: 'http://narrativegoldmine.com/metaverse#ADAS',
  term_id: 'AI-0348',
  preferred_term: 'ADAS',
  definition: 'Advanced Driver Assistance Systems (ADAS) are electronic systems that assist vehicle operators with driving and parking functions through automated technologies including adaptive cruise control, lane keeping assist, automatic emergency braking, blind spot detection, and parking assistance. ADAS represents SAE Level 1-2 automation, providing driver support whilst requiring continuous driver supervision and intervention capability.',
  domain: 'metaverse',
  source_domain: 'metaverse',
  owl_class: 'mv:ADAS',
  owl_physicality: 'ConceptualEntity',
  owl_role: 'Concept',
  status: 'draft',
  public_access: none,
  last_updated: '',
  maturity: 'draft',
  belongs_to_domains: ['MetaverseDomain'],
  source_file: '/home/user/logseq/mainKnowledgeGraph/pages/ADAS.md'
});


// ========================================================================
// Relationships
// ========================================================================

// Subclass relationships for AI-0348
MATCH (child:Concept {term_id: 'AI-0348'}), (parent:Concept {preferred_term: 'Metaverse'})
CREATE (child)-[:IS_SUBCLASS_OF]->(parent);


// ========================================================================
// Useful Queries
// ========================================================================

// Count concepts by domain:
// MATCH (c:Concept) RETURN c.domain, count(c) ORDER BY count(c) DESC;

// Find all AI concepts:
// MATCH (c:ArtificialIntelligence) RETURN c.term_id, c.preferred_term, c.definition LIMIT 25;

// Find all Blockchain concepts:
// MATCH (c:Blockchain) RETURN c.term_id, c.preferred_term, c.definition LIMIT 25;

// Find class hierarchy (subclass relationships):
// MATCH path = (child:Concept)-[:IS_SUBCLASS_OF*]->(parent:Concept)
// RETURN path LIMIT 50;

// Find top-level concepts (no parent):
// MATCH (c:Concept) WHERE NOT (c)-[:IS_SUBCLASS_OF]->()
// RETURN c.term_id, c.preferred_term, c.domain;

// Find cross-domain bridges:
// MATCH (source:Concept)-[r:BRIDGES_TO]->(target:Concept)
// WHERE source.domain <> target.domain
// RETURN source.term_id, source.domain, type(r), target.term_id, target.domain, r.via LIMIT 25;

// Find concepts by maturity level:
// MATCH (c:Concept {maturity: 'Mature'}) RETURN c.term_id, c.preferred_term LIMIT 25;

// Find high quality concepts (quality_score >= 0.8):
// MATCH (c:Concept) WHERE c.quality_score >= 0.8
// RETURN c.term_id, c.preferred_term, c.quality_score ORDER BY c.quality_score DESC;

// Full-text search in definitions:
// MATCH (c:Concept) WHERE c.definition CONTAINS 'learning'
// RETURN c.term_id, c.preferred_term, c.definition LIMIT 10;

