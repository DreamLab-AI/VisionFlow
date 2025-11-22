// Neo4j Ontology Schema Initialization Script
// VisionFlow Rich Ontology Metadata Schema V2
// Created: 2025-11-22
//
// This script initializes the Neo4j database with constraints, indexes,
// and example data for the enhanced ontology ingestion pipeline.
//
// Run with: cypher-shell < initialize-ontology-schema.cypher

// ============================================================================
// STEP 1: CREATE CONSTRAINTS
// ============================================================================

// Primary constraints for uniqueness
CREATE CONSTRAINT owl_class_iri_unique IF NOT EXISTS
FOR (c:OwlClass) REQUIRE c.iri IS UNIQUE;

CREATE CONSTRAINT owl_property_iri_unique IF NOT EXISTS
FOR (p:OwlProperty) REQUIRE p.iri IS UNIQUE;

CREATE CONSTRAINT owl_axiom_id_unique IF NOT EXISTS
FOR (a:OwlAxiom) REQUIRE a.id IS UNIQUE;

// Term ID uniqueness (for ontology blocks)
CREATE CONSTRAINT owl_class_term_id_unique IF NOT EXISTS
FOR (c:OwlClass) REQUIRE c.term_id IS UNIQUE;

// ============================================================================
// STEP 2: CREATE INDEXES FOR QUERY PERFORMANCE
// ============================================================================

// Core identification indexes
CREATE INDEX owl_class_label_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.label);
CREATE INDEX owl_class_preferred_term_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.preferred_term);
CREATE INDEX owl_class_term_id_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.term_id);

// Classification indexes
CREATE INDEX owl_class_source_domain_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.source_domain);
CREATE INDEX owl_class_status_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.status);
CREATE INDEX owl_class_maturity_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.maturity);

// Quality metric indexes
CREATE INDEX owl_class_quality_score_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.quality_score);
CREATE INDEX owl_class_authority_score_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.authority_score);

// OWL2 property indexes
CREATE INDEX owl_class_physicality_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.owl_physicality);
CREATE INDEX owl_class_role_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.owl_role);

// Domain relationship indexes
CREATE INDEX owl_class_belongs_to_domain_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.belongs_to_domain);
CREATE INDEX owl_class_bridges_to_domain_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.bridges_to_domain);

// Source tracking indexes
CREATE INDEX owl_class_source_file_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.source_file);
CREATE INDEX owl_class_version_idx IF NOT EXISTS FOR (c:OwlClass) ON (c.version);

// Property indexes
CREATE INDEX owl_property_label_idx IF NOT EXISTS FOR (p:OwlProperty) ON (p.label);
CREATE INDEX owl_property_type_idx IF NOT EXISTS FOR (p:OwlProperty) ON (p.property_type);

// Axiom indexes
CREATE INDEX owl_axiom_type_idx IF NOT EXISTS FOR (a:OwlAxiom) ON (a.axiom_type);
CREATE INDEX owl_axiom_inferred_idx IF NOT EXISTS FOR (a:OwlAxiom) ON (a.is_inferred);

// Relationship indexes
CREATE INDEX owl_relationship_type_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relationship_type);
CREATE INDEX owl_relationship_confidence_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.confidence);

// ============================================================================
// STEP 3: EXAMPLE DATA - Domain Ontology Classes
// ============================================================================

// AI Domain Example
MERGE (ai_gov:OwlClass {iri: 'http://narrativegoldmine.com/ai#AIGovernance'})
SET ai_gov.term_id = 'AI-0091',
    ai_gov.preferred_term = 'AI Governance',
    ai_gov.label = 'AI Governance',
    ai_gov.source_domain = 'ai',
    ai_gov.status = 'complete',
    ai_gov.maturity = 'mature',
    ai_gov.quality_score = 0.95,
    ai_gov.authority_score = 0.95,
    ai_gov.owl_physicality = 'VirtualEntity',
    ai_gov.owl_role = 'Process',
    ai_gov.belongs_to_domain = 'AIEthicsDomain',
    ai_gov.version = '1.0.0',
    ai_gov.public_access = true,
    ai_gov.description = 'Comprehensive system of organizational structures, policies, processes for AI oversight';

// Blockchain Domain Example
MERGE (blockchain:OwlClass {iri: 'http://narrativegoldmine.com/blockchain#Blockchain'})
SET blockchain.term_id = 'BC-0001',
    blockchain.preferred_term = 'Blockchain',
    blockchain.label = 'Blockchain',
    blockchain.source_domain = 'blockchain',
    blockchain.status = 'complete',
    blockchain.maturity = 'stable',
    blockchain.quality_score = 0.98,
    blockchain.authority_score = 0.98,
    blockchain.owl_physicality = 'VirtualEntity',
    blockchain.owl_role = 'Object',
    blockchain.belongs_to_domain = 'BlockchainDomain',
    blockchain.version = '2.1.0',
    blockchain.public_access = true,
    blockchain.description = 'Distributed ledger technology with cryptographic security';

// Metaverse Domain Example
MERGE (metaverse:OwlClass {iri: 'http://narrativegoldmine.com/metaverse#VirtualWorld'})
SET metaverse.term_id = 'MV-0001',
    metaverse.preferred_term = 'Virtual World',
    metaverse.label = 'Virtual World',
    metaverse.source_domain = 'metaverse',
    metaverse.status = 'in-progress',
    metaverse.maturity = 'emerging',
    metaverse.quality_score = 0.85,
    metaverse.authority_score = 0.87,
    metaverse.owl_physicality = 'VirtualEntity',
    metaverse.owl_role = 'Object',
    metaverse.belongs_to_domain = 'MetaverseDomain',
    metaverse.version = '1.0.0',
    metaverse.public_access = true;

// ============================================================================
// STEP 4: EXAMPLE RELATIONSHIPS
// ============================================================================

// Subclass relationships
MERGE (ai_ethics:OwlClass {iri: 'http://narrativegoldmine.com/ai#ArtificialIntelligence'})
SET ai_ethics.term_id = 'AI-0001',
    ai_ethics.preferred_term = 'Artificial Intelligence',
    ai_ethics.label = 'Artificial Intelligence',
    ai_ethics.source_domain = 'ai';

MERGE (ai_gov)-[:SUBCLASS_OF]->(ai_ethics);

// Semantic relationships with properties
MERGE (smart_contract:OwlClass {iri: 'http://narrativegoldmine.com/blockchain#SmartContract'})
SET smart_contract.term_id = 'BC-0478',
    smart_contract.preferred_term = 'Smart Contract',
    smart_contract.source_domain = 'blockchain';

MERGE (smart_contract)-[:RELATES_TO {
    relationship_type: 'uses',
    confidence: 0.95,
    is_inferred: false
}]->(blockchain);

MERGE (smart_contract)-[:RELATES_TO {
    relationship_type: 'enables',
    confidence: 0.90,
    is_inferred: false
}]->(ai_gov);

// Cross-domain bridge
MERGE (ai_gov)-[:BRIDGES_TO {
    via: 'regulation',
    confidence: 0.88
}]->(blockchain);

// ============================================================================
// STEP 5: EXAMPLE OWL PROPERTIES
// ============================================================================

MERGE (enables_prop:OwlProperty {iri: 'http://narrativegoldmine.com/dt#enables'})
SET enables_prop.label = 'enables',
    enables_prop.property_type = 'ObjectProperty',
    enables_prop.domain = 'Technology',
    enables_prop.range = 'Capability';

MERGE (requires_prop:OwlProperty {iri: 'http://narrativegoldmine.com/dt#requires'})
SET requires_prop.label = 'requires',
    requires_prop.property_type = 'ObjectProperty',
    requires_prop.domain = 'Technology',
    requires_prop.range = 'Resource';

// ============================================================================
// STEP 6: VERIFICATION QUERIES
// ============================================================================

// Return summary statistics
MATCH (c:OwlClass)
WITH count(c) as class_count
MATCH (p:OwlProperty)
WITH class_count, count(p) as property_count
MATCH ()-[r:RELATES_TO]->()
RETURN
    class_count as total_classes,
    property_count as total_properties,
    count(r) as total_relationships;

// Return classes by domain
MATCH (c:OwlClass)
RETURN c.source_domain as domain, count(c) as count
ORDER BY count DESC;

// Return quality distribution
MATCH (c:OwlClass)
WHERE c.quality_score IS NOT NULL
RETURN
    'quality_score' as metric,
    avg(c.quality_score) as average,
    min(c.quality_score) as minimum,
    max(c.quality_score) as maximum,
    count(c) as sample_size;

// ============================================================================
// INITIALIZATION COMPLETE
// ============================================================================
