# Neo4j Ontology Database Scripts

This directory contains scripts for initializing and managing the Neo4j ontology database for VisionFlow's enhanced data ingestion pipeline.

## Overview

VisionFlow uses **Neo4j as the single source of truth** for ontology data, with rich metadata from OntologyBlock headers in the Logseq knowledge graph.

## Database Schema

### Node Types

#### `:OwlClass` - Ontology Concepts
**Properties:**
- **Core Identification**: `iri`, `term_id`, `preferred_term`, `label`
- **Classification**: `source_domain`, `status`, `maturity`, `version`
- **Quality Metrics**: `quality_score`, `authority_score`
- **OWL2 Properties**: `owl_physicality`, `owl_role`
- **Domain Relations**: `belongs_to_domain`, `bridges_to_domain`
- **Source Tracking**: `source_file`, `file_sha1`, `markdown_content`
- **Content**: `description`, `definition`

#### `:OwlProperty` - Ontology Properties
**Properties:**
- `iri`, `label`, `property_type`, `domain`, `range`

#### `:OwlAxiom` - Logical Axioms
**Properties:**
- `id`, `axiom_type`, `subject`, `object`, `is_inferred`, `confidence`

### Relationship Types

- `:SUBCLASS_OF` - Taxonomic hierarchy
- `:RELATES_TO` - Semantic relationships (requires, enables, has-part, etc.)
  - Properties: `relationship_type`, `confidence`, `is_inferred`
- `:BRIDGES_TO` - Cross-domain connections
  - Properties: `via`, `confidence`

## Scripts

### `initialize-ontology-schema.cypher`

Complete database initialization script that:
1. Creates uniqueness constraints
2. Creates 30+ performance indexes
3. Loads example data for each domain (AI, Blockchain, Metaverse)
4. Creates sample relationships
5. Runs verification queries

**Usage:**
```bash
# Local Neo4j instance
cypher-shell < scripts/neo4j/initialize-ontology-schema.cypher

# Docker Neo4j instance
cat scripts/neo4j/initialize-ontology-schema.cypher | \\
  docker exec -i visionflow-neo4j cypher-shell -u neo4j -p your_password
```

## Integration with VisionFlow

### Rust Neo4j Adapter

The enhanced ontology parser automatically populates Neo4j with rich metadata:

```rust
use visionflow::adapters::neo4j_ontology_repository::{Neo4jOntologyRepository, Neo4jOntologyConfig};

// Initialize repository
let config = Neo4jOntologyConfig {
    uri: "neo4j://localhost:7687".to_string(),
    username: "neo4j".to_string(),
    password: std::env::var("NEO4J_PASSWORD")?,
    database: "neo4j".to_string(),
};

let repo = Neo4jOntologyRepository::new(config).await?;

// Add ontology class with full metadata
let class = OwlClass {
    iri: "http://narrativegoldmine.com/ai#LargeLanguageModel".to_string(),
    term_id: Some("AI-0850".to_string()),
    preferred_term: Some("Large Language Models".to_string()),
    quality_score: Some(0.95),
    maturity: Some("mature".to_string()),
    owl_physicality: Some("VirtualEntity".to_string()),
    belongs_to_domain: Some("AIEthicsDomain".to_string()),
    // ... other fields
};

repo.add_owl_class(&class).await?;
```

### Data Flow

```
GitHub Repository (Logseq Markdown)
    ↓
Local Sync (SHA1 differential updates)
    ↓
Enhanced Ontology Parser (Rust)
    ├─ Extract OntologyBlock headers
    ├─ Parse Tier 1/2/3 properties
    ├─ Detect domains, topics, relationships
    └─ Extract git metadata
    ↓
Neo4j Ontology Repository
    ├─ Store classes with rich metadata
    ├─ Create semantic relationships
    └─ Build cross-domain bridges
    ↓
GPU Force Calculation (semantic physics)
    ├─ Apply ontology-aware forces
    ├─ Cluster by physicality/role
    └─ Stage by maturity
    ↓
Real-time WebSocket Updates (clients)
```

## Common Queries

### Find High-Quality Classes
```cypher
MATCH (c:OwlClass)
WHERE c.quality_score >= 0.9
RETURN c.term_id, c.preferred_term, c.quality_score
ORDER BY c.quality_score DESC
LIMIT 20;
```

### Cross-Domain Bridges
```cypher
MATCH (c1:OwlClass)-[b:BRIDGES_TO]->(c2:OwlClass)
WHERE c1.source_domain <> c2.source_domain
RETURN c1.source_domain + ' → ' + c2.source_domain as bridge,
       count(*) as connections
ORDER BY connections DESC;
```

### Maturity Distribution
```cypher
MATCH (c:OwlClass)
WHERE c.maturity IS NOT NULL
RETURN c.maturity, count(*) as count
ORDER BY count DESC;
```

### Physicality Clustering
```cypher
MATCH (c:OwlClass)
WHERE c.owl_physicality IS NOT NULL
RETURN c.owl_physicality, c.owl_role, count(*) as count
ORDER BY c.owl_physicality, count DESC;
```

## Environment Variables

```bash
# Neo4j connection
export NEO4J_URI="neo4j://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_secure_password"
export NEO4J_DATABASE="neo4j"
```

## Performance Tuning

For optimal performance with large ontology graphs (1000+ classes):

```
# neo4j.conf settings
dbms.memory.heap.initial_size=2G
dbms.memory.heap.max_size=4G
dbms.memory.pagecache.size=2G
dbms.db.timezone=UTC
```

## Backup & Restore

```bash
# Backup
neo4j-admin database dump neo4j --to=/backups/ontology-$(date +%Y%m%d).dump

# Restore
neo4j-admin database load neo4j --from=/backups/ontology-20251122.dump --overwrite-destination=true
```

## See Also

- [Neo4j Schema V2 Documentation](../../docs/neo4j-rich-ontology-schema-v2.md)
- [Ontology Parser Implementation](../../docs/enhanced-ontology-parser-implementation.md)
- [Semantic Forces Guide](../../docs/guides/ontology-semantic-forces.md)
