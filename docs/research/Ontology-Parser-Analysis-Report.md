# Ontology Parser & OWL System Analysis Report

**Date**: 2025-11-02
**Author**: Research Agent
**Project**: WebXR Knowledge Graph System
**Purpose**: Analysis of ontology extraction capabilities and hybrid markdown format

---

## Executive Summary

This report analyzes the current ontology extraction system, documenting the hybrid markdown format used in GitHub repositories, examining the OWL class structure, and comparing the ontology parser with the knowledge graph parser. The system uses a specialized markdown format that embeds OWL ontology constructs within Logseq-style bullet points, enabling both human-readable documentation and machine-processable semantic information.

**Key Findings**:
- **Hybrid Format**: Rich Logseq markdown with embedded OWL axioms in Clojure functional syntax
- **Parser Capabilities**: Basic OWL extraction (classes, properties, axioms) but missing advanced constructs
- **Database Integration**: Full ontology-graph linkage with `owl_class_iri` fields on nodes/edges
- **Gap Analysis**: KG parser handles general relationships; ontology parser handles formal semantics

---

## 1. Ontology Parser Analysis

### 1.1 Current Implementation

**Location**: `src/services/parsers/ontology_parser.rs`

The `OntologyParser` extracts OWL constructs from markdown files marked with `- ### OntologyBlock`.

#### Extracted OWL Constructs

| OWL Construct | Pattern | Extraction Status |
|--------------|---------|-------------------|
| **Classes** | `owl_class:: <IRI>` | ✅ Full support |
| **Object Properties** | `objectProperty:: <IRI>` | ✅ Full support |
| **Data Properties** | `dataProperty:: <IRI>` | ✅ Full support |
| **SubClassOf Axioms** | `subClassOf:: <IRI>` | ✅ Full support |
| **Labels** | `label:: <text>` | ✅ Full support |
| **Descriptions** | `description:: <text>` | ✅ Full support |
| **Domain/Range** | `domain::` / `range::` | ✅ Full support |
| **Parent Classes** | Multiple `subClassOf` entries | ✅ Full support |

#### NOT Currently Extracted

| OWL Construct | Missing Pattern | Impact |
|--------------|-----------------|--------|
| **Disjoint Classes** | `owl:DisjointWith` | ⚠️ High - needed for constraints |
| **Equivalent Classes** | `owl:EquivalentClass` | ⚠️ Medium - semantic reasoning |
| **Inverse Properties** | `owl:inverseOf` | ⚠️ Medium - bidirectional relationships |
| **Property Characteristics** | `owl:Functional`, `owl:Transitive` | ⚠️ High - reasoning |
| **OWL Axioms Block** | Full functional syntax | ⚠️ High - formal semantics |
| **Individuals/Instances** | Instance declarations | ⚠️ Medium - ABox reasoning |
| **Complex Restrictions** | `ObjectSomeValuesFrom`, etc. | ⚠️ High - advanced constraints |

### 1.2 Code Structure

```rust
pub struct OntologyData {
    pub classes: Vec<OwlClass>,           // Extracted OWL classes
    pub properties: Vec<OwlProperty>,     // Object/Data properties
    pub axioms: Vec<OwlAxiom>,            // SubClassOf axioms only
    pub class_hierarchy: Vec<(String, String)>, // (child, parent) pairs
}
```

#### Parsing Flow

```
1. extract_ontology_section() → Find "### OntologyBlock" marker
2. extract_classes()          → Parse "owl_class::" declarations
3. extract_properties()       → Parse "objectProperty::" and "dataProperty::"
4. extract_axioms()           → Parse "subClassOf::" relationships
5. extract_class_hierarchy()  → Build hierarchy from axioms
```

#### Regex Patterns Used

```rust
// Classes (supports IRIs, prefixed names, and parenthesized forms)
r"owl:?_?class::\s*([a-zA-Z0-9_:/-]+(\([^)]+\))?)"

// Properties
r"objectProperty::\s*([a-zA-Z0-9_:/-]+)"
r"dataProperty::\s*([a-zA-Z0-9_:/-]+)"

// Axioms
r"subClassOf::\s*([a-zA-Z0-9_:/-]+)"
```

---

## 2. Hybrid Markdown Format Documentation

### 2.1 GitHub Repository Format

The system syncs from GitHub repositories containing **Logseq-style markdown** with embedded **OWL ontology blocks**.

#### Example: Metaverse.md

```markdown
- ### OntologyBlock
  id:: metaverse-ontology
  collapsed:: true
	- metaverseOntology:: true
	- term-id:: 20315
	- preferred-term:: Metaverse
	- definition:: A convergent network of persistent, synchronous 3D virtual worlds...
	- maturity:: mature
	- source:: [[ISO 23257]], [[ETSI GR MEC 032]], [[IEEE P2048]]
	- owl:class:: mv:Metaverse
	- owl:physicality:: VirtualEntity
	- owl:role:: Object
	- owl:inferred-class:: mv:VirtualObject
	- owl:functional-syntax:: true
	- belongsToDomain:: [[InfrastructureDomain]], [[VirtualSocietyDomain]]
	- implementedInLayer:: [[ApplicationLayer]]
	- #### Relationships
	  id:: metaverse-relationships
		- has-part:: [[Virtual World]], [[Avatar]], [[Digital Asset]]
		- requires:: [[3D Rendering]], [[Network Infrastructure]]
		- depends-on:: [[Internet]], [[Cloud Computing]], [[Extended Reality]]
		- enables:: [[Social VR]], [[Virtual Commerce]], [[Immersive Entertainment]]
	- #### OWL Axioms
	  id:: metaverse-owl-axioms
	  collapsed:: true
		- ```clojure
		  Declaration(Class(mv:Metaverse))

		  # Classification along two primary dimensions
		  SubClassOf(mv:Metaverse mv:VirtualEntity)
		  SubClassOf(mv:Metaverse mv:Object)

		  # Core architectural requirements
		  SubClassOf(mv:Metaverse
		    ObjectSomeValuesFrom(mv:hasPart mv:VirtualWorld)
		  )
		  SubClassOf(mv:Metaverse
		    ObjectSomeValuesFrom(mv:hasPart mv:Avatar)
		  )

		  # Domain classifications
		  SubClassOf(mv:Metaverse
		    ObjectSomeValuesFrom(mv:belongsToDomain mv:InfrastructureDomain)
		  )
		  ```
```

### 2.2 Metadata Properties Available

| Property Key | Type | Example | Purpose |
|-------------|------|---------|---------|
| `id::` | Logseq ID | `metaverse-ontology` | Block reference |
| `term-id::` | Integer | `20315` | Unique term identifier |
| `preferred-term::` | String | `Metaverse` | Display name |
| `definition::` | Text | "A convergent network..." | Human-readable definition |
| `maturity::` | Enum | `mature` | Development stage |
| `source::` | WikiLinks | `[[ISO 23257]]` | Standards references |
| `owl:class::` | IRI | `mv:Metaverse` | OWL class identifier |
| `owl:physicality::` | Class | `VirtualEntity` | Classification dimension |
| `owl:role::` | Class | `Object` | Classification dimension |
| `belongsToDomain::` | WikiLinks | `[[InfrastructureDomain]]` | Domain categorization |
| `implementedInLayer::` | WikiLink | `[[ApplicationLayer]]` | Architecture layer |

### 2.3 Relationship Types

Relationships are expressed as **semantic properties** with WikiLink targets:

```markdown
- has-part:: [[Virtual World]], [[Avatar]], [[Digital Asset]]
- requires:: [[3D Rendering]], [[Network Infrastructure]]
- depends-on:: [[Internet]], [[Cloud Computing]]
- enables:: [[Social VR]], [[Virtual Commerce]]
- binds-to:: [[Head-Mounted Display]], [[Rendered 3D Environment]]
```

### 2.4 OWL Axioms Section

The most advanced feature is the **OWL Axioms** block using **OWL Functional Syntax** in Clojure format:

```clojure
Declaration(Class(mv:Metaverse))

# Subsumption
SubClassOf(mv:Metaverse mv:VirtualEntity)

# Existential restrictions
SubClassOf(mv:Metaverse
  ObjectSomeValuesFrom(mv:hasPart mv:VirtualWorld)
)

# Domain constraints
SubClassOf(mv:Metaverse
  ObjectSomeValuesFrom(mv:belongsToDomain mv:InfrastructureDomain)
)
```

**This block is NOT currently parsed by the ontology parser!**

---

## 3. OWL Class System in Database

### 3.1 Database Schema

**Source**: `migration/unified_schema.sql`

#### `owl_classes` Table

```sql
CREATE TABLE owl_classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE NOT NULL,                    -- Full IRI
    local_name TEXT NOT NULL,                    -- Short name
    namespace_id INTEGER,                        -- Namespace reference
    label TEXT,
    comment TEXT,
    deprecated BOOLEAN DEFAULT 0,

    -- Hierarchy
    parent_class_iri TEXT,                       -- Direct parent

    -- Content tracking (CRITICAL for GitHub sync)
    markdown_content TEXT,                       -- Source markdown block
    file_sha1 TEXT,                              -- Checksum for cache invalidation
    source_file TEXT,                            -- Markdown file path

    -- Timestamps
    created_at TIMESTAMP,
    updated_at TIMESTAMP,

    FOREIGN KEY (parent_class_iri) REFERENCES owl_classes(iri)
);
```

#### `owl_properties` Table

```sql
CREATE TABLE owl_properties (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE NOT NULL,
    property_type TEXT CHECK(property_type IN ('object', 'datatype', 'annotation')),

    -- Domain/Range
    domain_class_iri TEXT,
    range_class_iri TEXT,

    -- OWL Characteristics (IMPORTANT!)
    is_functional BOOLEAN DEFAULT 0,
    is_inverse_functional BOOLEAN DEFAULT 0,
    is_transitive BOOLEAN DEFAULT 0,
    is_symmetric BOOLEAN DEFAULT 0,
    is_asymmetric BOOLEAN DEFAULT 0,
    is_reflexive BOOLEAN DEFAULT 0,
    is_irreflexive BOOLEAN DEFAULT 0,

    FOREIGN KEY (domain_class_iri) REFERENCES owl_classes(iri),
    FOREIGN KEY (range_class_iri) REFERENCES owl_classes(iri)
);
```

#### `owl_axioms` Table

```sql
CREATE TABLE owl_axioms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Axiom type (18 types supported!)
    axiom_type TEXT CHECK(axiom_type IN (
        'SubClassOf', 'DisjointClasses', 'EquivalentClasses',
        'SubObjectPropertyOf', 'EquivalentProperties',
        'FunctionalProperty', 'TransitiveProperty',
        'SymmetricProperty', 'ClassAssertion', ...
    )),

    subject_id INTEGER,
    object_id INTEGER,
    property_id INTEGER,

    -- Physics constraint parameters (GPU translation)
    strength REAL DEFAULT 1.0,
    priority INTEGER DEFAULT 5,
    distance REAL,
    user_defined BOOLEAN DEFAULT 0,

    -- Inference tracking
    inferred BOOLEAN DEFAULT 0,
    inference_method TEXT,
    source_axiom_id INTEGER
);
```

### 3.2 Graph-Ontology Integration

**Critical Link**: Nodes have `owl_class_iri` field!

```sql
CREATE TABLE graph_nodes (
    id INTEGER PRIMARY KEY,
    metadata_id TEXT UNIQUE NOT NULL,
    label TEXT,

    -- 3D Physics
    x REAL, y REAL, z REAL,
    vx REAL, vy REAL, vz REAL,
    mass REAL DEFAULT 1.0,

    -- ONTOLOGY LINKAGE
    owl_class_iri TEXT,                          -- Links to owl_classes(iri)
    owl_individual_iri TEXT,                     -- Links to owl_individuals(iri)

    FOREIGN KEY (owl_class_iri) REFERENCES owl_classes(iri)
);
```

#### Rust Node Model

```rust
pub struct Node {
    pub id: u32,
    pub metadata_id: String,
    pub label: String,

    // Physics
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub z: Option<f32>,
    pub mass: Option<f32>,

    // ONTOLOGY LINKAGE
    pub owl_class_iri: Option<String>,  // Links to OWL class!

    pub metadata: HashMap<String, String>,
}
```

**Builder Method**:
```rust
node.with_owl_class_iri("mv:Metaverse".to_string())
```

---

## 4. Knowledge Graph Parser Comparison

### 4.1 KG Parser Capabilities

**Source**: `src/services/parsers/knowledge_graph_parser.rs`

#### What KG Parser Extracts

| Feature | Pattern | Purpose |
|---------|---------|---------|
| **Page Node** | Filename | Main document node |
| **WikiLinks** | `[[Target Page]]` | Linked pages (creates edges) |
| **Tags** | `#tag` or `tag:: #tag` | Categorization |
| **Properties** | `key:: value` | Generic metadata |
| **File Metadata** | `public:: true` | Visibility control |

#### KG Parser Output

```rust
pub struct GraphData {
    nodes: Vec<Node>,              // Page node + linked nodes
    edges: Vec<Edge>,              // WikiLink edges
    metadata: MetadataStore,       // Generic properties
    id_to_metadata: HashMap<String, String>,
}
```

### 4.2 Ontology Parser Capabilities

#### What Ontology Parser Extracts

| Feature | Pattern | Purpose |
|---------|---------|---------|
| **OWL Classes** | `owl_class:: IRI` | Formal class definitions |
| **Properties** | `objectProperty::` / `dataProperty::` | Semantic relationships |
| **Axioms** | `subClassOf::` | Class hierarchy |
| **Labels** | `label::` | Human-readable names |
| **Descriptions** | `description::` | Documentation |
| **Domain/Range** | `domain::` / `range::` | Property constraints |

#### Ontology Parser Output

```rust
pub struct OntologyData {
    classes: Vec<OwlClass>,         // OWL classes
    properties: Vec<OwlProperty>,   // Object/Data properties
    axioms: Vec<OwlAxiom>,          // Formal axioms
    class_hierarchy: Vec<(String, String)>,  // Hierarchy
}
```

### 4.3 Gap Analysis: What Each Parser Misses

| Information Type | KG Parser | Ontology Parser | Solution |
|-----------------|-----------|-----------------|----------|
| **WikiLinks** | ✅ Extracted as edges | ❌ Ignored | Keep KG parser |
| **Tags** | ✅ Extracted | ❌ Ignored | Keep KG parser |
| **Generic Properties** | ✅ All `key::value` | ❌ Only specific patterns | Keep KG parser |
| **OWL Classes** | ❌ Generic property | ✅ Formal semantics | Keep ontology parser |
| **Class Hierarchy** | ❌ No understanding | ✅ Axioms | Keep ontology parser |
| **OWL Axioms Block** | ❌ Ignored | ❌ **NOT EXTRACTED** | **NEW PARSER NEEDED** |
| **Relationships** | ✅ As WikiLinks | ⚠️ Only domain/range | **BOTH NEEDED** |
| **Functional Syntax** | ❌ Not parsed | ❌ **NOT EXTRACTED** | **NEW PARSER NEEDED** |

---

## 5. Recommended Unified Ontology Extraction Strategy

### 5.1 Architecture: Dual Parser System

**DO NOT replace** one parser with the other. They serve complementary purposes:

```
┌─────────────────────────────────────────────────────┐
│         GitHub Markdown File (*.md)                 │
│                                                     │
│  - ### OntologyBlock                                │
│    - owl:class:: mv:Metaverse                      │
│    - has-part:: [[Virtual World]]                  │
│    - #### OWL Axioms                                │
│      - ```clojure                                   │
│        SubClassOf(mv:Metaverse mv:VirtualEntity)   │
│        ```                                          │
└─────────────────────────────────────────────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  StreamingSyncService          │
    │  (determines file type)        │
    └───────────────────────────────┘
                    │
        ┌───────────┴──────────┐
        ▼                      ▼
┌────────────────┐    ┌─────────────────────┐
│ KG Parser      │    │ Ontology Parser     │
│ (WikiLinks,    │    │ (OWL Classes,       │
│  Tags,         │    │  Properties,        │
│  Metadata)     │    │  Axioms)            │
└────────────────┘    └─────────────────────┘
        │                      │
        ▼                      ▼
┌────────────────┐    ┌─────────────────────┐
│ graph_nodes    │◄───┤ owl_classes         │
│ graph_edges    │    │ owl_properties      │
│ (physics)      │    │ owl_axioms          │
└────────────────┘    └─────────────────────┘
                              │
                              ▼
                      ┌─────────────────┐
                      │ Physics GPU     │
                      │ Constraints     │
                      └─────────────────┘
```

### 5.2 Enhanced Ontology Parser Requirements

**Phase 1: OWL Axioms Block Extraction** (HIGH PRIORITY)

Add extraction of the `#### OWL Axioms` section:

```rust
fn extract_owl_axioms_block(&self, section: &str) -> Option<String> {
    // Find "#### OWL Axioms" section
    // Extract ```clojure ... ``` block
    // Return raw OWL functional syntax
}
```

**Phase 2: Functional Syntax Parser** (CRITICAL)

Implement a full OWL Functional Syntax parser:

```rust
pub struct OwlFunctionalParser {
    // Parse OWL constructs:
    // - Declaration(Class(...))
    // - SubClassOf(...)
    // - ObjectSomeValuesFrom(...)
    // - DisjointClasses(...)
    // - etc.
}

pub enum OwlExpression {
    Class(String),
    ObjectSomeValuesFrom { property: String, filler: Box<OwlExpression> },
    ObjectIntersectionOf(Vec<OwlExpression>),
    // ... more constructs
}
```

**Phase 3: Relationship Property Extraction**

Extract semantic relationships from the `#### Relationships` section:

```rust
fn extract_relationships(&self, section: &str) -> Vec<RelationshipTriple> {
    // Parse lines like:
    // - has-part:: [[Virtual World]], [[Avatar]]
    // - requires:: [[3D Rendering]]
    //
    // Return: (subject, predicate, object) triples
}
```

### 5.3 Integration with Database

**Store Raw OWL Axioms**:

```sql
ALTER TABLE owl_classes ADD COLUMN owl_axioms_block TEXT;
```

**Parse and Store**:

```rust
// In ontology_parser.rs
pub struct OwlClass {
    pub iri: String,
    pub label: Option<String>,
    pub markdown_content: Option<String>,  // Full OntologyBlock
    pub owl_axioms_block: Option<String>,  // Raw functional syntax
    pub parsed_axioms: Vec<OwlExpression>, // Parsed expressions
}
```

### 5.4 Streaming Sync Service Integration

**Current Flow** (in `streaming_sync_service.rs`):

```rust
match classify_file_type(&content) {
    FileType::KnowledgeGraph => {
        // Parse with KG parser
        let graph_data = kg_parser.parse(&content, &file_name)?;
        kg_repo.save_incremental(&graph_data).await?;
    }
    FileType::Ontology => {
        // Parse with Ontology parser
        let ontology_data = onto_parser.parse(&content, &file_name)?;
        onto_repo.save_ontology(&classes, &properties, &axioms).await?;
    }
}
```

**Enhanced Flow** (PROPOSED):

```rust
match classify_file_type(&content) {
    FileType::KnowledgeGraph => {
        // 1. Parse general graph structure
        let graph_data = kg_parser.parse(&content, &file_name)?;

        // 2. Check for embedded ontology blocks
        if has_ontology_block(&content) {
            let ontology_data = onto_parser.parse(&content, &file_name)?;

            // 3. Link nodes to OWL classes
            for node in &mut graph_data.nodes {
                if let Some(class_iri) = find_owl_class_for_node(&node, &ontology_data) {
                    node.owl_class_iri = Some(class_iri);
                }
            }

            // 4. Save both
            onto_repo.save_ontology(&ontology_data).await?;
        }

        kg_repo.save_incremental(&graph_data).await?;
    }
    FileType::Ontology => {
        // Ontology-first files (formal definitions)
        let ontology_data = onto_parser.parse(&content, &file_name)?;
        onto_repo.save_ontology(&ontology_data).await?;

        // Optionally create graph nodes for classes
        let graph_nodes = ontology_to_graph_nodes(&ontology_data);
        kg_repo.save_incremental(&graph_nodes).await?;
    }
}
```

---

## 6. Detailed Recommendations

### 6.1 Immediate Actions (Week 1)

1. ✅ **Document current format** (DONE - this report)
2. **Add OWL Axioms block extraction**:
   - Locate `#### OWL Axioms` section
   - Extract ` ```clojure ... ``` ` blocks
   - Store as raw text in `owl_classes.owl_axioms_block`
3. **Add relationship extraction**:
   - Parse `#### Relationships` sections
   - Extract semantic properties (`has-part::`, `requires::`, etc.)
   - Store as OWL object properties

### 6.2 Short-term Improvements (Weeks 2-4)

1. **Implement OWL Functional Syntax Parser**:
   - Use existing Rust parser combinator libraries (`nom`, `pest`, `combine`)
   - Parse OWL 2 Functional Syntax
   - Convert to database axioms
2. **Enhance axiom type support**:
   - Add `DisjointClasses`
   - Add `EquivalentClasses`
   - Add property characteristics (`Functional`, `Transitive`, etc.)
3. **Improve IRI handling**:
   - Support namespace prefixes (`mv:`, `rdf:`, `owl:`)
   - Validate IRIs against namespace table
   - Resolve prefixed names to full IRIs

### 6.3 Long-term Enhancements (Month 2+)

1. **Bidirectional sync**:
   - Detect ontology changes in GitHub
   - Incremental updates (only changed classes)
   - Preserve user-defined constraints
2. **Reasoning integration**:
   - Use `horned_integration` module
   - Materialize inferred axioms
   - Store in `inference_results` table
3. **Validation**:
   - Check OWL consistency
   - Validate axioms against schema
   - Report errors back to GitHub

---

## 7. Test Coverage Analysis

### 7.1 Existing Tests

**Location**: `tests/ontology_parser_test.rs`

✅ Covered:
- Basic OWL class parsing
- Class hierarchy (SubClassOf)
- Object/Data properties
- Domain/Range extraction
- Multiple parent classes
- IRI format variations

❌ NOT Covered:
- OWL Axioms block extraction
- Functional syntax parsing
- Relationship properties
- Disjoint classes
- Property characteristics
- Complex restrictions

### 7.2 Required New Tests

```rust
#[test]
fn test_parse_owl_axioms_block() {
    // Test extraction of ```clojure ... ``` blocks
}

#[test]
fn test_parse_relationships() {
    // Test has-part::, requires::, etc.
}

#[test]
fn test_parse_functional_syntax() {
    // Test ObjectSomeValuesFrom, DisjointClasses, etc.
}

#[test]
fn test_hybrid_kg_ontology_file() {
    // Test files with both WikiLinks AND OWL classes
}
```

---

## 8. Conclusion

### 8.1 Current State Assessment

| Component | Completeness | Quality | Notes |
|-----------|-------------|---------|-------|
| **OWL Class Extraction** | 70% | Good | Basic patterns work well |
| **Property Extraction** | 60% | Good | Missing characteristics |
| **Axiom Extraction** | 30% | Fair | Only SubClassOf supported |
| **Functional Syntax** | 0% | N/A | Not implemented |
| **Relationship Properties** | 0% | N/A | Not implemented |
| **Database Schema** | 95% | Excellent | Well-designed, ready for full OWL |
| **Graph Integration** | 80% | Good | `owl_class_iri` linkage works |

### 8.2 Strategic Direction

**DO NOT unify parsers into one**. The dual-parser architecture is correct:

- **KG Parser**: General graph structure, WikiLinks, tags, generic metadata
- **Ontology Parser**: Formal OWL semantics, class hierarchy, axioms, constraints

**Instead**:
1. Enhance ontology parser to extract ALL OWL constructs from markdown
2. Parse the `#### OWL Axioms` functional syntax blocks
3. Extract relationship properties from `#### Relationships` sections
4. Maintain clear separation of concerns
5. Link the two systems via `owl_class_iri` field (already in place!)

### 8.3 Success Metrics

When refactoring is complete, the system should:

✅ Extract 100% of OWL constructs from markdown (vs. 30% today)
✅ Parse OWL Functional Syntax from axiom blocks
✅ Support all 18 axiom types in database schema
✅ Link graph nodes to OWL classes via `owl_class_iri`
✅ Support incremental GitHub sync without data loss
✅ Enable physics constraint generation from OWL axioms
✅ Maintain backward compatibility with existing files

---

## Appendix A: File Type Classification

**Current Logic** (in `streaming_sync_service.rs`):

```rust
fn classify_file_type(content: &str) -> FileType {
    if content.contains("- ### OntologyBlock") {
        FileType::Ontology
    } else if content.contains("public:: true") {
        FileType::KnowledgeGraph
    } else {
        FileType::Skip
    }
}
```

**Issue**: Files with BOTH `public:: true` AND `OntologyBlock` are classified as Ontology only!

**Proposed Fix**:

```rust
fn classify_file_type(content: &str) -> FileType {
    let has_ontology = content.contains("- ### OntologyBlock");
    let is_public = content.contains("public:: true");

    match (has_ontology, is_public) {
        (true, true) => FileType::HybridOntologyKG,  // Process with BOTH parsers
        (true, false) => FileType::Ontology,
        (false, true) => FileType::KnowledgeGraph,
        (false, false) => FileType::Skip,
    }
}
```

---

## Appendix B: Example Ontology Repository Files

**Analyzed Repository**: `/home/devuser/workspace/OntologyDesign/VisioningLab/`

Sample files with rich ontology annotations:
- `Metaverse.md` - Full OWL axioms, relationships, metadata
- `Virtual Reality (VR).md` - Hybrid entity with physical/virtual bindings
- 100+ concept files with consistent ontology structure

**Key Observation**: ALL files use Logseq bullet format with consistent property naming conventions.

---

## Appendix C: References

- **Code Locations**:
  - Ontology Parser: `src/services/parsers/ontology_parser.rs`
  - KG Parser: `src/services/parsers/knowledge_graph_parser.rs`
  - Streaming Sync: `src/services/streaming_sync_service.rs`
  - Database Schema: `migration/unified_schema.sql`
  - Node Model: `src/models/node.rs`

- **Test Files**:
  - `tests/ontology_parser_test.rs`
  - `tests/basic_ontology_test.rs`

- **Documentation**:
  - `docs/guides/ontology-parser.md`
  - `docs/architecture/github-sync-service-design.md`

---

**End of Report**
