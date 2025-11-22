# Canonical Ontology Block Schema

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Authoritative Specification
**Author:** Chief Architect - Swarm Ontology Standardization
**Purpose:** Definitive standard for all ontology blocks across 1,709 markdown files

---

## Executive Summary

This document defines THE canonical format for ontology blocks in the Logseq knowledge graph. It resolves inconsistencies across 6 existing patterns, corrects namespace issues, standardizes naming conventions, and establishes a hybrid OWL2/Logseq architecture that balances formal semantic rigor with practical usability.

**Key Design Decisions:**
- **Three-tier property system**: Required (Tier 1), Recommended (Tier 2), Optional (Tier 3)
- **Layered architecture**: Formal OWL2 semantics + Pragmatic Logseq metadata + Human documentation
- **Namespace standardization**: ai:, bc:, rb:, mv: with strict domain separation
- **CamelCase class naming**: Consistent across all domains
- **Status vs. maturity clarity**: Separate workflow state from conceptual maturity
- **OWL 2 DL profile**: For decidable automated reasoning
- **100% backward compatible**: Additive changes only

---

## Design Principles

### 1. Hybrid Architecture

The canonical format separates three distinct layers:

**Layer 1: Formal OWL2 Semantics**
- Machine-readable axioms in OWL Functional Syntax
- Reasoner-compatible (Pellet, HermiT, FaCT++)
- Exportable to standard formats (RDF/XML, Turtle, Manchester)
- OWL 2 DL profile for decidability

**Layer 2: Pragmatic Logseq Metadata**
- Native Logseq properties for workflow management
- Editorial status, versioning, quality metrics
- User-facing identifiers and labels
- Query-optimized property structure

**Layer 3: Human Documentation**
- Explanatory text sections
- Academic context and research references
- UK localization content
- Future directions and use cases

### 2. Multi-Domain Support

Four primary domain namespaces with clear boundaries:

| Namespace | Domain | Class Prefix | Example |
|-----------|--------|--------------|---------|
| `ai:` | Artificial Intelligence | AI-XXXX | `ai:LargeLanguageModel` |
| `bc:` | Blockchain & Cryptography | BC-XXXX | `bc:ConsensusMechanism` |
| `rb:` | Robotics & Autonomous Systems | RB-XXXX | `rb:AerialRobot` |
| `mv:` | Metaverse & Virtual Worlds | (numeric) | `mv:GameEngine` |

**Cross-domain concepts** use the most specific applicable namespace + multi-domain classification via `belongsToDomain`.

### 3. Semantic Completeness

Every ontology block must provide:
- **Identity**: Unique identifiers (Logseq ID, term-id, preferred-term)
- **Definition**: Comprehensive formal definition with concept links
- **Classification**: OWL class assignment with physicality/role dimensions
- **Taxonomy**: Parent class relationships (is-subclass-of)
- **Provenance**: Authoritative sources and review status
- **Status**: Workflow state (draft/complete) and conceptual maturity

### 4. Naming Conventions

**Logseq Properties**: `kebab-case` (lowercase with hyphens)
- `is-subclass-of`, `has-part`, `depends-on`, `source-domain`

**OWL Classes**: `PascalCase` (initial capitals, no separators)
- `MachineLearning`, `ConsensusMechanism`, `AerialRobot`

**OWL Properties**: `camelCase` (initial lowercase)
- `enablesTechnology`, `requiresInput`, `hasPart`

**Namespace Prefixes**: `lowercase:` (all lowercase)
- `ai:`, `bc:`, `rb:`, `mv:`, `owl:`, `rdfs:`

### 5. Status vs. Maturity Distinction

**status** - Editorial workflow state:
- `draft`: Initial creation, under development
- `in-progress`: Active editing and revision
- `complete`: Editorial review complete, ready for use
- `deprecated`: Superseded or obsolete

**maturity** - Conceptual stability and adoption:
- `draft`: Emerging concept, limited adoption
- `emerging`: Gaining traction, early research
- `mature`: Well-established, widely adopted
- `established`: Standardized, authoritative

**Example**: A newly written article about a mature technology should have `status:: complete` and `maturity:: mature`.

---

## Canonical Structure

### Complete Block Template

```markdown
- ### OntologyBlock
  id:: [domain]-[concept-slug]-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: [DOMAIN-NNNN]
    - preferred-term:: [Human Readable Name]
    - alt-terms:: [[Alternative Name 1]], [[Alternative Name 2]]
    - source-domain:: [ai | blockchain | robotics | metaverse | general]
    - status:: [draft | in-progress | complete | deprecated]
    - public-access:: [true | false]
    - version:: [M.m.p]
    - last-updated:: [YYYY-MM-DD]
    - quality-score:: [0.0-1.0]
    - cross-domain-links:: [number]

  - **Definition**
    - definition:: [Comprehensive formal definition with [[concept links]]. 2-5 sentences providing precise meaning, context, distinguishing characteristics, and purpose.]
    - maturity:: [draft | emerging | mature | established]
    - source:: [[Authoritative Source 1]], [[Source 2]]
    - authority-score:: [0.0-1.0]
    - scope-note:: [Optional: clarification of boundaries, context, usage constraints]

  - **Semantic Classification**
    - owl:class:: [namespace:ClassName]
    - owl:physicality:: [PhysicalEntity | VirtualEntity | AbstractEntity | HybridEntity]
    - owl:role:: [Object | Process | Agent | Quality | Relation | Concept]
    - owl:inferred-class:: [namespace:PhysicalityRole]
    - belongsToDomain:: [[PrimaryDomain]], [[SecondaryDomain]]
    - implementedInLayer:: [[LayerName]]

  - #### Relationships
    id:: [domain]-[concept-slug]-relationships

    - is-subclass-of:: [[ParentClass1]], [[ParentClass2]]
    - is-part-of:: [[WholeSystem]]
    - has-part:: [[Component1]], [[Component2]]
    - requires:: [[Requirement1]], [[Requirement2]]
    - depends-on:: [[Dependency1]]
    - enables:: [[EnabledCapability1]], [[EnabledCapability2]]
    - relates-to:: [[RelatedConcept1]], [[RelatedConcept2]]

  - #### CrossDomainBridges
    [Only include if concept bridges multiple domains]
    - bridges-to:: [[TargetDomainConcept]] via [relationship-type]
    - bridges-from:: [[SourceDomainConcept]] via [relationship-type]

  - #### OWL Axioms
    id:: [domain]-[concept-slug]-owl-axioms
    collapsed:: true
    [Include for key concepts requiring formal semantics]

    - ```clojure
      Prefix(:=<http://narrativegoldmine.com/[domain]#>)
      Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
      Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
      Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
      Prefix(dcterms:=<http://purl.org/dc/terms/>)

      Ontology(<http://narrativegoldmine.com/[domain]/[TERM-ID]>

        # Class Declaration
        Declaration(Class(:[ClassName]))

        # Taxonomic Hierarchy
        SubClassOf(:[ClassName] :[ParentClass])

        # Annotations
        AnnotationAssertion(rdfs:label :[ClassName] "[Preferred Term]"@en)
        AnnotationAssertion(rdfs:comment :[ClassName] "[Definition]"@en)
        AnnotationAssertion(dcterms:created :[ClassName] "[YYYY-MM-DD]"^^xsd:date)

        # Classification Axioms
        SubClassOf(:[ClassName] :[PhysicalityClass])
        SubClassOf(:[ClassName] :[RoleClass])

        # Property Restrictions
        SubClassOf(:[ClassName]
          ObjectSomeValuesFrom(:requires :[RequiredClass]))

        SubClassOf(:[ClassName]
          ObjectSomeValuesFrom(:hasPart :[ComponentClass]))

        # Property Characteristics (if defining properties)
        TransitiveObjectProperty(:isPartOf)
        AsymmetricObjectProperty(:requires)
        InverseObjectProperties(:hasPart :isPartOf)
      )
      ```
```

---

## Property Specifications

### Tier 1: Required Properties (MANDATORY)

All ontology blocks MUST include these properties:

#### Identification Section

**ontology** (boolean)
- **Required**: YES
- **Values**: `true`
- **Purpose**: Marks block as formal ontology entry vs. general note
- **Example**: `ontology:: true`

**term-id** (string)
- **Required**: YES
- **Format**: `{DOMAIN}-{NNNN}` where NNNN is zero-padded 4-digit number
- **Domain Prefixes**: AI, BC, RB, or numeric for metaverse
- **Example**: `term-id:: AI-0850`, `term-id:: RB-0010`, `term-id:: 20341`
- **Uniqueness**: MUST be globally unique across entire knowledge graph

**preferred-term** (string)
- **Required**: YES
- **Format**: Human-readable canonical name in Title Case
- **Purpose**: Primary display name for concept
- **Example**: `preferred-term:: Large Language Models`
- **Conventions**:
  - Use singular form for concepts (Machine Learning, not Machine Learnings)
  - Use plural for collections (Large Language Models refers to the class)
  - Include spaces and proper capitalization

**source-domain** (string)
- **Required**: YES
- **Values**: `ai`, `blockchain`, `robotics`, `metaverse`, `general`
- **Purpose**: Indicates originating knowledge domain
- **Example**: `source-domain:: ai`

**status** (enum)
- **Required**: YES
- **Values**: `draft`, `in-progress`, `complete`, `deprecated`
- **Purpose**: Editorial workflow state
- **Example**: `status:: complete`

**public-access** (boolean)
- **Required**: YES
- **Values**: `true`, `false`
- **Purpose**: Indicates if content is publicly shareable
- **Example**: `public-access:: true`

**last-updated** (date)
- **Required**: YES
- **Format**: ISO 8601 date `YYYY-MM-DD`
- **Purpose**: Tracks content currency
- **Example**: `last-updated:: 2025-11-21`

#### Definition Section

**definition** (string)
- **Required**: YES
- **Format**: 2-5 sentences providing comprehensive formal definition
- **Must Include**:
  - Precise meaning and scope
  - Contextual framing
  - Distinguishing characteristics
  - Links to related concepts using [[wiki links]]
- **Example**: `definition:: A Large Language Model is a type of artificial intelligence system based on deep neural network architectures (typically [[Transformer]] models) that has been trained on vast amounts of text data to understand and generate human-like text. These models demonstrate emergent capabilities including [[Few-Shot Learning]], contextual understanding, and multi-task performance without task-specific training.`

#### Semantic Classification Section

**owl:class** (IRI)
- **Required**: YES
- **Format**: `{namespace}:{ClassName}` in PascalCase
- **Purpose**: Formal OWL class identifier
- **Example**: `owl:class:: ai:LargeLanguageModel`
- **Conventions**:
  - Use appropriate namespace (ai:, bc:, rb:, mv:)
  - PascalCase with no separators
  - Descriptive and unambiguous

**owl:physicality** (enum)
- **Required**: YES
- **Values**: `PhysicalEntity`, `VirtualEntity`, `AbstractEntity`, `HybridEntity`
- **Purpose**: Distinguishes tangible vs. conceptual entities
- **Example**: `owl:physicality:: VirtualEntity`
- **Guidelines**:
  - PhysicalEntity: Physical objects, hardware, robots
  - VirtualEntity: Software, digital systems, algorithms
  - AbstractEntity: Concepts, theories, methodologies
  - HybridEntity: Cyber-physical systems combining physical and virtual

**owl:role** (enum)
- **Required**: YES
- **Values**: `Object`, `Process`, `Agent`, `Quality`, `Relation`, `Concept`
- **Purpose**: Functional classification
- **Example**: `owl:role:: Process`
- **Guidelines**:
  - Object: Things, entities, artifacts
  - Process: Activities, methods, operations
  - Agent: Autonomous actors with goals
  - Quality: Attributes, properties, characteristics
  - Relation: Connections, links, associations
  - Concept: Abstract ideas, categories, frameworks

#### Relationships Section

**is-subclass-of** (page link list)
- **Required**: YES (at least one parent class)
- **Format**: `[[ParentClass1]], [[ParentClass2]]`
- **Purpose**: Establishes taxonomic hierarchy
- **Example**: `is-subclass-of:: [[Machine Learning]], [[Neural Network Architecture]]`
- **Constraints**:
  - Multiple inheritance allowed
  - Must form valid DAG (no cycles)
  - Root concepts use `[[owl:Thing]]` as parent

---

### Tier 2: Recommended Properties (STRONGLY ENCOURAGED)

These properties significantly enhance ontology quality and should be included whenever applicable:

#### Identification Section (Tier 2)

**alt-terms** (page link list)
- **Format**: `[[Alternative Name 1]], [[Alternative Name 2]]`
- **Purpose**: Alternative names, synonyms, abbreviations
- **Example**: `alt-terms:: [[LLM]], [[Large Language AI]], [[Foundation Model]]`

**version** (semver string)
- **Format**: `M.m.p` (major.minor.patch)
- **Purpose**: Semantic versioning for content
- **Example**: `version:: 1.2.0`
- **Guidelines**:
  - Major: Breaking changes to definition or classification
  - Minor: Enhancements, additional relationships
  - Patch: Corrections, formatting fixes

**quality-score** (decimal)
- **Format**: `0.0` to `1.0`
- **Purpose**: Overall quality assessment
- **Example**: `quality-score:: 0.92`
- **Factors**: Definition completeness, source authority, relationship richness, OWL axiom quality

**cross-domain-links** (integer)
- **Purpose**: Count of links to concepts in other domains
- **Example**: `cross-domain-links:: 47`
- **Auto-computed**: Should be calculated by validation tools

#### Definition Section (Tier 2)

**maturity** (enum)
- **Values**: `draft`, `emerging`, `mature`, `established`
- **Purpose**: Conceptual stability and real-world adoption
- **Example**: `maturity:: mature`
- **Guidelines**:
  - draft: Newly proposed, speculative
  - emerging: Early research, limited adoption
  - mature: Well-understood, widely used
  - established: Standardized, authoritative consensus

**source** (page link list)
- **Format**: `[[Source 1]], [[Source 2]]`
- **Purpose**: Authoritative references for definition
- **Example**: `source:: [[OpenAI Research]], [[ISO/IEC 23257:2021]]`
- **Types**: Standards bodies, research papers, official documentation

**authority-score** (decimal)
- **Format**: `0.0` to `1.0`
- **Purpose**: Reliability and sourcing quality
- **Example**: `authority-score:: 0.95`
- **Factors**: Source credibility, peer review, standards alignment

**scope-note** (string)
- **Purpose**: Clarify boundaries, usage constraints, context
- **Example**: `scope-note:: This definition focuses on autoregressive language models; excludes masked language models like BERT.`

#### Semantic Classification Section (Tier 2)

**owl:inferred-class** (IRI)
- **Format**: `{namespace}:{PhysicalityRole}`
- **Purpose**: Computed classification from physicality + role
- **Example**: `owl:inferred-class:: ai:VirtualProcess`
- **Auto-computed**: Can be derived from physicality and role

**belongsToDomain** (page link list)
- **Format**: `[[Domain1]], [[Domain2]]`
- **Purpose**: Ontological domain classification
- **Example**: `belongsToDomain:: [[AI-GroundedDomain]], [[ComputationAndIntelligenceDomain]]`
- **Common Domains**:
  - [[AI-GroundedDomain]]
  - [[ComputationAndIntelligenceDomain]]
  - [[InfrastructureDomain]]
  - [[CryptographicDomain]]
  - [[DataManagementDomain]]
  - [[CreativeMediaDomain]]
  - [[RoboticsDomain]]
  - [[MetaverseDomain]]

#### Relationships Section (Tier 2)

**has-part** (page link list)
- **Purpose**: Compositional structure (whole has parts)
- **Example**: `has-part:: [[Encoder]], [[Decoder]], [[Attention Mechanism]]`
- **Inverse**: `is-part-of`

**is-part-of** (page link list)
- **Purpose**: Compositional membership (part belongs to whole)
- **Example**: `is-part-of:: [[Neural Network Architecture]]`
- **Inverse**: `has-part`

**requires** (page link list)
- **Purpose**: Technical dependencies and prerequisites
- **Example**: `requires:: [[Training Data]], [[Computational Resources]], [[GPU Infrastructure]]`
- **Inverse**: `is-required-by`

**depends-on** (page link list)
- **Purpose**: Logical or functional dependencies
- **Example**: `depends-on:: [[Transformer Architecture]], [[Attention Mechanism]]`
- **Inverse**: `is-dependency-of`

**enables** (page link list)
- **Purpose**: Capabilities and functionalities provided
- **Example**: `enables:: [[Few-Shot Learning]], [[Zero-Shot Learning]], [[Context Understanding]]`
- **Inverse**: `enabled-by`

**relates-to** (page link list)
- **Purpose**: General semantic associations
- **Example**: `relates-to:: [[Natural Language Processing]], [[Prompt Engineering]]`
- **Symmetric**: Bidirectional relationship

---

### Tier 3: Optional Properties (CONTEXTUAL)

These properties provide additional value for specific use cases:

#### Semantic Classification Section (Tier 3)

**implementedInLayer** (page link list)
- **Purpose**: Architectural layer assignment
- **Example**: `implementedInLayer:: [[ApplicationLayer]]`
- **Common Layers**: [[PlatformLayer]], [[ApplicationLayer]], [[SecurityLayer]], [[ProtocolLayer]]

#### CrossDomainBridges Section (Tier 3)

**bridges-to** (structured)
- **Format**: `[[TargetConcept]] via [relationship-type]`
- **Purpose**: Cross-domain outbound connections
- **Example**: `bridges-to:: [[Blockchain Verification]] via enables`

**bridges-from** (structured)
- **Format**: `[[SourceConcept]] via [relationship-type]`
- **Purpose**: Cross-domain inbound connections
- **Example**: `bridges-from:: [[Voice Interaction]] via has-part`

#### External Integration (Tier 3)

**doi** (string)
- **Purpose**: Digital Object Identifier for primary source
- **Example**: `doi:: 10.1038/nature14539`

**wikidata** (string)
- **Purpose**: Wikidata entity ID for cross-reference
- **Example**: `wikidata:: Q48806228`

**uri** (URI)
- **Purpose**: Persistent web identifier
- **Example**: `uri:: http://dbpedia.org/resource/Machine_learning`

#### Change Management (Tier 3)

**changelog** (string)
- **Purpose**: Summary of major changes
- **Example**: `changelog:: v1.2.0 - Added cross-domain bridges to robotics; v1.1.0 - Enhanced definition with 2025 context`

**replaces** (page link)
- **Purpose**: Deprecated concept superseded by this one
- **Example**: `replaces:: [[Legacy Machine Learning Definition]]`

**deprecated** (boolean)
- **Purpose**: Marks obsolete concepts
- **Example**: `deprecated:: true`

**superseded-by** (page link)
- **Purpose**: Current concept replacing this deprecated one
- **Example**: `superseded-by:: [[Modern AI Framework]]`

---

## Namespace Conventions

### Domain Namespaces

**ai: - Artificial Intelligence Domain**
- **Base URI**: `http://narrativegoldmine.com/ai#`
- **Term ID Prefix**: `AI-XXXX`
- **Class Example**: `ai:LargeLanguageModel`
- **Scope**: Machine learning, neural networks, AI governance, intelligent systems

**bc: - Blockchain Domain**
- **Base URI**: `http://narrativegoldmine.com/blockchain#`
- **Term ID Prefix**: `BC-XXXX`
- **Class Example**: `bc:ConsensusMechanism`
- **Scope**: Distributed ledgers, cryptocurrency, cryptographic protocols, smart contracts

**rb: - Robotics Domain**
- **Base URI**: `http://narrativegoldmine.com/robotics#`
- **Term ID Prefix**: `RB-XXXX`
- **Class Example**: `rb:AerialRobot`
- **Scope**: Robot types, control systems, sensors, actuators, autonomous systems
- **CRITICAL FIX**: All existing `mv:rb*` must change to `rb:*` (see migration rules)

**mv: - Metaverse Domain**
- **Base URI**: `http://narrativegoldmine.com/metaverse#`
- **Term ID Prefix**: Numeric (20001, 20002, etc.)
- **Class Example**: `mv:GameEngine`
- **Scope**: Virtual worlds, digital twins, spatial computing, VR/AR/XR

### Standard Namespaces

**owl: - Web Ontology Language**
- **URI**: `http://www.w3.org/2002/07/owl#`
- **Usage**: `owl:Class`, `owl:Thing`, `owl:ObjectProperty`

**rdfs: - RDF Schema**
- **URI**: `http://www.w3.org/2000/01/rdf-schema#`
- **Usage**: `rdfs:label`, `rdfs:comment`, `rdfs:subClassOf`

**xsd: - XML Schema Datatypes**
- **URI**: `http://www.w3.org/2001/XMLSchema#`
- **Usage**: `xsd:string`, `xsd:integer`, `xsd:decimal`, `xsd:date`

**dcterms: - Dublin Core Terms**
- **URI**: `http://purl.org/dc/terms/`
- **Usage**: `dcterms:created`, `dcterms:modified`, `dcterms:creator`

**skos: - Simple Knowledge Organization System**
- **URI**: `http://www.w3.org/2004/02/skos/core#`
- **Usage**: `skos:prefLabel`, `skos:altLabel`, `skos:broader`

---

## OWL Axioms Guidelines

### When to Include OWL Axioms

**INCLUDE for:**
- Core ontology classes (foundational concepts)
- Concepts with complex property restrictions
- Formally published terms (standards, specifications)
- Classes requiring automated reasoning
- Domain hubs with many relationships

**OMIT for:**
- Simple leaf concepts with minimal relationships
- Draft terms under active development
- Community-contributed general notes
- Purely descriptive content pages

### Axiom Structure

**Minimum Viable Axioms:**
```clojure
Declaration(Class(:ClassName))
SubClassOf(:ClassName :ParentClass)
AnnotationAssertion(rdfs:label :ClassName "Preferred Term"@en)
AnnotationAssertion(rdfs:comment :ClassName "Definition text"@en)
```

**Enhanced Axioms:**
```clojure
# Property restrictions
SubClassOf(:ClassName
  ObjectSomeValuesFrom(:requires :RequiredClass))

# Multiple inheritance
SubClassOf(:ClassName :ParentClass1)
SubClassOf(:ClassName :ParentClass2)

# Disjointness
DisjointClasses(:PhysicalEntity :VirtualEntity :AbstractEntity)

# Property characteristics
TransitiveObjectProperty(:isPartOf)
AsymmetricObjectProperty(:requires)
InverseObjectProperties(:hasPart :isPartOf)

# Domain and range
ObjectPropertyDomain(:enables :Technology)
ObjectPropertyRange(:enables :Capability)
```

### OWL 2 DL Profile Compliance

The canonical format targets **OWL 2 DL** for full expressivity with decidable reasoning:

**Constraints:**
- Strict separation of classes, properties, and individuals
- No punning (same IRI for multiple entity types)
- Type consistency in axioms
- Decidable consistency checking

**Profile Selection Guide:**
- **OWL 2 DL**: Default for rich ontologies requiring reasoning (RECOMMENDED)
- **OWL 2 EL**: Large ontologies with simple class hierarchies (optional optimization)
- **OWL 2 QL**: Query-heavy applications with database backends
- **OWL 2 RL**: Rule-based reasoning systems

---

## Validation Rules

### Structural Validation

1. **Required Properties Present**: All Tier 1 properties must exist
2. **Property Value Types**: Correct types (boolean, string, date, decimal, list)
3. **Date Format**: ISO 8601 `YYYY-MM-DD`
4. **Version Format**: Semantic versioning `M.m.p`
5. **Term ID Format**: `{PREFIX}-{NNNN}` or numeric
6. **Term ID Uniqueness**: No duplicates across entire graph

### Semantic Validation

7. **Parent Class Exists**: All `is-subclass-of` targets must be valid pages
8. **No Cycles**: Taxonomy must form DAG (directed acyclic graph)
9. **Namespace Correctness**: Domain namespace matches source-domain
10. **Class Name Convention**: PascalCase for OWL classes
11. **Property Name Convention**: kebab-case for Logseq properties

### Content Validation

12. **Definition Length**: 2-5 sentences (100-500 characters)
13. **Definition Has Links**: At least 2 [[wiki links]] to related concepts
14. **Source Attribution**: Tier 2+ concepts should cite sources
15. **Authority Score Range**: 0.0 ≤ authority-score ≤ 1.0
16. **Quality Score Range**: 0.0 ≤ quality-score ≤ 1.0

### OWL Validation

17. **OWL Syntax**: Valid OWL Functional Syntax if axioms present
18. **Namespace Declarations**: All prefixes declared in axioms
19. **Class Declarations**: Classes declared before use
20. **Consistency**: No unsatisfiable classes (reasoner check)

---

## Migration Strategy

### Priority Order

**Phase 1: Critical Fixes (Immediate)**
1. Fix robotics namespace (mv: → rb:)
2. Standardize class naming (CamelCase)
3. Resolve status/maturity confusions
4. Add missing required properties

**Phase 2: Structure (Weeks 1-2)**
1. Migrate to canonical structure
2. Move Relationships into OntologyBlock
3. Standardize indentation (2 spaces)
4. Remove empty sections

**Phase 3: Enhancement (Weeks 3-4)**
1. Add Tier 2 properties
2. Populate OWL axioms for core concepts
3. Add cross-domain bridges
4. Enhance definitions

**Phase 4: Formal Semantics (Weeks 5-6)**
1. Complete OWL axioms for all domains
2. Validate with reasoner
3. Generate machine-readable exports
4. Integrate with external tools

### Backward Compatibility

**Guaranteed:**
- All existing page links remain functional
- Existing block IDs preserved
- Existing Logseq queries continue working
- Property additions only (no deletions)

**Deprecated (but supported):**
- Flat property structure (migrate to sectioned)
- Tabs instead of spaces (migrate to 2 spaces)
- Duplicate metadata sections (consolidate)
- External Relationships sections (move inside OntologyBlock)

---

## Quality Assurance Checklist

**Before marking `status:: complete`:**

- [ ] All Tier 1 properties present
- [ ] Definition is comprehensive (2+ sentences)
- [ ] At least one `is-subclass-of` parent defined
- [ ] `source` cites authoritative references (for Tier 2+)
- [ ] `last-updated` reflects current date
- [ ] Wiki links resolve or are intentional forward references
- [ ] OWL axioms parse correctly (if present)
- [ ] Content sections follow recommended structure
- [ ] UK English spelling consistent
- [ ] No bare URLs (all formatted or annotated)
- [ ] Class naming follows PascalCase convention
- [ ] Namespace matches source-domain
- [ ] No duplicate metadata sections
- [ ] Reasoner validation passes (if OWL axioms present)

---

## Examples by Domain

See domain-specific template files:

- `/docs/ontology-migration/schemas/templates/template-ai.md` - Artificial Intelligence
- `/docs/ontology-migration/schemas/templates/template-blockchain.md` - Blockchain
- `/docs/ontology-migration/schemas/templates/template-robotics.md` - Robotics
- `/docs/ontology-migration/schemas/templates/template-metaverse.md` - Metaverse
- `/docs/ontology-migration/schemas/templates/template-general.md` - General Concepts

---

## References

1. W3C OWL 2 Web Ontology Language Document Overview (2012)
2. W3C OWL 2 Web Ontology Language Primer, Second Edition
3. Logseq Documentation - Block Properties and Queries
4. Noy, N. F., & McGuinness, D. L. (2001). Ontology Development 101
5. Hitzler, P., et al. (2012). OWL 2 Web Ontology Language Primer
6. Block Patterns Catalog v1.0 (internal analysis)
7. Best Practices Research v1.0 (internal research)
8. Semantic Patterns Analysis v1.0 (internal analysis)

---

**Document Control:**
- **Version**: 1.0.0
- **Status**: Authoritative
- **Approved By**: Chief Architect
- **Review Date**: 2025-11-21
- **Next Review**: 2025-12-21
- **Changelog**: Initial canonical specification
