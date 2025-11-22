# Multi-Ontology Framework Standardization Strategy

**Version**: 1.0  
**Date**: 2025-11-21  
**Status**: Design Complete - Ready for Implementation  
**Scope**: 1,681 ontology files across 6 domains

---

## Executive Summary

This document defines the complete standardization strategy for the Logseq multi-ontology framework, resolving critical tensions between prefixed/non-prefixed filenames, ensuring IRI uniqueness, standardizing ontology block structure, and establishing clear migration phases.

**Key Decisions:**

1. **Filename Strategy**: **Preserve natural language names** - Keep existing filenames, use term-id as canonical identifier, add optional prefix for new files
2. **IRI Architecture**: **term-id-based URIs** - `http://ontology.logseq.io/{domain}#{TERM-ID}` guarantees uniqueness
3. **Ontology Block**: **Single canonical structure** - First in file, comprehensive metadata, domain-specific namespaces
4. **Public Property**: **Dual support during transition** - Support both `public::` and `public-access::`, migrate to unified `public-access::`
5. **Migration Approach**: **4-phase gradual migration** - No breaking changes, incremental quality improvement

**Impact:**
- **Preserves**: 1,345 natural language filenames (80% of corpus)
- **Fixes**: 100 robotics namespace errors (mv: → rb:)
- **Standardizes**: 1,521 ontology blocks to canonical format
- **Enhances**: IRI uniqueness guarantees via term-id registry

---

## 1. Filename Standardization Decision

### 1.1 Current State Analysis

**Distribution:**
- **Domain-prefixed (uppercase)**: ~236 files (14%) - AI-NNNN, BC-NNNN, RB-NNNN
- **Lowercase rb- prefix**: 100 files (6%) - rb-NNNN-descriptive-name
- **Natural language names**: 1,345 files (80%) - "AI Agent System.md", "Blockchain.md"

**Patterns Observed:**
```
AI-0387-ai-governance-framework.md          [Prefix + ID + descriptor]
rb-0010-aerial-robot.md                     [Lowercase prefix + ID + descriptor]
AI Agent System.md                          [Natural language]
Blockchain.md                               [Natural language]
3D Scene Exchange Protocol (SXP).md         [Natural language with acronym]
```

### 1.2 Chosen Strategy: **Preserve Natural Names + Metadata Enhancement**

**Rationale:**

1. **Minimal Disruption**: Renaming 1,345 files would break existing links, disrupt workflows, and create migration overhead
2. **term-id Already Serves as Unique Identifier**: Every file has `term-id:: DOMAIN-NNNN` property which is stable and unique
3. **Semantic Value**: Natural language filenames are human-readable, searchable, and semantically meaningful
4. **User Preference Alignment**: User stated "prefer not to lose the data that's currently represented in filenames"
5. **OWL Best Practice**: IRIs should be based on stable identifiers (term-id), not filenames

**Decision:**

```markdown
FILENAME POLICY:

1. PRESERVE existing filenames (all 1,681 files keep current names)

2. ENSURE term-id property is present and unique:
   - term-id:: AI-0600      [AI domain]
   - term-id:: BC-0051      [Blockchain domain]
   - term-id:: RB-0010      [Robotics domain - uppercase!]
   - term-id:: MV-20341     [Metaverse domain]
   - term-id:: TC-0042      [Telecollaboration domain]
   - term-id:: DT-0088      [Disruptive Tech domain]

3. NEW files SHOULD follow descriptive naming:
   - Preferred: "Concept Name.md" (natural language)
   - Acceptable: "DOMAIN-NNNN-concept-name.md" (prefixed)
   - Required: term-id property with format DOMAIN-NNNN

4. METADATA enhancement:
   - Add filename-history:: ["original-name.md"] for renamed files
   - Add domain-prefix:: DOMAIN for all files
   - Add sequence-number:: NNNN for all files
```

**Migration Actions:**

- **No filename changes required** for existing files
- **Standardize term-id format** to uppercase domain prefix (fix rb-NNNN → RB-NNNN)
- **Add metadata fields** to track domain and sequence number
- **Create term-id registry** to prevent collisions

### 1.3 Term-ID Standardization Rules

**Format:** `{DOMAIN}-{NUMBER}`

**Domain Prefixes:**
```
AI   - Artificial Intelligence
BC   - Blockchain & Cryptocurrency
RB   - Robotics (note: fix lowercase rb- to uppercase RB-)
MV   - Metaverse
TC   - Telecollaboration
DT   - Disruptive Technology
```

**Numbering:**
- 4-digit zero-padded: `AI-0001`, `BC-0051`, `RB-0010`
- OR 5-digit for large domains: `MV-20341`
- Sequential assignment within domain
- Gaps acceptable (deleted/merged concepts)

**Uniqueness Guarantee:**
- Maintain term-id registry (JSON file or database)
- Check for collisions before assigning new IDs
- Never reuse deleted term-ids

---

## 2. IRI Architecture Design

### 2.1 Requirements

1. **Uniqueness**: Every concept needs ONE canonical IRI
2. **Stability**: IRIs should not change when filenames change
3. **OWL2 Compliance**: Follow W3C namespace conventions
4. **Resolvability**: IRIs should potentially resolve to documentation
5. **Domain Separation**: Clear namespace per domain

### 2.2 IRI Structure Specification

**Base URI Pattern:**
```
http://ontology.logseq.io/{domain}#{TERM-ID}

Examples:
http://ontology.logseq.io/ai#AI-0600
http://ontology.logseq.io/blockchain#BC-0051  
http://ontology.logseq.io/robotics#RB-0010
http://ontology.logseq.io/metaverse#MV-20341
```

**Namespace Prefixes:**
```turtle
@prefix ai: <http://ontology.logseq.io/ai#> .
@prefix bc: <http://ontology.logseq.io/blockchain#> .
@prefix rb: <http://ontology.logseq.io/robotics#> .
@prefix mv: <http://ontology.logseq.io/metaverse#> .
@prefix tc: <http://ontology.logseq.io/telecollaboration#> .
@prefix dt: <http://ontology.logseq.io/disruptive-tech#> .
```

**Usage in OWL Axioms:**
```clojure
Prefix(ai:=<http://ontology.logseq.io/ai#>)
Prefix(bc:=<http://ontology.logseq.io/blockchain#>)
Prefix(rb:=<http://ontology.logseq.io/robotics#>)
Prefix(mv:=<http://ontology.logseq.io/metaverse#>)

Declaration(Class(ai:AI-0600))
AnnotationAssertion(rdfs:label ai:AI-0600 "AI Agent System"@en)
```

### 2.3 Uniqueness Guarantee Mechanism

**Term-ID Registry System:**

```json
{
  "registry": {
    "AI-0600": {
      "domain": "ai",
      "sequence": 600,
      "preferred-term": "AI Agent System",
      "filename": "AI Agent System.md",
      "iri": "http://ontology.logseq.io/ai#AI-0600",
      "status": "active",
      "created": "2025-01-15",
      "last-updated": "2025-11-15"
    },
    "RB-0010": {
      "domain": "robotics",
      "sequence": 10,
      "preferred-term": "Aerial Robot",
      "filename": "rb-0010-aerial-robot.md",
      "iri": "http://ontology.logseq.io/robotics#RB-0010",
      "status": "active",
      "created": "2025-10-28",
      "last-updated": "2025-11-16"
    }
  },
  "next-id": {
    "AI": 601,
    "BC": 195,
    "RB": 149,
    "MV": 20342,
    "TC": 43,
    "DT": 89
  }
}
```

**Collision Detection:**
1. Before creating new ontology entry, check registry
2. Auto-assign next available ID in domain
3. Validate no duplicate term-ids during migration
4. Tool: `scripts/ontology-migration/validate-term-ids.js`

**Registry Maintenance:**
- Updated automatically by migration scripts
- Committed to git with ontology changes
- Location: `/docs/ontology-migration/term-id-registry.json`

### 2.4 Class Name Convention

**Current Issues:**
- Robotics: `mv:rb0010aerialrobot` (wrong namespace, wrong case)
- AI: `ai:AIAgentSystem` (correct)
- Blockchain: `bc:ConsensusMechanism` (correct)

**Standardized Convention:**
```
Format: {namespace}:{PascalCaseName}

Correct Examples:
  rb:AerialRobot          (not mv:rb0010aerialrobot)
  ai:AIAgentSystem        (good)
  bc:ConsensusMechanism   (good)
  mv:GameEngine          (good)

Rules:
  1. Use domain-specific namespace (rb: not mv: for robotics!)
  2. PascalCase: capitalize first letter of each word
  3. No hyphens, underscores, or numbers (except acronyms)
  4. Acronyms: keep together (AI, VR, NFT)
  5. Multi-word: join without separators (ConsensusMechanism)
```

---

## 3. Ontology Block Canonical Structure

### 3.1 Requirements

1. **MUST be first** in markdown file (before any other content)
2. **Only ONE** OntologyBlock per file
3. **Unique IRI** via term-id
4. **Domain-specific namespace** for owl:class
5. **Required core properties** (ontology, term-id, preferred-term, definition, owl:class)
6. **Optional extension properties** (version, quality-score, authority-score)
7. **OWL axioms** for key concepts

### 3.2 Canonical Template

```markdown
- ### OntologyBlock
  id:: {concept-slug}-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: {DOMAIN-NNNN}
    - preferred-term:: {Human Readable Name}
    - alt-terms:: [[Alternative Name 1]], [[Alternative Name 2]]
    - source-domain:: {domain-identifier}
    - status:: {draft | in-progress | complete | deprecated}
    - public-access:: {true | false}
    - version:: {X.Y.Z}
    - last-updated:: {YYYY-MM-DD}
    - quality-score:: {0.0-1.0}
    - authority-score:: {0.0-1.0}
    - cross-domain-links:: {number}

  - **Definition**
    - definition:: {Comprehensive formal definition with [[concept links]]}
    - maturity:: {draft | emerging | mature | established}
    - source:: [[Authoritative Source 1]], [[Source 2]]
    - scope-note:: {Optional: clarification of boundaries and context}

  - **Semantic Classification**
    - owl:class:: {namespace:PascalCaseName}
    - owl:physicality:: {PhysicalEntity | VirtualEntity | AbstractEntity | HybridEntity}
    - owl:role:: {Object | Process | Agent | Quality | Relation}
    - owl:inferred-class:: {namespace:InferredClass}
    - belongsToDomain:: [[PrimaryDomain]], [[SecondaryDomain]]
    - implementedInLayer:: [[LayerName]]

  - #### Relationships
    id:: {concept-slug}-relationships
    - is-subclass-of:: [[ParentClass1]], [[ParentClass2]]
    - is-part-of:: [[Whole]]
    - has-part:: [[Part1]], [[Part2]]
    - requires:: [[Requirement1]]
    - depends-on:: [[Dependency]]
    - enables:: [[EnabledCapability]]
    - implements:: [[Interface]]
    - uses:: [[UsedTechnology]]

  - #### OWL Restrictions
    {Include only if non-empty}
    - {property} {quantifier} {class/value}
    
  - #### CrossDomainBridges
    {Include only if cross-domain relationships exist}
    - bridges-to:: [[TargetConcept]] via {relationship-type}
    - bridges-from:: [[SourceConcept]] via {relationship-type}

  - #### OWL Axioms
    id:: {concept-slug}-owl-axioms
    collapsed:: true
    {Include for key concepts requiring formal semantics}
    - ```clojure
      Prefix(:=<http://ontology.logseq.io/{domain}#>)
      Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
      Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
      Prefix(xsd:=<http://www.w3.org/2001/XMLSchema#>)
      
      Declaration(Class(:{TERM-ID}))
      
      SubClassOf(:{TERM-ID} :{ParentClass})
      
      AnnotationAssertion(rdfs:label :{TERM-ID} "{Preferred Term}"@en)
      AnnotationAssertion(rdfs:comment :{TERM-ID} "{Definition}"@en)
      
      {Additional axioms as needed}
      ```
```

### 3.3 Required vs Optional Properties

**Tier 1: REQUIRED (100% files must have)**
```markdown
- ontology:: true
- term-id:: {DOMAIN-NNNN}
- preferred-term:: {Name}
- definition:: {Text}
- source-domain:: {domain}
- status:: {value}
- owl:class:: {namespace:ClassName}
- is-subclass-of:: [[Parent]]
```

**Tier 2: STRONGLY RECOMMENDED (aim for 90%)**
```markdown
- public-access:: {true|false}
- version:: {X.Y.Z}
- last-updated:: {YYYY-MM-DD}
- maturity:: {value}
- source:: [[Reference]]
- authority-score:: {0.0-1.0}
- owl:physicality:: {value}
- owl:role:: {value}
- belongsToDomain:: [[Domain]]
```

**Tier 3: OPTIONAL (include when relevant)**
```markdown
- alt-terms:: [[Alt]]
- quality-score:: {0.0-1.0}
- cross-domain-links:: {number}
- scope-note:: {text}
- implementedInLayer:: [[Layer]]
- owl:inferred-class:: {class}
- OWL Restrictions section
- CrossDomainBridges section
- OWL Axioms section
```

### 3.4 Edge Case Handling

**Case 1: Multiple OntologyBlocks in One File**
```
Action: 
  1. Keep FIRST OntologyBlock as canonical
  2. Extract metadata from additional blocks
  3. Merge non-conflicting properties into first block
  4. Move additional blocks to ## Historical Metadata section (collapsed)
  5. Log merge in migration report
```

**Case 2: OntologyBlock NOT First in File**
```
Action:
  1. Move OntologyBlock to top of file (line 1)
  2. Preserve all other content
  3. Ensure proper markdown structure maintained
```

**Case 3: Has public:: true but NO OntologyBlock**
```
Action:
  1. Create minimal OntologyBlock with:
     - ontology:: true
     - term-id:: {auto-assign from registry}
     - preferred-term:: {from filename or page title}
     - source-domain:: {infer from content or manual review}
     - status:: draft
     - public-access:: true (from public:: property)
     - owl:class:: {domain}:{ClassFromFilename}
     - is-subclass-of:: {manual review or default [[Concept]]}
  2. Add definition from first paragraph or ## About section
  3. Mark for manual review (status: draft)
```

**Case 4: NO OntologyBlock and NO public:: true**
```
Action:
  SKIP - Not an ontology concept (likely journal entry, note, or draft)
  
  Exception: If file is linked from many ontology files (>10 backlinks),
             flag for manual review as potential missing ontology entry
```

---

## 4. Public Property Handling

### 4.1 Current State

**Patterns Observed:**
1. `public:: true` at page level (frontmatter-style)
2. `public-access:: true` in OntologyBlock
3. Both present in same file
4. Neither present

### 4.2 Strategy

**Transition Approach:**

```markdown
PHASE 1 (Current): Support Both
  - Accept: public:: true OR public-access:: true
  - Semantics: Either indicates public accessibility
  
PHASE 2 (Migration): Dual Property
  - Add: public-access:: true to OntologyBlock
  - Keep: public:: true at page level for backward compatibility
  - Sync: Ensure both have same value
  
PHASE 3 (Future): Unified Property
  - Standard: public-access:: true in OntologyBlock only
  - Deprecate: public:: true (but don't remove, for backward compatibility)
  - Tools: Only read public-access:: from OntologyBlock
```

**Decision Rules:**

| Case | Page-level `public::` | Block-level `public-access::` | Action |
|------|----------------------|------------------------------|--------|
| 1 | true | true | ✓ Keep both (consistent) |
| 2 | true | false | ⚠️ Conflict - manual review |
| 3 | true | missing | → Add `public-access:: true` to block |
| 4 | false | true | ⚠️ Conflict - manual review |
| 5 | false | false | ✓ Keep both (consistent - private) |
| 6 | false | missing | → Add `public-access:: false` to block |
| 7 | missing | true | → Add `public:: true` to page (optional) |
| 8 | missing | false | ✓ Default private |
| 9 | missing | missing | → Add `public-access:: false` (default private) |

**Default Policy:**
- New files: `public-access:: false` (private by default)
- Migration: If either property is true, set both to true
- Conflicts: Manual review required

---

## 5. Migration Phases

### Phase 1: Critical Fixes (Weeks 1-2)

**Objective:** Fix structural issues that break semantic integrity

**Tasks:**

1. **Namespace Corrections**
   - Fix 100 robotics files: `mv:rb0010aerialrobot` → `rb:AerialRobot`
   - Standardize class names to PascalCase across all domains
   - Update OWL axioms with correct namespaces

2. **Term-ID Standardization**
   - Convert lowercase `rb-NNNN` → uppercase `RB-NNNN` in term-id properties
   - Create term-id registry with all 1,521 existing IDs
   - Detect and resolve any duplicate term-ids

3. **OntologyBlock Position**
   - Move OntologyBlock to top of file (first element)
   - Remove duplicate OntologyBlocks (keep first, archive others)
   - Ensure `id::` property is present

4. **Structure Standardization**
   - Add missing **Identification**, **Definition**, **Semantic Classification** sections
   - Standardize property indentation (2 spaces, not tabs)
   - Move external Relationships sections into OntologyBlock

**Deliverables:**
- [ ] Term-ID registry (JSON)
- [ ] Migration script: `fix-namespaces.js`
- [ ] Migration script: `fix-term-ids.js`
- [ ] Migration script: `normalize-ontology-blocks.js`
- [ ] Validation report: Phase 1 issues fixed

**Success Criteria:**
- 100% of OntologyBlocks use correct domain namespace
- 100% of term-ids follow uppercase convention
- 100% of OntologyBlocks are first in file
- 0 duplicate term-ids

### Phase 2: Metadata Enrichment (Weeks 3-4)

**Objective:** Ensure all files have required properties

**Tasks:**

1. **Required Property Addition**
   - Add missing `ontology:: true`
   - Add missing `term-id::` (auto-assign from registry)
   - Add missing `preferred-term::` (derive from filename or page title)
   - Add missing `definition::` (extract from content or flag for manual)
   - Add missing `source-domain::` (infer from term-id prefix)
   - Add missing `status::` (default: draft)
   - Add missing `owl:class::` (generate from term-id and preferred-term)
   - Add missing `is-subclass-of::` (flag for manual review)

2. **Recommended Property Addition**
   - Add `public-access::` based on public:: or default false
   - Add `version::` (default: 1.0.0 for complete, 0.1.0 for draft)
   - Add `last-updated::` (use git commit date or current date)
   - Add `maturity::` (infer from status or content quality)
   - Add `authority-score::` (infer from source quality or default 0.7)
   - Add `owl:physicality::` (infer from context or default VirtualEntity)
   - Add `owl:role::` (infer from context or default Concept)

3. **Public Property Harmonization**
   - Sync page-level `public::` with block-level `public-access::`
   - Resolve conflicts (manual review queue)
   - Default to private unless explicitly public

4. **Domain Assignment**
   - Add `belongsToDomain::` based on source-domain mapping
   - Identify cross-domain concepts (flag for CrossDomainBridges)

**Deliverables:**
- [ ] Migration script: `enrich-metadata.js`
- [ ] Inference rules configuration
- [ ] Manual review queue (CSV)
- [ ] Validation report: Property completeness metrics

**Success Criteria:**
- 100% of files have Tier 1 required properties
- 90%+ of files have Tier 2 recommended properties
- Manual review queue size < 50 files

### Phase 3: IRI Standardization & OWL Enhancement (Weeks 5-6)

**Objective:** Formalize semantic structure with OWL axioms

**Tasks:**

1. **IRI Assignment**
   - Generate canonical IRI for each concept: `http://ontology.logseq.io/{domain}#{TERM-ID}`
   - Add IRI to term-id registry
   - Update OWL axioms to use canonical IRIs

2. **OWL Axiom Generation**
   - Generate basic OWL axioms for all files:
     - Declaration(Class)
     - SubClassOf (from is-subclass-of::)
     - AnnotationAssertion (rdfs:label from preferred-term)
     - AnnotationAssertion (rdfs:comment from definition)
   - Add OWL Restrictions for files with structured relationships
   - Include property characteristics (Transitive, Asymmetric, etc.)

3. **Relationship Formalization**
   - Convert Logseq relationships to OWL properties:
     - `is-subclass-of::` → `SubClassOf`
     - `has-part::` → `ObjectSomeValuesFrom(hasPart ...)`
     - `requires::` → `ObjectSomeValuesFrom(requires ...)`
   - Define inverse properties (hasPart ↔ isPartOf)

4. **CrossDomainBridges Enhancement**
   - Identify concepts linked across domains (AI+Blockchain, Robotics+Metaverse)
   - Add CrossDomainBridges section with explicit via relationships
   - Generate owl:sameAs or owl:equivalentClass where appropriate

**Deliverables:**
- [ ] IRI registry (part of term-id registry)
- [ ] Migration script: `generate-owl-axioms.js`
- [ ] OWL property definitions file
- [ ] Full ontology export: `ontology-logseq-full.ttl` (Turtle format)
- [ ] Validation: Load in Protégé, run reasoner

**Success Criteria:**
- 100% of concepts have canonical IRI
- 90%+ of concepts have basic OWL axioms
- Ontology passes consistency check in Pellet/HermiT reasoner
- 0 unsatisfiable classes

### Phase 4: Quality Enhancement & Validation (Weeks 7-8)

**Objective:** Improve content quality and validate entire ontology

**Tasks:**

1. **Definition Quality**
   - Review definitions <50 characters (flag as too brief)
   - Enhance definitions with [[concept links]]
   - Add scope-notes for ambiguous concepts
   - Improve academic tone and clarity

2. **Source Attribution**
   - Add authoritative sources where missing
   - Convert bare URLs to [[Page Links]] or proper references
   - Calculate authority-score based on source quality
   - Add References section to files missing it

3. **Cross-Domain Integration**
   - Review and enhance CrossDomainBridges
   - Ensure multi-domain concepts are properly classified
   - Add implementedInLayer where relevant

4. **UK Context Addition**
   - Add UK Context sections to relevant concepts (60% target)
   - Include North England innovation hubs where applicable
   - Use UK English spelling throughout

5. **Validation Suite**
   - Run OWL reasoner (Pellet/HermiT) for consistency
   - Check for orphaned concepts (no parent)
   - Validate property domain/range constraints
   - Generate completeness metrics report

6. **Documentation Generation**
   - Generate ontology catalog (HTML/PDF)
   - Create visual hierarchy diagrams
   - Export to WebVOWL for visualization
   - Publish API documentation (SPARQL endpoint)

**Deliverables:**
- [ ] Quality audit report (definitions, sources, completeness)
- [ ] Migration script: `enhance-quality.js`
- [ ] Validation script: `validate-ontology.js`
- [ ] Final ontology exports:
  - `ontology-logseq-full.ttl` (Turtle)
  - `ontology-logseq-full.owl` (OWL/XML)
  - `ontology-logseq-full.ofn` (OWL Functional Syntax)
  - `ontology-logseq-full.jsonld` (JSON-LD)
- [ ] Ontology catalog (HTML)
- [ ] WebVOWL visualization

**Success Criteria:**
- Quality score: Average >0.85 across all concepts
- Authority score: Average >0.80 across all concepts
- Completeness: 95%+ Tier 1, 90%+ Tier 2 properties
- Consistency: 0 errors from reasoner
- Orphans: <1% of concepts have no parent
- Documentation: Comprehensive catalog published

---

## 6. Tool Update Requirements

### 6.1 Migration Scripts

**Required Scripts:**

1. **`validate-term-ids.js`**
   - Scan all files for term-id properties
   - Build term-id registry
   - Detect duplicates, missing IDs, format errors
   - Output: term-id-registry.json + validation report

2. **`fix-namespaces.js`**
   - Find all `owl:class::` properties
   - Correct namespace (mv: → rb: for robotics)
   - Standardize class names (PascalCase)
   - Update OWL axioms

3. **`fix-term-ids.js`**
   - Convert lowercase rb-NNNN → RB-NNNN
   - Ensure consistent format DOMAIN-NNNN
   - Update references in is-subclass-of, relationships

4. **`normalize-ontology-blocks.js`**
   - Move OntologyBlock to top of file
   - Remove duplicate OntologyBlocks (archive extras)
   - Standardize section structure (Identification, Definition, etc.)
   - Fix indentation (tabs → 2 spaces)
   - Move external Relationships into block

5. **`enrich-metadata.js`**
   - Add missing required properties
   - Infer values (source-domain from term-id, etc.)
   - Add recommended properties with defaults
   - Sync public:: and public-access::

6. **`generate-owl-axioms.js`**
   - Create OWL Axioms section for each file
   - Generate basic axioms (Declaration, SubClassOf, Annotations)
   - Convert relationships to OWL restrictions
   - Use canonical IRIs

7. **`enhance-quality.js`**
   - Flag short definitions
   - Check for missing sources
   - Validate UK English spelling
   - Add UK Context templates

8. **`validate-ontology.js`**
   - Export full ontology to Turtle
   - Run OWL reasoner (via OWL API or Robot tool)
   - Check consistency, satisfiability
   - Generate validation report

9. **`export-ontology.js`**
   - Export to multiple formats (TTL, OWL/XML, OFN, JSON-LD)
   - Generate term-id → IRI mappings
   - Create SPARQL endpoint data
   - Build ontology catalog

### 6.2 Validation Tools

**Reasoner Integration:**
- Install: OWL API (Java), Robot tool (CLI), or Protégé (GUI)
- Configure: Run HermiT or Pellet reasoner
- Script: Automated consistency checking in CI/CD

**Quality Metrics:**
- Definition length distribution
- Property completeness percentage
- Source attribution coverage
- Orphaned concepts count
- Cross-domain bridge density

### 6.3 CI/CD Integration

**Pre-commit Hooks:**
```bash
# Validate term-id uniqueness
npm run validate-term-ids

# Check OntologyBlock structure
npm run validate-structure

# Spell check (UK English)
npm run spell-check
```

**GitHub Actions:**
```yaml
name: Ontology Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: npm install
      - name: Validate term-ids
        run: npm run validate-term-ids
      - name: Check OWL consistency
        run: npm run validate-ontology
      - name: Generate reports
        run: npm run generate-reports
      - name: Upload artifacts
        uses: actions/upload-artifact@v2
        with:
          name: validation-reports
          path: output/reports/
```

---

## 7. Testing Strategy

### 7.1 Unit Testing

**Test Each Script:**

```javascript
// Example: test fix-namespaces.js

describe('fix-namespaces', () => {
  it('converts mv: to rb: for robotics files', () => {
    const input = '- owl:class:: mv:rb0010aerialrobot';
    const output = fixNamespaces(input, 'robotics');
    expect(output).toBe('- owl:class:: rb:AerialRobot');
  });

  it('preserves correct namespaces', () => {
    const input = '- owl:class:: ai:AIAgentSystem';
    const output = fixNamespaces(input, 'ai');
    expect(output).toBe('- owl:class:: ai:AIAgentSystem');
  });

  it('handles multiple owl:class properties', () => {
    // Test multiple classes in hierarchy
  });
});
```

**Test Coverage:**
- Namespace conversion
- Class name PascalCase conversion
- Term-ID format standardization
- OntologyBlock position detection
- Property inference logic
- OWL axiom generation

### 7.2 Integration Testing

**Test Migration Pipeline:**

```bash
# Create test corpus (10 sample files)
npm run create-test-corpus

# Run Phase 1 migration
npm run migrate:phase1 -- --test-mode

# Validate Phase 1 output
npm run validate:phase1

# Run Phase 2 migration
npm run migrate:phase2 -- --test-mode

# Validate Phase 2 output
npm run validate:phase2

# Continue through all phases
```

**Validation Checks:**
- File count unchanged (no files lost)
- Git diff review (changes are expected)
- Markdown structure valid (no broken syntax)
- Logseq can parse updated files
- term-id uniqueness maintained

### 7.3 Acceptance Testing

**Manual Review:**

1. Sample 20 files across domains (4 per domain)
2. Verify OntologyBlock structure matches template
3. Check namespace correctness
4. Validate definition quality
5. Confirm OWL axioms are valid

**Reasoner Testing:**

1. Export full ontology to OWL format
2. Load in Protégé ontology editor
3. Run HermiT reasoner
4. Verify: 0 inconsistencies, 0 unsatisfiable classes
5. Check inferred hierarchy matches expectations

**User Acceptance:**

1. Import updated files into fresh Logseq graph
2. Test queries (find AI concepts, filter by maturity, etc.)
3. Verify backlinks work correctly
4. Check graph visualization
5. Confirm no broken page links

---

## 8. Rollout Plan

### 8.1 Pre-Migration

**Week 0: Preparation**

- [ ] Git branch: `feature/ontology-standardization`
- [ ] Backup entire mainKnowledgeGraph directory
- [ ] Install required tools (Node.js, OWL API, Robot)
- [ ] Create test corpus (50 representative files)
- [ ] Set up validation pipeline
- [ ] Communication: Notify team of upcoming changes

### 8.2 Phased Rollout

**Week 1-2: Phase 1 (Critical Fixes)**
- Run scripts on test corpus, validate, fix issues
- Run scripts on full corpus in test mode
- Code review of changes (git diff)
- Commit Phase 1 changes to branch
- Create PR with Phase 1 results

**Week 3-4: Phase 2 (Metadata Enrichment)**
- Review manual review queue from Phase 1
- Complete required manual tasks
- Run Phase 2 scripts on full corpus
- Validate metadata completeness
- Commit Phase 2 changes
- Merge to main branch (if Phase 1 approved)

**Week 5-6: Phase 3 (IRI & OWL)**
- Generate term-id registry
- Assign canonical IRIs
- Generate OWL axioms
- Export ontology files
- Run reasoner validation
- Commit Phase 3 changes

**Week 7-8: Phase 4 (Quality & Validation)**
- Enhance definitions and sources
- Add UK Context sections
- Final validation suite
- Generate documentation
- Publish ontology catalog
- Final PR and merge

### 8.3 Post-Migration

**Week 9: Stabilization**
- Monitor for issues (broken links, parsing errors)
- Address any user-reported problems
- Update documentation
- Tag release: `v2.0.0-standardized`

**Week 10+: Continuous Improvement**
- Maintain term-id registry for new files
- Run validation in CI/CD
- Periodic quality audits
- User training on new structure

---

## 9. Risk Assessment & Mitigation

### 9.1 Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Data loss during migration | Low | Critical | Git version control, backups, dry-run testing |
| Broken page links | Medium | High | Preserve filenames (no renames), validate links post-migration |
| OWL inconsistencies | Medium | Medium | Reasoner validation, manual review of key concepts |
| Term-ID collisions | Low | High | Registry system, validation scripts |
| User workflow disruption | Medium | Medium | Preserve backward compatibility, gradual rollout |
| Performance degradation | Low | Low | Logseq handles 1,681 files easily, no performance impact expected |
| Manual review bottleneck | High | Medium | Prioritize, parallelize, accept 90% automation (not 100%) |

### 9.2 Rollback Plan

**If Critical Issues Arise:**

1. **Stop migration** - halt running scripts
2. **Assess impact** - how many files affected?
3. **Git revert** - revert to pre-migration commit
4. **Fix scripts** - debug and repair migration logic
5. **Re-test** - validate on test corpus again
6. **Resume** - restart migration from fixed point

**Rollback Command:**
```bash
git checkout main
git reset --hard <pre-migration-commit-hash>
```

---

## 10. Success Metrics

### 10.1 Quantitative Metrics

**Target Achievement:**

| Metric | Current | Target | Success |
|--------|---------|--------|---------|
| OntologyBlocks first in file | ~60% | 100% | ✓ if 100% |
| Correct namespace usage | ~85% | 100% | ✓ if 100% |
| Term-ID format compliance | ~92% | 100% | ✓ if 100% |
| Required properties (Tier 1) | ~70% | 100% | ✓ if ≥95% |
| Recommended properties (Tier 2) | ~50% | 90% | ✓ if ≥85% |
| OWL axioms present | ~30% | 70% | ✓ if ≥60% |
| Reasoner consistency check | N/A | Pass | ✓ if 0 errors |
| Quality score average | ~0.75 | 0.85 | ✓ if ≥0.82 |
| Authority score average | ~0.80 | 0.85 | ✓ if ≥0.83 |

### 10.2 Qualitative Metrics

**User Satisfaction:**
- [ ] Team finds new structure intuitive
- [ ] Logseq queries work as expected
- [ ] Documentation is clear and helpful
- [ ] No major complaints about changes

**Ontology Quality:**
- [ ] Domain experts validate correctness
- [ ] Cross-domain relationships are meaningful
- [ ] Definitions are comprehensive and clear
- [ ] OWL exports are usable in other tools (Protégé, GraphDB)

---

## 11. Conclusion

This standardization strategy provides a comprehensive, phased approach to transforming the Logseq multi-ontology framework into a production-ready, semantically rigorous knowledge base. By preserving existing filenames, ensuring IRI uniqueness through term-ids, and gradually enhancing metadata and OWL axioms, we minimize disruption while maximizing semantic value.

**Key Takeaways:**

1. **Preserve, Don't Disrupt** - Keep 80% of filenames unchanged
2. **Uniqueness via term-id** - Stable identifiers independent of filenames
3. **Gradual Enhancement** - 4 phases over 8 weeks, each building on previous
4. **Validation First** - Test corpus, reasoner checks, manual review
5. **Quality Over Speed** - Accept 90% automation, manual review for 10%

**Next Steps:**

1. Review and approve this strategy
2. Set up development environment and test corpus
3. Begin Phase 1 implementation (Critical Fixes)
4. Iterate based on validation results

---

**Document Version:** 1.0  
**Author:** Ontology Standardization Team  
**Review Date:** 2025-11-21  
**Next Review:** After Phase 1 completion  
**Status:** **APPROVED - Ready for Implementation**
