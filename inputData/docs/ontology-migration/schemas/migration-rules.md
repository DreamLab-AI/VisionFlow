# Ontology Block Migration Rules

**Version:** 1.0.0
**Date:** 2025-11-21
**Status:** Authoritative Specification
**Purpose:** Transformation rules for migrating existing ontology blocks to canonical format

---

## Executive Summary

This document provides precise transformation rules for migrating 1,709 markdown files from 6 existing patterns to the canonical ontology block schema. It addresses critical issues including namespace corrections, class naming standardization, structural reorganization, and property mapping.

**Migration Scope:**
- **Pattern 1** (Comprehensive Structured): Minor adjustments, 85% compliant
- **Pattern 2** (Blockchain with OWL Axioms): Minimal changes, 90% compliant
- **Pattern 3** (Robotics Simplified): Major namespace fix, 60% compliant
- **Pattern 4** (Logseq Minimal): Significant enhancement needed, 40% compliant
- **Pattern 5** (Metaverse Flat): Structure reorganization, 70% compliant
- **Pattern 6** (Extended Metadata): Merge with Pattern 1, 80% compliant

**Critical Fixes:**
1. **Robotics namespace**: `mv:rb*` → `rb:*` (affects 100% of robotics files)
2. **Class naming**: `rb0010aerialrobot` → `AerialRobot` (affects 80% of robotics files)
3. **Status/maturity**: Separate properties (affects 50% of robotics files)
4. **Duplicate sections**: Remove Technical Details duplicates (affects 20% of files)

---

## Priority 1: Critical Namespace Fixes

### Rule 1.1: Robotics Namespace Correction

**Issue:** All RB-* files incorrectly use `mv:` namespace instead of `rb:`

**Detection Pattern:**
```markdown
owl:class:: mv:rb[0-9]+.*
```

**Transformation:**
```markdown
# BEFORE (WRONG)
owl:class:: mv:rb0010aerialrobot

# AFTER (CORRECT)
owl:class:: rb:AerialRobot
```

**Automated Migration Script:**
```bash
# Find all files with robotics namespace error
grep -l "owl:class:: mv:rb" mainKnowledgeGraph/pages/rb-*.md

# Transform each file
sed -i 's/owl:class:: mv:rb[0-9]\+/owl:class:: rb:/g' file.md
```

**Manual Review Required:**
- Verify class name capitalization (see Rule 2.1)
- Check OWL axioms for consistency
- Update any references in other files

**Affected Files:** ~80 files (all RB-XXXX robotics entries)

**Priority:** CRITICAL - Must complete before any other migrations

---

## Priority 2: Class Naming Standardization

### Rule 2.1: Convert to PascalCase

**Issue:** Robotics files use lowercase concatenation; should be PascalCase

**Detection Pattern:**
```markdown
owl:class:: [a-z]+:[a-z0-9]+
```

**Transformation Algorithm:**
1. Remove term-id prefix from class name
2. Split on whitespace and hyphens
3. Capitalize first letter of each word
4. Remove spaces and hyphens

**Examples:**

| Original | Canonical |
|----------|-----------|
| `mv:rb0010aerialrobot` | `rb:AerialRobot` |
| `mv:rb0020swarmrobot` | `rb:SwarmRobot` |
| `bc:consensus-mechanism` | `bc:ConsensusMechanism` |
| `ai:large-language-model` | `ai:LargeLanguageModel` |

**Automated Transformation:**
```python
def to_pascal_case(term_id, preferred_term):
    """Convert preferred term to PascalCase class name."""
    # Remove punctuation, split on whitespace
    words = preferred_term.replace('-', ' ').split()
    # Capitalize each word
    pascal = ''.join(word.capitalize() for word in words)
    return pascal

# Example
to_pascal_case("RB-0010", "Aerial Robot")  # → "AerialRobot"
```

**Manual Review Required:**
- Acronyms: "AI" → "AI", not "Ai"
- Proper nouns: "Bitcoin" → "Bitcoin"
- Compound words: "Blockchain" vs "Block Chain"

**Affected Files:** ~100 files (mainly robotics domain)

**Priority:** CRITICAL

---

## Priority 3: Status and Maturity Separation

### Rule 3.1: Distinguish Status from Maturity

**Issue:** Robotics files have `status:: draft` and `maturity:: draft` for mature concepts

**Detection Pattern:**
```markdown
status:: draft
maturity:: draft
source:: \[\[ISO.*\]\]
```
(If authoritative source exists, concept is likely mature)

**Transformation Decision Tree:**

```
IF source includes standards (ISO, IEEE, NIST)
  THEN maturity = "mature" OR "established"
  AND status = evaluate_editorial_status()

IF source includes recent research (2023-2025)
  THEN maturity = "emerging" OR "mature"
  AND status = evaluate_editorial_status()

IF no authoritative sources
  THEN maturity = "draft" OR "emerging"
  AND status = "draft" OR "in-progress"
```

**Editorial Status Evaluation:**
```
IF comprehensive definition (>100 chars) AND relationships present AND sources cited
  THEN status = "complete"
ELSE IF definition exists AND basic properties present
  THEN status = "in-progress"
ELSE
  THEN status = "draft"
```

**Example Transformations:**

```markdown
# BEFORE
term-id:: RB-0010
preferred-term:: Aerial Robot
status:: draft
maturity:: draft
source:: [[ISO 8373:2021]]

# AFTER
term-id:: RB-0010
preferred-term:: Aerial Robot
status:: complete
maturity:: mature
source:: [[ISO 8373:2021]]
```

**Affected Files:** ~50 files (robotics domain overrepresented)

**Priority:** HIGH

---

## Priority 4: Structural Standardization

### Rule 4.1: Migrate Flat Structure to Sectioned

**Issue:** Metaverse files (Pattern 5) use flat property lists without **bold headers**

**Detection Pattern:**
```markdown
- ### OntologyBlock
  - ontology:: true
  - term-id::
  - preferred-term::
  [no **Identification** header]
```

**Transformation:**
```markdown
# BEFORE
- ### OntologyBlock
  - ontology:: true
  - term-id:: 20150
  - preferred-term:: Game Engine
  - definition:: ...
  - owl:class:: mv:GameEngine

# AFTER
- ### OntologyBlock
  id:: game-engine-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: 20150
    - preferred-term:: Game Engine
    - source-domain:: metaverse
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-21

  - **Definition**
    - definition:: ...
    - maturity:: mature
    - source:: ...

  - **Semantic Classification**
    - owl:class:: mv:GameEngine
    - owl:physicality:: VirtualEntity
    - owl:role:: Object
```

**Section Organization:**
1. Block header with `id::` and `collapsed:: true`
2. **Identification** section
3. **Definition** section
4. **Semantic Classification** section
5. **#### Relationships** subsection
6. **#### CrossDomainBridges** subsection (if applicable)
7. **#### OWL Axioms** subsection (if applicable)

**Affected Files:** ~50 files (metaverse flat pattern)

**Priority:** MEDIUM

---

### Rule 4.2: Move Relationships into OntologyBlock

**Issue:** Some files have `### Relationships` section outside OntologyBlock

**Detection Pattern:**
```markdown
- ### OntologyBlock
  [properties]

### Relationships
- is-subclass-of::
```

**Transformation:**
```markdown
# BEFORE
- ### OntologyBlock
  - owl:class:: ai:MachineLearning

### Relationships
- is-subclass-of:: [[Artificial Intelligence]]

# AFTER
- ### OntologyBlock
  - **Semantic Classification**
    - owl:class:: ai:MachineLearning

  - #### Relationships
    id:: machine-learning-relationships
    - is-subclass-of:: [[Artificial Intelligence]]
```

**Rules:**
- Move all relationship declarations inside OntologyBlock
- Create #### Relationships subsection if not present
- Add `id::` property to Relationships subsection
- Delete external ### Relationships section

**Affected Files:** ~30 files (older AI and general concepts)

**Priority:** MEDIUM

---

### Rule 4.3: Remove Duplicate Metadata Sections

**Issue:** Some files have duplicate "## Technical Details" or "## Metadata" sections

**Detection Pattern:**
```markdown
- ### OntologyBlock
  - term-id:: AI-0850

## Technical Details
- **Id**: ai-0850
- **Term ID**: AI-0850
```

**Transformation:**
- If duplicate section repeats OntologyBlock properties: DELETE
- If duplicate section adds new information: MERGE into OntologyBlock
- Keep only footer "## Metadata" section for provenance (Last Updated, Review Status, Curator)

**Example:**
```markdown
# BEFORE
- ### OntologyBlock
  - term-id:: AI-0850
  - status:: complete

## Technical Details
- **Term ID**: AI-0850
- **Status**: complete
[duplicate information]

# AFTER
- ### OntologyBlock
  - term-id:: AI-0850
  - status:: complete
[no duplicate section]

## Metadata
- **Last Updated**: 2025-11-21
- **Review Status**: Complete
[keep provenance-only footer]
```

**Affected Files:** ~40 files (Pattern 4 and older entries)

**Priority:** MEDIUM

---

### Rule 4.4: Standardize Indentation

**Issue:** Robotics files use tabs; should use 2 spaces

**Detection Pattern:**
```markdown
- ### OntologyBlock
\t- ontology:: true
```

**Transformation:**
```bash
# Convert tabs to 2 spaces
sed -i 's/\t/  /g' file.md
```

**Rules:**
- 2 spaces per indentation level
- Maintain Logseq bullet hierarchy
- No mixing of tabs and spaces

**Affected Files:** ~80 files (all robotics)

**Priority:** LOW (cosmetic, but improves consistency)

---

## Priority 5: Property Additions and Enhancements

### Rule 5.1: Add Missing Required Properties

**Required properties (Tier 1) that MUST be present:**

1. `ontology:: true`
2. `term-id:: [PREFIX-NNNN]`
3. `preferred-term:: [Name]`
4. `source-domain:: [domain]`
5. `status:: [draft|in-progress|complete|deprecated]`
6. `public-access:: [true|false]`
7. `last-updated:: [YYYY-MM-DD]`
8. `definition:: [text]`
9. `owl:class:: [namespace:ClassName]`
10. `owl:physicality:: [PhysicalEntity|VirtualEntity|AbstractEntity|HybridEntity]`
11. `owl:role:: [Object|Process|Agent|Quality|Relation|Concept]`
12. `is-subclass-of:: [[Parent]]`

**Missing Property Defaults:**

| Property | Default Value | Derivation |
|----------|---------------|------------|
| `status` | `"in-progress"` | If definition exists, else `"draft"` |
| `public-access` | `true` | Default assumption |
| `last-updated` | `[current date]` | Today's date |
| `source-domain` | Infer from term-id prefix | AI→ai, BC→blockchain, RB→robotics |
| `owl:physicality` | Infer from domain | See domain conventions |
| `owl:role` | `"Concept"` | Conservative default |

**Automated Gap Detection:**
```python
REQUIRED_PROPERTIES = [
    "ontology", "term-id", "preferred-term", "source-domain",
    "status", "public-access", "last-updated", "definition",
    "owl:class", "owl:physicality", "owl:role", "is-subclass-of"
]

def detect_missing_properties(block):
    missing = []
    for prop in REQUIRED_PROPERTIES:
        if prop not in block:
            missing.append(prop)
    return missing
```

**Affected Files:** ~200 files (Pattern 4 minimal blocks)

**Priority:** HIGH

---

### Rule 5.2: Add Recommended Properties (Tier 2)

**Recommended additions for quality concepts:**

1. `version:: 1.0.0` (start with 1.0.0 for complete entries)
2. `maturity:: [draft|emerging|mature|established]`
3. `source:: [[Source 1]], [[Source 2]]`
4. `authority-score:: [0.0-1.0]`
5. `quality-score:: [0.0-1.0]`
6. `belongsToDomain:: [[Domain]]`

**Quality Score Calculation:**
```python
def calculate_quality_score(block):
    score = 0.0

    # Definition quality (0-0.3)
    if len(block.get('definition', '')) > 200:
        score += 0.3
    elif len(block.get('definition', '')) > 100:
        score += 0.2

    # Sources present (0-0.2)
    if block.get('source'):
        score += 0.2

    # Relationships (0-0.2)
    if len(block.get('relationships', [])) >= 5:
        score += 0.2
    elif len(block.get('relationships', [])) >= 2:
        score += 0.1

    # OWL axioms (0-0.2)
    if block.get('owl-axioms'):
        score += 0.2

    # Completeness (0-0.1)
    if all_required_present(block):
        score += 0.1

    return round(score, 2)
```

**Authority Score Guidelines:**
```
1.0  - International standards (ISO, IEEE, NIST)
0.95 - Official specifications (W3C, IETF)
0.90 - Academic consensus (multiple peer-reviewed papers)
0.85 - Industry standards (Microsoft, Google, etc.)
0.80 - Well-established community definitions
0.70 - Emerging research concepts
0.60 - Experimental or speculative
```

**Affected Files:** ~500 files (all files should have Tier 2)

**Priority:** MEDIUM

---

### Rule 5.3: Populate OWL Axioms for Core Concepts

**Criteria for OWL Axiom Addition:**

**INCLUDE axioms if ANY:**
- File has term-id with domain prefix (AI-, BC-, RB-)
- Status is "complete" AND maturity is "mature" or "established"
- Authority-score >= 0.90
- Concept is referenced by 10+ other files (backlink count)

**OMIT axioms if ANY:**
- Status is "draft"
- No authoritative sources cited
- Concept is purely descriptive (no formal relationships)

**Minimum Axioms Template:**
```clojure
Prefix(:=<http://narrativegoldmine.com/[domain]#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Declaration(Class(:[ClassName]))
SubClassOf(:[ClassName] :[ParentClass])
AnnotationAssertion(rdfs:label :[ClassName] "[Preferred Term]"@en)
AnnotationAssertion(rdfs:comment :[ClassName] "[Definition]"@en)
```

**Generation Script:**
```python
def generate_minimum_axioms(block):
    namespace = get_namespace(block['owl:class'])
    class_name = get_class_name(block['owl:class'])
    parent_classes = block.get('is-subclass-of', [])

    axioms = f"""Prefix(:=<http://narrativegoldmine.com/{namespace}#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)

Declaration(Class(:{class_name}))
"""

    for parent in parent_classes:
        axioms += f"SubClassOf(:{class_name} :{to_pascal_case(parent)})\n"

    axioms += f"""
AnnotationAssertion(rdfs:label :{class_name} "{block['preferred-term']}"@en)
AnnotationAssertion(rdfs:comment :{class_name} "{block['definition']}"@en)
"""

    return axioms
```

**Affected Files:** ~300 files (core concepts without axioms)

**Priority:** LOW (enhancement, not required)

---

## Property Mapping Tables

### Pattern 1 → Canonical (Comprehensive Structured)

| Pattern 1 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| `ontology` | `ontology` | Direct copy |
| `term-id` | `term-id` | Direct copy |
| `preferred-term` | `preferred-term` | Direct copy |
| `source-domain` | `source-domain` | Direct copy |
| `status` | `status` | Validate values |
| `public-access` | `public-access` | Direct copy |
| `version` | `version` | Direct copy |
| `last-updated` | `last-updated` | Direct copy |
| `quality-score` | `quality-score` | Direct copy |
| `bitcoin-ai-relevance` | [remove] | Domain-specific, obsolete |
| `cross-domain-links` | `cross-domain-links` | Direct copy |
| `definition` | `definition` | Direct copy |
| `maturity` | `maturity` | Direct copy |
| `source` | `source` | Direct copy |
| `authority-score` | `authority-score` | Direct copy |
| `owl:class` | `owl:class` | Fix namespace, CamelCase |
| `owl:physicality` | `owl:physicality` | Direct copy |
| `owl:role` | `owl:role` | Direct copy |
| `owl:inferred-class` | `owl:inferred-class` | Fix namespace, CamelCase |
| `belongsToDomain` | `belongsToDomain` | Direct copy |
| `implementedInLayer` | `implementedInLayer` | Direct copy |
| `is-subclass-of` | `is-subclass-of` | Direct copy |

**Migration Complexity:** LOW (85% compliant)

---

### Pattern 2 → Canonical (Blockchain with OWL Axioms)

| Pattern 2 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| All Pattern 1 properties | (same as Pattern 1) | (same) |
| OWL Axioms block | OWL Axioms block | Verify syntax, add prefixes if missing |
| Relationships subsection | Relationships subsection | Add `id::` if missing |

**Additional Steps:**
- Validate OWL Functional Syntax
- Ensure all namespace prefixes declared
- Check for unsatisfiable classes with reasoner

**Migration Complexity:** LOW (90% compliant)

---

### Pattern 3 → Canonical (Robotics Simplified)

| Pattern 3 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| `ontology` | `ontology` | Direct copy |
| `term-id` | `term-id` | Direct copy |
| `preferred-term` | `preferred-term` | Clean formatting |
| `source-domain` | `source-domain` | Direct copy |
| `status` | `status` | VALIDATE (all "draft" suspect) |
| `definition` (embedded) | `definition` | Extract from embedded text |
| `maturity` | `maturity` | VALIDATE (all "draft" suspect) |
| `owl:class` (mv:*) | `owl:class` | **FIX namespace mv: → rb:** |
| `owl:class` (lowercase) | `owl:class` | **Convert to CamelCase** |
| `owl:physicality` | `owl:physicality` | ConceptualEntity → PhysicalEntity (robots are physical) |
| `owl:role` | `owl:role` | Concept → Agent or Object (validate) |
| `belongsToDomain` | `belongsToDomain` | RoboticsDomain |
| `is-subclass-of` | `is-subclass-of` | Direct copy |
| "## About" section | [restructure] | Move content to main section |
| "### Original Content" | [remove or collapse] | Archive if historical value |

**Critical Transformations:**
1. **Namespace fix**: `mv:rb0010aerialrobot` → `rb:AerialRobot`
2. **Physicality fix**: `ConceptualEntity` → `PhysicalEntity`
3. **Status review**: Evaluate all "draft" statuses
4. **Add missing properties**: version, public-access, last-updated, quality-score

**Migration Complexity:** HIGH (60% compliant, major changes needed)

---

### Pattern 4 → Canonical (Logseq Minimal)

| Pattern 4 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| `term-id` | `term-id` | Direct copy |
| `preferred-term` | `preferred-term` | Direct copy |
| `ontology` | `ontology` | Direct copy |
| `definition` | `definition` | Direct copy |
| `maturity` | `maturity` | Direct copy |
| `source` | `source` | Direct copy |
| `authority-score` | `authority-score` | Direct copy |
| `is-subclass-of` (external) | `is-subclass-of` (in block) | Move into OntologyBlock |
| [missing properties] | [many Tier 1 properties] | **Add defaults** |

**Properties to Add:**
- `source-domain`: Infer from term-id or domain context
- `status`: Default "in-progress"
- `public-access`: Default `true`
- `version`: Default "1.0.0" if complete
- `last-updated`: Current date
- `owl:class`: Generate from namespace + preferred-term
- `owl:physicality`: Infer from domain
- `owl:role`: Default "Concept"
- `quality-score`: Calculate
- `belongsToDomain`: Map from context

**Delete duplicate "## Technical Details" section**

**Migration Complexity:** HIGH (40% compliant, significant enhancement needed)

---

### Pattern 5 → Canonical (Metaverse Flat)

| Pattern 5 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| All flat properties | (organized properties) | Add **bold section headers** |
| `ontology` | `ontology` | Move to **Identification** |
| `term-id` | `term-id` | Move to **Identification** |
| `source-domain` | `source-domain` | Move to **Identification** |
| `status` | `status` | Move to **Identification** |
| `public-access` | `public-access` | Move to **Identification** |
| `preferred-term` | `preferred-term` | Move to **Identification** |
| `definition` | `definition` | Move to **Definition** |
| `maturity` | `maturity` | Move to **Definition** |
| `source` | `source` | Move to **Definition** |
| `owl:class` | `owl:class` | Move to **Semantic Classification** |
| `owl:physicality` | `owl:physicality` | Move to **Semantic Classification** |
| `owl:role` | `owl:role` | Move to **Semantic Classification** |
| `owl:inferred-class` | `owl:inferred-class` | Move to **Semantic Classification** |
| `belongsToDomain` | `belongsToDomain` | Move to **Semantic Classification** |
| `implementedInLayer` | `implementedInLayer` | Move to **Semantic Classification** |
| `#### Relationships` | `#### Relationships` | Keep as subsection |
| `#### OWL Axioms` | `#### OWL Axioms` | Keep as subsection |

**Structural Transformation:**
- Add `id::` to block header
- Add `collapsed:: true` to block header
- Group flat properties into sections
- Maintain subsection structure for Relationships and OWL Axioms

**Migration Complexity:** MEDIUM (70% compliant, structure reorganization)

---

### Pattern 6 → Canonical (Extended Metadata)

| Pattern 6 Property | Canonical Property | Transformation |
|--------------------|-------------------|----------------|
| All Pattern 1 properties | (same as Pattern 1) | (same) |
| `domain-prefix` | [merge into term-id] | Already in term-id (BC-0026) |
| `sequence-number` | [merge into term-id] | Already in term-id (BC-0026) |
| `filename-history` | [optional, keep if useful] | Archive in Metadata section |
| `is-subclass-of` (in Identification) | `is-subclass-of` (in Relationships) | Move to Relationships section |

**Transformation:**
- Pattern 6 is essentially Pattern 1 + extra metadata
- Keep extended metadata in optional Tier 3
- Move `is-subclass-of` from Identification to Relationships section

**Migration Complexity:** LOW (80% compliant, minor reorganization)

---

## Special Cases and Edge Cases

### Case 1: Files with No OntologyBlock

**Detection:** File lacks `### OntologyBlock` header

**Action:**
1. Check if file is documentation/guide (not ontology entry)
   - If yes: Leave unchanged
2. If file should be ontology entry:
   - Create OntologyBlock from scattered properties
   - Generate term-id if missing
   - Add all required Tier 1 properties

**Example:**
```markdown
# BEFORE
# Artificial Intelligence

AI is the simulation of human intelligence...

- is-subclass-of:: [[Computer Science]]

# AFTER
- ### OntologyBlock
  id:: artificial-intelligence-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: [assign new ID]
    - preferred-term:: Artificial Intelligence
    - source-domain:: ai
    - status:: in-progress
    - public-access:: true
    - last-updated:: 2025-11-21

  - **Definition**
    - definition:: [extract from content]

  [... rest of canonical structure]

# Artificial Intelligence

AI is the simulation of human intelligence...
```

---

### Case 2: Multiple OntologyBlocks in One File

**Detection:** Multiple `### OntologyBlock` headers in single file

**Action:**
1. Analyze if blocks represent:
   - **Variants**: Different versions/interpretations of same concept
   - **Related concepts**: Distinct but related concepts
2. If variants: **Merge** into single canonical block
3. If distinct: **Split** into separate files

**Example (merge):**
```markdown
# BEFORE
- ### OntologyBlock (Version 1)
  - term-id:: AI-0100
  - preferred-term:: Machine Learning

- ### OntologyBlock (Version 2)
  - term-id:: AI-0100
  - preferred-term:: Machine Learning
  - [updated properties]

# AFTER
- ### OntologyBlock
  id:: machine-learning-ontology
  collapsed:: true
  - term-id:: AI-0100
  - preferred-term:: Machine Learning
  - version:: 2.0.0
  - changelog:: v2.0.0 - Merged duplicate blocks, updated definition
  [... merged properties, keep most recent/complete]
```

---

### Case 3: Missing Parent Class (is-subclass-of)

**Detection:** No `is-subclass-of` property

**Action:**
1. Analyze term and domain
2. Assign appropriate parent:
   - AI domain: `[[Artificial Intelligence]]` or `[[Machine Learning]]`
   - Blockchain: `[[Blockchain Technology]]`
   - Robotics: `[[Robot]]` or `[[Robotic System]]`
   - Metaverse: `[[Virtual World]]` or `[[Metaverse Technology]]`
   - General: `[[owl:Thing]]` (root class)

**Example:**
```markdown
# BEFORE
- ### OntologyBlock
  - term-id:: AI-0250
  - preferred-term:: Neural Network
  [no is-subclass-of]

# AFTER
- ### OntologyBlock
  - term-id:: AI-0250
  - preferred-term:: Neural Network

  - #### Relationships
    id:: neural-network-relationships
    - is-subclass-of:: [[Machine Learning]], [[Computational Model]]
```

---

### Case 4: Broken Wiki Links

**Detection:** Links to non-existent pages

**Action:**
1. Check for typos or case variations
2. Identify if target should exist:
   - If yes: Create stub page with TODO
   - If no: Fix link or remove
3. Common fixes:
   - `[[machine learning]]` → `[[Machine Learning]]` (capitalize)
   - `[[AI-Agent]]` → `[[AI Agent System]]` (use preferred-term)

---

### Case 5: Empty or Placeholder Sections

**Detection:** Sections with no content or `[TBD]` markers

**Action:**
- **Empty OWL Restrictions**: DELETE section
- **Empty CrossDomainBridges**: DELETE section
- **Empty OWL Axioms**: Keep if `status:: complete`, otherwise DELETE
- **Placeholder definitions**: Set `status:: draft`

---

## Validation After Migration

### Automated Validation Checklist

Run validation script after migration of each file:

```python
def validate_ontology_block(file_path):
    """Validate migrated ontology block against canonical schema."""
    errors = []
    warnings = []

    block = parse_ontology_block(file_path)

    # TIER 1 REQUIRED PROPERTIES
    required = [
        "ontology", "term-id", "preferred-term", "source-domain",
        "status", "public-access", "last-updated", "definition",
        "owl:class", "owl:physicality", "owl:role", "is-subclass-of"
    ]

    for prop in required:
        if prop not in block:
            errors.append(f"Missing required property: {prop}")

    # NAMESPACE CORRECTNESS
    if block.get("source-domain") == "robotics":
        if block.get("owl:class", "").startswith("mv:"):
            errors.append("Robotics namespace error: using mv: instead of rb:")

    # CLASS NAMING CONVENTION
    if not is_pascal_case(get_class_name(block.get("owl:class", ""))):
        warnings.append("Class name not in PascalCase")

    # DATE FORMAT
    if not is_iso_date(block.get("last-updated", "")):
        errors.append("last-updated not in ISO 8601 format (YYYY-MM-DD)")

    # STATUS VALUES
    valid_statuses = ["draft", "in-progress", "complete", "deprecated"]
    if block.get("status") not in valid_statuses:
        errors.append(f"Invalid status value: {block.get('status')}")

    # SECTION STRUCTURE
    expected_sections = ["Identification", "Definition", "Semantic Classification"]
    for section in expected_sections:
        if section not in block.get("sections", []):
            warnings.append(f"Missing recommended section: {section}")

    # RELATIONSHIPS
    if "Relationships" in block.get("sections", []):
        if "id::" not in block.get("relationships_section", ""):
            warnings.append("Relationships section missing id:: property")

    return {
        "file": file_path,
        "errors": errors,
        "warnings": warnings,
        "valid": len(errors) == 0
    }
```

### Manual Review Checklist

For each migrated file, reviewer should verify:

- [ ] **Namespace correct** for domain
- [ ] **Class name** in PascalCase
- [ ] **Definition** comprehensive (2-5 sentences)
- [ ] **Parent class** appropriate and exists
- [ ] **Status/maturity** values make sense
- [ ] **Sources** are authoritative (if cited)
- [ ] **OWL axioms** parse correctly (if present)
- [ ] **Wiki links** resolve correctly
- [ ] **Indentation** consistent (2 spaces)
- [ ] **No duplicate sections**
- [ ] **Relationships** inside OntologyBlock
- [ ] **UK context** present (if applicable)

---

## Migration Workflow

### Phase 1: Critical Fixes (Week 1)

**Day 1-2: Robotics Namespace Fix**
```bash
# 1. Backup all files
cp -r mainKnowledgeGraph/pages mainKnowledgeGraph/pages.backup

# 2. Fix namespace in all RB- files
for file in mainKnowledgeGraph/pages/RB-*.md; do
  sed -i 's/owl:class:: mv:rb/owl:class:: rb:/g' "$file"
  echo "Fixed: $file"
done

# 3. Validate
grep "owl:class:: mv:rb" mainKnowledgeGraph/pages/RB-*.md
# Should return no results
```

**Day 3-4: Class Name Standardization**
```bash
# Run Python script to convert lowercase to PascalCase
python scripts/fix_class_names.py --domain robotics
python scripts/fix_class_names.py --domain blockchain
python scripts/fix_class_names.py --domain ai
```

**Day 5: Status/Maturity Correction**
```bash
# Run script to reassess status and maturity
python scripts/fix_status_maturity.py --domain robotics
```

### Phase 2: Structural Standardization (Week 2)

**Day 1-2: Add Missing Properties**
```bash
python scripts/add_missing_properties.py --tier 1
```

**Day 3: Reorganize Structure**
```bash
python scripts/migrate_flat_to_sectioned.py
python scripts/move_relationships_inside.py
```

**Day 4-5: Remove Duplicates**
```bash
python scripts/remove_duplicate_sections.py
```

### Phase 3: Enhancement (Week 3-4)

**Week 3: Add Tier 2 Properties**
```bash
python scripts/add_recommended_properties.py --tier 2
python scripts/calculate_quality_scores.py
```

**Week 4: Generate OWL Axioms**
```bash
python scripts/generate_owl_axioms.py --core-concepts-only
```

### Phase 4: Validation (Week 5)

**Week 5: Comprehensive Validation**
```bash
# Automated validation
python scripts/validate_all_ontology_blocks.py > validation_report.txt

# Generate compliance report
python scripts/generate_compliance_report.py

# Reasoner validation (if OWL axioms present)
python scripts/owl_reasoner_check.py
```

---

## Migration Scripts

### Script 1: fix_robotics_namespace.py

```python
#!/usr/bin/env python3
"""Fix robotics namespace from mv: to rb: and convert to CamelCase."""

import re
import glob
from pathlib import Path

def to_pascal_case(term):
    """Convert term to PascalCase."""
    # Handle special cases
    term = term.replace('-', ' ')
    words = term.split()
    return ''.join(word.capitalize() for word in words)

def fix_file(file_path):
    """Fix robotics namespace and class naming in a single file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find all mv:rb* class references
    pattern = r'owl:class:: mv:rb\d+([a-z]+(?:[a-z]+)*)'

    def replacer(match):
        lowercase_name = match.group(1)
        # Remove numbering, convert to PascalCase
        pascal_name = to_pascal_case(lowercase_name)
        return f'owl:class:: rb:{pascal_name}'

    new_content = re.sub(pattern, replacer, content)

    # Write back if changed
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False

def main():
    robotics_files = glob.glob('mainKnowledgeGraph/pages/RB-*.md')
    fixed_count = 0

    for file_path in robotics_files:
        if fix_file(file_path):
            fixed_count += 1
            print(f"✓ Fixed: {file_path}")
        else:
            print(f"  Skipped: {file_path} (no changes needed)")

    print(f"\nTotal files fixed: {fixed_count}/{len(robotics_files)}")

if __name__ == '__main__':
    main()
```

### Script 2: validate_ontology_block.py

```python
#!/usr/bin/env python3
"""Validate ontology blocks against canonical schema."""

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ValidationResult:
    file_path: str
    errors: List[str]
    warnings: List[str]

    @property
    def is_valid(self):
        return len(self.errors) == 0

REQUIRED_TIER1 = [
    "ontology", "term-id", "preferred-term", "source-domain",
    "status", "public-access", "last-updated", "definition",
    "owl:class", "owl:physicality", "owl:role", "is-subclass-of"
]

VALID_STATUSES = ["draft", "in-progress", "complete", "deprecated"]
VALID_MATURITIES = ["draft", "emerging", "mature", "established"]
VALID_PHYSICALITIES = ["PhysicalEntity", "VirtualEntity", "AbstractEntity", "HybridEntity"]
VALID_ROLES = ["Object", "Process", "Agent", "Quality", "Relation", "Concept"]

def extract_properties(content: str) -> Dict[str, str]:
    """Extract property:: value pairs from OntologyBlock."""
    properties = {}

    # Match property:: value patterns
    pattern = r'^\s*-\s+([a-z][a-z-]*)::\s*(.+)$'

    for line in content.split('\n'):
        match = re.match(pattern, line)
        if match:
            prop, value = match.groups()
            properties[prop] = value.strip()

    return properties

def validate_file(file_path: str) -> ValidationResult:
    """Validate a single ontology file."""
    errors = []
    warnings = []

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract OntologyBlock
    block_match = re.search(r'### OntologyBlock.*?(?=^##|\Z)', content, re.MULTILINE | re.DOTALL)

    if not block_match:
        errors.append("No OntologyBlock found")
        return ValidationResult(file_path, errors, warnings)

    block_content = block_match.group(0)
    properties = extract_properties(block_content)

    # Check required properties
    for prop in REQUIRED_TIER1:
        if prop not in properties:
            errors.append(f"Missing required property: {prop}")

    # Validate property values
    if properties.get('status') not in VALID_STATUSES:
        errors.append(f"Invalid status: {properties.get('status')}")

    if properties.get('maturity') and properties.get('maturity') not in VALID_MATURITIES:
        warnings.append(f"Invalid maturity: {properties.get('maturity')}")

    if properties.get('owl:physicality') not in VALID_PHYSICALITIES:
        errors.append(f"Invalid owl:physicality: {properties.get('owl:physicality')}")

    if properties.get('owl:role') not in VALID_ROLES:
        errors.append(f"Invalid owl:role: {properties.get('owl:role')}")

    # Check namespace correctness
    source_domain = properties.get('source-domain', '')
    owl_class = properties.get('owl:class', '')

    if source_domain == 'robotics' and owl_class.startswith('mv:'):
        errors.append("Robotics namespace error: using mv: instead of rb:")

    # Check class naming (PascalCase)
    if owl_class:
        class_name = owl_class.split(':')[-1]
        if not class_name[0].isupper() or '_' in class_name or '-' in class_name:
            warnings.append(f"Class name not PascalCase: {class_name}")

    # Check date format
    last_updated = properties.get('last-updated', '')
    if not re.match(r'^\d{4}-\d{2}-\d{2}$', last_updated):
        errors.append(f"Invalid date format for last-updated: {last_updated}")

    return ValidationResult(file_path, errors, warnings)

def main():
    import glob

    all_files = glob.glob('mainKnowledgeGraph/pages/*.md')

    total = len(all_files)
    valid = 0
    errors_count = 0
    warnings_count = 0

    print(f"Validating {total} files...\n")

    for file_path in all_files:
        result = validate_file(file_path)

        if result.is_valid:
            valid += 1
        else:
            print(f"\n❌ {file_path}")
            for error in result.errors:
                print(f"   ERROR: {error}")
                errors_count += 1
            for warning in result.warnings:
                print(f"   WARNING: {warning}")
                warnings_count += 1

    print(f"\n{'='*60}")
    print(f"Validation Complete:")
    print(f"  Total files: {total}")
    print(f"  Valid: {valid} ({valid/total*100:.1f}%)")
    print(f"  Errors: {errors_count}")
    print(f"  Warnings: {warnings_count}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
```

---

## Summary of Transformation Rules

| Priority | Rule | Pattern | Affected Files | Complexity |
|----------|------|---------|----------------|------------|
| 1 | Robotics namespace fix | Pattern 3 | ~80 | CRITICAL |
| 1 | Class name CamelCase | Pattern 3, 5 | ~100 | CRITICAL |
| 1 | Status/maturity separation | Pattern 3 | ~50 | HIGH |
| 2 | Flat → sectioned structure | Pattern 5 | ~50 | MEDIUM |
| 2 | Move relationships inside | Pattern 4 | ~30 | MEDIUM |
| 2 | Remove duplicate sections | Pattern 4 | ~40 | MEDIUM |
| 2 | Tabs → spaces | Pattern 3 | ~80 | LOW |
| 3 | Add missing Tier 1 properties | All patterns | ~200 | HIGH |
| 3 | Add Tier 2 properties | All patterns | ~500 | MEDIUM |
| 4 | Generate OWL axioms | All patterns | ~300 | LOW |

**Total Effort Estimate:**
- Phase 1 (Critical): 1 week
- Phase 2 (Structure): 1 week
- Phase 3 (Enhancement): 2 weeks
- Phase 4 (Validation): 1 week

**Grand Total: 5 weeks for complete migration of 1,709 files**

---

**Document Control:**
- **Version**: 1.0.0
- **Status**: Authoritative
- **Approved By**: Chief Architect
- **Review Date**: 2025-11-21
- **Next Review**: After Phase 1 completion
