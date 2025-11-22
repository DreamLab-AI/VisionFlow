# Ontology Standardization Quick Reference

**Version**: 1.0 | **Date**: 2025-11-21 | **1-Page Cheat Sheet**

---

## Filename Format

**KEEP existing filenames** - Do NOT rename files

```
✓ AI Agent System.md              (natural language - preferred)
✓ rb-0010-aerial-robot.md          (prefixed - acceptable)
✓ Blockchain.md                    (simple name - good)
```

**New files:** Use descriptive names
- Preferred: `Concept Name.md`
- Acceptable: `DOMAIN-NNNN-concept-name.md`

---

## Term-ID Format

**Required in every OntologyBlock:**

```markdown
- term-id:: {DOMAIN}-{NUMBER}
```

**Domain Prefixes:**
- `AI` - Artificial Intelligence
- `BC` - Blockchain & Cryptocurrency
- `RB` - Robotics (**uppercase R**, not rb-)
- `MV` - Metaverse
- `TC` - Telecollaboration
- `DT` - Disruptive Technology

**Examples:**
```markdown
- term-id:: AI-0600
- term-id:: BC-0051
- term-id:: RB-0010    ← Uppercase!
- term-id:: MV-20341
```

**Rules:**
- 4-digit zero-padded: `AI-0001`, `RB-0010`
- OR 5-digit for large domains: `MV-20341`
- Must be UNIQUE (check term-id-registry.json)

---

## IRI Format

**Canonical IRI for each concept:**

```
http://ontology.logseq.io/{domain}#{TERM-ID}
```

**Examples:**
```turtle
http://ontology.logseq.io/ai#AI-0600
http://ontology.logseq.io/blockchain#BC-0051
http://ontology.logseq.io/robotics#RB-0010
http://ontology.logseq.io/metaverse#MV-20341
```

**Namespace Prefixes:**
```clojure
Prefix(ai:=<http://ontology.logseq.io/ai#>)
Prefix(bc:=<http://ontology.logseq.io/blockchain#>)
Prefix(rb:=<http://ontology.logseq.io/robotics#>)
Prefix(mv:=<http://ontology.logseq.io/metaverse#>)
```

---

## OWL Class Names

**Format:** `{namespace}:{PascalCaseName}`

```markdown
✓ rb:AerialRobot               (correct)
✗ mv:rb0010aerialrobot          (wrong namespace + wrong case)

✓ ai:AIAgentSystem             (correct - keep acronyms together)
✓ bc:ConsensusMechanism        (correct - PascalCase)
✓ mv:GameEngine                (correct)
```

**Rules:**
1. Use domain-specific namespace (rb: for robotics, NOT mv:!)
2. PascalCase: capitalize first letter of each word
3. No hyphens, underscores, or numbers (except acronyms)
4. Join multi-word names without separators

---

## Required Properties (Tier 1)

**Every OntologyBlock MUST have:**

```markdown
- ontology:: true
- term-id:: {DOMAIN-NNNN}
- preferred-term:: {Human Readable Name}
- definition:: {Comprehensive definition with [[links]]}
- source-domain:: {domain}
- status:: {draft | in-progress | complete | deprecated}
- owl:class:: {namespace:ClassName}
- is-subclass-of:: [[ParentConcept]]
```

---

## OntologyBlock Template (Minimal)

```markdown
- ### OntologyBlock
  id:: {concept-slug}-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: AI-0600
    - preferred-term:: AI Agent System
    - source-domain:: ai
    - status:: complete
    - public-access:: true
    - version:: 1.0.0
    - last-updated:: 2025-11-21

  - **Definition**
    - definition:: An autonomous software entity that perceives its environment, makes decisions, and takes actions to achieve specific goals.
    - maturity:: mature
    - source:: [[Russell & Norvig AI: A Modern Approach]]
    - authority-score:: 0.95

  - **Semantic Classification**
    - owl:class:: ai:AIAgentSystem
    - owl:physicality:: VirtualEntity
    - owl:role:: Agent
    - belongsToDomain:: [[AI-GroundedDomain]]

  - #### Relationships
    id:: ai-agent-system-relationships
    - is-subclass-of:: [[Artificial Intelligence]]
    - has-part:: [[PerceptionSystem]], [[DecisionEngine]]
    - requires:: [[SensorInput]], [[ActionSpace]]
    - enables:: [[AutonomousOperation]]
```

---

## OWL Axioms Template (Optional)

**Include for key concepts:**

```markdown
  - #### OWL Axioms
    id:: {concept-slug}-owl-axioms
    collapsed:: true
    - ```clojure
      Prefix(ai:=<http://ontology.logseq.io/ai#>)
      Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
      Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
      
      Declaration(Class(ai:AI-0600))
      
      SubClassOf(ai:AI-0600 ai:ArtificialIntelligence)
      
      AnnotationAssertion(rdfs:label ai:AI-0600 "AI Agent System"@en)
      AnnotationAssertion(rdfs:comment ai:AI-0600 
        "An autonomous software entity..."@en)
      
      SubClassOf(ai:AI-0600
        ObjectSomeValuesFrom(ai:requires ai:SensorInput))
      ```
```

---

## Common Commands

**Validate term-ids:**
```bash
npm run validate-term-ids
```

**Fix namespaces:**
```bash
npm run fix-namespaces
```

**Normalize OntologyBlocks:**
```bash
npm run normalize-ontology-blocks
```

**Export ontology:**
```bash
npm run export-ontology
# Outputs: ontology-logseq-full.ttl, .owl, .ofn, .jsonld
```

**Run full validation:**
```bash
npm run validate-ontology
```

---

## Property Value Formats

**Single value:**
```markdown
- status:: complete
- version:: 1.0.0
```

**Logseq link:**
```markdown
- source:: [[Russell & Norvig AI: A Modern Approach]]
- is-subclass-of:: [[ArtificialIntelligence]]
```

**List of links:**
```markdown
- has-part:: [[Component1]], [[Component2]], [[Component3]]
- belongsToDomain:: [[Domain1]], [[Domain2]]
```

**Numeric:**
```markdown
- quality-score:: 0.92
- authority-score:: 0.95
- cross-domain-links:: 47
```

**Boolean:**
```markdown
- ontology:: true
- public-access:: false
```

**Date (ISO 8601):**
```markdown
- last-updated:: 2025-11-21
```

---

## Status Values

**Workflow Status:**
- `draft` - Initial creation, not reviewed
- `in-progress` - Under active development
- `complete` - Reviewed and approved
- `deprecated` - Superseded by another concept

**Content Maturity:**
- `draft` - Preliminary definition
- `emerging` - Newly established concept
- `mature` - Well-established, widely adopted
- `established` - Standard, authoritative definition

---

## Relationship Properties

**Hierarchical:**
```markdown
- is-subclass-of:: [[Parent]]
```

**Compositional:**
```markdown
- has-part:: [[Component]]
- is-part-of:: [[Whole]]
```

**Dependency:**
```markdown
- requires:: [[Requirement]]
- depends-on:: [[Dependency]]
```

**Functional:**
```markdown
- enables:: [[Capability]]
- implements:: [[Interface]]
- uses:: [[Technology]]
```

**Associative:**
```markdown
- relates-to:: [[RelatedConcept]]
```

---

## Migration Phases Checklist

**Phase 1: Critical Fixes**
- [ ] Fix robotics namespaces (mv: → rb:)
- [ ] Standardize term-ids (rb-NNNN → RB-NNNN)
- [ ] Move OntologyBlocks to top of file
- [ ] Remove duplicate blocks

**Phase 2: Metadata Enrichment**
- [ ] Add missing required properties
- [ ] Add recommended properties
- [ ] Sync public:: and public-access::
- [ ] Complete manual review queue

**Phase 3: IRI & OWL**
- [ ] Assign canonical IRIs
- [ ] Generate OWL axioms
- [ ] Export ontology files
- [ ] Validate with reasoner

**Phase 4: Quality & Validation**
- [ ] Enhance definitions
- [ ] Add authoritative sources
- [ ] Add UK Context sections
- [ ] Generate final reports

---

## Quick Validation Checks

**Is your OntologyBlock correct?**

✓ OntologyBlock is FIRST in file (before any other content)  
✓ Only ONE OntologyBlock per file  
✓ Has `ontology:: true`  
✓ Has unique `term-id::` (format: DOMAIN-NNNN)  
✓ Has `preferred-term::`  
✓ Has `definition::` (2+ sentences)  
✓ Has `owl:class::` with correct namespace  
✓ Has `is-subclass-of::` (at least one parent)  
✓ Uses 2-space indentation (not tabs)  
✓ Namespace matches domain (rb: for robotics, not mv:)

---

## Help & Resources

**Documentation:**
- Full Strategy: `/docs/ontology-migration/STANDARDIZATION-STRATEGY.md`
- Analysis: `/docs/ontology-migration/analysis/`

**Tools:**
- Term-ID Registry: `/docs/ontology-migration/term-id-registry.json`
- Migration Scripts: `/scripts/ontology-migration/`
- Validation: `npm run validate-ontology`

**Support:**
- Issues: Check validation reports in `/output/reports/`
- Questions: Refer to full strategy document
- Bugs: Report to development team

---

**Quick Reference Version**: 1.0  
**Last Updated**: 2025-11-21  
**For**: Ontology Standardization Project  
**Print this page for desk reference!**
