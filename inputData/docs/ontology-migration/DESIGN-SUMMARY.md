# Ontology Standardization Design - Executive Summary

**Date**: 2025-11-21  
**Status**: Design Complete  
**Documents Created**:
- `/docs/ontology-migration/STANDARDIZATION-STRATEGY.md` (comprehensive strategy)
- `/docs/ontology-migration/QUICK-REFERENCE.md` (one-page cheat sheet)
- `/docs/ontology-migration/term-id-registry-template.json` (registry template)

---

## Key Decisions

### 1. Filename Standardization: **PRESERVE Natural Names**

**Decision:** Keep existing filenames unchanged, use term-id as canonical identifier

**Rationale:**
- 80% of files (1,345) use natural language names
- Renaming would break existing links and workflows
- term-id already serves as stable unique identifier
- User preference: "prefer not to lose data in filenames"

**Impact:**
- ✓ Zero filename changes required
- ✓ Preserves user workflow
- ✓ Maintains semantic searchability
- → term-id becomes IRI basis (stable, independent of filenames)

**For New Files:**
- Preferred: `Concept Name.md` (natural language)
- Acceptable: `DOMAIN-NNNN-concept-name.md` (prefixed)
- Required: term-id property in OntologyBlock

---

### 2. IRI Architecture: **Term-ID-Based URIs**

**Structure:**
```
http://ontology.logseq.io/{domain}#{TERM-ID}

Examples:
  http://ontology.logseq.io/ai#AI-0600
  http://ontology.logseq.io/robotics#RB-0010
  http://ontology.logseq.io/blockchain#BC-0051
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

**Uniqueness Guarantee:**
- Term-ID registry (JSON file) tracks all assigned IDs
- Validation scripts check for collisions before assignment
- Format: `{DOMAIN}-{NUMBER}` (e.g., AI-0600, RB-0010)
- **Critical Fix:** Convert lowercase `rb-NNNN` → uppercase `RB-NNNN`

**Benefits:**
- ✓ Stable IRIs independent of filenames
- ✓ OWL2 compliant
- ✓ Potentially resolvable (future HTTP endpoint)
- ✓ Clear domain separation

---

### 3. Ontology Block Structure: **Single Canonical Format**

**Requirements:**
1. **MUST be first** in markdown file
2. **Only ONE** OntologyBlock per file
3. **Required properties** (Tier 1): ontology, term-id, preferred-term, definition, owl:class, is-subclass-of
4. **Recommended properties** (Tier 2): public-access, version, maturity, source, authority-score
5. **Optional sections**: OWL Restrictions, CrossDomainBridges, OWL Axioms

**Template:**
```markdown
- ### OntologyBlock
  id:: {concept-slug}-ontology
  collapsed:: true

  - **Identification**
    - ontology:: true
    - term-id:: {DOMAIN-NNNN}
    - preferred-term:: {Name}
    - source-domain:: {domain}
    - status:: {draft|in-progress|complete|deprecated}
    - public-access:: {true|false}
    - version:: {X.Y.Z}
    - last-updated:: {YYYY-MM-DD}

  - **Definition**
    - definition:: {Comprehensive definition with [[links]]}
    - maturity:: {draft|emerging|mature|established}
    - source:: [[Authoritative Source]]
    - authority-score:: {0.0-1.0}

  - **Semantic Classification**
    - owl:class:: {namespace:PascalCaseName}
    - owl:physicality:: {PhysicalEntity|VirtualEntity|AbstractEntity}
    - owl:role:: {Object|Process|Agent|Quality|Relation}
    - belongsToDomain:: [[Domain]]

  - #### Relationships
    - is-subclass-of:: [[Parent]]
    - has-part:: [[Component]]
    - requires:: [[Dependency]]
    - enables:: [[Capability]]

  - #### OWL Axioms (optional, for key concepts)
    - ```clojure
      Declaration(Class({namespace}:{TERM-ID}))
      SubClassOf({namespace}:{TERM-ID} {namespace}:{Parent})
      AnnotationAssertion(rdfs:label ...)
      ```
```

**Edge Case Handling:**
- Multiple OntologyBlocks → Keep first, archive others
- OntologyBlock not first → Move to top
- No OntologyBlock + public::true → Create minimal block
- No OntologyBlock + no public → Skip (not ontology concept)

---

### 4. Public Property: **Dual Support During Transition**

**Current State:**
- Some files: `public:: true` at page level
- Some files: `public-access:: true` in OntologyBlock
- Some files: both
- Some files: neither

**Strategy:**

**Phase 1 (Now):** Support both properties
```markdown
# Page level (frontmatter style)
public:: true

# Block level (in OntologyBlock)
- public-access:: true
```

**Phase 2 (Migration):** Harmonize
- If either is true → set both to true
- Resolve conflicts → manual review queue
- Default: false (private)

**Phase 3 (Future):** Unified standard
- Standard: `public-access:: true` in OntologyBlock only
- Deprecate: `public:: true` (keep for backward compatibility)

---

### 5. Migration Phases: **4-Phase Gradual Approach**

#### Phase 1: Critical Fixes (Weeks 1-2)
**Objective:** Fix structural issues that break semantic integrity

**Tasks:**
- Fix 100 robotics namespace errors: `mv:rb0010aerialrobot` → `rb:AerialRobot`
- Standardize term-ids: `rb-NNNN` → `RB-NNNN` (uppercase)
- Create term-id registry with all 1,521 existing entries
- Move OntologyBlocks to top of file
- Remove duplicate OntologyBlocks
- Standardize indentation (2 spaces, not tabs)

**Deliverables:**
- term-id-registry.json
- Migration scripts: fix-namespaces.js, fix-term-ids.js, normalize-ontology-blocks.js
- Validation report

**Success Criteria:**
- 100% correct namespace usage
- 100% uppercase term-id format
- 100% OntologyBlocks first in file
- 0 duplicate term-ids

---

#### Phase 2: Metadata Enrichment (Weeks 3-4)
**Objective:** Ensure all required properties are present

**Tasks:**
- Add missing Tier 1 properties (ontology, term-id, preferred-term, definition, owl:class, is-subclass-of)
- Add missing Tier 2 properties (public-access, version, maturity, source, authority-score)
- Harmonize public:: and public-access::
- Infer values where possible (source-domain from term-id, etc.)
- Generate manual review queue for missing definitions

**Deliverables:**
- Migration script: enrich-metadata.js
- Manual review queue (CSV with ~50-100 items)
- Property completeness report

**Success Criteria:**
- 100% Tier 1 property coverage
- 90% Tier 2 property coverage
- Manual review queue < 50 items

---

#### Phase 3: IRI Standardization & OWL Enhancement (Weeks 5-6)
**Objective:** Formalize semantic structure with OWL axioms

**Tasks:**
- Assign canonical IRI to each concept
- Generate basic OWL axioms (Declaration, SubClassOf, Annotations)
- Convert Logseq relationships to OWL restrictions
- Add OWL Axioms section to key concepts
- Export full ontology to Turtle, OWL/XML, JSON-LD

**Deliverables:**
- Migration script: generate-owl-axioms.js
- Full ontology exports: .ttl, .owl, .ofn, .jsonld
- Validation: Load in Protégé, run reasoner

**Success Criteria:**
- 100% concepts have canonical IRI
- 70% concepts have OWL axioms
- Ontology passes consistency check (0 errors)
- 0 unsatisfiable classes

---

#### Phase 4: Quality Enhancement & Validation (Weeks 7-8)
**Objective:** Improve content quality and validate entire ontology

**Tasks:**
- Enhance short definitions (<50 chars)
- Add authoritative sources where missing
- Add CrossDomainBridges for cross-domain concepts
- Add UK Context sections (60% coverage target)
- Run full validation suite
- Generate ontology catalog (HTML/PDF)
- Deploy SPARQL endpoint (optional)

**Deliverables:**
- Migration script: enhance-quality.js
- Validation script: validate-ontology.js
- Quality audit report
- Ontology catalog (HTML)
- WebVOWL visualization

**Success Criteria:**
- Average quality-score ≥ 0.85
- Average authority-score ≥ 0.85
- Reasoner validation: 0 errors
- Orphaned concepts < 1%
- Documentation published

---

## Critical Issues Fixed

### Issue 1: Robotics Namespace Misuse
**Problem:** 100 robotics files use `mv:` namespace instead of `rb:`
```markdown
WRONG: - owl:class:: mv:rb0010aerialrobot
RIGHT: - owl:class:: rb:AerialRobot
```
**Fix:** Phase 1 migration script corrects all instances

### Issue 2: Class Name Format
**Problem:** Inconsistent class naming
```markdown
WRONG: rb0010aerialrobot (lowercase, concatenated)
RIGHT: AerialRobot (PascalCase)
```
**Fix:** Phase 1 standardizes to PascalCase

### Issue 3: Term-ID Case Inconsistency
**Problem:** Lowercase vs uppercase prefixes
```markdown
WRONG: term-id:: rb-0010
RIGHT: term-id:: RB-0010
```
**Fix:** Phase 1 converts all to uppercase

### Issue 4: Multiple OntologyBlocks
**Problem:** Some files have 2+ OntologyBlocks
**Fix:** Phase 1 keeps first, archives others

### Issue 5: OntologyBlock Position
**Problem:** OntologyBlock not always first in file
**Fix:** Phase 1 moves to top

---

## Tool Requirements

### Migration Scripts (Node.js)
1. `validate-term-ids.js` - Build registry, detect duplicates
2. `fix-namespaces.js` - Correct rb: namespace, PascalCase
3. `fix-term-ids.js` - Uppercase term-id format
4. `normalize-ontology-blocks.js` - Position, structure, indentation
5. `enrich-metadata.js` - Add missing properties
6. `generate-owl-axioms.js` - Create OWL Axioms sections
7. `enhance-quality.js` - Improve definitions, sources
8. `validate-ontology.js` - Reasoner validation
9. `export-ontology.js` - Multi-format export

### Validation Tools
- OWL Reasoner: Pellet or HermiT (via OWL API or Protégé)
- Markdown validator
- Logseq parser (ensure files load correctly)
- Git diff review

### CI/CD Integration
- Pre-commit hooks: validate term-id uniqueness
- GitHub Actions: run validation suite on push
- Automated reporting

---

## Success Metrics

| Metric | Current | Target | Success Threshold |
|--------|---------|--------|-------------------|
| OntologyBlocks first in file | ~60% | 100% | ≥ 98% |
| Correct namespace usage | ~85% | 100% | 100% |
| Term-ID format compliance | ~92% | 100% | 100% |
| Required properties (Tier 1) | ~70% | 100% | ≥ 95% |
| Recommended properties (Tier 2) | ~50% | 90% | ≥ 85% |
| OWL axioms present | ~30% | 70% | ≥ 60% |
| Reasoner consistency | N/A | Pass | 0 errors |
| Quality score average | ~0.75 | 0.85 | ≥ 0.82 |
| Authority score average | ~0.80 | 0.85 | ≥ 0.83 |

---

## Timeline Summary

**Total Duration:** 8 weeks

- **Weeks 1-2:** Phase 1 (Critical Fixes)
- **Weeks 3-4:** Phase 2 (Metadata Enrichment)
- **Weeks 5-6:** Phase 3 (IRI & OWL)
- **Weeks 7-8:** Phase 4 (Quality & Validation)
- **Week 9:** Stabilization
- **Week 10+:** Continuous improvement

---

## Risk Mitigation

**Backup Strategy:**
- Git version control (all changes committed)
- Full backup before migration
- Dry-run testing on 50-file test corpus

**Rollback Plan:**
- Git revert to pre-migration commit if critical issues
- Fix scripts and re-test before resuming

**Quality Assurance:**
- Test corpus validation before full migration
- Manual review of 20 representative files per phase
- Reasoner validation at Phase 3
- User acceptance testing before final merge

---

## Recommended Actions

### Immediate (This Week)
1. Review and approve this strategy document
2. Set up development environment (Node.js, git branch)
3. Create 50-file test corpus from representative domains
4. Begin implementing Phase 1 scripts

### Short-term (Weeks 1-4)
1. Execute Phase 1 migration (Critical Fixes)
2. Validate Phase 1 results
3. Complete manual review queue from Phase 1
4. Execute Phase 2 migration (Metadata Enrichment)

### Medium-term (Weeks 5-8)
1. Execute Phase 3 migration (IRI & OWL)
2. Export ontology, run reasoner validation
3. Execute Phase 4 migration (Quality Enhancement)
4. Generate final documentation and catalog

### Long-term (Week 9+)
1. Stabilization and bug fixes
2. Integrate validation into CI/CD
3. Maintain term-id registry for new files
4. Periodic quality audits

---

## Documentation Deliverables

**Created:**
1. ✓ **STANDARDIZATION-STRATEGY.md** - Comprehensive 50+ page strategy
2. ✓ **QUICK-REFERENCE.md** - One-page cheat sheet
3. ✓ **term-id-registry-template.json** - Registry structure
4. ✓ **DESIGN-SUMMARY.md** - This executive summary

**To Create (During Migration):**
1. Migration scripts (9 scripts in `/scripts/ontology-migration/`)
2. Validation reports (per phase)
3. Term-ID registry (populated from existing files)
4. Ontology exports (.ttl, .owl, .ofn, .jsonld)
5. Ontology catalog (HTML/PDF)
6. User guide for new ontology creation

---

## Questions Answered

### Q1: Should we standardize filenames with prefixes?
**A:** NO. Keep existing natural language names. 80% of files already use this format, and renaming would cause massive disruption. Use term-id as the stable canonical identifier instead.

### Q2: How do we ensure IRI uniqueness?
**A:** Via term-id registry system. Each concept gets ONE unique term-id (format: DOMAIN-NNNN) which becomes the IRI fragment. Registry tracks all assignments and prevents collisions.

### Q3: What's the canonical ontology block structure?
**A:** Single block, first in file, with three main sections: Identification, Definition, Semantic Classification. Relationships and OWL Axioms as subsections. See template in QUICK-REFERENCE.md.

### Q4: How do we handle files with public:: true but no OntologyBlock?
**A:** Create a minimal OntologyBlock with auto-assigned term-id, inferred properties, and mark status as draft for manual review. See Phase 2 migration.

### Q5: Do we need to rename the 100 rb- prefixed files?
**A:** NO. Keep filenames as-is. Only standardize the term-id PROPERTY from lowercase `rb-NNNN` to uppercase `RB-NNNN`. This preserves links and workflow.

---

## Contact & Next Steps

**For Questions:**
- Refer to STANDARDIZATION-STRATEGY.md for detailed rationale
- Consult QUICK-REFERENCE.md for syntax and formats
- Review existing analysis docs in `/docs/ontology-migration/analysis/`

**To Begin Migration:**
1. Approve this design
2. Set up git branch: `feature/ontology-standardization`
3. Install dependencies: `npm install` (in project root)
4. Run test migration: `npm run migrate:test`
5. Review results and iterate

---

**Design Status:** ✅ **COMPLETE - Ready for Implementation**

**Next Milestone:** Phase 1 Script Development & Testing

**Estimated Completion:** 8 weeks from project start

---

*This summary captures the complete standardization strategy. For implementation details, refer to the full STANDARDIZATION-STRATEGY.md document.*
