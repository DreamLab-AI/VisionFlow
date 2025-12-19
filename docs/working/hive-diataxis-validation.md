---
title: "Diataxis Structure Validation Report"
description: "**Date**: 2025-12-19 **Total Files**: 303 **Active Documentation Files**: 223 (excluding archive/ and working/) **Compliance Score**: 55%"
category: explanation
tags:
  - documentation
updated-date: 2025-12-19
difficulty-level: intermediate
---

# Diataxis Structure Validation Report

**Date**: 2025-12-19
**Total Files**: 303
**Active Documentation Files**: 223 (excluding archive/ and working/)
**Compliance Score**: 55%

---

## Executive Summary

**Status**: ⚠️ **MODERATE COMPLIANCE** (Target: 95%+, Current: 55%)

The documentation corpus demonstrates **55% Diataxis compliance** with the following characteristics:
- **153 files** show content-category mismatches
- **6 files** lack category frontmatter
- Distribution approaches Diataxis targets but requires tutorial expansion

**Key Finding**: "howto" category normalized to "guide" per Diataxis framework (71 files reclassified)

---

## Category Distribution Analysis

### Current Distribution

| Category | Count | Percentage | Target Range | Status |
|----------|-------|------------|--------------|--------|
| **Tutorials** | 5 | 2.24% | 10-15% | ❌ FAIL - Need 22 more tutorials |
| **Guides** | 71 | 31.84% | 25-35% | ✅ PASS |
| **Reference** | 41 | 18.39% | 20-30% | ❌ FAIL |
| **Explanations** | 100 | 44.84% | 30-40% | ⚠️ HIGH - Consider converting some to tutorials |
| **Uncategorized** | 6 | 2.69% | 0% | ✅ GOOD |

### Distribution Visualization

```
Tutorials    [█░░░░░░░░░] 2.24%  (Target: 10-15%)
Guides       [████████░░] 31.84%  (Target: 25-35%)
Reference    [█████░░░░░] 18.39%  (Target: 20-30%)
Explanations [█████████░] 44.84%  (Target: 30-40%)
```

### Diataxis Compliance Breakdown

**Scoring Methodology**:
- Tutorials: 0/25 points (partial credit, below target)
- Guides: 25/25 points (within target)
- Reference: 15/25 points (partial credit, below target)
- Explanations: 15/25 points (above target)

**Current Score**: **55%** / 100%

---

## Structure Validation Results

### ✅ Directory Structure: PERFECT COMPLIANCE

- **Deep Nesting Violations**: 0 ✅
- **Invalid Root Directories**: 0 ✅
- **Naming Convention Violations**: 0 ✅

**Finding**: All files comply with structural requirements:
- ✅ Maximum 3 directory levels maintained
- ✅ Valid root directories only: `guides, explanations, reference, tutorials, diagrams, concepts, architecture, audits, analysis, assets, archive, working`
- ✅ Kebab-case naming convention throughout

---

## Category Mismatch Analysis

### Critical Mismatches (Top 20 of 153)


**1. `README.md`**
- Declared: `tutorial`
- Suggested: `guide`
- Reason: Content markers suggest guide (score: 203) vs tutorial (score: 46)

**2. `ARCHITECTURE_OVERVIEW.md`**
- Declared: `guide`
- Suggested: `reference`
- Reason: Content markers suggest reference (score: 154) vs howto (score: 0)

**3. `gpu-fix-summary.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 27) vs explanation (score: 2)

**4. `MAINTENANCE.md`**
- Declared: `reference`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 25) vs reference (score: 9)

**5. `QA_VALIDATION_FINAL.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 91) vs explanation (score: 23)

**6. `VALIDATION_CHECKLIST.md`**
- Declared: `reference`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 48) vs reference (score: 20)

**7. `NAVIGATION.md`**
- Declared: `reference`
- Suggested: `guide`
- Reason: Content markers suggest guide (score: 156) vs reference (score: 68)

**8. `TECHNOLOGY_CHOICES.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 89) vs explanation (score: 42)

**9. `DEVELOPER_JOURNEY.md`**
- Declared: `guide`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 84) vs howto (score: 0)

**10. `comfyui-service-integration.md`**
- Declared: `guide`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 28) vs howto (score: 0)

**11. `audits/README.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 59) vs explanation (score: 16)

**12. `audits/neo4j-settings-migration-audit.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 92) vs explanation (score: 17)

**13. `audits/neo4j-migration-action-plan.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 49) vs explanation (score: 2)

**14. `audits/neo4j-migration-summary.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 48) vs explanation (score: 11)

**15. `reference/physics-implementation.md`**
- Declared: `reference`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 44) vs reference (score: 7)

**16. `multi-agent-docker/x-fluxagent-adaptation-plan.md`**
- Declared: `explanation`
- Suggested: `reference`
- Reason: Content markers suggest reference (score: 113) vs explanation (score: 40)

**17. `multi-agent-docker/TERMINAL_GRID.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 19) vs explanation (score: 0)

**18. `multi-agent-docker/upstream-analysis.md`**
- Declared: `explanation`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 30) vs explanation (score: 8)

**19. `multi-agent-docker/SKILLS.md`**
- Declared: `guide`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 45) vs howto (score: 0)

**20. `multi-agent-docker/comfyui-sam3d-setup.md`**
- Declared: `guide`
- Suggested: `tutorial`
- Reason: Content markers suggest tutorial (score: 24) vs howto (score: 0)


### Mismatch Patterns

**By Suggested Category**:
- Tutorial candidates: 116 files
- Guide candidates: 9 files
- Reference candidates: 28 files
- Explanation candidates: 0 files

**Full list**: See `hive-diataxis-validation.json` for complete mismatch data

---

## Actionable Recommendations

### Priority 1: Expand Tutorial Content ⚠️ CRITICAL

**Current**: 5 tutorials (2.24%)
**Target**: 27 tutorials (12.5%)
**Action**: Create **22 new tutorials** or reclassify existing content

**Candidates for reclassification to tutorial**:
- gpu-fix-summary.md
- MAINTENANCE.md
- QA_VALIDATION_FINAL.md
- VALIDATION_CHECKLIST.md
- TECHNOLOGY_CHOICES.md
- DEVELOPER_JOURNEY.md
- comfyui-service-integration.md
- audits/README.md
- audits/neo4j-settings-migration-audit.md
- audits/neo4j-migration-action-plan.md

### Priority 2: Reduce Explanation Overrepresentation

**Current**: 100 explanations (44.84%)
**Target**: 78 explanations (35%)
**Action**: Convert **22 explanations** to tutorials or guides

### Priority 3: Categorize Uncategorized Files

**Action**: Add category frontmatter to 6 files:

```yaml
---
category: tutorial|guide|reference|explanation
---
```

### Priority 4: Validate Content Matches Category

**Action**: Review and update 153 mismatched files:

1. **Tutorial validation**: Ensure step-by-step instructions, prerequisites, "follow along" language
2. **Guide validation**: Ensure problem-solution format, task-oriented, "how to" language
3. **Reference validation**: Ensure completeness, accuracy, API specs, parameters
4. **Explanation validation**: Ensure context, background, "why" focus, conceptual depth

---

## Diataxis Category Criteria Reference

### Tutorial (Learning-Oriented)
**Characteristics**:
- Step-by-step instructions
- Linear progression
- "Follow along" language
- Prerequisites clearly stated
- Beginner-friendly

**Markers**: "step", "follow", "begin", "install", "##" (numbered sections)

### Guide (Task-Oriented)
**Characteristics**:
- Problem-solution format
- "How to" achieve specific goals
- Practical, actionable
- Assumes some knowledge

**Markers**: "how to", "guide", "implement", "configure", "solution"

### Reference (Information-Oriented)
**Characteristics**:
- Complete, accurate information
- API specifications
- Parameters, return values
- Schema definitions
- Dry, factual

**Markers**: "api", "reference", "specification", "endpoint", "parameter", "schema"

### Explanation (Understanding-Oriented)
**Characteristics**:
- Context and background
- "Why" focus
- Architectural decisions
- Conceptual depth
- Theory and reasoning

**Markers**: "why", "context", "background", "architecture", "decision", "concept"

---

## Validation Methodology

### Analysis Process

1. **Content Scanning**: Parse all `.md` files in active documentation
2. **Marker Scoring**: Count category-specific keywords and patterns
3. **Frontmatter Extraction**: Read declared category from YAML frontmatter
4. **Mismatch Detection**: Compare declared vs. suggested (score threshold: 2x)
5. **Structure Validation**: Check depth, naming, root directories

### Exclusions
- `archive/`: 80 archived files excluded
- `working/`: Active work-in-progress files excluded

### Scoring Formula

```python
compliance = (
    tutorial_score +    # 25% if 10-15%, 15% if >5%
    guide_score +       # 25% if 25-35%, 15% if >15%
    reference_score +   # 25% if 20-30%, 15% if >10%
    explanation_score   # 25% if 30-40%, 15% if >20%
)
```

---

## Next Steps for Hive Mind Agents

### Content Auditor
- Review 153 category mismatches
- Update frontmatter for 6 uncategorized files
- Propose reclassification plan

### Link Validator
- Ensure cross-references remain valid after recategorization
- Update internal links if files move categories

### ASCII Deprecator
- Convert remaining ASCII diagrams to Mermaid
- Ensure visual documentation accessibility

### Archive Curator
- Move outdated content to `archive/`
- Maintain archive index

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Total Files** | 303 |
| **Active Files** | 223 |
| **Archived Files** | 80 |
| **Categorized Files** | 217 |
| **Uncategorized Files** | 6 |
| **Category Mismatches** | 153 |
| **Structure Violations** | 0 |
| **Diataxis Compliance** | **55%** |

---

## Memory Store Results

**Key**: `hive/worker/structure-normaliser/results`

**Data**:
```json
{
  "compliance_percentage": 55,
  "total_files": 303,
  "active_files": 223,
  "categories": {
  "tutorial": {
    "count": 5,
    "percentage": 2.24
  },
  "explanation": {
    "count": 100,
    "percentage": 44.84
  },
  "reference": {
    "count": 41,
    "percentage": 18.39
  },
  "uncategorized": {
    "count": 6,
    "percentage": 2.69
  },
  "guide": {
    "count": 71,
    "percentage": 31.84
  }
},
  "violations": {
    "mismatches": 153,
    "structure": 0
  },
  "status": "moderate_compliance",
  "recommendations": [
    "Create 22 new tutorials",
    "Reclassify 153 mismatched files",
    "Categorize 6 uncategorized files"
  ]
}
```

---

**Generated by**: Structure Normaliser Agent (Diataxis Compliance Hive Mind)
**Validation Date**: 2025-12-19
**Data Source**: `/home/devuser/workspace/project/docs/working/hive-diataxis-validation.json`
**Target**: 95%+ compliance before production release

**Status**: ⚠️ **MODERATE COMPLIANCE** - Actionable improvements identified
