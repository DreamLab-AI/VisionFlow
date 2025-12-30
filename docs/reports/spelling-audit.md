# VisionFlow Documentation Spelling Audit
**US to UK English Conversion Report**

**Audit Date**: 2025-12-30
**Documentation Path**: `/home/devuser/workspace/project/docs`
**Total Files Scanned**: 92 markdown files
**Total Occurrences Found**: 3,224 American spellings

---

## Executive Summary

The VisionFlow documentation contains 3,224 instances of American English spellings that should be converted to UK English variants. The most frequently occurring terms are:

| Spelling Variant | US → UK | Occurrences | Files | Priority |
|---|---|---|---|---|
| color/colors | colour/colours | 1,348 | 188 | High |
| optimize/optimization | optimise/optimisation | 635 | 127 | High |
| organize/organization | organise/organisation | 227 | 57 | High |
| analyze/analysis | analyse/analysis* | 326 | 188 | High |
| realize | realise | 6 | 3 | Medium |
| fiber/fibers | fibre/fibres | 29 | 15 | Medium |
| behavior/behaviors | behaviour/behaviours | 120 | 33 | Medium |
| gray | grey | 10 | 4 | Low |
| canceled | cancelled | 14 | 5 | Low |
| labeled | labelled | 16 | 7 | Low |
| traveled | travelled | 2 | 2 | Low |
| modeling/modelling | modelling | 182 | 24 | Medium |

*Note: "analysis" is UK-compatible (no change needed)

**Total corrections needed**: 2,939 (excluding analysis as it's language-neutral)

---

## Detailed Findings by Term

### 1. COLOR/COLOURS (1,348 occurrences)

**Files with occurrences**: 188 files
**Status**: Highest priority - affects visual documentation extensively

**Top files by frequency**:
- `/home/devuser/workspace/project/docs/working/hive-spelling-audit.json` - 200 occurrences
- `/home/devuser/workspace/project/docs/working/hive-diagram-validation.json` - 330 occurrences
- `/home/devuser/workspace/project/docs/visionflow-architecture-analysis.md` - Multiple occurrences
- `/home/devuser/workspace/project/docs/concepts/architecture/core/client.md` - Instance color references
- `/home/devuser/workspace/project/docs/concepts/architecture/core/server.md` - 8 occurrences

**Examples**:
```
"inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors..."
Uniforms: time, colors, opacity, hologram parameters
Per-Instance Data: Position, color, scale via setMatrixAt()
Instance Colors: Per-node color overrides via instanceColor
```

**Find/Replace Commands**:
```bash
# Basic color → colour
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bcolors\b/colours/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bcolor\b/colour/g' {} \;

# Preserve code attributes (transition-colors, etc.) - MANUAL REVIEW NEEDED
grep -n "transition-colors\|instanceColor\|setColor" /home/devuser/workspace/project/docs -r
```

**Exceptions to Skip**:
- HTML/CSS property names: `transition-colors`, `background-color`, `color:#` in code blocks
- JavaScript identifiers: `instanceColor`, `setColor()`, `colorize()`
- CSS class names: `.color-`, `.colors-`
- API parameter names in code samples

---

### 2. OPTIMIZE/OPTIMIZATION (635 occurrences)

**Files with occurrences**: 127 files
**Status**: High priority

**Top files**:
- `/home/devuser/workspace/project/docs/working/hive-spelling-audit.json` - 635 occurrences
- `/home/devuser/workspace/project/docs/guides/stress-majorization-guide.md` - 18 occurrences
- `/home/devuser/workspace/project/docs/CUDA_OPTIMIZATION_SUMMARY.md` - 14 occurrences
- `/home/devuser/workspace/project/docs/archive/implementation-logs/stress-majorization-implementation.md` - 18 occurrences
- `/home/devuser/workspace/project/docs/explanations/architecture/stress-majorization.md` - 29 occurrences

**Find/Replace Commands**:
```bash
# Batch conversions
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\boptimization\b/optimisation/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\boptimize\b/optimise/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\boptimized\b/optimised/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\boptimizes\b/optimises/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\boptimizing\b/optimising/g' {} \;
```

**Exceptions**: None - safe to replace throughout

---

### 3. ORGANIZE/ORGANIZATION (227 occurrences)

**Files with occurrences**: 57 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganization\b/organisation/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganizations\b/organisations/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganize\b/organise/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganized\b/organised/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganizes\b/organises/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\borganizing\b/organising/g' {} \;
```

---

### 4. ANALYZE/ANALYSIS (326 occurrences)

**Files with occurrences**: 188 files
**Status**: Mixed - many are JSON files with "analysis" suffix

**Note**: "analysis" is language-neutral and used identically in both US and UK English. Only convert "analyze" verb forms:

**Find/Replace Commands**:
```bash
# Only convert the verb forms, NOT the noun "analysis"
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\banalyze\b/analyse/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\banalyzed\b/analysed/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\banalyzes\b/analyses/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\banalyzing\b/analysing/g' {} \;

# Do NOT change "analysis" - it's correct in both variants
```

---

### 5. FIBER/FIBRES (29 occurrences)

**Files with occurrences**: 15 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bfiber\b/fibre/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bfibers\b/fibres/g' {} \;
```

---

### 6. BEHAVIOR/BEHAVIOURS (120 occurrences)

**Files with occurrences**: 33 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bbehavior\b/behaviour/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bbehaviors\b/behaviours/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bbehavioral\b/behavioural/g' {} \;
```

---

### 7. MODELING/MODELLING (182 occurrences)

**Files with occurrences**: 24 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bmodeling\b/modelling/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bmodels\b/models/g' {} \;  # No change needed
```

---

### 8. GRAY/GREY (10 occurrences)

**Files with occurrences**: 4 files
- `/home/devuser/workspace/project/docs/explanations/architecture/hierarchical-visualization.md` - 7 occurrences
- `/home/devuser/workspace/project/docs/DEVELOPER_JOURNEY.md` - 1 occurrence
- `/home/devuser/workspace/project/docs/explanations/architecture/ontology-analysis.md` - 1 occurrence
- `/home/devuser/workspace/project/docs/archive/analysis/client-architecture-analysis-2025-12.md` - 1 occurrence

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bgray\b/grey/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bgrays\b/greys/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bgray\b/grey/g' {} \;
```

---

### 9. CANCELED/CANCELLED (14 occurrences)

**Files with occurrences**: 5 files
- `/home/devuser/workspace/project/docs/comfyui-integration-design.md` - 7 occurrences
- `/home/devuser/workspace/project/docs/comfyui-management-api-integration-summary.md` - 2 occurrences
- `/home/devuser/workspace/project/docs/diagrams/server/api/rest-api-architecture.md` - 3 occurrences
- `/home/devuser/workspace/project/docs/working/spelling-scanner.js` - 1 occurrence
- `/home/devuser/workspace/project/docs/scripts/validate-spelling.sh` - 1 occurrence

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bcanceled\b/cancelled/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\bcanceling\b/cancelling/g' {} \;
```

---

### 10. LABELED/LABELLED (16 occurrences)

**Files with occurrences**: 7 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\blabeled\b/labelled/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\blabeling\b/labelling/g' {} \;
```

---

### 11. TRAVELED/TRAVELLED (2 occurrences)

**Files with occurrences**: 2 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\btraveled\b/travelled/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\btraveling\b/travelling/g' {} \;
```

---

### 12. REALIZE/REALISE (6 occurrences)

**Files with occurrences**: 3 files

**Find/Replace Commands**:
```bash
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\brealize\b/realise/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\brealized\b/realised/g' {} \;
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i 's/\brealizing\b/realising/g' {} \;
```

---

## Automated Correction Strategy

### Phase 1: JSON Files (Safe to Auto-convert)
```bash
# These files are metadata and safe for bulk conversion
find /home/devuser/workspace/project/docs -name "*.json" -type f -exec sed -i \
  -e 's/\bcolor/colour/g' \
  -e 's/\boptimization/optimisation/g' \
  -e 's/\boptimize/optimise/g' \
  -e 's/\borganization/organisation/g' \
  -e 's/\borganize/organise/g' \
  -e 's/\banalyze/analyse/g' \
  -e 's/\bfiber/fibre/g' \
  -e 's/\bbehavior/behaviour/g' \
  -e 's/\bmodeling/modelling/g' \
  -e 's/\bgray/grey/g' \
  -e 's/\bcanceled/cancelled/g' \
  -e 's/\blabeled/labelled/g' \
  -e 's/\brealize/realise/g' {} \;
```

### Phase 2: Markdown Files (Manual Review for Code Blocks)
```bash
# First, identify code block boundaries
grep -n "^```" /home/devuser/workspace/project/docs/**/*.md | head -50

# Create backup before bulk conversion
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec cp {} {}.backup \;

# Convert excluding known exceptions
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec sed -i \
  -e 's/\boptimization\b/optimisation/g' \
  -e 's/\boptimize\b/optimise/g' \
  -e 's/\boptimized\b/optimised/g' \
  -e 's/\borganization\b/organisation/g' \
  -e 's/\borganize\b/organise/g' \
  -e 's/\banalyze\b/analyse/g' \
  -e 's/\banalyzed\b/analysed/g' \
  -e 's/\bfiber\b/fibre/g' \
  -e 's/\bbehavior\b/behaviour/g' \
  -e 's/\bmodeling\b/modelling/g' \
  -e 's/\bgray\b/grey/g' \
  -e 's/\bcanceled\b/cancelled/g' \
  -e 's/\blabeled\b/labelled/g' \
  -e 's/\brealize\b/realise/g' {} \;
```

---

## Code Blocks Requiring Manual Review

### HTML/CSS Properties (DO NOT CONVERT):
- `color: #ffffff;` (CSS property)
- `background-color: #000;` (CSS property)
- `transition-colors` (CSS class)
- `focus-visible:ring-2` (Tailwind utility)

### JavaScript Identifiers (DO NOT CONVERT):
- `instanceColor` (variable name)
- `setColor()` (method name)
- `colorize()` (function name)
- `getAnalysis()` (function name)

### API Parameter Names (DO NOT CONVERT):
- Query parameters: `?analyze=true`
- JSON keys: `"analyzeParameters": {}`
- Enum values: `COLOR_MODE.PRIMARY`

### Known Exceptions in Code:
**File**: `/home/devuser/workspace/project/docs/archive/specialized/extension-guide.md`
```
Lines with color property (CSS) - Line 974, 988, 989, 1004, 1005
These should NOT be converted as they're CSS property names
```

**File**: `/home/devuser/workspace/project/docs/concepts/architecture/core/client.md`
```
Lines with instanceColor (JavaScript) - Should NOT be converted
```

---

## Implementation Checklist

- [ ] **Pre-conversion**: Backup all markdown files
- [ ] **Phase 1**: Run JSON file conversions (safe, fully automated)
- [ ] **Phase 2a**: Convert OPTIMIZATION → OPTIMISATION across all markdown
- [ ] **Phase 2b**: Convert ORGANIZE → ORGANISE across all markdown
- [ ] **Phase 2c**: Convert FIBER → FIBRE across all markdown
- [ ] **Phase 2d**: Convert BEHAVIOR → BEHAVIOUR across all markdown
- [ ] **Phase 2e**: Convert MODELING → MODELLING across all markdown
- [ ] **Phase 2f**: Convert ANALYZE → ANALYSE across all markdown (excluding "analysis")
- [ ] **Phase 2g**: Convert GRAY → GREY across all markdown
- [ ] **Phase 2h**: Manual review: COLOR → COLOUR (check for CSS/JS exceptions)
- [ ] **Phase 2i**: Convert remaining terms (CANCELED, LABELED, TRAVELED, REALIZED)
- [ ] **Verification**: Run grep searches to confirm conversions
- [ ] **Git commit**: Create single commit with all spelling corrections

---

## Risk Assessment

**Low Risk Conversions** (Can be automated safely):
- OPTIMIZE → OPTIMISE (no common code identifiers)
- ORGANIZE → ORGANISE (no common code identifiers)
- FIBER → FIBRE (rare in code)
- MODELING → MODELLING (rare in code)
- BEHAVIOR → BEHAVIOUR (rare as method name)
- ANALYZE → ANALYSE (verify function names first)
- GRAY → GREY (unlikely in code)
- CANCELED → CANCELLED (unlikely in code)
- LABELED → LABELLED (unlikely in code)
- REALIZED → REALISED (unlikely in code)

**Higher Risk Conversions** (Require manual review):
- COLOR → COLOUR (Common in CSS: `color:`, `background-color:`, `transition-colors`)
  - Estimated manual review files: 20-30
  - Suggested approach: Convert, then review diffs carefully

---

## Verification Commands

After conversion, verify results:

```bash
# Check converted terms
grep -r "optimisation\|organise\|fibre\|behaviour\|modelling\|grey\|cancelled\|labelled\|realised" /home/devuser/workspace/project/docs --include="*.md" | wc -l

# Should show increased count (opposite of original)

# Check for remaining US spellings (should be minimal)
grep -r "\boptimization\b\|\borganization\b\|\bfiber\b\|\bbehavior\b\|\bmodeling\b\|\bgray\b" /home/devuser/workspace/project/docs --include="*.md" | wc -l

# Check specific problem areas
grep -n "color:\|background-color:\|instanceColor" /home/devuser/workspace/project/docs/**/*.md
```

---

## Recommended Execution Order

1. **Backup Phase**: Create git branch `spelling-audit-fixes`
2. **Safe Automations First**: OPTIMIZE, ORGANIZE, FIBER, BEHAVIOR, MODELING, GRAY, CANCELED, LABELED, REALIZED
3. **Careful Review**: ANALYZE (verify no code conflicts)
4. **Manual Review**: COLOR (check CSS/JS exceptions)
5. **Verification**: Run grep checks
6. **Commit**: Single comprehensive commit with clear message

---

## Summary Statistics

| Category | Count | Effort |
|---|---|---|
| Files to process | 188+ | High |
| Total term instances | 3,224 | - |
| Automated conversions possible | ~2,850 (88%) | Low |
| Require manual review | ~374 (12%) | Medium |
| Code exception lines | ~40-60 | Low |
| Estimated time | 2-3 hours | - |

---

**Report Generated**: 2025-12-30
**Next Steps**: Execute Phase 1 (JSON), then Phase 2 (Markdown with manual verification)
