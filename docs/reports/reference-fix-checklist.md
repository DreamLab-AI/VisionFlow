# Reference Directory Broken Links - Fix Checklist

**Total Broken Links**: 726
**Estimated Fix Time**: 60 minutes
**Impact**: Fixes 37.3% of all documentation broken links

---

## ✅ Phase 1: Symlinks for British Spellings (10 min)

**Impact**: Fixes 618 broken links (85.1%)

### Directory Symlinks

```bash
cd /home/devuser/workspace/project/docs/reference/agents

# Create optimisation → optimization symlink
ln -s optimization optimisation
```

**Fixes**: All 366 references to `reference/agents/optimisation/*`

---

### File Symlinks

```bash
cd /home/devuser/workspace/project/docs/reference/agents

# Analysis files
cd analysis
ln -s code-analyzer.md code-analyser.md

cd code-review
ln -s analyze-code-quality.md analyse-code-quality.md

# Optimization files
cd ../../optimization
ln -s topology-optimizer.md topology-optimiser.md

# Template files
cd ../templates
ln -s performance-analyzer.md performance-analyser.md
```

**Fixes**:
- 61 refs to `code-analyser.md`
- 61 refs to `analyse-code-quality.md`
- 61 refs to `topology-optimiser.md`
- 62 refs to `performance-analyser.md`

---

## ✅ Phase 2: Fix Double Path Errors (20 min)

**Impact**: Fixes 121 broken links (16.7%)

### Pattern 1: `reference/reference/agents`

**Files to update**: ~10 files

Search for:
```
../../reference/agents/
```

Replace with:
```
../../reference/agents/
```

Or for files at different depths:
```
../reference/agents/
→ ../reference/agents/
```

**Affected files**:
- `reference/agents/README.md`
- `reference/agents/analysis/code-analyzer.md`
- `reference/agents/analysis/code-review/analyze-code-quality.md`
- `reference/agents/architecture/system-design/arch-system-design.md`
- `reference/agents/consensus/README.md`
- `reference/agents/core/*.md`
- All other agent category files

### Pattern 2: `reference/agents/reference`

Search for:
```
reference/agents/
```

Replace with:
```
reference/agents/
```

---

## ✅ Phase 3: Create Missing Files (30 min)

**Impact**: Fixes 14 broken links (1.9%)

### Missing Architecture File

**File**: `docs/reference/architecture/architecture.md`
**References**: 4 broken links

**Content suggestion**:
```markdown
# System Architecture Overview

See also:
- [Hexagonal CQRS](./hexagonal-cqrs.md)
- [Actor System](./actor-system.md)
- [Database Schema](./database-schema.md)
```

---

### Missing Ontology Directory

**Directory**: `docs/reference/ontology/`
**Files needed**: 4

1. `api-reference.md` - OWL/RDF API documentation
2. `hornedowl.md` - HornedOWL library integration
3. `integration-summary.md` - Ontology integration overview
4. `system-overview.md` - Ontology system architecture

**Referenced by**: `concepts/ontology-and-validation.md`

---

### Missing Decision Records

**Directory**: `docs/reference/decisions/`
**Files needed**: 2

These already exist in `docs/concepts/decisions/`, just need symlinks:

```bash
mkdir -p docs/reference/decisions
cd docs/reference/decisions
ln -s ../../concepts/decisions/adr-001-unified-api-client.md
ln -s ../../concepts/decisions/adr-003-code-pruning-2025-10.md
```

---

### Missing SPARC Methodology

**File**: `docs/reference/sparc-methodology.md`
**References**: 2 broken links (from `/docs/reference/` absolute paths)

**Content suggestion**:
```markdown
# SPARC Methodology Reference

See also:
- [SPARC Agents](./agents/sparc/index.md)
- [SPARC Specification](./agents/sparc/specification.md)
- [SPARC Architecture](./agents/sparc/architecture.md)
```

---

## 📊 Verification Commands

### Before Fix

```bash
cd /home/devuser/workspace/project
python3 validate_links.py | grep "reference/" | wc -l
# Expected: 726
```

### After Phase 1

```bash
python3 validate_links.py | grep "reference/" | wc -l
# Expected: ~108 (726 - 618)
```

### After Phase 2

```bash
python3 validate_links.py | grep "reference/" | wc -l
# Expected: ~14 (108 - 94, accounting for overlap)
```

### After Phase 3

```bash
python3 validate_links.py | grep "reference/" | wc -l
# Expected: 0 or near 0
```

---

## 🎯 Success Criteria

- [ ] All symlinks created successfully
- [ ] No broken symlinks (`find -L -type l`)
- [ ] Double path patterns eliminated (grep verification)
- [ ] Missing files created or symlinked
- [ ] Validation script shows <10 broken reference/ links
- [ ] No regressions in other directories

---

## ⚠️ Potential Issues

### Symlink Compatibility

**Risk**: Symlinks may not work in all environments (Windows)

**Mitigation**:
- Git should handle symlinks correctly on modern systems
- Alternative: Use duplicate files instead of symlinks
- Document in README if symlinks are required

### Archive References

**Issue**: `archive/migration-legacy/reference/` has legitimate missing files

**Recommendation**: Leave archive broken links as-is (low priority)
- These are historical documents
- May be intentionally incomplete
- Don't affect active documentation

---

## 📈 Progress Tracking

### Phase 1 Checklist

- [ ] Create `optimisation` → `optimization` directory symlink
- [ ] Create `code-analyser.md` → `code-analyzer.md` symlink
- [ ] Create `analyse-code-quality.md` → `analyze-code-quality.md` symlink
- [ ] Create `topology-optimiser.md` → `topology-optimizer.md` symlink
- [ ] Create `performance-analyser.md` → `performance-analyzer.md` symlink
- [ ] Run validation: expect ~108 broken links remaining

### Phase 2 Checklist

- [ ] Identify all files with double path errors
- [ ] Batch replace `reference/reference/agents` → `reference/agents`
- [ ] Batch replace `reference/agents/reference/agents` → `reference/agents`
- [ ] Verify no instances remain: `grep -r "reference/reference" docs/`
- [ ] Run validation: expect ~14 broken links remaining

### Phase 3 Checklist

- [ ] Create `reference/architecture/architecture.md`
- [ ] Create `reference/ontology/` directory and 4 files
- [ ] Create `reference/decisions/` symlinks
- [ ] Create `reference/sparc-methodology.md`
- [ ] Run final validation: expect 0-5 broken links

---

## 🔄 Automation Opportunity

After manual fixes, consider creating a fix script:

```bash
#!/bin/bash
# scripts/fix-reference-links.sh

set -e

echo "Phase 1: Creating symlinks..."
cd docs/reference/agents
ln -sf optimization optimisation
# ... (all other symlinks)

echo "Phase 2: Fixing double paths..."
# Batch sed commands

echo "Phase 3: Creating missing files..."
# mkdir and template creation

echo "Done! Running validation..."
cd ../../..
python3 validate_links.py
```

This allows:
- Reproducible fixes
- Easy rollback if issues arise
- Documentation of fix approach

---

## 📝 Notes

### Why Symlinks vs Duplicates?

**Symlinks chosen because**:
- Don't duplicate content
- Updates automatically propagate
- Smaller repository size
- Clear "source of truth"

**Duplicates might be better if**:
- Cross-platform compatibility critical
- Symlinks cause issues in some tools
- Want to eventually deprecate British spellings

### Path Standardization

Consider standardizing all documentation paths to use:
- American spelling (optimization, analyzer)
- Relative paths (avoid absolute `/docs/` paths)
- Consistent depth (avoid `../../` spaghetti)

---

## ✨ Bonus: Style Guide Update

After fixes, update documentation style guide:

```markdown
## Link Conventions

1. **Spelling**: Use American English (optimization, not optimisation)
2. **Paths**: Use relative paths from current file
3. **Extensions**: Always include `.md` extension
4. **Case**: Use lowercase with hyphens (analyze-code-quality.md)
```

Add to: `docs/CONTRIBUTING.md` or `docs/reference/conventions.md`
