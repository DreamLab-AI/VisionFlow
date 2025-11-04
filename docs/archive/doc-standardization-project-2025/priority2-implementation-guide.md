# Priority 2 Implementation Guide - Step-by-Step Execution

**Status**: Ready for Implementation  
**Estimated Duration**: 4-6 hours  
**Complexity**: Medium (23 files, 27 link corrections)

---

## Quick Reference: All Files Requiring Fixes

### Category A: Architecture Path Updates (23 links across 21 files)

**Requires**: Change `../concepts/architecture/` → `../concepts/architecture/`

1. `guides/xr-setup.md` - 3 links
2. `guides/ontology-storage-guide.md` - 3 links  
3. `guides/vircadia-multi-user-guide.md` - 2 links
4. `reference/api/readme.md` - 1 link
5. `reference/api/03-websocket.md` - 1 link
6. `reference/api/rest-api-complete.md` - 1 link
7. `reference/api/rest-api-reference.md` - 2 links
8. `guides/navigation-guide.md` - 8 links
9. `getting-started/01-installation.md` - 1 link
10. `guides/developer/01-development-setup.md` - 1 link
11. `guides/migration/json-to-binary-protocol.md` - 1 link

### Category B: Double-Reference Path Fixes (4 links in 1 file)

**Requires**: Change `../reference/api/x.md` → `./x.md`

1. `reference/api/03-websocket.md` - 3 links (NOTE: Also in Category A)

---

## Implementation Strategy

### Phase 1: Automated Find & Replace (2-3 hours)

Use your editor's find and replace to fix bulk path issues. This section provides exact patterns.

#### Pattern 1: Fix `../concepts/architecture/` → `../concepts/architecture/`

**Files affected**: 11 files (guides/xr-setup.md, guides/ontology-storage-guide.md, guides/vircadia-multi-user-guide.md, reference/api/readme.md, reference/api/03-websocket.md, reference/api/rest-api-complete.md, reference/api/rest-api-reference.md, getting-started/01-installation.md, guides/developer/01-development-setup.md, guides/migration/json-to-binary-protocol.md)

**Bash command** (dry-run first):
```bash
# DRY RUN: See what will be changed
grep -rn "\.\./architecture/" /home/devuser/workspace/project/docs \
  --include="*.md" | grep -v "concepts/architecture"

# EXECUTE: Replace all occurrences
find /home/devuser/workspace/project/docs -name "*.md" -type f -exec \
  sed -i 's|\.\./architecture/|\.\./concepts/architecture/|g' {} +
```

**Verification after fix**:
```bash
# Verify replacements
grep -rn "\.\./concepts/architecture/" /home/devuser/workspace/project/docs --include="*.md" | wc -l
# Should show 21+ matches
```

---

#### Pattern 2: Fix `architecture/` → `concepts/architecture/` (no ../)

**Files affected**: guides/navigation-guide.md (8 links)

**Note**: This file uses relative paths from docs root (no `../` prefix)

**Bash command** (dry-run first):
```bash
# DRY RUN: Find lines with "architecture/" not preceded by "../"
grep -n "]\(architecture/" /home/devuser/workspace/project/docs/guides/navigation-guide.md

# EXECUTE: Replace in navigation-guide.md only
sed -i 's|](architecture/|](concepts/architecture/|g' \
  /home/devuser/workspace/project/docs/guides/navigation-guide.md
```

**Verification**:
```bash
# Verify all occurrences were updated
grep -n "architecture/" /home/devuser/workspace/project/docs/guides/navigation-guide.md
# Should all show "concepts/architecture/" now
```

---

#### Pattern 3: Fix `../../concepts/architecture/` → `../../concepts/architecture/`

**Files affected**: guides/migration/json-to-binary-protocol.md

**Bash command**:
```bash
# Fix this specific pattern
sed -i 's|../../concepts/architecture/|../../concepts/architecture/|g' \
  /home/devuser/workspace/project/docs/guides/migration/json-to-binary-protocol.md
```

---

### Phase 2: Manual Reference Fixes (1-2 hours)

#### Fix Double-Reference Paths in reference/api/03-websocket.md

**Required changes**:
```
OLD: ../reference/api/binary-protocol.md
NEW: ./binary-protocol.md

OLD: ../reference/api/rest-api.md  
NEW: ./rest-api.md

OLD: ../reference/performance-benchmarks.md
NEW: ../performance-benchmarks.md
```

**Manual steps** (file is complex, requires careful editing):

1. Open `/home/devuser/workspace/project/docs/reference/api/03-websocket.md`
2. Find line with: `../reference/api/binary-protocol.md`
3. Replace with: `./binary-protocol.md`
4. Find line with: `../reference/api/rest-api.md`
5. Replace with: `./rest-api.md`
6. Find line with: `../reference/performance-benchmarks.md`
7. Replace with: `../performance-benchmarks.md`

**Bash command** (if confident):
```bash
# Execute all three replacements
sed -i 's|\.\./reference/api/binary-protocol\.md|./binary-protocol.md|g' \
  /home/devuser/workspace/project/docs/reference/api/03-websocket.md

sed -i 's|\.\./reference/api/rest-api\.md|./rest-api.md|g' \
  /home/devuser/workspace/project/docs/reference/api/03-websocket.md

sed -i 's|\.\./reference/performance-benchmarks\.md|../performance-benchmarks.md|g' \
  /home/devuser/workspace/project/docs/reference/api/03-websocket.md
```

---

## Complete Bash Script (All-in-One)

Save this as `/tmp/fix-priority2.sh` and execute:

```bash
#!/bin/bash

set -e  # Exit on first error

DOCS-PATH="/home/devuser/workspace/project/docs"

echo "Starting Priority 2 Path Corrections..."
echo "========================================"

# Pattern 1: Fix ../concepts/architecture/ → ../concepts/architecture/
echo ""
echo "Phase 1: Updating ../concepts/architecture/ paths..."
echo "Files affected: 11"

find "$DOCS-PATH" -name "*.md" -type f -exec \
  sed -i 's|\.\./architecture/|\.\./concepts/architecture/|g' {} + && \
  echo "✓ Pattern 1 complete"

# Pattern 2: Fix architecture/ → concepts/architecture/ (no ../)
echo ""
echo "Phase 2: Updating architecture/ paths (no ../)..."
echo "File affected: navigation-guide.md"

sed -i 's|](architecture/|](concepts/architecture/|g' \
  "$DOCS-PATH/guides/navigation-guide.md" && \
  echo "✓ Pattern 2 complete"

# Pattern 3: Fix ../../concepts/architecture/ → ../../concepts/architecture/
echo ""
echo "Phase 3: Updating ../../concepts/architecture/ paths..."
echo "File affected: guides/migration/json-to-binary-protocol.md"

sed -i 's|../../concepts/architecture/|../../concepts/architecture/|g' \
  "$DOCS-PATH/guides/migration/json-to-binary-protocol.md" && \
  echo "✓ Pattern 3 complete"

# Pattern 4: Fix reference/api double-references
echo ""
echo "Phase 4: Fixing reference/api double-reference paths..."
echo "File affected: reference/api/03-websocket.md"

sed -i 's|\.\./reference/api/binary-protocol\.md|./binary-protocol.md|g' \
  "$DOCS-PATH/reference/api/03-websocket.md"

sed -i 's|\.\./reference/api/rest-api\.md|./rest-api.md|g' \
  "$DOCS-PATH/reference/api/03-websocket.md"

sed -i 's|\.\./reference/performance-benchmarks\.md|../performance-benchmarks.md|g' \
  "$DOCS-PATH/reference/api/03-websocket.md" && \
  echo "✓ Pattern 4 complete"

echo ""
echo "========================================"
echo "✓ All Priority 2 fixes complete!"
echo ""
echo "Verification summary:"
echo "---"

# Verification counts
echo ""
echo "Checking for ../concepts/architecture/ links (should be 21+):"
grep -r "\.\./concepts/architecture/" "$DOCS-PATH" --include="*.md" | wc -l

echo ""
echo "Checking for remaining ../concepts/architecture/ without concepts (should be 0):"
grep -r "\.\./architecture/" "$DOCS-PATH" --include="*.md" | grep -v "concepts/architecture" | wc -l

echo ""
echo "Checking for remaining ../reference/reference/ (should be 0):"
grep -r "\.\./reference/reference/" "$DOCS-PATH" --include="*.md" | wc -l

echo ""
echo "Task complete!"
```

**Execute the script**:
```bash
bash /tmp/fix-priority2.sh
```

---

## Verification Checklist

After implementing fixes, verify with these commands:

### Check 1: Architecture Path Consistency
```bash
# Should return 21+ matches of correct paths
grep -r "\.\./concepts/architecture/" /home/devuser/workspace/project/docs --include="*.md" | head -5
```

### Check 2: No Broken ../concepts/architecture/ Paths
```bash
# Should return 0 (no broken paths remaining)
grep -r "\]\(\.\./architecture/" /home/devuser/workspace/project/docs --include="*.md" | grep -v "concepts"
```

### Check 3: No Double-Reference Paths
```bash
# Should return 0 (no ../reference/reference/ paths)
grep -r "\.\./reference/reference/" /home/devuser/workspace/project/docs --include="*.md"
```

### Check 4: Validate Specific Files

**guides/xr-setup.md** (should have 3 fixes):
```bash
grep "architecture/" /home/devuser/workspace/project/docs/guides/xr-setup.md | grep "concepts"
# Should return 3 results
```

**guides/navigation-guide.md** (should have 8 fixes):
```bash
grep "architecture/" /home/devuser/workspace/project/docs/guides/navigation-guide.md | wc -l
# Should return 8+ results with "concepts/architecture/"
```

**reference/api/03-websocket.md** (should have paths fixed):
```bash
grep -E "binary-protocol|rest-api|performance" /home/devuser/workspace/project/docs/reference/api/03-websocket.md
# Should show ./binary-protocol.md and ./rest-api.md
```

---

## Manual Verification (Interactive)

For critical files, verify manually by opening and checking:

1. **guides/xr-setup.md**
   - Search for "architecture" links
   - Verify all contain "concepts/architecture/"

2. **guides/navigation-guide.md**
   - Search for all "[" and "](architecture/" patterns
   - Verify all updated to "concepts/architecture/"

3. **reference/api/03-websocket.md**
   - Check for "../reference/api/" patterns
   - Verify all changed to "./" for same-directory files

---

## Rollback Instructions

If something goes wrong, you can revert all changes:

```bash
# Revert all sed changes by restoring from git
cd /home/devuser/workspace/project
git checkout docs/

# Or revert specific files
git checkout docs/guides/xr-setup.md
git checkout docs/guides/navigation-guide.md
git checkout docs/reference/api/03-websocket.md
```

---

## Success Criteria

✅ All fixes are successful when:

1. **No broken ../concepts/architecture/ paths** remain (except ../concepts/architecture/)
2. **No ../reference/reference/** paths exist
3. **8 links in navigation-guide.md** all point to concepts/architecture/
4. **3 links in reference/api/03-websocket.md** follow correct relative paths
5. **All other architecture links** updated to ../concepts/architecture/

---

## Expected Timeline

| Phase | Task | Duration |
|-------|------|----------|
| Phase 1 | Automated replacements | 30 minutes |
| Phase 2 | Manual verification | 1 hour |
| Phase 3 | Testing & validation | 1-2 hours |
| Phase 4 | Documentation update | 30 minutes |
| **Total** | | **3-4 hours** |

---

## Next Steps After Priority 2

Once all Priority 2 fixes are complete:

1. **Commit changes** to git with message: "Fix Priority 2: Architecture path corrections (27 links)"
2. **Run full link validation** to confirm all Priority 2 issues resolved
3. **Begin Priority 3** (missing content creation)
4. **Update documentation-audit-completion-report.md** with completion status

---

**Script Version**: 1.0  
**Last Updated**: 2025-11-04  
**Ready**: YES ✅
