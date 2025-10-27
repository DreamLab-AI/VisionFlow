# Documentation Refactoring - Quick Start Guide

**Status**: Ready for Execution
**Target**: Category 1 - Critical Contradictions
**Estimated Time**: 7.5 hours (single person) or 3 hours (3-person team)

## What Was Done

I've analyzed 300+ documentation files and created an **executable plan** to resolve 5 critical contradictions:

1. **Binary Protocol** - Fixed "38 bytes" vs "36 bytes" contradiction
2. **API Ports** - Standardized to port 3030 (removed 8080/3001 references)
3. **Deployment** - Clarified Docker NOT implemented for main project
4. **Developer Setup** - Updated with actual dependencies and tools
5. **Testing Status** - Documented honest test coverage (no CI yet)

## Quick Execution

### Option 1: Run Full Plan (Recommended)

```bash
# Read the full plan
cat docs/refactoring/CATEGORY-1-EXECUTABLE-PLAN.md

# Execute tasks 1.1 through 1.6 in order
# Each task has step-by-step bash commands

# Validate when done
bash docs/refactoring/validate-category1.sh
```

### Option 2: Automated Quick Fixes

```bash
# Task 1.1: Fix binary protocol comment
sed -i 's/38 bytes per node/36 bytes per node/' src/utils/binary_protocol.rs

# Task 1.2: Fix API ports
find docs/ -type f -name "*.md" -exec sed -i \
  -e 's|localhost:8080|localhost:3030|g' \
  -e 's|localhost:3030|localhost:3030|g' \
  -e 's|:8080/|:3030/|g' \
  -e 's|:3030/|:3030/|g' \
  {} \;

# Validate
bash docs/refactoring/validate-category1.sh
```

## Files Created

| File | Purpose |
|------|---------|
| `docs/refactoring/CATEGORY-1-EXECUTABLE-PLAN.md` | Complete execution guide |
| `docs/refactoring/validate-category1.sh` | Validation script (22 tests) |
| `docs/refactoring/QUICK-START.md` | This file |

## What's in the Plan

### Task 1.1: Binary Protocol Standardization (30 min)
- **Problem**: Docs claim 38 bytes, code uses 36 bytes
- **Truth**: Protocol V2 is 36 bytes (u32 ID + 3 Vec3s + 2 floats)
- **Fix**: Update 1 source file comment
- **Files**: `src/utils/binary_protocol.rs`

### Task 1.2: API Port Standardization (1 hour)
- **Problem**: Mixed references to ports 3030, 8080, 3001
- **Truth**: Server uses port 3030 (SYSTEM_NETWORK_PORT env var)
- **Fix**: Replace all 8080/3001 → 3030 in docs
- **Files**: 12+ documentation files

### Task 1.3: Deployment Consolidation (2 hours)
- **Problem**: Contradictory deployment approaches
- **Truth**: No Docker for main project (Rust binary + Nginx)
- **Fix**: Create unified deployment README, clarify multi-agent-docker separation
- **Files**: Create `docs/deployment/README.md`, update 4 others

### Task 1.4: Developer Guide Update (2 hours)
- **Problem**: Outdated tools, wrong test framework
- **Truth**: Rust 1.75+, Actix-web, PostgreSQL, Vitest (not Jest)
- **Fix**: Rewrite developer setup guide with accurate stack
- **Files**: `docs/developer-guide/01-development-setup.md`, `05-testing.md`

### Task 1.5: Testing Documentation (1 hour)
- **Problem**: False claims of automated CI
- **Truth**: Manual testing only, no .github/workflows/
- **Fix**: Create honest testing status report
- **Files**: Create `docs/testing-status.md`, update CI claims

### Task 1.6: Cross-Cutting Updates (1 hour)
- **Fix**: Update index, create changelog, mark authoritative docs
- **Files**: `docs/00-INDEX.md`, `docs/refactoring/CHANGELOG.md`, `README.md`

## Validation Tests (22 total)

The `validate-category1.sh` script checks:

✅ No "38 bytes" references
✅ No ports 8080/3001 in docs
✅ Deployment README exists
✅ Docker separation clarified
✅ Developer guide has Rust version
✅ Testing docs mention Vitest (not Jest)
✅ Testing status documented
✅ CI claims accurate (not implemented)
✅ Documentation index updated
✅ Changelog exists
✅ `cargo build` succeeds
✅ `cargo check` passes

## Before You Start

### Prerequisites
- Git repository clean (no uncommitted changes)
- Rust toolchain installed (`cargo --version`)
- Read access to all documentation files
- Write access to docs/ directory

### Backup Strategy
Each task includes rollback procedures. The script creates `.bak` files automatically.

```bash
# Full backup before starting
tar -czf docs-backup-$(date +%Y%m%d).tar.gz docs/

# Rollback all if needed
tar -xzf docs-backup-YYYYMMDD.tar.gz
```

## Execution Strategy

### Single Person (7.5 hours)
Execute tasks 1.1 → 1.6 sequentially. Use the exact bash commands in the plan.

### Two People (4 hours)
- Person A: Tasks 1.1, 1.2 (1.5 hours)
- Person B: Tasks 1.3, 1.4 (4 hours, starts immediately)
- Both: Tasks 1.5, 1.6 together (1.5 hours)

### Three People (3 hours)
- Person A: Tasks 1.1, 1.2 (1.5 hours)
- Person B: Tasks 1.3, 1.5 (3 hours)
- Person C: Task 1.4 (2 hours)
- All: Task 1.6 together (1 hour)

## After Completion

1. **Run validation**: `bash docs/refactoring/validate-category1.sh`
2. **Commit changes**: Use commit messages from the plan
3. **Create PR**: Tag reviewers, link to this plan
4. **Update README**: Mark Category 1 as complete

## Troubleshooting

### Validation fails on test X
- See the plan's section for that task
- Check rollback procedure
- Review git diff for that file

### Merge conflicts
- Plan assumes clean repository
- Stash changes: `git stash`
- Apply plan
- Restore: `git stash pop`

### Build breaks after changes
- Most changes are docs-only (won't break build)
- Task 1.1 touches code but only comments
- If broken: `git checkout src/utils/binary_protocol.rs`

## Next Categories

After Category 1:
- **Category 2**: Outdated Information (remove deprecated features)
- **Category 3**: Documentation Gaps (add missing sections)
- **Category 4**: Structural Issues (reorganize)
- **Category 5**: Maintenance (automation)

## Support

- **Full Plan**: `docs/refactoring/CATEGORY-1-EXECUTABLE-PLAN.md`
- **Validation**: `docs/refactoring/validate-category1.sh`
- **Questions**: Open an issue with tag `[refactoring]`

---

**Ready to Execute**: Yes ✅
**Risk Level**: Low (mostly docs, one comment change)
**Reversible**: Yes (git revert + .bak files)
**Testing**: Automated validation script included
