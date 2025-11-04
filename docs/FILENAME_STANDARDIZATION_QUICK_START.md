# Filename Standardization - Quick Start Guide

**Last Updated:** 2025-11-04
**Full Plan:** See `FILENAME_STANDARDIZATION_EXECUTION_PLAN.md`

---

## 🚀 Quick Execution Steps

### Pre-Flight Checklist

```bash
# 1. Navigate to project
cd /home/devuser/workspace/project

# 2. Create feature branch
git checkout -b docs/filename-standardization

# 3. Create backup
tar -czf ~/docs-backup-$(date +%Y%m%d).tar.gz docs/

# 4. Verify scripts are executable
chmod +x docs/scripts/*.sh

# 5. Test validation script (dry run)
DRY_RUN=true docs/scripts/update-all-references.sh phase1
```

---

## 📋 Phase Execution Commands

### Phase 1: Critical Duplicates (2-3 hours)

**Action 1.1.1: Development Setup**
```bash
# Compare files
diff docs/guides/developer/development-setup.md \
     docs/guides/developer/01-development-setup.md

# Merge unique content manually
# Then delete duplicate
git rm docs/guides/developer/development-setup.md
git commit -m "docs: merge development-setup into 01-development-setup (Phase 1.1.1)"
```

**Action 1.1.2: Adding Features**
```bash
# Compare and merge
diff docs/guides/developer/adding-a-feature.md \
     docs/guides/developer/04-adding-features.md

git rm docs/guides/developer/adding-a-feature.md
git commit -m "docs: merge adding-a-feature into 04-adding-features (Phase 1.1.2)"
```

**Action 1.1.3: Testing Guide**
```bash
# Rename 05-testing to 05-testing-guide
git mv docs/guides/developer/testing-guide.md \
       docs/guides/developer/05-testing-guide.md

# Merge both testing-guide.md files into it
# Then delete duplicates
git rm docs/guides/developer/testing-guide.md
git rm docs/guides/testing-guide.md
git commit -m "docs: consolidate testing guides (Phase 1.1.3)"
```

**Action 1.1.4: XR Setup Differentiation**
```bash
# Edit both files to add cross-references
# Update titles and frontmatter
git add docs/guides/xr-setup.md docs/guides/user/xr-setup.md
git commit -m "docs: differentiate XR setup guides (Phase 1.1.4)"
```

**Update References**
```bash
# Update all references to renamed/merged files
docs/scripts/update-all-references.sh phase1

# Validate
docs/scripts/validate-links.sh

git add -A
git commit -m "docs: update references after Phase 1 merges"
```

---

### Phase 2: Numbering Conflicts (30 minutes)

**Action 2.1.1: Testing Status**
```bash
# Analyze content first
cat docs/guides/developer/testing-guide.md

# Option A: Move to reports
git mv docs/guides/developer/testing-guide.md \
       docs/reports/testing-status-2025-10-27.md

# Option B: Merge into 05-testing-guide.md
# (Manually merge content)
git rm docs/guides/developer/testing-guide.md

git commit -m "docs: resolve testing-status numbering conflict (Phase 2.1.1)"
```

**Action 2.2.1: API Reference Numbering**
```bash
# Create 02-rest-api.md by consolidating
# (Manually create from rest-api-reference.md and rest-api-complete.md)

git add docs/reference/api/02-rest-api.md
git commit -m "docs: create 02-rest-api.md to complete API sequence (Phase 2.2.1)"
```

---

### Phase 3: Case Normalization (1-2 hours)

**Action 3.1: Move Reports**
```bash
# Create report directories
mkdir -p docs/reports/{deprecation,audits}

# Move and rename GRAPHSERVICEACTOR files
for f in docs/GRAPHSERVICEACTOR_DEPRECATION_*.md; do
  basename=$(basename "$f" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  git mv "$f" "docs/reports/deprecation/$basename"
done

# Move audit reports
git mv docs/ALIGNMENT_REPORT.md \
       docs/reports/audits/alignment-report-2025-11-04.md

git mv docs/DOCUMENTATION_AUDIT_COMPLETION_REPORT.md \
       docs/reports/audits/documentation-audit-completion-2025-11-04.md

git mv docs/LINK_VALIDATION_REPORT.md \
       docs/reports/audits/link-validation-report-2025-11-04.md

git commit -m "docs: move reports to /reports/ directory (Phase 3.1)"
```

**Action 3.2: Rename Architecture Files**
```bash
cd docs/concepts/architecture

# Rename SCREAMING_SNAKE_CASE files
git mv 00-ARCHITECTURE-OVERVIEW.md 00-architecture-overview.md
git mv CQRS_DIRECTIVE_TEMPLATE.md cqrs-directive-template.md
git mv PIPELINE_INTEGRATION.md pipeline-integration.md
git mv PIPELINE_SEQUENCE_DIAGRAMS.md pipeline-sequence-diagrams.md
git mv QUICK_REFERENCE.md quick-reference.md

cd ../../..
git commit -m "docs: normalize architecture filenames to kebab-case (Phase 3.2)"
```

**Action 3.3: Rename Other Directories**
```bash
# Multi-agent-docker
cd docs/multi-agent-docker
git mv ARCHITECTURE.md architecture.md
git mv DOCKER-ENVIRONMENT.md docker-environment.md
git mv GOALIE-INTEGRATION.md goalie-integration.md
git mv PORT-CONFIGURATION.md port-configuration.md
git mv TOOLS.md tools.md
git mv TROUBLESHOOTING.md troubleshooting.md

cd ../..
git commit -m "docs: normalize multi-agent-docker filenames (Phase 3.3a)"

# Other files
git mv docs/guides/operations/PIPELINE_OPERATOR_RUNBOOK.md \
       docs/guides/operations/pipeline-operator-runbook.md

git mv docs/implementation/STRESS_MAJORIZATION_IMPLEMENTATION.md \
       docs/implementation/stress-majorization-implementation.md

git commit -m "docs: normalize remaining SCREAMING_SNAKE_CASE files (Phase 3.3b)"
```

**Update All References**
```bash
# Update references for all Phase 3 changes
docs/scripts/update-all-references.sh phase3

# Validate
docs/scripts/validate-links.sh

git add -A
git commit -m "docs: update references after Phase 3 case normalization"
```

---

### Phase 4: Disambiguation (1 hour)

**Action 4.1: Semantic Physics**
```bash
git mv docs/concepts/architecture/semantic-physics.md \
       docs/concepts/architecture/semantic-physics-overview.md

git mv docs/concepts/architecture/semantic-physics-system.md \
       docs/concepts/architecture/semantic-physics-architecture.md

git mv docs/reference/semantic-physics-implementation.md \
       docs/reference/semantic-physics-api-reference.md

git commit -m "docs: disambiguate semantic physics files (Phase 4.1)"
```

**Action 4.2: REST API**
```bash
git mv docs/reference/api/rest-api-complete.md \
       docs/reference/api/rest-api-detailed-spec.md

# Note: rest-api-reference.md becomes 02-rest-api.md in Phase 2

git commit -m "docs: disambiguate REST API documentation (Phase 4.2)"
```

**Action 4.3: Additional Disambiguation**
```bash
git mv docs/concepts/architecture/reasoning-tests-summary.md \
       docs/concepts/architecture/reasoning-test-results.md

git mv docs/concepts/ontology-reasoning.md \
       docs/concepts/ontology-reasoning-concepts.md

git commit -m "docs: additional file disambiguation (Phase 4.3)"
```

**Update All References**
```bash
# Update references for Phase 4 changes
docs/scripts/update-all-references.sh phase4

# Validate
docs/scripts/validate-links.sh

git add -A
git commit -m "docs: update references after Phase 4 disambiguation"
```

---

## 🔍 Validation Commands

### During Execution

```bash
# Check for broken links
docs/scripts/validate-links.sh

# Find orphaned files
docs/scripts/find-orphaned-files.sh

# Verify no SCREAMING_SNAKE_CASE remains (except README, CONTRIBUTING)
find docs -name "*[A-Z_][A-Z_]*.md" | grep -v "README\|CONTRIBUTING"

# Check numbering sequences
ls -1 docs/guides/developer/[0-9]*.md
ls -1 docs/reference/api/[0-9]*.md
```

### Final Validation

```bash
# Complete validation suite
docs/scripts/validate-links.sh
docs/scripts/find-orphaned-files.sh

# Check git status
git status

# Review all changes
git log --oneline docs/filename-standardization

# Count files changed
git diff --stat main..docs/filename-standardization
```

---

## 🔄 Rollback Commands

### Rollback Single Phase

```bash
# Find commit before phase
git log --oneline

# Reset to that commit
git reset --hard <commit-hash>
```

### Rollback Everything

```bash
# Switch back to main
git checkout main

# Delete feature branch
git branch -D docs/filename-standardization

# Restore from backup
tar -xzf ~/docs-backup-*.tar.gz -C /
```

---

## 📊 Progress Tracking

### Phase Checklist

- [ ] **Phase 1: Critical Duplicates**
  - [ ] 1.1.1 Development setup merged
  - [ ] 1.1.2 Adding features merged
  - [ ] 1.1.3 Testing guides consolidated
  - [ ] 1.1.4 XR setup differentiated
  - [ ] 1.2.1 Hierarchical visualization resolved
  - [ ] 1.2.2 Neo4j integration resolved
  - [ ] References updated
  - [ ] Links validated

- [ ] **Phase 2: Numbering Conflicts**
  - [ ] 2.1.1 Testing status resolved
  - [ ] 2.2.1 API reference sequence completed
  - [ ] Links validated

- [ ] **Phase 3: Case Normalization**
  - [ ] 3.1 Reports moved to /reports/
  - [ ] 3.2 Architecture files renamed
  - [ ] 3.3 Other directories normalized
  - [ ] References updated
  - [ ] Links validated

- [ ] **Phase 4: Disambiguation**
  - [ ] 4.1 Semantic physics files renamed
  - [ ] 4.2 REST API files disambiguated
  - [ ] 4.3 Additional files renamed
  - [ ] References updated
  - [ ] Links validated

- [ ] **Final Validation**
  - [ ] All links working
  - [ ] No orphaned files
  - [ ] No SCREAMING_SNAKE_CASE
  - [ ] All sequences correct
  - [ ] Git history clean

---

## 🆘 Common Issues & Solutions

### Issue: Merge Conflicts

**Solution:**
```bash
# Manually resolve conflicts
git status
# Edit conflicted files
git add <resolved-files>
git commit
```

### Issue: Broken Links After Update

**Solution:**
```bash
# Re-run reference update for specific phase
docs/scripts/update-all-references.sh phase1  # or phase3, phase4

# Check specific file
grep -n "broken-link.md" docs/**/*.md
```

### Issue: Lost Content During Merge

**Solution:**
```bash
# Check backup
tar -tzf ~/docs-backup-*.tar.gz | grep <filename>

# Extract specific file
tar -xzf ~/docs-backup-*.tar.gz docs/path/to/file.md
```

### Issue: Script Permissions

**Solution:**
```bash
chmod +x docs/scripts/*.sh
```

---

## 📈 Success Metrics

**Completed When:**
- ✅ 30 files processed according to plan
- ✅ Zero broken internal links
- ✅ All SCREAMING_SNAKE_CASE converted
- ✅ All numbering sequences valid
- ✅ All cross-references updated
- ✅ Git history clean and documented
- ✅ No orphaned files
- ✅ Documentation easily navigable

---

## 📞 Support

**Documentation:**
- Full plan: `FILENAME_STANDARDIZATION_EXECUTION_PLAN.md`
- Scripts: `docs/scripts/`

**Questions:**
- Create issue: Document in `/docs/reports/filename-standardization-issues.md`
- Rollback: See "Rollback Commands" section above

---

## 🎯 Quick Tips

1. **Always commit after each action** - Makes rollback easier
2. **Validate after each phase** - Catch issues early
3. **Keep backup** - Safety net for mistakes
4. **Test scripts in dry-run mode first** - Verify before executing
5. **Document issues** - Help future standardization efforts

---

**Ready to start?** Begin with Phase 1, Action 1.1.1 above! 🚀
