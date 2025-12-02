# Archive Index - Documentation Reports

Complete listing of all archived documentation alignment reports and legacy files.

## Archive: documentation-alignment-2025-12-02

**Date Archived:** 2025-12-02
**Source:** Root directory cleanup and legacy report archival
**Reason:** Consolidate legacy reports after initial documentation alignment audit

### Archived Files

#### Main Reports (4 files)

| File | Size | Description |
|------|------|-------------|
| `DOCUMENTATION_ALIGNMENT_COMPLETE.md` | 11.4 KB | Full audit completion report with 363 lines detailing the documentation alignment skill creation, execution results, and comprehensive findings |
| `DOCUMENTATION_ALIGNMENT_SUMMARY.md` | 9.4 KB | Executive summary of documentation alignment audit with key metrics and recommendations |
| `DEEPSEEK_SETUP_COMPLETE.md` | 6.6 KB | DeepSeek LLM integration setup completion notes and configuration details |
| `SWARM_EXECUTION_REPORT.md` | 22 KB | Swarm execution report for documentation alignment audit with agent coordination details |

#### JSON Reports (10 files)

Location: `json-reports/`

| File | Size | Type | Description |
|------|------|------|-------------|
| `archive-report.json` | 11.3 KB | Legacy | Working document identification results (non-scoped) |
| `ascii-report.json` | 2.4 KB | Legacy | ASCII diagram detection results (non-scoped) |
| `link-report.json` | 792.6 KB | Legacy | Link validation report with 1,881 broken links (non-scoped) |
| `mermaid-report.json` | 117.4 KB | Legacy | Mermaid diagram syntax validation (non-scoped) |
| `stubs-report.json` | 270.8 KB | Legacy | TODO/FIXME/stub scanning results (non-scoped) |

**Note:** Scoped versions of these reports are maintained in `.doc-alignment-reports/` with `-scoped.json` suffix.

### Audit Statistics

**From DOCUMENTATION_ALIGNMENT_COMPLETE.md:**

| Metric | Count | Status |
|--------|-------|--------|
| Valid Links | 21,940 | ✅ Excellent |
| Broken Links | 1,881 | ⚠️ High Priority |
| Orphan Documents | 2,684 | ⚠️ Needs Review |
| Valid Mermaid Diagrams | 124 | ✅ Good |
| Invalid Mermaid Diagrams | 35 | ⚠️ Medium Priority |
| ASCII Diagrams | 4 | ℹ️ Low Priority |
| Working Docs to Archive | 13 | ℹ️ Housekeeping |
| Critical Stubs | 10 | ⚠️ Must Fix |
| TODOs/FIXMEs | 193 | ⚠️ Track |

### Archive Actions Completed

1. ✅ Moved 4 legacy markdown reports from root/docs to archive
2. ✅ Copied 5 non-scoped JSON reports to archive
3. ✅ Created archive directory structure
4. ✅ Documented archive contents and metadata
5. ✅ Cleaned root directory of legacy reports
6. ✅ Organised working files into `/docs/working/`
7. ✅ Organised feature docs into `/docs/features/`
8. ✅ Organised architecture docs into `/docs/architecture/`
9. ✅ Created README files for new documentation directories
10. ✅ Moved context files (`pipeline-files.txt`, `TotalContext.txt`) to working directory

### Current Active Reports

**Not Archived** (still in active use):

| Location | Files | Purpose |
|----------|-------|---------|
| `.doc-alignment-reports/` | `*-scoped.json` | Current scoped validation reports |
| `docs/` | `DOCUMENTATION_ISSUES.md` | Live issues tracking |
| `multi-agent-docker/skills/docs-alignment/` | `SKILL.md` + scripts | Active alignment skill |

### Related Archives

Other archive directories in the project:

| Path | Contents |
|------|----------|
| `/archive/` | Legacy project files from 2025-11 |
| `/archive/phase-5-reports-2025-11-06/` | Phase 5 historical reports |
| `/archive/working-docs-2025-11-06/` | November working documents |
| `/docs/archive/` | Historical architecture and migration docs |

### Retrieval Instructions

**View archived report:**
```bash
cat /home/devuser/workspace/project/docs/archive/reports/documentation-alignment-2025-12-02/DOCUMENTATION_ALIGNMENT_COMPLETE.md
```

**Parse archived JSON:**
```bash
cd /home/devuser/workspace/project/docs/archive/reports/documentation-alignment-2025-12-02/json-reports
jq '.summary' link-report.json
jq '.diagrams[] | select(.valid == false)' mermaid-report.json
```

**Compare with current reports:**
```bash
# Legacy (archived)
jq '.total_links' docs/archive/reports/documentation-alignment-2025-12-02/json-reports/link-report.json

# Current (active)
jq '.total_links' .doc-alignment-reports/link-report-scoped.json
```

### Archive Metadata

| Property | Value |
|----------|-------|
| **Archive Created** | 2025-12-02 10:54 UTC |
| **Created By** | Archive Cleanup Agent |
| **Archive Size** | ~1.2 MB (reports + JSON) |
| **File Count** | 14 files (4 MD + 10 JSON) |
| **Retention Policy** | Indefinite (historical reference) |
| **Superseded By** | `.doc-alignment-reports/*-scoped.json` reports |

### Future Archive Policy

**Reports will be archived when:**
- New major validation runs complete
- Legacy report formats are replaced
- Significant codebase changes require re-validation
- Quarterly documentation audit cycles complete

**Archive naming convention:**
```
docs/archive/reports/documentation-alignment-YYYY-MM-DD/
```

---

**Archive Status:** ✅ Complete
**Index Version:** 1.0
**Last Updated:** 2025-12-02
**Next Review:** After next major documentation alignment audit
