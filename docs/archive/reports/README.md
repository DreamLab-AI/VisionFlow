---
title: Documentation Reports Archive
description: This directory contains archived documentation alignment reports and legacy status files.
type: archive
status: archived
---

# Documentation Reports Archive

This directory contains archived documentation alignment reports and legacy status files.

## Archive Structure

### documentation-alignment-2025-12-02/
Historical documentation alignment audit from December 2nd, 2025.

**Contents:**
- `DOCUMENTATION_ALIGNMENT_COMPLETE.md` - Full audit completion report
- `DOCUMENTATION_ALIGNMENT_SUMMARY.md` - Executive summary of findings
- `DEEPSEEK_SETUP_COMPLETE.md` - DeepSeek integration setup notes
- `json-reports/` - Legacy JSON validation reports (non-scoped versions)
  - `link-report.json` - Link validation results
  - `mermaid-report.json` - Mermaid diagram validation
  - `ascii-report.json` - ASCII diagram detection
  - `archive-report.json` - Working document identification
  - `stubs-report.json` - TODO/FIXME scanning results

**Key Metrics from Audit:**
- 21,940 valid links verified
- 1,881 broken links identified
- 2,684 orphan documents detected
- 159 diagrams validated (124 valid, 35 invalid)
- 193 TODOs/FIXMEs tracked
- 10 critical code stubs identified

## Active Reports

Current documentation alignment reports are maintained in:
- `.doc-alignment-reports/*-scoped.json` - Current scoped validation reports
- `docs/DOCUMENTATION_ISSUES.md` - Live issues tracking

## Archive Policy

Reports are archived when:
1. A new major validation run completes
2. Significant codebase changes require re-validation
3. Legacy formats are replaced by improved tooling

**Archive Retention:** Indefinite (historical reference)

## Viewing Archived Reports

All archived reports are standard markdown and JSON files:

```bash
# View completion report
cat docs/archive/reports/documentation-alignment-2025-12-02/DOCUMENTATION_ALIGNMENT_COMPLETE.md

# Parse JSON reports
jq '.summary' docs/archive/reports/documentation-alignment-2025-12-02/json-reports/link-report.json
```

## Related Documentation

- `/multi-agent-docker/skills/docs-alignment/` - Documentation alignment skill
- `/.doc-alignment-reports/` - Current validation reports
- `/docs/DOCUMENTATION_ISSUES.md` - Active issues list

---

**Last Updated:** 2025-12-02
**Maintained By:** Archive Cleanup Agent
