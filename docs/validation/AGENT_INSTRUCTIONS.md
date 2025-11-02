# Documentation Update & Validation - Agent Instructions

**Mission:** Update all VisionFlow documentation to reflect unified.db architecture

**Coordination:** All agents write findings to `/home/devuser/workspace/project/docs/validation/`

---

## Agent 1: README.md Updater

**Files:** `/home/devuser/workspace/project/README.md`

**Tasks:**
1. Update line 146: Change "Unified database design (single unified.db with all domain tables)" - Already correct ✅
2. Update line 245: Verify mermaid diagram shows single "unified.db" - NEEDS UPDATE
3. Update line 288: Verify data layer description mentions unified.db only - Already correct ✅
4. Remove any remaining references to dual/triple databases
5. Document FORCE_FULL_SYNC environment variable (line 330)

**Output:** `/home/devuser/workspace/project/docs/validation/readme-updates.md`

---

## Agent 2: Task Documentation Updater

**Files:**
- `/home/devuser/workspace/project/task.md`
- `/home/devuser/workspace/project/docs/task.md` (if exists)

**Tasks:**
1. Verify all database references point to unified.db ✅ (already complete in task.md)
2. Update pipeline diagrams to show unified architecture
3. Remove any legacy database cleanup tasks (already archived)
4. Ensure FORCE_FULL_SYNC is documented

**Output:** `/home/devuser/workspace/project/docs/validation/task-docs-updates.md`

---

## Agent 3: Architecture Documentation Updater

**Files:**
- `/home/devuser/workspace/project/docs/architecture/00-ARCHITECTURE-OVERVIEW.md`
- `/home/devuser/workspace/project/docs/architecture/04-database-schemas.md`
- `/home/devuser/workspace/project/docs/DATABASE_CLEANUP_PLAN.md`
- `/home/devuser/workspace/project/docs/CLEANUP_COMPLETION_REPORT.md`

**Tasks:**
1. Update 00-ARCHITECTURE-OVERVIEW.md lines 38-75 to reflect single unified.db
2. Remove "three separate databases" references
3. Update all mermaid diagrams to show unified database
4. Mark DATABASE_CLEANUP_PLAN.md as ARCHIVED/COMPLETED
5. Update CLEANUP_COMPLETION_REPORT.md with final status

**Output:** `/home/devuser/workspace/project/docs/validation/architecture-updates.md`

---

## Agent 4: Link Validator

**Files:** All 33 files with database references

**Tasks:**
1. Validate all internal markdown links work
2. Check for broken cross-references
3. Verify code block references point to existing files
4. Generate link validation report

**Output:** `/home/devuser/workspace/project/docs/validation/link-validation-report.md`

---

## Agent 5: Mermaid Diagram Validator

**Files:** All 85 files with mermaid diagrams

**Tasks:**
1. Parse all mermaid diagrams for syntax errors
2. Identify diagrams referencing old database architecture
3. Suggest fixes for broken diagrams
4. Generate mermaid validation report with line numbers

**Output:** `/home/devuser/workspace/project/docs/validation/mermaid-validation-report.md`

---

## Agent 6: API Documentation Updater

**Files:**
- `/home/devuser/workspace/project/docs/reference/api/`
- `/home/devuser/workspace/project/docs/api/`

**Tasks:**
1. Update API documentation to reflect unified database
2. Remove references to separate database endpoints
3. Document any new endpoints for unified architecture
4. Update database schema references

**Output:** `/home/devuser/workspace/project/docs/validation/api-docs-updates.md`

---

## Agent 7: Consistency Checker

**Files:** All documentation files

**Tasks:**
1. Cross-reference all 6 agent reports
2. Identify remaining inconsistencies
3. Verify all old database references removed
4. Generate master validation report
5. Create prioritized fix list

**Output:** `/home/devuser/workspace/project/docs/validation/MASTER_VALIDATION_REPORT.md`

---

## Coordination Protocol

**Before starting:**
```bash
npx claude-flow@alpha hooks pre-task --description "Documentation validation for unified.db"
npx claude-flow@alpha hooks session-restore --session-id "swarm-doc-validation"
```

**After completing:**
```bash
npx claude-flow@alpha hooks post-task --task-id "doc-validation-[agent-name]"
npx claude-flow@alpha hooks notify --message "Completed [agent-name] validation"
```

**Memory keys:**
- `swarm/doc-validation/[agent-name]/status`
- `swarm/doc-validation/[agent-name]/findings`
- `swarm/doc-validation/[agent-name]/completion_time`
