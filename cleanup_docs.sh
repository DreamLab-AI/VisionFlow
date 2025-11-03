#!/bin/bash
set -e
echo "Starting documentation cleanup..."

# Directories to be removed entirely
rm -rf docs/archive/
rm -rf docs/migration/
rm -rf docs/refactoring/
rm -rf docs/multi-agent-docker/docs/

# Specific files to be removed
rm -f docs/DOCUMENTATION_INDEX.md
rm -f docs/INDEX.md
rm -f docs/QUICK_NAVIGATION.md
rm -f docs/architecture/00-READ-ME-FIRST.md
rm -f docs/architecture/README_MIGRATION_STATUS.md
rm -f docs/architecture/component-status.md
rm -f docs/CODEBASE_AUDIT_FUNCTION_INVENTORY.md
rm -f docs/database-schema-diagrams.md
rm -f docs/DOCUMENTATION_ARCHITECTURE_DESIGN.md
rm -f docs/DOCUMENTATION_REFACTORING_COMPLETE.md
rm -f docs/DOCUMENTATION_REFACTORING_PLAN.md
rm -f docs/FINAL_REPORT_TASK_1.4.md
rm -f docs/GPU_CLEANUP_REPORT_2025_11_03.md
rm -f docs/gpu_consolidation_report_2025_11_03.md
rm -f docs/gpu_memory_consolidation_analysis.md
rm -f docs/gpu_memory_consolidation_report.md
rm -f docs/GPU_MEMORY_MIGRATION.md
rm -f docs/GPU_MEMORY_SUMMARY.txt
rm -f docs/http_response_migration_report.md
rm -f docs/MCP_CLIENT_CONSOLIDATION.md
rm -f docs/MIGRATION_COMPLETION_REPORT.md
rm -f docs/PHASE_4_CONSOLIDATION_REPORT.md
rm -f docs/PHASE_4_CONSOLIDATION_SUMMARY.md
rm -f docs/PHASE_5_COMPLETION_REPORT.md
rm -f docs/PHASE1_COMPLETION_REPORT.md
rm -f docs/PHASE1_COORDINATION_REPORT.md
rm -f docs/PHASE1_EXECUTION_GUIDE.md
rm -f docs/phase1-task-1.1-repository-architect.md
rm -f docs/phase1-task-1.2-completion-report.md
rm -f docs/phase1-task-1.2-safety-engineer.md
rm -f docs/phase1-task-1.3-completion-report.md
rm -f docs/phase1-task-1.3-json-specialist.md
rm -f docs/phase1-task-1.4-api-specialist.md
rm -f docs/phase1-task-1.5-time-specialist.md
rm -f docs/PHASE2_COMPLETION_REPORT.md
rm -f docs/PHASE2_TASK2.1_SUMMARY.md
rm -f docs/PHASE2_TASK2.6_COMPLETION_REPORT.md
rm -f docs/REFACTORING_NOTES.md
rm -f docs/REFACTORING_PLAN_SUMMARY.md
rm -f docs/REFACTORING_ROADMAP_DETAILED.md
rm -f docs/REFACTORING_VERIFICATION_REPORT.md
rm -f docs/REPOSITORY_DUPLICATION_ANALYSIS.md
rm -f docs/architecture/CQRS_MIGRATION_COMPLETE.md
rm -f docs/architecture/CQRS_MIGRATION_SUMMARY.md
rm -f docs/architecture/PIPELINE_INTEGRATION_COMPLETE.md

echo "Cleanup complete."