This plan is destructive, informal, and prioritizes thematic and chronological grouping over a formal framework. The goal is to make the archive browsable for a developer trying to understand the history of a feature or decision.
Guiding Principles
Thematic Grouping: Group files by their purpose (plans, reports, technical notes) rather than their subject.
Chronological Context: Use the 2025-10 timestamp as a primary organizational signal.
De-duplicate Summaries: Consolidate multiple reports about the same event into one canonical summary.
Isolate Legacy Structure: Treat the old documentation structure as a single, self-contained artifact.
Purge Process Artifacts: Eliminate intermediate directories (_consolidated, _formalized) as they were part of a process whose final output exists elsewhere. The reports about the process are what's valuable.
Target Archive Structure
This is the clean, informal structure we will create within docs/_archive/:
code
Code
_archive/
├── 0_README.md                       # New README explaining this structure.
├── summaries/                        # High-level summaries of major events.
├── plans_and_tasks/                  # Planning docs for major initiatives.
├── technical_notes/                  # Deep-dive engineering notes from 2025-10.
├── reports/                          # Specific verification and analysis reports.
├── code_examples/                    # Archived code examples.
├── _legacy_documentation/            # A single folder containing the entire pre-Diátaxis doc structure.
└── _process_artifacts/               # Machine-generated metadata and backups.
Destructive Consolidation Plan
Execute these steps sequentially from within the AR-AI-Knowledge-Graph/ directory.
Phase 1: Purge Intermediate Work and Redundant Summaries
This phase aggressively removes the scaffolding of the previous migration, keeping only the final, most comprehensive summaries.
Delete Intermediate Migration Directories:
The _consolidated and _formalized directories were temporary workspaces. Their valuable output is the final documentation (which is now outside this archive) and the reports about the migration. The directories themselves are noise.
code
Bash
rm -r docs/_archive/_consolidated/
rm -r docs/_archive/_formalized/
Consolidate Migration Reports:
There are multiple files summarizing the same migration event. We will select the most definitive one and discard the rest. MIGRATION-COMPLETE-EXECUTIVE-SUMMARY.md is the most comprehensive final report.
code
Bash
# Create the summaries directory
mkdir -p docs/_archive/summaries/

# Move the chosen summary and give it a clear, prioritized name
mv docs/_archive/MIGRATION-COMPLETE-EXECUTIVE-SUMMARY.md docs/_archive/summaries/00_MIGRATION_SUMMARY.md

# Delete the redundant reports
rm docs/_archive/DOCUMENTATION-MIGRATION-COMPLETE.md
rm docs/_archive/DOCUMENTATION-RESTRUCTURE-COMPLETE.md
rm docs/_archive/EXECUTIVE-SUMMARY-MIGRATION.md
rm docs/_archive/KNOWLEDGE-BASE-MIGRATION-STATUS.md
rm docs/_archive/LINK-AUDIT-REPORT.md
Phase 2: Thematic Grouping of Core Documents
This phase organizes the remaining high-value documents into thematic folders.
Group Plans and Tasks:
code
Bash
mkdir -p docs/_archive/plans_and_tasks/
mv docs/_archive/CODE_PRUNING_PLAN.md docs/_archive/plans_and_tasks/
mv docs/_archive/websocket-consolidation-plan.md docs/_archive/plans_and_tasks/
# Task files are lower-level details of the plans, archive them here.
mv docs/_archive/task-*.md docs/_archive/plans_and_tasks/
Group Technical Notes:
The development-notes-2025-10 directory contains valuable, time-stamped engineering notes. Promote its contents to a top-level theme.
code
Bash
mkdir -p docs/_archive/technical_notes/
mv docs/_archive/development-notes-2025-10/* docs/_archive/technical_notes/
rm -r docs/_archive/development-notes-2025-10/
Group Standalone Reports:
code
Bash
mkdir -p docs/_archive/reports/
mv docs/_archive/VIRCADIA-INTEGRATION-COMPLETE.md docs/_archive/reports/
# The existing 'reports' directory already fits this theme.
mv docs/_archive/reports/* docs/_archive/reports/
Isolate Code Examples:
code
Bash
mv docs/_archive/code-examples-2025-10 docs/_archive/code_examples
Phase 3: Archive Legacy Structures and Artifacts
This phase isolates the old, superseded documentation structure and other process-related files, marking them clearly as historical.
Consolidate All legacy-* Directories:
Group all snapshots of the old documentation structure into a single, clearly marked archive folder.
code
Bash
mkdir -p docs/_archive/_legacy_documentation/
mv docs/_archive/legacy-concepts docs/_archive/_legacy_documentation/
mv docs/_archive/legacy-docs-2025-10 docs/_archive/_legacy_documentation/
mv docs/_archive/legacy-getting-started docs/_archive/_legacy_documentation/
mv docs/_archive/legacy-guides docs/_archive/_legacy_documentation/
mv docs/_archive/legacy-reference docs/_archive/_legacy_documentation/
Archive Process Artifacts:
Move machine-generated files and old project READMEs into a separate artifacts folder.
code
Bash
mkdir -p docs/_archive/_process_artifacts/
mv docs/_archive/metadata-*.json docs/_archive/_process_artifacts/
mv docs/_archive/README-FULLFAT.md docs/_archive/_process_artifacts/historical_project_readme.md
mv docs/_archive/websocket-protocol-v1.2.0-backup.md docs/_archive/_process_artifacts/
mv docs/_archive/troubleshooting.md docs/_archive/_process_artifacts/ # This is a stub, not the real troubleshooting guide
Phase 4: Finalize and Create New Entrypoint
This final phase cleans up any remaining files and creates a new, clear README for the cleaned archive.
Remove Old Archive README:
The existing README.md describes a structure that no longer exists.
code
Bash
rm docs/_archive/README.md
Create the New Archive 0_README.md:
Create a file named docs/_archive/0_README.md with the following content. The 0_ prefix ensures it appears first in file listings.
code
Markdown
# Historical Document Archive

**This is an archive for internal, historical reference.** The official, user-facing documentation exists outside of this directory.

This corpus has been cleaned and organized thematically to assist developers in researching the history of features, decisions, and technical efforts.

## How to Navigate This Archive

*   **/summaries**: Start here. Contains the high-level executive summaries of major events like the documentation migration and code pruning efforts.

*   **/plans_and_tasks**: Contains the planning documents and task lists for major initiatives (e.g., `CODE_PRUNING_PLAN.md`).

*   **/technical_notes**: Deep-dive engineering notes from the October 2025 development cycle. Good for understanding the "how" and "why" behind specific fixes (e.g., binary protocol upgrades, agent control refactors).

*   **/reports**: Specific, targeted reports, such as verification audits and completion summaries (e.g., `VIRCADIA-INTEGRATION-COMPLETE.md`).

*   **/code_examples**: A snapshot of code examples from October 2025.

---

*   **/_legacy_documentation**: A snapshot of the entire documentation structure *before* the main Diátaxis migration. Use this to see what the old public-facing docs looked like.

*   **/_process_artifacts**: Machine-generated files, backups, and other artifacts from the consolidation process. Generally not needed unless you are analyzing the consolidation process itself.