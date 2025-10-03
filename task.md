Project: Knowledge Base Consolidation Plan
1. Vision & Goals
The primary goal is to transform the existing docs directory from a development-focused, somewhat disorganized collection into a polished, public-facing knowledge base. This new structure will serve as the single source of truth for all users, developers, and contributors.
Success Criteria:
Clarity & Accessibility: Information is easy to find, read, and understand for all target audiences.
Consistency: Uniform structure, naming conventions, and formatting are applied across the entire corpus.
Navigability: A robust, bidirectional linking system allows users to seamlessly move between related topics.
Accuracy: All duplicated content is removed, and information is consolidated into canonical documents.
Maintainability: The new structure is logical and easy for the team to maintain and extend.
2. Proposed Information Architecture
The new structure is based on the Diátaxis framework, organizing content by user need into four distinct categories. This is a standard and effective model for technical documentation.
Target Directory Structure:
code
Code
AR-AI-Knowledge-Graph/
├── README.md              # High-level project overview, value prop, quick start, link to docs
└── docs/
    ├── index.md             # Knowledge base landing page, navigation hub
    ├── getting-started/     # TUTORIALS: For first-time users
    │   ├── 01-installation.md
    │   └── 02-first-graph-and-agents.md
    ├── guides/              # HOW-TO GUIDES: For specific tasks
    │   ├── deployment.md
    │   ├── development-workflow.md
    │   ├── orchestrating-agents.md
    │   ├── extending-the-system.md
    │   ├── troubleshooting.md
    │   ├── working-with-gui-sandbox.md
    │   └── xr-setup.md
    ├── concepts/            # EXPLANATION: Core ideas and architecture
    │   ├── index.md
    │   ├── system-architecture.md
    │   ├── agentic-workers.md
    │   ├── gpu-compute.md
    │   ├── networking-and-protocols.md
    │   ├── data-flow.md
    │   ├── security-model.md
    │   ├── ontology-and-validation.md
    │   └── decisions/         # Consolidated ADRs
    │       ├── index.md
    │       └── adr-*.md
    ├── reference/           # REFERENCE: Detailed technical information
    │   ├── index.md
    │   ├── glossary.md
    │   ├── configuration.md
    │   ├── api/
    │   │   ├── index.md
    │   │   ├── rest-api.md
    │   │   ├── websocket-protocol.md
    │   │   ├── binary-protocol.md
    │   │   ├── voice-api.md
    │   │   └── openapi-spec.yml
    │   └── agents/
    │       ├── index.md
    │       ├── conventions.md
    │       ├── core/
    │       ├── github/
    │       ├── consensus/
    │       ├── optimization/
    │       ├── sparc/
    │       ├── swarm/
    │       └── templates/
    ├── contributing.md        # Contribution guidelines
    └── archive/               # Old, deprecated files for historical context
        ├── README.md
        └── ...
3. Phased Execution Plan
This project is broken down into five distinct phases.
Phase 1: Preparation & Scaffolding ✅ COMPLETE
Objective: Prepare the repository for the migration by creating the new structure and defining standards.

**Completed Tasks:**
- ✅ Created new directory structure (getting-started, guides, concepts/decisions, reference/api, reference/agents, archive)
- ✅ Created placeholder index files
- ✅ Fixed "VisionsFlow" → "VisionFlow" typos (8 occurrences in 3 files)
- ✅ Moved ADR-001 to docs/concepts/decisions/adr-001-unified-api-client.md
- ✅ Moved ADR-003 to docs/concepts/decisions/adr-003-code-pruning-2025-10.md
- ✅ Moved OpenAPI spec to docs/reference/api/openapi-spec.yml
- ✅ Verified contributing.md and glossary.md already in correct locations

**External Services Identified for Documentation:**
- RAGFlow: External docker network (docker_ragflow) - RAG engine
- Whisper: STT service at whisper-webui-backend:8000
- Kokoro: TTS service at kokoro-tts-container:8880
- Vircadia: Integrated XR/AR system in client

Tasks:
Branch Creation: Create a new feature branch for this work (e.g., feature/knowledge-base-restructure).
Establish Conventions:
File Naming: All new files and directories will use kebab-case. Number prefixes are only for ordered tutorials in getting-started.
Linking: All links must be relative repository links (e.g., ../concepts/system-architecture.md).
Language: Standardize on UK English (optimisation, colour, etc.) as noted in 00-INDEX.md.
Typo Correction: Globally find and replace "VisionsFlow" with "VisionFlow".
Create New Directory Structure: Execute the following commands to create the new scaffolding.
code
Bash
# Run from the root of the AR-AI-Knowledge-Graph repository
mkdir -p docs/getting-started docs/guides docs/concepts/decisions docs/reference/api docs/reference/agents docs/archive
touch docs/index.md docs/contributing.md docs/archive/README.md
touch docs/concepts/index.md docs/concepts/decisions/index.md
touch docs/reference/index.md docs/reference/api/index.md docs/reference/agents/index.md
Initial File Movement (No Content Changes): Move files that require minimal changes to their new locations to begin populating the structure.
code
Bash
# Move ADRs
mv docs/architecture/decisions/ADR-001-unified-api-client.md docs/concepts/decisions/adr-001-unified-api-client.md
mv docs/decisions/003-code-pruning-2025-10.md docs/concepts/decisions/adr-003-code-pruning-2025-10.md

# Move Glossary
mv docs/reference/glossary.md docs/reference/glossary.md

# Move Contributing Guide
mv docs/contributing.md docs/contributing.md

# Move OpenAPI Spec
mv docs/ontology-openapi-spec.yml docs/reference/api/openapi-spec.yml
Phase 2: Content Consolidation & Migration ⏳ IN PROGRESS
Objective: Systematically process, merge, and rewrite the existing documents into the new, canonical structure.

**Completed Migrations:**
- ✅ **02-first-graph-and-agents.md**: Merged 02-quick-start.md + quickstart.md into cohesive user journey guide
  - Comprehensive 10-minute tutorial from installation to first multi-agent deployment
  - Documented all external services: RAGFlow, Whisper STT, Kokoro TTS, Vircadia XR
  - UK English throughout
  - Extensive troubleshooting and next steps

**In Progress:**
- ⏳ Deployment guide consolidation
- ⏳ System architecture synthesis

Methodology: For each new target file, identify source files, merge their content, remove duplication, rewrite for a knowledge base audience, and establish preliminary links.

Detailed Migration Tasks:
Target File	Source File(s)	Action Plan
docs/getting-started/01-installation.md	docs/getting-started/01-installation.md	Review & Refine: This file is mostly good. Standardize formatting, ensure clarity for a new user, and add links to troubleshooting.md and configuration.md.
docs/getting-started/02-first-graph-and-agents.md	docs/getting-started/02-quick-start.md<br>docs/getting-started/quickstart.md	Merge & Rewrite: Combine both quick start guides into a single, cohesive tutorial. Focus on the user journey: install -> run -> see graph -> spawn first agent. Remove development-specific jargon.
docs/guides/deployment.md	docs/guides/01-deployment.md<br>docs/development/deployment.md<br>docs/deployment/vircadia-docker-deployment.md	Merge & Structure: Create a master deployment guide. Use content from vircadia-docker-deployment.md as the primary "How-To". Integrate general principles from the other two files. Add sections for different environments (Local, Staging, Production).
docs/guides/development-workflow.md	docs/guides/02-development-workflow.md	Move & Refine: This is a good guide. Move it to the new location, clean up formatting, and link heavily to contributing.md and the new docs/concepts/decisions/ directory.
docs/guides/orchestrating-agents.md	docs/guides/04-orchestrating-agents.md	Move & Expand: Move the file. Expand with practical examples drawn from the reference/agents/ directory to show how to use different agent types and coordination patterns.
docs/guides/extending-the-system.md	docs/guides/05-extending-the-system.md	Move & Refine: Move the file and ensure it provides clear, actionable steps for developers looking to add custom agents or tools. Link to the reference/agents/templates/ section.
docs/guides/troubleshooting.md	docs/guides/06-troubleshooting.md<br>docs/multi-agent-docker/TROUBLESHOOTING.md	Merge & Categorize: Combine all troubleshooting information into one central guide. Create clear sections like "Installation," "Docker," "Agent," "Performance."
docs/guides/working-with-gui-sandbox.md	docs/guides/03-using-the-gui-sandbox.md	Move & Clarify: Move the file. Ensure the purpose of the GUI sandbox and its relationship with MCP tools is crystal clear from the start. Add a summary of available tools with links to the multi-agent-docker/TOOLS.md reference.
docs/guides/xr-setup.md	docs/guides/xr-quest3-setup.md	Move & Generalize: Rename to xr-setup.md. While focusing on Quest 3, frame it as a general WebXR setup guide, making it more future-proof.
docs/concepts/system-architecture.md	docs/architecture/overview.md<br>docs/architecture/system-overview.md<br>docs/concepts/01-system-overview.md<br>docs/architecture/hybrid_docker_mcp_architecture.md<br>docs/multi-agent-docker/ARCHITECTURE.md	Synthesize & Rewrite: This is a major consolidation task. Create a definitive, high-level explanation of the system. Start with the diagrams from overview.md, integrate details from system-overview.md and hybrid_docker_mcp_architecture.md. Use a C4 model approach (System Context, Containers, Components).
docs/concepts/agentic-workers.md	docs/concepts/02-agentic-workers.md	Move & Refine: This is a good conceptual document. Move it, improve diagrams, and ensure it links to the practical guides/orchestrating-agents.md and the detailed reference/agents/ section.
docs/concepts/gpu-compute.md	docs/concepts/03-gpu-compute.md<br>docs/architecture/gpu/*	Merge & Explain: Use 03-gpu-compute.md as the base. Integrate the more technical details from architecture/gpu/ (communication flow, optimizations, stability) as subsections. The goal is to explain why and how GPU compute is used.
docs/concepts/networking-and-protocols.md	docs/concepts/04-networking.md	Move & Expand: This is a good base. Integrate details from the various protocol documents (websocket-protocol.md, binary-protocol.md, mcp-protocol.md) to provide a holistic view of the communication layers.
docs/concepts/data-flow.md	docs/concepts/06-data-flow.md	Move & Refine: Move the file. Enhance the diagrams and ensure the explanations are clear for someone unfamiliar with the system. Link to the API reference for specifics on data structures.
docs/concepts/security-model.md	docs/architecture/security.md<br>docs/concepts/05-security.md<br>docs/guides/security.md	Merge & Structure: Use architecture/security.md as the master document. Create a comprehensive explanation of the security model, covering authentication, authorization, data protection, etc.
docs/concepts/ontology-and-validation.md	docs/concepts/ontology-validation.md<br>docs/specialized/ontology/*	Synthesize & Summarize: Create a new conceptual document explaining the role of ontology. Use ontology-validation.md as a starting point. Summarize the key ideas from the specialized/ontology files and link to them for deep dives (they will be moved to reference/ontology/).
docs/reference/configuration.md	docs/reference/configuration.md<br>docs/reference/cuda-parameters.md<br>docs/multi-agent-docker/PORT-CONFIGURATION.md	Merge & Organize: Create a single, authoritative configuration reference. Organize by component (System, Database, GPU, Agents, Network). Use tables for clarity.
docs/reference/api/websocket-protocol.md	docs/architecture/components/websocket-protocol.md<br>docs/reference/api/websocket-api.md	Merge & Finalize: Use websocket-protocol.md for the low-level technical details and websocket-api.md for the high-level usage patterns. Combine them into a single, comprehensive document covering both aspects.
docs/reference/api/binary-protocol.md	docs/reference/binary-protocol.md<br>docs/reference/api/binary-protocol.md	Merge & Clarify: These two files describe different formats (28-byte vs 34-byte). The technical-verification-report.md confirms the 34-byte format is the one in use. Create a single, definitive binary-protocol.md based on the 34-byte format and archive the 28-byte version.
docs/reference/agents/*	docs/reference/agents/*	Restructure & Index: The agent documentation is extensive and well-structured but needs better entry points. Create a main index.md that explains the agent system and links to sub-indexes for each category (core, github, etc.). Standardize all agent files to use kebab-case.md.
Phase 3: Root-Level & Index Restructuring (2 Days)
Objective: Create the main entry points for the project and the knowledge base, ensuring a smooth user journey from the GitHub repository landing page into the documentation.
Tasks:
Create the New Root README.md:
Source: README-FULLFAT.md and the existing README.md.
Action: Create a new README.md from scratch. It should be visually engaging and serve as a "shop window" for the project.
Content:
Project Title & Badges.
A concise, powerful one-liner explaining the project.
Key feature highlights (from README-FULLFAT.md).
A compelling GIF or screenshot.
A "Why VisionFlow?" section explaining the value proposition.
A super simple "Quick Start" section (clone, docker-compose up, open URL).
A prominent link to the full Knowledge Base: [📚 Full Documentation](docs/index.md).
Technology stack summary.
Link to contributing guidelines.
Create the New docs/index.md (Knowledge Base Hub):
Source: 00-INDEX.md and docs/index.md.
Action: This page is the central navigation hub. It should not be a giant table of contents but a guided entry point.
Content:
Welcome message.
"Start Here" Section: Links to the three main user paths:
"I'm new and want to try it out" -> getting-started/01-installation.md
"I want to accomplish a specific task" -> guides/
"I want to understand how it works" -> concepts/
Main Sections: Cards or links for each top-level directory (Getting Started, Guides, Concepts, Reference) with a brief description of what's inside.
A search bar placeholder (if a search tool like Algolia is planned).
Link to contributing.md.
Phase 4: Linking, Navigation & Verification (3 Days)
Objective: Transform the collection of documents into a cohesive, interconnected web of knowledge.
Tasks:
Implement Breadcrumbs: Add a breadcrumb navigation link at the top of every document (except the root index.md).
Example for docs/guides/deployment.md: [Knowledge Base](../index.md) > [Guides](./index.md) > Deployment
Add "Related Articles" Sections: At the bottom of each document, add a section linking to 2-4 other relevant documents. This is crucial for discoverability.
Conduct a Full Link Audit:
Use a script or tool (like markdown-link-check) to verify that there are no broken links.
Manually review every document to ensure all internal links are relative and correct.
Ensure that for every forward link, there's a logical backward link (either a breadcrumb or a link in the "Related" section).
Review Navigation Flow:
Simulate user journeys:
Can a new user go from README.md to a running instance without getting lost?
Can a developer find the REST API reference for a specific endpoint easily?
Can a contributor find the architectural decision records?
Adjust links and section indexes based on this review.
Phase 5: Finalization & Archival (1 Day)
Objective: Clean up the repository, removing all old structures and leaving only the new, polished knowledge base.
Tasks:
Archive Remaining Files:
Move all original files that were consolidated (and not deleted) into the docs/archive/ directory.
Create a README.md in the archive explaining that the content is historical and may be outdated, pointing to the new canonical documents.
This includes files like 00-INDEX.md, README-FULLFAT.md, DOCUMENTATION-MIGRATION-COMPLETE.md, and all the original, now-merged documents.
Delete Old Directories: Remove all the old, now-empty directories (docs/architecture, docs/implementation, docs/reports, etc.).
Final Review: Perform a final pass over the entire docs structure. Check for:
Consistent formatting and naming.
Presence of breadcrumbs and related links.
Clarity of the main README.md and docs/index.md.
Merge the Branch: Submit the feature/knowledge-base-restructure branch for review and merge it into the main branch.
This detailed plan provides a systematic path to transform the document corpus into a high-quality, user-centric knowledge base that will be a valuable asset for the project.

---

## 📊 Migration Progress Summary (2025-10-03)

### ✅ COMPLETED

#### Phase 1: Preparation & Scaffolding - 100% Complete
- New directory structure created (Diátaxis framework)
- "VisionsFlow" → "VisionFlow" typos fixed (8 occurrences)
- ADRs moved to concepts/decisions/
- OpenAPI spec moved to reference/api/
- Core files verified in correct locations

#### Phase 2: Content Consolidation - 18% Complete (1 of 18 critical files)
- ✅ **02-first-graph-and-agents.md** - Comprehensive 10-minute tutorial
  - Merged 02-quick-start.md + quickstart.md
  - User journey: install → graph → agents
  - All external services documented (RAGFlow, Whisper, Kokoro, Vircadia)
  - UK English throughout
  - Quality template for remaining migrations

### ⏳ IN PROGRESS

#### Immediate Next Steps (2-3 Days)
1. deployment.md - Merge 3 deployment guides
2. troubleshooting.md - Consolidate all troubleshooting
3. system-architecture.md - Major synthesis with Mermaid diagrams
4. Create root README.md and docs/index.md (Phase 3)

### 📋 PENDING

#### High Priority (1 Week)
- 6 remaining user onboarding guides
- 4 concept/architecture documents
- 4 reference consolidations

#### Phases 3-5 (2-3 Weeks)
- Root-level navigation files
- Comprehensive linking system
- Final archival and cleanup

### 📁 Key Documents Created
- [EXECUTIVE-SUMMARY-MIGRATION.md](docs/EXECUTIVE-SUMMARY-MIGRATION.md) - Executive overview
- [KNOWLEDGE-BASE-MIGRATION-STATUS.md](docs/KNOWLEDGE-BASE-MIGRATION-STATUS.md) - Detailed status
- [02-first-graph-and-agents.md](docs/getting-started/02-first-graph-and-agents.md) - Quality template

### 🎯 External Services Validated
- **RAGFlow**: docker_ragflow network (RAG engine)
- **Whisper**: whisper-webui-backend:8000 (STT)
- **Kokoro**: kokoro-tts-container:8880 (TTS)
- **Vircadia**: Client-integrated XR/AR (Quest 3)

### 💡 For Executive Team
1. ✅ Approve Phase 1 completion - Foundation is solid
2. 📋 Review quality template (02-first-graph-and-agents.md)
3. 👥 Allocate 1-2 technical writers for 10-15 days
4. 🎯 Prioritize critical user onboarding path

**Estimated Total Completion**: 10-15 working days
**Quality Standard**: World-class, user-centric knowledge base